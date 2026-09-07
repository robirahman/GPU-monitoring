#!/usr/bin/env python3
"""MN-P: hide a too-big-for-one-node training job by choosing its parallelism.

WHY THIS ATTACK, AND WHY IT IS NOT A SHAPING KNOB
-------------------------------------------------
Findings section 22 measured the host NIC on the 2x H100 pair and found the
inter-node channel separates big training from big sharded inference by 1123x:

    s16_train_fsdp30b                34,650 Mbps median    2.8% idle
    s16_nonml_comm                    1,650 Mbps          53.0% idle
    s16_infershard_qwen72b_tp8pp2        30.9 Mbps         10.3% idle
    s16_inferrep_qwen1_5b                 0.1 Mbps          100% idle

The larger model is the QUIET one, so the discriminator is not scale -- it is
which collectives cross the node boundary. FSDP shards parameters and optimizer
state over all 16 ranks, so every step all-gathers parameters and
reduce-scatters gradients over the NIC. Tensor-parallel inference keeps its
collectives inside the node on NVLink and sends only pipeline activations
between stages. No amount of reshaping an FSDP job closes three orders of
magnitude.

So MN-P changes the parallelism instead of the signal:

    FSDP (or TP) WITHIN a node   -> parameter traffic rides intra-node NVLink
    PIPELINE across nodes        -> the NIC carries only activations and
                                    activation gradients

That is the same KIND of traffic as the PP inference it imitates, at roughly 2x
the volume (inference sends activations forward only; training also sends the
gradient back). Predicted inter-node bytes per optimizer step:

    2 (fwd + bwd) * microbatches * local_ranks * batch * seq * hidden * 2 bytes

For 30B (hidden 6656) at batch 1, seq 512, 8 local ranks, 1 microbatch that is
109 MB/step against FSDP's multi-GB, i.e. a ~60x reduction -- enough to test
whether the detector's interconnect advantage survives, and to price it.

WHAT IT COSTS: the pipeline bubble. With `microbatches` = 1 a stage sits idle
while the other computes, so utilisation halves; GPipe with K microbatches cuts
the bubble to roughly 1/(1+K). Evasion cost here is a THROUGHPUT penalty from
the parallelism choice, a fundamentally different quantity from MN-I's 1/F, and
it is what EVASION_TAX_DATA reports so evasion_cost_eval.py can price it.

    torchrun --nnodes=2 --nproc-per-node=8 ... \
        workloads/adversarial_multinode_pipeline.py --size 30b --microbatches 4

STAGE 0 holds the embeddings and the first half of the blocks; the last stage
holds the rest plus the norm and the LM head. Stage index = node rank, so a
stage never straddles the NIC, and every rank's pipeline peer is the rank with
the same local id one node over.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads.fsdp_training import (
        MODEL_CONFIGS, BATCH_DEFAULTS, RandomTransformerLM, build_fsdp_model,
    )
except ImportError:  # launched as `torchrun workloads/adversarial_multinode_pipeline.py`
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown
    from fsdp_training import (
        MODEL_CONFIGS, BATCH_DEFAULTS, RandomTransformerLM, build_fsdp_model,
    )


# ── the cross-node hop ───────────────────────────────────────────────────────
# Two autograd Functions rather than one: a rank is either the sender or the
# receiver of a given edge, never both, and folding that into one Function
# would mean branching on stage inside forward/backward -- exactly the kind of
# rank-dependent collective that has deadlocked this project's multislice runs
# before. Here the send/recv ORDER is a function of the microbatch index only,
# which is identical on every rank.

# NOTE ON WHY THERE ARE NO autograd.Function WRAPPERS HERE
#
# The first version put dist.send/dist.recv inside two autograd Functions, which
# is the tidy way to express a pipeline edge. It deadlocked, and the trace shows
# exactly how: stage 1 finished step 0 and moved on to step 1's recv while
# stage 0 was still blocked inside step 0's backward waiting for the gradient
# stage 1 had already sent.
#
#   node0: [trace r0 step0.0] fwd done -> activation sent -> (blocked forever)
#   node1: [trace r8 step0.0] recv done -> fwd done -> bwd done, grad sent
#          [trace r8 step1.0] recv begin
#
# P2P issued from inside a backward runs on the autograd engine's thread and
# stream, so the two peers' operations stop being a matched pair in a single
# well-defined order, and NCCL eventually gives up with
# "ncclRemoteError: ... a remote process exiting prematurely".
#
# Doing the transfers on the MAIN thread and calling backward() explicitly with
# the received gradient keeps one strict order per edge:
#
#   stage 0:  send(activation) -> recv(grad) -> h.backward(grad)
#   stage 1:  recv(activation) -> forward/backward -> send(h_in.grad)
#
# which is what every hand-rolled pipeline implementation ends up doing.

# ── stage models ─────────────────────────────────────────────────────────────

class PipelineStage(nn.Module):
    """One contiguous slice of RandomTransformerLM.

    Built by taking the whole model on `meta` and dropping the layers this
    stage does not own, so the split matches fsdp_training's model exactly and
    the baseline and the disguise are the same architecture.
    """

    def __init__(self, config_name, first, last, n_stages, stage):
        super().__init__()
        full = RandomTransformerLM(config_name)
        self.vocab_size = full.vocab_size
        self.is_first = stage == 0
        self.is_last = stage == n_stages - 1
        self.blocks = nn.ModuleList(list(full.blocks)[first:last])
        if self.is_first:
            self.tok_emb = full.tok_emb
            self.pos_emb = full.pos_emb
        if self.is_last:
            self.norm = full.norm
            self.lm_head = full.lm_head

    def forward(self, x):
        if self.is_first:
            B, T = x.shape
            pos = torch.arange(T, device=x.device).unsqueeze(0)
            x = self.tok_emb(x) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        if self.is_last:
            x = self.norm(x)
            x = self.lm_head(x)
        return x


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="30b", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--microbatches", type=int, default=1,
                    help="GPipe depth. 1 is the worst bubble (a stage idles "
                         "while the other computes) and the quietest NIC; "
                         "larger K trades inter-node bytes for utilisation.")
    ap.add_argument("--pp-split", type=int, default=None,
                    help="layers on stage 0 (default: even split)")
    ap.add_argument("--activation-ckpt", type=int, default=1)
    ap.add_argument("--trace", type=int, default=0,
                    help="per-phase progress from one rank per stage; for "
                         "diagnosing a pipeline stall, not for collection")
    args = ap.parse_args()

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])
    local_ws = int(os.environ.get("LOCAL_WORLD_SIZE", torch.cuda.device_count()))
    node_rank = rank // local_ws
    n_stages = world_size // local_ws
    if n_stages < 2:
        raise SystemExit("MN-P needs at least 2 nodes -- the disguise IS the "
                         "node boundary")

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    hidden, n_layers, n_heads, ffn, vocab = MODEL_CONFIGS[args.size]
    batch_size = args.batch_size or BATCH_DEFAULTS[args.size]

    # Intra-node group: FSDP shards the stage over the 8 GPUs of THIS node, so
    # every parameter all-gather stays on NVLink. Created by all ranks (it is a
    # collective) even though each rank only uses its own.
    node_groups = [
        dist.new_group(ranks=list(range(s * local_ws, (s + 1) * local_ws)))
        for s in range(n_stages)
    ]
    my_group = node_groups[node_rank]

    split = args.pp_split or (n_layers // n_stages)
    bounds = [(s * split, min((s + 1) * split, n_layers)) for s in range(n_stages)]
    bounds[-1] = (bounds[-1][0], n_layers)      # last stage absorbs the remainder
    first, last = bounds[node_rank]

    peer_next = rank + local_ws if node_rank < n_stages - 1 else None
    peer_prev = rank - local_ws if node_rank > 0 else None

    # A DEDICATED 2-rank group per pipeline edge, rather than P2P on the world
    # group. Doing send/recv as the world group's first operation failed at the
    # very first collective:
    #   ncclRemoteError: A call failed possibly due to a network error or a
    #   remote process exiting prematurely   (rank 15, "last enqueued NCCL
    #   work: 0")
    # because the world communicator's init is a rendezvous every member has to
    # reach, and the stage-0 ranks were still inside their intra-node FSDP
    # collective while the stage-1 ranks sat in recv. A 2-rank group has
    # unambiguous membership and both its members arrive together.
    # new_group is itself collective, so EVERY rank builds EVERY edge.
    edge_group = {}
    for _lr in range(local_ws):
        for _s in range(n_stages - 1):
            _a = _s * local_ws + _lr
            _b = (_s + 1) * local_ws + _lr
            edge_group[(_a, _b)] = dist.new_group(ranks=[_a, _b])
    g_next = edge_group.get((rank, peer_next)) if peer_next is not None else None
    g_prev = edge_group.get((peer_prev, rank)) if peer_prev is not None else None

    if rank == 0:
        print(f"MN-P pipeline disguise: {args.size} over {n_stages} stages "
              f"x {local_ws} FSDP ranks", flush=True)
        print(f"  stage layer bounds: {bounds}  (n_layers={n_layers})")
        print(f"  microbatches={args.microbatches} batch={batch_size} "
              f"seq={args.seq_len} hidden={hidden}")
        act_mb = batch_size * args.seq_len * hidden * 2 / 1e6
        print(f"  activation tensor: {act_mb:.1f} MB per rank per hop")
        print(f"  predicted inter-node bytes/step: "
              f"{2 * args.microbatches * local_ws * act_mb:.0f} MB "
              f"(fwd activations + bwd grads, {local_ws} ranks)")
        print(f"  Duration: {args.duration}s", flush=True)

    # Build on meta, then materialise ON THE HOST before FSDP shards it.
    #
    # FSDP's own meta-init (param_init_fn) does not reach the ROOT parameters of
    # a sliced stage: stage 1's lm_head stayed on meta and flattening it failed
    # with "setStorage: sizes [32000, 2048] ... storage of size 0". Doing it
    # ourselves is also the only way this scales -- a 30b stage is 16.25B
    # parameters, which is 65 GB in fp32 and will not fit on one 40 GB A100
    # before sharding. bf16 on the host is 32.5 GB per rank, and the pair has
    # 680 GB of RAM per node, so 8 ranks fit with room to spare.
    _prev_dtype = torch.get_default_dtype()
    torch.set_default_dtype(torch.bfloat16)
    try:
        with torch.device("meta"):
            stage_mod = PipelineStage(args.size, first, last, n_stages, node_rank)
        stage_mod.to_empty(device="cpu")
        with torch.no_grad():
            for prm in stage_mod.parameters():
                prm.normal_(mean=0.0, std=0.02)
            for buf in stage_mod.buffers():
                buf.zero_()
    finally:
        torch.set_default_dtype(_prev_dtype)

    model = build_fsdp_model(stage_mod, device,
                             activation_ckpt=bool(args.activation_ckpt),
                             cpu_offload=False, process_group=my_group,
                             materialized=True)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    if rank == 0:
        print(f"  Post-init GPU memory: "
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB", flush=True)

    # Warm the edge communicators before the clock starts: NCCL builds a
    # communicator on first use, and paying for it inside the measured window
    # would show up as a stall in the telemetry we are trying to characterise.
    _warm = torch.zeros(1, device=device, dtype=torch.bfloat16)
    if peer_next is not None:
        dist.send(_warm, dst=peer_next, group=g_next)
    if peer_prev is not None:
        dist.recv(_warm, src=peer_prev, group=g_prev)
    dist.barrier()
    if rank == 0:
        print("  edge communicators warm", flush=True)

    deadline = CollectiveDeadline(args.duration, device)
    wall_start = time.time()
    step = 0
    tokens_per_step = batch_size * local_ws * args.seq_len * args.microbatches
    useful_tokens = 0
    inter_node_bytes = 0
    loss_val = float("nan")
    last_print = wall_start

    while not deadline.expired():
        optimizer.zero_grad(set_to_none=True)

        # ---- 1F1B, not GPipe: each microbatch is a COMPLETE forward+backward
        # before the next one starts, and gradients accumulate until the
        # optimizer step.
        #
        # GPipe (all forwards, then all backwards) is what a throughput-focused
        # implementation would do, and it is what the first version did -- but
        # FSDP FULL_SHARD frees the unsharded flat parameter after each forward
        # and re-gathers it inside ITS OWN backward hooks, so a second backward
        # over a separately retained graph finds freed storage:
        #   RuntimeError: setStorage: sizes [32000, 2048] ... storage of size 0
        # (the lm_head, on every stage-1 rank). Retaining K forward graphs
        # across an FSDP reshard is simply not supported. 1F1B keeps exactly
        # one live graph, which is the shape FSDP is built for, and it costs
        # only the overlap we were never going to get from a 2-stage pipeline
        # anyway -- the bubble is the honest price of this disguise.
        for _mb in range(args.microbatches):
            def _tr(msg, _s=step, _m=_mb):
                if args.trace and local_rank == 0:
                    print(f"  [trace r{rank} step{_s}.{_m}] {msg}", flush=True)

            if peer_prev is None:
                # FIRST stage: compute, ship activations on, wait for their
                # gradient, then run our own backward with it.
                ids = torch.randint(0, vocab, (batch_size, args.seq_len),
                                    device=device)
                h = model(ids)
                _tr("fwd done")
                dist.send(h.detach().contiguous(), dst=peer_next, group=g_next)
                inter_node_bytes += h.numel() * h.element_size()
                grad = torch.empty_like(h)
                dist.recv(grad, src=peer_next, group=g_next)
                inter_node_bytes += grad.numel() * grad.element_size()
                _tr("grad received")
                h.backward(grad)
                _tr("bwd done")
            else:
                h_in = torch.empty(batch_size, args.seq_len, hidden,
                                   device=device, dtype=torch.bfloat16)
                dist.recv(h_in, src=peer_prev, group=g_prev)
                inter_node_bytes += h_in.numel() * h_in.element_size()
                h_in.requires_grad_(True)
                _tr("activation received")
                out = model(h_in)
                if peer_next is None:
                    # LAST stage: the loss lives here.
                    tgt = torch.randint(0, vocab, (batch_size, args.seq_len),
                                        device=device)
                    loss = F.cross_entropy(
                        out.reshape(-1, out.size(-1)).float(), tgt.reshape(-1))
                    loss_val = float(loss.detach())
                    (loss / args.microbatches).backward()
                else:
                    # MIDDLE stage: pass activations on, get their grad back,
                    # then backward into our own slice. Order matches both
                    # neighbours: recv(act) -> send(act) -> recv(grad) -> send(grad).
                    dist.send(out.detach().contiguous(), dst=peer_next,
                              group=g_next)
                    inter_node_bytes += out.numel() * out.element_size()
                    g_out = torch.empty_like(out)
                    dist.recv(g_out, src=peer_next, group=g_next)
                    inter_node_bytes += g_out.numel() * g_out.element_size()
                    out.backward(g_out)
                _tr("bwd done")
                dist.send(h_in.grad.detach().contiguous(), dst=peer_prev,
                          group=g_prev)
                inter_node_bytes += h_in.grad.numel() * h_in.grad.element_size()
                _tr("grad sent")

        optimizer.step()
        useful_tokens += tokens_per_step
        step += 1

        if rank == 0:
            emit_step("train", step, tokens=tokens_per_step, passes=1,
                      node=node_rank, rank=rank, loss=f"{loss_val:.4f}")
            now = time.time()
            if now - last_print >= 30:
                el = now - wall_start
                print(f"  {el:.0f}s: steps={step} "
                      f"useful={useful_tokens / el:.0f} tok/s "
                      f"internode={inter_node_bytes / el / 1e6:.1f} MB/s "
                      f"loss={loss_val:.4f}", flush=True)
                last_print = now

    elapsed = time.time() - wall_start
    if rank == 0:
        print(f"\nMN-P complete: {step} steps in {elapsed:.1f}s")
        print(f"  Useful tokens: {useful_tokens:,} "
              f"({useful_tokens / elapsed:.0f} tok/s)")
        print(f"  Inter-node bytes (this rank): {inter_node_bytes / 1e9:.2f} GB "
              f"= {inter_node_bytes / elapsed / 1e6:.1f} MB/s "
              f"= {inter_node_bytes * 8 / elapsed / 1e6:.0f} Mbps")
        # Same schema evasion_cost_eval.py already parses. There is no diluted
        # work here: every step is a real optimizer step, so attributed ==
        # useful and the whole cost shows up as a lower tok/s.
        print(f"  EVASION_TAX_DATA: strategy=MN-P world_size={world_size} "
              f"train_frac=1.0 dilution= steps={step} train_steps={step} "
              f"useful_tokens={useful_tokens} attributed_tokens={useful_tokens} "
              f"duration={elapsed:.1f} microbatches={args.microbatches} "
              f"size={args.size} internode_mbps="
              f"{inter_node_bytes * 8 / elapsed / 1e6:.1f}", flush=True)

    sync_teardown(device)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
