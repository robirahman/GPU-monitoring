#!/usr/bin/env python3
"""Local SGD / DiLoCo training -- distributed training that is almost never talking.

Why this workload exists
------------------------
The 2026-09-30 leave-one-family-out put held-out training recall at 0.5687 with
train_ddp at 0.0255 and train_lora at 0.2060, and the ablations showed that more
data and more NIC coverage do not move it: train_ddp already has sidecars on
80.6% of its windows. What the corpus lacks is training programs whose
COMMUNICATION MACHINERY the held-out fold has never seen. Every training family
we have talks on every step:

    FSDP          all-gather + reduce-scatter per layer, per step   16077 Mbps
    contrastive   all-gather of embeddings, per step                 1947
    diffusion     DDP-shaped                                          580
    pipe          point-to-point activations, per microbatch          513
    DDP           one all-reduce of the full gradient, per step       184
    LoRA          all-reduce of tiny adapter grads, per step            0.003

Local SGD breaks that invariant. Each replica trains INDEPENDENTLY for H steps
and the ranks only meet every H-th step, where they average parameters through
one enormous all-reduce. Between syncs there is no inter-node traffic at all, so
a 16-GPU job looks exactly like 16 unrelated single-GPU jobs -- which is the
multi-node disguise this project exists to detect, arrived at honestly rather
than adversarially. DiLoCo (Douillard et al.) is a real published method for
training across poorly-connected nodes, so this is not a strawman: an adversary
with a slow link has an ordinary engineering reason to train exactly this way.

The traffic shape is the point. At --sync-every 200 the NIC is silent for
minutes and then moves the entire parameter set at once. Long flat stretches
punctuated by rare huge bursts is a signature no other family in the corpus
produces, and it is the one that most resembles independent inference replicas.

    torchrun --nproc-per-node=8 workloads/train_localsgd.py --size 1b --sync-every 50
    torchrun --nproc-per-node=8 workloads/train_localsgd.py --size 1b --sync-every 200

Memory
------
Every rank holds a FULL replica -- that is what makes it local SGD rather than
FSDP -- so GPU memory is params + grads + Adam moments with no sharding to hide
behind. The outer optimizer's state (the global parameter copy and its momentum
buffer) lives on the CPU: it is touched once per H steps, so the transfer is
free at these periods, and keeping it on the GPU would add another 2x params to
a budget that is already the binding constraint. 1b in fp32 needs ~21 GB and
fits a 40 GB card; anything larger should be gated with need_gb.
"""

import argparse
import os
import sys
import time

try:
    from workloads._step import emit_step
    from workloads._seed import add_seed_argument, record_seed, record_progress
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads import fsdp_training as _fsdp
except ImportError:  # launched as `torchrun workloads/train_localsgd.py`
    from _step import emit_step
    from _seed import add_seed_argument, record_seed, record_progress
    from dist_deadline import CollectiveDeadline, sync_teardown
    import fsdp_training as _fsdp

RandomTransformerLM = _fsdp.RandomTransformerLM

import torch
import torch.nn as nn
import torch.distributed as dist

try:
    from workloads.gradcompress import (
        CompressedAllReduce, EFState, make_codec, ring_allreduce_bytes)
except ImportError:
    from gradcompress import (
        CompressedAllReduce, EFState, make_codec, ring_allreduce_bytes)


class _EFSlot:
    """Views one parameter's slot in a shared EFState.

    CompressedAllReduce indexes residuals by POSITION in the tensor list it is
    handed, and here that list is a single delta -- so every parameter would
    share slot 0 and overwrite each other's residual. This maps the caller's
    parameter index onto slot 0 of a per-parameter view.
    """

    def __init__(self, ef, idx):
        self._ef, self._i = ef, idx

    def get(self, _i, like):
        return self._ef.get(self._i, like) if self._ef is not None else \
            torch.zeros_like(like)

    def set(self, _i, v):
        if self._ef is not None:
            self._ef.set(self._i, v)

# A replica-per-rank workload cannot lean on sharding, so it wants a smaller
# default than the FSDP cells. 350m keeps the whole thing comfortable on a
# 40 GB card at seq 1024 while still producing a multi-GB sync burst.
#
# It has to be registered in fsdp_training's OWN dict, not a local copy:
# RandomTransformerLM resolves the name against the module global, so a local
# override would pass argparse's choices check and then KeyError inside the
# constructor.
_fsdp.MODEL_CONFIGS.setdefault("350m", (1024, 24, 16, 4096, 32000))
MODEL_CONFIGS = _fsdp.MODEL_CONFIGS

BATCH_DEFAULTS = {"350m": 8, "1b": 4, "7b": 1}


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--size", default="1b", choices=sorted(MODEL_CONFIGS))
    p.add_argument("--duration", type=int, default=600,
                   help="wall-clock seconds; every rank stops on the same step")
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--sync-every", type=int, default=50,
                   help="inner steps between parameter syncs. 1 reproduces "
                        "synchronous data parallelism; the interesting values "
                        "are large enough that the NIC goes quiet in between.")
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--outer-lr", type=float, default=0.7,
                   help="DiLoCo outer step size on the pseudo-gradient")
    p.add_argument("--outer-momentum", type=float, default=0.9)
    p.add_argument("--compress", default="none",
                   help="none | e3m0 | e3m0:sr | int8 | topk:0.01 | topk:0.01,noef. "
                        "Compression switches the sync to DELTA form; see "
                        "local_sgd_sync. Note that at one replica per rank "
                        "(world 16) the compressed all-gather costs W/2 times an "
                        "all-reduce, so e3m0 nets 1.00x and int8 nets 0.50x -- "
                        "see workloads/gradcompress.py")
    p.add_argument("--outer-opt", default="nesterov", choices=("nesterov", "avg"),
                   help="'avg' is plain parameter averaging with no outer state, "
                        "for when GPU/CPU memory is the binding constraint")
    add_seed_argument(p)
    return p.parse_args()


@torch.no_grad()
def local_sgd_sync(model, outer, args, world_size, codec=None, ef=None):
    """Average replicas across ranks and take the DiLoCo outer step.

    Returns the number of BYTES ON THE WIRE this rank transmitted, so the caller
    can report interconnect volume honestly. With a codec that is the compressed
    payload under the all-gather cost model, not the parameter size.

    The averaging itself is what an adversary cannot avoid: however long the
    inner loop runs, the replicas have to meet eventually or they are not
    training one model.

    DELTA FORM, and this is the part that must not be got wrong.  Uncompressed,
    reducing `p.data` directly is fine. Compressed, it is not: top-k on a
    PARAMETER tensor keeps the largest 1% of the weights and zeroes the other
    99%, destroying the model in one sync. Top-k on a DELTA keeps the largest 1%
    of the CHANGE and merely postpones the rest, which error feedback then
    carries into the next round. So with a codec the quantity reduced is
    `p - prev`, and `prev` -- the outer optimizer's CPU copy -- is already
    maintained for exactly this. With codec=None the two forms are
    algebraically identical, and scripts/test_localsgd_sync.py asserts that.
    """
    moved = 0
    for i, (name, p) in enumerate(model.named_parameters()):
        if codec is None:
            dist.all_reduce(p.data, op=dist.ReduceOp.SUM)
            p.data /= world_size
            moved += ring_allreduce_bytes(
                p.data.numel() * p.data.element_size(), world_size)
        else:
            prev_cpu = outer[name][0] if args.outer_opt != "avg" else None
            base = (prev_cpu.to(p.device) if prev_cpu is not None
                    else p.data.detach().clone())
            delta = [p.data.float() - base]
            car = CompressedAllReduce(codec, world_size=world_size)
            moved += car.mean_(delta, ef=_EFSlot(ef, i))
            p.data.copy_((base + delta[0]).to(p.data.dtype))

        if args.outer_opt == "avg":
            continue
        # DiLoCo: the pseudo-gradient is how far the averaged replica moved away
        # from the global weights over the whole inner loop, and the outer
        # optimizer applies it with Nesterov momentum. Doing this on CPU keeps
        # two more full parameter copies off the GPU; at one touch per H steps
        # the transfer does not show up in the telemetry.
        prev, buf = outer[name]
        pseudo_grad = prev - p.data.detach().float().to("cpu")  # global - averaged
        buf.mul_(args.outer_momentum).add_(pseudo_grad)
        update = pseudo_grad.add(buf, alpha=args.outer_momentum)
        prev.add_(update, alpha=-args.outer_lr)
        p.data.copy_(prev.to(p.data.device, non_blocking=True))
    return moved


def main():
    args = parse_args()
    record_seed(args.seed)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 8)) or 8
    node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", 0)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    batch_size = args.batch_size or BATCH_DEFAULTS.get(args.size, 4)
    vocab = MODEL_CONFIGS[args.size][4]

    # Every replica must start from the SAME weights or the first average is
    # meaningless -- so build on rank 0's values and broadcast, then give each
    # rank its own DATA stream. Identical data would make the replicas identical
    # and the sync a no-op, which would quietly turn this into a workload that
    # transfers a large tensor for no reason.
    torch.manual_seed(args.seed)
    model = RandomTransformerLM(args.size).to(device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    torch.manual_seed(args.seed + rank)

    codec = make_codec(args.compress) if args.compress != "none" else None
    ef = EFState() if (codec is not None and getattr(codec, "uses_error_feedback",
                                                     False)) else None
    inner = torch.optim.AdamW(model.parameters(), lr=args.inner_lr)
    outer = {}
    if args.outer_opt == "nesterov":
        for name, p in model.named_parameters():
            prev = p.data.detach().to("cpu").clone()
            outer[name] = (prev, torch.zeros_like(prev))

    if rank == 0:
        params = sum(p.numel() for p in model.parameters())
        print(f"Local SGD / DiLoCo: size={args.size} params={params/1e9:.2f}B "
              f"world_size={world_size} sync_every={args.sync_every} "
              f"outer={args.outer_opt} batch={batch_size} seq={args.seq_len}",
              flush=True)
        print(f"  replica memory: {torch.cuda.max_memory_allocated()/2**30:.1f} GB "
              f"after init", flush=True)

    # Piggyback the deadline vote on the sync boundary. The vote is a 1-element
    # all_reduce -- trivial bandwidth, but on THIS workload a per-step collective
    # would stamp a steady inter-node rhythm onto a job whose entire signature is
    # the absence of one, and the cross-rank features would read it.
    deadline = CollectiveDeadline(args.duration, device, check_every=args.sync_every)

    wall_start = time.time()
    step = 0
    syncs = 0
    inter_node_bytes = 0
    tokens = 0
    while not deadline.expired():
        input_ids = torch.randint(0, vocab, (batch_size, args.seq_len), device=device)
        logits = model(input_ids)
        loss = nn.functional.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1),
        )
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        inner.step()
        inner.zero_grad(set_to_none=True)

        step += 1
        tokens += batch_size * args.seq_len

        if step % args.sync_every == 0:
            moved = local_sgd_sync(model, outer, args, world_size,
                                   codec=codec, ef=ef)
            syncs += 1
            # Only the traffic that leaves this node counts as inter-node. Take
            # the ranks-per-node figure from torchrun rather than assuming 8:
            # the s8 cells run this same file with half the world.
            if world_size > local_world:
                # `moved` is already wire bytes from the cost model, so this
                # only takes the share of them that crosses the node boundary.
                inter_node_bytes += moved * (world_size - local_world) // world_size
            if rank == 0:
                emit_step("train", step, tokens=tokens, node=node_rank, rank=rank,
                          sync=1, sync_mb=round(moved / 2**20, 1))
        elif rank == 0:
            emit_step("train", step, tokens=tokens, node=node_rank, rank=rank, sync=0)

    elapsed = time.time() - wall_start
    record_progress(args.seed, step, tokens, elapsed)
    if rank == 0:
        quiet = 100.0 * (step - syncs) / max(step, 1)
        print(f"\nLocal SGD complete: {step} steps, {syncs} syncs in {elapsed:.1f}s",
              flush=True)
        print(f"  Useful tokens: {tokens:,} ({tokens/elapsed:.0f} tok/s)", flush=True)
        print(f"  Steps with NO inter-node traffic: {quiet:.1f}%", flush=True)
        print(f"  Inter-node bytes (this rank): {inter_node_bytes/1e9:.2f} GB "
              f"= {inter_node_bytes/elapsed/1e6:.1f} MB/s "
              f"= {inter_node_bytes*8/elapsed/1e6:.0f} Mbps", flush=True)
        print(f"  EVASION_TAX_DATA: strategy=LOCALSGD world_size={world_size} "
              f"train_frac=1.0 dilution= steps={step} train_steps={step} "
              f"useful_tokens={tokens} attributed_tokens={tokens} "
              f"duration={elapsed:.1f} sync_every={args.sync_every} "
              f"syncs={syncs} size={args.size} internode_mbps="
              f"{inter_node_bytes*8/elapsed/1e6:.1f}", flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
