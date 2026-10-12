#!/usr/bin/env python3
"""Streaming and Decoupled DiLoCo: staggered fragment syncs, compressed.

Why this workload exists
------------------------
`train_localsgd.py` is plain DiLoCo: the NIC is silent for H steps and then
moves the entire parameter set in one enormous burst. Streaming DiLoCo
(arXiv 2501.18512) breaks that burst into P fragments on staggered schedules, so
the same total volume leaves as a continuous low-amplitude comb instead of a
periodic spike -- a different point in feature space, not more of the same one.
Decoupled DiLoCo (arXiv 2604.21428) then removes the global barrier entirely: a
syncer merges whatever quorum of learners has reported, and stragglers are
excluded rather than waited for.

Both are named in the user's paper (arXiv 2605.29359) as the methods that make
training over a 100 Mbps consumer link feasible, which is what makes them the
right families for this corpus.

    --mode streaming   staggered fragments, tau-overlapped, alpha-mixed
    --mode decoupled   the above plus a quorum-K syncer that never blocks

A REPLICA IS A NODE, NOT A RANK
-------------------------------
This is the one design decision that matters, and it comes out of arithmetic
rather than taste. A compressed reduction cannot use `all_reduce` -- the sum of
quantised values is not the quantisation of the sum -- so it goes by all-gather,
which costs W/2 times an all-reduce for the same payload
(see workloads/gradcompress.py). At W=16, one replica per rank, E3M0's 8x
payload reduction nets to exactly 1.00x on the wire and int8 nets to 0.50x: the
"compression" would send twice as much as sending nothing.

At W=2 the penalty is exactly 1.0 and the nominal ratio is fully realised. So
the replicas here are the two NODES: inner steps are ordinary data-parallel
all-reduces within a node's 8 ranks (NVLink, never the NIC), and the outer
fragment exchange is a 2-rank collective between node leaders carrying exactly
the compressed fragment and nothing else. That also makes the reported
inter-node byte count a measurement rather than the
`moved * (world - local) / world` estimate `train_localsgd.py` has to make.

`--replica-ranks 1` reproduces the 16-replica geometry as an ablation, and is
the configuration in which compression is expected to be worthless -- collecting
it is how that prediction gets tested rather than asserted.
"""

from __future__ import annotations

import argparse
import os
import re
import time
from datetime import timedelta

try:
    from workloads._step import emit_step
    from workloads._seed import add_seed_argument, record_seed
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads.gradcompress import make_codec, ring_allgather_bytes
    from workloads import fsdp_training as _fsdp
except ImportError:  # launched as `torchrun workloads/train_diloco_stream.py`
    from _step import emit_step
    from _seed import add_seed_argument, record_seed
    from dist_deadline import CollectiveDeadline, sync_teardown
    from gradcompress import make_codec, ring_allgather_bytes
    import fsdp_training as _fsdp

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

_fsdp.MODEL_CONFIGS.setdefault("350m", (1024, 24, 16, 4096, 32000))
RandomTransformerLM = _fsdp.RandomTransformerLM


# ── fragmentation ───────────────────────────────────────────────────────────

def build_fragments(model: nn.Module, n_fragments: int, mode: str = "strided"):
    """Partition every parameter name into n disjoint, EXHAUSTIVE fragments.

    The paper fragments transformer blocks, but a model is not only its blocks:
    tok_emb, pos_emb, the final norm and lm_head are not in any block. Leaving
    them out means they never sync -- at 1b that is ~130M parameters diverging
    across replicas forever while the job runs at full rate, moves the right
    bytes for the other 90%, and reports perfect throughput. So the non-block
    parameters are dealt round-robin over the fragments, and the test asserts
    the partition covers `named_parameters()` exactly.

    mode='strided' (the paper's preference) sends block i to fragment i % P, so
    each fragment spans the depth of the network; 'consecutive' gives contiguous
    slabs.
    """
    # The block index may be at the start of the name ("blocks.3.mlp.0.weight")
    # or nested under a parent ("body.blocks.3..."), depending on how the caller
    # wrapped the model -- match both rather than assuming a leading dot.
    blk_re = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    blocks, others = {}, []
    for name, _ in model.named_parameters():
        m = blk_re.search(name)
        if m:
            blocks.setdefault(int(m.group(1)), []).append(name)
        else:
            others.append(name)

    frags = [[] for _ in range(n_fragments)]
    n_blocks = len(blocks)
    for i in sorted(blocks):
        p = (i % n_fragments) if mode == "strided" else min(
            i * n_fragments // max(n_blocks, 1), n_fragments - 1)
        frags[p].extend(blocks[i])
    for j, name in enumerate(others):
        frags[j % n_fragments].append(name)
    return frags


def fragment_schedule(n_fragments: int, H: int):
    """Offset per fragment: fragment p syncs at steps where t % H == offset_p.

    H=100, P=2 gives offsets [0, 50] -- fragment 0 at 100, 200, ... and
    fragment 1 at 150, 250, ... An off-by-one here collapses every fragment onto
    the same step, which silently reproduces plain DiLoCo with extra machinery:
    the NIC signature reverts to train_localsgd's and the cell buys nothing.
    """
    return [int(round(p * H / n_fragments)) % H for p in range(n_fragments)]


# ── decoupled-mode arithmetic ───────────────────────────────────────────────

def token_weights(c_tokens, c_steps):
    """Decoupled DiLoCo's merge weight: quantity x quality.

        w_m = c_tokens_m * (c_tokens_m / c_steps_m)

    so a learner that processed twice the tokens at the same step count carries
    FOUR times the weight, not two. Normalised to sum to 1.
    """
    t = np.asarray(c_tokens, dtype=np.float64)
    s = np.asarray(c_steps, dtype=np.float64)
    w = t * (t / np.maximum(s, 1.0))
    total = w.sum()
    return w / total if total > 0 else np.full_like(w, 1.0 / max(len(w), 1))


def rda_merge(grads, weights):
    """Radial-Directional Averaging: average norms and directions separately.

        r = sum_i w_i * ||g_i||
        d = normalise(sum_i w_i * g_i / ||g_i||)
        out = r * d

    Equal to the weighted arithmetic mean only when the inputs are parallel.
    The point is that averaging many outer gradients shrinks their norm toward
    zero as M grows, and RDA preserves the step SIZE while averaging only the
    direction.
    """
    w = np.asarray(weights, dtype=np.float64)
    w = w / w.sum() if w.sum() > 0 else np.full_like(w, 1.0 / len(w))
    out = {}
    for name in grads[0]:
        norms = [float(g[name].norm()) for g in grads]
        r = float(sum(wi * n for wi, n in zip(w, norms)))
        d = torch.zeros_like(grads[0][name])
        for wi, g in zip(w, grads):
            n = float(g[name].norm())
            if n > 0:
                d += wi * g[name] / n
        dn = float(d.norm())
        out[name] = (d / dn * r) if dn > 0 else torch.zeros_like(d)
    return out


def select_cohort(control, quorum_k: int, grace: int, tick: int):
    """Which learners this tick's merge includes. PURE, so every rank agrees.

    `control[:, 1]` is the tick each learner last reported ready. A learner is
    eligible if that is within `grace` ticks of now. If fewer than `quorum_k`
    are eligible the syncer SKIPS the merge -- it does not wait, because waiting
    is what strands the world-group deadline vote.
    """
    eligible = [r for r in range(control.shape[0])
                if 0 <= tick - int(control[r, 1]) <= grace]
    if len(eligible) < quorum_k:
        return [], list(range(control.shape[0]))
    excluded = [r for r in range(control.shape[0]) if r not in eligible]
    return eligible, excluded


# ── the staggered, overlapped, compressed exchange ──────────────────────────

class FragmentExchange:
    """One fragment's outer sync: launch at step t, apply at t + (tau - 1).

    tau=1 blocks immediately (the paper's main setting); tau>1 lets inner steps
    run while the transfer is in flight.
    """

    def __init__(self, codec, group, world, device, alpha: float = 0.5):
        self.codec = codec
        self.group = group
        self.world = world
        self.device = device
        self.alpha = alpha
        self.pending = {}
        self.wire = 0

    def launch(self, frag: int, names, model, outer, step: int, tau: int):
        """Encode this replica's delta for the fragment and start the exchange."""
        deltas, metas, payloads = {}, {}, []
        for n in names:
            p = dict(model.named_parameters())[n]
            d = (p.data.float() - outer[n].to(p.device)).reshape(-1)
            payload, meta = self.codec.encode(d)
            deltas[n] = d
            metas[n] = meta
            payloads.append(payload.reshape(-1).view(torch.uint8))
        flat = torch.cat(payloads) if payloads else torch.zeros(
            1, dtype=torch.uint8, device=self.device)
        gathered = [torch.empty_like(flat) for _ in range(self.world)]
        handle = dist.all_gather(gathered, flat, group=self.group, async_op=True)
        self.pending[frag] = dict(names=names, metas=metas, gathered=gathered,
                                  handle=handle, ready_at=step + max(tau - 1, 0),
                                  nbytes=flat.numel())
        return flat.numel()

    def ready(self, frag: int, step: int) -> bool:
        p = self.pending.get(frag)
        return p is not None and step >= p["ready_at"]

    def apply(self, frag: int, model, outer, outer_mom, lr: float,
              momentum: float) -> int:
        """Wait the transfer, merge, take the outer step, mix back into locals."""
        p = self.pending.pop(frag)
        p["handle"].wait()
        params = dict(model.named_parameters())

        off = 0
        merged = {}
        for n in p["names"]:
            meta = p["metas"][n]
            numel_bytes = self._payload_bytes(meta, params[n].numel())
            acc = torch.zeros(meta["numel"], dtype=torch.float32,
                              device=self.device)
            for r in range(self.world):
                raw = p["gathered"][r][off:off + numel_bytes]
                acc += self.codec.decode(self._cast(raw, meta), meta).reshape(-1)
            merged[n] = acc / self.world
            off += numel_bytes

        for n in p["names"]:
            # DiLoCo's outer step, on the CPU-resident global weights.
            pseudo = -merged[n].reshape(outer[n].shape).cpu()
            buf = outer_mom[n]
            buf.mul_(momentum).add_(pseudo)
            outer[n].add_(pseudo.add(buf, alpha=momentum), alpha=-lr)
            # Streaming DiLoCo mixes rather than overwrites: the local replica
            # has moved on since the delta was computed, and alpha=0.5 is the
            # uniform average the paper uses at M=2.
            g = outer[n].to(params[n].device)
            params[n].data.mul_(1.0 - self.alpha).add_(g, alpha=self.alpha)

        self.wire += ring_allgather_bytes(p["nbytes"], self.world)
        return p["nbytes"]

    def flush(self, model, outer, outer_mom, lr, momentum) -> None:
        """Complete every outstanding exchange BEFORE sync_teardown().

        tau>1 can leave transfers in flight when the deadline fires, and
        destroying the communicator under enqueued NCCL work is the
        ncclRemoteError dist_deadline.py exists to prevent, arriving by a
        different door.
        """
        for frag in list(self.pending):
            self.apply(frag, model, outer, outer_mom, lr, momentum)

    def _payload_bytes(self, meta, numel):
        name = self.codec.name
        if name == "e3m0":
            return (meta["numel"] + 1) // 2
        if name == "int8":
            return meta["numel"]
        if name == "topk":
            return 8 * meta["k"]
        return meta["numel"] * 4

    def _cast(self, raw_u8, meta):
        if self.codec.name in ("e3m0", "int8"):
            return raw_u8 if self.codec.name == "e3m0" else raw_u8.view(torch.int8)
        return raw_u8.view(torch.float32)


# ── main ────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--mode", default="streaming", choices=("streaming", "decoupled"))
    p.add_argument("--size", default="350m", choices=sorted(_fsdp.MODEL_CONFIGS))
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--fragments", type=int, default=8)
    p.add_argument("--frag-mode", default="strided", choices=("strided", "consecutive"))
    p.add_argument("--sync-every", type=int, default=100, help="H")
    p.add_argument("--tau", type=int, default=1, help="overlap depth")
    p.add_argument("--alpha", type=float, default=0.5)
    p.add_argument("--compress", default="e3m0")
    p.add_argument("--inner-lr", type=float, default=1e-4)
    p.add_argument("--outer-lr", type=float, default=0.7)
    p.add_argument("--outer-momentum", type=float, default=0.9)
    p.add_argument("--replica-ranks", type=int, default=0,
                   help="ranks per replica; 0 means one replica per NODE, which "
                        "is the only geometry where compression pays")
    p.add_argument("--quorum", type=int, default=1, help="decoupled: K")
    p.add_argument("--grace", type=int, default=1, help="decoupled: grace ticks")
    p.add_argument("--speed-jitter", type=float, default=0.0,
                   help="decoupled: fraction of extra work on odd replicas")
    p.add_argument("--pg-timeout-min", type=int,
                   default=int(os.environ.get("SPAR_PG_TIMEOUT_MIN", "45")))
    add_seed_argument(p)
    return p.parse_args()


def main():
    args = parse_args()
    record_seed(args.seed)
    dist.init_process_group(
        backend="nccl" if torch.cuda.is_available() else "gloo",
        timeout=timedelta(minutes=args.pg_timeout_min))
    rank = dist.get_rank()
    world = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 8)) or 8
    node = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", 0)))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")

    rr = args.replica_ranks or min(local_world, world)
    n_replicas = max(world // rr, 1)
    replica = rank // rr
    is_leader = (rank % rr == 0)

    # Inner group: this replica's ranks (NVLink). Leader group: one rank per
    # replica, and the ONLY group whose traffic crosses the node boundary.
    inner_group = None
    for i in range(n_replicas):
        g = dist.new_group(ranks=list(range(i * rr, (i + 1) * rr)))
        if i == replica:
            inner_group = g
    leader_group = dist.new_group(ranks=[i * rr for i in range(n_replicas)])

    vocab = _fsdp.MODEL_CONFIGS[args.size][4]
    torch.manual_seed(args.seed)
    model = RandomTransformerLM(args.size).to(device)
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    torch.manual_seed(args.seed + replica)          # replicas differ, ranks within do not

    inner = torch.optim.AdamW(model.parameters(), lr=args.inner_lr)
    outer = {n: p.data.detach().float().cpu().clone()
             for n, p in model.named_parameters()}
    outer_mom = {n: torch.zeros_like(v) for n, v in outer.items()}

    frags = build_fragments(model, args.fragments, args.frag_mode)
    offsets = fragment_schedule(args.fragments, args.sync_every)
    codec = make_codec(args.compress)
    ex = FragmentExchange(codec, leader_group, n_replicas, device, args.alpha)

    if rank == 0:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"{args.mode} DiLoCo: params={n_par/1e6:.0f}M replicas={n_replicas} "
              f"({rr} ranks each) fragments={args.fragments}/{args.frag_mode} "
              f"H={args.sync_every} tau={args.tau} alpha={args.alpha} "
              f"compress={args.compress} offsets={offsets}", flush=True)

    deadline = CollectiveDeadline(args.duration, device, check_every=5)
    t0 = time.time()
    step = 0
    syncs = 0
    tokens = 0
    c_tokens = 0
    while not deadline.expired():
        ids = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
        logits = model(ids)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, logits.size(-1)),
                               ids[:, 1:].reshape(-1))
        loss.backward()
        if args.speed_jitter and (replica % 2 == 1):
            # Decoupled DiLoCo's own heterogeneity experiment adds ~10% speed
            # variance; this is the same idea, deterministic and reproducible.
            extra = int(args.speed_jitter * 10)
            for _ in range(extra):
                _ = (logits[:, :8, :8] @ logits[:, :8, :8].transpose(1, 2)).sum()
        if rr > 1:
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=inner_group)
                    p.grad /= rr
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        inner.step()
        inner.zero_grad(set_to_none=True)
        step += 1
        tokens += args.batch_size * args.seq_len
        c_tokens += args.batch_size * args.seq_len

        # Launch any fragment whose staggered slot is this step.
        if is_leader and n_replicas > 1:
            for p_idx, off in enumerate(offsets):
                if step % args.sync_every == off and p_idx not in ex.pending:
                    ex.launch(p_idx, frags[p_idx], model, outer, step, args.tau)
            for p_idx in list(ex.pending):
                if ex.ready(p_idx, step):
                    ex.apply(p_idx, model, outer, outer_mom,
                             args.outer_lr, args.outer_momentum)
                    syncs += 1
                    c_tokens = 0
        # Every rank in a replica adopts the leader's merged weights.
        if rr > 1 and step % args.sync_every == 0:
            for p in model.parameters():
                dist.broadcast(p.data, src=replica * rr, group=inner_group)

        if rank == 0:
            emit_step("train", step, tokens=tokens, node=node, rank=rank,
                      sync=int(syncs))

    ex.flush(model, outer, outer_mom, args.outer_lr, args.outer_momentum)
    elapsed = time.time() - t0
    if rank == 0:
        logical = sum(v.numel() * 4 for v in outer.values())
        print(f"\n{args.mode} DiLoCo complete: {step} steps, {syncs} fragment "
              f"syncs in {elapsed:.1f}s", flush=True)
        print(f"  Inter-node bytes: {ex.wire/1e9:.3f} GB "
              f"= {ex.wire*8/elapsed/1e6:.1f} Mbps", flush=True)
        print(f"  EVASION_TAX_DATA: strategy=DILOCO-{args.mode.upper()} "
              f"world_size={world} replicas={n_replicas} train_frac=1.0 "
              f"dilution= steps={step} train_steps={step} "
              f"useful_tokens={tokens} attributed_tokens={tokens} "
              f"duration={elapsed:.1f} fragments={args.fragments} "
              f"sync_every={args.sync_every} tau={args.tau} "
              f"compress={args.compress} logical_bytes={logical} "
              f"internode_mbps={ex.wire*8/elapsed/1e6:.1f}", flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
