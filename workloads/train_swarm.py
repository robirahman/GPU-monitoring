#!/usr/bin/env python3
"""SWARM parallelism: stochastic pipeline routing over unreliable peers.

Why this workload exists
------------------------
SWARM (arXiv 2301.11913) trains large models on poorly connected, unreliable
devices, and the user's paper cites it as one of the methods that makes
sub-threshold distributed training practical. Its traffic is unlike anything
else in the corpus:

  * peers serving one pipeline stage form a "swarm", and each microbatch is
    routed to whichever peer in the next stage is least loaded -- so the
    edge set is STOCHASTIC and changes microbatch to microbatch, where every
    other pipeline cell here has a fixed topology;
  * load is deliberately uneven, and a straggler is routed AROUND rather than
    waited for;
  * every so often a peer MIGRATES between stages, which is a one-off
    multi-hundred-megabyte parameter transfer at an arbitrary mid-run moment
    with nothing like it before or after;
  * activations and gradients cross the wire int8-compressed.

    torchrun --nproc-per-node=8 workloads/train_swarm.py --stages 4 --layout strided
    torchrun --nproc-per-node=8 workloads/train_swarm.py --stages 4 --layout blocked

--layout decides whether a stage straddles the node boundary (strided: stage s
holds ranks s, s+4, s+8, s+12, so roughly half of every hop crosses the NIC) or
sits inside one node (blocked: stages 0-1 on node0, 2-3 on node1, so only one
edge crosses). Same workload, either side of the boundary -- the same contrast
s16_train_tp16 against s16_train_tp8 already exploits.

What is simplified, and why
---------------------------
Real SWARM peers join and leave a DHT, and surviving peer LOSS is its whole
value proposition. torchrun plus NCCL has no fault tolerance: killing a rank
aborts the world. So a "failed" peer here is BANNED, not killed -- it stays in
the world, announces itself unavailable, takes no microbatches and keeps ticking
the deadline. What that reproduces is the rerouting and the load shift; what it
cannot reproduce is survival of an actual process death, and no claim should be
made that it does.

Routing decisions are made by each stage's lowest-numbered rank and published
through the control plane rather than taken independently per peer, because an
independently chosen destination is a receive that nobody posted. The IWRR
itself is real -- it runs on measured per-peer processing times. See
workloads/dist_control.py for why every transfer has to come from a plan that
every rank derives identically.
"""

from __future__ import annotations

import argparse
import os
import time
from datetime import timedelta

try:
    from workloads._step import emit_step
    from workloads._seed import add_seed_argument, record_seed
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads.dist_control import (
        ControlPlane, P2PChannel, Transfer, assert_plan_agreed)
    from workloads.gradcompress import Int8Codec
except ImportError:  # launched as `torchrun workloads/train_swarm.py`
    from _step import emit_step
    from _seed import add_seed_argument, record_seed
    from dist_deadline import CollectiveDeadline, sync_teardown
    from dist_control import (
        ControlPlane, P2PChannel, Transfer, assert_plan_agreed)
    from gradcompress import Int8Codec

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist


# ── routing ─────────────────────────────────────────────────────────────────

class SwarmRouter:
    """Interleaved Weighted Round-Robin over one stage's peers.

    Each peer carries its ACCUMULATED processing time; the next microbatch goes
    to the smallest. A router keyed on a cost that is never updated degenerates
    to plain round-robin, at which point the straggler injection does nothing
    and the cell quietly stops being interesting -- nothing fails, the telemetry
    just goes flat. `test_swarm_routing.py` measures the resulting share.
    """

    def __init__(self, peers):
        self.peers = sorted(peers)
        self.cost = {p: 0.0 for p in self.peers}
        self.banned = set()

    def available(self):
        return [p for p in self.peers if p not in self.banned]

    def update_cost(self, peer: int, seconds: float) -> None:
        if peer in self.cost:
            self.cost[peer] += float(seconds)

    def ban(self, peer: int) -> None:
        self.banned.add(peer)

    def unban(self, peer: int) -> None:
        self.banned.discard(peer)

    def assign(self, n_micro: int):
        """One destination peer per microbatch. Deterministic given the costs.

        Ties break on the peer index, never on dict order -- two ranks that
        tie-break differently produce different plans, which is a hang.
        """
        avail = self.available()
        if not avail:
            return []
        work = {p: self.cost[p] for p in avail}
        out = []
        for _ in range(n_micro):
            nxt = min(avail, key=lambda p: (work[p], p))
            out.append(nxt)
            work[nxt] += 1.0          # one microbatch of notional work
        return out


class StageMap:
    """rank -> stage, mutable by rebalance().

    `rebalance` must be a PURE function of the gathered queue depths: every rank
    computes the same move from the same vector, or two ranks hold different
    stage maps and every subsequent plan disagrees.
    """

    def __init__(self, world: int, n_stages: int, layout: str = "strided"):
        self.world = world
        self.n_stages = n_stages
        self.layout = layout
        if layout == "strided":
            self._stage = {r: r % n_stages for r in range(world)}
        else:
            per = max(world // n_stages, 1)
            self._stage = {r: min(r // per, n_stages - 1) for r in range(world)}

    def stage_of(self, rank: int) -> int:
        return self._stage[rank]

    def peers(self, stage: int):
        return sorted(r for r, s in self._stage.items() if s == stage)

    def rebalance(self, queue_depth):
        """Move ONE peer from the most under-utilised stage to the most
        over-utilised. Returns (rank, from_stage, to_stage) or None.

        Refuses to empty a stage: a stage with no peers is an unroutable
        pipeline, i.e. an immediate hang.
        """
        q = np.asarray(queue_depth, dtype=np.float64)
        order = sorted(range(self.n_stages), key=lambda s: (q[s], s))
        src, dst = order[0], order[-1]
        if src == dst or q[dst] <= q[src]:
            return None
        donors = self.peers(src)
        if len(donors) <= 1:
            return None
        mover = donors[-1]                      # deterministic: highest rank
        self._stage[mover] = dst
        return (mover, src, dst)


def plan_micro_round(smap: StageMap, routers, control, n_micro: int,
                     tag_base: int = 0):
    """The agreed transfer plan for one forward micro-round across all stage
    boundaries. PURE in (smap, routers, control)."""
    plan = []
    tag = tag_base
    for s in range(smap.n_stages - 1):
        srcs = [r for r in smap.peers(s) if int(control[r, 1]) == 1]
        if not srcs:
            continue
        dests = routers[s + 1].assign(n_micro)
        if not dests:
            continue
        for m in range(n_micro):
            src = srcs[m % len(srcs)]
            dst = dests[m]
            if src == dst:
                continue
            plan.append(Transfer(src=src, dst=dst, numel=1, tag=tag))
            tag += 1
    return plan


# ── model ───────────────────────────────────────────────────────────────────

class StageModule(nn.Module):
    """One pipeline stage. Stage 0 embeds, the last stage carries the head."""

    def __init__(self, hidden, layers, heads, ffn, vocab, stage, n_stages):
        super().__init__()
        self.stage, self.n_stages = stage, n_stages
        self.is_first, self.is_last = (stage == 0), (stage == n_stages - 1)
        if self.is_first:
            self.tok_emb = nn.Embedding(vocab, hidden)
            self.pos_emb = nn.Embedding(4096, hidden)
        self.blocks = nn.ModuleList([
            nn.TransformerEncoderLayer(hidden, heads, ffn, batch_first=True,
                                       norm_first=True)
            for _ in range(layers)])
        if self.is_last:
            self.norm = nn.LayerNorm(hidden)
            self.head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, x):
        if self.is_first:
            pos = torch.arange(x.shape[1], device=x.device).unsqueeze(0)
            x = self.tok_emb(x) + self.pos_emb(pos)
        for b in self.blocks:
            x = b(x)
        if self.is_last:
            return self.head(self.norm(x))
        return x


def stage_param_numel(args, stage: int, _cache={}) -> int:
    """How many parameters a given stage holds, without allocating any.

    Every rank needs this for a stage it does not serve, because a migration's
    transfer size has to be in the plan and the plan has to be identical on all
    of them. `torch.device("meta")` builds the module's shape graph and no
    storage, so asking costs nothing.
    """
    if stage not in _cache:
        with torch.device("meta"):
            m = StageModule(args.hidden, args.layers_per_stage, args.heads,
                            args.ffn, args.vocab, stage, args.stages)
        _cache[stage] = sum(p.numel() for p in m.parameters())
    return _cache[stage]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--stages", type=int, default=4)
    p.add_argument("--layout", default="strided", choices=("strided", "blocked"))
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers-per-stage", type=int, default=4)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ffn", type=int, default=4096)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--micro", type=int, default=4)
    p.add_argument("--straggler-ranks", default="",
                   help="comma-separated ranks given extra work, so IWRR routes "
                        "away from them -- which is the behaviour to record")
    p.add_argument("--straggler-factor", type=float, default=3.0)
    p.add_argument("--ban-rank", type=int, default=-1)
    p.add_argument("--ban-at", type=int, default=-1, help="tick to ban at")
    p.add_argument("--rebalance-every", type=int, default=300,
                   help="seconds; the paper's T")
    p.add_argument("--act-compress", default="int8", choices=("int8", "none"))
    p.add_argument("--lr", type=float, default=1e-4)
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

    smap = StageMap(world, args.stages, args.layout)
    routers = {s: SwarmRouter(smap.peers(s)) for s in range(args.stages)}
    stragglers = {int(x) for x in args.straggler_ranks.split(",") if x.strip()}

    torch.manual_seed(args.seed + smap.stage_of(rank))
    model = StageModule(args.hidden, args.layers_per_stage, args.heads,
                        args.ffn, args.vocab, smap.stage_of(rank),
                        args.stages).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
    codec = Int8Codec() if args.act_compress == "int8" else None

    cp = ControlPlane(device, width=4)
    ch = P2PChannel()
    ch.warm(rank, world, device)

    if rank == 0:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"SWARM: stages={args.stages} layout={args.layout} "
              f"peers/stage={world // args.stages} params/stage={n_par/1e6:.0f}M "
              f"micro={args.micro} compress={args.act_compress} "
              f"stragglers={sorted(stragglers)}", flush=True)

    deadline = CollectiveDeadline(args.duration, device, check_every=2)
    t0 = time.time()
    tick = 0
    wire = 0
    migrations = 0
    last_rebalance = t0
    act_shape = (args.batch_size, args.seq_len, args.hidden)
    act_numel = int(np.prod(act_shape))

    while not deadline.expired():
        tick += 1
        my_stage = smap.stage_of(rank)
        banned = (args.ban_rank == rank and 0 <= args.ban_at <= tick)
        avail = 0 if banned else 1

        # Local compute for this tick, timed so IWRR has a real cost to sort on.
        c0 = time.time()
        if not banned:
            # Keyed on the MODULE, not on the stage map. The two can disagree
            # for exactly one tick around a migration, and when they did -- with
            # the map already moved and the module not yet -- rank 3 fed a float
            # activation into stage 0's embedding and killed s16_train_swarm_blocked
            # after 18,267 clean ticks.
            if model.is_first:
                x = torch.randint(0, args.vocab, (args.batch_size, args.seq_len),
                                  device=device)
            else:
                x = torch.randn(act_shape, device=device)
            out = model(x)
            if rank in stragglers:
                reps = max(int(args.straggler_factor) - 1, 0)
                for _ in range(reps):
                    out = out + 0.0 * model(x)
            loss = out.float().pow(2).mean()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        dt = time.time() - c0

        # EVERY input to the plan comes out of the control matrix, because the
        # plan has to be byte-identical on all sixteen ranks and a wall-clock
        # measurement is the one thing that can never be. The obvious version of
        # this loop -- `routers[my_stage].update_cost(rank, dt)` -- seeds each
        # rank's own stage router with its own timing and leaves every other
        # stage's router pristine, so rank 1 and rank 0 disagree about where
        # stage 1 should route before the first microbatch moves. That cost
        # s16_train_swarm_s4 its entire 1560 s timeout on 2026-10-02: ten
        # distinct plans across sixteen ranks on tick 1, the four that agreed
        # emitting their STEP lines and the rest blocked in recv.
        #
        # So the cost is published as integer microseconds and EVERY rank
        # accumulates it into EVERY router, from the one matrix all of them see.
        cost_us = int(max(min(dt, 60.0), 0.0) * 1e6)
        ctl = cp.exchange(torch.tensor([avail, cost_us], dtype=torch.int64,
                                       device=device))
        for s, r in routers.items():
            for p in r.peers:
                if int(ctl[p, 1]) == 0:
                    r.ban(p)
                else:
                    r.unban(p)
                r.update_cost(p, float(ctl[p, 2]) / 1e6)

        plan = plan_micro_round(smap, routers, ctl, args.micro,
                                tag_base=(tick % 64) * 512)
        if tick == 1:
            # One extra collective, once, on the DEFAULT group -- never the P2P
            # one -- turning the whole class of plan-divergence bugs from a
            # silent timeout into a named exception on the first tick.
            assert_plan_agreed(plan, device, tick=tick)
        payload = torch.randn(act_numel, device=device)
        if codec is not None:
            enc, meta = codec.encode(payload)
            send_t = enc.view(torch.uint8).float()
        else:
            send_t = payload
        plan = [t._replace(numel=send_t.numel()) for t in plan]

        _, sent = ch.run(plan, rank,
                         get_send=lambda t: send_t,
                         make_recv=lambda t: torch.zeros(t.numel, device=device),
                         paranoid=(tick == 1), world=world)
        wire += sent

        # Rebalance on the paper's wall-clock period, from a gathered depth
        # vector every rank sees identically.
        if time.time() - last_rebalance >= args.rebalance_every:
            depth = np.array([sum(1 for p in smap.peers(s)
                                  if int(ctl[p, 1]) == 1)
                              for s in range(args.stages)], dtype=float)
            # depth is peers-available; the OVER-utilised stage is the one with
            # fewest peers, so invert before asking for a move.
            mv = smap.rebalance(-depth)
            if mv is not None:
                migrations += 1
                mover, to_stage = mv[0], mv[2]
                # A migration that moves the MAP and not the PARAMETERS is not a
                # migration -- it is a rank being told to serve a stage whose
                # weights it does not have. SWARM's own description is that the
                # moved peer's parameters arrive before it serves, and the
                # arrival is most of what a migration costs on the wire, so it
                # is the part worth measuring.
                donor = next((p for p in smap.peers(to_stage) if p != mover),
                             None)
                if donor is not None:
                    n_par = stage_param_numel(args, to_stage)
                    xfer = [Transfer(src=donor, dst=mover, numel=n_par,
                                     tag=40000 + (tick % 64))]
                    got, moved = ch.run(
                        xfer, rank,
                        get_send=lambda t: torch.cat(
                            [q.data.reshape(-1) for q in model.parameters()]),
                        make_recv=lambda t: torch.zeros(t.numel, device=device),
                        world=world)
                    wire += moved
                    if rank == mover:
                        flat = next(iter(got.values()))
                        torch.manual_seed(args.seed + to_stage)
                        model = StageModule(args.hidden, args.layers_per_stage,
                                            args.heads, args.ffn, args.vocab,
                                            to_stage, args.stages).to(device)
                        off = 0
                        with torch.no_grad():
                            for q in model.parameters():
                                k = q.numel()
                                q.copy_(flat[off:off + k].view_as(q))
                                off += k
                        opt = torch.optim.AdamW(model.parameters(), lr=args.lr)
                    # Routing has to follow the map, on every rank, or the next
                    # plan is derived from peer sets that no longer exist.
                    routers = {st: SwarmRouter(smap.peers(st))
                               for st in range(args.stages)}
                if rank == 0:
                    print(f"  rebalance: rank {mv[0]} stage {mv[1]} -> {mv[2]}"
                          f" ({stage_param_numel(args, to_stage)/1e6:.0f}M "
                          f"params pulled from rank {donor})", flush=True)
            last_rebalance = time.time()

        emit_step("train", tick, tokens=args.batch_size * args.seq_len,
                  node=node, rank=rank, stage=my_stage, avail=avail)

    elapsed = time.time() - t0
    tot = torch.tensor([float(wire)], device=device)
    dist.all_reduce(tot)
    if rank == 0:
        w = float(tot.item())
        inter = w * (world - local_world) / max(world - 1, 1)
        print(f"\nSWARM complete: {tick} ticks, {migrations} migrations in "
              f"{elapsed:.1f}s", flush=True)
        print(f"  Inter-node bytes: {inter/1e9:.2f} GB "
              f"= {inter*8/elapsed/1e6:.0f} Mbps", flush=True)
        print(f"  EVASION_TAX_DATA: strategy=SWARM world_size={world} "
              f"train_frac=1.0 dilution= steps={tick} train_steps={tick} "
              f"useful_tokens={tick * args.batch_size * args.seq_len} "
              f"attributed_tokens={tick * args.batch_size * args.seq_len} "
              f"duration={elapsed:.1f} stages={args.stages} layout={args.layout} "
              f"migrations={migrations} compress={args.act_compress} "
              f"internode_mbps={inter*8/elapsed/1e6:.1f}", flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
