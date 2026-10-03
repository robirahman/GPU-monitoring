#!/usr/bin/env python3
"""Agreed-schedule point-to-point: the only P2P pattern safe beside CollectiveDeadline.

Why this exists
---------------
Decoupled DiLoCo, SWARM and the distributed-RL architectures all need ranks to
do *different* things at the same moment -- a syncer merging while learners
train, a banned peer idling while its stage reroutes, an actor rolling out while
the learner steps. Every one of them is a deadlock waiting to happen next to
`workloads/dist_deadline.py`, for two reasons that are easy to miss:

1. **`CollectiveDeadline.expired()` votes with an `all_reduce` on the world
   group, gated on a LOCAL call counter** (`self._calls % self.check_every`). A
   rank that calls `expired()` a different number of times issues its collective
   at a different call index, and the job hangs. So every role-dependent branch
   must live INSIDE the tick body, never around it.

2. **No P2P may be outstanding when that vote fires.** If rank A blocks in a
   recv from B while B enters the world `all_reduce`, neither returns. This is
   the same failure `adversarial_multinode_pipeline.py` documents as
   `ncclRemoteError` on rank 15 with "last enqueued NCCL work: 0".

So the pattern is BOUNDED ASYNCHRONY:

    while not deadline.expired():          # rank-uniform, quiesced
        ctl  = control.exchange(local)     # one tiny all_gather, world group
        plan = make_plan(ctl)              # PURE: identical on every rank
        run_plan(plan, ...)                # posts recvs, then sends, waits all
                                           # -- no handle survives the call

`make_plan` being a pure function of the all-gathered control matrix is what
makes this deadlock-free: no rank ever posts a recv that another rank has not
already been told to match, and exclusion of a straggler is expressed by leaving
it OUT of the plan rather than by one side giving up on a recv -- which would
strand the sender's message in the NCCL queue and corrupt the next tick.

This is an honest simplification of true asynchrony, and it should be documented
as such wherever it is used: telemetry is sampled at 1 Hz and windowed at 30 s,
so asynchrony finer than a tick is not observable in the corpus anyway. What
matters -- ranks doing different amounts of work at different times, stragglers,
quorum exclusion, migration -- all survives the quantisation.

P2P on its own process group: a precaution, not a measured fix
--------------------------------------------------------------
`P2PChannel` owns a group created once at startup so transfers and collectives
do not share a communicator. NCCL documents that concurrent point-to-point and
collective operations on one communicator can deadlock, and keeping them apart
costs one extra communicator, so it is worth doing.

To be accurate about what was actually observed here: on gloo/CPU, transfers run
on the default group work fine, and so do transfers on a dedicated group. The
deadlock this module was first written to explain turned out to be an
unreverted test mutation that dropped the handle waits -- not a group-sharing
problem. The separation stays because the NCCL guidance is real and the cost is
one communicator; it is not a claim that this module measured a gloo failure.

Why world-group isend/irecv rather than per-edge process groups
---------------------------------------------------------------
`adversarial_multinode_pipeline.py` builds a `new_group` per pipeline edge, which
is right for it: the edge set is fixed and tiny. SWARM's edge set changes at
every rebalance, and `new_group` is itself a collective, so it cannot be called
mid-run on changing membership. Pre-creating all W*(W-1)/2 pairs costs 120 NCCL
communicators at world 16. Tagged P2P on the world group is the right tool --
provided `warm_p2p_ring()` has run and `run_plan` never interleaves P2P with a
collective.
"""

from __future__ import annotations

import hashlib
import os
from collections import namedtuple

import torch
import torch.distributed as dist

# src/dst are GLOBAL ranks; numel is the payload length; tag must be unique
# within a tick across the whole plan.
Transfer = namedtuple("Transfer", "src dst numel tag")


class PlanError(RuntimeError):
    """Raised when a plan is malformed, instead of letting it hang."""


class ControlPlane:
    """One all_gather of a small int64 vector per tick, on the world group.

    The first slot is reserved for the tick index and is checked for agreement
    across ranks.

    Be precise about what that check can and cannot do. `all_gather` is itself a
    collective, so call N on one rank always pairs with call N on another: a rank
    that simply calls `exchange()` a different NUMBER of times does not produce
    mismatched ticks, it hangs on the unmatched call. Nothing inside this class
    can catch that, and nothing tries -- the defence against it is the discipline
    in the module docstring (every role branch inside the tick body) plus
    `plan_digest` agreement in tests.

    What the tick check does catch is counter divergence at equal call counts:
    a ControlPlane constructed fresh on some ranks mid-run while others keep
    theirs -- e.g. one built inside a per-phase helper rather than once per job.
    That resets the counter on those ranks only, every subsequent plan is
    derived from a matrix whose rows describe different ticks, and without this
    check the symptom would surface much later as a mismatched transfer.
    """

    TICK_SLOT = 0

    def __init__(self, device, width: int = 8, group=None):
        if width < 1:
            raise ValueError("width must leave room for the tick slot")
        self.device = device
        self.width = int(width)
        self.group = group
        self.world = dist.get_world_size(group) if dist.is_initialized() else 1
        self.rank = dist.get_rank(group) if dist.is_initialized() else 0
        self._tick = 0

    def exchange(self, local: torch.Tensor | None = None) -> torch.Tensor:
        """Publish this rank's control vector, return the (world, width) matrix.

        `local` may be None or shorter than `width`; the tick slot is always
        overwritten by this class.
        """
        buf = torch.zeros(self.width, dtype=torch.int64, device=self.device)
        if local is not None:
            v = local.reshape(-1).to(torch.int64)
            n = min(v.numel(), self.width - 1)
            buf[1:1 + n] = v[:n]
        buf[self.TICK_SLOT] = self._tick
        self._tick += 1

        if self.world < 2 or not dist.is_initialized():
            return buf.unsqueeze(0)

        out = [torch.empty_like(buf) for _ in range(self.world)]
        dist.all_gather(out, buf, group=self.group)
        mat = torch.stack(out, dim=0)

        ticks = mat[:, self.TICK_SLOT]
        if not bool((ticks == ticks[0]).all()):
            bad = {int(r): int(t) for r, t in enumerate(ticks.tolist())}
            raise PlanError(
                "tick divergence across ranks: "
                f"{bad}. Some rank's tick body took a different number of "
                "control exchanges -- every role branch must be INSIDE the "
                "tick, not around it, or CollectiveDeadline's vote will hang.")
        return mat


def assert_plan_matched(plan, world: int) -> None:
    """Every transfer well-formed and every (src,dst,tag) unique.

    Cheap enough to call every tick. A malformed plan is a hang; this turns it
    into an exception with the offending transfer named.
    """
    seen = set()
    for t in plan:
        if not (0 <= t.src < world and 0 <= t.dst < world):
            raise PlanError(f"{t}: rank out of range for world {world}")
        if t.src == t.dst:
            raise PlanError(f"{t}: self-transfer")
        if t.numel <= 0:
            raise PlanError(f"{t}: non-positive numel")
        key = (t.src, t.dst, t.tag)
        if key in seen:
            raise PlanError(f"{t}: duplicate (src,dst,tag) -- the two payloads "
                            f"would be matched to each other's recv")
        seen.add(key)


def _is_gloo(group=None) -> bool:
    """True when `group`'s transport is gloo, so payloads must be on the host."""
    if not dist.is_initialized():
        return False
    try:
        return "gloo" in str(dist.get_backend(group)).lower()
    except Exception:                               # noqa: BLE001
        return False


def assert_plan_agreed(plan, device, group=None, tick: int | None = None) -> None:
    """All-gather the plan digest; raise unless every rank derived the same one.

    `assert_plan_matched` only checks a plan against ITSELF. It cannot see that
    two ranks built DIFFERENT plans, and that is the failure that actually costs
    a cell its whole timeout: on 2026-10-02 s16_train_swarm_s4 derived ten
    distinct plans across sixteen ranks on its very first tick, because each
    rank had seeded its router with its own wall-clock timing. The four ranks
    that happened to agree finished the tick and emitted their STEP lines; the
    other twelve blocked in recv and the cell burned 1560 s at 100% utilisation
    before the queue's timeout killed it, with nothing in the log but silence.

    MUST be called from every rank in `group`, which is why it takes the DEFAULT
    group rather than the P2P one: a collective on the point-to-point
    communicator is the deadlock `run_plan` exists to avoid. Nothing may be
    outstanding when it runs, so call it before posting any transfer.
    """
    if not dist.is_initialized():
        return
    world = dist.get_world_size(group)
    if world < 2:
        return
    mine = torch.tensor([int(plan_digest(plan)[:15], 16)],
                        dtype=torch.int64, device=device)
    out = [torch.empty_like(mine) for _ in range(world)]
    dist.all_gather(out, mine, group=group)
    vals = [int(t.item()) for t in out]
    if len(set(vals)) == 1:
        return
    groups: dict[int, list[int]] = {}
    for r, v in enumerate(vals):
        groups.setdefault(v, []).append(r)
    summary = "; ".join(f"{ranks} -> {v:x}" for v, ranks in
                        sorted(groups.items(), key=lambda kv: -len(kv[1])))
    raise PlanError(
        f"ranks derived {len(groups)} different plans"
        f"{'' if tick is None else f' at tick {tick}'}: {summary}. "
        "Every input to the plan must come from the control matrix every rank "
        "sees identically -- never from a local measurement.")


def plan_digest(plan) -> str:
    """Stable hash of a plan, for asserting every rank derived the same one.

    Order matters: two ranks that agree on the SET of transfers but disagree on
    their order will post sends and recvs in different sequences, which is a
    hang rather than a mismatch, so the digest deliberately includes order.
    """
    h = hashlib.sha1()
    for t in plan:
        h.update(f"{t.src}:{t.dst}:{t.numel}:{t.tag}|".encode())
    return h.hexdigest()[:16]


def run_plan(plan, rank: int, get_send, make_recv, group=None,
             paranoid: bool = False, world: int | None = None):
    """Execute a plan that is byte-identical on every rank.

    `group` should be a dedicated point-to-point group (see `P2PChannel`), so
    transfers never share a communicator with collectives -- NCCL documents that
    the two can deadlock when concurrent.

    `get_send(transfer) -> Tensor` supplies the payload for a transfer this rank
    sends; `make_recv(transfer) -> Tensor` supplies the buffer for one it
    receives. Returns `(received: dict[Transfer, Tensor], bytes_sent: int)`.

    ALL receives are posted before ANY send. Posting sends first can fill the
    transport's buffers with nothing draining them, which deadlocks at exactly
    the payload sizes that matter. Every handle is waited before returning, so
    nothing is outstanding when the caller next touches a collective -- that
    guarantee is the entire contract with CollectiveDeadline.
    """
    if paranoid:
        assert_plan_matched(plan, world or dist.get_world_size(group))

    # gloo moves CPU tensors only, so a CUDA payload is staged through host
    # memory. `stage` is empty on a NCCL group and the code below is the same
    # either way.
    host = _is_gloo(group)

    recv_handles, recv_bufs, stage = [], {}, []
    for t in plan:
        if t.dst == rank:
            buf = make_recv(t)
            recv_bufs[t] = buf
            into = buf
            if host and buf.device.type != "cpu":
                into = torch.empty_like(buf, device="cpu")
                stage.append((buf, into))
            recv_handles.append(dist.irecv(into, src=t.src, group=group,
                                           tag=t.tag))

    send_handles, sent = [], 0
    for t in plan:
        if t.src == rank:
            payload = get_send(t)
            sent += payload.numel() * payload.element_size()
            if host and payload.device.type != "cpu":
                payload = payload.detach().to("cpu", copy=True)
            send_handles.append(dist.isend(payload, dst=t.dst, group=group,
                                           tag=t.tag))

    for h in recv_handles:
        h.wait()
    for h in send_handles:
        h.wait()
    for dev_buf, host_buf in stage:
        dev_buf.copy_(host_buf)
    return recv_bufs, sent


def warm_p2p_ring(rank: int, world: int, device, group=None) -> None:
    """Bring the P2P group's communicator up with a RANK-UNIFORM COLLECTIVE.

    The name is historical; the ring is now opt-in and almost certainly not what
    you want. What this function exists for is the one thing that genuinely has
    to happen before a partial plan can run.

    `new_group` builds its NCCL communicator LAZILY, on first use, and that
    build is collective over the whole group. A plan where only some ranks have
    a transfer -- which is the entire point of `run_plan`, and of excluding a
    straggler by absence -- therefore cannot be the first thing to touch a fresh
    P2P group: the ranks with nothing to do skip `run_plan` entirely and race on
    to the next collective, the ranks that do have a transfer block inside the
    communicator build waiting for them, and neither side ever moves.

    Observed in full on 2026-10-03, s16_train_impala_r15, py-spy across all 16
    ranks: rank 15 (the learner) in `irecv`, ranks 9 and 12 in `isend`, and the
    ten ranks with no transfer this tick already spinning in
    CollectiveDeadline.expired(). Nothing wrong with the plan -- every rank had
    derived the same one -- and nothing wrong with the ordering inside run_plan.
    The group's communicator simply had no quorum to come up with.

    An all_reduce of one float is that quorum. Every rank calls it, it is a
    collective and so can neither deadlock on P2P ordering nor hit the
    ncclInvalidUsage that a send/recv as the group's first op does, and
    afterwards partial plans work because the group is already up.

    The old ring warm-up -- one send/recv hop per adjacent pair, to pre-pay the
    per-pair setup inside the measured window -- is kept behind SPAR_P2P_WARM=1.
    It bought a few hundred milliseconds and cost two rounds: by rank parity it
    raised "NCCL Error 5: invalid usage" on all 16 ranks, and recv-first it
    deadlocked, every rank blocked on its predecessor around a closed ring.
    """
    if not dist.is_initialized() or world < 2:
        return

    # The quorum. Not optional, and deliberately the first thing: everything
    # below is an optimisation, and an optimisation must not be able to decide
    # whether the cell runs.
    probe_dev = "cpu" if _is_gloo(group) else device
    dist.all_reduce(torch.zeros(1, dtype=torch.float32, device=probe_dev),
                    group=group)

    if os.environ.get("SPAR_P2P_WARM", "") != "1":
        return
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    send = torch.zeros(1, dtype=torch.float32, device=probe_dev)
    recv = torch.zeros(1, dtype=torch.float32, device=probe_dev)
    try:
        if rank % 2 == 0:
            dist.send(send, dst=nxt, group=group)
            dist.recv(recv, src=prv, group=group)
        else:
            dist.recv(recv, src=prv, group=group)
            dist.send(send, dst=nxt, group=group)
    except Exception as e:                      # noqa: BLE001
        print(f"[dist_control] P2P ring warm-up skipped ({type(e).__name__}: "
              f"{str(e)[:120]}); the group is up either way", flush=True)


def new_p2p_group(ranks=None, backend: str | None = None):
    """A process group reserved for point-to-point traffic, on GLOO.

    Created once, at startup, by every rank -- `new_group` is itself collective,
    so it cannot be called mid-run or on a subset.

    WHY GLOO AND NOT NCCL. Measured on two H100 nodes, 2026-10-03, with a
    sixteen-rank reproduction of one tick: a default-group all_gather, then a
    plan in which five ranks send to one, then a default-group all_reduce that
    the ten ranks with no transfer reach immediately.

        NCCL subgroup   senders never leave tick 1; the ten uninvolved ranks
                        run all six ticks and the job dies on a timeout
        NCCL default    tick 1 takes 36.3 s, then ncclRemoteError
        GLOO subgroup   six ticks, 4.2 s, every rank through

    The same plan WITHOUT the surrounding collectives completes on NCCL in
    1.3 s, so neither the plan nor run_plan's ordering is at fault: it is NCCL
    point-to-point under partial participation, interleaved with collectives
    that the non-participating ranks race ahead into. That is the shape of
    every workload here -- excluding a straggler by its ABSENCE from the plan
    is the whole design -- so the transport has to be the thing that changes.

    What this costs, stated plainly because it changes what the telemetry
    means: payloads cross host memory and TCP rather than NVLink and the
    GPU-direct path, so transfers no longer hold the GPU busy and the cells'
    utilisation during a transfer drops. The inter-node BYTES are unchanged and
    the NIC sidecar still measures them honestly, which is the measurement
    these cells exist for. Collectives are untouched -- they stay on NCCL, on
    the default group, where they have always worked.
    """
    if not dist.is_initialized():
        return None
    if ranks is None:
        ranks = list(range(dist.get_world_size()))
    backend = backend or os.environ.get("SPAR_P2P_BACKEND", "gloo")
    try:
        return dist.new_group(ranks=ranks, backend=backend)
    except Exception as e:                          # noqa: BLE001
        print(f"[dist_control] could not build a {backend} P2P group "
              f"({type(e).__name__}: {str(e)[:100]}); falling back to the "
              f"default backend, which deadlocks under partial plans",
              flush=True)
        return dist.new_group(ranks=ranks)


class P2PChannel:
    """Owns the dedicated P2P group so it cannot be forgotten at a call site.

        ch = P2PChannel()                  # once, every rank, before the loop
        ch.warm(rank, world, device)
        while not deadline.expired():
            ctl  = control.exchange(local)
            recv, sent = ch.run(make_plan(ctl), rank, get_send, make_recv)
    """

    def __init__(self, ranks=None, backend: str | None = None):
        self.group = new_p2p_group(ranks, backend=backend)
        self.bytes_sent = 0

    def warm(self, rank: int, world: int, device) -> None:
        warm_p2p_ring(rank, world, device, group=self.group)

    def run(self, plan, rank: int, get_send, make_recv,
            paranoid: bool = False, world: int | None = None):
        recv, sent = run_plan(plan, rank, get_send, make_recv,
                              group=self.group, paranoid=paranoid, world=world)
        self.bytes_sent += sent
        return recv, sent
