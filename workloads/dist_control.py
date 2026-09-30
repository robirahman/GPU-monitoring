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

    recv_handles, recv_bufs = [], {}
    for t in plan:
        if t.dst == rank:
            buf = make_recv(t)
            recv_bufs[t] = buf
            recv_handles.append(dist.irecv(buf, src=t.src, group=group, tag=t.tag))

    send_handles, sent = [], 0
    for t in plan:
        if t.src == rank:
            payload = get_send(t)
            sent += payload.numel() * payload.element_size()
            send_handles.append(dist.isend(payload, dst=t.dst, group=group,
                                           tag=t.tag))

    for h in recv_handles:
        h.wait()
    for h in send_handles:
        h.wait()
    return recv_bufs, sent


def warm_p2p_ring(rank: int, world: int, device, group=None) -> None:
    """One 1-element send/recv around a ring, before the measured window opens.

    NCCL builds a P2P communicator lazily on first use between a given pair. A
    workload that first touches P2P inside its timed loop pays that setup as a
    multi-hundred-millisecond stall, which lands in the telemetry as a feature
    of the workload rather than of the harness -- and doing the FIRST one
    concurrently with a collective is the documented ncclRemoteError.

    Rank r sends to r+1 and receives from r-1, so every adjacent pair is warmed.
    A full mesh would be W*(W-1) transfers; the ring is enough to force the
    communicator machinery up without paying for pairs most plans never use.
    """
    if world < 2 or not dist.is_initialized():
        return
    nxt, prv = (rank + 1) % world, (rank - 1) % world
    send = torch.zeros(1, dtype=torch.float32, device=device)
    recv = torch.zeros(1, dtype=torch.float32, device=device)
    # Even ranks send first, odd ranks receive first: a symmetric blocking
    # send/recv pair would deadlock on a transport with no buffering.
    if rank % 2 == 0:
        dist.send(send, dst=nxt, group=group)
        dist.recv(recv, src=prv, group=group)
    else:
        dist.recv(recv, src=prv, group=group)
        dist.send(send, dst=nxt, group=group)


def new_p2p_group(ranks=None):
    """A process group reserved for point-to-point traffic.

    Created once, at startup, by every rank -- `new_group` is itself collective,
    so it cannot be called mid-run or on a subset.
    """
    if not dist.is_initialized():
        return None
    if ranks is None:
        ranks = list(range(dist.get_world_size()))
    return dist.new_group(ranks=ranks)


class P2PChannel:
    """Owns the dedicated P2P group so it cannot be forgotten at a call site.

        ch = P2PChannel()                  # once, every rank, before the loop
        ch.warm(rank, world, device)
        while not deadline.expired():
            ctl  = control.exchange(local)
            recv, sent = ch.run(make_plan(ctl), rank, get_send, make_recv)
    """

    def __init__(self, ranks=None):
        self.group = new_p2p_group(ranks)
        self.bytes_sent = 0

    def warm(self, rank: int, world: int, device) -> None:
        warm_p2p_ring(rank, world, device, group=self.group)

    def run(self, plan, rank: int, get_send, make_recv,
            paranoid: bool = False, world: int | None = None):
        recv, sent = run_plan(plan, rank, get_send, make_recv,
                              group=self.group, paranoid=paranoid, world=world)
        self.bytes_sent += sent
        return recv, sent
