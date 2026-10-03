#!/usr/bin/env python3
"""CPU tests for workloads/dist_control.py.

Every failure this guards against is a HANG, not a wrong number -- and a hang
costs the cell its entire timeout and, on a rented pair, the rest of the round's
hours. So the tests check the invariants that make a hang impossible rather than
waiting to observe one:

  * plans must be byte-identical on every rank (a pure function of the
    all-gathered control matrix), including their ORDER;
  * every send must have a matching recv and vice versa;
  * no handle may survive run_plan, or CollectiveDeadline's world-group vote
    deadlocks against it;
  * exclusion must be expressed by leaving a rank out of the plan, never by
    abandoning a recv, which strands the sender's message for the next tick;
  * a rank whose tick body diverged must be caught at the control exchange with
    a message, not minutes later in an unrelated collective.

    python3 scripts/test_dist_control.py
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.dist_control import (  # noqa: E402
    ControlPlane, P2PChannel, PlanError, Transfer, assert_plan_matched,
    assert_plan_agreed, plan_digest, run_plan, warm_p2p_ring)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


# ── pure-CPU part: the planner contract ─────────────────────────────────────

def make_plan(ctl):
    """A representative pure planner: every READY rank sends to the coordinator.

    Deliberately written the way a real one is -- iterating a sorted range, never
    a set or dict -- because iteration order is what makes plans diverge.
    """
    ready = [r for r in range(ctl.shape[0]) if int(ctl[r, 1]) == 1]
    coord = 0
    return [Transfer(src=r, dst=coord, numel=8, tag=100 + r)
            for r in ready if r != coord]


def test_planner_purity():
    ctl = torch.zeros(6, 4, dtype=torch.int64)
    ctl[:, 1] = torch.tensor([1, 1, 0, 1, 0, 1])
    digests = {plan_digest(make_plan(ctl.clone())) for _ in range(20)}
    check(len(digests) == 1, "planner is not deterministic across calls")
    # Excluded ranks appear nowhere -- not as a skipped recv, not as a zero-size
    # transfer. That is the invariant that keeps a straggler's message from
    # being stranded and consumed by the next tick.
    plan = make_plan(ctl)
    srcs = {t.src for t in plan}
    check(2 not in srcs and 4 not in srcs,
          f"excluded ranks leaked into the plan: {sorted(srcs)}")
    check(srcs == {1, 3, 5}, f"plan sources {sorted(srcs)} != [1,3,5]")


def test_plan_validation():
    W = 4
    assert_plan_matched([Transfer(0, 1, 8, 1), Transfer(1, 0, 8, 2)], W)
    for bad, why in [
        ([Transfer(0, 0, 8, 1)], "self-transfer"),
        ([Transfer(0, 9, 8, 1)], "rank out of range"),
        ([Transfer(0, 1, 0, 1)], "non-positive numel"),
        ([Transfer(0, 1, 8, 1), Transfer(0, 1, 8, 1)], "duplicate (src,dst,tag)"),
    ]:
        try:
            assert_plan_matched(bad, W)
            FAILS.append(f"assert_plan_matched accepted a plan with {why}")
        except PlanError:
            pass


def test_digest_is_order_sensitive():
    a = [Transfer(1, 0, 8, 1), Transfer(2, 0, 8, 2)]
    b = [Transfer(2, 0, 8, 2), Transfer(1, 0, 8, 1)]
    check(plan_digest(a) != plan_digest(b),
          "plan_digest ignores order -- two ranks posting the same transfers "
          "in different sequences hang, and this would not detect it")


# ── distributed part ────────────────────────────────────────────────────────

def _worker(rank, world, initfile, results):
    dist.init_process_group(backend="gloo", init_method=f"file://{initfile}",
                            rank=rank, world_size=world)
    dev = torch.device("cpu")
    out = {}

    # A dedicated P2P group, so transfers never share a communicator with the
    # control exchanges. NCCL documents that concurrent P2P and collective ops
    # on one communicator can deadlock; on gloo/CPU both arrangements work, so
    # this is a precaution carried for the GPU path, not a measured fix.
    ch = P2PChannel()
    ch.warm(rank, world, dev)
    out["warmed"] = True

    cp = ControlPlane(dev, width=4)

    # ── tick 1: both ranks ready, rank 1 -> rank 0 ──
    local = torch.tensor([1], dtype=torch.int64)
    ctl = cp.exchange(local)
    out["ctl_shape"] = tuple(ctl.shape)
    plan = make_plan(ctl)
    out["digest_t1"] = plan_digest(plan)

    payload = torch.full((8,), float(rank + 1))
    recvd, sent = ch.run(
        plan, rank,
        get_send=lambda t: payload,
        make_recv=lambda t: torch.zeros(t.numel),
        paranoid=True, world=world)
    out["sent_t1"] = sent
    out["recv_t1"] = {f"{t.src}->{t.dst}": float(v[0]) for t, v in recvd.items()}

    # ── tick 2: rank 1 declares itself NOT ready; the plan must be empty and
    #    neither rank may block. This is the exclusion path. ──
    ctl2 = cp.exchange(torch.tensor([0 if rank == 1 else 1], dtype=torch.int64))
    plan2 = make_plan(ctl2)
    out["digest_t2"] = plan_digest(plan2)
    out["plan2_len"] = len(plan2)
    _, sent2 = ch.run(plan2, rank,
                      get_send=lambda t: payload,
                      make_recv=lambda t: torch.zeros(t.numel))
    out["sent_t2"] = sent2

    # ── tick 3: a collective immediately after run_plan. If any handle had
    #    survived, this is where it would deadlock -- which is exactly the
    #    interaction with CollectiveDeadline's vote. ──
    probe = torch.ones(1)
    dist.all_reduce(probe)
    out["post_plan_allreduce"] = float(probe.item())

    # ── tick 4: counter divergence at EQUAL call counts. Rank 1 builds a fresh
    #    ControlPlane (tick resets to 0) while rank 0 keeps the one it has been
    #    using -- the shape of the bug where the plane is constructed inside a
    #    per-phase helper instead of once per job. Both ranks still call
    #    exchange() exactly once, so this is not a hang; it is two ranks
    #    describing different ticks in the same matrix, and it must be reported.
    #
    #    (A rank calling exchange() a DIFFERENT number of times is not testable
    #    here and not detectable there: all_gather pairs call N with call N, so
    #    the odd call simply has no partner and blocks. That is what the
    #    rank-uniform tick discipline exists to prevent.)
    out["divergence_caught"] = False
    try:
        plane = ControlPlane(dev, width=4) if rank == 1 else cp
        plane.exchange(local)
    except PlanError as e:
        out["divergence_caught"] = True
        out["divergence_msg"] = str(e)[:80]

    # ── tick 5: assert_plan_agreed. The plan-divergence guard is the only
    #    thing standing between a wrong routing input and a cell that burns its
    #    whole timeout in silence, so it is exercised both ways: an agreed plan
    #    must pass without comment, and a plan one rank built differently must
    #    raise on EVERY rank -- a guard that fires on only the odd rank out
    #    leaves the others blocked in recv, which is the hang it exists to
    #    replace.
    agreed = [Transfer(src=1, dst=0, numel=8, tag=7)]
    out["agree_ok"] = True
    try:
        assert_plan_agreed(agreed, dev, tick=5)
    except PlanError as e:
        out["agree_ok"] = False
        out["agree_err"] = str(e)[:120]

    skewed = agreed if rank == 0 else [Transfer(src=1, dst=0, numel=8, tag=9)]
    out["disagree_caught"] = False
    try:
        assert_plan_agreed(skewed, dev, tick=6)
    except PlanError as e:
        out["disagree_caught"] = True
        out["disagree_msg"] = str(e)[:140]

    results[rank] = out
    try:
        dist.destroy_process_group()
    except Exception:
        pass


def test_distributed():
    world = 2
    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as td:
        mgr = mp.Manager()
        results = mgr.dict()
        mp.spawn(_worker, args=(world, os.path.join(td, "rdzv"), results),
                 nprocs=world, join=True)
        results = dict(results)

    if len(results) != world:
        print(f"FAIL\n  only {len(results)}/{world} ranks reported -- a hang")
        return

    r0, r1 = results[0], results[1]

    check(r0["ctl_shape"] == (world, 4),
          f"control matrix shape {r0['ctl_shape']} != {(world, 4)}")

    # The invariant: identical plans on every rank, byte for byte.
    check(r0["digest_t1"] == r1["digest_t1"],
          f"ranks derived DIFFERENT plans: {r0['digest_t1']} vs {r1['digest_t1']}")
    check(r0["digest_t2"] == r1["digest_t2"], "plans diverged on the excluded tick")

    # Rank 1 sent 8 floats; rank 0 sent nothing and received rank 1's value.
    check(r1["sent_t1"] == 32, f"rank 1 sent {r1['sent_t1']} B, expected 32")
    check(r0["sent_t1"] == 0, f"rank 0 (the coordinator) sent {r0['sent_t1']} B")
    check(r0["recv_t1"].get("1->0") == 2.0,
          f"rank 0 received {r0['recv_t1']}, expected {{'1->0': 2.0}}")

    # Exclusion: no transfers at all, and nobody blocked.
    check(r0["plan2_len"] == 0,
          f"excluded tick still planned {r0['plan2_len']} transfers")
    check(r0["sent_t2"] == 0 and r1["sent_t2"] == 0,
          "bytes moved on a tick whose plan was empty")

    # A collective right after run_plan must complete -- the quiescence contract.
    check(r0["post_plan_allreduce"] == 2.0,
          "the all_reduce after run_plan did not see both ranks; a handle "
          "survived run_plan and the deadline vote would deadlock on it")

    check(r0["divergence_caught"] or r1["divergence_caught"],
          "tick-counter divergence was NOT caught -- a ControlPlane rebuilt on "
          "some ranks mid-run would silently derive plans from a matrix whose "
          "rows describe different ticks")

    # assert_plan_agreed, both directions.
    check(r0["agree_ok"] and r1["agree_ok"],
          f"assert_plan_agreed rejected a plan both ranks agreed on: "
          f"{r0.get('agree_err') or r1.get('agree_err')}")
    check(r0["disagree_caught"] and r1["disagree_caught"],
          f"plan divergence was not raised on BOTH ranks "
          f"(rank0={r0['disagree_caught']}, rank1={r1['disagree_caught']}) -- a "
          f"guard that fires on only one rank leaves the other blocked in recv, "
          f"which is the hang it exists to replace")
    check("2 different plans" in r0.get("disagree_msg", ""),
          f"the divergence message does not name the split: "
          f"{r0.get('disagree_msg')!r}")


def main():
    test_planner_purity()
    test_plan_validation()
    test_digest_is_order_sensitive()
    test_distributed()
    if FAILS:
        print("FAIL")
        for f in FAILS:
            print("  " + f)
        return 1
    print("OK: plans identical across ranks, sends and recvs matched, nothing "
          "outstanding after run_plan, exclusion silent, tick divergence caught")
    return 0


if __name__ == "__main__":
    sys.exit(main())
