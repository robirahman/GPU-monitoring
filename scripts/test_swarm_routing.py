#!/usr/bin/env python3
"""CPU tests for workloads/train_swarm.py's routing and stage map.

The planner and StageMap are deliberately PURE functions so all 16 ranks can be
simulated in one process -- which is the only practical way to assert the
property that matters: every rank must derive a byte-identical plan. A plan two
ranks disagree about is not a wrong answer, it is a hang, and a hang costs the
cell its whole timeout and the round its remaining hours.

The quieter failure is an IWRR keyed on a cost that never updates: it degenerates
to plain round-robin, the straggler injection stops doing anything, and the cell
reduces to an ordinary pipeline with extra machinery. Nothing fails; the
telemetry just stops being interesting, which no later analysis would catch.

    python3 scripts/test_swarm_routing.py
"""

import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.dist_control import assert_plan_matched, plan_digest  # noqa: E402
from workloads.train_swarm import (  # noqa: E402
    StageMap, SwarmRouter, plan_micro_round,
)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def test_iwrr_is_weighted():
    """Four peers, one of them 3x slower. IWRR must route away from it."""
    r = SwarmRouter([0, 1, 2, 3])
    costs = {0: 1.0, 1: 1.0, 2: 1.0, 3: 3.0}
    counts = {p: 0 for p in r.peers}
    for _ in range(400):
        for p in r.assign(1):
            counts[p] += 1
            r.update_cost(p, costs[p])
    total = sum(counts.values())
    slow_share = counts[3] / total
    fast = [counts[p] / total for p in (0, 1, 2)]
    check(0.08 <= slow_share <= 0.20,
          f"slow peer took {slow_share:.1%} of microbatches; expected ~1/6 "
          f"(0.25 would mean the router is round-robin and the cost is ignored)")
    check(max(fast) - min(fast) < 0.05,
          f"fast peers got uneven shares {['%.3f' % f for f in fast]}")


def test_ban_reroutes():
    r = SwarmRouter([0, 1, 2, 3])
    r.ban(3)
    counts = {p: 0 for p in r.peers}
    for _ in range(400):
        for p in r.assign(1):
            counts[p] += 1
            r.update_cost(p, 1.0)
    check(counts[3] == 0,
          f"banned peer still received {counts[3]} microbatches -- those are "
          f"sends with no matching recv")
    check(all(counts[p] > 100 for p in (0, 1, 2)),
          f"work was not spread over the survivors: {counts}")
    r.unban(3)
    check(3 in r.available(), "unban did not restore the peer")


def test_router_is_deterministic():
    """Routing must depend only on the SET of peers and their costs.

    Each rank constructs its own SwarmRouter, and nothing guarantees they build
    it from an identically ORDERED peer list. If assign() is sensitive to that
    order, two ranks route differently and the plan deadlocks.

    Note on what this can and cannot catch: replacing `sorted(peers)` with
    `list(set(peers))` is NOT detectable here, because CPython hashes small
    ints to themselves so set order over rank indices is stable. The mutation
    is benign for integer ranks. What the test asserts instead is the property
    itself -- order-independence -- which stays true whatever the container.
    """
    digests = set()
    for _ in range(50):
        r = SwarmRouter([3, 1, 0, 2])        # deliberately unsorted input
        digests.add(tuple(r.assign(8)))
    check(len(digests) == 1,
          f"assign() gave {len(digests)} different answers from an identical "
          f"state -- two ranks would route differently and deadlock")

    # Same peers, four different input orders, identical cost history.
    outs = set()
    for order in ([0, 1, 2, 3], [3, 2, 1, 0], [2, 0, 3, 1], [1, 3, 0, 2]):
        r = SwarmRouter(order)
        for p, c in ((0, 2.0), (1, 1.0), (2, 1.0), (3, 5.0)):
            r.update_cost(p, c)
        outs.add(tuple(r.assign(12)))
    check(len(outs) == 1,
          f"peer-list ORDER changed the routing ({len(outs)} distinct "
          f"assignments) -- ranks that build their router differently would "
          f"send to peers that posted no receive")

    # And with every cost tied, the choice must still be index-ordered rather
    # than whatever the container yields first.
    r = SwarmRouter([3, 1, 0, 2])
    check(r.assign(4) == [0, 1, 2, 3],
          f"tied costs gave {r.assign(4)}, expected index order [0,1,2,3]")


def test_stage_map_layouts():
    s = StageMap(16, 4, "strided")
    check(s.peers(0) == [0, 4, 8, 12],
          f"strided stage 0 is {s.peers(0)}, expected [0,4,8,12] -- it must "
          f"straddle both nodes")
    b = StageMap(16, 4, "blocked")
    check(b.peers(0) == [0, 1, 2, 3],
          f"blocked stage 0 is {b.peers(0)}, expected [0,1,2,3]")
    check(all(st < 8 for st in b.peers(0) + b.peers(1)),
          "blocked layout put stages 0-1 across the node boundary")


def test_rebalance_is_pure_and_safe():
    depth = [-4.0, -4.0, -1.0, -4.0]        # stage 2 has fewest peers
    moves = set()
    for _ in range(32):
        m = StageMap(16, 4, "strided").rebalance(depth)
        moves.add(m)
    check(len(moves) == 1,
          f"rebalance returned {len(moves)} different moves from one depth "
          f"vector -- ranks would hold different stage maps")
    mv = moves.pop()
    check(mv is not None and mv[2] == 2,
          f"rebalance moved to stage {mv[2] if mv else None}, expected the "
          f"most over-utilised stage 2")

    # Must refuse to empty a stage: an unroutable pipeline is an instant hang.
    tiny = StageMap(4, 4, "strided")          # exactly one peer per stage
    check(tiny.rebalance([-1.0, -1.0, 0.0, -1.0]) is None,
          "rebalance emptied a single-peer stage, leaving the pipeline "
          "unroutable")
    # All-equal depths must produce no move at all.
    check(StageMap(16, 4, "strided").rebalance([-4.0] * 4) is None,
          "rebalance moved a peer when every stage was equally loaded")


def test_plan_is_identical_across_ranks():
    """Simulate all 16 ranks building the plan from one control matrix."""
    world, stages, micro = 16, 4, 4
    ctl = torch.ones(world, 4, dtype=torch.int64)
    ctl[7, 1] = 0                                  # rank 7 unavailable

    digests, plans = set(), []
    for _ in range(world):                         # each rank builds its own
        smap = StageMap(world, stages, "strided")
        routers = {s: SwarmRouter(smap.peers(s)) for s in range(stages)}
        for s, r in routers.items():
            for p in r.peers:
                if int(ctl[p, 1]) == 0:
                    r.ban(p)
        plan = plan_micro_round(smap, routers, ctl, micro)
        digests.add(plan_digest(plan))
        plans.append(plan)
    check(len(digests) == 1,
          f"the {world} ranks derived {len(digests)} different plans")

    plan = plans[0]
    try:
        assert_plan_matched(plan, world)
    except Exception as e:
        FAILS.append(f"plan is malformed: {e}")
    check(all(t.dst != 7 and t.src != 7 for t in plan),
          "the unavailable rank 7 appears in the plan -- exclusion must be "
          "absence, never an abandoned recv")
    tags = [t.tag for t in plan]
    check(len(tags) == len(set(tags)),
          "tag collision within one micro-round: two payloads would be matched "
          "to each other's receive")


def main():
    test_iwrr_is_weighted()
    test_ban_reroutes()
    test_router_is_deterministic()
    test_stage_map_layouts()
    test_rebalance_is_pure_and_safe()
    test_plan_is_identical_across_ranks()
    if FAILS:
        print("FAIL")
        for f in FAILS:
            print("  " + f)
        return 1
    print("OK: IWRR routes away from the straggler, bans reroute, the stage map "
          "rebalances deterministically without emptying a stage, and all 16 "
          "ranks derive one identical plan")
    return 0


if __name__ == "__main__":
    sys.exit(main())
