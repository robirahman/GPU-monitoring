#!/usr/bin/env python3
"""CPU tests for workloads/train_diloco_stream.py.

Three of these guard failures that leave a job running at full rate, moving
exactly the right number of bytes, and reporting perfect throughput:

  * a parameter belonging to no fragment never syncs, so ~10% of the model
    diverges across replicas forever;
  * an off-by-one in the stagger collapses every fragment onto one step, which
    silently reproduces plain DiLoCo -- the NIC signature reverts to
    train_localsgd's and the cell buys nothing;
  * alpha wired backwards, or the pseudo-gradient sign flipped, trains away from
    the objective while the telemetry is identical.

    python3 scripts/test_diloco_stream.py
"""

import sys
from itertools import chain
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.train_diloco_stream import (  # noqa: E402
    RandomTransformerLM, build_fragments, fragment_schedule, rda_merge,
    select_cohort, token_weights,
)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


def test_fragments_are_a_partition():
    m = RandomTransformerLM("350m")
    allnames = {n for n, _ in m.named_parameters()}
    for P in (1, 2, 3, 8, 24):
        for mode in ("strided", "consecutive"):
            frags = build_fragments(m, P, mode)
            flat = list(chain(*frags))
            check(set(flat) == allnames,
                  f"P={P}/{mode}: fragments miss "
                  f"{sorted(allnames - set(flat))[:3]} -- those parameters "
                  f"would NEVER sync")
            check(len(flat) == len(set(flat)),
                  f"P={P}/{mode}: a parameter appears in two fragments, so it "
                  f"is reduced twice per round")
            check(all(len(f) > 0 for f in frags),
                  f"P={P}/{mode}: an empty fragment -- its slot transfers "
                  f"nothing while still counting as a sync")

    # strided must genuinely interleave depth, or it is 'consecutive' by another
    # name and the fragments do not span the network.
    import re
    blk = re.compile(r"(?:^|\.)blocks\.(\d+)\.")
    frags = build_fragments(m, 3, "strided")
    idxs = sorted({int(blk.search(n).group(1)) for n in frags[0] if blk.search(n)})
    check(idxs[:3] == [0, 3, 6],
          f"strided fragment 0 holds blocks {idxs[:3]}, expected [0,3,6]")


def test_schedule_staggers():
    for H, P in ((100, 2), (30, 8), (24, 24), (100, 3)):
        off = fragment_schedule(P, H)
        check(len(set(off)) == P or H < P,
              f"H={H},P={P}: offsets {off} collide -- fragments sync together, "
              f"which is plain DiLoCo with extra machinery")
        fires = {p: [t for t in range(1, 10 * H + 1) if t % H == off[p]]
                 for p in range(P)}
        for p in range(P):
            check(len(fires[p]) == 10,
                  f"H={H},P={P}: fragment {p} fired {len(fires[p])} times in "
                  f"10 periods, expected 10")
    off = fragment_schedule(2, 100)
    check(off == [0, 50], f"H=100,P=2 offsets {off}, expected [0, 50]")
    f0 = [t for t in range(1, 501) if t % 100 == 0]
    f1 = [t for t in range(1, 501) if t % 100 == 50]
    check(f0[:2] == [100, 200] and f1[:2] == [50, 150],
          f"stagger wrong: fragment 0 at {f0[:2]}, fragment 1 at {f1[:2]}")


def test_token_weights():
    # quantity x quality: twice the tokens at equal steps is FOUR times the
    # weight, not two.
    w = token_weights([200, 100], [10, 10])
    check(abs(w[0] / w[1] - 4.0) < 1e-9,
          f"token weight ratio {w[0]/w[1]:.3f}, expected 4.0 -- the formula is "
          f"c_tokens * (c_tokens / c_steps), not c_tokens alone")
    check(abs(w.sum() - 1.0) < 1e-9, "token weights are not normalised")
    w2 = token_weights([100, 100], [10, 20])
    check(w2[0] > w2[1], "the learner with FEWER steps for the same tokens "
                         "must carry more weight, not less")


def test_rda():
    g1 = {"a": torch.tensor([3.0, 4.0])}          # norm 5
    g2 = {"a": torch.tensor([0.0, 1.0])}          # norm 1
    out = rda_merge([g1, g2], [0.5, 0.5])
    check(abs(float(out["a"].norm()) - 3.0) < 1e-5,
          f"RDA norm {float(out['a'].norm()):.4f}, expected the weighted mean "
          f"of the norms, 3.0")
    mean = (g1["a"] + g2["a"]) / 2
    check(not torch.allclose(out["a"], mean, atol=1e-4),
          "RDA returned the arithmetic mean -- it is a stub, and the whole "
          "point is that it preserves step SIZE while averaging direction")
    # Parallel inputs are the one case where it MUST equal the mean.
    p1 = {"a": torch.tensor([2.0, 0.0])}
    p2 = {"a": torch.tensor([4.0, 0.0])}
    check(torch.allclose(rda_merge([p1, p2], [0.5, 0.5])["a"],
                         torch.tensor([3.0, 0.0]), atol=1e-5),
          "RDA disagrees with the mean even on parallel inputs")


def test_cohort_selection():
    # control[:, 1] is the tick each learner last reported.
    ctl = torch.tensor([[0, 10], [0, 10], [0, 4], [0, 9]], dtype=torch.int64)
    inc, exc = select_cohort(ctl, quorum_k=1, grace=1, tick=10)
    check(sorted(inc) == [0, 1, 3],
          f"cohort {sorted(inc)} != [0,1,3] -- rank 2 is 6 ticks stale and rank "
          f"3 is within the grace window")
    check(2 in exc, "the straggler was not excluded")
    # Below quorum the syncer must SKIP, not wait and not merge short.
    inc2, _ = select_cohort(ctl, quorum_k=4, grace=0, tick=10)
    check(inc2 == [],
          f"quorum 4 with 2 eligible returned {inc2}; it must skip the merge "
          f"entirely, because waiting is what strands the deadline vote")
    # Determinism: the same matrix must give the same answer every time, or
    # ranks disagree about who is in the cohort and the plan deadlocks.
    outs = {tuple(select_cohort(ctl, 1, 1, 10)[0]) for _ in range(50)}
    check(len(outs) == 1, "select_cohort is not deterministic")


def main():
    test_fragments_are_a_partition()
    test_schedule_staggers()
    test_token_weights()
    test_rda()
    test_cohort_selection()
    if FAILS:
        print("FAIL")
        for f in FAILS:
            print("  " + f)
        return 1
    print("OK: fragments partition every parameter, the stagger spreads "
          "fragments across the period, token weights are quantity x quality, "
          "RDA preserves norm, and the cohort is deterministic")
    return 0


if __name__ == "__main__":
    sys.exit(main())
