#!/usr/bin/env python3
"""Regression test: build_step_log_map must MERGE both nodes' logs, not replace.

`rid_for()` in multinode_queue.sh is deterministic and identical on both nodes,
and both nodes' launcher logs are gathered into logs/ and globbed together. The
original `out[rid] = {...}` therefore discarded whichever log was scanned first.

That was invisible because every multi-node workload so far emits STEP lines
from rank 0 only, so node1's log carries none and is dropped by the `steps.empty`
guard before it can do any harm. Distributed RL breaks the assumption on
purpose: an actor rank runs inference while the learner trains, so every rank
must emit, and the overwrite would have silently thrown away half the ground
truth for exactly the cells whose per-rank attribution is the point.

    python3 scripts/test_step_log_merge.py
"""

import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.window_labels import build_step_log_map, parse_step_log  # noqa: E402

RID = "abc123def456"


def _write_log(path, node, ranks, kinds, t0, n):
    """One node's launcher log: n STEP lines per rank, interleaved."""
    with open(path, "w") as f:
        f.write(f"Run ID: {RID}\n")
        for i in range(n):
            for rk, kind in zip(ranks, kinds):
                f.write(f"STEP ts={t0 + i:.3f} kind={kind} batch_idx={i} "
                        f"tokens=128 passes=1 node={node} rank={rk}\n")


def main():
    fails = []
    with tempfile.TemporaryDirectory() as td:
        td = Path(td)
        # node0: ranks 0-1, the learner side (train). node1: ranks 8-9, actors.
        # Disjoint timestamps so a merge is provable by count, and overlapping
        # roles so the by_rank split is provable by kind.
        _write_log(td / "s__n0__cell.log", 0, [0, 1], ["train", "train"], 1000.0, 5)
        _write_log(td / "s__n1__cell.log", 1, [8, 9], ["infer", "infer"], 2000.0, 7)

        m = build_step_log_map([str(td / "*.log")], [RID])

        if RID not in m:
            print("FAIL\n  run_id missing from map entirely")
            return 1
        e = m[RID]

        # 1. The merge itself: 2 ranks x 5 train lines, 2 ranks x 7 infer lines.
        if len(e["train_ts"]) != 10:
            fails.append(f"train_ts has {len(e['train_ts'])}, expected 10 "
                         f"(node0's log was overwritten)")
        if len(e["infer_ts"]) != 14:
            fails.append(f"infer_ts has {len(e['infer_ts'])}, expected 14 "
                         f"(node1's log was overwritten)")

        # 2. Sorted, because window_activity bisects into these arrays.
        for k in ("train_ts", "infer_ts"):
            if not np.all(np.diff(e[k]) >= 0):
                fails.append(f"{k} is not sorted after the merge")

        # 3. Both logs named, so provenance survives.
        if e["log"].count(",") != 1:
            fails.append(f"log field {e['log']!r} does not name both logs")

        # 4. Rank attribution: the thing the flat arrays cannot express.
        want = {(0, 0), (0, 1), (1, 8), (1, 9)}
        got = set(e["by_rank"])
        if got != want:
            fails.append(f"by_rank keys {sorted(got)} != {sorted(want)}")
        else:
            if len(e["by_rank"][(0, 0)]["train_ts"]) != 5:
                fails.append("by_rank[(0,0)] lost its train steps")
            if len(e["by_rank"][(0, 0)]["infer_ts"]) != 0:
                fails.append("learner rank (0,0) has infer steps it never emitted")
            if len(e["by_rank"][(1, 8)]["infer_ts"]) != 7:
                fails.append("by_rank[(1,8)] lost its infer steps")
            if len(e["by_rank"][(1, 8)]["train_ts"]) != 0:
                fails.append("actor rank (1,8) has train steps it never emitted")

        # 5. Legacy subset-dialect logs (no node=/rank=) must still parse, and
        #    must not invent a rank 0 that would collide with a real one.
        legacy = td / "legacy.log"
        with open(legacy, "w") as f:
            f.write(f"Run ID: {RID}\n")
            f.write("STEP ts=3000.0 kind=train batch_idx=0\n")
        _, df = parse_step_log(str(legacy))
        if len(df) != 1:
            fails.append("subset-dialect STEP line no longer parses")
        elif int(df["rank"].iloc[0]) != -1:
            fails.append(f"missing rank= became {df['rank'].iloc[0]}, not -1 "
                         f"-- an unattributed line must not masquerade as rank 0")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print(f"OK: merged 2 logs -> {len(m[RID]['train_ts'])} train + "
          f"{len(m[RID]['infer_ts'])} infer steps, 4 ranks attributed separately")
    return 0


if __name__ == "__main__":
    sys.exit(main())
