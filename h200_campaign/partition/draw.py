#!/usr/bin/env python3
"""Constrained random draw of held-out test lineages (see PREREGISTRATION.md).

Inputs: lineages.json (register), constraints.json (pre-registered counts and
rules, including the seed), feasibility.json (pre-draw feasibility results).
Attempt i draws with random.Random(f"{seed}:{i}") from the sorted eligible ids
of each pool. An attempt that violates a written constraint is logged with the
violated rule and the next attempt is used. The first valid attempt is the draw.

  python -m h200_campaign.partition.draw --out draw_result.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path

HERE = Path(__file__).resolve().parent


def eligible(register: dict, feas: dict) -> dict:
    pools = {}
    for pool, members in register["candidates"].items():
        ok = sorted(k for k in members if feas.get(k, {}).get("feasible") is True)
        pools[pool] = ok
    return pools


def check(draw: dict, register: dict, cons: dict) -> list[str]:
    bad = []
    strata = {k: v["stratum"] for k, v in register["candidates"]["training_capable"].items()}
    tr = [strata[k] for k in draw["training_capable"]]
    for need in cons["test_training_must_include_strata_any_of"]:
        if not any(s in need for s in tr):
            bad.append(f"no test training lineage from strata {need}")
    if len(set(tr)) < cons["test_training_min_distinct_strata"]:
        bad.append(f"test training lineages span {len(set(tr))} strata < {cons['test_training_min_distinct_strata']}")
    return bad


def run(register: dict, cons: dict, feas: dict, max_attempts: int = 1000) -> dict:
    pools = eligible(register, feas)
    k = cons["test_counts"]
    for pool, n in k.items():
        if len(pools[pool]) < n:
            raise SystemExit(f"infeasible before drawing: pool {pool} has {len(pools[pool])} eligible < {n}; "
                             "constraints must not be relaxed silently")
    log = []
    for i in range(max_attempts):
        rng = random.Random(f"{cons['seed']}:{i}")
        draw = {pool: sorted(rng.sample(pools[pool], k[pool])) for pool in sorted(k)}
        bad = check(draw, register, cons)
        log.append({"attempt": i, "draw": draw, "violations": bad})
        if not bad:
            test = {x for v in draw.values() for x in v}
            dev = {p: [x for x in pools[p] if x not in test] for p in pools}
            return {"seed": cons["seed"], "attempts": log, "accepted_attempt": i, "test": draw,
                    "calibration_additions": dev, "eligible_pools": pools,
                    "inputs_sha256": {n: hashlib.sha256(json.dumps(o, sort_keys=True).encode()).hexdigest()
                                      for n, o in (("lineages", register), ("constraints", cons),
                                                   ("feasibility", feas))}}
    raise SystemExit("no valid draw within max_attempts; report, do not relax constraints")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--lineages", default=str(HERE / "lineages.json"))
    ap.add_argument("--constraints", default=str(HERE / "constraints.json"))
    ap.add_argument("--feasibility", default=str(HERE / "feasibility.json"))
    ap.add_argument("--out", default=str(HERE / "draw_result.json"))
    args = ap.parse_args()
    reg = json.loads(Path(args.lineages).read_text())
    cons = json.loads(Path(args.constraints).read_text())
    feas = json.loads(Path(args.feasibility).read_text())["results"]
    res = run(reg, cons, feas)
    Path(args.out).write_text(json.dumps(res, indent=1))
    print(json.dumps({"accepted_attempt": res["accepted_attempt"], "test": res["test"],
                      "rejected_attempts": [a for a in res["attempts"] if a["violations"]]}, indent=1))


if __name__ == "__main__":
    main()
