#!/usr/bin/env python3
"""Table 7's LOO column: one classifier trained on the WHOLE corpus, minus one
evasion family at a time.

This is a different quantity from the `unseen` column of
scripts/paper_table5_armsrace.py, and the two should not be confused:

  armsrace `unseen`   fit on corpus_N minus family f, where corpus_N holds only
                      rounds 1..N. A Round 1 family is therefore scored by a
                      detector that has seen nothing past Round 1. It answers
                      "at the time, did hardening generalise to a sibling?"
  this script         fit on EVERYTHING except family f -- every other family,
                      every round, all the benign data. It answers "given the
                      finished corpus, would a new instance of this family be
                      caught?" That is the deployment question, and it is what
                      Table 7's published "LOO mean" column claims to report.

Neither cross-validates: the family is wholly held out and predicted in one
shot. So the holdout key (job_id / run_id / workload) does not enter, and the
same numbers stand under all three protocols -- PROVIDED no job or run spans
two families, which the script asserts rather than assumes.

Control families are scored too, on their NON-training windows, where the same
number is a false-positive rate on a benign family the detector has never seen.

    PYTHONPATH=. python3 scripts/paper_table7_loo.py
    PAPER_MODEL=xgb PYTHONPATH=. python3 scripts/paper_table7_loo.py
"""
from __future__ import annotations

import argparse, json, os, sys, time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "scripts"))
from paper_table5_armsrace import (  # noqa: E402
    family_of, FAMILY_SPEC, DROP_CELLS, INSTRUMENT_CELLS, out_of_scope_runs,
    forest, _model_kind)
from classifier.threeway_improved import get_feature_cols  # noqa: E402
from classifier.workload_labels import canonical_workload  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "cache/win_30_r6.parquet"))
    ap.add_argument("--max-gpus", type=int, default=8)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    t0 = time.time()
    kind = _model_kind()

    win = pd.read_parquet(a.cache)
    win["cell"] = [canonical_workload(l) for l in win.workload_label]
    oos, _ = out_of_scope_runs(a.max_gpus, True)
    drop = win.run_id.astype(str).isin(oos) | win.cell.isin(DROP_CELLS | INSTRUMENT_CELLS)
    win = win.loc[~drop].reset_index(drop=True)
    win["family"] = [family_of(l) for l in win.workload_label]

    feats = [c for c in get_feature_cols(win)
             if c not in {"round", "cell", "family", "subgroup"}]
    X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    y = (win.threeway_label == "ml_training").to_numpy(int)
    fam = win.family.to_numpy(object)
    print(f"model={kind}  {len(win):,} windows  {len(feats)} features  "
          f"{int(y.sum()):,} training / {int((1-y).sum()):,} not", flush=True)

    # Does holding out a FAMILY also hold out whole jobs and whole workloads?
    # If a job or run straddled two families, family-LOO would leak at job level
    # and the holdout key would matter after all.
    leaks = {}
    for key in ("job_id", "run_id", "cell"):
        g = win.groupby(win[key].astype(str)).family.nunique(dropna=False)
        leaks[key] = int((g > 1).sum())
    print(f"  groups spanning >1 family:  "
          + "  ".join(f"{k} {v}" for k, v in leaks.items()), flush=True)
    if any(leaks.values()):
        print("  WARNING: family-LOO does not cleanly nest inside those groups; "
              "the holdout key is NOT moot.", flush=True)

    fams = sorted({f for f in win.family.dropna().unique()})
    rows = []
    for f in fams:
        is_ev = FAMILY_SPEC[f][1]
        # Score each family on the class it actually carries. Four of the
        # controls (undisguised PPO, grad-accum PPO, distributed RL, the AdamW
        # positive control) are TRAINING workloads, so for them this number is
        # a detection rate -- a miss there is a false negative on an honest job.
        # Only the genuinely non-training controls yield a false-positive rate.
        sel = (fam == f) & (y == 1)
        scored = "detection"
        if not sel.sum():
            sel = (fam == f) & (y == 0)
            scored = "false_positive"
        if not sel.sum():
            continue
        tr = fam != f
        clf = forest().fit(X[tr], y[tr])
        rate = float((clf.predict(X[sel]) == 1).mean())
        rows.append({
            "family": f, "round": FAMILY_SPEC[f][0], "is_evasion": is_ev,
            "scored": scored,
            "windows": int(sel.sum()),
            "cells": int(win.loc[sel, "cell"].nunique()),
            "runs": int(win.loc[sel, "run_id"].nunique()),
            "train_windows": int(tr.sum()),
            "loo_rate": rate,
            "by_cell": {c: float((clf.predict(X[sel & (win.cell == c).to_numpy()]) == 1).mean())
                        for c in sorted(win.loc[sel, "cell"].unique())},
        })
        r = rows[-1]
        print(f"  {f:24s} R{r['round']} {r['scored'][:9]:9s} n={r['windows']:5d} "
              f"cells={r['cells']:2d} -> {rate*100:6.2f}%   [{time.time()-t0:.0f}s]",
              flush=True)

    ev = [r for r in rows if r["is_evasion"]]
    mean_unw = float(np.mean([r["loo_rate"] for r in ev]))
    mean_w = float(np.average([r["loo_rate"] for r in ev],
                              weights=[r["windows"] for r in ev]))
    out = {"table": 7, "column": "LOO (whole-corpus, leave-one-family-out)",
           "model": kind, "window_sec": 30.0, "cache": a.cache,
           "holdout_key": "not applicable: no cross-validation, family held out whole",
           "groups_spanning_multiple_families": leaks,
           "n_evasion_families": len(ev),
           "loo_mean_unweighted": mean_unw, "loo_mean_window_weighted": mean_w,
           "rows": rows}
    tag = a.tag or kind
    p = PROJECT_ROOT / f"docs/camera_ready_results_20261022/paper_table7_loo_{tag}.json"
    p.write_text(json.dumps(out, indent=2) + "\n")
    print(f"\nLOO mean over {len(ev)} evasion families: "
          f"{mean_unw*100:.2f}% unweighted, {mean_w*100:.2f}% window-weighted")
    print(f"wrote {p}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
