#!/usr/bin/env python3
"""Regenerate the headline binary accuracy on the frozen camera-ready corpus.

One row per combination of the two corrections, so every change to a published
number is attributable to a specific fix rather than to "the pipeline changed":

  published   run_level_mode="orig",   job labels     -- reproduces 98.2%
  causal      run_level_mode="causal", job labels     -- the leak removed
  windows     run_level_mode="causal", window labels  -- work observed, not assumed
  steps-only  as above, restricted to label_source=="steps"

Grouped by workload, not by run: grouping by run leaks workload identity, since
every held-out run has siblings of the same workload in the training fold
(see threeway_improved.workload_group).

  PYTHONPATH=. python3 scripts/regen_headline.py --tier A_camera_ready
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.threeway_improved import (get_feature_cols, load_all_data,  # noqa: E402
                                          sliding_windows, workload_group)

WINDOW, STRIDE, FOLDS = 30.0, 15.0, 5


def evaluate(win, tag, steps_only=False):
    w = win
    if "label_source" in w.columns:
        w = w[w.label_source.ne("unresolved")]
        if steps_only:
            w = w[w.label_source.eq("steps")]
    if w.empty or w.threeway_label.nunique() < 2:
        return dict(variant=tag, windows=int(len(w)), accuracy=None,
                    note="not enough labelled windows to score")
    feats = get_feature_cols(w)
    X = w[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    y = (w.threeway_label == "ml_training").to_numpy(int)
    groups = workload_group(w.workload_label.values)
    n_splits = min(FOLDS, len(np.unique(groups)))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = []
    for tr, te in cv.split(X, y, groups):
        clf = RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        scores.append(float((clf.predict(X[te]) == y[te]).mean()))
    return dict(variant=tag, windows=int(len(w)), workloads=int(len(np.unique(groups))),
                folds=n_splits, accuracy=float(np.mean(scores)), std=float(np.std(scores)),
                training_share=float(y.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="A_camera_ready")
    ap.add_argument("--out", default="docs/camera_ready_results_20261022/headline_binary.json")
    a = ap.parse_args()

    raw = load_all_data(tier=a.tier)
    print(f"{len(raw):,} rows, {raw.run_id.nunique():,} runs", flush=True)

    results = []
    for tag, mode, events in (("published (leaky, job labels)", "orig", None),
                              ("causal, job labels", "causal", None),
                              ("causal, window labels", "causal", "auto")):
        win = sliding_windows(raw, WINDOW, STRIDE, run_level_mode=mode, work_events=events)
        print(f"  {tag}: {len(win):,} windows", flush=True)
        results.append(evaluate(win, tag))
        if events == "auto":
            results.append(evaluate(win, "causal, window labels, steps-only", steps_only=True))
        print(f"    -> {results[-1]}", flush=True)

    out = PROJECT_ROOT / a.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(dict(tier=a.tier, window_sec=WINDOW, stride_sec=STRIDE,
                                   results=results), indent=2) + "\n")
    print("\n" + pd.DataFrame(results).to_string(index=False))
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
