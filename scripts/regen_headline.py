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


def _paper_model():
    """The classifier under test. PAPER_MODEL=xgb swaps in XGBoost.

    Both configurations are the ones Table 6 compared, so a stack run with
    PAPER_MODEL=xgb stays comparable with that measurement. paper_figure3.py
    imports evaluate() from here, so patching this one site covers the window
    sweep as well as the headline.
    """
    import os
    nj = int(os.environ.get("PAPER_NJOBS", "0")) or -1
    if os.environ.get("PAPER_MODEL", "rf").strip().lower() in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                             n_jobs=nj, random_state=42,
                             eval_metric="logloss", tree_method="hist")
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=nj)
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.threeway_improved import (get_feature_cols, load_all_data,  # noqa: E402
                                          sliding_windows, workload_group)

WINDOW, STRIDE, FOLDS = 30.0, 15.0, 5


def _group_key(w, group_by):
    """CV grouping key.

    "workload" holds out whole workloads: the deployment question is whether an
    HPC job the detector has never seen trips it. "job_id" holds out whole
    physical executions, every rank together. "run_id" holds out single
    telemetry streams, so the other ranks of the same 8-GPU job remain in the
    training fold -- that is the paper's original protocol, and it is part of
    why it scores higher.
    """
    if group_by == "workload":
        return workload_group(w.workload_label.values)
    if group_by not in w.columns:
        raise KeyError(f"{group_by!r} is not a column of the window frame")
    return w[group_by].astype(str).to_numpy()


def evaluate(win, tag, steps_only=False, group_by="workload"):
    w = win
    if "label_source" in w.columns:
        w = w[w.label_source.ne("unresolved")]
        if steps_only:
            w = w[w.label_source.eq("steps")]
    if w.empty or w.threeway_label.nunique() < 2:
        return dict(variant=tag, group_by=group_by, windows=int(len(w)), accuracy=None,
                    note="not enough labelled windows to score")
    feats = get_feature_cols(w)
    X = w[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    y = (w.threeway_label == "ml_training").to_numpy(int)
    groups = _group_key(w, group_by)
    n_groups = int(len(np.unique(groups)))
    n_splits = min(FOLDS, n_groups)
    if n_splits < 2:
        return dict(variant=tag, group_by=group_by, windows=int(len(w)), accuracy=None,
                    note=f"only {n_groups} group(s) under {group_by}")
    jobs = w["job_id"].astype(str).to_numpy() if "job_id" in w.columns else None
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores, oof, job_leak = [], np.full(len(y), -1, dtype=int), 0
    for tr, te in cv.split(X, y, groups):
        clf = _paper_model()
        clf.fit(X[tr], y[tr])
        pred = clf.predict(X[te])
        oof[te] = pred
        scores.append(float((pred == y[te]).mean()))
        if jobs is not None:
            job_leak += len(set(jobs[tr]) & set(jobs[te]))

    seen = oof >= 0
    tp = int(((oof == 1) & (y == 1) & seen).sum())
    fp = int(((oof == 1) & (y == 0) & seen).sum())
    tn = int(((oof == 0) & (y == 0) & seen).sum())
    fn = int(((oof == 0) & (y == 1) & seen).sum())

    def ratio(num, den):
        return float(num / den) if den else None

    prec, rec = ratio(tp, tp + fp), ratio(tp, tp + fn)
    spec, npv = ratio(tn, tn + fp), ratio(tn, tn + fn)

    def f1(p, r):
        return float(2 * p * r / (p + r)) if p and r else None

    f1_pos, f1_neg = f1(prec, rec), f1(npv, spec)
    return dict(variant=tag, group_by=group_by, windows=int(len(w)),
                groups=n_groups,
                workloads=int(len(np.unique(workload_group(w.workload_label.values)))),
                folds=n_splits,
                accuracy=float(np.mean(scores)), std=float(np.std(scores)),
                fold_accuracies=[round(s, 6) for s in scores],
                precision=prec, recall=rec, specificity=spec,
                false_positive_rate=(None if spec is None else float(1.0 - spec)),
                macro_f1=(None if f1_pos is None or f1_neg is None
                          else float((f1_pos + f1_neg) / 2)),
                confusion=dict(tp=tp, fp=fp, tn=tn, fn=fn),
                jobs_split_across_folds=(None if jobs is None else int(job_leak)),
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
