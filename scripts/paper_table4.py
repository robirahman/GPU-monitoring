#!/usr/bin/env python3
"""Table 4: classification accuracy by GPU -- single-GPU CV and LOGO.

Published Table 4 reports, per GPU, accuracy training and testing on that GPU
alone, and accuracy when the GPU is held out entirely (leave-one-GPU-out).

Reads the prebuilt window cache so it shares one code path with every other
regenerated table; building windows in here would both waste ~26 minutes and
reintroduce the serial/parallel discrepancy.

LOGO needs no grouping: the training set spans the other eight models and the
held-out GPU supplies the whole test set. Single-GPU CV does need a grouping
key, since one GPU's runs repeat the same workloads.

    PYTHONPATH=. python3 scripts/paper_table4.py --group-by run_id
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from classifier.threeway_improved import get_feature_cols, workload_group  # noqa: E402
from scripts.regen_headline import FOLDS  # noqa: E402

CACHE = PROJECT_ROOT / "cache"


def _model_kind():
    import os
    return os.environ.get("PAPER_MODEL", "rf").strip().lower()


def forest():
    """The classifier under test. PAPER_MODEL=xgb swaps in XGBoost.

    Both configurations are the ones Table 6 compared, so a stack run with
    PAPER_MODEL=xgb is comparable with that measurement rather than being a
    third tuning. Note XGBoost has no class_weight; Table 6 ran it without one
    and the corpus is 57% positive, so the imbalance correction is small.
    """
    import os
    nj = int(os.environ.get("PAPER_NJOBS", "0")) or -1
    if _model_kind() in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                             n_jobs=nj, random_state=42,
                             eval_metric="logloss", tree_method="hist")
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=nj)


def key(w, group_by):
    if group_by == "workload":
        return workload_group(w.workload_label.values)
    return w[group_by].astype(str).to_numpy()


def single_gpu_cv(w, feats, group_by):
    y = (w.threeway_label == "ml_training").to_numpy(int)
    if len(np.unique(y)) < 2:
        return None, "only one class present on this GPU"
    X = w[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    g = key(w, group_by)
    n = min(FOLDS, len(np.unique(g)))
    if n < 2:
        return None, f"only {len(np.unique(g))} group(s) under {group_by}"
    sc = []
    for tr, te in StratifiedGroupKFold(n_splits=n, shuffle=True,
                                       random_state=42).split(X, y, g):
        if len(np.unique(y[tr])) < 2:
            continue
        sc.append(float((forest().fit(X[tr], y[tr]).predict(X[te]) == y[te]).mean()))
    return (float(np.mean(sc)), None) if sc else (None, "no scorable fold")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--group-by", default="run_id",
                    choices=["run_id", "job_id", "workload"])
    ap.add_argument("--cache", help="explicit window-cache parquet")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    a = ap.parse_args()
    out = PROJECT_ROOT / f"docs/camera_ready_results_20261022/paper_table4_{a.group_by}{a.tag}.json"

    t0 = time.time()
    win = pd.read_parquet(Path(a.cache) if a.cache
                          else CACHE / f"win_{int(a.window)}.parquet")
    win = win[win.paper_gpu.notna()]
    feats = get_feature_cols(win)
    X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    y = (win.threeway_label == "ml_training").to_numpy(int)
    gpu = win.paper_gpu.to_numpy()
    print(f"{len(win):,} windows, {win.paper_gpu.nunique()} GPU classes, "
          f"{len(feats)} features, group_by={a.group_by}", flush=True)

    rows = []
    for name in sorted(pd.unique(gpu)):
        m = gpu == name
        cv_acc, cv_note = single_gpu_cv(win[m], feats, a.group_by)
        tr = ~m
        if len(np.unique(y[tr])) < 2 or not m.sum():
            lo_acc, lo_note = None, "degenerate split"
        else:
            lo_acc = float((forest().fit(X[tr], y[tr]).predict(X[m]) == y[m]).mean())
            lo_note = None
        rows.append({"gpu": name, "windows": int(m.sum()),
                     "train_share": float(y[m].mean()),
                     "single_gpu_cv": cv_acc, "single_gpu_cv_note": cv_note,
                     "logo": lo_acc, "logo_note": lo_note})
        cvs = f"{cv_acc*100:7.2f}%" if cv_acc is not None else f"{cv_note:>8}"
        los = f"{lo_acc*100:7.2f}%" if lo_acc is not None else f"{lo_note:>8}"
        print(f"  {name:18s} n={int(m.sum()):6,}  CV {cvs}  LOGO {los}  "
              f"[{time.time()-t0:.0f}s]", flush=True)

    out.write_text(json.dumps(
        {"table": 4, "group_by": a.group_by, "window_sec": a.window,
         "folds": FOLDS, "n_features": len(feats), "rows": rows}, indent=2) + "\n")
    print(f"\n[{time.time()-t0:.0f}s] wrote {out}")


if __name__ == "__main__":
    main()
