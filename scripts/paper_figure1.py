#!/usr/bin/env python3
"""Figure 1: cross-GPU transfer matrix, plus the LOGO row.

Cell (i, j) is accuracy when training on GPU i and testing on GPU j. Training
happens ONCE per source GPU and is then scored against all nine targets, so the
matrix costs nine fits, not eighty-one. The LOGO row trains on the other eight
and tests on the held-out one: nine more.

The diagonal is NOT a generalisation estimate -- it trains and tests on the same
GPU's windows, so it is reported for completeness and should be read as an upper
bound, with Table 4's grouped single-GPU CV being the honest within-GPU number.

    PYTHONPATH=. python3 scripts/paper_figure1.py --window 30
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

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from classifier.threeway_improved import get_feature_cols  # noqa: E402

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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--cache", help="explicit window-cache parquet")
    ap.add_argument("--tag", default="", help="suffix for the output file")
    a = ap.parse_args()
    out = PROJECT_ROOT / f"docs/camera_ready_results_20261022/paper_figure1_{int(a.window)}s{a.tag}.json"

    win = pd.read_parquet(Path(a.cache) if a.cache
                          else CACHE / f"win_{int(a.window)}.parquet")
    win = win[win.paper_gpu.notna()]
    feats = get_feature_cols(win)
    y = (win.threeway_label == "ml_training").to_numpy(int)
    X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    gpu = win.paper_gpu.to_numpy()
    names = sorted(pd.unique(gpu))
    print(f"{len(win):,} windows, {len(names)} GPU classes, {len(feats)} features",
          flush=True)

    t0 = time.time()
    matrix, logo_row, diag_note = {}, {}, {}
    for src in names:
        m = gpu == src
        if len(np.unique(y[m])) < 2:
            diag_note[src] = "single class on this GPU; cannot train"
            continue
        clf = forest().fit(X[m], y[m])
        matrix[src] = {}
        for tgt in names:
            t = gpu == tgt
            matrix[src][tgt] = float((clf.predict(X[t]) == y[t]).mean())
        print(f"  trained on {src:18s} [{time.time()-t0:5.0f}s]  "
              f"self {matrix[src][src]:.4f}", flush=True)

    for held in names:
        tr, te = gpu != held, gpu == held
        if len(np.unique(y[tr])) < 2:
            logo_row[held] = None
            continue
        clf = forest().fit(X[tr], y[tr])
        logo_row[held] = float((clf.predict(X[te]) == y[te]).mean())
        print(f"  LOGO hold out {held:18s} [{time.time()-t0:5.0f}s]  "
              f"{logo_row[held]:.4f}", flush=True)

    doc = {"figure": 1, "window_sec": a.window, "gpus": names,
           "transfer": matrix, "logo": logo_row, "untrainable": diag_note,
           "note": ("Diagonal trains and tests on the same GPU without grouping; "
                    "it is an upper bound, not a generalisation estimate. "
                    "Table 4's grouped single-GPU CV is the honest within-GPU figure.")}
    out.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"\n[{time.time()-t0:.0f}s] wrote {out}")

    tab = pd.DataFrame(matrix).T.reindex(index=names, columns=names)
    print("\ntrain (row) -> test (col):")
    print((tab * 100).round(1).to_string())
    print("\nLOGO:", {k: None if v is None else round(v * 100, 1)
                      for k, v in logo_row.items()})


if __name__ == "__main__":
    main()
