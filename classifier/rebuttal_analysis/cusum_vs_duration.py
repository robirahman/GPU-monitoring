#!/usr/bin/env python3
"""How long must a disguised (L4 fused-write) run be before CUSUM catches it?

Deconfounds model size from run duration. The earlier ladder confounded them
(1.1B was short, 7B was long). Here we score a *long* 1.1B L4 run and a *short*
7B L4 run, then ask two things:

  1. Per-window signal: is P(train) per window the same across model sizes? If so,
     catchability is set by run length, not size.
  2. Time-to-catch: sweeping the run length (prefixes of the long 1.1B run),
     at what duration does the CUSUM cross the alert threshold, for a given
     false-alert budget?

Classifier: binary RF (ml_training vs rest), trained on the causal-windowed
corpus with the two deconfound runs held out (in-distribution monitor). CUSUM
threshold h is calibrated on benign inference runs to a false-alert budget, as
in cusum_detect.py.
"""
from __future__ import annotations
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols)
from classifier.rebuttal_analysis.cusum_detect import cusum_first_alert

CK = 0.25          # CUSUM reference/drift
LONG = "symgemm_L4_sgdfused_long1b"
SHORT = "symgemm_L4_sgdfused_short7b"
STRIDE = 15.0      # s per window step (30 s window / 15 s stride)


def cusum_traj(p, k):
    S, out = 0.0, []
    for pi in p:
        S = max(0.0, S + (pi - k))
        out.append(S)
    return np.array(out)


def main():
    print("Loading + windowing corpus (causal)...")
    win = sliding_windows(load_all_data(), 30, 15, run_level_mode="causal").reset_index(drop=True)
    feats = get_feature_cols(win)
    wl = win["workload_label"].astype(str)

    is_new = wl.str.startswith(LONG) | wl.str.startswith(SHORT)
    train = win[~is_new]
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(train[feats].fillna(0).values.astype(np.float32),
            (train["threeway_label"].values == "ml_training").astype(int))
    i1 = list(clf.classes_).index(1)

    def p_series(prefix):
        m = wl.str.startswith(prefix)
        g = win[m].sort_values("window_start")
        if len(g) == 0:
            return None
        return clf.predict_proba(g[feats].fillna(0).values.astype(np.float32))[:, i1]

    # Calibrate h on benign inference runs to false-alert budgets.
    benign = win[(win["threeway_label"] == "ml_inference") & ~wl.str.startswith("symgemm_")]
    b_runs = []
    for rid, g in benign.groupby("run_id"):
        g = g.sort_values("window_start")
        p = clf.predict_proba(g[feats].fillna(0).values.astype(np.float32))[:, i1]
        dur = float(g["window_end"].iloc[-1] - g["window_start"].iloc[0])
        b_runs.append((p, max(dur, 1.0) / 3600.0))
    gpu_h = sum(d for _, d in b_runs)

    def h_for_budget(budget):
        for h in np.arange(2.0, 80.0, 0.5):
            fp = sum(1 for p, _ in b_runs if cusum_first_alert(p, CK, h) is not None) / gpu_h
            if fp <= budget:
                return h, fp
        return 80.0, float("nan")

    pl, ps = p_series(LONG), p_series(SHORT)
    print(f"\nbenign pool {len(b_runs)} runs, {gpu_h:.1f} GPU-h")
    for name, p in [("1.1B LONG", pl), ("7B SHORT", ps)]:
        if p is None:
            print(f"  {name}: NOT FOUND (collection incomplete?)"); continue
        mins = len(p) * STRIDE / 60
        print(f"  {name:10s}: {len(p):4d} windows (~{mins:.0f} min)  "
              f"per-window P(train) mean={p.mean():.3f} std={p.std():.3f} p10={np.percentile(p,10):.3f}")

    print("\n== Time-to-catch vs false-alert budget ==")
    print(f"{'budget/GPU-h':>12s} {'h':>5s} {'realized FP':>11s} "
          f"{'1.1B-long cross':>16s} {'7B-short':>10s}")
    for budget in (0.20, 0.10, 0.05):
        h, fp = h_for_budget(budget)
        def cross(p):
            if p is None:
                return "n/a"
            i = cusum_first_alert(p, CK, h)
            return f"{(i+1)*STRIDE/60:.1f} min (w{i+1})" if i is not None else f"NO (<{len(p)}w)"
        print(f"{budget:>12.2f} {h:>5.1f} {fp:>11.3f} {cross(pl):>16s} {cross(ps):>10s}")

    # Minimum run length to be caught: prefix sweep of the long 1.1B run.
    if pl is not None:
        print("\n== Min run length for the 1.1B disguised run to be caught (prefix sweep) ==")
        for budget in (0.20, 0.10, 0.05):
            h, _ = h_for_budget(budget)
            i = cusum_first_alert(pl, CK, h)
            if i is None:
                print(f"  budget {budget}/GPU-h (h={h:.1f}): not caught within {len(pl)} windows")
            else:
                print(f"  budget {budget}/GPU-h (h={h:.1f}): caught at window {i+1} "
                      f"= {(i+1)*STRIDE/60:.1f} min of runtime")


if __name__ == "__main__":
    main()
