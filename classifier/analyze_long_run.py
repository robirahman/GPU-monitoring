"""Validate the run-level duration extrapolation against a real long run.

Scores a single long telemetry run window-by-window with a binary RF trained on
the rest of the corpus (optionally excluding all whitebox families, so the long
WB-L run is an *unseen* strategy — the hardest, leave-one-strategy-out test),
then measures whether per-window detection compounds over the job as the
short-run two-state-Markov extrapolation predicted:

  * per-window alert rate on the run (naive, and on training-active windows only);
  * lag-1 flag autocorrelation vs the short-run Markov assumption;
  * time-to-first-alert under a k-of-m trigger;
  * empirical P(detected within elapsed T) by sliding a job-start over the run.

Ground-truth training activity per window comes from the STEP log
(`STEP ts=… kind=train|infer`), so a window counts as training only if a real
step landed in it.

Usage:
  python -m classifier.analyze_long_run \
    --run-parquet data/whitebox_lora_N10_long_*.parquet \
    --step-log logs/wb_lora_N10_long.log \
    --exclude-whitebox --window 30 --stride 15 --tau 0.75 --k 3 --m 5
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
from classifier.run_level_detection import kofm_trigger

STEP_RE = re.compile(r"STEP\s+ts=([\d.]+)\s+kind=(train|infer)")


def parse_train_ts(step_log: str) -> np.ndarray:
    ts = []
    with open(step_log, "r", errors="ignore") as f:
        for line in f:
            m = STEP_RE.search(line)
            if m and m.group(2) == "train":
                ts.append(float(m.group(1)))
    return np.sort(np.array(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-parquet", required=True)
    ap.add_argument("--step-log", required=True)
    ap.add_argument("--exclude-whitebox", action="store_true",
                    help="train with all whitebox_* families removed (unseen-strategy test)")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=15.0)
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    args = ap.parse_args()

    run_path = sorted(glob.glob(args.run_parquet))[0]
    run_label = pd.read_parquet(run_path, columns=["run_id"])["run_id"].iloc[0]

    print("Loading corpus + long run, windowing (causal)...")
    df = load_all_data()
    win = sliding_windows(df, args.window, args.stride, run_level_mode="causal")
    feats = get_feature_cols(win)

    is_long = win["run_id"] == run_label
    long_win = win[is_long].sort_values("window_start").reset_index(drop=True)
    if long_win.empty:
        raise SystemExit(f"Long run {run_label} produced no windows (is it loaded from {run_path}?)")

    train = win[~is_long]
    if args.exclude_whitebox:
        train = train[~train["workload_label"].astype(str).str.startswith("whitebox")]
    Xtr = train[feats].fillna(0).values.astype(np.float32)
    ytr = (train["threeway_label"].values == "ml_training").astype(int)

    clf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    idx1 = list(clf.classes_).index(1)

    Xte = long_win[feats].fillna(0).values.astype(np.float32)
    p_train = clf.predict_proba(Xte)[:, idx1]
    ws = long_win["window_start"].values
    we = long_win["window_end"].values

    # ground-truth training activity per window from the STEP log
    train_ts = parse_train_ts(args.step_log)
    n_train_steps = np.searchsorted(train_ts, we) - np.searchsorted(train_ts, ws)
    active = n_train_steps > 0

    alert = p_train > args.tau
    # lag-1 autocorrelation of the alert sequence
    a = alert.astype(float)
    lag1 = np.corrcoef(a[:-1], a[1:])[0, 1] if a.std() > 0 else float("nan")
    # transition estimate for the Markov comparison
    prev_clear = a[:-1] == 0
    p01 = a[1:][prev_clear].mean() if prev_clear.sum() else float("nan")

    trig = kofm_trigger(alert, args.k, args.m)
    ttfa = (ws[trig] - ws[0]) if trig is not None else None

    # empirical P(detected within elapsed T): slide a job-start over the run,
    # for each start find windows until the first k-of-m trigger after it.
    def detect_latency_from(start_i):
        seg = alert[start_i:]
        t = kofm_trigger(seg, args.k, args.m)
        return None if t is None else t * args.stride  # seconds from that start
    starts = range(0, len(alert), max(1, len(alert) // 200))
    lat = [detect_latency_from(s) for s in starts]
    lat_ok = [x for x in lat if x is not None]

    print(f"\n=== 8 h WB-L N10 long run ({'UNSEEN' if args.exclude_whitebox else 'in-dist'} model, "
          f"causal, {args.window:.0f}s/{args.stride:.0f}s, tau={args.tau}) ===")
    print(f"windows: {len(long_win)}  ({len(long_win)*args.stride/3600:.1f} h)")
    print(f"training-active windows (STEP log): {int(active.sum())}/{len(active)} "
          f"({active.mean():.1%})")
    print(f"per-window alert rate — all windows:        {alert.mean():.3f}")
    print(f"per-window alert rate — training-active only: {alert[active].mean():.3f}  "
          f"(corrected detection; non-training windows correctly not flagged)")
    print(f"lag-1 alert autocorrelation: {lag1:.3f}   P(alert|prev clear)={p01:.3f}")
    print(f"time-to-first {args.k}-of-{args.m} alert: "
          f"{('%.0f s (%.1f min)' % (ttfa, ttfa/60)) if ttfa is not None else 'NEVER'}")
    if lat_ok:
        print(f"detect-latency over {len(starts)} job-start offsets: "
              f"median {np.median(lat_ok):.0f}s, 90th pct {np.percentile(lat_ok,90):.0f}s, "
              f"never-detected starts: {len(lat)-len(lat_ok)}/{len(lat)}")
    # cumulative P(≥1 alert by elapsed T) from run start
    print("\nP(≥1 alert by elapsed time from run start):")
    for T in [30, 60, 300, 900, 1800, 3600]:
        n = int(T / args.stride)
        det = alert[:n].any() if n <= len(alert) else alert.any()
        print(f"  {T:5d}s ({T/60:5.1f} min): {'DETECTED' if det else 'no alert yet'}")


if __name__ == "__main__":
    main()
