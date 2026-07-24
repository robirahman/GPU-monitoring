"""Run-level detection analysis from per-window predictions.

Window-level accuracy answers "given a 30 s snippet, is this training?"; a
governance monitor instead asks "over a whole job, do we ever alert, how often
do we false-alarm, and how fast do we catch a real run?". This script turns the
per-window predictions from dump_window_predictions.py into run-level operating
metrics, and extrapolates single-window detection to arbitrary job durations.

It answers three reviewer/author points directly:
  * M7wU Q1 — run-level detection with a fixed threshold + aggregation rule:
    missed-run rate, false alerts per GPU-hour, time-to-first-alert.
  * the "40% per window is not 40% per run" argument — a two-state Markov
    model of the per-window alert sequence gives the in-between of the
    independent (1-(1-p)^N) and perfectly-correlated (p) bounds.
  * whitebox families — run-level detection is reported per held-out family.

Aggregation rules:
  * threshold  : per-window alert = p_train > tau.
  * k-of-m     : run alerts when k of any m consecutive windows are alerts
                 (debounces isolated spikes).

tau is calibrated on benign (non-training) runs to a target false-alert rate
per GPU-hour, then applied to training runs.

Usage:
  python -m classifier.run_level_detection \
      --predictions classifier/results/window_predictions.parquet \
      --eval twostage_cv_rf --variant causal \
      --target-fp-per-gpu-hour 0.1 --k 3 --m 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"

# Adversarial-training families count as positives (a monitor must catch a
# disguised training run). Set at runtime from the label taxonomy.
TRAINING_LABEL = "ml_training"


def kofm_trigger(alerts: np.ndarray, k: int, m: int) -> int | None:
    """Index of the first window at which k of the last m windows are alerts."""
    if len(alerts) == 0:
        return None
    for i in range(len(alerts)):
        lo = max(0, i - m + 1)
        if alerts[lo:i + 1].sum() >= k:
            return i
    return None


def per_run_sequences(df: pd.DataFrame):
    """Yield (run_id, workload_label, is_training, sorted p_train array, stride)."""
    for (run_id,), g in df.groupby(["run_id"]):
        g = g.sort_values("window_start")
        is_train = (g["threeway_label"].iloc[0] == TRAINING_LABEL)
        ws = g["window_start"].values
        stride = float(np.median(np.diff(ws))) if len(ws) > 1 else np.nan
        yield run_id, g["workload_label"].iloc[0], is_train, g["p_train"].values, ws, stride


def gpu_hours(seqs, stride_default=15.0):
    """Approximate GPU-hours = sum over runs of n_windows * stride / 3600."""
    tot = 0.0
    for _, _, _, p, ws, stride in seqs:
        s = stride if np.isfinite(stride) else stride_default
        tot += len(p) * s / 3600.0
    return tot


def calibrate_tau(benign_seqs, k, m, target_fp_per_gpu_hour, taus):
    """Smallest tau whose benign false-alert rate ≤ target (per GPU-hour)."""
    gh = gpu_hours(benign_seqs)
    best = None
    table = []
    for tau in taus:
        fp_events = 0
        for _, _, _, p, ws, stride in benign_seqs:
            if kofm_trigger(p > tau, k, m) is not None:
                fp_events += 1
        rate = fp_events / gh if gh > 0 else np.nan
        table.append((tau, fp_events, rate))
        if rate <= target_fp_per_gpu_hour and best is None:
            best = tau
    return best, pd.DataFrame(table, columns=["tau", "fp_runs", "fp_per_gpu_hour"]), gh


def markov_extrapolation(train_seqs, tau, stride, durations_min):
    """Fit a two-state Markov model of the per-window alert sequence and
    extrapolate P(≥1 alert) to long durations, bracketed by the independent
    and perfectly-correlated bounds."""
    # Pool per-window alert transitions across training runs.
    n0 = n01 = 0          # windows in no-alert state; of those, transitions to alert
    n_alert = n_total = 0
    stat0_runs = []
    for _, _, _, p, ws, s in train_seqs:
        a = (p > tau).astype(int)
        n_alert += a.sum()
        n_total += len(a)
        for i in range(1, len(a)):
            if a[i - 1] == 0:
                n0 += 1
                n01 += a[i]
    p_win = n_alert / n_total if n_total else np.nan          # marginal per-window alert prob
    p01 = n01 / n0 if n0 else np.nan                          # P(alert | prev no-alert)
    # Stationary no-alert probability from the marginal.
    pi0 = 1.0 - p_win
    rows = []
    for dmin in durations_min:
        N = max(1, int(round(dmin * 60.0 / stride)))
        indep = 1.0 - (1.0 - p_win) ** N
        corr = p_win                                          # perfectly correlated bound
        # Markov: P(all N windows no-alert) ≈ pi0 * (1-p01)^(N-1)
        markov = 1.0 - (pi0 * (1.0 - p01) ** (N - 1)) if np.isfinite(p01) else np.nan
        rows.append({"duration_min": dmin, "n_windows": N,
                     "P_detect_independent": indep,
                     "P_detect_markov": markov,
                     "P_detect_correlated": corr})
    meta = {"p_window_alert": p_win, "p01_alert_given_prev_clear": p01,
            "lag1_persistence_p11": (1 - (n0 - n01) / n0) if n0 else np.nan}
    return pd.DataFrame(rows), meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default=str(RESULTS_DIR / "window_predictions.parquet"))
    ap.add_argument("--eval", default="twostage_cv_rf")
    ap.add_argument("--variant", default="causal")
    ap.add_argument("--target-fp-per-gpu-hour", type=float, default=0.1)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--out-prefix", default=str(RESULTS_DIR / "run_level"))
    args = ap.parse_args()

    df = pd.read_parquet(args.predictions)
    df = df[(df["eval_name"] == args.eval) & (df["feature_variant"] == args.variant)]
    if df.empty:
        raise SystemExit(f"No rows for eval={args.eval} variant={args.variant}. "
                         f"Available: {sorted(df['eval_name'].unique())}")
    if "p_train" not in df or df["p_train"].isna().all():
        raise SystemExit("This eval has no p_train probabilities; pick a 3-class eval.")

    seqs = list(per_run_sequences(df))
    benign = [s for s in seqs if not s[2]]
    train = [s for s in seqs if s[2]]
    stride = float(np.nanmedian([s[5] for s in seqs if np.isfinite(s[5])]))
    print(f"{len(seqs)} runs ({len(train)} training, {len(benign)} benign); "
          f"median stride ≈ {stride:.0f}s; benign GPU-h ≈ {gpu_hours(benign):.2f}")

    taus = np.round(np.arange(0.05, 0.96, 0.05), 2)
    tau, cal_table, benign_gh = calibrate_tau(benign, args.k, args.m,
                                              args.target_fp_per_gpu_hour, taus)
    if tau is None:
        tau = 0.9
        print(f"[warn] target FP rate unreachable in sweep; falling back to tau={tau}")
    print(f"\nCalibrated tau = {tau} for ≤{args.target_fp_per_gpu_hour} false alerts/GPU-h "
          f"(k={args.k}, m={args.m})")
    cal_table.to_csv(f"{args.out_prefix}_calibration.csv", index=False)

    # Run-level metrics on training runs at the calibrated operating point.
    missed = 0
    ttfa = []
    fam_rows = []
    for run_id, wl, is_train, p, ws, s in train:
        idx = kofm_trigger(p > tau, args.k, args.m)
        if idx is None:
            missed += 1
            fam_rows.append({"run_id": run_id, "workload_label": wl, "detected": 0, "ttfa_s": np.nan})
        else:
            t = ws[idx] - ws[0]
            ttfa.append(t)
            fam_rows.append({"run_id": run_id, "workload_label": wl, "detected": 1, "ttfa_s": t})
    missed_rate = missed / len(train) if train else np.nan

    # Realized benign false-alert rate at tau.
    fp_events = sum(1 for _, _, _, p, ws, s in benign if kofm_trigger(p > tau, args.k, args.m) is not None)
    fp_rate = fp_events / benign_gh if benign_gh else np.nan

    print(f"\n── Run-level operating point ──")
    print(f"  training runs:           {len(train)}")
    print(f"  missed-run rate:         {missed_rate:.1%}  ({missed}/{len(train)})")
    print(f"  median time-to-first-alert: {np.median(ttfa):.0f}s" if ttfa else "  (no detections)")
    print(f"  false alerts / GPU-hour: {fp_rate:.3f}  ({fp_events} benign runs alerted)")

    fam_df = pd.DataFrame(fam_rows)
    fam_df.to_csv(f"{args.out_prefix}_per_run.csv", index=False)
    # Per-family run-level detection (whitebox families of interest).
    fam_summary = (fam_df.assign(family=fam_df["workload_label"])
                   .groupby("family")["detected"].mean().reset_index()
                   .rename(columns={"detected": "run_level_detection"}))
    fam_summary.to_csv(f"{args.out_prefix}_per_family.csv", index=False)

    # Duration extrapolation.
    ext, meta = markov_extrapolation(train, tau, stride,
                                     durations_min=[0.5, 5, 30, 60, 240, 1440, 10080, 43200])
    ext.to_csv(f"{args.out_prefix}_duration_extrapolation.csv", index=False)
    print(f"\n── Per-window alert stats (training runs) ──")
    print(f"  P(alert | window)          = {meta['p_window_alert']:.3f}")
    print(f"  P(alert | prev clear)      = {meta['p01_alert_given_prev_clear']:.3f}")
    print(f"  persistence P(alert|prev alert) = {meta['lag1_persistence_p11']:.3f}")
    print("\n── P(≥1 alert) vs job duration ──")
    with pd.option_context("display.width", 200):
        print(ext.to_string(index=False))
    print(f"\nWrote {args.out_prefix}_*.csv")


if __name__ == "__main__":
    main()
