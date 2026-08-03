#!/usr/bin/env python3
"""Re-score whitebox detection by actual per-window training activity.

Motivation
----------
A disguised-training run (WB-D / WB-L / WB-O) spends most wall-clock doing
inference and only trains on a small fraction of batches. The paper's
detection metric counts EVERY window of such a run as "should be flagged
training", so a window in which the evader did zero training but was correctly
called inference is scored as a miss. That is the wrong denominator: pausing
training to look benign is not a classifier failure — it is the evader forgoing
training. This script recomputes detection counting a window as a missed
training window only if a real training step occurred inside it.

Ground truth comes from the per-step ``STEP ts=... kind=train|infer`` lines
emitted by workloads/adversarial_whitebox.py, captured by teeing the
run_workload.py launcher (which also prints ``Run ID: <id>``) to a log:

    python scripts/run_workload.py --workload whitebox_lora_N10 -- \
        python workloads/adversarial_whitebox.py --strategy WB-L --dilution 10 \
        2>&1 | tee logs/whitebox_lora_N10_run1.log

The STEP timestamps share the time.time() epoch with the 1 Hz telemetry
collector, so they bin directly into the prediction windows recorded in
window_predictions.parquet (window_start / window_end are in the same epoch).

Usage
-----
    python scripts/label_windows_by_training.py \
        --predictions classifier/results/window_predictions.parquet \
        --step-logs logs/whitebox_*.log \
        --out classifier/results/whitebox_rescored.parquet
"""

from __future__ import annotations

import argparse
import glob
from pathlib import Path

import numpy as np
import pandas as pd

# Single source of truth for STEP-log parsing / window activity.
from classifier.window_labels import parse_step_log, match_run_id, window_activity


def rescore(predictions: pd.DataFrame, logs: list[str], run_id_overrides: dict) -> pd.DataFrame:
    """Annotate whitebox windows with training activity and corrected labels."""
    pred_ids = predictions["run_id"].unique()
    annotated = []

    for log_path in logs:
        log_run_id, steps = parse_step_log(log_path)
        rid = match_run_id(log_run_id, run_id_overrides.get(log_path), pred_ids)
        if rid is None:
            print(f"[warn] {log_path}: could not match run_id "
                  f"(log said {log_run_id!r}); skipping")
            continue
        if steps.empty:
            print(f"[warn] {log_path}: no STEP lines found; skipping")
            continue

        train_ts = steps.loc[steps["kind"] == "train", "ts"].values
        infer_ts = steps.loc[steps["kind"] == "infer", "ts"].values
        rows = predictions[predictions["run_id"] == rid].copy()
        if rows.empty:
            print(f"[warn] {log_path}: run {rid} not in predictions; skipping")
            continue

        act = window_activity(rows["window_start"].values,
                              rows["window_end"].values, train_ts, infer_ts)
        rows["n_train_steps"] = act["n_train_steps"]
        rows["n_infer_steps"] = act["n_infer_steps"]
        rows["train_active"] = act["train_active"]
        rows["step_log"] = Path(log_path).name
        annotated.append(rows)
        print(f"[ok ] {log_path}: run {rid}, {len(rows)} windows, "
              f"{len(train_ts)} train-steps, {int((n_train>0).sum())} windows with training")

    if not annotated:
        raise SystemExit("No logs matched any prediction runs — nothing to re-score.")
    return pd.concat(annotated, ignore_index=True)


def summarize(annotated: pd.DataFrame) -> pd.DataFrame:
    """Naive vs corrected detection per (run_id, workload_label).

    naive     = fraction of ALL windows predicted training
                (the paper's denominator).
    corrected = fraction of TRAINING-ACTIVE windows predicted training
                (a window with zero training steps cannot be a miss).
    """
    out = []
    key = ["step_log", "run_id", "workload_label"]
    for keyvals, g in annotated.groupby(key):
        active = g[g["train_active"]]
        naive = (g["y_pred"] == "ml_training").mean()
        corrected = (active["y_pred"] == "ml_training").mean() if len(active) else np.nan
        row = dict(zip(key, keyvals))
        row.update({
            "n_windows": len(g),
            "n_train_active_windows": int(g["train_active"].sum()),
            "naive_detection": naive,
            "corrected_detection": corrected,
        })
        out.append(row)
    return pd.DataFrame(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--predictions", default="classifier/results/window_predictions.parquet")
    ap.add_argument("--step-logs", nargs="+", required=True,
                    help="teed launcher logs (globs ok)")
    ap.add_argument("--out", default="classifier/results/whitebox_rescored.parquet")
    ap.add_argument("--run-id", nargs="*", default=[],
                    help="optional log=run_id overrides, e.g. logs/a.log=abcd1234")
    args = ap.parse_args()

    logs = []
    for pattern in args.step_logs:
        logs.extend(sorted(glob.glob(pattern)))
    if not logs:
        raise SystemExit("No step logs matched.")

    overrides = {}
    for item in args.run_id:
        if "=" in item:
            k, v = item.split("=", 1)
            overrides[k] = v

    predictions = pd.read_parquet(args.predictions)
    annotated = rescore(predictions, logs, overrides)
    annotated.to_parquet(args.out, index=False)
    print(f"\nWrote {args.out} ({len(annotated)} windows)")

    summ = summarize(annotated)
    summ_path = str(Path(args.out).with_name("whitebox_rescored_summary.csv"))
    summ.to_csv(summ_path, index=False)
    print(f"Wrote {summ_path}\n")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(summ.to_string(index=False))


if __name__ == "__main__":
    main()
