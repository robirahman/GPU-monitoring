"""
Controlled Dilution Sweep Evaluation
=====================================

Evaluates the dilution sweep data (dilution_sweep_N1..N20 + inference)
under three classifier conditions to produce the evasion tax curve:

  1. Baseline: Train on clean data only, test on each dilution level.
  2. LOO: Train on clean + all adversarial EXCEPT dilution family.
  3. Hardened: Train on clean + ALL adversarial incl. original L_diluted
     (but NOT the controlled sweep data itself).

Runs at multiple window sizes (5s, 15s, 30s, 60s) to validate detection
latency claims.

Produces:
  - classifier/results/dilution_sweep_eval.csv

Usage:
    python -m classifier.eval_dilution_sweep
"""

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols,
    compute_sample_weights,
    TRAINING_LABELS, INFERENCE_LABELS, OTHER_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

DILUTION_SWEEP_LABELS = [
    "dilution_sweep_N1", "dilution_sweep_N2", "dilution_sweep_N3",
    "dilution_sweep_N5", "dilution_sweep_N10", "dilution_sweep_N20",
    "dilution_sweep_inference",
]

ALL_ADVERSARIAL_PREFIXES = ["adversarial_", "dilution_sweep_"]

DILUTION_FAMILY_LABELS = {
    "adversarial_L_diluted_2", "adversarial_L_diluted_5",
    "adversarial_L_diluted_10", "adversarial_L_diluted_20",
    "dilution_sweep_N1", "dilution_sweep_N2", "dilution_sweep_N3",
    "dilution_sweep_N5", "dilution_sweep_N10", "dilution_sweep_N20",
    "dilution_sweep_inference",
}


def dilution_n(label):
    if "inference" in label:
        return float("inf")
    return int(label.split("_N")[-1])


def train_frac(label):
    n = dilution_n(label)
    return 0.0 if n == float("inf") else 1.0 / n


def evaluate_condition(clf, le, df_win, feature_cols, sweep_present,
                       condition_name, win_sec):
    """Evaluate one classifier condition on all sweep labels."""
    rows = []
    for label in sweep_present:
        mask = df_win["workload_label"] == label
        if mask.sum() == 0:
            continue
        X_test = df_win.loc[mask, feature_cols].fillna(0).values.astype(np.float32)
        y_pred = le.inverse_transform(clf.predict(X_test))

        is_training = label != "dilution_sweep_inference"
        n_t = int((y_pred == "ml_training").sum())
        n_i = int((y_pred == "ml_inference").sum())
        n_o = int((y_pred == "other").sum())

        if is_training:
            det = (y_pred == "ml_training").mean()
        else:
            det = (y_pred == "ml_inference").mean()

        rows.append({
            "condition": condition_name,
            "window_sec": win_sec,
            "label": label,
            "dilution_N": dilution_n(label),
            "train_fraction": train_frac(label),
            "detection_rate": det if is_training else np.nan,
            "inference_correct_rate": det if not is_training else np.nan,
            "n_windows": int(mask.sum()),
            "pred_training": n_t,
            "pred_inference": n_i,
            "pred_other": n_o,
        })

        tag = f"N={dilution_n(label)}" if is_training else "inference"
        log.info(f"    {label:<30s} [{tag:>10s}]  det={det:5.1%}  "
                 f"(train={n_t}, infer={n_i}, other={n_o})  n={mask.sum()}")

    return rows


def main():
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    log.info("=" * 70)
    log.info("CONTROLLED DILUTION SWEEP EVALUATION")
    log.info("=" * 70)

    log.info("Loading all telemetry data...")
    df_raw = load_all_data()

    present = set(df_raw["workload_label"].unique())
    sweep_present = [l for l in DILUTION_SWEEP_LABELS if l in present]
    log.info(f"Sweep labels found: {sweep_present}")

    le = LabelEncoder()
    le.fit(sorted(["ml_training", "ml_inference", "other"]))

    xgb_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )

    all_results = []

    for WIN_SEC, STRIDE in [(5, 5), (15, 10), (30, 15), (60, 30)]:
        log.info(f"\n{'#'*70}")
        log.info(f"WINDOW SIZE: {WIN_SEC}s (stride {STRIDE}s)")
        log.info(f"{'#'*70}")

        df_win = sliding_windows(df_raw, window_sec=WIN_SEC, stride_sec=STRIDE)
        feature_cols = get_feature_cols(df_win)
        log.info(f"  Total windows: {len(df_win)} | Features: {len(feature_cols)}")

        all_labels = set(df_win["workload_label"].unique())
        adversarial_labels = {l for l in all_labels
                              if any(l.startswith(p) for p in ALL_ADVERSARIAL_PREFIXES)}
        clean_labels = all_labels - adversarial_labels

        clean_mask = df_win["workload_label"].isin(clean_labels)
        non_dilution_adv = adversarial_labels - DILUTION_FAMILY_LABELS
        non_dilution_adv_mask = df_win["workload_label"].isin(non_dilution_adv)
        hardening_adv = adversarial_labels - set(DILUTION_SWEEP_LABELS)
        hardening_adv_mask = df_win["workload_label"].isin(hardening_adv)

        sweep_windows = df_win["workload_label"].isin(set(sweep_present)).sum()
        log.info(f"  Clean: {clean_mask.sum()} | Sweep test: {sweep_windows} | "
                 f"Non-dilution adv: {non_dilution_adv_mask.sum()} | "
                 f"Hardening adv: {hardening_adv_mask.sum()}")

        # ── Baseline ────────────────────────────────────────────────────
        log.info(f"\n  BASELINE (clean only):")
        X_tr = df_win.loc[clean_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_tr = le.transform(df_win.loc[clean_mask, "threeway_label"].values)
        groups_tr = df_win.loc[clean_mask, "run_id"].values
        w_tr = compute_sample_weights(y_tr)
        log.info(f"    Class-weighted: {len(y_tr)} windows (all data retained)")
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        all_results.extend(evaluate_condition(
            clf, le, df_win, feature_cols, sweep_present, "baseline", WIN_SEC))

        # ── LOO ─────────────────────────────────────────────────────────
        log.info(f"\n  LOO (clean + non-dilution adversarial):")
        loo_mask = clean_mask | non_dilution_adv_mask
        X_tr = df_win.loc[loo_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_tr = le.transform(df_win.loc[loo_mask, "threeway_label"].values)
        w_tr = compute_sample_weights(y_tr)
        log.info(f"    Class-weighted: {len(y_tr)} windows (all data retained)")
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        all_results.extend(evaluate_condition(
            clf, le, df_win, feature_cols, sweep_present, "LOO", WIN_SEC))

        # ── Hardened ────────────────────────────────────────────────────
        log.info(f"\n  HARDENED (clean + all adversarial incl. L_diluted):")
        hard_mask = clean_mask | hardening_adv_mask
        X_tr = df_win.loc[hard_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_tr = le.transform(df_win.loc[hard_mask, "threeway_label"].values)
        w_tr = compute_sample_weights(y_tr)
        log.info(f"    Class-weighted: {len(y_tr)} windows (all data retained)")
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X_tr, y_tr, sample_weight=w_tr)
        all_results.extend(evaluate_condition(
            clf, le, df_win, feature_cols, sweep_present, "hardened", WIN_SEC))

    # ── Save all results ────────────────────────────────────────────────
    df_results = pd.DataFrame(all_results)
    out_path = RESULTS_DIR / "dilution_sweep_eval.csv"
    df_results.to_csv(out_path, index=False)
    log.info(f"\nSaved: {out_path}")

    # ── Summary tables per window size ──────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("EVASION TAX CURVE SUMMARY")
    log.info("=" * 70)

    training_only = df_results[df_results["train_fraction"] > 0].copy()

    for ws in sorted(training_only["window_sec"].unique()):
        ws_data = training_only[training_only["window_sec"] == ws]
        log.info(f"\n  Window: {ws}s")
        log.info(f"  {'N':>5s}  {'Frac':>5s}  {'Cost':>5s}  {'Baseline':>8s}  {'LOO':>8s}  {'Hardened':>8s}  {'nWin':>5s}")
        log.info("  " + "-" * 52)

        for n_val in sorted(ws_data["dilution_N"].unique()):
            tf = 1.0 / n_val
            for cond in ["baseline", "LOO", "hardened"]:
                sub = ws_data[(ws_data["dilution_N"] == n_val) & (ws_data["condition"] == cond)]
                if cond == "baseline":
                    base = sub["detection_rate"].values[0] if len(sub) > 0 else float("nan")
                    n_win = sub["n_windows"].values[0] if len(sub) > 0 else 0
                elif cond == "LOO":
                    loo = sub["detection_rate"].values[0] if len(sub) > 0 else float("nan")
                else:
                    hard = sub["detection_rate"].values[0] if len(sub) > 0 else float("nan")

            log.info(f"  {n_val:>5.0f}  {tf:>4.0%}  {n_val:>4.0f}x  {base:>6.1%}  {loo:>6.1%}  {hard:>6.1%}    {n_win:>3d}")

    log.info("\n" + "=" * 70)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    main()
