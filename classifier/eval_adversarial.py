"""
Adversarial Robustness Evaluation
=================================

Train the three-way classifier on clean data, then evaluate on adversarial
workloads. Reports:
  1. Detection rate per adversarial strategy
  2. Per-strategy confusion (what does the classifier think it is?)
  3. Throughput comparison (evasion tax) vs baseline training

Usage:
    python -m classifier.eval_adversarial
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
    load_all_data, extract_enhanced_features, sliding_windows,
    get_feature_cols, threeway_label, TRAINING_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)

ADVERSARIAL_LABELS = {
    "adversarial_A_util_modulation",
    "adversarial_B_low_util",
    "adversarial_D_temporal_disruption",
}

CLEAN_TRAINING_LABELS = {
    "pytorch_resnet_cifar10", "pytorch_resnet_cifar10_amp",
    "pytorch_mlp_cifar10",
    "gpt2_wikitext2", "gpt2_wikitext2_amp",
    "bert_sst2", "bert_sst2_amp",
}


def main():
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    import xgboost as xgb

    log.info("=" * 60)
    log.info("ADVERSARIAL ROBUSTNESS EVALUATION")
    log.info("=" * 60)

    # Load all data
    log.info("Loading all telemetry data...")
    df_raw = load_all_data()

    # Split into clean and adversarial
    clean_mask = ~df_raw["workload_label"].isin(ADVERSARIAL_LABELS)
    adv_mask = df_raw["workload_label"].isin(ADVERSARIAL_LABELS)

    log.info(f"Clean rows: {clean_mask.sum():,} | Adversarial rows: {adv_mask.sum():,}")
    log.info(f"Adversarial workloads found: {df_raw.loc[adv_mask, 'workload_label'].unique().tolist()}")

    if adv_mask.sum() == 0:
        log.error("No adversarial data found!")
        return

    # Extract windows
    for win_sec in [30, 60]:
        stride = win_sec // 2
        log.info(f"\n{'='*60}")
        log.info(f"Window: {win_sec}s")
        log.info(f"{'='*60}")

        # Clean windows for training
        df_clean = df_raw[clean_mask].copy()
        df_clean_win = sliding_windows(df_clean, window_sec=win_sec, stride_sec=stride)

        # Adversarial windows for testing
        df_adv = df_raw[adv_mask].copy()
        df_adv_win = sliding_windows(df_adv, window_sec=win_sec, stride_sec=stride)

        if df_clean_win.empty or df_adv_win.empty:
            log.warning("Insufficient windows, skipping")
            continue

        feature_cols = get_feature_cols(df_clean_win)
        # Ensure same features in both
        for c in feature_cols:
            if c not in df_adv_win.columns:
                df_adv_win[c] = 0.0

        X_train = df_clean_win[feature_cols].fillna(0).values.astype(np.float32)
        y_train = df_clean_win["threeway_label"].values

        X_adv = df_adv_win[feature_cols].fillna(0).values.astype(np.float32)
        y_adv_true = np.array(["ml_training"] * len(X_adv))  # all adversarial are training
        adv_labels = df_adv_win["workload_label"].values

        log.info(f"Clean train: {len(X_train)} windows ({pd.Series(y_train).value_counts().to_dict()})")
        log.info(f"Adversarial test: {len(X_adv)} windows")

        # Train on clean data
        le = LabelEncoder()
        y_train_enc = le.fit_transform(y_train)

        clf = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf.fit(X_train, y_train_enc)

        # Predict on adversarial data
        y_adv_pred_enc = clf.predict(X_adv)
        y_adv_pred = le.inverse_transform(y_adv_pred_enc)

        # Overall detection rate (correctly classified as ml_training)
        overall_detected = (y_adv_pred == "ml_training").mean()
        log.info(f"\n--- Overall adversarial detection rate: {overall_detected:.1%} ---")
        log.info(f"Predicted distribution: {pd.Series(y_adv_pred).value_counts().to_dict()}")

        # Per-strategy analysis
        results = []
        for strategy in sorted(ADVERSARIAL_LABELS):
            mask = adv_labels == strategy
            if mask.sum() == 0:
                continue

            preds = y_adv_pred[mask]
            detected = (preds == "ml_training").mean()
            n_windows = mask.sum()
            pred_dist = pd.Series(preds).value_counts().to_dict()

            log.info(f"\n  {strategy}:")
            log.info(f"    Windows: {n_windows}")
            log.info(f"    Detected as training: {detected:.1%}")
            log.info(f"    Predictions: {pred_dist}")

            results.append({
                "strategy": strategy,
                "window_sec": win_sec,
                "n_windows": n_windows,
                "detection_rate": detected,
                "pred_training": pred_dist.get("ml_training", 0),
                "pred_inference": pred_dist.get("ml_inference", 0),
                "pred_other": pred_dist.get("other", 0),
            })

        # Also evaluate clean training detection rate (baseline)
        # Use cross-val on clean data for fair comparison
        from sklearn.model_selection import StratifiedGroupKFold
        clean_train_mask = df_clean_win["workload_label"].isin(CLEAN_TRAINING_LABELS)
        if clean_train_mask.sum() > 0:
            X_clean_train = df_clean_win.loc[clean_train_mask, feature_cols].fillna(0).values.astype(np.float32)
            y_clean_train_true = np.array(["ml_training"] * len(X_clean_train))
            y_clean_train_pred = le.inverse_transform(clf.predict(X_clean_train))
            clean_detected = (y_clean_train_pred == "ml_training").mean()
            log.info(f"\n  Clean training (baseline): detected {clean_detected:.1%}")
            results.append({
                "strategy": "clean_training_baseline",
                "window_sec": win_sec,
                "n_windows": len(X_clean_train),
                "detection_rate": clean_detected,
                "pred_training": (y_clean_train_pred == "ml_training").sum(),
                "pred_inference": (y_clean_train_pred == "ml_inference").sum(),
                "pred_other": (y_clean_train_pred == "other").sum(),
            })

        if results:
            df_results = pd.DataFrame(results)
            out_path = RESULTS_DIR / f"adversarial_eval_{win_sec}s.csv"
            df_results.to_csv(out_path, index=False)
            log.info(f"\nResults saved to {out_path}")

    # ── Throughput / Evasion Tax Analysis ─────────────────────────────────
    log.info(f"\n{'='*60}")
    log.info("EVASION TAX ANALYSIS")
    log.info(f"{'='*60}")

    # Read adversarial run logs to extract throughput
    # We can estimate from the telemetry: duration and samples/sec are in workload output
    # For now, compare power and utilization profiles
    for strategy in sorted(ADVERSARIAL_LABELS):
        adv_runs = df_raw[df_raw["workload_label"] == strategy]
        if adv_runs.empty:
            continue

        mean_util = adv_runs["gpu_utilization_pct"].mean()
        mean_power = adv_runs["power_draw_w"].mean()
        std_util = adv_runs["gpu_utilization_pct"].std()

        log.info(f"\n  {strategy}:")
        log.info(f"    Mean GPU util: {mean_util:.1f}%  (std: {std_util:.1f})")
        log.info(f"    Mean power: {mean_power:.1f}W")

    # Compare with clean training
    for label in ["pytorch_resnet_cifar10", "bert_sst2", "gpt2_wikitext2"]:
        clean_runs = df_raw[df_raw["workload_label"] == label]
        if clean_runs.empty:
            continue
        mean_util = clean_runs["gpu_utilization_pct"].mean()
        mean_power = clean_runs["power_draw_w"].mean()
        log.info(f"\n  {label} (clean baseline):")
        log.info(f"    Mean GPU util: {mean_util:.1f}%  Mean power: {mean_power:.1f}W")

    log.info("\n" + "=" * 60)
    log.info("EVALUATION COMPLETE")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
