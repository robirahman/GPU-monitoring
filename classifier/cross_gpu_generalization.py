"""
Cross-GPU Generalization Analysis (A1) + Detection Latency Curve (A5)
=====================================================================

A1: Train classifier on one GPU's data, test on another's.
    - All pairwise transfers across GPU types
    - GPUs: RTX 4050, RTX 5080, A100, H200, B200 (+ H100 NVL/80GB if present)
    - Pool all GPUs → retrain
    - Feature stability analysis across GPUs

A5: Run three-way classifier at window sizes 5s-120s to produce
    accuracy vs detection latency curve.

Usage:
    python -m classifier.cross_gpu_generalization
"""

import json
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# Reuse data loading and feature extraction from threeway_improved
from classifier.threeway_improved import (
    load_all_data, extract_enhanced_features, sliding_windows,
    get_feature_cols, RAW_SIGNALS, threeway_label,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def identify_gpu_type(gpu_name: str) -> str:
    """Map GPU name strings to canonical types."""
    if gpu_name is None or pd.isna(gpu_name):
        return "unknown"
    gpu_name = str(gpu_name)
    if "A100" in gpu_name:
        return "A100"
    if "H100" in gpu_name and "NVL" in gpu_name:
        return "H100_NVL"
    if "H100" in gpu_name:
        return "H100_80GB"
    if "H200" in gpu_name:
        return "H200"
    if "B200" in gpu_name:
        return "B200"
    if "5080" in gpu_name:
        return "RTX_5080"
    if "4050" in gpu_name:
        return "RTX_4050"
    return "unknown"


def add_gpu_type(df: pd.DataFrame) -> pd.DataFrame:
    """Add gpu_type column based on gpu_name."""
    if "gpu_name" in df.columns:
        df["gpu_type"] = df["gpu_name"].apply(identify_gpu_type)
    else:
        df["gpu_type"] = "unknown"
    return df


# ── A1: Cross-GPU Generalization ─────────────────────────────────────────────

def run_cross_gpu_experiments():
    """Train on one GPU type, test on another."""
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    import xgboost as xgb
    from sklearn.preprocessing import LabelEncoder

    log.info("=" * 60)
    log.info("A1: CROSS-GPU GENERALIZATION")
    log.info("=" * 60)

    # Load all data with gpu_name preserved
    log.info("Loading all telemetry data...")
    df_raw = load_all_data()
    df_raw = add_gpu_type(df_raw)

    # Check GPU distribution
    gpu_run_counts = df_raw.groupby("gpu_type")["run_id"].nunique()
    log.info(f"GPU type distribution (runs):\n{gpu_run_counts.to_string()}")

    # Extract windows at 30s (good balance of accuracy and sample count)
    log.info("Extracting 30s sliding windows...")
    df_win = sliding_windows(df_raw, window_sec=30, stride_sec=15)
    if df_win.empty:
        log.error("No windows generated")
        return {}

    # Propagate gpu_type to windows
    run_gpu = df_raw.groupby("run_id")["gpu_type"].first().to_dict()
    df_win["gpu_type"] = df_win["run_id"].map(run_gpu).fillna("unknown")

    feature_cols = get_feature_cols(df_win)
    log.info(f"Windows: {len(df_win)} | Features: {len(feature_cols)}")
    log.info(f"Per-GPU window counts:\n{df_win['gpu_type'].value_counts().to_string()}")

    results = []

    # Get GPU types that have enough data
    gpu_types = [g for g in df_win["gpu_type"].unique() if g != "unknown"]
    gpu_types = [g for g in gpu_types if (df_win["gpu_type"] == g).sum() >= 20]
    log.info(f"GPU types with sufficient data: {gpu_types}")

    # Cross-GPU transfer experiments
    for train_gpu in gpu_types:
        for test_gpu in gpu_types:
            train_mask = df_win["gpu_type"] == train_gpu
            test_mask = df_win["gpu_type"] == test_gpu

            X_train = df_win.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
            y_train = df_win.loc[train_mask, "threeway_label"].values
            X_test = df_win.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)
            y_test = df_win.loc[test_mask, "threeway_label"].values

            if len(X_train) < 10 or len(X_test) < 10:
                continue

            # Check class coverage
            train_classes = set(y_train)
            test_classes = set(y_test)
            if len(train_classes) < 2 or len(test_classes) < 2:
                continue

            le = LabelEncoder()
            le.fit(list(train_classes | test_classes))
            y_train_enc = le.transform(y_train)
            y_test_enc = le.transform(y_test)

            clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1, verbosity=0,
            )
            clf.fit(X_train, y_train_enc)
            y_pred_enc = clf.predict(X_test)
            y_pred = le.inverse_transform(y_pred_enc)

            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average="macro", zero_division=0)

            is_same = "same-GPU" if train_gpu == test_gpu else "cross-GPU"
            log.info(f"  Train={train_gpu:12s} → Test={test_gpu:12s} [{is_same}]  "
                     f"acc={acc:.4f}  f1={f1:.4f}  "
                     f"(train={len(X_train)}, test={len(X_test)})")

            if train_gpu != test_gpu:
                report = classification_report(y_test, y_pred, zero_division=0)
                log.info(f"\n{report}")

            results.append({
                "train_gpu": train_gpu,
                "test_gpu": test_gpu,
                "type": is_same,
                "accuracy": acc,
                "f1_macro": f1,
                "train_samples": len(X_train),
                "test_samples": len(X_test),
            })

    # Pooled experiment: train on all GPUs combined
    log.info("\n--- Pooled (all GPUs) ---")
    known_mask = df_win["gpu_type"] != "unknown"
    X_all = df_win.loc[known_mask, feature_cols].fillna(0).values.astype(np.float32)
    y_all = df_win.loc[known_mask, "threeway_label"].values
    groups_all = df_win.loc[known_mask, "run_id"].values

    if len(X_all) >= 30:
        from sklearn.model_selection import StratifiedGroupKFold

        le = LabelEncoder()
        y_enc = le.fit_transform(y_all)

        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        accs, f1s = [], []
        for tr_idx, te_idx in cv.split(X_all, y_enc, groups_all):
            clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1, verbosity=0,
            )
            clf.fit(X_all[tr_idx], y_enc[tr_idx])
            y_pred = clf.predict(X_all[te_idx])
            accs.append(accuracy_score(y_enc[te_idx], y_pred))
            f1s.append(f1_score(y_enc[te_idx], y_pred, average="macro", zero_division=0))

        log.info(f"  Pooled (all GPUs) 5-fold CV:  acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
                 f"f1={np.mean(f1s):.4f}  (n={len(X_all)})")
        results.append({
            "train_gpu": "pooled", "test_gpu": "pooled", "type": "pooled",
            "accuracy": np.mean(accs), "f1_macro": np.mean(f1s),
            "train_samples": len(X_all), "test_samples": len(X_all),
        })

    # Feature stability analysis
    log.info("\n--- Feature Stability Across GPUs ---")
    feature_means = {}
    for gpu in gpu_types:
        mask = df_win["gpu_type"] == gpu
        if mask.sum() < 10:
            continue
        means = df_win.loc[mask, feature_cols].mean()
        stds = df_win.loc[mask, feature_cols].std()
        feature_means[gpu] = {"mean": means, "std": stds}

    if len(feature_means) >= 2:
        # Compute coefficient of variation of feature means across GPUs
        mean_df = pd.DataFrame({gpu: fm["mean"] for gpu, fm in feature_means.items()})
        cv_across_gpus = mean_df.std(axis=1) / (mean_df.mean(axis=1).abs() + 1e-9)
        cv_across_gpus = cv_across_gpus.sort_values()

        log.info("\nMost STABLE features across GPUs (low CV of means):")
        for feat, cv in cv_across_gpus.head(15).items():
            vals = {gpu: f"{mean_df.loc[feat, gpu]:.3f}" for gpu in mean_df.columns}
            log.info(f"  {feat:45s} CV={cv:.4f}  {vals}")

        log.info("\nMost UNSTABLE features across GPUs (high CV of means):")
        for feat, cv in cv_across_gpus.tail(15).items():
            vals = {gpu: f"{mean_df.loc[feat, gpu]:.3f}" for gpu in mean_df.columns}
            log.info(f"  {feat:45s} CV={cv:.4f}  {vals}")

        # Save stability analysis
        cv_across_gpus.to_csv(RESULTS_DIR / "feature_stability_across_gpus.csv")

    # Save results
    if results:
        pd.DataFrame(results).to_csv(RESULTS_DIR / "cross_gpu_results.csv", index=False)
        log.info(f"\nResults saved to {RESULTS_DIR / 'cross_gpu_results.csv'}")

    return results


# ── A5: Detection Latency Curve ──────────────────────────────────────────────

def run_detection_latency():
    """Accuracy vs window size for the three-way classifier."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    log.info("\n" + "=" * 60)
    log.info("A5: DETECTION LATENCY CURVE")
    log.info("=" * 60)

    df_raw = load_all_data()

    window_sizes = [5, 10, 15, 20, 30, 45, 60, 90, 120]
    results = []

    for win_sec in window_sizes:
        stride = max(win_sec // 2, 2)
        log.info(f"\nWindow: {win_sec}s (stride: {stride}s)")

        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride)
        if df_win.empty or len(df_win) < 20:
            log.warning(f"  Insufficient windows ({len(df_win)}), skipping")
            continue

        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        groups = df_win["run_id"].values

        # Check enough classes
        unique = np.unique(y)
        if len(unique) < 2:
            continue

        le = LabelEncoder()
        y_enc = le.fit_transform(y)

        # Determine number of CV folds
        run_class_counts = df_win.groupby("run_id")["threeway_label"].first().value_counts()
        n_splits = min(5, run_class_counts.min())
        if n_splits < 2:
            log.warning(f"  Not enough runs per class for CV (min={run_class_counts.min()})")
            continue

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs, f1s = [], []

        for tr_idx, te_idx in cv.split(X, y_enc, groups):
            clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1, verbosity=0,
            )
            clf.fit(X[tr_idx], y_enc[tr_idx])
            y_pred = clf.predict(X[te_idx])
            accs.append(accuracy_score(y_enc[te_idx], y_pred))
            f1s.append(f1_score(y_enc[te_idx], y_pred, average="macro", zero_division=0))

        mean_acc = np.mean(accs)
        std_acc = np.std(accs)
        mean_f1 = np.mean(f1s)

        log.info(f"  {win_sec:3d}s window: acc={mean_acc:.4f}±{std_acc:.4f}  "
                 f"f1={mean_f1:.4f}  (n_windows={len(X)}, {n_splits}-fold CV)")

        results.append({
            "window_sec": win_sec,
            "accuracy_mean": mean_acc,
            "accuracy_std": std_acc,
            "f1_macro": mean_f1,
            "n_windows": len(X),
            "n_folds": n_splits,
        })

    if results:
        df_results = pd.DataFrame(results)
        df_results.to_csv(RESULTS_DIR / "detection_latency.csv", index=False)
        log.info(f"\nDetection latency results saved to {RESULTS_DIR / 'detection_latency.csv'}")

        # Find minimum window for >90% and >95%
        for threshold in [0.90, 0.95]:
            above = df_results[df_results["accuracy_mean"] >= threshold]
            if not above.empty:
                min_win = above["window_sec"].min()
                log.info(f"  Minimum window for >{threshold*100:.0f}% accuracy: {min_win}s")
            else:
                log.info(f"  No window achieves >{threshold*100:.0f}% accuracy")

    return results


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    cross_gpu_results = run_cross_gpu_experiments()
    latency_results = run_detection_latency()

    log.info("\n" + "=" * 60)
    log.info("ALL EXPERIMENTS COMPLETE")
    log.info("=" * 60)

    if cross_gpu_results:
        log.info("\nCross-GPU Summary:")
        for r in cross_gpu_results:
            log.info(f"  {r['train_gpu']:>12s} → {r['test_gpu']:<12s}  "
                     f"acc={r['accuracy']:.4f}  f1={r['f1_macro']:.4f}")

    if latency_results:
        log.info("\nDetection Latency Summary:")
        for r in latency_results:
            marker = " ✓" if r["accuracy_mean"] >= 0.95 else ""
            log.info(f"  {r['window_sec']:3d}s: acc={r['accuracy_mean']:.4f}±{r['accuracy_std']:.4f}{marker}")


if __name__ == "__main__":
    main()
