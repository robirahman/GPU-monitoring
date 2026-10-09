"""
Evasion Tax Lower Bound Analysis
=================================

1. Sweep idle fractions: collect + evaluate composite adversarial at each level
2. Feature ablation: test hardened classifiers with restricted feature subsets
3. Produce the evasion tax curve: detection rate vs throughput loss

Usage:
    # Phase 1: Collect data (run on GPU)
    python -m classifier.evasion_bound --collect --gpu 0

    # Phase 2: Evaluate (run on CPU after collection)
    python -m classifier.evasion_bound --evaluate
"""

import argparse
import json
import logging
import os
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)
SWEEP_DIR = PROJECT_ROOT / "data" / "h100_nvl" / "evasion_sweep"


# ── Feature subsets for ablation ─────────────────────────────────────────────

FEATURE_SUBSETS = {
    "full_159": None,  # Use all features
    "power_memory_pcie": [
        # Power features (~15)
        "power_draw_w_mean", "power_draw_w_std", "power_draw_w_min", "power_draw_w_max",
        "power_draw_w_p25", "power_draw_w_p50", "power_draw_w_p75", "power_draw_w_p95",
        "power_draw_w_iqr", "power_draw_w_range", "power_draw_w_cv", "power_draw_w_skew", "power_draw_w_kurt",
        "power_cv", "power_rolling_var", "power_range_frac",
        "power_fft_period", "power_fft_peak_power",
        # Memory features (~20)
        "mem_used_mb_mean", "mem_used_mb_std", "mem_used_mb_min", "mem_used_mb_max",
        "mem_used_mb_p25", "mem_used_mb_p50", "mem_used_mb_p75", "mem_used_mb_p95",
        "mem_used_mb_iqr", "mem_used_mb_range", "mem_used_mb_cv", "mem_used_mb_skew", "mem_used_mb_kurt",
        "mem_slope", "mem_slope_first_half", "mem_slope_second_half", "mem_slope_ratio", "mem_second_half_cv",
        "mem_used_fft_peak_power",
        # PCIe features (~10)
        "pcie_tx_mbps_mean", "pcie_tx_mbps_std", "pcie_tx_mbps_cv",
        "pcie_rx_mbps_mean", "pcie_rx_mbps_std", "pcie_rx_mbps_cv",
        "pcie_total_mean", "pcie_cv", "pcie_fft_peak_power", "pcie_nonzero_frac",
        # Cross-signal
        "power_per_util",
    ],
    "memory_only": [
        "mem_used_mb_mean", "mem_used_mb_std", "mem_used_mb_min", "mem_used_mb_max",
        "mem_used_mb_p25", "mem_used_mb_p50", "mem_used_mb_p75", "mem_used_mb_p95",
        "mem_used_mb_iqr", "mem_used_mb_range", "mem_used_mb_cv", "mem_used_mb_skew", "mem_used_mb_kurt",
        "mem_slope", "mem_slope_first_half", "mem_slope_second_half", "mem_slope_ratio", "mem_second_half_cv",
        "mem_used_fft_peak_power",
    ],
    "power_only": [
        "power_draw_w_mean", "power_draw_w_std", "power_draw_w_min", "power_draw_w_max",
        "power_draw_w_p25", "power_draw_w_p50", "power_draw_w_p75", "power_draw_w_p95",
        "power_draw_w_iqr", "power_draw_w_range", "power_draw_w_cv", "power_draw_w_skew", "power_draw_w_kurt",
        "power_cv", "power_rolling_var", "power_range_frac",
        "power_fft_period", "power_fft_peak_power",
    ],
    "minimal_3": [
        "mem_slope_ratio", "power_cv", "pcie_cv",
    ],
}


# ── Phase 1: Collection ─────────────────────────────────────────────────────

def collect_sweep(gpu_index: int):
    """Run composite adversarial at each idle fraction and collect telemetry."""
    SWEEP_DIR.mkdir(parents=True, exist_ok=True)

    idle_fractions = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
    n_runs = 2  # 2 runs per fraction for speed (120s each)

    for frac in idle_fractions:
        label = f"adversarial_composite_{int(frac*100):02d}"
        for run in range(1, n_runs + 1):
            log.info(f"Collecting {label} run {run}/{n_runs} on GPU {gpu_index}...")
            cmd = [
                sys.executable, str(PROJECT_ROOT / "scripts" / "run_workload.py"),
                "--workload", label if label in _get_registered_labels() else "custom",
                "--gpu-index", str(gpu_index),
                "--output-dir", str(SWEEP_DIR),
                "--no-dcgm",
            ]
            # For non-registered fractions, use command mode
            if label not in _get_registered_labels():
                cmd = [
                    sys.executable, str(PROJECT_ROOT / "scripts" / "run_workload.py"),
                    "--workload", label,
                    "--gpu-index", str(gpu_index),
                    "--output-dir", str(SWEEP_DIR),
                    "--no-dcgm",
                    "--",
                    sys.executable, str(PROJECT_ROOT / "workloads" / "adversarial_composite.py"),
                    "--idle-fraction", str(frac),
                    "--duration", "120",
                ]

            result = subprocess.run(cmd, capture_output=True, text=True, cwd=str(PROJECT_ROOT))
            if result.returncode != 0:
                log.warning(f"  Run failed: {result.stderr[-200:] if result.stderr else 'no stderr'}")
            else:
                log.info(f"  Done: {label} run {run}")

    log.info(f"Collection complete. Files in {SWEEP_DIR}")


def _get_registered_labels():
    sys.path.insert(0, str(PROJECT_ROOT))
    from workloads.registry import WORKLOAD_REGISTRY
    return set(WORKLOAD_REGISTRY.keys())


# ── Phase 2: Evaluation ─────────────────────────────────────────────────────

def evaluate_sweep():
    """Evaluate classifier on collected sweep data."""
    from classifier.threeway_improved import (
        load_all_data, sliding_windows, get_feature_cols,
        extract_enhanced_features, TRAINING_LABELS,
    )
    from sklearn.preprocessing import LabelEncoder
    from sklearn.metrics import accuracy_score, f1_score
    import xgboost as xgb

    log.info("=" * 60)
    log.info("EVASION TAX BOUND ANALYSIS")
    log.info("=" * 60)

    # Load all clean data for training
    log.info("Loading clean telemetry data...")
    df_raw = load_all_data()

    # Separate clean vs adversarial composite
    composite_labels = {l for l in df_raw["workload_label"].unique()
                       if "adversarial_composite" in str(l)}
    all_adversarial = {l for l in df_raw["workload_label"].unique()
                      if "adversarial" in str(l)}
    clean_mask = ~df_raw["workload_label"].isin(all_adversarial)

    log.info(f"Composite adversarial labels found: {sorted(composite_labels)}")
    log.info(f"Clean rows: {clean_mask.sum():,}")

    # Extract windows
    win_sec = 30
    stride = 15

    df_clean = df_raw[clean_mask].copy()
    df_clean_win = sliding_windows(df_clean, window_sec=win_sec, stride_sec=stride)

    if df_clean_win.empty:
        log.error("No clean windows")
        return

    all_feature_cols = get_feature_cols(df_clean_win)
    log.info(f"Clean windows: {len(df_clean_win)} | Features: {len(all_feature_cols)}")

    # ── Feature ablation on clean data ───────────────────────────────────
    log.info("\n" + "=" * 60)
    log.info("FEATURE ABLATION: Clean Data Accuracy")
    log.info("=" * 60)

    X_clean = df_clean_win[all_feature_cols].fillna(0).values.astype(np.float32)
    y_clean = df_clean_win["threeway_label"].values
    groups_clean = df_clean_win["run_id"].values

    from sklearn.model_selection import StratifiedGroupKFold

    ablation_results = []
    trained_classifiers = {}

    for subset_name, subset_cols in FEATURE_SUBSETS.items():
        if subset_cols is None:
            cols = all_feature_cols
        else:
            cols = [c for c in subset_cols if c in all_feature_cols]
            if not cols:
                log.warning(f"  {subset_name}: no matching features, skipping")
                continue

        col_indices = [all_feature_cols.index(c) for c in cols if c in all_feature_cols]
        X_sub = X_clean[:, col_indices]

        le = LabelEncoder()
        y_enc = le.fit_transform(y_clean)

        n_splits = min(5, pd.Series(y_clean).value_counts().min())
        if n_splits < 2:
            continue

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        for tr_idx, te_idx in cv.split(X_sub, y_enc, groups_clean):
            clf = xgb.XGBClassifier(
                n_estimators=400, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1, verbosity=0,
            )
            clf.fit(X_sub[tr_idx], y_enc[tr_idx])
            y_pred = clf.predict(X_sub[te_idx])
            accs.append(accuracy_score(y_enc[te_idx], y_pred))

        mean_acc = np.mean(accs)
        log.info(f"  {subset_name:25s} ({len(cols):3d} features): "
                 f"acc={mean_acc:.4f}±{np.std(accs):.4f}")

        ablation_results.append({
            "feature_set": subset_name,
            "n_features": len(cols),
            "clean_accuracy": mean_acc,
            "clean_accuracy_std": np.std(accs),
        })

        # Train a final classifier on all clean data for adversarial eval
        clf_final = xgb.XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )
        clf_final.fit(X_sub, y_enc)
        trained_classifiers[subset_name] = (clf_final, le, col_indices)

    # ── Evaluate each idle fraction against each classifier ──────────────
    log.info("\n" + "=" * 60)
    log.info("EVASION TAX CURVE")
    log.info("=" * 60)

    sweep_results = []

    for label in sorted(composite_labels):
        # Parse idle fraction from label
        try:
            frac_str = label.split("_")[-1]
            idle_frac = int(frac_str) / 100.0
        except (ValueError, IndexError):
            continue

        adv_mask = df_raw["workload_label"] == label
        if adv_mask.sum() < 5:
            continue

        df_adv = df_raw[adv_mask].copy()
        df_adv_win = sliding_windows(df_adv, window_sec=win_sec, stride_sec=stride)
        if df_adv_win.empty:
            continue

        # Ensure feature columns match
        for c in all_feature_cols:
            if c not in df_adv_win.columns:
                df_adv_win[c] = 0.0

        X_adv = df_adv_win[all_feature_cols].fillna(0).values.astype(np.float32)

        # Measured cost, if the workload logged it. adversarial_composite.py:182
        # has always emitted an EVASION_TAX_DATA line carrying the real
        # samples/sec, and nothing ever read it -- so the throughput reported
        # here was the ASSUMED 1-idle_frac below. Prefer the measurement, and
        # say which one was used so a reader can tell.
        measured_tp = _measured_throughput_fraction(idle_frac)

        # Compute telemetry summary
        mean_power = df_adv["power_draw_w"].mean() if "power_draw_w" in df_adv.columns else 0
        mean_util = df_adv["gpu_utilization_pct"].mean() if "gpu_utilization_pct" in df_adv.columns else 0
        mean_mem = df_adv["mem_used_mb"].mean() if "mem_used_mb" in df_adv.columns else 0
        mean_temp = df_adv["temperature_c"].mean() if "temperature_c" in df_adv.columns else 0

        for subset_name, (clf, le, col_indices) in trained_classifiers.items():
            X_sub = X_adv[:, col_indices]
            y_pred_enc = clf.predict(X_sub)
            y_pred = le.inverse_transform(y_pred_enc)

            detection_rate = (y_pred == "ml_training").mean()
            n_windows = len(X_sub)

            # Measured when the run logged it; the 1-idle_frac fallback is an
            # upper bound on utility (it assumes the non-idle fraction is spent
            # at full honest throughput, which idle padding does not guarantee).
            if measured_tp is not None:
                throughput_fraction = measured_tp
                tp_source = "measured"
            else:
                throughput_fraction = 1.0 - idle_frac
                tp_source = "assumed(1-idle)"

            log.info(f"  idle={idle_frac:.0%} | {subset_name:25s} | "
                     f"detected={detection_rate:.1%} ({n_windows} win) | "
                     f"throughput={throughput_fraction:.0%} [{tp_source}]")

            sweep_results.append({
                "idle_fraction": idle_frac,
                "throughput_source": tp_source,
                "throughput_fraction": throughput_fraction,
                "feature_set": subset_name,
                "detection_rate": detection_rate,
                "n_windows": n_windows,
                "mean_power_w": mean_power,
                "mean_util_pct": mean_util,
                "mean_mem_mb": mean_mem,
                "mean_temp_c": mean_temp,
            })

    # ── Save results ─────────────────────────────────────────────────────
    if ablation_results:
        df_abl = pd.DataFrame(ablation_results)
        df_abl.to_csv(RESULTS_DIR / "feature_ablation.csv", index=False)
        log.info(f"\nFeature ablation saved to {RESULTS_DIR / 'feature_ablation.csv'}")

    if sweep_results:
        df_sweep = pd.DataFrame(sweep_results)
        df_sweep.to_csv(RESULTS_DIR / "evasion_tax_curve.csv", index=False)
        log.info(f"Evasion tax curve saved to {RESULTS_DIR / 'evasion_tax_curve.csv'}")

        # ── Find crossover points ────────────────────────────────────
        log.info("\n" + "=" * 60)
        log.info("EVASION TAX BOUNDS")
        log.info("=" * 60)

        for subset_name in FEATURE_SUBSETS:
            sub = df_sweep[df_sweep["feature_set"] == subset_name].sort_values("idle_fraction")
            if sub.empty:
                continue

            # Find minimum idle fraction where detection < 50%
            below_50 = sub[sub["detection_rate"] < 0.50]
            if not below_50.empty:
                min_idle = below_50["idle_fraction"].min()
                throughput_loss = min_idle * 100
                log.info(f"  {subset_name:25s}: evade (<50% detection) requires "
                         f"≥{throughput_loss:.0f}% throughput loss (idle≥{min_idle:.0%})")
            else:
                log.info(f"  {subset_name:25s}: NO idle fraction achieves <50% detection "
                         f"(best: {sub['detection_rate'].min():.1%} at idle={sub.loc[sub['detection_rate'].idxmin(), 'idle_fraction']:.0%})")

            # Find minimum idle fraction where detection < 20%
            below_20 = sub[sub["detection_rate"] < 0.20]
            if not below_20.empty:
                min_idle = below_20["idle_fraction"].min()
                log.info(f"  {subset_name:25s}: near-complete evasion (<20%) requires "
                         f"≥{min_idle*100:.0f}% throughput loss")

    log.info("\n" + "=" * 60)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 60)


# ── Main ─────────────────────────────────────────────────────────────────────

def _measured_throughput_fraction(idle_frac, log_glob="logs/*.log"):
    """Real throughput of the idle_frac run over the idle_frac=0 run, or None.

    Reads the EVASION_TAX_DATA lines that adversarial_composite.py already
    emits. Returns None when the logs are absent, so callers fall back to the
    old assumption rather than failing.
    """
    import glob as _glob
    import re as _re

    tax_re = _re.compile(r"EVASION_TAX_DATA:\s*(?P<body>.*)")
    field_re = _re.compile(r"(\w+)=([-\w.]*)")
    by_idle = {}
    for path in _glob.glob(log_glob):
        try:
            with open(path, errors="ignore") as fh:
                for line in fh:
                    m = tax_re.search(line)
                    if not m:
                        continue
                    f = dict(field_re.findall(m.group("body")))
                    try:
                        k = round(float(f.get("idle_frac", "nan")), 3)
                        tp = float(f["throughput"])
                    except (KeyError, ValueError):
                        continue
                    by_idle.setdefault(k, []).append(tp)
        except OSError:
            continue
    base = by_idle.get(0.0)
    mine = by_idle.get(round(float(idle_frac), 3))
    if not base or not mine:
        return None
    b = sum(base) / len(base)
    return (sum(mine) / len(mine)) / b if b else None


def main():
    parser = argparse.ArgumentParser(description="Evasion tax bound analysis")
    parser.add_argument("--collect", action="store_true", help="Phase 1: collect sweep data")
    parser.add_argument("--evaluate", action="store_true", help="Phase 2: evaluate and produce bounds")
    parser.add_argument("--gpu", type=int, default=0, help="GPU index for collection")
    args = parser.parse_args()

    if args.collect:
        collect_sweep(args.gpu)
    if args.evaluate:
        evaluate_sweep()
    if not args.collect and not args.evaluate:
        print("Specify --collect and/or --evaluate")


if __name__ == "__main__":
    main()
