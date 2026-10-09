"""Cross-architecture transfer eval: NVIDIA → AMD-Windows.

The headline question: does a classifier trained only on NVIDIA telemetry
correctly classify the same workload categories when run on a Windows-AMD
machine (RX 9070 XT, RDNA 4, DirectML / AMF / Vulkan compute stacks)?

This is the natural extension of the report's cross-GPU heatmap (§5.1,
Figure 1, Table 4) — which only covered NVIDIA-to-NVIDIA transfer — into
the cross-vendor case discussed in §E.4 (Limitations, "future work should
generalize ... using the analogous telemetry signals available for other
hardware").

Protocol:
  1. Load all parquets via classifier.threeway_improved.load_all_data().
  2. Slice into 30s windows, stride 15s, per-run grouped.
  3. Split: train = NVIDIA rows; test = AMD-Windows rows. Both are matched
     to the cross-vendor schema by load_all_data + _normalize_columns.
  4. Fit TwoStageClassifier (RF backend) on the NVIDIA partition.
  5. Predict on the AMD partition. Report:
       - overall three-way accuracy
       - binary "is this training?" accuracy (the governance question)
       - per-workload breakdown (predicted vs true)
       - confusion matrix

Notes:
  - PdhCollector reports `pcie_{tx,rx}_mbps` as NaN on AMD-Windows (PDH
    doesn't expose PCIe throughput on AMD without amd-smi). We fillna(0)
    to keep RandomForest happy. This is conservative — those features are
    information the model loses on AMD, but cross-vendor accuracy still
    has to come from the remaining 7 raw signals.
  - LHM provides `power_draw_w`, `temperature_c`, `sm_clock_mhz`,
    `mem_clock_mhz` on AMD; PDH provides `gpu_utilization_pct`,
    `mem_utilization_pct`, `mem_used_mb`. So all 7 non-PCIe signals are
    populated.

Usage:
    python -m classifier.eval_amd_transfer
    python classifier/eval_amd_transfer.py --backend xgboost
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

import sys
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.threeway_improved import (
    load_all_data,
    sliding_windows,
    get_feature_cols,
)
from classifier.twostage import TwoStageClassifier


def _vendor_of(gpu_name: str) -> str:
    if not isinstance(gpu_name, str):
        return "unknown"
    s = gpu_name.lower()
    if "nvidia" in s or "geforce" in s or "tesla" in s:
        return "nvidia"
    if "amd" in s or "radeon" in s:
        return "amd"
    if "tpu" in s:
        # Without this a "Google TPU v6e" row returns "unknown" and is dropped
        # by the train=nvidia/test=amd partition with no count logged.
        return "tpu"
    return "unknown"


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--backend", choices=["random_forest", "xgboost"], default="random_forest")
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--stride-sec", type=float, default=15.0)
    ap.add_argument("--output", type=str,
                    default=str(PROJECT_ROOT / "classifier" / "results" / "amd_transfer.csv"))
    args = ap.parse_args()

    log.info("Loading all telemetry parquets ...")
    raw = load_all_data()

    # Drop smoke-test runs that polluted data/raw_telemetry/ during pipeline
    # validation. These have e2e/smoke/test in the workload_label or run_id.
    smoke_pat = r"smoke|_e2e\b|sensor_test|pdh_[a-z]+_test"
    smoke_mask = (
        raw["workload_label"].astype(str).str.contains(smoke_pat, case=False, regex=True)
        | raw["run_id"].astype(str).str.contains(smoke_pat, case=False, regex=True)
    )
    if smoke_mask.any():
        log.info(f"Excluding {smoke_mask.sum():,} smoke-test rows "
                 f"({raw.loc[smoke_mask, 'run_id'].nunique()} runs)")
        raw = raw.loc[~smoke_mask].copy()

    # Tag vendor before windowing — we propagate it via run_id later
    raw["vendor"] = raw["gpu_name"].apply(_vendor_of)
    log.info("Vendor distribution by run:")
    log.info(raw.groupby("vendor")["run_id"].nunique().to_string())

    # AMD parquets currently have pcie_tx/rx_mbps as NaN (PDH limitation).
    # Fill them with 0 so RandomForest doesn't reject the rows. NVIDIA rows
    # are unaffected.
    for c in ("pcie_tx_mbps", "pcie_rx_mbps"):
        if c in raw.columns:
            raw[c] = raw[c].fillna(0.0)

    amd_run_ids = set(raw.loc[raw["vendor"] == "amd", "run_id"].unique())
    nvidia_run_ids = set(raw.loc[raw["vendor"] == "nvidia", "run_id"].unique())
    log.info(f"AMD runs: {len(amd_run_ids)}; NVIDIA runs: {len(nvidia_run_ids)}")
    if not amd_run_ids:
        log.error("No AMD telemetry rows found — collect some first via "
                  "scripts/run_workload_windows.py.")
        return 2
    if not nvidia_run_ids:
        log.error("No NVIDIA telemetry rows found — load_all_data should have "
                  "found them under data/.")
        return 2

    log.info(f"Sliding windows: window={args.window_sec}s, stride={args.stride_sec}s")
    windows = sliding_windows(raw, args.window_sec, args.stride_sec)
    log.info(f"Total windows: {len(windows)}")

    # Map run_id -> vendor for splitting
    run_vendor = raw.groupby("run_id")["vendor"].first()
    windows["vendor"] = windows["run_id"].map(run_vendor)
    train_mask = windows["vendor"] == "nvidia"
    test_mask = windows["vendor"] == "amd"

    log.info(f"Train (NVIDIA) windows: {train_mask.sum()}")
    log.info(f"Test  (AMD)    windows: {test_mask.sum()}")

    feature_cols = get_feature_cols(windows)
    log.info(f"Feature cols: {len(feature_cols)}")

    X_tr = windows.loc[train_mask, feature_cols].values
    y_tr = windows.loc[train_mask, "threeway_label"].values
    g_tr = windows.loc[train_mask, "run_id"].values

    X_te = windows.loc[test_mask, feature_cols].values
    y_te = windows.loc[test_mask, "threeway_label"].values
    g_te = windows.loc[test_mask, "run_id"].values
    wl_te = windows.loc[test_mask, "workload_label"].values

    # Some feature cells can still be NaN from variance-of-constant computations.
    X_tr = np.nan_to_num(X_tr, nan=0.0)
    X_te = np.nan_to_num(X_te, nan=0.0)

    log.info(f"Fitting TwoStageClassifier (backend={args.backend}) on NVIDIA ...")
    clf = TwoStageClassifier(backend=args.backend)
    clf.fit(X_tr, y_tr, groups=g_tr, feature_cols=feature_cols)

    log.info("Predicting on AMD ...")
    proba = clf.predict_proba(X_te)            # (n, 3) in clf.classes_ order
    pred = np.array(clf.classes_)[proba.argmax(axis=1)]

    classes = list(clf.classes_)
    log.info(f"Classes: {classes}")

    # Three-way per-window accuracy
    correct = (pred == y_te)
    overall_acc = correct.mean() if len(correct) else float("nan")
    log.info(f"Overall three-way per-window accuracy: {overall_acc:.4f} "
             f"({correct.sum()}/{len(correct)})")

    # Binary "is this training?" accuracy
    is_train_pred = (pred == "ml_training")
    is_train_true = (y_te == "ml_training")
    binary_acc = (is_train_pred == is_train_true).mean() if len(y_te) else float("nan")
    log.info(f"Binary is-training per-window accuracy: {binary_acc:.4f} "
             f"({(is_train_pred == is_train_true).sum()}/{len(y_te)})")

    # Per-workload breakdown
    by_wl = []
    for wl in sorted(set(wl_te)):
        m = wl_te == wl
        true_label = y_te[m][0]   # all rows for one workload share the label
        n = m.sum()
        n_correct = (pred[m] == true_label).sum()
        # Most common predicted class for this workload
        u, c = np.unique(pred[m], return_counts=True)
        majority_pred = u[c.argmax()]
        n_majority = c.max()
        # Per-class probability means
        means = {cls: proba[m, classes.index(cls)].mean() for cls in classes}
        by_wl.append({
            "workload": wl,
            "true_class": true_label,
            "n_windows": int(n),
            "n_correct": int(n_correct),
            "accuracy": n_correct / n if n else float("nan"),
            "majority_pred": majority_pred,
            "majority_n": int(n_majority),
            "p_other_mean": round(means.get("other", float("nan")), 3),
            "p_train_mean": round(means.get("ml_training", float("nan")), 3),
            "p_infer_mean": round(means.get("ml_inference", float("nan")), 3),
        })
    df_wl = pd.DataFrame(by_wl)
    log.info("\nPer-workload breakdown (AMD test set):")
    log.info("\n" + df_wl.to_string(index=False))

    # Confusion matrix
    log.info("\nConfusion matrix (rows=true, cols=predicted):")
    cm = pd.crosstab(
        pd.Series(y_te, name="true"),
        pd.Series(pred, name="predicted"),
        margins=True,
    )
    log.info("\n" + cm.to_string())

    # Save the per-workload breakdown
    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df_wl.to_csv(out_path, index=False)
    log.info(f"\nWrote per-workload breakdown to {out_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
