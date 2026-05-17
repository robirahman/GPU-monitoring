"""Whitebox detection-rate eval (Round 5) on AMD RX 9070 XT.

Trains a hardened cross-vendor classifier on the full corpus minus the
whitebox parquets, then measures detection rate on each WB-* family
(WB-I, WB-D, WB-L).

The paper's headline (Table 5, NVIDIA-trained vs NVIDIA-tested):
  - WB-L (LoRA + dilution): 43-87% detection on unseen variants
  - WB-D (diluted): 100% hardened in-distribution, 87-100% LOO mean
  - WB-I (inference): 100% (correct: it's pure inference, not training)

For AMD we add the analogous question: does a hardened CROSS-VENDOR
classifier (NVIDIA + AMD-clean pooled) still catch the AMD whitebox
attempts at the same rate, or does cross-vendor training make the
classifier more or less vulnerable to whitebox attack?

Treats "detected as training" as the positive class for all WB variants
that *are* training (WB-D, WB-L). For WB-I, the ground truth is inference
so we report inference-detection accuracy.

Usage:
    python -m classifier.eval_whitebox_amd
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

from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
from classifier.twostage import TwoStageClassifier


def _vendor_of(gpu_name: str) -> str:
    if not isinstance(gpu_name, str): return "unknown"
    s = gpu_name.lower()
    if "nvidia" in s or "geforce" in s or "tesla" in s: return "nvidia"
    if "amd" in s or "radeon" in s: return "amd"
    return "unknown"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["random_forest", "xgboost"], default="random_forest")
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--stride-sec", type=float, default=15.0)
    ap.add_argument("--output", type=str,
                    default=str(PROJECT_ROOT / "classifier" / "results" / "whitebox_amd.csv"))
    args = ap.parse_args()

    raw = load_all_data()
    raw["vendor"] = raw["gpu_name"].apply(_vendor_of)
    for c in ("pcie_tx_mbps", "pcie_rx_mbps"):
        if c in raw.columns:
            raw[c] = raw[c].where(raw[c] >= 0, np.nan).fillna(0.0)

    # Identify the AMD WB-* runs to hold out
    wb_mask = raw["workload_label"].astype(str).str.startswith("whitebox_WB-")
    amd_mask = raw["vendor"] == "amd"
    held_out = wb_mask & amd_mask
    n_held = raw.loc[held_out, "run_id"].nunique()
    log.info(f"Held-out AMD WB-* runs: {n_held}")
    if n_held == 0:
        log.error("No AMD whitebox runs found; collect Tier 3 first.")
        return 2

    train_pool = raw.loc[~held_out].copy()
    test_pool = raw.loc[held_out].copy()
    log.info(f"Train pool: {train_pool['run_id'].nunique()} runs, "
             f"{train_pool['vendor'].value_counts().to_dict()}")
    log.info(f"Test pool:  {test_pool['run_id'].nunique()} runs (AMD whitebox)")

    log.info(f"Sliding windows: {args.window_sec}s / stride {args.stride_sec}s")
    train_windows = sliding_windows(train_pool, args.window_sec, args.stride_sec)
    test_windows = sliding_windows(test_pool, args.window_sec, args.stride_sec)
    log.info(f"Train windows: {len(train_windows)}, Test windows: {len(test_windows)}")

    feature_cols = get_feature_cols(train_windows)
    X_tr = np.nan_to_num(train_windows[feature_cols].values, nan=0.0)
    y_tr = train_windows["threeway_label"].values
    g_tr = train_windows["run_id"].values

    log.info(f"Fitting cross-vendor hardened classifier (backend={args.backend})...")
    clf = TwoStageClassifier(backend=args.backend)
    clf.fit(X_tr, y_tr, groups=g_tr, feature_cols=feature_cols)
    classes = list(clf.classes_)
    log.info(f"Classes: {classes}")

    # Predict per WB family
    rows = []
    for strategy in sorted(test_windows["workload_label"].unique()):
        sub = test_windows[test_windows["workload_label"] == strategy].copy()
        if len(sub) == 0:
            continue
        X_te = np.nan_to_num(sub[feature_cols].values, nan=0.0)
        proba = clf.predict_proba(X_te)
        pred = np.array(classes)[proba.argmax(axis=1)]
        true_label = sub["threeway_label"].iloc[0]
        n = len(sub)
        n_train = int((pred == "ml_training").sum())
        n_correct = int((pred == true_label).sum())
        # Predicted-class mix
        u, c = np.unique(pred, return_counts=True)
        pred_mix = dict(zip(u, c))
        p_train_mean = float(proba[:, classes.index("ml_training")].mean())
        p_infer_mean = float(proba[:, classes.index("ml_inference")].mean())
        p_other_mean = float(proba[:, classes.index("other")].mean())

        # "Detection" rate definition:
        #   For WB-D, WB-L (true training): detection = fraction predicted as training
        #   For WB-I (true inference):       detection = fraction predicted as inference
        if true_label == "ml_training":
            detection_rate = n_train / n
        elif true_label == "ml_inference":
            detection_rate = (pred == "ml_inference").sum() / n
        else:
            detection_rate = n_correct / n

        rows.append({
            "strategy": strategy,
            "true_class": true_label,
            "n_windows": n,
            "n_runs": int(sub["run_id"].nunique()),
            "detection_rate": round(detection_rate, 4),
            "three_way_accuracy": round(n_correct / n, 4),
            "p_train_mean": round(p_train_mean, 3),
            "p_infer_mean": round(p_infer_mean, 3),
            "p_other_mean": round(p_other_mean, 3),
            "predicted_mix": pred_mix,
        })

    df = pd.DataFrame(rows)
    log.info("\n" + df.drop(columns=["predicted_mix"]).to_string(index=False))
    log.info("\nPredicted class mix per strategy:")
    for r in rows:
        log.info(f"  {r['strategy']:35s} {r['predicted_mix']}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.drop(columns=["predicted_mix"]).to_csv(args.output, index=False)
    log.info(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
