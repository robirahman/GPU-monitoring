"""Cross-vendor classifier eval (NVIDIA ↔ AMD-Linux, both directions).

Extends classifier/eval_amd_transfer.py with:
  - Linux-amdsmi PCIe -1 sentinel normalization (the older Windows-AMD
    pipeline reported NaN; the new Linux/amdsmi pipeline reports -1 because
    consumer cards don't expose PCIe TX/RX bandwidth via amdsmi).
  - Reverse direction (AMD-trained → NVIDIA-test).
  - AMD-only 5-fold grouped CV (sanity check that the AMD signal is even
    learnable internally; isolates "AMD has weaker signal" from "NVIDIA
    features don't transfer to AMD").
  - Reports both three-way and the headline binary "is this training?".

Usage:
    python -m classifier.eval_cross_vendor
    python classifier/eval_cross_vendor.py --backend xgboost
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

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
    return "unknown"


def _normalize_amd_pcie_sentinel(raw: pd.DataFrame) -> pd.DataFrame:
    # amdsmi reports pcie_tx/rx_mbps == -1 when the card doesn't expose it
    # (RX 9070 XT, most consumer cards). Old Windows-AMD pipeline used NaN.
    # Either way, normalize to NaN-then-zero so the value carries no signal.
    for c in ("pcie_tx_mbps", "pcie_rx_mbps"):
        if c in raw.columns:
            raw[c] = raw[c].where(raw[c] >= 0, np.nan).fillna(0.0)
    return raw


def _drop_smoke_runs(raw: pd.DataFrame) -> pd.DataFrame:
    smoke_pat = r"smoke|_e2e\b|sensor_test|pdh_[a-z]+_test"
    smoke_mask = (
        raw["workload_label"].astype(str).str.contains(smoke_pat, case=False, regex=True)
        | raw["run_id"].astype(str).str.contains(smoke_pat, case=False, regex=True)
    )
    if smoke_mask.any():
        log.info(f"Excluding {smoke_mask.sum():,} smoke-test rows "
                 f"({raw.loc[smoke_mask, 'run_id'].nunique()} runs)")
        raw = raw.loc[~smoke_mask].copy()
    return raw


def _eval(name, clf, X_te, y_te, wl_te, classes):
    proba = clf.predict_proba(X_te)
    pred = np.array(classes)[proba.argmax(axis=1)]
    correct = (pred == y_te)
    overall_acc = correct.mean() if len(correct) else float("nan")
    is_train_pred = (pred == "ml_training")
    is_train_true = (y_te == "ml_training")
    binary_acc = (is_train_pred == is_train_true).mean() if len(y_te) else float("nan")

    log.info(f"[{name}] three-way per-window accuracy: {overall_acc:.4f}   "
             f"binary training-vs-rest: {binary_acc:.4f}   "
             f"(n={len(y_te)})")
    cm = pd.crosstab(pd.Series(y_te, name="true"),
                     pd.Series(pred, name="predicted"),
                     margins=True)
    log.info(f"[{name}] confusion matrix:\n{cm.to_string()}")
    return {
        "name": name,
        "n_windows": int(len(y_te)),
        "three_way_acc": float(overall_acc),
        "binary_train_vs_rest_acc": float(binary_acc),
    }


def _per_workload_breakdown(name, clf, X_te, y_te, wl_te, classes):
    proba = clf.predict_proba(X_te)
    pred = np.array(classes)[proba.argmax(axis=1)]
    by_wl = []
    for wl in sorted(set(wl_te)):
        m = wl_te == wl
        true_label = y_te[m][0]
        n = int(m.sum())
        n_correct = int((pred[m] == true_label).sum())
        u, c = np.unique(pred[m], return_counts=True)
        majority_pred = u[c.argmax()]
        means = {cls: proba[m, classes.index(cls)].mean() for cls in classes}
        by_wl.append({
            "experiment": name,
            "workload": wl,
            "true_class": true_label,
            "n_windows": n,
            "n_correct": n_correct,
            "accuracy": round(n_correct / n, 4) if n else float("nan"),
            "majority_pred": majority_pred,
            "p_other_mean": round(means.get("other", float("nan")), 3),
            "p_train_mean": round(means.get("ml_training", float("nan")), 3),
            "p_infer_mean": round(means.get("ml_inference", float("nan")), 3),
        })
    return pd.DataFrame(by_wl)


def _amd_only_cv(windows, feature_cols, backend="random_forest", n_splits=5):
    amd = windows[windows["vendor"] == "amd"].copy()
    groups = amd["run_id"].values
    X = amd[feature_cols].values
    X = np.nan_to_num(X, nan=0.0)
    y = amd["threeway_label"].values
    wl = amd["workload_label"].values

    n_groups = len(set(groups))
    if n_groups < n_splits:
        log.info(f"AMD-only CV: only {n_groups} groups, dropping splits to {n_groups}")
        n_splits = max(2, n_groups)

    log.info(f"AMD-only CV ({n_splits}-fold grouped): n={len(X)} windows, {n_groups} runs")
    gkf = GroupKFold(n_splits=n_splits)
    accs, binary_accs = [], []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=groups)):
        clf = TwoStageClassifier(backend=backend)
        clf.fit(X[tr], y[tr], groups=groups[tr], feature_cols=feature_cols)
        classes = list(clf.classes_)
        proba = clf.predict_proba(X[te])
        pred = np.array(classes)[proba.argmax(axis=1)]
        acc = (pred == y[te]).mean()
        binary_acc = ((pred == "ml_training") == (y[te] == "ml_training")).mean()
        log.info(f"  fold {fold + 1}: three-way={acc:.4f}  binary={binary_acc:.4f}  n_test={len(te)}")
        accs.append(acc)
        binary_accs.append(binary_acc)
    log.info(f"AMD-only CV mean: three-way={np.mean(accs):.4f} ± {np.std(accs):.4f}   "
             f"binary={np.mean(binary_accs):.4f} ± {np.std(binary_accs):.4f}")
    return {
        "name": "amd_only_cv",
        "three_way_acc": float(np.mean(accs)),
        "three_way_acc_std": float(np.std(accs)),
        "binary_train_vs_rest_acc": float(np.mean(binary_accs)),
        "binary_train_vs_rest_acc_std": float(np.std(binary_accs)),
    }


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--backend", choices=["random_forest", "xgboost"], default="random_forest")
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--stride-sec", type=float, default=15.0)
    ap.add_argument("--output-dir", type=str,
                    default=str(PROJECT_ROOT / "classifier" / "results"))
    ap.add_argument("--no-amd-cv", action="store_true",
                    help="Skip the AMD-only CV (slow)")
    args = ap.parse_args()

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    log.info("Loading all telemetry parquets ...")
    raw = load_all_data()
    raw = _drop_smoke_runs(raw)
    raw["vendor"] = raw["gpu_name"].apply(_vendor_of)
    log.info("Vendor distribution by run:")
    log.info(raw.groupby("vendor")["run_id"].nunique().to_string())
    raw = _normalize_amd_pcie_sentinel(raw)

    log.info(f"Sliding windows: {args.window_sec}s / stride {args.stride_sec}s")
    windows = sliding_windows(raw, args.window_sec, args.stride_sec)
    run_vendor = raw.groupby("run_id")["vendor"].first()
    windows["vendor"] = windows["run_id"].map(run_vendor)

    feature_cols = get_feature_cols(windows)
    log.info(f"Features: {len(feature_cols)}")

    nv_mask = windows["vendor"] == "nvidia"
    amd_mask = windows["vendor"] == "amd"
    log.info(f"NVIDIA windows: {nv_mask.sum()}   AMD windows: {amd_mask.sum()}")

    results = []
    breakdowns = []

    def _build_xy(mask):
        X = np.nan_to_num(windows.loc[mask, feature_cols].values, nan=0.0)
        y = windows.loc[mask, "threeway_label"].values
        g = windows.loc[mask, "run_id"].values
        wl = windows.loc[mask, "workload_label"].values
        return X, y, g, wl

    # --- (a) NVIDIA-trained → AMD-test ---
    Xtr, ytr, gtr, _ = _build_xy(nv_mask)
    Xte, yte, _, wlte = _build_xy(amd_mask)
    log.info("\n=== (a) Train: NVIDIA  →  Test: AMD ===")
    clf = TwoStageClassifier(backend=args.backend)
    clf.fit(Xtr, ytr, groups=gtr, feature_cols=feature_cols)
    classes = list(clf.classes_)
    results.append(_eval("nvidia_to_amd", clf, Xte, yte, wlte, classes))
    breakdowns.append(_per_workload_breakdown("nvidia_to_amd", clf, Xte, yte, wlte, classes))

    # --- (b) AMD-trained → NVIDIA-test ---
    Xtr, ytr, gtr, _ = _build_xy(amd_mask)
    Xte, yte, _, wlte = _build_xy(nv_mask)
    log.info("\n=== (b) Train: AMD  →  Test: NVIDIA ===")
    clf = TwoStageClassifier(backend=args.backend)
    clf.fit(Xtr, ytr, groups=gtr, feature_cols=feature_cols)
    classes = list(clf.classes_)
    results.append(_eval("amd_to_nvidia", clf, Xte, yte, wlte, classes))
    breakdowns.append(_per_workload_breakdown("amd_to_nvidia", clf, Xte, yte, wlte, classes))

    # --- (c) Pooled grouped 5-fold CV across both vendors ---
    pooled_mask = nv_mask | amd_mask
    X, y, g, wl = _build_xy(pooled_mask)
    n_groups = len(set(g))
    log.info(f"\n=== (c) Pooled 5-fold grouped CV (n={len(X)}, {n_groups} runs) ===")
    gkf = GroupKFold(n_splits=min(5, n_groups))
    accs = []
    binary_accs = []
    for fold, (tr, te) in enumerate(gkf.split(X, y, groups=g)):
        clf = TwoStageClassifier(backend=args.backend)
        clf.fit(X[tr], y[tr], groups=g[tr], feature_cols=feature_cols)
        classes = list(clf.classes_)
        proba = clf.predict_proba(X[te])
        pred = np.array(classes)[proba.argmax(axis=1)]
        acc = (pred == y[te]).mean()
        bacc = ((pred == "ml_training") == (y[te] == "ml_training")).mean()
        log.info(f"  fold {fold + 1}: three-way={acc:.4f}  binary={bacc:.4f}  n_test={len(te)}")
        accs.append(acc)
        binary_accs.append(bacc)
    log.info(f"Pooled CV mean: three-way={np.mean(accs):.4f} ± {np.std(accs):.4f}   "
             f"binary={np.mean(binary_accs):.4f} ± {np.std(binary_accs):.4f}")
    results.append({
        "name": "pooled_5fold_cv",
        "n_windows": int(len(X)),
        "three_way_acc": float(np.mean(accs)),
        "three_way_acc_std": float(np.std(accs)),
        "binary_train_vs_rest_acc": float(np.mean(binary_accs)),
        "binary_train_vs_rest_acc_std": float(np.std(binary_accs)),
    })

    # --- (d) AMD-only CV: does the AMD signal even support classification? ---
    if not args.no_amd_cv:
        log.info("\n=== (d) AMD-only 5-fold grouped CV ===")
        results.append(_amd_only_cv(windows, feature_cols, backend=args.backend))

    # Persist
    pd.DataFrame(results).to_csv(out_dir / "cross_vendor_summary.csv", index=False)
    pd.concat(breakdowns, ignore_index=True).to_csv(out_dir / "cross_vendor_per_workload.csv", index=False)
    log.info(f"\nSummary: {out_dir / 'cross_vendor_summary.csv'}")
    log.info(f"Per-workload: {out_dir / 'cross_vendor_per_workload.csv'}")

    log.info("\nSummary table:")
    log.info("\n" + pd.DataFrame(results).to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
