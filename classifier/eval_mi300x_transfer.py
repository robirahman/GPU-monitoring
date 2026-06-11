"""Zero-shot cross-vendor transfer: NVIDIA (NVML) -> AMD MI300X (datacenter).

Trains the two-stage RandomForest on NVIDIA telemetry only, then tests it
zero-shot on the MI300X VF Phase-1 corpus. This is the cross-vendor extension
of the report's NVIDIA-only cross-GPU heatmap.

Restricted to MI300X (gpu_name contains "MI300X") so the test set is one clean
hardware family, not mixed with consumer AMD-Windows runs.

Harmonization for the VF:
  - Raw signals coerced to numeric (some sources store "N/A").
  - temperature_c (edge) is masked on the VF (sentinel -1) -> substitute the
    real temperature_hotspot_c, which is the comparable die temperature to
    NVML's GPU temp.
  - pcie_tx/rx are -1 on the VF (host-privileged) -> NaN -> 0, same treatment
    the original cross-vendor eval used for the AMD-Windows PDH gap. The model
    simply loses that signal on AMD.
  - Only the 9 shared Tier-1 RAW_SIGNALS are used (the NVIDIA model never saw
    rdc_* / hotspot / throttle), so this is a fair shared-feature transfer.

Usage:
    python -m classifier.eval_mi300x_transfer
"""
from __future__ import annotations

import argparse, logging, warnings
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
    load_all_data, sliding_windows, get_feature_cols, RAW_SIGNALS,
)
from classifier.twostage import TwoStageClassifier


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["random_forest", "xgboost"], default="random_forest")
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--stride-sec", type=float, default=15.0)
    ap.add_argument("--output", default=str(PROJECT_ROOT / "classifier" / "results" / "mi300x_transfer.csv"))
    args = ap.parse_args()

    raw = load_all_data()

    # Coerce numeric telemetry to float (the VF stores "N/A" for masked fields
    # such as mm_activity-derived encoder/decoder util).
    for c in (RAW_SIGNALS + ["temperature_hotspot_c", "encoder_util_pct",
                             "decoder_util_pct", "fan_speed_pct"]):
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")

    name = raw["gpu_name"].astype(str)
    is_mi300x = name.str.contains("MI300X", case=False, na=False)
    is_nvidia = name.str.contains("nvidia|geforce|tesla|h100|h200|a100|rtx|b200", case=False, na=False)

    # VF harmonization on the MI300X rows.
    if "temperature_hotspot_c" in raw.columns:
        sub = is_mi300x & (~(raw["temperature_c"] > 0))   # edge masked (-1/NaN)
        raw.loc[sub, "temperature_c"] = raw.loc[sub, "temperature_hotspot_c"]
    for c in ("temperature_c", "pcie_tx_mbps", "pcie_rx_mbps"):
        raw.loc[is_mi300x & (raw[c] < 0), c] = np.nan
    for c in ("pcie_tx_mbps", "pcie_rx_mbps"):
        raw[c] = raw[c].fillna(0.0)

    raw["vendor"] = np.where(is_mi300x, "mi300x", np.where(is_nvidia, "nvidia", "other"))
    keep = raw["vendor"].isin(["mi300x", "nvidia"])
    raw = raw.loc[keep].copy()
    log.info("Runs by vendor:\n" + raw.groupby("vendor")["run_id"].nunique().to_string())

    windows = sliding_windows(raw, args.window_sec, args.stride_sec)
    run_vendor = raw.groupby("run_id")["vendor"].first()
    windows["vendor"] = windows["run_id"].map(run_vendor)
    tr = windows["vendor"] == "nvidia"
    te = windows["vendor"] == "mi300x"
    log.info(f"NVIDIA train windows: {tr.sum()} | MI300X test windows: {te.sum()}")

    feats = get_feature_cols(windows)
    log.info(f"Feature cols: {len(feats)} (shared Tier-1 stats)")
    X_tr = np.nan_to_num(windows.loc[tr, feats].values, nan=0.0)
    y_tr = windows.loc[tr, "threeway_label"].values
    g_tr = windows.loc[tr, "run_id"].values
    X_te = np.nan_to_num(windows.loc[te, feats].values, nan=0.0)
    y_te = windows.loc[te, "threeway_label"].values
    wl_te = windows.loc[te, "workload_label"].values

    log.info(f"Fitting TwoStageClassifier ({args.backend}) on NVIDIA ...")
    clf = TwoStageClassifier(backend=args.backend)
    clf.fit(X_tr, y_tr, groups=g_tr, feature_cols=feats)

    proba = clf.predict_proba(X_te)
    classes = list(clf.classes_)
    pred = np.array(classes)[proba.argmax(axis=1)]

    overall = (pred == y_te).mean()
    is_tr_p, is_tr_t = (pred == "ml_training"), (y_te == "ml_training")
    binary = (is_tr_p == is_tr_t).mean()
    # binary precision/recall for training detection
    tp = (is_tr_p & is_tr_t).sum(); fp = (is_tr_p & ~is_tr_t).sum(); fn = (~is_tr_p & is_tr_t).sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    log.info(f"\n=== NVIDIA -> MI300X zero-shot (n={len(y_te)} windows) ===")
    log.info(f"Three-way per-window accuracy : {overall:.4f}")
    log.info(f"Binary is-training accuracy   : {binary:.4f}")
    log.info(f"Training detection precision  : {prec:.4f}  recall: {rec:.4f}")

    rows = []
    for wl in sorted(set(wl_te)):
        m = wl_te == wl
        t = y_te[m][0]; n = int(m.sum())
        u, c = np.unique(pred[m], return_counts=True)
        rows.append({"workload": wl, "true": t, "n": n,
                     "acc": round((pred[m] == t).mean(), 3),
                     "majority_pred": u[c.argmax()]})
    df = pd.DataFrame(rows)
    log.info("\nPer-workload (MI300X):\n" + df.to_string(index=False))
    log.info("\nConfusion (rows=true, cols=pred):\n" +
             pd.crosstab(pd.Series(y_te, name="true"), pd.Series(pred, name="pred"), margins=True).to_string())

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output, index=False)
    log.info(f"\nWrote {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
