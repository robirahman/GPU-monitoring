"""Tier-2 value on the hard evasions (white-box LoRA + dilution), MI300X.

Trains a binary is-training detector on the Phase-1 MI300X corpus (clean +
easy adversarial) and tests detection on the HELD-OUT hard-evasion runs
(whitebox WB-*/lora/dilution). Compares Tier-1 (NVML-equivalent) vs Tier-2
(rocprofiler) vs combined. The question: do the Tier-2 counters catch
disguised training that Tier-1 misses — the paper's <50% white-box case?

Detection rate = fraction of a workload's windows flagged "training".
For the disguised-training workloads that is recall (higher = better);
for whitebox_WB-I_inference it is the false-positive rate (lower = better).
"""
from __future__ import annotations
import logging, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__); warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))

from classifier.threeway_improved import RAW_SIGNALS
from classifier.eval_mi300x_experiments import load_harmonized, _window_tier, TIER2
from sklearn.ensemble import RandomForestClassifier


def main():
    raw = load_harmonized()
    mi = raw[raw["vendor"] == "mi300x"].copy()
    t1_cols = [c for c in RAW_SIGNALS if c in mi.columns]
    w = _window_tier(mi, t1_cols + TIER2)

    is_hard = w["workload"].str.startswith("whitebox_")
    train_w = w[~is_hard]; test_w = w[is_hard]
    log.info(f"Train (Phase-1) windows={len(train_w)} runs={train_w['run_id'].nunique()}")
    log.info(f"Test  (hard-evasion) windows={len(test_w)} runs={test_w['run_id'].nunique()}")

    ytr = (train_w["label"] == "ml_training").astype(int).values
    t1f = [f"{c}_{s}" for c in t1_cols for s in ("mean","std","min","max","p50")]
    t2f = [f"{c}_{s}" for c in TIER2 for s in ("mean","std","min","max","p50")]

    def fit_pred(cols):
        rf = RandomForestClassifier(n_estimators=400, class_weight="balanced", random_state=0, n_jobs=-1)
        rf.fit(train_w[cols].values, ytr)
        return rf.predict(test_w[cols].values)

    preds = {"Tier-1": fit_pred(t1f), "Tier-2": fit_pred(t2f), "T1+T2": fit_pred(t1f + t2f)}

    rows = []
    for wl in sorted(test_w["workload"].unique()):
        m = (test_w["workload"] == wl).values
        true_tr = (test_w.loc[test_w["workload"] == wl, "label"].iloc[0] == "ml_training")
        row = {"workload": wl, "true_training": true_tr, "n": int(m.sum())}
        for k, p in preds.items():
            row[f"{k}_detect%"] = round(100 * p[m].mean(), 1)
        rows.append(row)
    df = pd.DataFrame(rows)
    log.info("\nDetection rate (% windows flagged training) on held-out hard evasions:\n"
             + df.to_string(index=False))

    # Aggregate recall over the genuinely-training hard evasions
    tr_mask = (test_w["label"] == "ml_training").values
    log.info("\nRecall on disguised-training hard evasions (higher=better):")
    for k, p in preds.items():
        log.info(f"  {k:7} recall={100*p[tr_mask].mean():.1f}%")
    # False-positive rate on the inference control
    inf_mask = ~tr_mask
    if inf_mask.any():
        log.info("False-positive rate on whitebox inference control (lower=better):")
        for k, p in preds.items():
            log.info(f"  {k:7} FPR={100*p[inf_mask].mean():.1f}%")

    out = ROOT / "classifier" / "results" / "mi300x_hardevasion_tier2.csv"
    df.to_csv(out, index=False); log.info(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
