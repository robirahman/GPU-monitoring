"""Accuracy vs telemetry sampling rate (AMD RX 9070 XT).

Reads the 4-workload sampling sweep (idle, pytorch_resnet_cifar10_amp,
llm_infer_phi3_mini, adversarial_K_online_learning) collected at 5 rates
(0.5, 1, 2, 5, 10 Hz) × 2 reps, trains a binary RandomForest at each
rate, and plots accuracy vs rate.

Future-work bullet from Appendix E: "minimum sample rate that is still
difficult to evade." This script answers it for the AMD/RX 9070 XT
single-GPU regime.

Outputs:
  report/figures/accuracy_vs_sample_rate_amd.{png,pdf}
  classifier/results/accuracy_vs_sample_rate_amd.csv
"""
from __future__ import annotations
import logging
import re
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "classifier" / "results"
RES.mkdir(parents=True, exist_ok=True)

import sys
sys.path.insert(0, str(ROOT))
from classifier.threeway_improved import sliding_windows, get_feature_cols


RATE_PATTERN = re.compile(r"_rate([0-9]+p[0-9]+|[0-9]+)h?_")

def parse_rate(fname: str) -> float | None:
    m = RATE_PATTERN.search(fname)
    if not m:
        return None
    s = m.group(1).replace("p", ".")
    return float(s)


def load_sampling_sweep() -> pd.DataFrame:
    rows = []
    for p in sorted((ROOT / "data").glob("*AMD_Radeon_RX_9070_XT_rate*.parquet")):
        rate = parse_rate(p.name)
        if rate is None:
            continue
        df = pd.read_parquet(p)
        df["sample_rate_hz"] = rate
        df["source_file"] = p.name
        rows.append(df)
    if not rows:
        raise RuntimeError("No sampling-rate parquets found")
    out = pd.concat(rows, ignore_index=True)
    log.info(f"Loaded {len(rows)} parquets, total {len(out):,} rows")
    return out


def _label_threeway(workload_label: str) -> str:
    if workload_label.startswith("adversarial") or workload_label == "pytorch_resnet_cifar10_amp":
        return "ml_training"
    if workload_label.startswith("llm_infer") or workload_label.startswith("resnet50_inference"):
        return "ml_inference"
    return "other"


def main():
    raw = load_sampling_sweep()
    raw["threeway_label"] = raw["workload_label"].apply(_label_threeway)
    raw["binary_label"] = (raw["threeway_label"] == "ml_training").astype(int)
    log.info("Per-rate row counts:")
    log.info(raw.groupby(["sample_rate_hz", "workload_label"]).size().to_string())

    results = []
    for rate, sub in raw.groupby("sample_rate_hz"):
        log.info(f"--- rate {rate} Hz ---")
        # 30-second window at this rate = ceil(30 * rate) samples
        n_per_window = max(2, int(round(30.0 * rate)))
        stride = max(1, int(round(15.0 * rate)))
        # We use sliding_windows but it needs interval-agnostic windowing.
        # Use a simpler home-grown windower because the project's sliding_windows
        # assumes 1 Hz internally for some feature computations.
        wins = []
        for rid, grp in sub.groupby("run_id"):
            grp = grp.sort_values("timestamp_epoch").reset_index(drop=True)
            for s in range(0, len(grp) - n_per_window + 1, stride):
                w = grp.iloc[s:s + n_per_window]
                wins.append({
                    "run_id": rid,
                    "binary_label": int(w["binary_label"].iloc[0]),
                    "workload_label": w["workload_label"].iloc[0],
                    # Compact feature set: mean/std/min/max of 6 raw channels
                    "util_mean": w["gpu_utilization_pct"].mean(),
                    "util_std": w["gpu_utilization_pct"].std(),
                    "util_max": w["gpu_utilization_pct"].max(),
                    "memmb_mean": w["mem_used_mb"].mean(),
                    "memmb_std": w["mem_used_mb"].std(),
                    "memmb_max": w["mem_used_mb"].max(),
                    "power_mean": w["power_draw_w"].mean(),
                    "power_std": w["power_draw_w"].std(),
                    "power_max": w["power_draw_w"].max(),
                    "smclk_mean": w["sm_clock_mhz"].mean(),
                    "smclk_std": w["sm_clock_mhz"].std(),
                    "memutil_mean": w["mem_utilization_pct"].mean(),
                    "memutil_std": w["mem_utilization_pct"].std(),
                    "temp_mean": w["temperature_c"].mean(),
                    "temp_max": w["temperature_c"].max(),
                })
        if not wins:
            log.info(f"  (no windows; skipping)")
            continue
        df = pd.DataFrame(wins).fillna(0)
        feat = [c for c in df.columns if c not in ("run_id", "binary_label", "workload_label")]
        X = df[feat].values.astype(np.float32)
        y = df["binary_label"].values
        g = df["run_id"].values
        if len(set(y)) < 2:
            log.info("  (single class only — skipping)")
            continue
        # Need enough groups per class for grouped CV
        run_labels = df.groupby("run_id")["binary_label"].first()
        min_class_groups = int(run_labels.value_counts().min())
        if min_class_groups < 2:
            log.info(f"  (only {min_class_groups} groups in minority class — skipping)")
            continue
        n_splits = max(2, min(5, min_class_groups))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs = []
        for tri, tei in cv.split(X, y, g):
            if len(tri) == 0 or len(tei) == 0:
                continue
            clf = RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            )
            clf.fit(X[tri], y[tri])
            accs.append(accuracy_score(y[tei], clf.predict(X[tei])))
        if not accs:
            continue
        results.append({
            "rate_hz": rate,
            "n_windows": len(df),
            "n_runs": int(df["run_id"].nunique()),
            "n_splits": n_splits,
            "binary_accuracy_mean": float(np.mean(accs)),
            "binary_accuracy_std": float(np.std(accs)),
        })
        log.info(f"  binary acc = {np.mean(accs):.4f} ± {np.std(accs):.4f} "
                 f"(n_windows={len(df)})")

    rdf = pd.DataFrame(results).sort_values("rate_hz")
    log.info("\nFinal:\n" + rdf.to_string(index=False))

    # Plot
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.errorbar(rdf["rate_hz"], rdf["binary_accuracy_mean"],
                yerr=rdf["binary_accuracy_std"],
                marker="o", linewidth=1.5, capsize=4, color="C0")
    ax.set_xscale("log")
    ax.set_xlabel("Telemetry sample rate (Hz)")
    ax.set_ylabel("Binary classifier accuracy")
    ax.set_ylim(0.5, 1.02)
    ax.set_title("Accuracy vs telemetry sample rate — AMD RX 9070 XT")
    ax.grid(True, alpha=0.4)
    for _, r in rdf.iterrows():
        ax.annotate(f"{r['binary_accuracy_mean']*100:.1f}%",
                    (r["rate_hz"], r["binary_accuracy_mean"]),
                    textcoords="offset points", xytext=(0, 8),
                    ha="center", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG / "accuracy_vs_sample_rate_amd.png", dpi=150)
    fig.savefig(FIG / "accuracy_vs_sample_rate_amd.pdf")
    rdf.to_csv(RES / "accuracy_vs_sample_rate_amd.csv", index=False)
    log.info(f"Saved plot + CSV")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
