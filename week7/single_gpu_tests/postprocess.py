#!/usr/bin/env python3
"""Postprocess single GPU parquet files: run classifier, generate plots, save reports."""
import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("postprocess")

DATA_DIR    = Path(__file__).parent / "data"
RESULTS_DIR = Path(__file__).parent / "results"
PLOTS_DIR   = Path(__file__).parent / "plots"
for d in [RESULTS_DIR, PLOTS_DIR]: d.mkdir(parents=True, exist_ok=True)

B200_TDP_W = 1000.0

# Load all parquets
all_dfs = {}
for pq in sorted(DATA_DIR.glob("*.parquet")):
    label = pq.stem.rsplit("_", 1)[0]   # strip hex suffix
    df = pd.read_parquet(pq)
    if label not in all_dfs:
        all_dfs[label] = df
    else:
        all_dfs[label] = pd.concat([all_dfs[label], df], ignore_index=True)

log.info("Loaded %d workloads: %s", len(all_dfs), list(all_dfs.keys()))

# Summary CSV
METRICS_COLS = ["gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
                "power_draw_w", "temperature_c", "sm_clock_mhz"]
summary_rows = []
for label, df in all_dfs.items():
    row = {"workload": label, "n_samples": len(df)}
    gpu_name_col = df["gpu_name"].iloc[0] if "gpu_name" in df.columns else "B200"
    row["gpu"] = gpu_name_col
    for col in METRICS_COLS:
        if col not in df.columns: continue
        v = pd.to_numeric(df[col], errors="coerce").dropna().values
        if len(v) == 0: continue
        row[f"{col}_mean"] = float(v.mean())
        row[f"{col}_std"]  = float(v.std())
        row[f"{col}_cv"]   = float(v.std() / (abs(v.mean()) + 1e-9))
    summary_rows.append(row)

summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(RESULTS_DIR / "single_gpu_summary.csv", index=False)
log.info("Saved single_gpu_summary.csv")

# Classifier
TRAIN_LABELS = {"resnet_fp32", "resnet_amp", "mlp_training"}
feat_rows = []
for label, df in all_dfs.items():
    sub = df[METRICS_COLS].apply(pd.to_numeric, errors="coerce").dropna()
    if len(sub) < 5: continue
    feat = {"workload": label, "is_training": int(label in TRAIN_LABELS)}
    for col in METRICS_COLS:
        v = sub[col].values
        feat[f"{col}_mean"] = v.mean()
        feat[f"{col}_std"]  = v.std()
        feat[f"{col}_cv"]   = v.std() / (abs(v.mean()) + 1e-9)
    feat_rows.append(feat)

feat_df = pd.DataFrame(feat_rows)
feat_cols = [c for c in feat_df.columns if c not in ("workload", "is_training")]
X = feat_df[feat_cols].fillna(0).to_numpy(dtype=float)
y = feat_df["is_training"].to_numpy(dtype=int)
labels_arr = np.array(feat_df["workload"].tolist())

clf_results = {}
if len(set(y)) >= 2 and len(y) >= 4:
    X_tr, X_te, y_tr, y_te, l_tr, l_te = train_test_split(
        X, y, labels_arr, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc = accuracy_score(y_te, y_pred)
    log.info("Binary classifier accuracy: %.3f", acc)
    clf_results = {
        "accuracy": acc,
        "report": classification_report(y_te, y_pred,
                                        target_names=["non_training", "training"],
                                        output_dict=True),
        "feature_importances": dict(zip(feat_cols, clf.feature_importances_.tolist())),
    }
    with open(RESULTS_DIR / "classifier_results.json", "w") as f:
        json.dump(clf_results, f, indent=2)
    log.info("Saved classifier_results.json")
else:
    log.warning("Insufficient data for classifier (n=%d, classes=%s)", len(y), set(y))

# Plots
colors = plt.cm.tab10.colors
labels_order = list(all_dfs.keys())
x = range(len(labels_order))

# GPU Util
fig, ax = plt.subplots(figsize=(14, 5))
means = [all_dfs[l]["gpu_utilization_pct"].apply(pd.to_numeric, errors="coerce").mean()
         for l in labels_order]
stds  = [all_dfs[l]["gpu_utilization_pct"].apply(pd.to_numeric, errors="coerce").std()
         for l in labels_order]
ax.bar(x, means, yerr=stds, capsize=4,
       color=[colors[i % 10] for i in range(len(labels_order))],
       edgecolor="black", linewidth=0.7)
ax.set_xticks(list(x))
ax.set_xticklabels(labels_order, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("GPU Utilization (%)"); ax.set_ylim(0, 105)
ax.set_title("Week 7 — B200 Single GPU: GPU Utilization by Workload", fontweight="bold")
ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "gpu_util_by_workload.png", dpi=120, bbox_inches="tight")
plt.close()

# Power
fig, ax = plt.subplots(figsize=(14, 5))
pmeans = [all_dfs[l]["power_draw_w"].apply(pd.to_numeric, errors="coerce").mean()
          for l in labels_order]
ax.bar(x, pmeans,
       color=[colors[i % 10] for i in range(len(labels_order))],
       edgecolor="black", linewidth=0.7)
ax.set_xticks(list(x))
ax.set_xticklabels(labels_order, rotation=35, ha="right", fontsize=9)
ax.set_ylabel("Power Draw (W)")
ax.set_title("Week 7 — B200 Single GPU: Power Draw by Workload", fontweight="bold")
ax.axhline(B200_TDP_W, color="red", ls="--", lw=1.2, label=f"B200 TDP ({B200_TDP_W}W)")
ax.legend(); ax.grid(True, alpha=0.3, axis="y")
plt.tight_layout()
plt.savefig(PLOTS_DIR / "power_by_workload.png", dpi=120, bbox_inches="tight")
plt.close()

# Time-series per workload
for label, df in all_dfs.items():
    if "gpu_utilization_pct" not in df.columns: continue
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    fig.suptitle(f"Week 7 — {label} — B200 Telemetry", fontsize=12, fontweight="bold")
    t = pd.to_numeric(df["timestamp_epoch"], errors="coerce").values
    t = t - t[0]
    for ax, (col, ylabel, color) in zip(axes, [
        ("gpu_utilization_pct", "GPU Util (%)", "steelblue"),
        ("power_draw_w",        "Power (W)",    "firebrick"),
        ("mem_used_mb",         "Memory (MB)",  "darkorange"),
    ]):
        if col in df.columns:
            v = pd.to_numeric(df[col], errors="coerce").values
            ax.plot(t, v, color=color, lw=0.9)
        ax.set_ylabel(ylabel); ax.grid(True, alpha=0.3)
    axes[-1].set_xlabel("Elapsed (s)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / f"timeseries_{label}.png", dpi=100, bbox_inches="tight")
    plt.close()

# Model scaling
scale_csv = RESULTS_DIR / "model_scaling.csv"
if scale_csv.exists():
    sc = pd.read_csv(scale_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.suptitle("Week 7 — Model Width Scaling (B200 Single GPU)", fontweight="bold")
    axes[0].plot(sc["n_ch"], sc["mean_step_ms"], marker="o", color="steelblue")
    axes[0].set(xlabel="n_ch", ylabel="Step time (ms)", title="Step Time vs Width")
    axes[0].grid(True, alpha=0.3)
    axes[1].plot(sc["n_params"] / 1e6, sc["tflops"], marker="s", color="firebrick")
    axes[1].set(xlabel="Params (M)", ylabel="TFLOPS", title="Compute Efficiency")
    axes[1].grid(True, alpha=0.3)
    axes[2].loglog(sc["n_params"], sc["mean_step_ms"], marker="^", color="darkorange")
    axes[2].set(xlabel="Params (log)", ylabel="Step (ms, log)", title="Roofline Proxy")
    axes[2].grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "model_scaling.png", dpi=120, bbox_inches="tight")
    plt.close()

log.info("All plots saved → %s", PLOTS_DIR)

# Print summary
print("\n" + "="*70)
print(f"{'Workload':<20} {'GPU Util%':>9} {'Power W':>8} {'Mem MB':>8}")
print("-"*70)
for row in summary_rows:
    print(f"{row['workload']:<20} "
          f"{row.get('gpu_utilization_pct_mean', float('nan')):>9.1f} "
          f"{row.get('power_draw_w_mean', float('nan')):>8.1f} "
          f"{row.get('mem_used_mb_mean', float('nan')):>8.0f}")
print("="*70)
if clf_results:
    print(f"\nBinary classifier accuracy: {clf_results['accuracy']:.1%}")
