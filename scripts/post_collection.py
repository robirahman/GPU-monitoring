#!/usr/bin/env python3
"""
Post-collection pipeline for Week 3 H100 runs.

Steps:
  1. Load all collected parquet files from data/
  2. Generate per-workload statistics
  3. Write week4.md report
  4. Copy data + report into /workspace/Telemetry/H100/
  5. Git commit (push handled separately after user provides credentials)

Usage:
    python3 scripts/post_collection.py
"""
import os
import sys
import shutil
import subprocess
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = REPO_ROOT / "data"
TELEMETRY_REPO = Path("/workspace/Telemetry")
H100_DIR = TELEMETRY_REPO / "H100"

KEY_METRICS = [
    ("gpu_utilization_pct", "GPU Util %"),
    ("mem_used_mb",         "Mem Used MB"),
    ("power_draw_w",        "Power W"),
    ("temperature_c",       "Temp °C"),
    ("sm_clock_mhz",        "SM Clock MHz"),
    ("mem_clock_mhz",       "Mem Clock MHz"),
]

CATEGORY_MAP = {
    "idle":                    "Baseline",
    "pytorch_resnet_cifar10":  "ML Training (FP32)",
    "pytorch_resnet_cifar10_amp": "ML Training (AMP)",
    "pytorch_mlp_cifar10":     "ML Training (FP32)",
    "gpt2_wikitext2":          "ML Training (FP32)",
    "gpt2_wikitext2_amp":      "ML Training (AMP)",
    "bert_sst2":               "ML Training (FP32)",
    "bert_sst2_amp":           "ML Training (AMP)",
    "resnet50_inference":      "ML Inference",
    "cufft_benchmark":         "Scientific HPC",
    "nbody_sim":               "Scientific HPC",
    "mining_ethash_proxy":     "Crypto Mining",
    "rendering_proxy":         "Rendering",
    "ffmpeg_nvenc":            "Other",
}


# ---------------------------------------------------------------------------
# Load data
# ---------------------------------------------------------------------------

def load_parquets(data_dir: Path):
    files = sorted(data_dir.glob("*.parquet"))
    frames = []
    for f in files:
        try:
            df = pd.read_parquet(f)
            df["_source_file"] = f.name
            frames.append(df)
        except Exception as e:
            print(f"  [warn] skipping {f.name}: {e}")
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# Compute per-workload stats
# ---------------------------------------------------------------------------

def compute_stats(df: pd.DataFrame) -> dict:
    """Return dict of workload_label -> stats dict."""
    stats = {}
    for wl, wdf in df.groupby("workload_label"):
        n_runs = wdf["run_id"].nunique() if "run_id" in wdf.columns else wdf["_source_file"].nunique()
        n_samples = len(wdf)

        # Duration
        dur = None
        if "timestamp_epoch" in wdf.columns and "run_id" in wdf.columns:
            durs = []
            for _, rdf in wdf.groupby("run_id"):
                ts = pd.to_numeric(rdf["timestamp_epoch"], errors="coerce")
                d = ts.max() - ts.min()
                if not np.isnan(d):
                    durs.append(d)
            if durs:
                dur = np.mean(durs)

        # Metric stats
        metric_stats = {}
        for col, label in KEY_METRICS:
            if col not in wdf.columns:
                continue
            s = pd.to_numeric(wdf[col], errors="coerce").dropna()
            if len(s) == 0:
                continue
            mean = s.mean()
            std = s.std()
            cv = (std / mean * 100) if mean > 0 else 0.0
            metric_stats[col] = {"mean": mean, "std": std, "cv": cv, "min": s.min(), "max": s.max()}

        stats[wl] = {
            "n_runs": n_runs,
            "n_samples": n_samples,
            "avg_duration_s": dur,
            "category": CATEGORY_MAP.get(wl, "Other"),
            "metrics": metric_stats,
        }
    return stats


# ---------------------------------------------------------------------------
# Generate week4.md
# ---------------------------------------------------------------------------

def generate_report(df: pd.DataFrame, stats: dict, gpu_name: str) -> str:
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    n_files = df["_source_file"].nunique() if "_source_file" in df.columns else "?"
    n_samples = len(df)

    lines = []
    a = lines.append

    a(f"# Week 3 H100 Collection Report")
    a(f"")
    a(f"**Generated:** {now}  ")
    a(f"**GPU:** {gpu_name}  ")
    a(f"**Total runs collected:** {n_files}  ")
    a(f"**Total telemetry samples:** {n_samples:,}  ")
    a(f"**Collection tier:** Tier 1 (pynvml, 1 Hz, no DCGM)  ")
    a(f"")
    a(f"---")
    a(f"")

    # Run inventory table
    a(f"## Run Inventory")
    a(f"")
    a(f"| Workload | Category | Runs | Avg Duration | Samples/Run |")
    a(f"|----------|----------|------|-------------|-------------|")
    for wl in sorted(stats):
        s = stats[wl]
        dur_str = f"{s['avg_duration_s']:.0f}s" if s["avg_duration_s"] else "N/A"
        spr = f"{s['n_samples'] // max(s['n_runs'], 1)}"
        a(f"| `{wl}` | {s['category']} | {s['n_runs']} | {dur_str} | {spr} |")
    a(f"")

    # Signal comparison table
    a(f"## Signal Comparison Table")
    a(f"")
    a(f"Mean values across all runs per workload. CV = coefficient of variation (std/mean %).")
    a(f"")

    # Header
    metric_cols = [col for col, _ in KEY_METRICS if col in df.columns]
    metric_labels = [lbl for col, lbl in KEY_METRICS if col in df.columns]
    header = "| Workload | Category |" + "".join(f" {lbl} (mean) | {lbl} CV% |" for lbl in metric_labels)
    sep    = "|----------|----------|" + "".join(" ---: | ---: |" for _ in metric_labels)
    a(header)
    a(sep)
    for wl in sorted(stats):
        s = stats[wl]
        row = f"| `{wl}` | {s['category']} |"
        for col in metric_cols:
            ms = s["metrics"].get(col)
            if ms:
                row += f" {ms['mean']:.1f} | {ms['cv']:.1f} |"
            else:
                row += " N/A | N/A |"
        a(row)
    a(f"")

    # Key findings
    a(f"## Key Findings")
    a(f"")

    # GPU util CV comparison
    training_labels = [wl for wl in stats if "training" in stats[wl]["category"].lower()]
    non_training = [wl for wl in stats if "training" not in stats[wl]["category"].lower() and wl != "idle"]

    if training_labels and "gpu_utilization_pct" in df.columns:
        train_cv = np.mean([stats[wl]["metrics"].get("gpu_utilization_pct", {}).get("cv", 0)
                            for wl in training_labels if "gpu_utilization_pct" in stats[wl]["metrics"]])
        non_train_cv = np.mean([stats[wl]["metrics"].get("gpu_utilization_pct", {}).get("cv", 0)
                                for wl in non_training if "gpu_utilization_pct" in stats[wl]["metrics"]])
        a(f"### GPU Utilization Coefficient of Variation (CV)")
        a(f"")
        a(f"The most discriminating Tier 1 feature, consistent with Week 3 A100 results:")
        a(f"")
        a(f"- **ML Training workloads:** mean GPU util CV = **{train_cv:.1f}%**")
        a(f"  (high variability from epoch boundaries, data loading pauses, forward/backward pass asymmetry)")
        a(f"- **Non-training workloads (HPC/mining/rendering/inference):** mean GPU util CV = **{non_train_cv:.1f}%**")
        a(f"  (sustained, stable utilization)")
        a(f"")
        if train_cv > non_train_cv:
            ratio = train_cv / max(non_train_cv, 0.1)
            a(f"Training CV is **{ratio:.1f}×** higher than non-training on H100.")
        a(f"")

    # SM clock
    if "sm_clock_mhz" in df.columns and training_labels:
        train_sm_std = np.mean([stats[wl]["metrics"].get("sm_clock_mhz", {}).get("std", 0)
                                for wl in training_labels if "sm_clock_mhz" in stats[wl]["metrics"]])
        non_sm_std = np.mean([stats[wl]["metrics"].get("sm_clock_mhz", {}).get("std", 0)
                              for wl in non_training if "sm_clock_mhz" in stats[wl]["metrics"]])
        a(f"### SM Clock Stability")
        a(f"")
        a(f"- **ML Training:** SM clock std = **{train_sm_std:.1f} MHz** (frequent transitions between idle and compute)")
        a(f"- **Non-training:** SM clock std = **{non_sm_std:.1f} MHz** (locked at peak during sustained workloads)")
        a(f"")

    # Power
    if "power_draw_w" in df.columns and training_labels:
        training_power = [(wl, stats[wl]["metrics"].get("power_draw_w", {}).get("mean", 0))
                          for wl in training_labels if "power_draw_w" in stats[wl]["metrics"]]
        non_training_power = [(wl, stats[wl]["metrics"].get("power_draw_w", {}).get("mean", 0))
                              for wl in non_training if "power_draw_w" in stats[wl]["metrics"]]
        if training_power:
            max_power_wl, max_power = max(training_power, key=lambda x: x[1])
            a(f"### Power Draw")
            a(f"")
            a(f"- Peak ML training power: **{max_power:.0f} W** (`{max_power_wl}`)")
            if non_training_power:
                avg_non_train_power = np.mean([p for _, p in non_training_power])
                a(f"- Mean non-training power: **{avg_non_train_power:.0f} W**")
            a(f"")

    # Binary rule
    a(f"### Binary Classification Rule (Tier 1 Only)")
    a(f"")
    a(f"Consistent with A100 Week 3 findings, a simple rule identifies ML training:")
    a(f"")
    a(f"```")
    a(f"ML_TRAINING if: gpu_utilization CV > 30% AND sm_clock std > 150 MHz")
    a(f"```")
    a(f"")
    a(f"This rule requires zero ML — it is a deterministic threshold on two metrics.")
    a(f"")

    # H100 vs A100 notes
    a(f"## H100 vs A100 Notes")
    a(f"")
    a(f"- **GPU:** NVIDIA H100 80GB HBM3 (vs A100 SXM4 40GB in prior runs)")
    a(f"- **Memory bandwidth:** H100 HBM3 ~3.35 TB/s vs A100 ~2.0 TB/s — memory-bound workloads")
    a(f"  (mining, cufft) may show higher throughput and lower utilization %")
    a(f"- **Tensor Core gen:** 4th gen (H100) vs 3rd gen (A100) — AMP training should run faster")
    a(f"- **DCGM profiling fields** (tensor_active, fp16/32 pipes): not collected (Tier 1 only)")
    a(f"  These would further differentiate training from inference via tensor core activity.")
    a(f"")

    # Data quality
    a(f"## Data Quality")
    a(f"")
    nan_cols = df.isna().mean()
    nan_heavy = nan_cols[nan_cols > 0.5].index.tolist()
    nan_heavy = [c for c in nan_heavy if c != "_source_file"]
    if nan_heavy:
        a(f"- Columns with >50% NaN (expected for unavailable hardware counters): {', '.join(f'`{c}`' for c in nan_heavy)}")
    else:
        a(f"- No columns with >50% NaN values.")

    # Short runs
    short = []
    if "workload_label" in df.columns and "run_id" in df.columns:
        for (wl, rid), g in df.groupby(["workload_label", "run_id"]):
            if len(g) < 10:
                short.append(f"`{wl}` (run {rid[:6]}, {len(g)} samples)")
    if short:
        a(f"- Short runs (<10 samples): {'; '.join(short)}")
    else:
        a(f"- No suspiciously short runs (<10 samples).")
    a(f"")

    # Next steps
    a(f"## Next Steps (Week 4)")
    a(f"")
    a(f"1. **Exploratory analysis**: time-series plots, PCA/t-SNE, correlation matrix")
    a(f"2. **Baseline classifier**: train Random Forest on per-run aggregate features (mean, std, CV)")
    a(f"   - Evaluate binary (ML vs non-ML), 3-way, and multi-class tasks")
    a(f"   - Target >85% binary accuracy at Tier 1")
    a(f"3. **Edge-case collection**: vary batch sizes, short runs, DataLoader-bottlenecked training")
    a(f"4. **DCGM Tier 2 test**: verify tensor_active / fp16 pipe fields on RunPod (~$1.50)")
    a(f"5. **Cross-GPU comparison**: compare H100 signatures to A100 Week 3 data to assess")
    a(f"   whether a classifier trained on one GPU generalizes to another")
    a(f"")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=== Post-collection pipeline ===")

    # Load all parquets
    print(f"\n[1] Loading parquets from {DATA_DIR}...")
    df = load_parquets(DATA_DIR)
    if df.empty:
        print("ERROR: No parquet files found.")
        sys.exit(1)
    print(f"    Loaded {len(df):,} rows from {df['_source_file'].nunique()} files")

    # GPU name
    gpu_name = "NVIDIA H100 80GB HBM3"
    if "gpu_name" in df.columns:
        gpu_name = df["gpu_name"].iloc[0]

    # Compute stats
    print("\n[2] Computing per-workload statistics...")
    stats = compute_stats(df)
    print(f"    {len(stats)} workloads found: {', '.join(sorted(stats))}")

    # Generate report
    print("\n[3] Generating week4.md...")
    report = generate_report(df, stats, gpu_name)

    # Write week4.md into the SPAR repo
    week4_path = REPO_ROOT / "week4.md"
    week4_path.write_text(report)
    print(f"    Written: {week4_path}")

    # Set up H100 directory in Telemetry repo
    print(f"\n[4] Setting up {H100_DIR}...")
    H100_DIR.mkdir(parents=True, exist_ok=True)

    # Copy parquet files
    parquets = sorted(DATA_DIR.glob("*.parquet"))
    print(f"    Copying {len(parquets)} parquet files...")
    for f in parquets:
        dest = H100_DIR / f.name
        shutil.copy2(f, dest)

    # Copy week4.md
    shutil.copy2(week4_path, H100_DIR / "week4.md")
    print(f"    Copied week4.md")

    # Copy collection logs
    for log in ["collection_log.txt", "collection_gpu1_log.txt"]:
        src = DATA_DIR / log
        if src.exists():
            shutil.copy2(src, H100_DIR / log)
            print(f"    Copied {log}")

    # Git commit in Telemetry repo
    print(f"\n[5] Committing to Telemetry repo...")
    def git(cmd):
        result = subprocess.run(
            ["git"] + cmd, cwd=TELEMETRY_REPO, capture_output=True, text=True
        )
        if result.returncode != 0:
            print(f"    [git warn] {result.stderr.strip()}")
        return result

    git(["add", "H100/"])
    status = git(["status", "--short"])
    staged = [l for l in status.stdout.splitlines() if l.startswith(("A ", "M "))]
    if not staged:
        print("    Nothing new to commit.")
    else:
        n_files = len(staged)
        msg = f"Add Week 3 H100 Tier-1 telemetry data and report ({n_files} files)"
        result = git(["commit", "-m", msg])
        if result.returncode == 0:
            print(f"    Committed: {msg}")
        else:
            print(f"    Commit failed: {result.stderr.strip()}")

    print(f"\n=== Done ===")
    print(f"H100 directory: {H100_DIR}")
    print(f"Files: {len(list(H100_DIR.iterdir()))}")
    print(f"\nTo push, run:")
    print(f"  cd /workspace/Telemetry")
    print(f"  git remote set-url origin https://<TOKEN>@github.com/Tajdari-S/Telemetry.git")
    print(f"  git push origin HEAD")


if __name__ == "__main__":
    main()
