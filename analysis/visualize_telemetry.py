#!/usr/bin/env python3
"""GPU Telemetry Visualization — SPAR Week 3 Data

Generates plots from collected Parquet telemetry files:
  1. Time-series per metric for each workload (one file per metric)
  2. Box plots comparing all workloads per metric
  3. Heatmap: mean metric values by workload (signal comparison table)
  4. Correlation matrix between metrics
  5. Power vs GPU utilization scatter (coloured by workload)
  6. Memory usage timelines per workload category

Usage:
    python analysis/visualize_telemetry.py
    python analysis/visualize_telemetry.py --data-dir data/ --output-dir analysis/plots/
"""

import argparse
import glob
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
import pandas as pd

# ── colour palette ────────────────────────────────────────────────────────────
WORKLOAD_COLOURS = {
    "idle":                       "#aaaaaa",
    "pytorch_resnet_cifar10":     "#1f77b4",
    "pytorch_resnet_cifar10_amp": "#4fa8e0",
    "pytorch_mlp_cifar10":        "#aec7e8",
    "gpt2_wikitext2":             "#ff7f0e",
    "bert_sst2":                  "#d62728",
    "resnet50_inference":         "#2ca02c",
    "cufft_benchmark":            "#9467bd",
    "nbody_sim":                  "#8c564b",
    "mining_ethash_proxy":        "#e377c2",
    "rendering_proxy":            "#17becf",
}

METRIC_LABELS = {
    "gpu_utilization_pct": "GPU Utilization (%)",
    "mem_utilization_pct": "Memory Utilization (%)",
    "mem_used_mb":         "Memory Used (MB)",
    "power_draw_w":        "Power Draw (W)",
    "temperature_c":       "Temperature (°C)",
    "sm_clock_mhz":        "SM Clock (MHz)",
    "mem_clock_mhz":       "Memory Clock (MHz)",
    "pcie_tx_mbps":        "PCIe TX (MB/s)",
    "pcie_rx_mbps":        "PCIe RX (MB/s)",
}

NUMERIC_METRICS = list(METRIC_LABELS.keys())

WORKLOAD_CATEGORIES = {
    "Baseline":   ["idle"],
    "ML Training (FP32)": ["pytorch_resnet_cifar10", "pytorch_mlp_cifar10", "gpt2_wikitext2", "bert_sst2"],
    "ML Training (AMP)":  ["pytorch_resnet_cifar10_amp"],
    "Inference":  ["resnet50_inference"],
    "HPC":        ["cufft_benchmark", "nbody_sim"],
    "Crypto Mining": ["mining_ethash_proxy"],
    "Rendering":  ["rendering_proxy"],
}


# ── helpers ───────────────────────────────────────────────────────────────────

def load_data(data_dir: str) -> pd.DataFrame:
    files = glob.glob(os.path.join(data_dir, "*.parquet"))
    if not files:
        raise FileNotFoundError(f"No parquet files found in {data_dir}")
    dfs = []
    for f in files:
        df = pd.read_parquet(f)
        dfs.append(df)
    combined = pd.concat(dfs, ignore_index=True)
    combined["timestamp"] = pd.to_datetime(combined["timestamp_utc"], utc=True)
    return combined


def add_elapsed(df: pd.DataFrame) -> pd.DataFrame:
    """Add elapsed_s column per run_id, relative to each run's first sample."""
    df = df.copy()
    df["elapsed_s"] = df.groupby("run_id")["timestamp_epoch"].transform(lambda s: s - s.min())
    return df


def save(fig, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {path}")


# ── plot 1: time-series per metric ────────────────────────────────────────────

def plot_timeseries(df: pd.DataFrame, output_dir: str):
    """One figure per metric, one trace per (workload × run), x = elapsed seconds."""
    print("Plot 1: time-series per metric")
    ts_dir = os.path.join(output_dir, "timeseries")

    df_rel = add_elapsed(df)

    for metric, ylabel in METRIC_LABELS.items():
        fig, ax = plt.subplots(figsize=(14, 5))

        for label, colour in WORKLOAD_COLOURS.items():
            runs = df_rel[df_rel["workload_label"] == label]
            if runs.empty:
                continue
            first = True
            for run_id, run in runs.groupby("run_id"):
                run = run.sort_values("elapsed_s")
                ax.plot(
                    run["elapsed_s"],
                    run[metric],
                    color=colour,
                    alpha=0.55,
                    linewidth=1.0,
                    label=label if first else "_nolegend_",
                )
                first = False

        ax.set_xlabel("Elapsed time (s)")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — all workloads (all runs)")
        ax.legend(loc="upper right", fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)
        save(fig, os.path.join(ts_dir, f"{metric}.png"))


# ── plot 2: box plots per metric ──────────────────────────────────────────────

def plot_boxplots(df: pd.DataFrame, output_dir: str):
    """One figure per metric: box per workload showing distribution of sample values."""
    print("Plot 2: box plots per metric")
    box_dir = os.path.join(output_dir, "boxplots")

    workloads_present = sorted(df["workload_label"].unique())
    colours = [WORKLOAD_COLOURS.get(w, "#888888") for w in workloads_present]

    for metric, ylabel in METRIC_LABELS.items():
        data_by_wl = [df[df["workload_label"] == w][metric].dropna().values
                      for w in workloads_present]

        fig, ax = plt.subplots(figsize=(14, 6))
        bp = ax.boxplot(data_by_wl, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, colour in zip(bp["boxes"], colours):
            patch.set_facecolor(colour)
            patch.set_alpha(0.75)

        ax.set_xticks(range(1, len(workloads_present) + 1))
        ax.set_xticklabels(workloads_present, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel)
        ax.set_title(f"{ylabel} — distribution by workload")
        ax.grid(True, axis="y", alpha=0.3)
        save(fig, os.path.join(box_dir, f"{metric}.png"))


# ── plot 3: heatmap — mean metric by workload ─────────────────────────────────

def plot_heatmap(df: pd.DataFrame, output_dir: str):
    """Normalised heatmap: rows = workloads, columns = metrics."""
    print("Plot 3: mean-metric heatmap")
    means = df.groupby("workload_label")[NUMERIC_METRICS].mean()

    # z-score normalise each column so colour scale is comparable
    normed = (means - means.mean()) / (means.std() + 1e-9)

    fig, ax = plt.subplots(figsize=(13, 7))
    im = ax.imshow(normed.values, aspect="auto", cmap="RdYlGn", vmin=-2, vmax=2)

    ax.set_xticks(range(len(NUMERIC_METRICS)))
    ax.set_xticklabels([METRIC_LABELS[m] for m in NUMERIC_METRICS],
                       rotation=35, ha="right", fontsize=9)
    ax.set_yticks(range(len(means)))
    ax.set_yticklabels(means.index, fontsize=9)

    # Annotate cells with raw mean values
    for i, wl in enumerate(means.index):
        for j, metric in enumerate(NUMERIC_METRICS):
            val = means.loc[wl, metric]
            fmt = f"{val:.0f}" if val >= 10 else f"{val:.1f}"
            ax.text(j, i, fmt, ha="center", va="center", fontsize=7,
                    color="black")

    plt.colorbar(im, ax=ax, label="Z-score (column-normalised)")
    ax.set_title("Mean GPU Metrics by Workload  (z-score normalised per column)")
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "heatmap_mean_metrics.png"))


# ── plot 4: correlation matrix ────────────────────────────────────────────────

def plot_correlation(df: pd.DataFrame, output_dir: str):
    """Pearson correlation between all numeric metrics across all samples."""
    print("Plot 4: correlation matrix")
    corr = df[NUMERIC_METRICS].corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="coolwarm", vmin=-1, vmax=1)

    labels = [METRIC_LABELS[m] for m in NUMERIC_METRICS]
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=8)

    for i in range(len(NUMERIC_METRICS)):
        for j in range(len(NUMERIC_METRICS)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="white" if abs(corr.values[i, j]) > 0.6 else "black")

    plt.colorbar(im, ax=ax, label="Pearson r")
    ax.set_title("Metric Correlation Matrix (all workloads combined)")
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "correlation_matrix.png"))


# ── plot 5: power vs GPU utilisation scatter ──────────────────────────────────

def plot_power_vs_util(df: pd.DataFrame, output_dir: str):
    """Scatter: power_draw_w (y) vs gpu_utilization_pct (x), coloured by workload."""
    print("Plot 5: power vs GPU utilisation scatter")
    fig, ax = plt.subplots(figsize=(10, 7))

    for label, colour in WORKLOAD_COLOURS.items():
        sub = df[df["workload_label"] == label]
        if sub.empty:
            continue
        ax.scatter(
            sub["gpu_utilization_pct"],
            sub["power_draw_w"],
            c=colour,
            alpha=0.25,
            s=8,
            label=label,
            rasterized=True,
        )

    ax.set_xlabel("GPU Utilization (%)")
    ax.set_ylabel("Power Draw (W)")
    ax.set_title("Power Draw vs GPU Utilization — all workloads")
    ax.legend(loc="upper left", fontsize=8, markerscale=3)
    ax.grid(True, alpha=0.3)
    save(fig, os.path.join(output_dir, "scatter_power_vs_util.png"))


# ── plot 6: memory usage timeline per category ───────────────────────────────

def plot_memory_by_category(df: pd.DataFrame, output_dir: str):
    """Faceted time-series of mem_used_mb, one subplot per workload category."""
    print("Plot 6: memory usage by workload category")
    df_rel = add_elapsed(df)

    n_cats = len(WORKLOAD_CATEGORIES)
    fig, axes = plt.subplots(n_cats, 1, figsize=(14, 3.5 * n_cats), sharex=False)
    if n_cats == 1:
        axes = [axes]

    for ax, (cat_name, workloads) in zip(axes, WORKLOAD_CATEGORIES.items()):
        for wl in workloads:
            colour = WORKLOAD_COLOURS.get(wl, "#888888")
            runs = df_rel[df_rel["workload_label"] == wl]
            first = True
            for run_id, run in runs.groupby("run_id"):
                run = run.sort_values("elapsed_s")
                ax.plot(run["elapsed_s"], run["mem_used_mb"],
                        color=colour, alpha=0.6, linewidth=1.2,
                        label=wl if first else "_nolegend_")
                first = False
        ax.set_title(cat_name)
        ax.set_ylabel("Memory Used (MB)")
        ax.set_xlabel("Elapsed time (s)")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("GPU Memory Usage Timeline by Workload Category", fontsize=13, y=1.01)
    fig.tight_layout()
    save(fig, os.path.join(output_dir, "memory_timeline_by_category.png"))


# ── plot 7: per-workload mean ± std bar chart ─────────────────────────────────

def plot_mean_bar(df: pd.DataFrame, output_dir: str):
    """For key metrics: grouped bar chart with mean ± 1 std across all samples."""
    print("Plot 7: mean ± std bar charts for key metrics")
    bar_dir = os.path.join(output_dir, "bars")
    key_metrics = ["gpu_utilization_pct", "power_draw_w", "mem_used_mb", "temperature_c", "sm_clock_mhz"]

    for metric in key_metrics:
        stats = df.groupby("workload_label")[metric].agg(["mean", "std"]).reset_index()
        stats = stats.sort_values("mean", ascending=False)
        colours = [WORKLOAD_COLOURS.get(w, "#888888") for w in stats["workload_label"]]

        fig, ax = plt.subplots(figsize=(13, 5))
        x = range(len(stats))
        bars = ax.bar(x, stats["mean"], yerr=stats["std"], capsize=4,
                      color=colours, alpha=0.8, error_kw={"elinewidth": 1.5})

        ax.set_xticks(list(x))
        ax.set_xticklabels(stats["workload_label"], rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(METRIC_LABELS[metric])
        ax.set_title(f"{METRIC_LABELS[metric]} — mean ± 1 std by workload")
        ax.grid(True, axis="y", alpha=0.3)
        save(fig, os.path.join(bar_dir, f"{metric}_bar.png"))


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Visualise SPAR GPU telemetry data")
    parser.add_argument("--data-dir",    default="data/",           help="Directory containing .parquet files")
    parser.add_argument("--output-dir",  default="analysis/plots/", help="Directory to save plots")
    args = parser.parse_args()

    # Resolve paths relative to repo root
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir   = os.path.join(repo_root, args.data_dir)
    output_dir = os.path.join(repo_root, args.output_dir)

    print(f"Loading data from: {data_dir}")
    df = load_data(data_dir)
    print(f"Loaded {len(df):,} samples across {df['workload_label'].nunique()} workloads\n")

    plot_timeseries(df, output_dir)
    plot_boxplots(df, output_dir)
    plot_heatmap(df, output_dir)
    plot_correlation(df, output_dir)
    plot_power_vs_util(df, output_dir)
    plot_memory_by_category(df, output_dir)
    plot_mean_bar(df, output_dir)

    print(f"\nAll plots saved to: {output_dir}")


if __name__ == "__main__":
    main()
