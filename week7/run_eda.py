#!/usr/bin/env python3
"""
Week 7 Step 2: Exploratory Data Analysis
- Load all parquet files (existing week7 B200 edge-case data)
- Time-series visualizations per workload category
- Summary statistics and autocorrelation
- PCA and t-SNE projections of aggregate features
- Correlation matrices

Outputs plots to week7/plots/, stats to week4/results/eda_stats.csv
"""

import sys
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.cm import get_cmap

warnings.filterwarnings("ignore")

REPO_ROOT = Path(__file__).parent.parent
DATA_DIRS = [REPO_ROOT / "data", REPO_ROOT / "week7" / "data"]
PLOTS_DIR = REPO_ROOT / "week7" / "plots"
RESULTS_DIR = REPO_ROOT / "week7" / "results"
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Metrics of interest for plots and analysis
METRICS = [
    "gpu_utilization_pct",
    "mem_utilization_pct",
    "power_draw_w",
    "temperature_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "mem_used_mb",
    "pcie_tx_mbps",
    "pcie_rx_mbps",
]

WORKLOAD_COLORS = {
    "idle": "#9E9E9E",
    "pytorch_resnet_cifar10": "#2196F3",
    "pytorch_resnet_cifar10_amp": "#1565C0",
    "pytorch_mlp_cifar10": "#00BCD4",
    "gpt2_wikitext2": "#4CAF50",
    "gpt2_wikitext2_amp": "#2E7D32",
    "bert_sst2": "#FF9800",
    "bert_sst2_amp": "#E65100",
    "resnet50_inference": "#9C27B0",
    "cufft_benchmark": "#F44336",
    "nbody_sim": "#795548",
    "mining_ethash_proxy": "#FF5722",
    "rendering_proxy": "#607D8B",
    # Edge cases
    "resnet18_small_batch": "#42A5F5",
    "resnet18_large_batch": "#1E88E5",
    "resnet18_short_run": "#0D47A1",
    "mlp_small_batch": "#26C6DA",
    "mlp_large_batch": "#0097A7",
    "resnet50_inference_small": "#AB47BC",
    "resnet50_inference_large": "#7B1FA2",
    "cufft_short": "#EF5350",
    "cufft_concurrent": "#B71C1C",
    "nbody_short": "#8D6E63",
    "resnet50_concurrent": "#BA68C8",
}

CATEGORY_MAP = {
    "idle": "baseline",
    "pytorch_resnet_cifar10": "ml_training",
    "pytorch_resnet_cifar10_amp": "ml_training",
    "pytorch_mlp_cifar10": "ml_training",
    "gpt2_wikitext2": "ml_training",
    "gpt2_wikitext2_amp": "ml_training",
    "bert_sst2": "ml_training",
    "bert_sst2_amp": "ml_training",
    "resnet50_inference": "ml_inference",
    "cufft_benchmark": "scientific_hpc",
    "nbody_sim": "scientific_hpc",
    "mining_ethash_proxy": "crypto_mining",
    "rendering_proxy": "rendering",
    # Edge cases
    "resnet18_small_batch": "ml_training",
    "resnet18_large_batch": "ml_training",
    "resnet18_short_run": "ml_training",
    "mlp_small_batch": "ml_training",
    "mlp_large_batch": "ml_training",
    "resnet50_inference_small": "ml_inference",
    "resnet50_inference_large": "ml_inference",
    "cufft_short": "scientific_hpc",
    "cufft_concurrent": "scientific_hpc",
    "nbody_short": "scientific_hpc",
    "resnet50_concurrent": "ml_inference",
    "idle_h100": "baseline",
}


def load_all_parquets(dirs):
    dfs = []
    for d in dirs:
        d = Path(d)
        if not d.exists():
            continue
        for f in sorted(d.glob("*.parquet")):
            try:
                df = pd.read_parquet(f)
                if "workload_label" not in df.columns:
                    continue
                # Normalise label: strip _NVIDIA_* suffix if present
                df["workload_label"] = df["workload_label"].str.split("_NVIDIA").str[0]
                df["source_file"] = f.name
                dfs.append(df)
            except Exception as e:
                print(f"  WARN: could not load {f.name}: {e}")
    if not dfs:
        print("ERROR: No parquet files found.")
        return pd.DataFrame()
    full = pd.concat(dfs, ignore_index=True)
    # Add category column
    full["category"] = full["workload_label"].map(
        lambda x: CATEGORY_MAP.get(x, "unknown")
    )
    # Parse timestamp
    if "timestamp_utc" in full.columns:
        full["timestamp_utc"] = pd.to_datetime(full["timestamp_utc"], utc=True, errors="coerce")
    return full


def compute_relative_time(df):
    """Add elapsed_sec column (seconds since start of each run)."""
    if "run_id" not in df.columns or "timestamp_epoch" not in df.columns:
        df["elapsed_sec"] = np.arange(len(df))
        return df
    df = df.copy()
    df["elapsed_sec"] = df.groupby("run_id")["timestamp_epoch"].transform(
        lambda s: s - s.min()
    )
    return df


def plot_timeseries_by_category(df, metric="gpu_utilization_pct"):
    """Plot time-series of a metric, one subplot per category."""
    cats = sorted(df["category"].unique())
    n = len(cats)
    fig, axes = plt.subplots(n, 1, figsize=(14, 3.5 * n), sharex=False)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Time-Series: {metric}", fontsize=14, fontweight="bold")

    for ax, cat in zip(axes, cats):
        cat_df = df[df["category"] == cat]
        ax.set_title(cat.replace("_", " ").title(), fontsize=11)
        for run_id, grp in cat_df.groupby("run_id"):
            grp = grp.sort_values("elapsed_sec")
            label = grp["workload_label"].iloc[0]
            color = WORKLOAD_COLORS.get(label, "#777777")
            if metric in grp.columns:
                ax.plot(grp["elapsed_sec"], grp[metric], alpha=0.6,
                        linewidth=0.8, color=color, label=label)
        ax.set_ylabel(metric.replace("_", " "))
        ax.set_xlabel("Elapsed (s)")
        # Legend for unique labels only
        handles, labels_legend = ax.get_legend_handles_labels()
        by_label = dict(zip(labels_legend, handles))
        ax.legend(by_label.values(), by_label.keys(), fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / f"timeseries_{metric}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_summary_boxplot(df):
    """Box plots of key metrics across workload labels."""
    metrics_to_plot = [
        ("gpu_utilization_pct", "GPU Util (%)"),
        ("power_draw_w", "Power (W)"),
        ("mem_used_mb", "Memory Used (MB)"),
        ("sm_clock_mhz", "SM Clock (MHz)"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    axes = axes.flatten()
    fig.suptitle("Distribution of Key Metrics by Workload", fontsize=14, fontweight="bold")

    for ax, (metric, ylabel) in zip(axes, metrics_to_plot):
        if metric not in df.columns:
            ax.set_visible(False)
            continue
        labels_order = sorted(df["workload_label"].unique())
        data_list = [
            df[df["workload_label"] == lbl][metric].dropna().values
            for lbl in labels_order
        ]
        bp = ax.boxplot(data_list, patch_artist=True, notch=False,
                        medianprops={"color": "black", "linewidth": 2})
        for patch, lbl in zip(bp["boxes"], labels_order):
            patch.set_facecolor(WORKLOAD_COLORS.get(lbl, "#AAAAAA"))
            patch.set_alpha(0.7)
        ax.set_xticks(range(1, len(labels_order) + 1))
        ax.set_xticklabels(labels_order, rotation=45, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    out = PLOTS_DIR / "summary_boxplots.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def plot_correlation_matrix(df):
    """Correlation heat map of numeric metrics."""
    numeric_cols = [c for c in METRICS if c in df.columns]
    corr = df[numeric_cols].dropna().corr()

    fig, ax = plt.subplots(figsize=(10, 8))
    im = ax.imshow(corr.values, cmap="RdYlGn", vmin=-1, vmax=1)
    plt.colorbar(im, ax=ax, shrink=0.8)
    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.index)))
    ax.set_xticklabels(corr.columns, rotation=45, ha="right", fontsize=9)
    ax.set_yticklabels(corr.index, fontsize=9)
    ax.set_title("Telemetry Metric Correlation Matrix", fontsize=13, fontweight="bold")
    for i in range(len(corr.index)):
        for j in range(len(corr.columns)):
            ax.text(j, i, f"{corr.values[i, j]:.2f}",
                    ha="center", va="center", fontsize=7,
                    color="black" if abs(corr.values[i, j]) < 0.7 else "white")
    plt.tight_layout()
    out = PLOTS_DIR / "correlation_matrix.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def compute_aggregate_features(df):
    """Extract per-run aggregate features for PCA/t-SNE and classifier input."""
    numeric_cols = [c for c in METRICS if c in df.columns]
    records = []
    for run_id, grp in df.groupby("run_id"):
        feat = {"run_id": run_id,
                "workload_label": grp["workload_label"].iloc[0],
                "category": grp["category"].iloc[0]}
        for col in numeric_cols:
            vals = grp[col].dropna()
            if len(vals) == 0:
                continue
            feat[f"{col}_mean"]  = vals.mean()
            feat[f"{col}_std"]   = vals.std()
            feat[f"{col}_min"]   = vals.min()
            feat[f"{col}_max"]   = vals.max()
            feat[f"{col}_cv"]    = vals.std() / (vals.mean() + 1e-9)
            feat[f"{col}_p25"]   = vals.quantile(0.25)
            feat[f"{col}_p75"]   = vals.quantile(0.75)
            # FFT dominant frequency (energy in 0.1–5 Hz band)
            if len(vals) >= 16:
                fft_amp = np.abs(np.fft.rfft(vals.values - vals.mean()))
                freqs = np.fft.rfftfreq(len(vals), d=1.0)
                band = (freqs >= 0.1) & (freqs <= 5.0)
                feat[f"{col}_fft_energy"] = fft_amp[band].sum() if band.any() else 0.0
                feat[f"{col}_fft_peak"] = freqs[fft_amp.argmax()] if len(fft_amp) > 0 else 0.0
            # Memory growth rate (linear regression slope)
        if "mem_used_mb" in grp.columns:
            vals = grp["mem_used_mb"].dropna()
            if len(vals) >= 4:
                x = np.arange(len(vals))
                slope = np.polyfit(x, vals.values, 1)[0]
                feat["mem_growth_rate"] = slope
        records.append(feat)
    return pd.DataFrame(records)


def plot_pca_tsne(feature_df):
    """PCA and t-SNE projections of aggregate run features."""
    from sklearn.preprocessing import StandardScaler
    from sklearn.decomposition import PCA

    feat_cols = [c for c in feature_df.columns
                 if c not in ("run_id", "workload_label", "category")
                 and feature_df[c].dtype in (np.float64, np.float32, float)]
    X = feature_df[feat_cols].fillna(0).values
    labels = feature_df["workload_label"].values
    unique_labels = sorted(set(labels))

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # PCA
    pca = PCA(n_components=min(10, X_scaled.shape[1]))
    X_pca = pca.fit_transform(X_scaled)
    explained = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    fig.suptitle("PCA Projection of Per-Run Aggregate Features", fontsize=13, fontweight="bold")

    cmap = get_cmap("tab20")
    label_to_idx = {l: i for i, l in enumerate(unique_labels)}

    for ax, (comp_x, comp_y, title) in zip(
        axes,
        [(0, 1, f"PC1 vs PC2 (var {explained[0]:.1%} + {explained[1]:.1%})"),
         (2, 3, f"PC3 vs PC4 (var {explained[2]:.1%} + {explained[3]:.1%})" if len(explained) > 3 else "")],
    ):
        if comp_y >= X_pca.shape[1]:
            ax.set_visible(False)
            continue
        for lbl in unique_labels:
            mask = labels == lbl
            color = WORKLOAD_COLORS.get(lbl, cmap(label_to_idx[lbl] / len(unique_labels)))
            ax.scatter(X_pca[mask, comp_x], X_pca[mask, comp_y],
                       label=lbl, color=color, s=60, alpha=0.8, edgecolors="k", linewidths=0.3)
        ax.set_title(title, fontsize=10)
        ax.set_xlabel(f"PC{comp_x + 1}")
        ax.set_ylabel(f"PC{comp_y + 1}")
        ax.legend(fontsize=7, ncol=2)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    out = PLOTS_DIR / "pca_projection.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")

    # t-SNE (only if enough samples)
    if len(feature_df) >= 10:
        try:
            from sklearn.manifold import TSNE
            perp = min(30, max(5, len(feature_df) // 3))
            tsne = TSNE(n_components=2, perplexity=perp, random_state=42, max_iter=1000)
            X_tsne = tsne.fit_transform(X_scaled)

            fig, ax = plt.subplots(figsize=(12, 8))
            for lbl in unique_labels:
                mask = labels == lbl
                color = WORKLOAD_COLORS.get(lbl, cmap(label_to_idx[lbl] / len(unique_labels)))
                ax.scatter(X_tsne[mask, 0], X_tsne[mask, 1],
                           label=lbl, color=color, s=80, alpha=0.85,
                           edgecolors="k", linewidths=0.4)
            ax.set_title(f"t-SNE Projection (perplexity={perp})", fontsize=13, fontweight="bold")
            ax.set_xlabel("t-SNE dim 1")
            ax.set_ylabel("t-SNE dim 2")
            ax.legend(fontsize=8, ncol=2, loc="best")
            ax.grid(True, alpha=0.3)
            plt.tight_layout()
            out = PLOTS_DIR / "tsne_projection.png"
            plt.savefig(out, dpi=120, bbox_inches="tight")
            plt.close()
            print(f"  Saved: {out.name}")
        except Exception as e:
            print(f"  t-SNE skipped: {e}")

    return X_pca, explained, feat_cols


def plot_autocorrelation(df, metric="gpu_utilization_pct", n_lags=60):
    """Autocorrelation plots per workload category."""
    from pandas.plotting import autocorrelation_plot

    cats = sorted(df["category"].unique())
    n = len(cats)
    fig, axes = plt.subplots(n, 1, figsize=(12, 3 * n), sharex=True)
    if n == 1:
        axes = [axes]
    fig.suptitle(f"Autocorrelation of {metric}", fontsize=13, fontweight="bold")

    for ax, cat in zip(axes, cats):
        cat_df = df[df["category"] == cat]
        ax.set_title(cat.replace("_", " ").title(), fontsize=10)
        plotted = 0
        for run_id, grp in cat_df.groupby("run_id"):
            grp = grp.sort_values("elapsed_sec")
            if metric not in grp.columns or len(grp) < 20:
                continue
            series = grp[metric].dropna().values
            lags = min(n_lags, len(series) // 2)
            acf = [np.corrcoef(series[:-l], series[l:])[0, 1] if l > 0 else 1.0
                   for l in range(lags + 1)]
            color = WORKLOAD_COLORS.get(grp["workload_label"].iloc[0], "#666666")
            ax.plot(range(len(acf)), acf, alpha=0.6, linewidth=0.9, color=color)
            plotted += 1
        ax.axhline(0, color="black", linewidth=0.8, linestyle="--")
        ax.axhline(0.2, color="red", linewidth=0.5, linestyle=":")
        ax.axhline(-0.2, color="red", linewidth=0.5, linestyle=":")
        ax.set_ylabel("ACF")
        ax.set_ylim(-1.1, 1.1)
        ax.grid(True, alpha=0.3)

    axes[-1].set_xlabel("Lag (s)")
    plt.tight_layout()
    out = PLOTS_DIR / f"autocorrelation_{metric}.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"  Saved: {out.name}")


def compute_summary_stats(df):
    """Compute per-workload summary statistics."""
    numeric_cols = [c for c in METRICS if c in df.columns]
    stats = df.groupby("workload_label")[numeric_cols].agg(["mean", "std", "min", "max"])
    stats.columns = ["_".join(c) for c in stats.columns]
    stats["n_samples"] = df.groupby("workload_label").size()
    stats["n_runs"] = df.groupby("workload_label")["run_id"].nunique()
    return stats


def main():
    print("=" * 60)
    print("Week 7 EDA")
    print("=" * 60)

    print("\n[1/7] Loading parquet files...")
    df = load_all_parquets(DATA_DIRS)
    if df.empty:
        print("No data loaded. Run collect_edge_cases.py first.")
        sys.exit(0)
    df = compute_relative_time(df)
    print(f"  Loaded {len(df):,} rows, {df['run_id'].nunique()} runs, "
          f"{df['workload_label'].nunique()} workload types")
    print(f"  Workloads: {sorted(df['workload_label'].unique())}")

    print("\n[2/7] Summary statistics...")
    stats = compute_summary_stats(df)
    stats_path = RESULTS_DIR / "eda_summary_stats.csv"
    stats.to_csv(stats_path)
    print(f"  Saved: {stats_path.name}")
    print(stats[["gpu_utilization_pct_mean", "power_draw_w_mean",
                 "n_samples", "n_runs"]].to_string())

    print("\n[3/7] Time-series plots...")
    for metric in ["gpu_utilization_pct", "power_draw_w", "mem_used_mb"]:
        if metric in df.columns:
            plot_timeseries_by_category(df, metric)

    print("\n[4/7] Summary boxplots...")
    plot_summary_boxplot(df)

    print("\n[5/7] Correlation matrix...")
    plot_correlation_matrix(df)

    print("\n[6/7] Autocorrelation...")
    for metric in ["gpu_utilization_pct", "power_draw_w"]:
        if metric in df.columns:
            plot_autocorrelation(df, metric)

    print("\n[7/7] PCA and t-SNE projections...")
    feature_df = compute_aggregate_features(df)
    feat_path = RESULTS_DIR / "run_features.csv"
    feature_df.to_csv(feat_path, index=False)
    print(f"  Features for {len(feature_df)} runs saved to {feat_path.name}")
    if len(feature_df) >= 4:
        X_pca, explained, feat_cols = plot_pca_tsne(feature_df)
        pca_summary = pd.DataFrame({
            "component": range(1, len(explained) + 1),
            "explained_variance_ratio": explained,
            "cumulative_variance": np.cumsum(explained),
        })
        pca_summary.to_csv(RESULTS_DIR / "pca_variance.csv", index=False)
        print(f"  PCA: top 3 components explain {explained[:3].sum():.1%} of variance")
    else:
        print("  Skipping PCA/t-SNE (too few runs)")

    print("\n" + "=" * 60)
    print(f"EDA complete. Plots saved to: {PLOTS_DIR}")
    print(f"Stats saved to:  {RESULTS_DIR}")
    print("=" * 60)


if __name__ == "__main__":
    main()
