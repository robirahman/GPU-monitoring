#!/usr/bin/env python3
"""
Week 7 — Roofline Model + Tier 1 Telemetry Analysis Figures

Generates:
  1. roofline_single_gpu.png  — Roofline model: TFLOPS vs arithmetic intensity (B200 single GPU)
  2. roofline_two_gpu.png     — Roofline model: 2-GPU DataParallel width sweep
  3. roofline_comparison.png  — Roofline overlay: B200 vs H100 Week 4 reference
  4. tier1_heatmap.png        — Tier 1 feature heatmap: all workloads × all metrics
  5. tier1_cv_analysis.png    — Coefficient of variation across workloads (key discriminator)
  6. tier1_pca.png            — PCA projection of 6-metric Tier 1 feature space
  7. tier1_power_vs_util.png  — Power draw vs GPU utilization scatter (workload fingerprint)
  8. tier1_clock_analysis.png — SM clock and memory clock dynamics (B200 new feature)
  9. nvlink_roofline.png      — NVLink bandwidth curve with theoretical ceiling annotation
"""

import sys
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger("roofline_tier1")

WEEK7        = Path(__file__).parent
SG_RESULTS   = WEEK7 / "single_gpu_tests" / "results"
SG_DATA      = WEEK7 / "single_gpu_tests" / "data"
SG_PLOTS     = WEEK7 / "single_gpu_tests" / "plots"
TG_RESULTS   = WEEK7 / "two_gpu_tests" / "results"
EC_RESULTS   = WEEK7 / "edge_cases" / "results"
OUT_PLOTS    = WEEK7 / "single_gpu_tests" / "plots"   # primary output location
OUT_PLOTS.mkdir(exist_ok=True)
(WEEK7 / "two_gpu_tests" / "plots").mkdir(exist_ok=True)

# ── Hardware ceilings ──────────────────────────────────────────────────────────
B200_FP16_TFLOPS  = 4500.0   # single GPU
B200_FP32_TFLOPS  =  140.0
B200_MEM_BW_GBPS  = 8000.0   # HBM3e
B200_NVLINK5_BW   =  900.0   # per direction
B200_TDP_W        = 1000.0
H100_FP16_TFLOPS  = 1979.0   # Week 4 reference
H100_MEM_BW_GBPS  = 3350.0
H100_NVLINK4_BW   =  150.0   # Week 4 measured peak (NV6)

BYTES_PER_FLOAT   = 4

# ─────────────────────────────────────────────────────────────────────────────
# 1. ROOFLINE MODEL — Single GPU B200
# ─────────────────────────────────────────────────────────────────────────────

def plot_roofline_single_gpu():
    # Use real B200 results from scale_model_accuracy.py — contains ai and achieved_tflops
    sc_path = WEEK7 / "results" / "model_accuracy" / "model_accuracy_results.csv"
    sc = pd.read_csv(sc_path)
    # Rename columns to match expected names
    if "param_count" in sc.columns and "n_params" not in sc.columns:
        sc = sc.rename(columns={"param_count": "n_params"})
    if "n_ch" not in sc.columns:
        sc["n_ch"] = [8, 16, 32, 64, 128, 256, 512][:len(sc)]

    # Use pre-computed arithmetic intensity and TFLOPS from model_accuracy_results.csv
    arith_intensity = sc["ai"].values           # FLOP/byte, already computed correctly
    achieved_tflops_fp16 = sc["achieved_tflops"].values  # real B200 measured TFLOPS

    # Roofline ceilings
    ai_range = np.logspace(-2, 4, 500)
    roof_compute_fp16 = np.full_like(ai_range, B200_FP16_TFLOPS)
    roof_compute_fp32 = np.full_like(ai_range, B200_FP32_TFLOPS)
    roof_memory       = ai_range * (B200_MEM_BW_GBPS / 1e3)   # TB/s → TFLOPS

    # Actual roofline ceiling = min(compute, memory)
    roof_actual_fp16 = np.minimum(roof_compute_fp16, roof_memory)
    roof_actual_fp32 = np.minimum(roof_compute_fp32, roof_memory)

    ridge_fp16 = B200_FP16_TFLOPS / (B200_MEM_BW_GBPS / 1e3)
    ridge_fp32 = B200_FP32_TFLOPS / (B200_MEM_BW_GBPS / 1e3)

    fig, ax = plt.subplots(figsize=(11, 7))

    # Roofline ceiling lines
    ax.loglog(ai_range, roof_actual_fp16, color="#1565C0", lw=2.5, ls="-",
              label=f"B200 FP16 ceiling ({B200_FP16_TFLOPS:,.0f} TFLOPS)")
    ax.loglog(ai_range, roof_actual_fp32, color="#E53935", lw=2.5, ls="--",
              label=f"B200 FP32 ceiling ({B200_FP32_TFLOPS:,.0f} TFLOPS)")

    # Memory-bandwidth bound slope label
    slope_x = [1e-2, ridge_fp16 * 0.5]
    slope_y = [v * (B200_MEM_BW_GBPS / 1e3) for v in slope_x]
    ax.annotate("Memory-bandwidth\nbound slope\n(8,000 GB/s HBM3e)",
                xy=(0.05, 0.05 * B200_MEM_BW_GBPS / 1e3), fontsize=8,
                color="#555", rotation=35, ha="left")

    # Ridge points
    ax.axvline(ridge_fp16, color="#1565C0", ls=":", lw=1, alpha=0.5)
    ax.axvline(ridge_fp32, color="#E53935", ls=":", lw=1, alpha=0.5)
    ax.text(ridge_fp16 * 1.1, 0.5, f"Ridge FP16\n{ridge_fp16:.1f}", fontsize=7,
            color="#1565C0", rotation=90, va="bottom")
    ax.text(ridge_fp32 * 1.1, 0.5, f"Ridge FP32\n{ridge_fp32:.2f}", fontsize=7,
            color="#E53935", rotation=90, va="bottom")

    # Measured datapoints
    colors_map = plt.cm.plasma(np.linspace(0.1, 0.9, len(sc)))
    for i, row in sc.iterrows():
        ai     = arith_intensity[i]
        tflops = achieved_tflops_fp16[i]
        ax.scatter(ai, tflops, s=140, color=colors_map[i], zorder=10,
                   edgecolors="black", linewidths=0.7)
        ax.annotate(f"n_ch={int(row['n_ch'])}\n({row['n_params']/1e6:.1f}M)",
                    (ai, tflops), fontsize=7, ha="left", va="bottom",
                    xytext=(3, 4), textcoords="offset points")

    # Efficiency lines (25%, 50%, 75% of FP16 ceiling)
    for pct, alpha in [(0.25, 0.15), (0.50, 0.15), (0.75, 0.15)]:
        ax.loglog(ai_range, roof_actual_fp16 * pct, color="gray", lw=0.8,
                  ls=":", alpha=alpha)
        # Label at right edge
        idx = len(ai_range) - 1
        ax.text(ai_range[idx], roof_actual_fp16[idx] * pct,
                f"{int(pct*100)}%", fontsize=6, color="gray", va="center")

    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (TFLOPS)", fontsize=12)
    ax.set_title("Roofline Model — Week 7 B200 Single GPU (Model Width Sweep)",
                 fontsize=13, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-3, B200_FP16_TFLOPS * 2)

    # Annotation box
    ax.text(0.98, 0.05,
            f"B200 HBM3e: {B200_MEM_BW_GBPS:,.0f} GB/s\n"
            f"B200 FP16:  {B200_FP16_TFLOPS:,.0f} TFLOPS\n"
            f"Peak util:  {100*achieved_tflops_fp16.max()/B200_FP16_TFLOPS:.2f}% of FP16",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.8))

    plt.tight_layout()
    out = OUT_PLOTS / "roofline_single_gpu.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 2. ROOFLINE MODEL — 2-GPU DataParallel
# ─────────────────────────────────────────────────────────────────────────────

def plot_roofline_two_gpu():
    # Use model_accuracy_results.csv for proper fwd_flops and AI (same ResNet models)
    sg = pd.read_csv(WEEK7 / "results" / "model_accuracy" / "model_accuracy_results.csv")
    if "param_count" in sg.columns and "n_params" not in sg.columns:
        sg = sg.rename(columns={"param_count": "n_params"})
    if "n_ch" not in sg.columns:
        sg["n_ch"] = [8, 16, 32, 64, 128, 256, 512][:len(sg)]
    sg_by_nch = sg.set_index("n_ch")

    wd = pd.read_csv(TG_RESULTS / "model_width_sweep.csv")
    bs = pd.read_csv(TG_RESULTS / "batch_size_sweep.csv")

    # Width sweep: use profiled AI, compute 2-GPU aggregate TFLOPS from real fwd_flops
    ai_wd, tflops_wd = [], []
    for _, row in wd.iterrows():
        nch = int(row["n_ch"])
        sg_row = sg_by_nch.loc[nch]
        ai_wd.append(sg_row["ai"])
        # Aggregate TFLOPS = total fwd_flops (same batch=64 total) / 2-GPU step time
        tflops_wd.append(sg_row["fwd_flops"] / (row["mean_step_ms"] * 1e-3) / 1e12)
    ai_wd = np.array(ai_wd)
    tflops_wd = np.array(tflops_wd)

    # Batch sweep: fixed n_ch=128, scale fwd_flops by batch size
    sg_128 = sg_by_nch.loc[128]
    fwd_flops_per_sample = sg_128["fwd_flops"] / sg_128["batch_size"]
    ai_bs, tflops_bs = [], []
    for _, row in bs.iterrows():
        bsz = int(row["batch_size"])
        total_fwd_flops = fwd_flops_per_sample * bsz
        ai_bs.append(sg_128["ai"])  # AI ≈ constant for same architecture
        tflops_bs.append(total_fwd_flops / (row["mean_step_ms"] * 1e-3) / 1e12)
    ai_bs = np.array(ai_bs)
    tflops_bs = np.array(tflops_bs)

    ai_range = np.logspace(-2, 4, 500)
    roof_fp16_2gpu = np.minimum(np.full_like(ai_range, B200_FP16_TFLOPS * 2),
                                ai_range * (B200_MEM_BW_GBPS / 1e3) * 2)
    roof_fp16_1gpu = np.minimum(np.full_like(ai_range, B200_FP16_TFLOPS),
                                ai_range * (B200_MEM_BW_GBPS / 1e3))

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.loglog(ai_range, roof_fp16_2gpu, color="#1565C0", lw=2.5,
              label=f"2× B200 FP16 ceiling ({B200_FP16_TFLOPS*2:,.0f} TFLOPS)")
    ax.loglog(ai_range, roof_fp16_1gpu, color="#1565C0", lw=1.5, ls="--",
              label=f"1× B200 FP16 ceiling ({B200_FP16_TFLOPS:,.0f} TFLOPS)", alpha=0.5)

    # Width sweep points
    c_wd = plt.cm.Blues(np.linspace(0.4, 0.9, len(wd)))
    for i, row in wd.iterrows():
        ax.scatter(ai_wd[i], tflops_wd[i], s=120, color=c_wd[i], zorder=10,
                   edgecolors="#1565C0", linewidths=0.8, marker="o")
        ax.annotate(f"n_ch={int(row['n_ch'])}\n({row['n_params']/1e6:.1f}M)",
                    (ai_wd[i], tflops_wd[i]),
                    fontsize=7, ha="left", xytext=(3, 3), textcoords="offset points")

    # Batch sweep points
    c_bs = plt.cm.Oranges(np.linspace(0.4, 0.9, len(bs)))
    for i, row in bs.iterrows():
        ax.scatter(ai_bs[i], tflops_bs[i], s=100, color=c_bs[i], zorder=10,
                   edgecolors="#E65100", linewidths=0.8, marker="s")
        ax.annotate(f"bs={int(row['batch_size'])}",
                    (ai_bs[i], tflops_bs[i]),
                    fontsize=7, ha="right", xytext=(-3, 3), textcoords="offset points")

    legend_elements = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#5C9BD5",
               markeredgecolor="#1565C0", markersize=9, label="Width sweep (n_ch)"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="#FFA040",
               markeredgecolor="#E65100", markersize=9, label="Batch size sweep"),
    ]
    ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0][:2],
              loc="upper left", fontsize=9)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (TFLOPS)", fontsize=12)
    ax.set_title("Roofline Model — Week 7 2× B200 DataParallel\n"
                 "(model width sweep and batch size sweep)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-1, B200_FP16_TFLOPS * 4)

    # Annotation box
    ax.text(0.98, 0.05,
            f"B200 HBM3e: {B200_MEM_BW_GBPS:,.0f} GB/s\n"
            f"2× B200 FP16: {B200_FP16_TFLOPS*2:,.0f} TFLOPS\n"
            f"Peak achieved: {tflops_wd.max():.1f} TFLOPS ({100*tflops_wd.max()/(B200_FP16_TFLOPS*2):.2f}% of 2-GPU peak)\n"
            f"DP overhead vs 1-GPU: see comparison plot",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.8))

    plt.tight_layout()
    out = WEEK7 / "two_gpu_tests" / "plots" / "roofline_two_gpu.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 3. ROOFLINE COMPARISON — B200 vs H100 (Week 4 reference)
# ─────────────────────────────────────────────────────────────────────────────

def plot_roofline_comparison():
    """B200 single-GPU vs 2-GPU DataParallel roofline comparison."""
    # Single-GPU data from model_accuracy_results.csv
    sg = pd.read_csv(WEEK7 / "results" / "model_accuracy" / "model_accuracy_results.csv")
    if "param_count" in sg.columns and "n_params" not in sg.columns:
        sg = sg.rename(columns={"param_count": "n_params"})
    if "n_ch" not in sg.columns:
        sg["n_ch"] = [8, 16, 32, 64, 128, 256, 512][:len(sg)]
    sg_by_nch = sg.set_index("n_ch")

    ai_1gpu = sg["ai"].values
    tflops_1gpu = sg["achieved_tflops"].values

    # 2-GPU DP data — recompute TFLOPS using profiled fwd_flops
    wd = pd.read_csv(TG_RESULTS / "model_width_sweep.csv")
    ai_2gpu, tflops_2gpu = [], []
    for _, row in wd.iterrows():
        nch = int(row["n_ch"])
        sg_row = sg_by_nch.loc[nch]
        ai_2gpu.append(sg_row["ai"])
        tflops_2gpu.append(sg_row["fwd_flops"] / (row["mean_step_ms"] * 1e-3) / 1e12)
    ai_2gpu = np.array(ai_2gpu)
    tflops_2gpu = np.array(tflops_2gpu)

    ai_range = np.logspace(-2, 4, 500)
    roof_1gpu = np.minimum(np.full_like(ai_range, B200_FP16_TFLOPS),
                           ai_range * (B200_MEM_BW_GBPS / 1e3))
    roof_2gpu = np.minimum(np.full_like(ai_range, B200_FP16_TFLOPS * 2),
                           ai_range * (B200_MEM_BW_GBPS / 1e3) * 2)

    fig, ax = plt.subplots(figsize=(11, 7))
    ax.loglog(ai_range, roof_2gpu, color="#1565C0", lw=2.5,
              label=f"2× B200 FP16 ceiling ({B200_FP16_TFLOPS*2:,.0f} TFLOPS)")
    ax.loglog(ai_range, roof_1gpu, color="#F57F17", lw=2.5,
              label=f"1× B200 FP16 ceiling ({B200_FP16_TFLOPS:,.0f} TFLOPS)")

    # 1-GPU measured (diamonds)
    for i, row in sg.iterrows():
        ax.scatter(ai_1gpu[i], tflops_1gpu[i], s=140, color="#F57F17", zorder=10,
                   edgecolors="#E65100", linewidths=0.8, marker="D")
        if row["n_ch"] in [32, 64, 128, 256, 512]:
            ax.annotate(f"n_ch={int(row['n_ch'])}",
                        (ai_1gpu[i], tflops_1gpu[i]),
                        fontsize=7, color="#E65100", ha="left",
                        xytext=(4, 3), textcoords="offset points")

    # 2-GPU DP measured (circles)
    for i, row in wd.iterrows():
        ax.scatter(ai_2gpu[i], tflops_2gpu[i], s=120, color="#1565C0", zorder=10,
                   edgecolors="#0D47A1", linewidths=0.8, marker="o")
        nch = int(row["n_ch"])
        if nch in [32, 64, 128, 256, 512]:
            ax.annotate(f"n_ch={nch}",
                        (ai_2gpu[i], tflops_2gpu[i]),
                        fontsize=7, color="#0D47A1", ha="right",
                        xytext=(-4, -8), textcoords="offset points")

    # Connecting lines between 1-GPU and 2-GPU for same model
    for i, row in wd.iterrows():
        nch = int(row["n_ch"])
        sg_idx = sg[sg["n_ch"] == nch].index[0]
        ax.plot([ai_1gpu[sg_idx], ai_2gpu[i]],
                [tflops_1gpu[sg_idx], tflops_2gpu[i]],
                color="gray", ls=":", lw=0.8, alpha=0.5, zorder=1)

    legend_elements = [
        Line2D([0],[0], color="#F57F17", lw=2.5, label="1× B200 roofline"),
        Line2D([0],[0], color="#1565C0", lw=2.5, label="2× B200 roofline"),
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#F57F17",
               markeredgecolor="#E65100", markersize=9, label="1-GPU measured"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1565C0",
               markeredgecolor="#0D47A1", markersize=9, label="2-GPU DP measured"),
        Line2D([0],[0], color="gray", ls=":", lw=0.8, label="DP overhead (same model)"),
    ]
    ax.legend(handles=legend_elements, loc="upper left", fontsize=9)
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=12)
    ax.set_ylabel("Achieved Performance (TFLOPS)", fontsize=12)
    ax.set_title("B200 Roofline: 1-GPU vs 2-GPU DataParallel\n"
                 "Model Width Sweep — AMP Training (Week 7)",
                 fontsize=12, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    ax.set_xlim(1e-2, 1e4)
    ax.set_ylim(1e-1, B200_FP16_TFLOPS * 4)

    # Annotation: DP efficiency
    peak_1gpu = tflops_1gpu.max()
    peak_2gpu = tflops_2gpu.max()
    ax.text(0.98, 0.05,
            f"B200 HBM3e: {B200_MEM_BW_GBPS:,.0f} GB/s\n"
            f"1-GPU peak: {peak_1gpu:.1f} TFLOPS ({100*peak_1gpu/B200_FP16_TFLOPS:.2f}% of FP16)\n"
            f"2-GPU DP peak: {peak_2gpu:.1f} TFLOPS ({100*peak_2gpu/(B200_FP16_TFLOPS*2):.2f}% of 2-GPU FP16)\n"
            f"DP scaling efficiency: {peak_2gpu/peak_1gpu:.2f}× (ideal = 2.0×)",
            transform=ax.transAxes, fontsize=9, va="bottom", ha="right",
            bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="gray", alpha=0.8))

    plt.tight_layout()
    out = OUT_PLOTS / "roofline_comparison_1gpu_vs_2gpu.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 4. TIER 1 HEATMAP — All workloads × all metrics
# ─────────────────────────────────────────────────────────────────────────────

def load_all_parquets():
    """Load and consolidate all single-GPU parquets."""
    all_dfs = {}
    for pq in sorted(SG_DATA.glob("*.parquet")):
        label = "_".join(pq.stem.split("_")[:-1])
        df = pd.read_parquet(pq)
        if label not in all_dfs:
            all_dfs[label] = df
        else:
            all_dfs[label] = pd.concat([all_dfs[label], df], ignore_index=True)
    return all_dfs


def plot_tier1_heatmap(all_dfs):
    METRICS = ["gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
               "power_draw_w", "temperature_c", "sm_clock_mhz", "mem_clock_mhz"]
    METRIC_LABELS = ["GPU Util %", "Mem Util %", "Mem Used MB",
                     "Power W", "Temp °C", "SM Clock MHz", "Mem Clock MHz"]

    labels = sorted(all_dfs.keys())
    stats = ["mean", "std", "cv"]
    STAT_LABELS = ["Mean", "Std Dev", "CV (σ/μ)"]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle("Week 7 — Tier 1 NVML Feature Heatmap (B200 Single GPU)",
                 fontsize=14, fontweight="bold")

    for ax, stat, stat_label in zip(axes, stats, STAT_LABELS):
        matrix = np.zeros((len(labels), len(METRICS)))
        for i, lbl in enumerate(labels):
            df = all_dfs[lbl]
            for j, col in enumerate(METRICS):
                if col not in df.columns:
                    matrix[i, j] = np.nan
                    continue
                v = pd.to_numeric(df[col], errors="coerce").dropna().values
                if len(v) == 0:
                    matrix[i, j] = np.nan
                    continue
                if stat == "mean":
                    matrix[i, j] = v.mean()
                elif stat == "std":
                    matrix[i, j] = v.std()
                elif stat == "cv":
                    matrix[i, j] = v.std() / (abs(v.mean()) + 1e-9)

        # Normalize each column to [0,1] for visualization
        col_min = np.nanmin(matrix, axis=0)
        col_max = np.nanmax(matrix, axis=0)
        norm = (matrix - col_min) / (col_max - col_min + 1e-9)

        im = ax.imshow(norm, aspect="auto", cmap="YlOrRd", vmin=0, vmax=1)

        ax.set_xticks(range(len(METRICS)))
        ax.set_xticklabels(METRIC_LABELS, rotation=40, ha="right", fontsize=8)
        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.set_title(stat_label, fontweight="bold", fontsize=11)

        # Annotate with raw values
        for i in range(len(labels)):
            for j in range(len(METRICS)):
                val = matrix[i, j]
                if np.isnan(val):
                    continue
                text_color = "white" if norm[i, j] > 0.6 else "black"
                if stat == "mean":
                    txt = f"{val:.0f}" if val > 100 else f"{val:.1f}"
                elif stat == "std":
                    txt = f"{val:.1f}"
                else:
                    txt = f"{val:.2f}"
                ax.text(j, i, txt, ha="center", va="center",
                        fontsize=6.5, color=text_color)

        plt.colorbar(im, ax=ax, shrink=0.6, label="Normalized (0-1)")

    plt.tight_layout()
    out = OUT_PLOTS / "tier1_heatmap.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 5. TIER 1 CV ANALYSIS — Key discriminative feature
# ─────────────────────────────────────────────────────────────────────────────

def plot_tier1_cv_analysis(all_dfs):
    METRICS = ["gpu_utilization_pct", "power_draw_w", "sm_clock_mhz", "mem_clock_mhz", "mem_used_mb"]
    METRIC_LABELS = ["GPU Util %", "Power W", "SM Clock MHz", "Mem Clock MHz", "Mem Used MB"]
    TRAIN_LABELS = {"resnet_fp32", "resnet_amp", "mlp_training"}

    labels = sorted(all_dfs.keys())
    colors = ["#E53935" if l in TRAIN_LABELS else "#1565C0" for l in labels]

    fig, axes = plt.subplots(1, len(METRICS), figsize=(18, 6))
    fig.suptitle("Week 7 — Tier 1 Coefficient of Variation (CV) by Workload\n"
                 "(Red = ML training, Blue = non-training | CV = key discriminator)",
                 fontsize=12, fontweight="bold")

    for ax, col, col_label in zip(axes, METRICS, METRIC_LABELS):
        cvs = []
        for lbl in labels:
            df = all_dfs[lbl]
            if col not in df.columns:
                cvs.append(0)
                continue
            v = pd.to_numeric(df[col], errors="coerce").dropna().values
            cv = v.std() / (abs(v.mean()) + 1e-9) if len(v) > 1 else 0
            cvs.append(cv)

        bars = ax.bar(range(len(labels)), cvs, color=colors, edgecolor="black", linewidth=0.6)
        ax.set_xticks(range(len(labels)))
        ax.set_xticklabels(labels, rotation=50, ha="right", fontsize=7)
        ax.set_title(col_label, fontweight="bold", fontsize=9)
        ax.set_ylabel("CV (σ/μ)" if axes[0] == ax else "")
        ax.grid(True, alpha=0.3, axis="y")

        # Week 3 A100 threshold for gpu_utilization_pct
        if col == "gpu_utilization_pct":
            ax.axhline(0.30, color="green", ls="--", lw=1.2, label="W3 threshold (CV=0.30)")
            ax.legend(fontsize=7)

    legend_elements = [
        mpatches.Patch(facecolor="#E53935", edgecolor="black", label="ML Training"),
        mpatches.Patch(facecolor="#1565C0", edgecolor="black", label="Non-training"),
    ]
    fig.legend(handles=legend_elements, loc="lower center", ncol=2, fontsize=10,
               bbox_to_anchor=(0.5, -0.02))
    plt.tight_layout()
    out = OUT_PLOTS / "tier1_cv_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 6. TIER 1 PCA
# ─────────────────────────────────────────────────────────────────────────────

def plot_tier1_pca(all_dfs):
    METRICS = ["gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
               "power_draw_w", "sm_clock_mhz", "mem_clock_mhz"]
    TRAIN_LABELS = {"resnet_fp32", "resnet_amp", "mlp_training"}

    rows = []
    for label, df in all_dfs.items():
        sub = df[METRICS].apply(pd.to_numeric, errors="coerce").dropna()
        if len(sub) < 3:
            continue
        feat = {"workload": label, "is_training": label in TRAIN_LABELS}
        for col in METRICS:
            v = sub[col].values
            feat[f"{col}_mean"] = v.mean()
            feat[f"{col}_std"]  = v.std()
            feat[f"{col}_cv"]   = v.std() / (abs(v.mean()) + 1e-9)
        rows.append(feat)

    if len(rows) < 3:
        log.warning("Not enough data for PCA")
        return

    feat_df = pd.DataFrame(rows)
    feat_cols = [c for c in feat_df.columns if c not in ("workload", "is_training")]
    X = feat_df[feat_cols].fillna(0).to_numpy(dtype=float)
    X_scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2)
    Xp = pca.fit_transform(X_scaled)

    fig, ax = plt.subplots(figsize=(10, 7))
    for i, row in feat_df.iterrows():
        color = "#E53935" if row["is_training"] else "#1565C0"
        marker = "D" if row["is_training"] else "o"
        ax.scatter(Xp[i, 0], Xp[i, 1], c=color, s=160, marker=marker,
                   zorder=5, edgecolors="black", linewidths=0.8)
        ax.annotate(row["workload"], (Xp[i, 0], Xp[i, 1]),
                    fontsize=8, ha="center", va="bottom",
                    xytext=(0, 6), textcoords="offset points")

    legend_elements = [
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#E53935",
               markeredgecolor="black", markersize=11, label="ML Training"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="#1565C0",
               markeredgecolor="black", markersize=11, label="Non-training"),
    ]
    ax.legend(handles=legend_elements, fontsize=10)
    ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% variance)", fontsize=11)
    ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% variance)", fontsize=11)
    ax.set_title("Week 7 — Tier 1 PCA Projection (B200 Single GPU)\n"
                 f"18-feature space ({len(METRICS)} metrics × mean/std/cv), "
                 f"total variance explained: {sum(pca.explained_variance_ratio_)*100:.1f}%",
                 fontsize=11, fontweight="bold")
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    out = OUT_PLOTS / "tier1_pca.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 7. POWER vs UTIL SCATTER — Workload fingerprint
# ─────────────────────────────────────────────────────────────────────────────

def plot_tier1_power_vs_util(all_dfs):
    TRAIN_LABELS = {"resnet_fp32", "resnet_amp", "mlp_training"}
    colors_map = plt.cm.tab10.colors

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("Week 7 — Tier 1 Workload Fingerprints (B200)", fontsize=13, fontweight="bold")

    # Left: power vs GPU utilization (mean)
    ax = axes[0]
    for i, (label, df) in enumerate(sorted(all_dfs.items())):
        util = pd.to_numeric(df["gpu_utilization_pct"], errors="coerce").dropna()
        power = pd.to_numeric(df["power_draw_w"], errors="coerce").dropna()
        if len(util) == 0 or len(power) == 0:
            continue
        color = "#E53935" if label in TRAIN_LABELS else colors_map[i % 10]
        ax.scatter(util.mean(), power.mean(), s=200, color=color, zorder=5,
                   edgecolors="black", linewidths=0.8,
                   marker="D" if label in TRAIN_LABELS else "o")
        # Error bars
        ax.errorbar(util.mean(), power.mean(), xerr=util.std(), yerr=power.std(),
                    fmt="none", color=color, alpha=0.4, capsize=3)
        ax.annotate(label, (util.mean(), power.mean()), fontsize=7,
                    ha="left", xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("Mean GPU Utilization (%)", fontsize=11)
    ax.set_ylabel("Mean Power Draw (W)", fontsize=11)
    ax.set_title("Mean Power vs Utilization\n(error bars = 1σ)", fontsize=10, fontweight="bold")
    ax.axhline(B200_TDP_W, color="red", ls=":", lw=1, label=f"TDP {B200_TDP_W}W", alpha=0.5)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3)

    # Right: SM clock vs mem clock (B200 new feature)
    ax = axes[1]
    for i, (label, df) in enumerate(sorted(all_dfs.items())):
        if "sm_clock_mhz" not in df.columns or "mem_clock_mhz" not in df.columns:
            continue
        sm = pd.to_numeric(df["sm_clock_mhz"], errors="coerce").dropna()
        mc = pd.to_numeric(df["mem_clock_mhz"], errors="coerce").dropna()
        if len(sm) == 0 or len(mc) == 0:
            continue
        color = "#E53935" if label in TRAIN_LABELS else colors_map[i % 10]
        ax.scatter(sm.mean(), mc.mean(), s=200, color=color, zorder=5,
                   edgecolors="black", linewidths=0.8,
                   marker="D" if label in TRAIN_LABELS else "o")
        ax.errorbar(sm.mean(), mc.mean(), xerr=sm.std(), yerr=mc.std(),
                    fmt="none", color=color, alpha=0.4, capsize=3)
        ax.annotate(label, (sm.mean(), mc.mean()), fontsize=7,
                    ha="left", xytext=(4, 3), textcoords="offset points")

    ax.set_xlabel("SM Clock (MHz)", fontsize=11)
    ax.set_ylabel("Memory Clock (MHz)", fontsize=11)
    ax.set_title("SM Clock vs Memory Clock\n(B200: both vary — new discriminative feature)",
                 fontsize=10, fontweight="bold")
    ax.grid(True, alpha=0.3)

    legend_elements = [
        Line2D([0],[0], marker="D", color="w", markerfacecolor="#E53935",
               markeredgecolor="black", markersize=10, label="ML Training"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="gray",
               markeredgecolor="black", markersize=10, label="Non-training"),
    ]
    for ax in axes:
        ax.legend(handles=legend_elements + ax.get_legend_handles_labels()[0],
                  fontsize=8, loc="upper left")

    plt.tight_layout()
    out = OUT_PLOTS / "tier1_power_vs_util.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 8. CLOCK ANALYSIS — B200 new feature
# ─────────────────────────────────────────────────────────────────────────────

def plot_tier1_clock_analysis(all_dfs):
    fig, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    fig.suptitle("Week 7 — B200 Clock Dynamics (Tier 1 New Feature)\n"
                 "Memory clock varies 157→1965 MHz — unlike A100 (locked at 1215 MHz)",
                 fontsize=12, fontweight="bold")

    TRAIN_LABELS = {"resnet_fp32", "resnet_amp", "mlp_training"}
    colors_map = plt.cm.tab10.colors

    labels_order = sorted(all_dfs.keys())
    x = np.arange(len(labels_order))

    for ax, (col, ylabel) in zip(axes, [
        ("sm_clock_mhz",  "SM Clock (MHz)"),
        ("mem_clock_mhz", "Memory Clock (MHz)"),
    ]):
        means, stds, bar_colors = [], [], []
        for label in labels_order:
            df = all_dfs[label]
            if col not in df.columns:
                means.append(0); stds.append(0)
            else:
                v = pd.to_numeric(df[col], errors="coerce").dropna().values
                means.append(v.mean() if len(v) else 0)
                stds.append(v.std() if len(v) else 0)
            bar_colors.append("#E53935" if label in TRAIN_LABELS else "#1565C0")

        bars = ax.bar(x, means, yerr=stds, capsize=4, color=bar_colors,
                      edgecolor="black", linewidth=0.6)
        ax.set_xticks(x)
        ax.set_xticklabels(labels_order, rotation=35, ha="right", fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.grid(True, alpha=0.3, axis="y")

        if col == "mem_clock_mhz":
            ax.axhline(1215, color="orange", ls="--", lw=1.5,
                       label="A100 locked at 1,215 MHz (Week 3)")
            ax.axhline(1965, color="purple", ls=":", lw=1.5,
                       label="B200 peak 1,965 MHz")
            ax.legend(fontsize=8)

    legend_elements = [
        mpatches.Patch(facecolor="#E53935", edgecolor="black", label="ML Training"),
        mpatches.Patch(facecolor="#1565C0", edgecolor="black", label="Non-training"),
    ]
    fig.legend(handles=legend_elements, loc="upper right", fontsize=9)
    plt.tight_layout()
    out = OUT_PLOTS / "tier1_clock_analysis.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# 9. NVLINK ROOFLINE
# ─────────────────────────────────────────────────────────────────────────────

def plot_nvlink_roofline():
    bw_df = pd.read_csv(TG_RESULTS / "nvlink_bandwidth.csv")
    uni = bw_df[bw_df.get("note", pd.Series(dtype=str)).isna()].copy() if "note" in bw_df.columns else bw_df.copy()
    bidir_row = bw_df[bw_df.get("note", pd.Series(dtype=str)).notna()] if "note" in bw_df.columns else pd.DataFrame()

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle("B200 NVLink 5 Bandwidth Characterization (Week 7)",
                 fontsize=13, fontweight="bold")

    # Left: bandwidth vs transfer size
    ax = axes[0]
    ax.semilogx(uni["size_mb"], uni["bandwidth_gbps"], marker="o", lw=2.5,
                color="#F57F17", label="B200 NVLink 5 (measured)")
    ax.axhline(B200_NVLINK5_BW, color="#F57F17", ls="--", lw=1.5, alpha=0.7,
               label=f"Theoretical peak ({B200_NVLINK5_BW} GB/s per dir)")
    if not bidir_row.empty:
        ax.axhline(bidir_row.iloc[0]["bandwidth_gbps"] / 2, color="#E53935", ls=":",
                   lw=1.5, label=f"Bidir/2 ({bidir_row.iloc[0]['bandwidth_gbps']/2:.0f} GB/s)")
    ax.set_xlabel("Transfer size (MB, log scale)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("Unidirectional Bandwidth vs Transfer Size")
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")

    # Annotate peak achieved
    if len(uni) > 0:
        peak_bw = uni["bandwidth_gbps"].max()
        peak_size = uni.loc[uni["bandwidth_gbps"].idxmax(), "size_mb"]
        efficiency = 100 * peak_bw / B200_NVLINK5_BW
        ax.annotate(f"Peak: {peak_bw:.1f} GB/s\n({efficiency:.1f}% of {B200_NVLINK5_BW} GB/s)\nat {peak_size} MB",
                    (peak_size, peak_bw), fontsize=8, ha="left",
                    xytext=(10, -20), textcoords="offset points",
                    arrowprops=dict(arrowstyle="->", color="gray"),
                    bbox=dict(boxstyle="round,pad=0.3", fc="lightyellow", ec="gray", alpha=0.8))

    # Right: bandwidth efficiency (% of theoretical peak)
    ax = axes[1]
    efficiency_pct = 100 * uni["bandwidth_gbps"] / B200_NVLINK5_BW
    ax.semilogx(uni["size_mb"], efficiency_pct, marker="^", lw=2.5, color="#2E7D32")
    ax.axhline(100, color="gray", ls="--", lw=1, label="100% theoretical")
    ax.fill_between(uni["size_mb"], 0, efficiency_pct, alpha=0.15, color="#2E7D32")
    ax.set_xlabel("Transfer size (MB, log scale)")
    ax.set_ylabel("Bandwidth Efficiency (% of theoretical)")
    ax.set_title("NVLink 5 Bandwidth Efficiency")
    ax.set_ylim(0, 110)
    ax.legend(fontsize=8); ax.grid(True, alpha=0.3, which="both")
    for x_val, y_val in zip(uni["size_mb"], efficiency_pct):
        ax.annotate(f"{y_val:.0f}%", (x_val, y_val), fontsize=8,
                    ha="center", va="bottom", xytext=(0, 4), textcoords="offset points")

    plt.tight_layout()
    out = WEEK7 / "two_gpu_tests" / "plots" / "nvlink_roofline.png"
    plt.savefig(out, dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Saved %s", out.name)


# ─────────────────────────────────────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────────────────────────────────────

def main():
    log.info("=" * 60)
    log.info("Generating roofline + Tier 1 analysis figures")
    log.info("=" * 60)

    # Roofline plots
    log.info("1/9  Roofline — Single GPU B200...")
    plot_roofline_single_gpu()

    log.info("2/9  Roofline — 2-GPU DataParallel B200...")
    plot_roofline_two_gpu()

    log.info("3/9  Roofline Comparison — B200 1-GPU vs 2-GPU DP...")
    plot_roofline_comparison()

    # Load Tier 1 parquets
    log.info("Loading Tier 1 telemetry parquets...")
    all_dfs = load_all_parquets()
    log.info("  Loaded %d workloads: %s", len(all_dfs), list(all_dfs.keys()))

    log.info("4/9  Tier 1 Heatmap...")
    plot_tier1_heatmap(all_dfs)

    log.info("5/9  Tier 1 CV Analysis...")
    plot_tier1_cv_analysis(all_dfs)

    log.info("6/9  Tier 1 PCA...")
    plot_tier1_pca(all_dfs)

    log.info("7/9  Tier 1 Power vs Util + Clock scatter...")
    plot_tier1_power_vs_util(all_dfs)

    log.info("8/9  Tier 1 Clock Analysis (B200 new feature)...")
    plot_tier1_clock_analysis(all_dfs)

    log.info("9/9  NVLink Roofline + Speedup...")
    plot_nvlink_roofline()

    # Copy all roofline and tier1 plots to main plots/ directory
    import shutil
    main_plots = WEEK7 / "plots"
    main_plots.mkdir(exist_ok=True)
    for src_dir in [OUT_PLOTS, WEEK7 / "two_gpu_tests" / "plots"]:
        for png in src_dir.glob("*.png"):
            dst = main_plots / png.name
            shutil.copy2(png, dst)
            log.info("  Copied %s -> plots/%s", png.name, png.name)

    # Clean up old H100 comparison file if it exists
    old_comparison = OUT_PLOTS / "roofline_comparison_b200_vs_h100.png"
    if old_comparison.exists():
        old_comparison.unlink()
        log.info("  Removed stale %s", old_comparison.name)
    old_main = main_plots / "roofline_comparison_b200_vs_h100.png"
    if old_main.exists():
        old_main.unlink()
        log.info("  Removed stale %s from plots/", old_main.name)

    log.info("=" * 60)
    log.info("All figures generated and copied to plots/")
    log.info("=" * 60)

    all_out = sorted(main_plots.glob("*.png"))
    print(f"\nAll plots in {main_plots}:")
    for p in all_out:
        print(f"  {p.name}")


if __name__ == "__main__":
    main()
