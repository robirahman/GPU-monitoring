#!/usr/bin/env python3
"""
make_scaling_plots.py — Improved roofline visualisation for scale_to_bottleneck results.

Generates three figures:
  1. scaling_roofline_combined.png  — ONE roofline with all model sizes, forward and
                                       backward+allreduce shown as separate markers,
                                       NVLink shift arrows between them.
  2. scaling_roofline_regimes.png  — Zoomed annotated view of the width sweep showing
                                       the three regime boundaries.
  3. scaling_step_breakdown.png    — Step-time breakdown (fwd/bwd/allreduce/opt) for
                                       both sweeps side-by-side.
"""

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.colors as mcolors
import matplotlib.cm as cm
from pathlib import Path

RESULTS_JSON = Path("/root/Telemetry/week7/results/scaling/scale_results.json")
PLOTS_DIR    = Path("/root/Telemetry/week7/plots")

# H100 hardware ceilings
B200_FP16_TFLOPS = 4500.0
H100_FP16_TFLOPS = 4500.0
B200_FP32_TFLOPS = 140.0
H100_FP32_TFLOPS = 140.0
B200_MEM_BW_GBPS = 8000.0
H100_MEM_BW_GBPS = 8000.0
NVLINK_BW_GBPS   =  900.0
ELEM_BYTES       =      4

RIDGE_FP16 = H100_FP16_TFLOPS / (H100_MEM_BW_GBPS / 1e3)   # ~591 FLOP/B
RIDGE_FP32 = H100_FP32_TFLOPS / (H100_MEM_BW_GBPS / 1e3)   # ~20 FLOP/B


def load_data() -> pd.DataFrame:
    with open(RESULTS_JSON) as f:
        raw = json.load(f)
    df = pd.DataFrame(raw)
    if "oom" not in df.columns:
        df["oom"] = False
    df = df[~df["oom"]].copy()

    # Derive per-pass metrics
    # Forward: fwd_flops / fwd_ms
    df["fwd_perf_tflops"] = df["fwd_flops"] / (df["fwd_ms"] * 1e-3) / 1e12
    df["fwd_ai"]          = df["fwd_flops"] / df["fwd_bytes"]

    # Backward compute only (subtract estimated allreduce from bwd_ms)
    df["bwd_compute_ms"]  = (df["bwd_ms"] - df["nvlink_allreduce_ms"]).clip(lower=0.01)
    df["bwd_compute_perf_tflops"] = (2 * df["fwd_flops"] /
                                     (df["bwd_compute_ms"] * 1e-3) / 1e12)
    df["bwd_compute_ai"]  = df["fwd_ai"]   # same ops as fwd, same intensity

    # Backward total (includes allreduce): lower perf, lower effective AI
    df["bwd_total_perf_tflops"] = (2 * df["fwd_flops"] /
                                   (df["bwd_ms"] * 1e-3) / 1e12)
    # Effective AI when allreduce bytes are included in denominator
    grad_bytes = df["param_count"] * ELEM_BYTES
    df["bwd_total_ai"] = (2 * df["fwd_flops"]) / (2 * df["fwd_bytes"] + grad_bytes)

    return df


def roofline_ceil(ai_arr: np.ndarray):
    fp16 = np.minimum(H100_FP16_TFLOPS, ai_arr * H100_MEM_BW_GBPS / 1e3)
    fp32 = np.minimum(H100_FP32_TFLOPS, ai_arr * H100_MEM_BW_GBPS / 1e3)
    return fp16, fp32


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: Combined roofline — all model sizes, forward AND backward
# ─────────────────────────────────────────────────────────────────────────────
def plot_combined_roofline(df: pd.DataFrame):
    fig, ax = plt.subplots(figsize=(14, 9))

    ai_arr = np.logspace(-1, 5, 1000)
    fp16, fp32 = roofline_ceil(ai_arr)

    # ── Ceilings ──────────────────────────────────────────────────────────────
    ax.loglog(ai_arr, fp16, color="#1565C0", lw=2.2, zorder=2,
              label=f"H100 FP16 tensor-core ceiling  {H100_FP16_TFLOPS:.0f} TFLOPS")
    ax.loglog(ai_arr, fp32, color="#0097A7", lw=1.6, ls="--", zorder=2,
              label=f"H100 FP32 CUDA-core ceiling  {H100_FP32_TFLOPS:.0f} TFLOPS")

    # Ridge-point vertical lines
    ax.axvline(RIDGE_FP16, color="#1565C0", lw=0.9, ls=":", alpha=0.5)
    ax.axvline(RIDGE_FP32, color="#0097A7", lw=0.9, ls=":", alpha=0.5)
    ax.text(RIDGE_FP16 * 1.06, 1e-2,
            f"FP16 ridge\n{RIDGE_FP16:.0f} FLOP/B",
            fontsize=7.5, color="#1565C0", va="bottom")
    ax.text(RIDGE_FP32 * 1.06, 1e-2,
            f"FP32 ridge\n{RIDGE_FP32:.1f} FLOP/B",
            fontsize=7.5, color="#0097A7", va="bottom")

    # ── Regime shading ────────────────────────────────────────────────────────
    ax.axvspan(0.1, RIDGE_FP32, alpha=0.04, color="royalblue", label="_nolegend_")
    ax.axvspan(RIDGE_FP32, RIDGE_FP16, alpha=0.04, color="gold",   label="_nolegend_")
    ax.axvspan(RIDGE_FP16, 1e5,        alpha=0.04, color="salmon", label="_nolegend_")

    ax.text(0.13, 1800, "Memory-\nbound", fontsize=8, color="royalblue", alpha=0.8,
            fontstyle="italic")
    ax.text(RIDGE_FP32 * 1.2, 1800, "Transition", fontsize=8, color="#9E6C00",
            alpha=0.8, fontstyle="italic")
    ax.text(RIDGE_FP16 * 1.2, 1800, "Compute-\nbound", fontsize=8, color="firebrick",
            alpha=0.8, fontstyle="italic")

    # ── Batch-sweep points (n_ch=64) ─────────────────────────────────────────
    batch = df[df["experiment"] == "batch_sweep"].sort_values("batch_size")
    bs_vals = batch["batch_size"].values
    norm_bs = mcolors.LogNorm(vmin=bs_vals.min(), vmax=bs_vals.max())
    cmap_bs = cm.Blues

    for _, row in batch.iterrows():
        c = cmap_bs(norm_bs(row["batch_size"]))
        # Forward
        ax.scatter(row["fwd_ai"], row["fwd_perf_tflops"],
                   marker="^", s=80, color=c, edgecolors="#1a237e", lw=0.6,
                   zorder=5, alpha=0.85)
        # Backward+allreduce (negligible allreduce for n_ch=64, so near forward)
        ax.scatter(row["bwd_total_ai"], row["bwd_total_perf_tflops"],
                   marker="s", s=60, color=c, edgecolors="#1a237e", lw=0.6,
                   zorder=5, alpha=0.85)
        # Label largest batch
        if row["batch_size"] in [1, 64, 1024]:
            ax.annotate(f"bs={int(row['batch_size'])}",
                        xy=(row["fwd_ai"], row["fwd_perf_tflops"]),
                        xytext=(5, 4), textcoords="offset points", fontsize=7,
                        color="#1a237e", alpha=0.85)

    # ── Width-sweep points ───────────────────────────────────────────────────
    width = df[df["experiment"] == "width_sweep"].sort_values("n_ch")
    n_ch_vals = sorted(width["n_ch"].unique())
    cmap_w = cm.YlOrRd
    norm_w = mcolors.LogNorm(vmin=min(n_ch_vals), vmax=max(n_ch_vals))

    for _, row in width.iterrows():
        c = cmap_w(norm_w(row["n_ch"]))
        # Forward point (triangle up, hollow)
        ax.scatter(row["fwd_ai"], row["fwd_perf_tflops"],
                   marker="^", s=140, facecolors="none", edgecolors=c, lw=2.0,
                   zorder=6)
        # Backward compute (circle, hollow)
        ax.scatter(row["bwd_compute_ai"], row["bwd_compute_perf_tflops"],
                   marker="o", s=120, facecolors="none", edgecolors=c, lw=2.0,
                   zorder=6)
        # Backward+allreduce (square, filled) — shifted left by allreduce bytes
        ax.scatter(row["bwd_total_ai"], row["bwd_total_perf_tflops"],
                   marker="s", s=130, color=c, edgecolors="k", lw=0.8,
                   zorder=6)

        # Arrow from bwd-compute to bwd+allreduce (shows NVLink drag)
        if abs(row["bwd_total_ai"] - row["bwd_compute_ai"]) > 1:
            ax.annotate(
                "", xy=(row["bwd_total_ai"], row["bwd_total_perf_tflops"]),
                xytext=(row["bwd_compute_ai"], row["bwd_compute_perf_tflops"]),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.2,
                                mutation_scale=10),
                zorder=4)

        # Label the forward point with n_ch
        ax.annotate(
            f"n={int(row['n_ch'])}\n{int(row['param_count']/1e6)}M",
            xy=(row["fwd_ai"], row["fwd_perf_tflops"]),
            xytext=(6, 6), textcoords="offset points",
            fontsize=7.5, color="k",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.6, lw=0))

        # Label comm fraction on bwd+allreduce point for large models
        if row["comm_fraction"] > 0.1:
            ax.annotate(
                f"{row['comm_fraction']*100:.0f}% NVLink",
                xy=(row["bwd_total_ai"], row["bwd_total_perf_tflops"]),
                xytext=(-55, -16), textcoords="offset points",
                fontsize=6.5, color="saddlebrown",
                arrowprops=dict(arrowstyle="-", color="saddlebrown", lw=0.6))

    # ── Legend ────────────────────────────────────────────────────────────────
    legend_elements = [
        mpatches.Patch(facecolor="none", label="Ceilings / boundaries", linewidth=0),
        plt.Line2D([0], [0], color="#1565C0", lw=2.2,
                   label="H100 FP16 ceiling  1979 TFLOPS"),
        plt.Line2D([0], [0], color="#0097A7", lw=1.6, ls="--",
                   label="H100 FP32 ceiling  67 TFLOPS"),
        mpatches.Patch(facecolor="none", label="", linewidth=0),
        mpatches.Patch(facecolor="none", label="Markers", linewidth=0),
        plt.scatter([], [], marker="^", s=90, facecolors="none",
                    edgecolors="gray", lw=1.8, label="Forward pass (width sweep)"),
        plt.scatter([], [], marker="o", s=80, facecolors="none",
                    edgecolors="gray", lw=1.8, label="Backward compute only (width sweep)"),
        plt.scatter([], [], marker="s", s=90, color="gray", edgecolors="k",
                    label="Backward + NVLink allreduce (width sweep)"),
        plt.scatter([], [], marker="^", s=70, color="#6baed6", edgecolors="#1a237e",
                    label="Forward / bwd (batch sweep, n_ch=64)"),
        plt.Line2D([0], [0], color="gray", lw=1.2,
                   marker=">", ms=5, label="NVLink drag (arrow = allreduce shift)"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left",
              framealpha=0.85, ncol=1)

    # Colourbars
    sm_w  = plt.cm.ScalarMappable(cmap=cmap_w, norm=norm_w)
    sm_bs = plt.cm.ScalarMappable(cmap=cmap_bs, norm=norm_bs)
    sm_w.set_array([]);  sm_bs.set_array([])
    cb_w  = fig.colorbar(sm_w,  ax=ax, shrink=0.35, pad=0.01, location="right",
                          anchor=(0.0, 0.9))
    cb_bs = fig.colorbar(sm_bs, ax=ax, shrink=0.35, pad=0.01, location="right",
                          anchor=(0.0, 0.45))
    cb_w.set_label("Width sweep n_ch", fontsize=8)
    cb_bs.set_label("Batch sweep batch_size", fontsize=8)

    ax.set_xlabel("Arithmetic Intensity  (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Achieved Throughput  (TFLOPS)", fontsize=12)
    ax.set_title(
        "Roofline Model — All Scales, Forward vs Backward+NVLink AllReduce\n"
        "2× H100 80GB HBM3, NVLink 4.0 NV6  |  DDP ResNet-style CNN",
        fontsize=11, fontweight="bold")
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(0.1, 3e4)
    ax.set_ylim(0.05, H100_FP16_TFLOPS * 1.8)

    out = PLOTS_DIR / "scaling_roofline_combined.png"
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: Width-sweep regime zoom with annotated transitions
# ─────────────────────────────────────────────────────────────────────────────
def plot_regime_zoom(df: pd.DataFrame):
    width = df[df["experiment"] == "width_sweep"].sort_values("n_ch")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # ── Left: roofline zoom for width sweep ──────────────────────────────────
    ax = axes[0]
    ai_arr = np.logspace(1, 5, 600)
    fp16, fp32 = roofline_ceil(ai_arr)
    ax.loglog(ai_arr, fp16, "#1565C0", lw=2.2,
              label=f"FP16 ceiling {H100_FP16_TFLOPS:.0f} TFLOPS")
    ax.loglog(ai_arr, fp32, "#0097A7", lw=1.6, ls="--",
              label=f"FP32 ceiling {H100_FP32_TFLOPS:.0f} TFLOPS")
    ax.axvline(RIDGE_FP16, color="#1565C0", lw=1, ls=":", alpha=0.6)
    ax.axvline(RIDGE_FP32, color="#0097A7", lw=1, ls=":", alpha=0.6)

    colors = plt.cm.plasma(np.linspace(0.15, 0.9, len(width)))
    for i, (_, row) in enumerate(width.iterrows()):
        c = colors[i]
        # Plot three points per model: fwd, bwd_compute, bwd+allreduce
        ax.scatter(row["fwd_ai"], row["fwd_perf_tflops"],
                   marker="^", s=160, color=c, edgecolors="k", lw=0.7, zorder=7)
        ax.scatter(row["bwd_compute_ai"], row["bwd_compute_perf_tflops"],
                   marker="o", s=130, facecolors="none", edgecolors=c, lw=2.2, zorder=7)
        ax.scatter(row["bwd_total_ai"], row["bwd_total_perf_tflops"],
                   marker="s", s=150, color=c, edgecolors="k", lw=0.7, zorder=7,
                   alpha=0.75)

        # NVLink arrow: bwd_compute → bwd_total
        if abs(row["bwd_total_ai"] - row["bwd_compute_ai"]) > 0.5:
            ax.annotate(
                "", xy=(row["bwd_total_ai"], row["bwd_total_perf_tflops"]),
                xytext=(row["bwd_compute_ai"], row["bwd_compute_perf_tflops"]),
                arrowprops=dict(arrowstyle="-|>", color="gray", lw=1.3,
                                connectionstyle="arc3,rad=-0.1", mutation_scale=11),
                zorder=5)

        # Label
        xoff, yoff = (6, 8) if i % 2 == 0 else (-55, -18)
        ax.annotate(
            f"n={int(row['n_ch'])} ({int(row['param_count']/1e6)}M params)\n"
            f"grad={row['grad_mb']:.0f} MB  comm={row['comm_fraction']*100:.0f}%",
            xy=(row["fwd_ai"], row["fwd_perf_tflops"]),
            xytext=(xoff, yoff), textcoords="offset points",
            fontsize=7, color=c,
            bbox=dict(boxstyle="round,pad=0.25", fc="white", alpha=0.7, lw=0))

    legend_elements = [
        plt.scatter([], [], marker="^", s=100, color="gray", edgecolors="k",
                    lw=0.7, label="Forward pass"),
        plt.scatter([], [], marker="o", s=100, facecolors="none",
                    edgecolors="gray", lw=2.0, label="Backward compute only"),
        plt.scatter([], [], marker="s", s=100, color="gray", edgecolors="k",
                    lw=0.7, alpha=0.75, label="Backward + NVLink allreduce"),
        plt.Line2D([0], [0], color="gray", lw=1.3, marker=">", ms=5,
                   label="NVLink drag arrow"),
    ]
    ax.legend(handles=legend_elements, fontsize=8, loc="upper left")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=11)
    ax.set_ylabel("Achieved Throughput (TFLOPS)", fontsize=11)
    ax.set_title("Width Sweep Roofline — Forward vs Backward vs Backward+AllReduce\n"
                 "Gray arrows show NVLink overhead shift", fontsize=10)
    ax.grid(True, which="both", alpha=0.22)
    ax.set_xlim(20, 5e4)
    ax.set_ylim(0.1, H100_FP16_TFLOPS * 1.5)

    # ── Right: time breakdown bars showing fwd / bwd_compute / allreduce / opt ─
    ax2 = axes[1]
    x = np.arange(len(width))
    bw = 0.55

    fwd_ms   = width["fwd_ms"].values
    bwd_c_ms = width["bwd_compute_ms"].values
    allr_ms  = width["nvlink_allreduce_ms"].values
    opt_ms   = width["opt_ms"].values

    b1 = ax2.bar(x, fwd_ms,         width=bw, color="#4C72B0", label="Forward")
    b2 = ax2.bar(x, bwd_c_ms,       width=bw, bottom=fwd_ms,
                 color="#DD8452", label="Backward compute")
    b3 = ax2.bar(x, allr_ms,        width=bw, bottom=fwd_ms + bwd_c_ms,
                 color="#C44E52", label="NVLink all-reduce")
    b4 = ax2.bar(x, opt_ms,         width=bw,
                 bottom=fwd_ms + bwd_c_ms + allr_ms,
                 color="#55A868", label="Optimizer step")

    ax2.set_xticks(x)
    ax2.set_xticklabels(
        [f"n={int(r['n_ch'])}\n{int(r['param_count']/1e6)}M" for _, r in width.iterrows()],
        fontsize=8.5)
    ax2.set_ylabel("Time per training step (ms)", fontsize=11)
    ax2.set_title("Step-Time Breakdown by Model Width\n"
                  "Red = NVLink allreduce overhead", fontsize=10)
    ax2.legend(fontsize=9, loc="upper left")
    ax2.grid(axis="y", alpha=0.3)

    # Annotate allreduce % on each bar
    for i, (_, row) in enumerate(width.iterrows()):
        pct = row["comm_fraction"] * 100
        y_pos = row["fwd_ms"] + row["bwd_compute_ms"] + row["nvlink_allreduce_ms"] / 2
        ax2.text(i, y_pos, f"{pct:.0f}%", ha="center", va="center",
                 fontsize=7.5, color="white", fontweight="bold")

    plt.suptitle("Width Sweep: Three Performance Regimes on 2× H100 NV6\n"
                 "Memory-bound → Compute-bound → NVLink all-reduce bound",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "scaling_roofline_regimes.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Step breakdown both sweeps + NVLink fraction
# ─────────────────────────────────────────────────────────────────────────────
def plot_step_breakdown(df: pd.DataFrame):
    batch = df[df["experiment"] == "batch_sweep"].sort_values("batch_size")
    width = df[df["experiment"] == "width_sweep"].sort_values("n_ch")

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    for ax, sub, xlabel_fn, title in [
        (axes[0], batch,
         lambda r: f"bs={int(r['batch_size'])}",
         "Batch-size Sweep (n_ch=64)"),
        (axes[1], width,
         lambda r: f"n={int(r['n_ch'])}\n{int(r['param_count']/1e6)}M",
         "Width Sweep (batch=64)"),
    ]:
        x = np.arange(len(sub))
        bw = 0.6
        fwd   = sub["fwd_ms"].values
        bwd_c = sub["bwd_compute_ms"].values
        allr  = sub["nvlink_allreduce_ms"].values
        opt   = sub["opt_ms"].values

        ax.bar(x, fwd,   width=bw, color="#4C72B0", label="Forward")
        ax.bar(x, bwd_c, width=bw, bottom=fwd, color="#DD8452", label="Backward compute")
        ax.bar(x, allr,  width=bw, bottom=fwd + bwd_c,
               color="#C44E52", label="NVLink all-reduce")
        ax.bar(x, opt,   width=bw, bottom=fwd + bwd_c + allr,
               color="#55A868", label="Optimizer")

        ax.set_xticks(x)
        ax.set_xticklabels([xlabel_fn(r) for _, r in sub.iterrows()],
                           fontsize=8, rotation=30 if len(sub) > 7 else 0,
                           ha="right" if len(sub) > 7 else "center")
        ax.set_ylabel("Time per step (ms)", fontsize=11)
        ax.set_title(title, fontsize=10)
        ax.legend(fontsize=8)
        ax.grid(axis="y", alpha=0.3)

        # Secondary axis: throughput (imgs/s)
        ax2 = ax.twinx()
        imgs_per_s = sub["batch_size"] / (sub["step_ms"] * 1e-3)
        ax2.plot(x, imgs_per_s, "k--o", ms=5, lw=1.5, label="Throughput (img/s)")
        ax2.set_ylabel("Throughput (images / second)", fontsize=9, color="k")
        ax2.legend(fontsize=8, loc="upper right")

    plt.suptitle("Training Step Breakdown: Forward / Backward / NVLink AllReduce / Optimizer\n"
                 "2× H100 DDP with NVLink 4 NV6",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = PLOTS_DIR / "scaling_step_breakdown.png"
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {out}")


if __name__ == "__main__":
    df = load_data()
    print(f"Loaded {len(df)} configs")
    plot_combined_roofline(df)
    plot_regime_zoom(df)
    plot_step_breakdown(df)
    print("All plots saved to", PLOTS_DIR)
