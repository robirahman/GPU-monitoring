#!/usr/bin/env python3
"""
week7/edge_cases/plot_roofline_edge_cases.py
=============================================
Roofline plot showing all 10 edge-case runs overlaid on the
Week 7 Tier-1 training sweep (B200 single-GPU reference).

Similar in spirit to week5/corner_cases/plots/roofline_vs_training.png
but for B200 edge cases.

Formula (matches scale_model_accuracy.py):
  achieved_tflops = flop_mult * fwd_flops * (batch / 64) / step_ms / 1e12

Where flop_mult:
  training  → 3  (fwd + bwd + bwd ≈ 3× fwd, same as tier-1)
  inference → 1  (fwd only)

Output:  week7/edge_cases/plots/roofline_all_runs.png
"""

import sys
from pathlib import Path

sys.path.insert(0, "/tmp/pynvml_pkg")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

WEEK7     = Path(__file__).parent.parent
EC_DIR    = Path(__file__).parent
PLOTS_DIR = EC_DIR / "plots"
PLOTS_DIR.mkdir(exist_ok=True)

TIER1_CSV = WEEK7 / "results" / "model_accuracy" / "model_accuracy_results.csv"
EC_CSV    = EC_DIR / "results" / "edge_cases_summary.csv"

# ── B200 hardware ceilings ─────────────────────────────────────────────────────
B200_FP16_TFLOPS = 4500.0   # TFLOPS
B200_FP32_TFLOPS =  140.0   # TFLOPS
B200_MEM_BW_GBPS = 8000.0   # GB/s

RIDGE_FP16 = B200_FP16_TFLOPS / (B200_MEM_BW_GBPS / 1e3)  # 562
RIDGE_FP32 = B200_FP32_TFLOPS / (B200_MEM_BW_GBPS / 1e3)  # 17.5

# ── Tier-1 model reference: n_ch → (ai, fwd_flops) at batch=64 ────────────────
TIER1_REF = {
    8:   ( 45.404958,   858_865_664),
    32:  (174.792470,  13_401_128_960),
    128: (594.665874, 213_055_176_704),
    256: (990.263626, 851_312_115_712),
}
REF_BATCH = 64

# N_STEPS + N_WARMUP used by run_edge_cases.py — for elapsed-based estimation
N_TOTAL = 80 + 5   # 85 iterations

# Per-case definition:
#   csv_label, display, n_ch, batch, flop_mult, mode, marker, color, label_offset
# flop_mult: 3=training (fwd+bwd), 1=inference (fwd only)
# label_offset: (dx_pts, dy_pts) for annotation text, or None for default
CASES = [
    # csv_label         display                n_ch  batch  mult  mode        mk   color      offset
    ("baseline_train", "Baseline\nTrain",       128,   64,   3, "train",    "o", "#37474F", ( 8, 10)),
    ("baseline_infer", "Baseline\nInfer",        128,   64,   1, "infer",    "s", "#78909C", ( 8,-18)),
    ("EC1_phantom",    "EC1\nPhantom",           128,   64,   1, "infer",    "^", "#E53935", (-52,  8)),
    ("EC2_silent",     "EC2\nSilent",            128,   64,   3, "train",    "v", "#FB8C00", ( 8,  5)),
    ("EC3_sparse",     "EC3\nSparse",            128,   64,   3, "train",    "<", "#FDD835", (-52, -8)),
    ("EC4_mining",     "EC4\nMining",            256,  512,   1, "infer",    ">", "#8E24AA", (  8,  5)),
    ("EC5_frozen",     "EC5\nFrozen",            128,   64,   1, "train",    "D", "#00ACC1", (-52,-18)),
    ("EC6_low_int",    "EC6\nLow-Int",             8,    4,   3, "train",    "P", "#43A047", (  8, -5)),
    ("EC7_amp_mask",   "EC7\nAMP",                32,   64,   3, "train",    "*", "#1E88E5", (  8,  5)),
    ("EC8_mem_idle",   "EC8\nMem Idle\n(no compute)", 128, 64, 0, "idle", "X", "#6D4C41", (  8,  5)),
]


def step_ms_for(row, n_ch, batch, flop_mult):
    """
    Return (step_ms, source) where source is 'measured' or 'estimated'.
    - measured:  mean_step_ms column is present and finite
    - estimated: fall back to elapsed_s / N_TOTAL  (baseline_infer has no per-step timing)
    - None:      idle case (EC8)
    """
    if flop_mult == 0:
        return None, "idle"
    ms = row.get("mean_step_ms", float("nan"))
    if pd.notna(ms) and ms > 0:
        return float(ms), "measured"
    elapsed = row.get("elapsed_s", float("nan"))
    if pd.notna(elapsed) and elapsed > 0:
        return float(elapsed) * 1000.0 / N_TOTAL, "estimated"
    return None, "missing"


def compute_point(n_ch, batch, flop_mult, ms):
    """Return (ai, achieved_tflops) using the same formula as scale_model_accuracy.py."""
    ai, fwd_flops_ref = TIER1_REF[n_ch]
    fwd_flops = fwd_flops_ref * (batch / REF_BATCH)
    tflops    = flop_mult * fwd_flops / (ms * 1e-3) / 1e12
    return ai, tflops


def plot_roofline():
    tier1 = pd.read_csv(TIER1_CSV)
    if "param_count" in tier1.columns and "n_params" not in tier1.columns:
        tier1 = tier1.rename(columns={"param_count": "n_params"})
    if "n_ch" not in tier1.columns:
        tier1["n_ch"] = [8, 16, 32, 64, 128, 256, 512][: len(tier1)]

    ec_df = pd.read_csv(EC_CSV).set_index("label")

    fig, ax = plt.subplots(figsize=(13, 8))

    # ── Roofline ceiling curves ───────────────────────────────────────────────
    ai_range  = np.logspace(0, 4, 1000)
    bw_line   = ai_range * (B200_MEM_BW_GBPS / 1e3)
    roof_fp16 = np.minimum(B200_FP16_TFLOPS, bw_line)
    roof_fp32 = np.minimum(B200_FP32_TFLOPS, bw_line)

    ax.loglog(ai_range, roof_fp16, color="#1565C0", lw=2.5,
              label=f"B200 FP16 ceiling  ({B200_FP16_TFLOPS:,.0f} TFLOPS)")
    ax.loglog(ai_range, roof_fp32, color="#C62828", lw=2.5, ls="--",
              label=f"B200 FP32 ceiling  ({B200_FP32_TFLOPS:,.0f} TFLOPS)")

    # Ridge lines
    ax.axvline(RIDGE_FP16, color="#1565C0", ls=":", lw=1.0, alpha=0.5)
    ax.axvline(RIDGE_FP32, color="#C62828",  ls=":", lw=1.0, alpha=0.5)
    ax.text(RIDGE_FP16 * 1.06, 0.15, f"FP16 ridge {RIDGE_FP16:.0f}",
            fontsize=7, color="#1565C0", va="bottom", rotation=90)
    ax.text(RIDGE_FP32 * 0.55, 0.15, f"FP32 ridge {RIDGE_FP32:.1f}",
            fontsize=7, color="#C62828",  va="bottom", rotation=90)

    # Regime shading
    ax.axvspan(1,           RIDGE_FP32, alpha=0.04, color="blue",  zorder=0)
    ax.axvspan(RIDGE_FP32,  RIDGE_FP16, alpha=0.04, color="green", zorder=0)
    ax.axvspan(RIDGE_FP16,  1e4,        alpha=0.04, color="red",   zorder=0)

    # ── Tier-1 training sweep (reference) ────────────────────────────────────
    t1_ai  = tier1["ai"].values
    t1_tf  = tier1["achieved_tflops"].values
    t1_nch = tier1["n_ch"].values

    ax.scatter(t1_ai, t1_tf, s=110, c="black", marker="o",
               zorder=8, alpha=0.30, label="Tier-1 training sweep (n_ch width)")
    ax.plot(t1_ai, t1_tf, "k-", lw=0.8, alpha=0.20, zorder=7)
    for ai_v, tf_v, nch in zip(t1_ai, t1_tf, t1_nch):
        ax.annotate(f"n={nch}", (ai_v, tf_v), fontsize=6, color="#888",
                    xytext=(3, 3), textcoords="offset points")

    # Training AI range band
    ax.axvspan(t1_ai.min(), t1_ai.max(), alpha=0.05, color="navy",
               label="Tier-1 training AI range")

    # ── Edge case points ──────────────────────────────────────────────────────
    legend_entries = []
    for (csv_label, display, n_ch, batch, flop_mult, mode,
         marker, color, offset) in CASES:
        if csv_label not in ec_df.index:
            print(f"  WARNING: {csv_label} not in CSV, skipping")
            continue
        row = ec_df.loc[csv_label]

        ms, src = step_ms_for(row, n_ch, batch, flop_mult)

        if src == "idle":
            # EC8: pure memory idle, no meaningful TFLOPS
            # Place at AI~1 (well below FP32 ridge) to signal memory-bound / no compute
            ai_v, tflops_v = 1.5, 0.008
        elif ms is None:
            print(f"  WARNING: no timing for {csv_label}, skipping")
            continue
        else:
            ai_v, tflops_v = compute_point(n_ch, batch, flop_mult, ms)
            if src == "estimated":
                print(f"  {csv_label}: step_ms estimated from elapsed_s ({ms:.2f} ms)")

        size   = 180 if mode in ("train",) else 130
        zorder = 14 if csv_label.startswith("baseline") else 12
        ax.scatter(ai_v, tflops_v, s=size, color=color, marker=marker,
                   edgecolors="white", linewidths=0.8, zorder=zorder, alpha=0.93)

        arrowprops = dict(arrowstyle="-", color=color, lw=0.7,
                          connectionstyle="arc3,rad=0.0")
        ax.annotate(
            display,
            xy=(ai_v, tflops_v),
            xytext=(offset[0], offset[1]),
            textcoords="offset points",
            fontsize=7,
            color=color,
            fontweight="bold" if csv_label.startswith("baseline") else "normal",
            arrowprops=arrowprops,
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.6),
        )
        legend_entries.append((display.replace("\n", " "), color, marker))

    # ── Legend ────────────────────────────────────────────────────────────────
    handles = [
        Line2D([0], [0], color="#1565C0", lw=2.5,
               label=f"B200 FP16 ceiling  ({B200_FP16_TFLOPS:,.0f} TFLOPS)"),
        Line2D([0], [0], color="#C62828",  lw=2.5, ls="--",
               label=f"B200 FP32 ceiling  ({B200_FP32_TFLOPS:,.0f} TFLOPS)"),
        Line2D([0], [0], marker="o", color="black", ls="none", ms=8, alpha=0.35,
               label="Tier-1 training sweep"),
    ]
    for disp, color, marker in legend_entries:
        handles.append(
            Line2D([0], [0], marker=marker, color=color, ls="none", ms=8,
                   markeredgecolor="white", markeredgewidth=0.5, label=disp)
        )
    ax.legend(handles=handles, fontsize=7.5, loc="upper left", ncol=2,
              framealpha=0.93, edgecolor="#ccc")

    # ── Axes and formatting ───────────────────────────────────────────────────
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP / byte)", fontsize=12)
    ax.set_ylabel("Achieved TFLOPS", fontsize=12)
    ax.set_title(
        "Week 7 Edge Cases — All Runs vs B200 Roofline\n"
        "(overlaid on Tier-1 training sweep; TFLOPS = 3 × fwd_flops / step_ms  for training, "
        "1 × for inference)",
        fontsize=11, fontweight="bold",
    )
    ax.set_xlim(1, 3000)
    ax.set_ylim(0.002, B200_FP16_TFLOPS * 1.8)
    ax.grid(True, which="both", alpha=0.20)

    # Hardware info box
    ax.text(
        0.985, 0.035,
        f"B200 HBM3e : {B200_MEM_BW_GBPS:,.0f} GB/s\n"
        f"B200 FP16  : {B200_FP16_TFLOPS:,.0f} TFLOPS\n"
        f"B200 FP32  : {B200_FP32_TFLOPS:,.0f} TFLOPS",
        transform=ax.transAxes, fontsize=8.5, va="bottom", ha="right",
        bbox=dict(boxstyle="round,pad=0.4", fc="lightyellow", ec="#ccc", alpha=0.9),
    )

    plt.tight_layout()
    out = PLOTS_DIR / "roofline_all_runs.png"
    plt.savefig(out, dpi=140, bbox_inches="tight")
    plt.close()
    print(f"Saved: {out}")


if __name__ == "__main__":
    plot_roofline()
