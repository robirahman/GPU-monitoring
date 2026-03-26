#!/venv/main/bin/python
"""
week5/corner_cases/plot_rooflines.py
======================================
Generate roofline plots for all corner cases.

Plots produced
--------------
1. roofline_all.png          — all cases on one roofline, colored by group
2. roofline_vs_training.png  — corner cases overlaid on week4 training sweep
3. roofline_cnn.png          — CC-A CNN inference breakdown
4. roofline_llm.png          — CC-B/C LLM prefill vs decode
5. roofline_quant.png        — CC-D quantisation comparison
6. roofline_vit.png          — CC-F ViT sweep
7. regime_scatter.png        — TFLOPS vs AI scatter with regime boundaries
"""

import sys, json
from pathlib import Path

sys.path.insert(0, "/tmp/pynvml_pkg")

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D

WEEK5   = Path(__file__).parent.parent
WEEK4   = WEEK5.parent / "week4"
CC_DIR  = Path(__file__).parent
RESULTS = CC_DIR / "results"
PLOTS   = CC_DIR / "plots"
PLOTS.mkdir(exist_ok=True)

# ── H100 NVL ceilings ─────────────────────────────────────────────────────────
FP16_CEIL  = 1979.0      # TFLOPS
FP32_CEIL  =   67.0      # TFLOPS
HBM_GBps   = 3900.0      # GB/s
RIDGE_FP32 = FP32_CEIL * 1e3 / HBM_GBps   # AI where FP32 BW = FP32 compute
RIDGE_FP16 = FP16_CEIL * 1e3 / HBM_GBps   # AI where FP16 BW = FP16 compute

# Week4 training AI/TFLOPS reference points (from scale_to_bottleneck results)
W4_TRAIN = [
    # (AI,  TFLOPS, n_ch)
    ( 45.4,   0.40,  8),
    ( 89.8,   1.86, 16),
    (174.8,   7.41, 32),
    (330.4,  31.74, 64),
    (594.7, 110.03,128),
    (990.3, 242.44,256),
    (1483.5,347.07,512),
]

GROUP_STYLE = {
    "CC-A": dict(color="#2196F3", marker="s", label="CC-A CNN Inference"),
    "CC-B": dict(color="#FF9800", marker="^", label="CC-B LLM Prefill"),
    "CC-C": dict(color="#9C27B0", marker="v", label="CC-C LLM Decode"),
    "CC-D": dict(color="#4CAF50", marker="D", label="CC-D Quantisation"),
    "CC-E": dict(color="#F44336", marker="P", label="CC-E Training Variant"),
    "CC-F": dict(color="#00BCD4", marker="*", label="CC-F ViT Inference"),
}

DTYPE_ALPHA = {"fp16": 0.9, "bf16": 0.8, "fp32": 0.6}


def roofline_ax(ax, ai_min=1, ai_max=5000, annotate=True):
    """Draw roofline ceilings and ridge lines on ax."""
    ai   = np.logspace(np.log10(ai_min), np.log10(ai_max), 500)

    # FP32 roofline
    roof_fp32 = np.minimum(FP32_CEIL, ai * HBM_GBps / 1e3)
    ax.plot(ai, roof_fp32, "k-",  lw=2.0, label="FP32 roofline")

    # FP16 roofline
    roof_fp16 = np.minimum(FP16_CEIL, ai * HBM_GBps / 1e3)
    ax.plot(ai, roof_fp16, "k--", lw=1.5, alpha=0.6, label="FP16 roofline")

    # Ridge lines
    ax.axvline(RIDGE_FP32, color="steelblue", ls=":", lw=1.2, alpha=0.7)
    ax.axvline(RIDGE_FP16, color="darkorange", ls=":", lw=1.2, alpha=0.7)

    # Regime shading
    ax.axvspan(ai_min, RIDGE_FP32, alpha=0.04, color="blue",   label="Memory-bound")
    ax.axvspan(RIDGE_FP32, RIDGE_FP16, alpha=0.04, color="green", label="Compute (FP32)")
    ax.axvspan(RIDGE_FP16, ai_max,    alpha=0.04, color="red",  label="Compute (FP16)")

    if annotate:
        ax.text(RIDGE_FP32 * 1.08, FP32_CEIL * 0.12, f"FP32 ridge\n{RIDGE_FP32:.0f}",
                fontsize=7, color="steelblue", va="bottom")
        ax.text(RIDGE_FP16 * 1.08, FP16_CEIL * 0.04, f"FP16 ridge\n{RIDGE_FP16:.0f}",
                fontsize=7, color="darkorange", va="bottom")
        ax.text(ai_max * 0.6, FP16_CEIL * 1.02, "FP16 Tensor Core",
                fontsize=8, color="gray")
        ax.text(ai_max * 0.6, FP32_CEIL * 1.05, "FP32 CUDA",
                fontsize=8, color="gray")

    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("Arithmetic Intensity (FLOP/byte)", fontsize=11)
    ax.set_ylabel("Achieved TFLOPS", fontsize=11)
    ax.grid(True, which="both", alpha=0.2)
    ax.set_xlim(ai_min, ai_max)
    ax.set_ylim(0.001, FP16_CEIL * 2)


def add_week4_training(ax, alpha=0.35):
    """Overlay week4 DDP training reference points."""
    ai_vals = [p[0] for p in W4_TRAIN]
    tf_vals = [p[1] for p in W4_TRAIN]
    ax.scatter(ai_vals, tf_vals, s=120, c="black", marker="o",
               zorder=10, alpha=alpha, label="Week4 DDP training")
    ax.plot(ai_vals, tf_vals, "k-", lw=1.0, alpha=alpha * 0.6)


def load_results():
    p = RESULTS / "corner_cases_results.json"
    if not p.exists():
        raise FileNotFoundError(p)
    with open(p) as f:
        return json.load(f)


# ── Plot 1: All cases on one roofline ─────────────────────────────────────────

def plot_all(records):
    fig, ax = plt.subplots(figsize=(13, 8))
    roofline_ax(ax)
    add_week4_training(ax)

    for r in records:
        g = r.get("group", "CC-?")
        s = GROUP_STYLE.get(g, dict(color="gray", marker="o", label=g))
        alpha = DTYPE_ALPHA.get(r.get("dtype", "fp16"), 0.7)
        ms = 60 if r.get("workload", "").startswith("training") else 40
        ax.scatter(r["ai"], r["achieved_tflops"],
                   c=s["color"], marker=s["marker"], s=ms,
                   alpha=alpha, edgecolors="none", zorder=5)

    # Legend
    handles = [Line2D([0], [0], color="k", lw=2, label="FP32 roofline"),
               Line2D([0], [0], color="k", lw=1.5, ls="--", alpha=0.6, label="FP16 roofline"),
               Line2D([0], [0], marker="o", color="black", ls="none", ms=10, alpha=0.4,
                      label="Week4 DDP training")]
    for g, s in GROUP_STYLE.items():
        handles.append(Line2D([0], [0], marker=s["marker"], color=s["color"],
                               ls="none", ms=8, label=s["label"]))
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Corner Cases — All Groups vs H100 NVL Roofline", fontsize=13)
    plt.tight_layout()
    p = PLOTS / "roofline_all.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p.name}")


# ── Plot 2: Roofline vs Training (highlight overlap) ──────────────────────────

def plot_vs_training(records):
    fig, ax = plt.subplots(figsize=(12, 7))
    roofline_ax(ax)

    # Week4 training band
    ai_tr = [p[0] for p in W4_TRAIN]
    ax.axvspan(min(ai_tr), max(ai_tr), alpha=0.08, color="red",
               label="Week4 training AI range")
    add_week4_training(ax, alpha=0.5)

    for r in records:
        g = r.get("group", "CC-?")
        s = GROUP_STYLE.get(g, dict(color="gray", marker="o"))
        # Highlight cases within training AI range
        in_range = min(ai_tr) <= r["ai"] <= max(ai_tr)
        edge = "black" if in_range else "none"
        lw   = 1.5 if in_range else 0
        ax.scatter(r["ai"], r["achieved_tflops"],
                   c=s["color"], marker=s["marker"],
                   s=80 if in_range else 30,
                   alpha=0.9 if in_range else 0.3,
                   edgecolors=edge, linewidths=lw, zorder=5 if in_range else 3)

    handles = [mpatches.Patch(color="red", alpha=0.15, label="Training AI range")]
    handles += [Line2D([0], [0], marker=s["marker"], color=s["color"],
                       ls="none", ms=8, label=s["label"])
                for g, s in GROUP_STYLE.items()]
    ax.legend(handles=handles, fontsize=8, loc="upper left")
    ax.set_title("Corner Cases Overlaid on Week4 Training Range\n"
                 "(Bold border = AI falls within training range)", fontsize=12)
    plt.tight_layout()
    p = PLOTS / "roofline_vs_training.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p.name}")


# ── Plot 3: CNN inference sweep ───────────────────────────────────────────────

def plot_cnn(records):
    cnn = [r for r in records if r.get("group") == "CC-A"]
    if not cnn: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: roofline colored by batch size
    ax = axes[0]
    roofline_ax(ax)
    add_week4_training(ax, alpha=0.3)
    batches = sorted(set(r["batch"] for r in cnn))
    cmap = plt.cm.plasma
    for r in cnn:
        idx   = batches.index(r["batch"]) / max(len(batches) - 1, 1)
        color = cmap(idx)
        m     = "^" if r.get("dtype") == "fp16" else "s"
        ax.scatter(r["ai"], r["achieved_tflops"], c=[color], marker=m,
                   s=50, alpha=0.8, edgecolors="none")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(batches[0], batches[-1]))
    plt.colorbar(sm, ax=ax, label="Batch size")
    ax.set_title("CC-A CNN Inference — colored by batch size", fontsize=10)

    # Right: TFLOPS vs batch for each arch
    ax2 = axes[1]
    archs = sorted(set(r.get("arch", "") for r in cnn))
    colors_a = plt.cm.tab10(np.linspace(0, 0.5, len(archs)))
    for i, arch in enumerate(archs):
        sub = sorted([r for r in cnn if r.get("arch") == arch and r.get("dtype") == "fp16"],
                     key=lambda x: x["batch"])
        if not sub: continue
        bs = [r["batch"] for r in sub]
        tf = [r["achieved_tflops"] for r in sub]
        ax2.plot(bs, tf, "o-", color=colors_a[i], label=arch, lw=2)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.axhline(67, color="steelblue", ls=":", lw=1.2, label="FP32 ceiling 67T")
    ax2.axhline(1979, color="darkorange", ls=":", lw=1.2, label="FP16 ceiling 1979T")
    ax2.set_xlabel("Batch size"); ax2.set_ylabel("TFLOPS")
    ax2.set_title("CC-A: TFLOPS vs Batch Size (FP16)", fontsize=10)
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.suptitle("CNN Inference Sweep — Roofline Analysis", fontsize=13)
    plt.tight_layout()
    p = PLOTS / "roofline_cnn.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p.name}")


# ── Plot 4: LLM prefill vs decode ─────────────────────────────────────────────

def plot_llm(records):
    prefill = [r for r in records if r.get("group") == "CC-B"]
    decode  = [r for r in records if r.get("group") == "CC-C"]
    if not prefill and not decode: return

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    roofline_ax(ax)
    add_week4_training(ax, alpha=0.3)
    models = sorted(set(r.get("arch", "") for r in prefill + decode))
    cmap = plt.cm.Set1
    for i, m_name in enumerate(models):
        col = cmap(i / len(models))
        pr  = [r for r in prefill if r.get("arch") == m_name]
        dc  = [r for r in decode  if r.get("arch") == m_name]
        if pr:
            ax.scatter([r["ai"] for r in pr], [r["achieved_tflops"] for r in pr],
                       c=[col], marker="^", s=60, alpha=0.8, label=f"{m_name} prefill")
        if dc:
            ax.scatter([r["ai"] for r in dc], [r["achieved_tflops"] for r in dc],
                       c=[col], marker="v", s=60, alpha=0.5, label=f"{m_name} decode")
    ax.legend(fontsize=7, loc="upper left"); ax.set_title("LLM Prefill (▲) vs Decode (▼)")

    ax2 = axes[1]
    seq_lens = sorted(set(r.get("seq_len", 0) for r in prefill if r.get("seq_len")))
    cmap2 = plt.cm.viridis
    for r in prefill:
        if "seq_len" not in r: continue
        idx   = seq_lens.index(r["seq_len"]) / max(len(seq_lens) - 1, 1)
        color = cmap2(idx)
        ax2.scatter(r["batch"], r["achieved_tflops"], c=[color], s=60,
                    alpha=0.8, edgecolors="none")
    sm = plt.cm.ScalarMappable(cmap=cmap2,
                                norm=plt.Normalize(seq_lens[0], seq_lens[-1]))
    plt.colorbar(sm, ax=ax2, label="Sequence length")
    ax2.axhline(67, color="steelblue", ls=":", lw=1.2)
    ax2.axhline(1979, color="darkorange", ls=":", lw=1.2)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Batch size"); ax2.set_ylabel("TFLOPS")
    ax2.set_title("LLM Prefill TFLOPS vs Batch (colored by seq_len)")
    ax2.grid(True, alpha=0.3)
    plt.suptitle("LLM Inference — Prefill vs Decode Roofline", fontsize=13)
    plt.tight_layout()
    p = PLOTS / "roofline_llm.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p.name}")


# ── Plot 5: Quantisation comparison ──────────────────────────────────────────

def plot_quant(records):
    quant = [r for r in records if r.get("group") == "CC-D"]
    if not quant: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    roofline_ax(ax)
    add_week4_training(ax, alpha=0.3)
    d_colors = {"fp32": "#F44336", "fp16": "#2196F3", "bf16": "#4CAF50"}
    for r in quant:
        c = d_colors.get(r.get("dtype", "fp16"), "gray")
        ax.scatter(r["ai"], r["achieved_tflops"], c=c, s=60, alpha=0.8, edgecolors="none")
    handles = [mpatches.Patch(color=c, label=d) for d, c in d_colors.items()]
    ax.legend(handles=handles, title="Precision", fontsize=9)
    ax.set_title("CC-D Quantisation — Roofline by precision")

    ax2 = axes[1]
    archs = sorted(set(r.get("arch", "") for r in quant))
    x = np.arange(len(archs))
    w = 0.25
    for i, (dtype, c) in enumerate(d_colors.items()):
        vals = []
        for arch in archs:
            sub = [r["achieved_tflops"] for r in quant
                   if r.get("arch") == arch and r.get("dtype") == dtype]
            vals.append(np.mean(sub) if sub else 0)
        ax2.bar(x + i * w, vals, w, label=dtype, color=c, alpha=0.8)
    ax2.set_xticks(x + w)
    ax2.set_xticklabels(archs, rotation=20)
    ax2.set_ylabel("Mean TFLOPS"); ax2.set_yscale("log")
    ax2.legend(title="Precision", fontsize=9); ax2.grid(True, axis="y", alpha=0.3)
    ax2.set_title("Mean TFLOPS by Arch and Precision")
    plt.suptitle("CC-D: Quantisation Effect on Roofline Position", fontsize=13)
    plt.tight_layout()
    p = PLOTS / "roofline_quant.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p.name}")


# ── Plot 6: ViT sweep ─────────────────────────────────────────────────────────

def plot_vit(records):
    vit = [r for r in records if r.get("group") == "CC-F"]
    if not vit: return
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    ax = axes[0]
    roofline_ax(ax)
    add_week4_training(ax, alpha=0.3)
    vit_models = sorted(set(r.get("arch", "") for r in vit))
    cmap = plt.cm.tab10
    for i, vm in enumerate(vit_models):
        sub = [r for r in vit if r.get("arch") == vm]
        ax.scatter([r["ai"] for r in sub], [r["achieved_tflops"] for r in sub],
                   c=[cmap(i / len(vit_models))], s=60, alpha=0.8, label=vm)
    ax.legend(fontsize=8, title="ViT variant")
    ax.set_title("CC-F ViT Inference — Roofline by model")

    ax2 = axes[1]
    for i, vm in enumerate(vit_models):
        sub = sorted([r for r in vit if r.get("arch") == vm], key=lambda x: x["batch"])
        if not sub: continue
        ax2.plot([r["batch"] for r in sub], [r["achieved_tflops"] for r in sub],
                 "o-", color=cmap(i / len(vit_models)), label=vm, lw=2)
    ax2.axhline(67, color="steelblue", ls=":", lw=1.2)
    ax2.axhline(1979, color="darkorange", ls=":", lw=1.2)
    ax2.set_xscale("log"); ax2.set_yscale("log")
    ax2.set_xlabel("Batch size"); ax2.set_ylabel("TFLOPS")
    ax2.set_title("ViT TFLOPS vs Batch size")
    ax2.legend(fontsize=8); ax2.grid(True, alpha=0.3)
    plt.suptitle("CC-F: Vision Transformer Roofline", fontsize=13)
    plt.tight_layout()
    p = PLOTS / "roofline_vit.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p.name}")


# ── Plot 7: Regime scatter ─────────────────────────────────────────────────────

def plot_regime_scatter(records):
    fig, ax = plt.subplots(figsize=(12, 7))
    roofline_ax(ax, annotate=True)
    add_week4_training(ax, alpha=0.5)

    for r in records:
        g = r.get("group", "CC-?")
        s = GROUP_STYLE.get(g, dict(color="gray", marker="o"))
        # Size by batch
        bs = r.get("batch", 1)
        size = max(20, min(200, 10 * math.log2(bs + 1)))
        ax.scatter(r["ai"], r["achieved_tflops"],
                   c=s["color"], marker=s["marker"],
                   s=size, alpha=0.75, edgecolors="none", zorder=5)

    handles = [Line2D([0], [0], marker=s["marker"], color=s["color"],
                      ls="none", ms=9, label=s["label"])
               for g, s in GROUP_STYLE.items()]
    handles += [Line2D([0], [0], marker="o", color="black", ls="none", ms=10,
                       alpha=0.5, label="Week4 DDP training")]
    ax.legend(handles=handles, fontsize=8, loc="upper left", ncol=2)
    ax.set_title("Regime Scatter — all corner cases vs H100 NVL Roofline\n"
                 "(point size ∝ log batch size)", fontsize=12)
    plt.tight_layout()
    p = PLOTS / "regime_scatter.png"
    plt.savefig(p, dpi=130, bbox_inches="tight"); plt.close()
    print(f"Saved: {p.name}")


# ── Main ───────────────────────────────────────────────────────────────────────

import math

def main():
    records = load_results()
    print(f"Loaded {len(records)} records")

    plot_all(records)
    plot_vs_training(records)
    plot_cnn(records)
    plot_llm(records)
    plot_quant(records)
    plot_vit(records)
    plot_regime_scatter(records)
    print(f"\nAll plots saved to {PLOTS}/")


if __name__ == "__main__":
    main()
