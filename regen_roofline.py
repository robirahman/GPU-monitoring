"""Regenerate Appendix F roofline figure from raw corner-case data.

Source: week5/corner_cases/results/corner_cases_results.json (286
H100-NVL measurements across 6 workload families: CNN inference, LLM
prefill, LLM decode, quantization variants, training-variant reference,
ViT inference).

Output: report/figures/roofline_corner_cases.{png,pdf}

Shows: (1) H100-NVL FP32/FP16 rooflines with ridge lines; (2) each
corner-case configuration plotted as arithmetic intensity vs achieved
TFLOPS, coloured by workload family; (3) training reference band from
week4 dense-training sweeps overlaid to highlight the overlap region.
"""
import json
import logging
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # ICML: TrueType, avoid Type-3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
SRC = ROOT / "week5" / "corner_cases" / "results" / "corner_cases_results.json"

# H100 NVL roofline ceilings (dense tensor-core peak TFLOPS, HBM GB/s)
FP16_CEIL = 1979.0
FP32_CEIL = 67.0
HBM_GBps = 3900.0
RIDGE_FP32 = FP32_CEIL * 1e3 / HBM_GBps
RIDGE_FP16 = FP16_CEIL * 1e3 / HBM_GBps

# Week-4 dense-training reference points (AI, TFLOPS) — training band anchor
TRAIN_REF = [
    (45.4, 0.40), (89.8, 1.86), (174.8, 7.41), (330.4, 31.74),
    (594.7, 110.03), (990.3, 242.44), (1483.5, 347.07),
]

GROUP_STYLE = {
    "CC-A": dict(color="#2196F3", marker="s", label="CNN inference"),
    "CC-B": dict(color="#FF9800", marker="^", label="LLM prefill"),
    "CC-C": dict(color="#9C27B0", marker="v", label="LLM decode"),
    "CC-D": dict(color="#4CAF50", marker="D", label="Quantisation variants"),
    "CC-E": dict(color="#F44336", marker="P", label="Forward-only training variant"),
    "CC-F": dict(color="#00BCD4", marker="*", label="ViT inference"),
}

with open(SRC) as f:
    data = json.load(f)
log.info(f"Loaded {len(data)} corner-case runs from {SRC.name}")

fig, ax = plt.subplots(figsize=(11.0, 5.5))

# --- rooflines + ridges ---
ai_axis = np.logspace(0, 3.8, 500)  # 1 to ~6300
roof_fp32 = np.minimum(FP32_CEIL, ai_axis * HBM_GBps / 1e3)
roof_fp16 = np.minimum(FP16_CEIL, ai_axis * HBM_GBps / 1e3)
ax.plot(ai_axis, roof_fp32, "k-", lw=2.0, label="FP32 roofline (67 TFLOPS)")
ax.plot(ai_axis, roof_fp16, "k--", lw=1.5, alpha=0.6, label="FP16 roofline (1979 TFLOPS)")

# Regime shading
ax.axvspan(1, RIDGE_FP32, alpha=0.06, color="blue")
ax.axvspan(RIDGE_FP32, RIDGE_FP16, alpha=0.06, color="green")
ax.axvspan(RIDGE_FP16, ai_axis.max(), alpha=0.06, color="red")

# Ridge annotations
ax.axvline(RIDGE_FP32, color="steelblue", ls=":", lw=1.0, alpha=0.6)
ax.axvline(RIDGE_FP16, color="darkorange", ls=":", lw=1.0, alpha=0.6)
ax.text(RIDGE_FP32 * 1.1, 0.08, f"FP32 ridge\n{RIDGE_FP32:.0f}", fontsize=7,
        color="steelblue", va="bottom")
ax.text(RIDGE_FP16 * 1.1, 0.08, f"FP16 ridge\n{RIDGE_FP16:.0f}", fontsize=7,
        color="darkorange", va="bottom")

# --- training reference band ---
train_ai = [p[0] for p in TRAIN_REF]
train_tf = [p[1] for p in TRAIN_REF]
ax.fill_betweenx(
    np.array(train_tf),
    np.array(train_ai) * 0.6, np.array(train_ai) * 1.6,
    color="#999999", alpha=0.25, label="Training reference band (dense FP32)",
)
ax.plot(train_ai, train_tf, color="black", lw=0.8, linestyle="-", alpha=0.5)
ax.scatter(train_ai, train_tf, c="black", s=18, marker="x", zorder=5,
           label="Dense FP32 training sweep")

# --- corner-case points, coloured by group ---
for group, style in GROUP_STYLE.items():
    pts = [p for p in data if p.get("group") == group]
    if not pts:
        continue
    ai_vals = [p["ai"] for p in pts]
    tf_vals = [p["achieved_tflops"] for p in pts]
    ax.scatter(ai_vals, tf_vals,
               c=style["color"], marker=style["marker"], s=35,
               edgecolors="black", linewidths=0.3, alpha=0.75,
               label=f"{style['label']} (n={len(pts)})", zorder=3)

ax.set_xscale("log")
ax.set_yscale("log")
ax.set_xlim(1, ai_axis.max())
ax.set_ylim(0.05, FP16_CEIL * 2)
ax.set_xlabel("Arithmetic intensity (FLOP / byte)")
ax.set_ylabel("Achieved throughput (TFLOPS)")
ax.grid(which="both", alpha=0.2)
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, framealpha=0.95)

fig.tight_layout()
fig.savefig(FIG / "roofline_corner_cases.png", dpi=150)
fig.savefig(FIG / "roofline_corner_cases.pdf")
log.info(f"Saved roofline_corner_cases.png/.pdf")
plt.close(fig)
