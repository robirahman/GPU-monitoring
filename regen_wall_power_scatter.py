"""Regenerate wall-power scatter (Appendix D, Fig A1) from
classifier/results/wall_power_correlation.csv.
Output: report/figures/wall_power_scatter.{png,pdf}
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "classifier" / "results"

df = pd.read_csv(RES / "wall_power_correlation.csv")
log.info(f"Loaded {len(df)} workloads")

def category(name: str) -> str:
    n = name.lower()
    if n == "idle":
        return "idle"
    if n.startswith("adversarial"):
        return "adversarial"
    if "mining" in n:
        return "mining"
    if "rendering" in n:
        return "rendering"
    if "cufft" in n or "nbody" in n:
        return "HPC"
    if "inference" in n:
        return "ML inference"
    return "ML training"

df["category"] = df["workload"].map(category)

# Pearson r and linear fit
r, p = pearsonr(df["gpu_power_mean_w"], df["wall_power_mean_w"])
slope, intercept = np.polyfit(df["gpu_power_mean_w"], df["wall_power_mean_w"], 1)
log.info(f"Pearson r={r:.4f}  R^2={r**2:.4f}  p={p:.2g}")
log.info(f"Linear fit: wall = {slope:.3f} * NVML + {intercept:.1f} W")

CAT_COLORS = {
    "idle":          "#888888",
    "ML training":   "#2ca02c",
    "ML inference":  "#1f77b4",
    "HPC":           "#9467bd",
    "mining":        "#8c564b",
    "rendering":     "#e377c2",
    "adversarial":   "#d62728",
}

fig, ax = plt.subplots(figsize=(8.0, 5.5))

# Fit line first so points render on top
x_line = np.linspace(0, df["gpu_power_mean_w"].max() * 1.08, 100)
y_line = slope * x_line + intercept
ax.plot(x_line, y_line, linestyle="--", color="black", alpha=0.7, linewidth=1.4,
        label=f"wall = {slope:.2f}·NVML + {intercept:.0f} W   ($R^2$ = {r**2:.2f})")

# y = x reference (perfect proxy line, no overhead)
ax.plot(x_line, x_line, linestyle=":", color="gray", alpha=0.5, linewidth=0.9,
        label="y = x  (no system overhead)")

# Scatter per category
for cat, color in CAT_COLORS.items():
    sub = df[df["category"] == cat]
    if len(sub) == 0:
        continue
    ax.scatter(sub["gpu_power_mean_w"], sub["wall_power_mean_w"],
               c=color, s=75, edgecolors="black", linewidths=0.5,
               label=f"{cat} (n={len(sub)})", zorder=3)

# Label notable points. Tuned offsets to avoid overlap.
notable = {
    "idle":                            ("idle",            (12, -4)),
    "cufft_benchmark":                 ("cuFFT",           (10,   6)),
    "nbody_sim":                       ("N-body",          (-40, -14)),
    "gpt2_wikitext2":                  ("GPT-2 train",     (10,  -4)),
    "adversarial_K_online_learning":   ("K_online\n(CPU burn)", (-80, -20)),
    "adversarial_L_diluted_5":         ("L_diluted_5",     (10,   0)),
    "adversarial_L_diluted_10":        ("L_diluted_10",    (10,   2)),
    "adversarial_L_diluted_20":        ("L_diluted_20",    (10,  -8)),
    "adversarial_L_diluted_2":         ("L_diluted_2",     (10,   2)),
    "mining_ethash_proxy":             ("mining",          (10, -14)),
    "resnet50_inference":              ("ResNet-50 infer", (10,   6)),
    "rendering_proxy":                 ("rendering",       (-70,  -8)),
    "bert_sst2":                       ("BERT train",      (-60,  12)),
    "pytorch_mlp_cifar10":             ("MLP",             (10,  -8)),
}
for wl, (label, (dx, dy)) in notable.items():
    row = df[df["workload"] == wl]
    if len(row) == 0:
        continue
    x, y = row["gpu_power_mean_w"].iloc[0], row["wall_power_mean_w"].iloc[0]
    ax.annotate(label, (x, y), xytext=(dx, dy), textcoords="offset points",
                fontsize=7, color="black",
                arrowprops=dict(arrowstyle="-", color="gray", alpha=0.5, lw=0.5))

ax.set_xlabel("NVML $power\\_draw\\_w$ (GPU only, mean over workload)")
ax.set_ylabel("Wall power (Tapo P115, mean over workload)")
ax.set_title(f"NVML vs. wall-plug power on RTX 5080\n"
             f"Pearson $r$ = {r:.3f},  $R^2$ = {r**2:.2f},  n = {len(df)} workloads",
             fontsize=11)
ax.set_xlim(0, df["gpu_power_mean_w"].max() * 1.15)
ax.set_ylim(0, df["wall_power_mean_w"].max() * 1.10)
ax.grid(alpha=0.3)
# Legend outside plot on the right
ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=8,
          framealpha=0.95, borderaxespad=0.0)

fig.tight_layout()
fig.savefig(FIG / "wall_power_scatter.png", dpi=150)
fig.savefig(FIG / "wall_power_scatter.pdf")
log.info(f"Saved wall_power_scatter.png/.pdf")
plt.close(fig)
