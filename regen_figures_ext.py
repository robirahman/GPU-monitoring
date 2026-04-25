"""Regenerate Figs 2 (dilution curve) and 4 (LOO bar) from result CSVs.
Outputs to report/figures/.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
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
RES = ROOT / "classifier" / "results"

# ── Figure 4: LOO detection bar chart ─────────────────────────────────────
loo = pd.read_csv(RES / "leave_one_out_twostage.csv")
# Aggregate per family: unweighted mean across variants (matches Table 7).
# Each variant counts equally regardless of how many windows it contributes.
agg = (loo.groupby("held_out_family")
          .agg(detection_rate=("detection_rate", "mean"),
               n_windows=("n_windows", "sum"),
               min_rate=("detection_rate", "min"),
               max_rate=("detection_rate", "max"),
               n_variants=("detection_rate", "count")))
# Sort ascending so highest rate ends up at top of horizontal bar chart
agg = agg.sort_values("detection_rate")

DDP_FAMS = {"K_ddp", "K_ddp_accum", "L_ddp", "L_ddp_stagger"}
agg["is_ddp"] = agg.index.to_series().isin(DDP_FAMS)
colors = ["#1f77b4" if is_ddp else "#d62728" if rate < 0.5
          else "#ff7f0e" if rate < 0.8 else "#2ca02c"
          for rate, is_ddp in zip(agg["detection_rate"], agg["is_ddp"])]

# Asymmetric error bars: [rate - min, max - rate] (in percent).
# Clip tiny negative drift from windows-weighted mean falling just outside
# [min, max] due to FP precision; matplotlib rejects negative xerr.
xerr = np.vstack([
    np.clip((agg["detection_rate"] - agg["min_rate"]).values, 0, None),
    np.clip((agg["max_rate"] - agg["detection_rate"]).values, 0, None),
]) * 100

fig, ax = plt.subplots(figsize=(10.0, 8.0))
y = np.arange(len(agg))
ax.barh(y, agg["detection_rate"] * 100, color=colors,
        edgecolor="black", linewidth=0.3,
        xerr=xerr,
        error_kw={"ecolor": "black", "capsize": 3,
                  "elinewidth": 0.9, "capthick": 0.9})
ax.set_yticks(y)
ax.set_yticklabels(agg.index, fontsize=9)
ax.set_xlabel("LOO detection rate (%)")
ax.set_xlim(0, 132)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.axvline(50, color="gray", linestyle=":", linewidth=0.8)
ax.axvline(90, color="gray", linestyle=":", linewidth=0.8)
for i, (rate, n, mx, nv) in enumerate(zip(
        agg["detection_rate"], agg["n_windows"],
        agg["max_rate"], agg["n_variants"])):
    # Place label just past the upper error-bar cap
    x_text = mx * 100 + 1.5
    label = f"{rate*100:.1f}% (n={n}"
    label += f", k={nv})" if nv > 1 else ")"
    ax.text(x_text, i, label, va="center", fontsize=8)
ax.set_ylim(-0.6, len(agg) - 0.4)
from matplotlib.patches import Patch
legend_elems = [
    Patch(facecolor="#1f77b4", label="Multi-GPU (DDP)"),
    Patch(facecolor="#2ca02c", label="Single-GPU, ≥80%"),
    Patch(facecolor="#ff7f0e", label="Single-GPU, 50-80%"),
    Patch(facecolor="#d62728", label="Single-GPU, <50%"),
]
ax.legend(handles=legend_elems, loc="lower right", fontsize=9,
          framealpha=0.95)
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG / "loo_detection_bar.png", dpi=150)
fig.savefig(FIG / "loo_detection_bar.pdf")
log.info(f"Saved loo_detection_bar.png/.pdf — {len(agg)} families")
plt.close(fig)

log.info("LOO per-family detection rates:")
for fam, row in agg.iterrows():
    log.info(f"  {fam:25s}  {row['detection_rate']*100:5.1f}%  n={int(row['n_windows'])}")

# ── Figure 2: Dilution curve ──────────────────────────────────────────────
dil = pd.read_csv(RES / "dilution_sweep_eval.csv")
# Keep 5s windows, drop inference-only row, keep train_fraction>0
sub = dil[(dil["window_sec"] == 5) & (dil["dilution_N"] != np.inf)].copy()
sub["dilution_N"] = sub["dilution_N"].astype(int)

fig, ax = plt.subplots(figsize=(6.5, 4.0))
styles = {
    "baseline":  ("#888888", "o", "Baseline"),
    "LOO":       ("#d62728", "s", "LOO (unseen dilution)"),
    "hardened":  ("#2ca02c", "^", "Hardened (class-balanced)"),
}
for cond, (color, marker, label) in styles.items():
    s = sub[sub["condition"] == cond].sort_values("dilution_N")
    ax.plot(s["dilution_N"], s["detection_rate"] * 100,
            marker=marker, color=color, label=label, linewidth=1.8, markersize=7)

ax.set_xscale("log")
ax.set_xticks([1, 2, 3, 5, 10, 20])
ax.set_xticklabels([1, 2, 3, 5, 10, 20])
ax.set_xlabel("Dilution N (cost multiplier ≈ N)")
ax.set_ylabel("Detection rate (%)")
ax.set_ylim(0, 105)
ax.axhline(50, color="gray", linestyle=":", linewidth=0.8, alpha=0.7)
ax.set_title("Dilution curve — composite idle padding\n(ResNet-50, RTX 5080, 5s windows)")
ax.grid(alpha=0.3)
ax.legend(loc="lower left", fontsize=9)
# Secondary axis: throughput fraction
ax2 = ax.twiny()
ax2.set_xscale("log")
ax2.set_xlim(ax.get_xlim())
ax2.set_xticks([1, 2, 3, 5, 10, 20])
ax2.set_xticklabels(["100%", "50%", "33%", "20%", "10%", "5%"])
ax2.set_xlabel("Training throughput fraction (1/N)")
fig.tight_layout()
fig.savefig(FIG / "dilution_curve.png", dpi=150)
fig.savefig(FIG / "dilution_curve.pdf")
log.info(f"Saved dilution_curve.png/.pdf")
plt.close(fig)

log.info(f"All done. Figures in {FIG}")
