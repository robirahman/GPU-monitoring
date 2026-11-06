"""Figure 4: leave-one-family-out detection rate per evasion family, XGBoost.

Plots the `unseen` column of the arms-race run -- for each evasion family, the
classifier is hardened on corpus_N minus that family and then asked to detect
it. This is the same number as Table 7's LOO mean column, so the figure and the
table cannot drift apart.

Error bars are min/max across the workload cells inside a family, which needs
the per-cell breakdown that `paper_table5_armsrace.py --unseen-only` records in
`unseen_by_cell`. The ordinary arms-race JSON stores the family mean only; if
this script is pointed at one, it draws the bars without error bars and says so
rather than inventing a spread.

Note this replaces report/figures/loo_detection_bar.png, which was built by
regen_figures_ext.py from classifier/results/leave_one_out_twostage.csv -- a
RandomForest run, at 60 s windows, on a smaller corpus (18.7k training windows
against today's 28.1k). Three things changed at once, so the old figure is not
comparable cell by cell.

Usage:
  python3 scripts/paper_figure4_plot.py \
      --json docs/camera_ready_results_20261022/paper_table5_armsrace_xgb_unseen.json \
      --out report/figures/loo_detection_bar_xgb
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt
from matplotlib.patches import Patch

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--title",
                default="Leave-one-family-out detection (XGBoost, 30 s windows,\n"
                        "hardened on every other family in the round)")
a = ap.parse_args()

J = json.loads(Path(a.json).read_text())
fams = [r for r in J["by_family"]
        if r.get("is_evasion") and r.get("unseen") is not None]
fams.sort(key=lambda r: r["unseen"])

name = [r["family"] for r in fams]
rate = np.array([r["unseen"] for r in fams]) * 100
nwin = [r["windows"] for r in fams]
ncell = [r["cells"] for r in fams]

have_cells = all(r.get("unseen_by_cell") for r in fams)
if have_cells:
    lo = np.array([min(r["unseen_by_cell"].values()) for r in fams]) * 100
    hi = np.array([max(r["unseen_by_cell"].values()) for r in fams]) * 100
    xerr = np.vstack([np.clip(rate - lo, 0, None), np.clip(hi - rate, 0, None)])
else:
    xerr = None
    hi = rate
    print("WARNING: no unseen_by_cell in this JSON -- drawing bars with no "
          "error bars. Re-run paper_table5_armsrace.py with --unseen-only to "
          "get the per-cell spread.")

DDP = {"K_ddp", "K_ddp_accum", "L_ddp", "L_ddp_stagger"}
colors = ["#1f77b4" if n in DDP else
          "#d62728" if r < 50 else "#ff7f0e" if r < 80 else "#2ca02c"
          for n, r in zip(name, rate)]

fig, ax = plt.subplots(figsize=(10.0, 8.0))
y = np.arange(len(fams))
ax.barh(y, rate, color=colors, edgecolor="black", linewidth=0.3, xerr=xerr,
        error_kw={"ecolor": "black", "capsize": 3, "elinewidth": 0.9,
                  "capthick": 0.9})
ax.set_yticks(y)
ax.set_yticklabels(name, fontsize=9)
ax.set_xlabel("Leave-one-family-out detection rate (%)")
ax.set_xlim(0, 132)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.axvline(50, color="gray", ls=":", lw=0.8)
ax.axvline(90, color="gray", ls=":", lw=0.8)
for i, (r, n, k, h) in enumerate(zip(rate, nwin, ncell, hi)):
    lab = f"{r:.1f}% (n={n:,}" + (f", k={k})" if k > 1 else ")")
    ax.text(h + 1.5, i, lab, va="center", fontsize=8)
ax.set_ylim(-0.6, len(fams) - 0.4)
ax.legend(handles=[Patch(facecolor="#1f77b4", label="Multi-GPU (DDP)"),
                   Patch(facecolor="#2ca02c", label="Single-GPU, ≥80%"),
                   Patch(facecolor="#ff7f0e", label="Single-GPU, 50-80%"),
                   Patch(facecolor="#d62728", label="Single-GPU, <50%")],
          loc="lower right", fontsize=9, framealpha=0.95)
ax.grid(axis="x", alpha=0.3)
ax.set_title(a.title, fontsize=11)
fig.tight_layout()

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".png"), dpi=150)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)

print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
print(f"  {len(fams)} evasion families, error bars: "
      f"{'min/max across cells' if have_cells else 'NONE'}")
for r in fams:
    extra = ""
    if r.get("unseen_by_cell"):
        v = [x * 100 for x in r["unseen_by_cell"].values()]
        extra = f"   cells {min(v):5.1f}-{max(v):5.1f}%  (k={len(v)})"
    print(f"  {r['family']:24s} {r['unseen']*100:5.1f}%  n={r['windows']:5,}{extra}")
