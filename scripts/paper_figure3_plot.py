"""Figure 3: classification accuracy vs window size, XGBoost, job-level holdout.

Plots docs/camera_ready_results_20261022/paper_figure3_job_id_xgb.json, which
sweeps 5/10/15/20/30/45/60/90/120 s windows through the same evaluate() path as
the headline number. Error bars are the standard deviation across the five CV
folds, not a confidence interval: whole jobs move between folds together, so a
single large job landing in one fold moves that fold's accuracy by more than any
window-size effect in this sweep. Showing the spread is the point -- the
accuracy differences between window sizes are smaller than the fold-to-fold
spread at most sizes, which is what makes the "latency is nearly free" claim
defensible.

Accuracy only: specificity is reported per window size in the source JSON and in
Table 2, and plotting it here made the figure argue two things at once.

Usage:
  python3 scripts/paper_figure3_plot.py \
      --json docs/camera_ready_results_20261022/paper_figure3_job_id_xgb.json \
      --out report/figures/accuracy_vs_window_xgb
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--out", required=True)
ap.add_argument("--title", default="Accuracy vs window size (XGBoost, job-level holdout)")
a = ap.parse_args()

J = json.loads(Path(a.json).read_text())
rows = sorted(J["rows"], key=lambda r: r["window_sec"])
w = np.array([r["window_sec"] for r in rows], float)
acc = np.array([r["accuracy"] for r in rows]) * 100
sd = np.array([r["std"] for r in rows]) * 100

fig, ax = plt.subplots(figsize=(7.0, 4.2))
ax.fill_between(w, acc - sd, acc + sd, color="#1f77b4", alpha=0.15,
                label="±1 sd across folds")
ax.plot(w, acc, marker="o", color="#1f77b4", lw=1.9, ms=6, label="Accuracy")

for x, yv in zip(w, acc):
    ax.annotate(f"{yv:.2f}", (x, yv), textcoords="offset points",
                xytext=(0, 9), ha="center", fontsize=7.5, color="#13527e")

ax.set_xscale("log")
ax.set_xticks(w)
ax.set_xticklabels([f"{int(v)}" for v in w])
ax.minorticks_off()
ax.set_xlabel("Window size (s)")
ax.set_ylabel("Accuracy (%)")
ax.set_ylim(min(acc - sd) - 0.4, min(max(acc + sd) + 0.4, 100.3))
ax.grid(alpha=0.3)
ax.set_title(a.title, fontsize=11)
ax.legend(loc="lower left", fontsize=8.5, framealpha=0.95)
fig.tight_layout()

out = Path(a.out)
out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".png"), dpi=150)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)

best = max(rows, key=lambda r: r["accuracy"])
ge15 = [r for r in rows if r["window_sec"] >= 15]
print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
print(f"  sizes          {[int(v) for v in w]}")
print(f"  accuracy       {min(acc):.2f}-{max(acc):.2f}%")
print(f"  >=15 s         {min(r['accuracy'] for r in ge15)*100:.2f}-"
      f"{max(r['accuracy'] for r in ge15)*100:.2f}%")
print(f"  best window    {best['window_sec']} s at {best['accuracy']*100:.2f}%")
print(f"  5 s            {rows[0]['accuracy']*100:.2f}% "
      f"(gap to best {best['accuracy']*100 - rows[0]['accuracy']*100:.2f} pp)")
