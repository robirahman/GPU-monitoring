"""Figures for the run-level evaluation appendix.

Panel A/B: ROC and PR stratified by evasion family, window level. These are the
curves Reviewer M7wU asked for; whitebox_lora, B_low_util and the Round 5
white-box families are drawn in colour and everything else in grey, because the
point of the figure is that the hard families are hard, not that the mean is
high.

Panel C/D: run-level ROC and PR for the three aggregation rules on the same
axes. The gap between the window-level curve and the run-level curves is the
whole argument for aggregating, and it is invisible if only one of them is
plotted.

Usage:
  python3 scripts/paper_runlevel_plot.py \
      --json docs/camera_ready_results_20261022/paper_runlevel_xgb.json \
      --out report/figures/runlevel_roc_pr_xgb
"""
import argparse, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

HARD = ["symgemm_L4_fused_sgd", "whitebox_lora", "B_low_util",
        "A_util_modulation", "whitebox_full", "O_composite_idle_pad"]
COL = {"symgemm_L4_fused_sgd": "#d62728", "whitebox_lora": "#ff7f0e",
       "B_low_util": "#9467bd", "A_util_modulation": "#8c564b",
       "whitebox_full": "#2ca02c", "O_composite_idle_pad": "#1f77b4"}

ap = argparse.ArgumentParser()
ap.add_argument("--json", required=True)
ap.add_argument("--out", required=True)
a = ap.parse_args()
J = json.loads(Path(a.json).read_text())
cv, byf = J["curves"], {r["family"]: r for r in J["by_family"]}

fig, ax = plt.subplots(2, 2, figsize=(11.5, 9.0))

# ---- A: window ROC by family ----
axA = ax[0, 0]
for f, c in cv["by_family"].items():
    x, y = c["roc"]
    hard = f in HARD
    axA.plot(x, y, lw=1.9 if hard else 0.8,
             color=COL.get(f, "#bbbbbb"), zorder=3 if hard else 1,
             label=f"{f} ({byf[f]['window_roc_auc']:.3f})" if hard else None)
x, y = cv["window_overall"]["roc"]
axA.plot(x, y, lw=2.4, color="black", ls="--", zorder=4, label="all windows")
axA.set_xscale("log"); axA.set_xlim(1e-4, 1.0); axA.set_ylim(0.5, 1.005)
axA.set_xlabel("False positive rate (log)"); axA.set_ylabel("True positive rate")
axA.set_title("A. Window-level ROC by evasion family\n"
              "(log FPR: every curve saturates on a linear axis)", fontsize=10.5)
axA.legend(fontsize=7.2, loc="lower right", framealpha=0.95)
axA.grid(alpha=0.3)

# ---- B: window PR by family (log-x to show the low-recall tail) ----
axB = ax[0, 1]
for f, c in cv["by_family"].items():
    x, y = c["pr"]
    hard = f in HARD
    axB.plot(x, y, lw=1.9 if hard else 0.8,
             color=COL.get(f, "#bbbbbb"), zorder=3 if hard else 1,
             label=f"{f} (AP {byf[f]['window_ap']:.3f})" if hard else None)
x, y = cv["window_overall"]["pr"]
axB.plot(x, y, lw=2.4, color="black", ls="--", zorder=4, label="all windows")
axB.set_xlabel("Recall"); axB.set_ylabel("Precision")
axB.set_title("B. Window-level precision-recall by evasion family", fontsize=10.5)
axB.legend(fontsize=7.2, loc="lower left", framealpha=0.95)
axB.grid(alpha=0.3)

# ---- C/D: run-level, three aggregation rules ----
seed0 = J["per_seed"][0]
names = [("cusum", "CUSUM", "#d62728"), ("3of5", "3-of-5 @ τ=0.75", "#1f77b4"),
         ("maxp", "max P(train)", "#7f7f7f")]
axC = ax[1, 0]
for key, lab, c in names:
    x, y = cv["run_overall"][f"roc_{key}"]
    auc = seed0[f"run_auc_score_{key}"]
    axC.plot(x, y, lw=2.0, color=c, label=f"{lab} ({auc:.4f})")
x, y = cv["window_overall"]["roc"]
axC.plot(x, y, lw=1.4, color="black", ls="--",
         label=f"window level ({np.mean(J['window_level']['roc_auc']):.4f})")
axC.set_xlim(0, 0.10); axC.set_ylim(0.85, 1.002)
axC.set_xlabel("False positive rate (zoomed to 0-10%)")
axC.set_ylabel("True positive rate")
axC.set_title("C. Run-level ROC by aggregation rule\n"
              "(in-distribution; all rules are near-perfect here)", fontsize=10.5)
axC.legend(fontsize=8, loc="lower right", framealpha=0.95); axC.grid(alpha=0.3)

axD = ax[1, 1]
L = [r for r in J.get("lofo", []) if r["run_alert_hybrid"] < 0.999 or
     r["run_alert_3of5"] < 0.999]
L.sort(key=lambda r: r["run_alert_cusum"])
if L:
    yy = np.arange(len(L)); bh = 0.38
    axD.barh(yy + bh / 2, [r["run_alert_3of5"] * 100 for r in L], bh,
             color="#1f77b4", edgecolor="black", lw=0.3, label="3-of-5 @ τ=0.75")
    axD.barh(yy - bh / 2, [r["run_alert_cusum"] * 100 for r in L], bh,
             color="#d62728", edgecolor="black", lw=0.3, label="CUSUM (matched false alarms)")
    for i, r in enumerate(L):
        axD.text(r["run_alert_3of5"] * 100 + 1, i + bh / 2,
                 f"{r['run_alert_3of5']*100:.0f}%", va="center", fontsize=7.5)
        axD.text(r["run_alert_cusum"] * 100 + 1, i - bh / 2,
                 f"{r['run_alert_cusum']*100:.0f}%", va="center", fontsize=7.5,
                 fontweight="bold" if r["run_alert_cusum"] > r["run_alert_3of5"] else "normal")
    axD.set_yticks(yy); axD.set_yticklabels([r["family"] for r in L], fontsize=8.5)
    axD.set_xlim(0, 118); axD.set_xlabel("Run-level alert rate (%)")
    axD.set_title("D. Transfer to UNSEEN families: leave-one-family-out\n"
                  "(families below 100%; the other "
                  f"{len(J.get('lofo', [])) - len(L)} are at 100% for both)",
                  fontsize=10.5)
    axD.legend(fontsize=8, loc="lower right", framealpha=0.95)
    axD.grid(axis="x", alpha=0.3)
else:
    axD.axis("off")

fig.suptitle("Run-level evaluation: XGBoost, 30 s windows / 15 s stride, "
             "job-level holdout", fontsize=11.5)
fig.tight_layout(rect=[0, 0, 1, 0.975])
out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(out.with_suffix(".png"), dpi=150)
fig.savefig(out.with_suffix(".pdf"))
plt.close(fig)
print(f"wrote {out.with_suffix('.png')} and {out.with_suffix('.pdf')}")
for r in sorted(J["by_family"], key=lambda r: r["window_roc_auc"]):
    print(f"  {r['family']:24s} AUC {r['window_roc_auc']:.4f}  AP {r['window_ap']:.4f}  "
          f"TPR@1%FPR {r['window_tpr_at_1pct_fpr']:.3f}  "
          f"run 3of5 {('%.3f' % r['run_alert_3of5']) if r['run_alert_3of5'] is not None else '-':>6s}  "
          f"run CUSUM {('%.3f' % r['run_alert_cusum']) if r['run_alert_cusum'] is not None else '-':>6s}")
