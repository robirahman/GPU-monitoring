"""Plot binary (training vs non-training) accuracy vs window size.

Reads classifier/results/binary_results.csv (produced by retrain_binary.py)
and writes report/figures/accuracy_vs_window_binary.{png,pdf}.

Cheap to run — pure plot, no training. Edit freely.
"""
import logging
from pathlib import Path
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
CSV = ROOT / "classifier" / "results" / "binary_results.csv"

res = pd.read_csv(CSV)
log.info(f"Loaded {len(res)} rows from {CSV.name}")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
for clf_name in res["classifier"].unique():
    sub = res[res["classifier"] == clf_name].sort_values("window_sec")
    ax.errorbar(sub["window_sec"], sub["acc_mean"], yerr=sub["acc_std"],
                marker="o", capsize=3, label=clf_name, linewidth=1.5)
ax.set_xlabel("Window size (s)")
ax.set_ylabel("Accuracy (5-fold grouped CV)")
ax.set_ylim(0.95, 1.00)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()

png = FIG / "accuracy_vs_window_binary.png"
pdf = FIG / "accuracy_vs_window_binary.pdf"
fig.savefig(png, dpi=150)
fig.savefig(pdf)
log.info(f"Saved {png.name} / {pdf.name}")
