"""Plot the binary-RF cross-GPU transfer heatmap from the cached CSV.

Reads classifier/results/cross_gpu_transfer_binary_rf.csv (produced by
regen_crossgpu_heatmap_binary_rf.py) and writes
report/figures/cross_gpu_heatmap_binary_rf.{png,pdf}.

Cheap to run — pure plot, no training. Edit freely.
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
CSV = ROOT / "classifier" / "results" / "cross_gpu_transfer_binary_rf.csv"

df = pd.read_csv(CSV, index_col=0)
log.info(f"Loaded {df.shape[0]}x{df.shape[1]} matrix from {CSV.name}")

row_labels = list(df.index)
col_labels = list(df.columns)
acc_mat = df.values

fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(col_labels) + 2.5),
                                max(4.0, 0.55 * len(row_labels) + 1.5)))
im = ax.imshow(acc_mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(col_labels)))
ax.set_yticks(range(len(row_labels)))
ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_xlabel("Test GPU")
ax.set_ylabel("Train GPU")
for i in range(acc_mat.shape[0]):
    for j in range(acc_mat.shape[1]):
        v = acc_mat[i, j]
        if np.isnan(v):
            continue
        color = "white" if v < 0.75 else "black"
        ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                color=color, fontsize=8)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="accuracy")
fig.tight_layout()

png = FIG / "cross_gpu_heatmap_binary_rf.png"
pdf = FIG / "cross_gpu_heatmap_binary_rf.pdf"
fig.savefig(png, dpi=150)
fig.savefig(pdf)
log.info(f"Saved {png.name} / {pdf.name}")
