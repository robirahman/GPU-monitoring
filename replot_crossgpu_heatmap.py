"""Replot the cross-GPU transfer heatmap from the cached CSV without retraining.

Applies cleaner short names and the canonical GPU ordering.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42  # ICML: TrueType, avoid Type-3
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent
CSV = ROOT / "classifier" / "results" / "cross_gpu_transfer.csv"
FIG = ROOT / "report" / "figures"

_SHORT_NAME_OVERRIDES = {
    "A100-SXM4-40GB": "A100 40GB",
    "A100-SXM4-80GB": "A100 80GB",
    "RTX 4050 Laptop GPU": "RTX 4050 Laptop",
    "H100 80GB HBM3": "H100 80GB",
}

GPU_ORDER = [
    "RTX 4050 Laptop",
    "RTX 5080",
    "RTX 5090",
    "A100 40GB",
    "A100 80GB",
    "H100 80GB",
    "H100 NVL",
    "H200",
    "B200",
]


def short_name(g):
    s = str(g).replace("NVIDIA ", "").replace("GeForce ", "")
    return _SHORT_NAME_OVERRIDES.get(s, s)


def gpu_sort_key(g):
    s = short_name(g)
    if s in GPU_ORDER:
        return (0, GPU_ORDER.index(s))
    return (1, s)


df = pd.read_csv(CSV, index_col=0)

df.columns = [short_name(c) for c in df.columns]
new_index = []
for i in df.index:
    if str(i).startswith("Pooled"):
        new_index.append(i)
    else:
        new_index.append(short_name(i))
df.index = new_index

train_rows = [i for i in df.index if not i.startswith("Pooled") and i != "unknown"]
pooled_rows = [i for i in df.index if i.startswith("Pooled")]
row_order = sorted(train_rows, key=gpu_sort_key) + pooled_rows
# Match columns to train-row GPUs so the figure is square on named GPUs.
col_order = sorted(train_rows, key=gpu_sort_key)

df = df.loc[row_order, col_order]

acc_mat = df.values
row_labels = df.index.tolist()
col_labels = df.columns.tolist()

fig, ax = plt.subplots(figsize=(max(6.0, 0.9 * len(col_labels) + 2.5),
                                 max(4.0, 0.55 * len(row_labels) + 1.5)))
im = ax.imshow(acc_mat, cmap="RdYlGn", vmin=0.5, vmax=1.0, aspect="auto")
ax.set_xticks(range(len(col_labels)))
ax.set_yticks(range(len(row_labels)))
ax.set_xticklabels(col_labels, rotation=35, ha="right", fontsize=9)
ax.set_yticklabels(row_labels, fontsize=9)
ax.set_xlabel("Test GPU")
ax.set_ylabel("Train GPU")
ax.set_title("Cross-GPU transfer — three-way accuracy (XGBoost, 30s windows)")
for i in range(acc_mat.shape[0]):
    for j in range(acc_mat.shape[1]):
        v = acc_mat[i, j]
        if pd.isna(v):
            continue
        color = "white" if v < 0.75 else "black"
        ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                color=color, fontsize=8)
ax.axhline(len(row_labels) - 1.5, color="black", linewidth=1.2)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="accuracy")
fig.tight_layout()
fig.savefig(FIG / "cross_gpu_heatmap.png", dpi=150)
fig.savefig(FIG / "cross_gpu_heatmap.pdf")
print(f"Saved {FIG / 'cross_gpu_heatmap.png'}")
print(f"Saved {FIG / 'cross_gpu_heatmap.pdf'}")

df.to_csv(CSV)
print(f"Rewrote {CSV} with canonical labels and ordering")
print()
print("Final row order:", row_labels)
print("Final col order:", col_labels)
