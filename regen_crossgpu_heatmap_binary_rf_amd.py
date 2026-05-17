"""Cross-GPU transfer heatmap with AMD row/column — binary, RF backend.

Vendor-extended version of regen_crossgpu_heatmap_binary_rf.py:
adds 'Radeon RX 9070 XT' (RDNA 4, consumer) as the 10th GPU.

Same protocol:
  * 30s windows / stride 15s
  * Same-GPU diagonal = StratifiedGroupKFold(k=5)
  * Cross-GPU cells = train on one GPU, test on another
  * Binary label: ml_training vs everything else
  * RandomForest (n=400, balanced class_weight)

Outputs:
  report/figures/cross_gpu_heatmap_binary_rf_amd.{png,pdf}
  classifier/results/cross_gpu_transfer_binary_rf_amd.csv
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["pdf.fonttype"] = 42
matplotlib.rcParams["ps.fonttype"] = 42
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)
RES = ROOT / "classifier" / "results"
RES.mkdir(parents=True, exist_ok=True)

MIN_RUNS_PER_CLASS = 3

_SHORT = {
    "A100-SXM4-40GB": "A100 40GB",
    "A100-SXM4-80GB": "A100 80GB",
    "RTX 4050 Laptop GPU": "RTX 4050 Laptop",
    "H100 80GB HBM3": "H100 80GB",
    "Radeon RX 9070 XT": "RX 9070 XT",   # AMD addition
}
GPU_ORDER = [
    "RTX 4050 Laptop", "RTX 5080", "RTX 5090",
    "A100 40GB", "A100 80GB", "H100 80GB",
    "H100 NVL", "H200", "B200",
    "RX 9070 XT",
]


def short_name(g):
    s = (str(g)
         .replace("NVIDIA ", "")
         .replace("GeForce ", "")
         .replace("AMD ", ""))
    return _SHORT.get(s, s)


def gpu_sort_key(g):
    s = short_name(g)
    if s in GPU_ORDER:
        return (0, GPU_ORDER.index(s))
    return (1, s)


def make_rf():
    return RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )


log.info("Loading all telemetry...")
df_raw = load_all_data()
if "gpu_name" not in df_raw.columns:
    df_raw["gpu_name"] = "unknown"

# AMD's amdsmi reports `pcie_tx/rx_mbps == -1`; normalize to 0 so RF doesn't
# learn "-1 means AMD" as a confounder.
for c in ("pcie_tx_mbps", "pcie_rx_mbps"):
    if c in df_raw.columns:
        df_raw[c] = df_raw[c].where(df_raw[c] >= 0, np.nan).fillna(0.0)

run_to_gpu = (df_raw.dropna(subset=["gpu_name"])
                   .groupby("run_id")["gpu_name"].first().to_dict())

df_win = sliding_windows(df_raw, window_sec=30, stride_sec=15)
df_win["gpu_name"] = df_win["run_id"].map(run_to_gpu).fillna("unknown")
df_win["binary_label"] = (df_win["threeway_label"] == "ml_training").astype(int)

feature_cols = get_feature_cols(df_win)
X = df_win[feature_cols].fillna(0).values.astype(np.float32)
y = df_win["binary_label"].values
groups = df_win["run_id"].values
gpus = df_win["gpu_name"].values
log.info(f"30s windows: {len(df_win):,} | features: {len(feature_cols)}")
log.info(f"Class balance — train:{(y==1).sum():,}  non-train:{(y==0).sum():,}")

run_first = df_win.groupby("run_id").agg(
    gpu=("gpu_name", "first"), label=("binary_label", "first"))
class_counts = run_first.groupby(["gpu", "label"]).size().unstack(fill_value=0)
log.info(f"Runs per GPU × binary class:\n{class_counts}")

eligible = [g for g in class_counts.index if str(g) not in ("unknown", "nan")]
train_gpus = sorted(
    [g for g in eligible
     if (class_counts.loc[g] >= MIN_RUNS_PER_CLASS).all()],
    key=gpu_sort_key,
)
all_gpus = train_gpus
log.info(f"GPUs eligible: {[short_name(g) for g in train_gpus]}")

acc_mat = np.full((len(train_gpus), len(all_gpus)), np.nan)
for i, src in enumerate(train_gpus):
    tr_mask = (gpus == src)
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    clf = make_rf()
    clf.fit(X_tr, y_tr)
    log.info(f"Trained RF on {short_name(src)} ({tr_mask.sum():,} windows)")

    for j, tgt in enumerate(all_gpus):
        te_mask = (gpus == tgt)
        if te_mask.sum() == 0:
            continue
        if src == tgt:
            X_s, y_s, g_s = X[te_mask], y[te_mask], groups[te_mask]
            min_class = int(pd.Series(y_s).value_counts().min())
            n_splits = max(2, min(5, min_class))
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            accs = []
            for tri, tei in cv.split(X_s, y_s, g_s):
                c = make_rf()
                c.fit(X_s[tri], y_s[tri])
                accs.append(accuracy_score(y_s[tei], c.predict(X_s[tei])))
            acc_mat[i, j] = float(np.mean(accs))
            log.info(f"  {short_name(src):20s} -> self (CV) : {acc_mat[i,j]:.4f}")
        else:
            acc_mat[i, j] = accuracy_score(y[te_mask], clf.predict(X[te_mask]))
            log.info(f"  {short_name(src):20s} -> {short_name(tgt):20s}: {acc_mat[i,j]:.4f}")

row_labels = [short_name(g) for g in train_gpus]
col_labels = [short_name(g) for g in all_gpus]

fig, ax = plt.subplots(figsize=(max(6.5, 0.9 * len(col_labels) + 2.5),
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

png_path = FIG / "cross_gpu_heatmap_binary_rf_amd.png"
pdf_path = FIG / "cross_gpu_heatmap_binary_rf_amd.pdf"
fig.savefig(png_path, dpi=150)
fig.savefig(pdf_path)
log.info(f"Saved {png_path.name} / {pdf_path.name}")

csv_path = RES / "cross_gpu_transfer_binary_rf_amd.csv"
pd.DataFrame(acc_mat, index=row_labels, columns=col_labels).to_csv(csv_path)
log.info(f"Saved {csv_path}")
