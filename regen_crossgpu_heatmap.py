"""Figure 1: cross-GPU transfer heatmap.
Trains XGBoost on windows from each GPU type; evaluates on every other GPU.
Includes a 'pooled (5-fold CV)' row. 30s windows.
"""
import logging
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols, compute_sample_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

MIN_RUNS_PER_CLASS = 3  # require each class present with ≥3 runs for a GPU to train-on

log.info("Loading all telemetry...")
df_raw = load_all_data()
# Build run_id → gpu_name map before windowing (windows drop gpu_name)
if "gpu_name" not in df_raw.columns:
    df_raw["gpu_name"] = "unknown"
run_to_gpu = (df_raw.dropna(subset=["gpu_name"])
                   .groupby("run_id")["gpu_name"].first().to_dict())

df_win = sliding_windows(df_raw, window_sec=30, stride_sec=15)
df_win["gpu_name"] = df_win["run_id"].map(run_to_gpu).fillna("unknown")
feature_cols = get_feature_cols(df_win)
X = df_win[feature_cols].fillna(0).values.astype(np.float32)
y = df_win["threeway_label"].values
groups = df_win["run_id"].values
gpus = df_win["gpu_name"].values
log.info(f"30s windows: {len(df_win):,} | features: {len(feature_cols)}")

# Which GPU types have enough data per class to serve as training source?
run_first = df_win.groupby("run_id").agg(
    gpu=("gpu_name", "first"), label=("threeway_label", "first"))
class_counts = run_first.groupby(["gpu", "label"]).size().unstack(fill_value=0)
log.info(f"Runs per GPU × class:\n{class_counts}")

train_gpus = [g for g in class_counts.index
              if (class_counts.loc[g] >= MIN_RUNS_PER_CLASS).all()]
all_gpus = sorted(class_counts.index.tolist())
log.info(f"GPUs eligible as training source: {train_gpus}")
log.info(f"All GPUs (as eval targets): {all_gpus}")


def short_name(g):
    """Shorten 'NVIDIA GeForce RTX 5080' → 'RTX 5080'."""
    s = str(g).replace("NVIDIA ", "").replace("GeForce ", "")
    return s


# ── Per-GPU → per-GPU transfer matrix ─────────────────────────────────────
le = LabelEncoder().fit(y)
acc_mat = np.full((len(train_gpus) + 1, len(all_gpus)), np.nan)

for i, src in enumerate(train_gpus):
    tr_mask = (gpus == src)
    X_tr, y_tr = X[tr_mask], y[tr_mask]
    y_tr_enc = le.transform(y_tr)
    w_tr = compute_sample_weights(y_tr_enc)
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_tr, y_tr_enc, sample_weight=w_tr)
    log.info(f"  Trained XGB on {short_name(src)} ({tr_mask.sum():,} windows)")

    for j, tgt in enumerate(all_gpus):
        te_mask = (gpus == tgt)
        if te_mask.sum() == 0:
            continue
        if src == tgt:
            # Same-GPU: grouped 5-fold CV to avoid leakage
            accs = []
            X_s, y_s, g_s = X[te_mask], y[te_mask], groups[te_mask]
            cv = StratifiedGroupKFold(n_splits=min(5, pd.Series(y_s).value_counts().min()),
                                      shuffle=True, random_state=42)
            for tri, tei in cv.split(X_s, y_s, g_s):
                y_tri_enc = le.transform(y_s[tri])
                w = compute_sample_weights(y_tri_enc)
                c = xgb.XGBClassifier(
                    n_estimators=500, max_depth=6, learning_rate=0.05,
                    subsample=0.8, colsample_bytree=0.8,
                    use_label_encoder=False, eval_metric="mlogloss",
                    random_state=42, n_jobs=-1, verbosity=0,
                )
                c.fit(X_s[tri], y_tri_enc, sample_weight=w)
                y_pred = le.inverse_transform(c.predict(X_s[tei]))
                accs.append(accuracy_score(y_s[tei], y_pred))
            acc_mat[i, j] = float(np.mean(accs))
            log.info(f"    {short_name(src):20s} → self (CV): {acc_mat[i,j]:.4f}")
        else:
            y_pred = le.inverse_transform(clf.predict(X[te_mask]))
            acc_mat[i, j] = accuracy_score(y[te_mask], y_pred)
            log.info(f"    {short_name(src):20s} → {short_name(tgt):20s}: {acc_mat[i,j]:.4f}")

# ── Pooled row: 5-fold grouped CV, evaluated per-GPU on held-out fold ─────
log.info("Pooled classifier (5-fold grouped CV)...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
per_gpu_correct = {g: [0, 0] for g in all_gpus}  # [correct, total]
for tr_idx, te_idx in cv.split(X, y, groups):
    y_tr_enc = le.transform(y[tr_idx])
    w_tr = compute_sample_weights(y_tr_enc)
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X[tr_idx], y_tr_enc, sample_weight=w_tr)
    y_pred = le.inverse_transform(clf.predict(X[te_idx]))
    for g in all_gpus:
        m = (gpus[te_idx] == g)
        if m.sum() == 0:
            continue
        per_gpu_correct[g][0] += (y_pred[m] == y[te_idx][m]).sum()
        per_gpu_correct[g][1] += m.sum()

for j, g in enumerate(all_gpus):
    c, t = per_gpu_correct[g]
    acc_mat[-1, j] = c / t if t > 0 else np.nan

log.info(f"Pooled per-GPU accuracies: " +
         ", ".join(f"{short_name(g)}={acc_mat[-1,j]:.3f}" for j, g in enumerate(all_gpus)))

# ── Plot heatmap ─────────────────────────────────────────────────────────
row_labels = [short_name(g) for g in train_gpus] + ["Pooled (5-fold CV)"]
col_labels = [short_name(g) for g in all_gpus]

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
        if np.isnan(v):
            continue
        color = "white" if v < 0.75 else "black"
        ax.text(j, i, f"{v*100:.1f}", ha="center", va="center",
                color=color, fontsize=8)
# Separator above pooled row
ax.axhline(len(row_labels) - 1.5, color="black", linewidth=1.2)
fig.colorbar(im, ax=ax, fraction=0.04, pad=0.03, label="accuracy")
fig.tight_layout()
fig.savefig(FIG / "cross_gpu_heatmap.png", dpi=150)
fig.savefig(FIG / "cross_gpu_heatmap.pdf")
log.info(f"Saved cross_gpu_heatmap.png/.pdf")

# Save the numeric matrix too
pd.DataFrame(acc_mat, index=row_labels, columns=col_labels).to_csv(
    ROOT / "classifier" / "results" / "cross_gpu_transfer.csv")
log.info("Saved classifier/results/cross_gpu_transfer.csv")
