"""Regenerate figures that depend on the retrained classifier:
  - Figure: Confusion matrix (best model: RF 30s)
  - Figure: Feature importance (top 15 XGBoost)
  - Figure: Accuracy vs window size
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

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols, compute_sample_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
FIG_DIR = PROJECT_ROOT / "report" / "figures"
FIG_DIR.mkdir(parents=True, exist_ok=True)

# ── 1. Accuracy vs window size ────────────────────────────────────────────
res = pd.read_csv(PROJECT_ROOT / "classifier" / "results" / "threeway_results.csv")
log.info(f"Loaded results: {len(res)} rows")

fig, ax = plt.subplots(figsize=(5.5, 3.5))
for clf_name in res["classifier"].unique():
    sub = res[res["classifier"] == clf_name].sort_values("window_sec")
    ax.errorbar(sub["window_sec"], sub["acc_mean"], yerr=sub["acc_std"],
                marker="o", capsize=3, label=clf_name, linewidth=1.5)
ax.set_xlabel("Window size (s)")
ax.set_ylabel("Accuracy (5-fold grouped CV)")
ax.set_title("Three-way classification accuracy vs window size")
ax.set_ylim(0.95, 1.00)
ax.grid(alpha=0.3)
ax.legend()
fig.tight_layout()
fig.savefig(FIG_DIR / "accuracy_vs_window.png", dpi=150)
fig.savefig(FIG_DIR / "accuracy_vs_window.pdf")
log.info(f"Saved accuracy_vs_window.png/.pdf")
plt.close(fig)

# ── 2. Load data and extract 30s windows (best config) ─────────────────────
log.info("Loading all telemetry for confusion matrix + feature importance...")
df_raw = load_all_data()
df_win = sliding_windows(df_raw, window_sec=30, stride_sec=15)
feature_cols = get_feature_cols(df_win)
X = df_win[feature_cols].fillna(0).values.astype(np.float32)
y = df_win["threeway_label"].values
groups = df_win["run_id"].values
log.info(f"30s windows: {len(df_win)} | features: {len(feature_cols)}")

# ── 3. Confusion matrix via cross-validated predictions (RandomForest 30s) ─
log.info("Computing cross-validated predictions for confusion matrix (RF 30s)...")
cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
y_true_all, y_pred_all = [], []
for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
    rf = RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1,
    )
    rf.fit(X[tr_idx], y[tr_idx])
    y_pred_all.extend(rf.predict(X[te_idx]))
    y_true_all.extend(y[te_idx])
    log.info(f"  fold {fold_idx+1}/5 done")

labels = ["ml_training", "ml_inference", "other"]
cm = confusion_matrix(y_true_all, y_pred_all, labels=labels)
cm_norm = cm / cm.sum(axis=1, keepdims=True)

fig, ax = plt.subplots(figsize=(5.5, 4.5))
im = ax.imshow(cm_norm, cmap="Blues", vmin=0, vmax=1)
ax.set_xticks(range(3))
ax.set_yticks(range(3))
ax.set_xticklabels([l.replace("ml_", "") for l in labels], rotation=0)
ax.set_yticklabels([l.replace("ml_", "") for l in labels])
ax.set_xlabel("Predicted")
ax.set_ylabel("True")
ax.set_title("Confusion matrix — RandomForest, 30s windows\n"
             f"(n={len(y_true_all):,} windows, grouped 5-fold CV)")
for i in range(3):
    for j in range(3):
        color = "white" if cm_norm[i, j] > 0.5 else "black"
        ax.text(j, i, f"{cm[i, j]:,}\n({cm_norm[i, j]*100:.1f}%)",
                ha="center", va="center", color=color, fontsize=10)
fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
fig.tight_layout()
fig.savefig(FIG_DIR / "confusion_matrix_rf30s.png", dpi=150)
fig.savefig(FIG_DIR / "confusion_matrix_rf30s.pdf")
log.info(f"Saved confusion_matrix_rf30s.png/.pdf")
plt.close(fig)

# ── 4. Feature importance (XGBoost trained on all 30s data) ─────────────────
log.info("Training XGBoost on all 30s data for feature importance...")
le = LabelEncoder()
y_enc = le.fit_transform(y)
w = compute_sample_weights(y_enc)
xgb_clf = xgb.XGBClassifier(
    n_estimators=500, max_depth=6, learning_rate=0.05,
    subsample=0.8, colsample_bytree=0.8,
    use_label_encoder=False, eval_metric="mlogloss",
    random_state=42, n_jobs=-1, verbosity=0,
)
xgb_clf.fit(X, y_enc, sample_weight=w)
importances = xgb_clf.feature_importances_
top_n = 15
top_idx = np.argsort(importances)[::-1][:top_n]
top_names = [feature_cols[i] for i in top_idx]
top_vals = importances[top_idx]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
y_pos = np.arange(top_n)
ax.barh(y_pos, top_vals[::-1], color="steelblue")
ax.set_yticks(y_pos)
ax.set_yticklabels([n.replace("_", " ") for n in top_names[::-1]], fontsize=8)
ax.set_xlabel("XGBoost feature importance (gain)")
ax.set_title(f"Top-{top_n} features — XGBoost, 30s windows")
ax.grid(axis="x", alpha=0.3)
fig.tight_layout()
fig.savefig(FIG_DIR / "feature_importance_top15.png", dpi=150)
fig.savefig(FIG_DIR / "feature_importance_top15.pdf")
log.info(f"Saved feature_importance_top15.png/.pdf")
plt.close(fig)

log.info("Top 15 features:")
for name, val in zip(top_names, top_vals):
    log.info(f"  {name:50s}  {val:.4f}")

log.info(f"All figures saved under {FIG_DIR}")
