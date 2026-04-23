"""Regenerate Appendix G classifier comparison.

Evaluates four classifiers head-to-head on the full post-Round-5b dataset
at 30 s windows with 5-fold StratifiedGroupKFold:
  - RandomForest (headline)
  - XGBoost
  - SVM-RBF (with per-feature standardisation inside the CV fold)
  - Logistic regression (with standardisation)

Outputs:
  - classifier/results/classifier_comparison.csv
  - report/figures/classifier_comparison.{png,pdf}

Run when time permits — SVM-RBF on 26k windows × 166 features with 5-fold
CV takes tens of minutes; LogReg is fast.
"""
import logging
from pathlib import Path
import time
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
import xgboost as xgb

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols, compute_sample_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent
RES = ROOT / "classifier" / "results"
FIG = ROOT / "report" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

WIN_SEC, STRIDE_SEC = 30, 15

log.info("Loading telemetry data...")
df_raw = load_all_data()
df_win = sliding_windows(df_raw, window_sec=WIN_SEC, stride_sec=STRIDE_SEC)
feat_cols = get_feature_cols(df_win)
X = df_win[feat_cols].fillna(0).values.astype(np.float32)
y = df_win["threeway_label"].values
groups = df_win["run_id"].values
log.info(f"Windows: {len(df_win)} | Features: {len(feat_cols)} | Runs: {df_win['run_id'].nunique()}")

cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
le = LabelEncoder()
y_enc = le.fit_transform(y)

classifiers = {
    "RandomForest": lambda: RandomForestClassifier(
        n_estimators=400, max_depth=None, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced",
        random_state=42, n_jobs=-1),
    "XGBoost": lambda: xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0),
    "SVM-RBF": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("svm", SVC(kernel="rbf", C=1.0, gamma="scale",
                    class_weight="balanced", cache_size=2000,
                    random_state=42))]),
    "LogReg": lambda: Pipeline([
        ("scale", StandardScaler()),
        ("lr", LogisticRegression(max_iter=1000, class_weight="balanced",
                                   solver="saga", multi_class="multinomial",
                                   random_state=42, n_jobs=-1))]),
}

rows = []
for name, factory in classifiers.items():
    log.info(f"=== {name} ===")
    accs, f1s, times = [], [], []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y_enc, groups)):
        t0 = time.time()
        clf = factory()
        if name == "XGBoost":
            w_tr = compute_sample_weights(y_enc[tr])
            clf.fit(X[tr], y_enc[tr], sample_weight=w_tr)
        else:
            clf.fit(X[tr], y_enc[tr])
        y_pred = clf.predict(X[te])
        accs.append(accuracy_score(y_enc[te], y_pred))
        f1s.append(f1_score(y_enc[te], y_pred, average="macro", zero_division=0))
        times.append(time.time() - t0)
        log.info(f"  fold {fold_idx+1}/5 acc={accs[-1]:.4f} f1={f1s[-1]:.4f} t={times[-1]:.0f}s")
    mean_acc, std_acc, mean_f1 = np.mean(accs), np.std(accs), np.mean(f1s)
    log.info(f"  [{name}] acc={mean_acc:.4f} +/- {std_acc:.4f}  f1={mean_f1:.4f}  total_t={sum(times):.0f}s")
    rows.append(dict(classifier=name, accuracy=mean_acc, accuracy_std=std_acc,
                     f1_macro=mean_f1, fit_seconds=sum(times)))

out = pd.DataFrame(rows)
out.to_csv(RES / "classifier_comparison.csv", index=False)
log.info(f"Saved {RES/'classifier_comparison.csv'}")

# ── Bar chart ─────────────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(6.5, 4.0))
x_pos = np.arange(len(out))
bars_acc = ax.bar(x_pos - 0.2, out["accuracy"] * 100, 0.4,
                  yerr=out["accuracy_std"] * 100, capsize=4,
                  color="#1f77b4", label="Accuracy", edgecolor="black", linewidth=0.4)
bars_f1 = ax.bar(x_pos + 0.2, out["f1_macro"] * 100, 0.4,
                 color="#2ca02c", label="Macro-F1", edgecolor="black", linewidth=0.4)
ax.set_xticks(x_pos)
ax.set_xticklabels(out["classifier"], rotation=0)
ax.set_ylabel("Score (%)")
ax.set_ylim(80, 101)
ax.axhline(98.42, color="gray", linestyle=":", linewidth=0.8,
           label="RF 30 s headline (98.42 %)")
ax.set_title(f"Classifier comparison — 3-way, 30 s windows, 5-fold grouped CV\n"
             f"n = {len(df_win):,} windows across {df_win['run_id'].nunique()} runs")
ax.legend(loc="lower right", fontsize=8)
ax.grid(axis="y", alpha=0.3)
for bar in list(bars_acc) + list(bars_f1):
    h = bar.get_height()
    ax.annotate(f"{h:.1f}", xy=(bar.get_x() + bar.get_width()/2, h),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=7)
fig.tight_layout()
fig.savefig(FIG / "classifier_comparison.png", dpi=150)
fig.savefig(FIG / "classifier_comparison.pdf")
log.info(f"Saved classifier_comparison.png/.pdf")
plt.close(fig)
