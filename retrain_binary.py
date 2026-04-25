"""Retrain XGBoost + RF for binary (training vs non-training), 5-fold grouped CV.

Mirrors retrain_fast.py. Writes classifier/results/binary_results.csv with one
row per (classifier, window_sec). Plotting is decoupled — see
plot_accuracy_vs_window_binary.py.
"""
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
import xgboost as xgb

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols, compute_sample_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
results_dir = PROJECT_ROOT / "classifier" / "results"
results_dir.mkdir(exist_ok=True)
OUT_CSV = results_dir / "binary_results.csv"

WINDOWS = [(5, 2), (15, 7), (30, 15), (60, 30)]

# Resume support: keep rows already present in OUT_CSV; only train missing
# (window_sec, classifier) combinations.
existing = pd.DataFrame()
if OUT_CSV.exists():
    existing = pd.read_csv(OUT_CSV)
    log.info(f"Found existing {OUT_CSV.name} with {len(existing)} rows — will skip already-trained pairs")
existing_keys = (
    set(zip(existing["window_sec"].astype(int), existing["classifier"]))
    if not existing.empty else set()
)

# Decide which window sizes still need any work; skip telemetry load if none.
need_windows = [(w, s) for w, s in WINDOWS
                if {(w, "XGBoost"), (w, "RandomForest")} - existing_keys]
if not need_windows:
    log.info("All (window, classifier) pairs already present — nothing to do.")
    raise SystemExit(0)

log.info(f"Need to train: {[w for w, _ in need_windows]}")
log.info("Loading all telemetry data...")
df_raw = load_all_data()

all_results = existing.to_dict("records") if not existing.empty else []

for win_sec, stride_sec in need_windows:
    log.info(f"=== Window: {win_sec}s, stride: {stride_sec}s ===")
    df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
    if df_win.empty:
        continue

    feature_cols = get_feature_cols(df_win)
    X = df_win[feature_cols].fillna(0).values.astype(np.float32)
    # Binary label: 1 = ml_training, 0 = anything else (inference + other)
    y = (df_win["threeway_label"].values == "ml_training").astype(int)
    groups = df_win["run_id"].values

    log.info(f"Windows: {len(df_win)} | Features: {len(feature_cols)}")
    log.info(f"Class dist: training={int((y==1).sum())}  non-training={int((y==0).sum())}")

    # Per-run class for grouped CV split sizing
    run_first = df_win.groupby("run_id")["threeway_label"].first()
    run_binary = (run_first == "ml_training").astype(int)
    n_splits = min(5, int(run_binary.value_counts().min()))

    classifiers = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced",
            random_state=42, n_jobs=-1,
        ),
    }

    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    for clf_name, clf in classifiers.items():
        if (win_sec, clf_name) in existing_keys:
            log.info(f"  [{clf_name:15s}] already in CSV — skipping")
            continue
        accs, f1s = [], []
        all_y_true, all_y_pred = [], []
        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if clf_name == "XGBoost":
                w_tr = compute_sample_weights(y_tr)
                clf.fit(X_tr, y_tr, sample_weight=w_tr)
            else:
                clf.fit(X_tr, y_tr)
            y_pred = clf.predict(X_te)
            accs.append(accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
            all_y_true.extend(y_te)
            all_y_pred.extend(y_pred)

        mean_acc, std_acc, mean_f1 = np.mean(accs), np.std(accs), np.mean(f1s)
        log.info(f"  [{clf_name:15s}] acc={mean_acc:.4f}+/-{std_acc:.4f}  f1={mean_f1:.4f}")
        log.info("\n" + classification_report(
            all_y_true, all_y_pred,
            target_names=["non_training", "training"], zero_division=0))
        cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1])
        cm_df = pd.DataFrame(cm,
                             index=["non_training", "training"],
                             columns=["non_training", "training"])
        log.info(f"Confusion matrix ({clf_name}, {win_sec}s):\n{cm_df}")

        all_results.append({
            "classifier": clf_name, "window_sec": win_sec,
            "acc_mean": mean_acc, "acc_std": std_acc, "f1_macro": mean_f1,
            "n_windows": len(df_win), "n_runs": df_win["run_id"].nunique(),
        })

    pd.DataFrame(all_results).to_csv(OUT_CSV, index=False)
    log.info(f"Saved partial results ({len(all_results)} rows) -> {OUT_CSV}")

log.info("=== SUMMARY ===")
for r in sorted(all_results, key=lambda x: -x["acc_mean"]):
    log.info(f"  {r['classifier']:15s} {r['window_sec']}s  "
             f"acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  f1={r['f1_macro']:.4f}")
log.info("Done.")
