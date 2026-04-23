"""Fast retrain: XGBoost + RF only, saves after each window size."""
import sys
import logging
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import classification_report, accuracy_score, f1_score, confusion_matrix
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols, compute_sample_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent
results_dir = PROJECT_ROOT / "classifier" / "results"
results_dir.mkdir(exist_ok=True)

log.info("Loading all telemetry data...")
df_raw = load_all_data()

all_results = []

for win_sec, stride_sec in [(15, 7), (30, 15), (60, 30)]:
    log.info(f"=== Window: {win_sec}s, stride: {stride_sec}s ===")
    df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
    if df_win.empty:
        continue

    feature_cols = get_feature_cols(df_win)
    X = df_win[feature_cols].fillna(0).values.astype(np.float32)
    y = df_win["threeway_label"].values
    groups = df_win["run_id"].values

    log.info(f"Windows: {len(df_win)} | Features: {len(feature_cols)}")
    log.info(f"Class dist: {pd.Series(y).value_counts().to_dict()}")

    run_class_counts = df_win.groupby("run_id")["threeway_label"].first().value_counts()
    n_splits = min(5, run_class_counts.min())

    classifiers = {
        "XGBoost": xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            use_label_encoder=False, eval_metric="mlogloss",
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
        accs, f1s = [], []
        all_y_true, all_y_pred = [], []
        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
            X_tr, X_te = X[tr_idx], X[te_idx]
            y_tr, y_te = y[tr_idx], y[te_idx]
            if clf_name == "XGBoost":
                le = LabelEncoder()
                y_tr_enc = le.fit_transform(y_tr)
                w_tr = compute_sample_weights(y_tr_enc)
                clf.fit(X_tr, y_tr_enc, sample_weight=w_tr)
                y_pred = le.inverse_transform(clf.predict(X_te))
            else:
                clf.fit(X_tr, y_tr)
                y_pred = clf.predict(X_te)
            accs.append(accuracy_score(y_te, y_pred))
            f1s.append(f1_score(y_te, y_pred, average="macro", zero_division=0))
            all_y_true.extend(y_te)
            all_y_pred.extend(y_pred)

        mean_acc, std_acc, mean_f1 = np.mean(accs), np.std(accs), np.mean(f1s)
        log.info(f"  [{clf_name:15s}] acc={mean_acc:.4f}+/-{std_acc:.4f}  f1={mean_f1:.4f}")
        log.info("\n" + classification_report(all_y_true, all_y_pred, zero_division=0))
        # Confusion matrix
        labels = ["ml_training", "ml_inference", "other"]
        cm = confusion_matrix(all_y_true, all_y_pred, labels=labels)
        cm_df = pd.DataFrame(cm, index=labels, columns=labels)
        log.info(f"Confusion matrix ({clf_name}, {win_sec}s):\n{cm_df}")

        all_results.append({
            "classifier": clf_name, "window_sec": win_sec,
            "acc_mean": mean_acc, "acc_std": std_acc, "f1_macro": mean_f1,
            "n_windows": len(df_win), "n_runs": df_win["run_id"].nunique(),
        })

    # Save after each window size in case of interruption
    pd.DataFrame(all_results).to_csv(results_dir / "threeway_results.csv", index=False)
    log.info(f"Saved partial results ({len(all_results)} rows)")

log.info("=== SUMMARY ===")
for r in sorted(all_results, key=lambda x: -x["acc_mean"]):
    log.info(f"  {r['classifier']:15s} {r['window_sec']}s  acc={r['acc_mean']:.4f}+/-{r['acc_std']:.4f}  f1={r['f1_macro']:.4f}")
log.info("Done.")
