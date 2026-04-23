"""
Compare TwoStageClassifier backends (XGBoost vs RandomForest)
=============================================================

Head-to-head: same fold splits, same features, same rescue rule; only
the Stage 1 / Stage 2 classifier family differs.

Metrics reported:
  1. Pooled 5-fold StratifiedGroupKFold accuracy + macro-F1 at 30s / 60s.
  2. Oracle regression on `whitebox_inference` (held out at train time):
     median / mean / p95 / frac>0.5 of P(train) for each backend.

Output:
  - classifier/results/twostage_backends.csv
  - classifier/results/twostage_backends.md

Usage:
    python -m classifier.compare_twostage_backends
"""

from __future__ import annotations

import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"

BACKENDS = ("xgboost", "random_forest")


def run_pooled_cv(window_secs=(30, 60)) -> pd.DataFrame:
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score
    from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
    from classifier.twostage import TwoStageClassifier

    log.info("Loading telemetry data...")
    df_raw = load_all_data()

    rows = []
    for win_sec in window_secs:
        stride_sec = win_sec / 2
        log.info(f"=== window={win_sec}s stride={stride_sec}s ===")
        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
        if df_win.empty:
            continue
        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        groups = df_win["run_id"].values

        run_class_counts = df_win.groupby("run_id")["threeway_label"].first().value_counts()
        n_splits = max(2, min(5, int(run_class_counts.min())))
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        per_backend_accs: dict = {b: [] for b in BACKENDS}
        per_backend_f1s: dict = {b: [] for b in BACKENDS}

        for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups)):
            for backend in BACKENDS:
                clf = TwoStageClassifier(backend=backend).fit(
                    X[tr], y[tr], groups=groups[tr], feature_cols=feature_cols,
                )
                y_pred = clf.predict(X[te])
                acc = accuracy_score(y[te], y_pred)
                f1 = f1_score(y[te], y_pred, average="macro", zero_division=0)
                per_backend_accs[backend].append(acc)
                per_backend_f1s[backend].append(f1)
                log.info(f"  fold {fold_idx+1}/{n_splits}  {backend:15s} acc={acc:.4f} f1={f1:.4f}")

        for backend in BACKENDS:
            row = {
                "window_sec": win_sec,
                "backend": backend,
                "acc_mean": float(np.mean(per_backend_accs[backend])),
                "acc_std": float(np.std(per_backend_accs[backend])),
                "f1_macro": float(np.mean(per_backend_f1s[backend])),
                "n_folds": n_splits,
            }
            rows.append(row)
            log.info(
                f"  [{backend} @{win_sec}s] acc={row['acc_mean']:.4f}±{row['acc_std']:.4f}  f1={row['f1_macro']:.4f}"
            )
    return pd.DataFrame(rows)


def run_oracle_regression(target_label: str = "whitebox_inference",
                          window_sec: int = 60) -> list[dict]:
    from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
    from classifier.twostage import TwoStageClassifier

    df_raw = load_all_data()
    mask_target = df_raw["workload_label"] == target_label
    df_target = df_raw[mask_target].copy()
    df_train = df_raw[~mask_target].copy()
    log.info(
        f"Oracle regression: target={target_label} ({df_target['run_id'].nunique()} runs), "
        f"training pool = {df_train['run_id'].nunique()} runs"
    )

    df_train_win = sliding_windows(df_train, window_sec=window_sec, stride_sec=window_sec / 2)
    feature_cols = get_feature_cols(df_train_win)
    X_tr = df_train_win[feature_cols].fillna(0).values.astype(np.float32)
    y_tr = df_train_win["threeway_label"].values
    g_tr = df_train_win["run_id"].values

    df_target_win = sliding_windows(df_target, window_sec=window_sec, stride_sec=window_sec / 2)
    if df_target_win.empty:
        raise RuntimeError("No windows for target label")
    X_te = df_target_win[feature_cols].fillna(0).values.astype(np.float32)

    rows = []
    for backend in BACKENDS:
        clf = TwoStageClassifier(backend=backend).fit(
            X_tr, y_tr, groups=g_tr, feature_cols=feature_cols,
        )
        idx = int(np.where(clf.classes_ == "ml_training")[0][0])
        p_train = clf.predict_proba(X_te)[:, idx]
        rows.append({
            "target_label": target_label,
            "backend": backend,
            "window_sec": window_sec,
            "n_windows": int(len(X_te)),
            "P_train_median": float(np.median(p_train)),
            "P_train_mean": float(np.mean(p_train)),
            "P_train_p95": float(np.percentile(p_train, 95)),
            "P_train_gt_0p5_frac": float((p_train > 0.5).mean()),
        })
        log.info(
            f"  [{backend}] median={rows[-1]['P_train_median']:.4f}  "
            f"mean={rows[-1]['P_train_mean']:.4f}  "
            f"p95={rows[-1]['P_train_p95']:.4f}  "
            f"frac>0.5={rows[-1]['P_train_gt_0p5_frac']:.3f}"
        )
    return rows


def write_markdown(cv_df: pd.DataFrame, oracle_rows: list[dict], out: Path):
    lines = [
        "# TwoStage backend comparison — XGBoost vs RandomForest",
        "",
        "Generated by `classifier/compare_twostage_backends.py`.",
        "",
        "## 1. Pooled 5-fold CV (same splits for both backends)",
        "",
        "| window | backend       | accuracy            | macro-F1 |",
        "|---:|---|---:|---:|",
    ]
    for _, r in cv_df.iterrows():
        lines.append(
            f"| {int(r['window_sec'])}s | {r['backend']} | "
            f"{r['acc_mean']:.4f}±{r['acc_std']:.4f} | {r['f1_macro']:.4f} |"
        )
    lines += [
        "",
        "## 2. Oracle regression on `whitebox_inference` (held out at train time)",
        "",
        "| backend | median P(train) | mean | p95 | frac>0.5 |",
        "|---|---:|---:|---:|---:|",
    ]
    for r in oracle_rows:
        lines.append(
            f"| {r['backend']} | {r['P_train_median']:.4f} | "
            f"{r['P_train_mean']:.4f} | {r['P_train_p95']:.4f} | "
            f"{r['P_train_gt_0p5_frac']:.3f} |"
        )
    lines += [
        "",
        "Lower is better on the oracle regression — `whitebox_inference` is",
        "real (forward-only) inference held out at training time. Both",
        "backends should stay well below the 0.5 governance threshold.",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote {out}")


def main():
    RESULTS_DIR.mkdir(exist_ok=True)
    cv_df = run_pooled_cv(window_secs=(30, 60))
    cv_df.to_csv(RESULTS_DIR / "twostage_backends.csv", index=False)
    log.info(f"Saved {RESULTS_DIR/'twostage_backends.csv'}")
    oracle_rows = run_oracle_regression()
    write_markdown(cv_df, oracle_rows, RESULTS_DIR / "twostage_backends.md")
    log.info("Done.")


if __name__ == "__main__":
    main()
