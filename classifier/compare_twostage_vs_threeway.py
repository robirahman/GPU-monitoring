"""
A/B Harness: Two-Stage vs. Three-Way Classifier (Round 5b)
==========================================================

Matched evaluation of both classifiers on identical windowed data:

  1. Pooled 5-fold StratifiedGroupKFold accuracy + macro-F1 @ 15/30/60s.
  2. Oracle regression test on Round-5 WB-O parquets — key success metric
     for the defender response: median P(train) on `whitebox_inference`
     should drop from ~0.99 (three-way) to <0.5 (two-stage).
  3. Per-class breakdown and confusion-matrix diff.

Outputs:
  - classifier/results/twostage_vs_threeway.csv
  - classifier/results/twostage_vs_threeway.md

Usage:
    python -m classifier.compare_twostage_vs_threeway
    python -m classifier.compare_twostage_vs_threeway --windows 60
"""

from __future__ import annotations

import argparse
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


def _train_threeway_on_fold(X_tr, y_tr, groups_tr):
    """Train the original three-way XGBoost on a fold, with the same weighting
    as threeway_improved.train_and_evaluate."""
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    from classifier.threeway_improved import compute_sample_weights

    le = LabelEncoder()
    y_enc = le.fit_transform(y_tr)
    w = compute_sample_weights(y_enc, groups=groups_tr)
    clf = xgb.XGBClassifier(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        use_label_encoder=False, eval_metric="mlogloss",
        random_state=42, n_jobs=-1, verbosity=0,
    )
    clf.fit(X_tr, y_enc, sample_weight=w)
    return clf, le


def _threeway_predict(clf, le, X_te) -> np.ndarray:
    return le.inverse_transform(clf.predict(X_te))


def _threeway_proba(clf, le, X_te, class_name: str = "ml_training") -> np.ndarray:
    """Return P(class_name) column from the three-way classifier."""
    idx = int(np.where(le.classes_ == class_name)[0][0])
    return clf.predict_proba(X_te)[:, idx]


def run_pooled_cv(window_secs: tuple[int, ...] = (15, 30, 60)) -> pd.DataFrame:
    """Run matched 5-fold CV for both classifiers at each window size."""
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import accuracy_score, f1_score, classification_report
    from classifier.threeway_improved import (
        load_all_data, sliding_windows, get_feature_cols,
    )
    from classifier.twostage import TwoStageClassifier

    log.info("Loading all telemetry data once for matched eval...")
    df_raw = load_all_data()

    rows = []
    for win_sec in window_secs:
        stride_sec = win_sec / 2
        log.info(f"\n{'='*60}\nWindow: {win_sec}s, stride: {stride_sec}s\n{'='*60}")
        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
        if df_win.empty:
            continue

        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        groups = df_win["run_id"].values

        unique_classes = np.unique(y)
        if len(unique_classes) < 3:
            log.warning(f"Skipping {win_sec}s: fewer than 3 classes")
            continue

        run_class_counts = df_win.groupby("run_id")["threeway_label"].first().value_counts()
        n_splits = min(5, int(run_class_counts.min()))
        if n_splits < 2:
            n_splits = 2

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        three_acc, three_f1, two_acc, two_f1 = [], [], [], []
        three_preds_all, two_preds_all, y_true_all = [], [], []

        for tr_idx, te_idx in cv.split(X, y, groups):
            # Three-way
            clf3, le3 = _train_threeway_on_fold(X[tr_idx], y[tr_idx], groups[tr_idx])
            y3 = _threeway_predict(clf3, le3, X[te_idx])
            three_acc.append(accuracy_score(y[te_idx], y3))
            three_f1.append(f1_score(y[te_idx], y3, average="macro", zero_division=0))
            three_preds_all.extend(y3)

            # Two-stage
            clf2 = TwoStageClassifier()
            clf2.fit(X[tr_idx], y[tr_idx], groups=groups[tr_idx], feature_cols=feature_cols)
            y2 = clf2.predict(X[te_idx])
            two_acc.append(accuracy_score(y[te_idx], y2))
            two_f1.append(f1_score(y[te_idx], y2, average="macro", zero_division=0))
            two_preds_all.extend(y2)

            y_true_all.extend(y[te_idx])

        row = {
            "window_sec": win_sec,
            "three_way_acc": float(np.mean(three_acc)),
            "three_way_acc_std": float(np.std(three_acc)),
            "three_way_f1": float(np.mean(three_f1)),
            "two_stage_acc": float(np.mean(two_acc)),
            "two_stage_acc_std": float(np.std(two_acc)),
            "two_stage_f1": float(np.mean(two_f1)),
            "acc_delta": float(np.mean(two_acc) - np.mean(three_acc)),
            "f1_delta": float(np.mean(two_f1) - np.mean(three_f1)),
        }
        log.info(
            f"  @{win_sec}s  three={row['three_way_acc']:.4f}±{row['three_way_acc_std']:.4f}  "
            f"two={row['two_stage_acc']:.4f}±{row['two_stage_acc_std']:.4f}  "
            f"Δacc={row['acc_delta']:+.4f}  Δf1={row['f1_delta']:+.4f}"
        )
        log.info("Three-way per-class:\n" + classification_report(y_true_all, three_preds_all, zero_division=0))
        log.info("Two-stage per-class:\n" + classification_report(y_true_all, two_preds_all, zero_division=0))
        rows.append(row)

    return pd.DataFrame(rows)


def run_oracle_regression_test(target_label: str = "whitebox_inference", window_sec: int = 60) -> dict:
    """Train both classifiers on all data EXCEPT target_label, then run them
    over target_label parquets window-by-window and report P(train) stats.

    This is the key Round 5b success metric: median P(train) on
    `whitebox_inference` should be <0.5 for two-stage (was ~0.99 for three-way).
    """
    from classifier.threeway_improved import (
        load_all_data, sliding_windows, get_feature_cols,
    )
    from classifier.twostage import TwoStageClassifier

    df_raw = load_all_data()
    mask_target = df_raw["workload_label"] == target_label
    df_target = df_raw[mask_target].copy()
    df_train = df_raw[~mask_target].copy()
    log.info(
        f"Oracle regression test: target={target_label} "
        f"({df_target['run_id'].nunique()} runs), "
        f"training pool = {df_train['run_id'].nunique()} runs"
    )

    # Window the training pool and fit both classifiers
    df_train_win = sliding_windows(df_train, window_sec=window_sec, stride_sec=window_sec / 2)
    feature_cols = get_feature_cols(df_train_win)
    X_tr = df_train_win[feature_cols].fillna(0).values.astype(np.float32)
    y_tr = df_train_win["threeway_label"].values
    g_tr = df_train_win["run_id"].values

    clf3, le3 = _train_threeway_on_fold(X_tr, y_tr, g_tr)
    clf2 = TwoStageClassifier().fit(X_tr, y_tr, groups=g_tr, feature_cols=feature_cols)

    # Window the target data and collect per-window probabilities
    df_target_win = sliding_windows(df_target, window_sec=window_sec, stride_sec=window_sec / 2)
    if df_target_win.empty:
        raise RuntimeError(f"No windows produced for target label {target_label}")
    # Ensure same feature column alignment
    X_te = df_target_win[feature_cols].fillna(0).values.astype(np.float32)

    p_train_three = _threeway_proba(clf3, le3, X_te, class_name="ml_training")
    p_train_two = clf2.predict_proba(X_te)[:, int(np.where(clf2.classes_ == "ml_training")[0][0])]

    result = {
        "target_label": target_label,
        "n_windows": int(len(X_te)),
        "three_way_P_train_median": float(np.median(p_train_three)),
        "three_way_P_train_mean": float(np.mean(p_train_three)),
        "three_way_P_train_p95": float(np.percentile(p_train_three, 95)),
        "three_way_P_train_gt_0p5_frac": float((p_train_three > 0.5).mean()),
        "two_stage_P_train_median": float(np.median(p_train_two)),
        "two_stage_P_train_mean": float(np.mean(p_train_two)),
        "two_stage_P_train_p95": float(np.percentile(p_train_two, 95)),
        "two_stage_P_train_gt_0p5_frac": float((p_train_two > 0.5).mean()),
    }
    log.info(f"Oracle regression on `{target_label}`:")
    for k, v in result.items():
        log.info(f"  {k}: {v}")
    return result


def write_markdown(cv_df: pd.DataFrame, oracle: dict, out: Path) -> None:
    lines = [
        "# Two-Stage vs Three-Way Classifier — A/B Comparison",
        "",
        "Generated by `classifier/compare_twostage_vs_threeway.py`.",
        "",
        "## 1. Pooled 5-fold CV",
        "",
        "| window | three-way acc | two-stage acc | Δacc | three-way F1 | two-stage F1 | ΔF1 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for _, r in cv_df.iterrows():
        lines.append(
            f"| {int(r['window_sec'])}s "
            f"| {r['three_way_acc']:.4f}±{r['three_way_acc_std']:.4f} "
            f"| {r['two_stage_acc']:.4f}±{r['two_stage_acc_std']:.4f} "
            f"| {r['acc_delta']:+.4f} "
            f"| {r['three_way_f1']:.4f} "
            f"| {r['two_stage_f1']:.4f} "
            f"| {r['f1_delta']:+.4f} |"
        )
    lines += [
        "",
        f"## 2. Oracle regression test on `{oracle['target_label']}`",
        "",
        f"Trained on all data except `{oracle['target_label']}`, then evaluated "
        f"on {oracle['n_windows']} held-out windows.",
        "",
        "| metric | three-way | two-stage |",
        "|---|---:|---:|",
        f"| median P(train) | {oracle['three_way_P_train_median']:.4f} | {oracle['two_stage_P_train_median']:.4f} |",
        f"| mean   P(train) | {oracle['three_way_P_train_mean']:.4f} | {oracle['two_stage_P_train_mean']:.4f} |",
        f"| p95    P(train) | {oracle['three_way_P_train_p95']:.4f} | {oracle['two_stage_P_train_p95']:.4f} |",
        f"| frac P>0.5      | {oracle['three_way_P_train_gt_0p5_frac']:.4f} | {oracle['two_stage_P_train_gt_0p5_frac']:.4f} |",
        "",
        "## Gate checklist (from Round 5b plan)",
        "",
        f"- [ ] Pooled 60s acc(two-stage) ≥ acc(three-way) − 0.5 pp",
        f"- [ ] Median P(train) on `whitebox_inference` < 0.5 for two-stage",
        f"- [ ] Leave-one-strategy-out mean ≥ 98% (see `leave_one_out.csv`)",
        f"- [ ] No per-family regression > 5 pp (see LOO output)",
        "",
    ]
    out.write_text("\n".join(lines), encoding="utf-8")
    log.info(f"Wrote markdown report: {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", nargs="+", type=int, default=[15, 30, 60])
    ap.add_argument("--oracle-target", default="whitebox_inference")
    ap.add_argument("--oracle-window", type=int, default=60)
    ap.add_argument("--skip-cv", action="store_true")
    ap.add_argument("--skip-oracle", action="store_true")
    args = ap.parse_args()

    RESULTS_DIR.mkdir(exist_ok=True)

    if args.skip_cv:
        cv_df = pd.DataFrame()
    else:
        cv_df = run_pooled_cv(tuple(args.windows))
        cv_df.to_csv(RESULTS_DIR / "twostage_vs_threeway.csv", index=False)
        log.info(f"Saved CV results: {RESULTS_DIR / 'twostage_vs_threeway.csv'}")

    if args.skip_oracle:
        oracle = {
            "target_label": args.oracle_target, "n_windows": 0,
            "three_way_P_train_median": float("nan"), "three_way_P_train_mean": float("nan"),
            "three_way_P_train_p95": float("nan"), "three_way_P_train_gt_0p5_frac": float("nan"),
            "two_stage_P_train_median": float("nan"), "two_stage_P_train_mean": float("nan"),
            "two_stage_P_train_p95": float("nan"), "two_stage_P_train_gt_0p5_frac": float("nan"),
        }
    else:
        oracle = run_oracle_regression_test(args.oracle_target, args.oracle_window)

    write_markdown(cv_df, oracle, RESULTS_DIR / "twostage_vs_threeway.md")


if __name__ == "__main__":
    main()
