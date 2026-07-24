"""Dump per-window predictions for CV and LOO evaluations, across feature variants.

The paper's result CSVs only ever stored aggregates (mean accuracy, per-class
window counts). Reviewers ask for ROC/PR curves, false-positive rates,
confusion matrices, and run-level aggregation — none of which can be recovered
from aggregates. This script produces the missing raw table: one row per test
window with the true label, predicted label, and class probabilities.

Each evaluation is run under three ``run_level_mode`` feature variants so the
data-leakage fix can be quantified before/after:
  - ``orig``   : whole-run features attached to every window (the leaky
                 pre-fix behavior).
  - ``none``   : run-level features dropped (leak-free floor).
  - ``causal`` : run-level features recomputed per window from past+current
                 samples only.

Evaluations (mirroring the existing scripts exactly):
  - ``binary_cv``     : RandomForest, binary (ml_training vs rest), grouped
                        5-fold CV  (retrain_binary.py, the 98.21% headline).
  - ``threeway_cv``   : XGBoost three-way, grouped 5-fold CV
                        (threeway_improved.train_and_evaluate).
  - ``twostage_cv``   : TwoStageClassifier (RF backend), grouped 5-fold CV
                        (twostage.train_and_evaluate_twostage — deployed model).
  - ``loo_twostage``  : leave-one-strategy-out, TwoStage RF backend
                        (leave_one_strategy_out.py --classifier twostage).
  - ``loo_threeway``  : leave-one-strategy-out, XGBoost three-way.

Outputs (under classifier/results/):
  - window_preds/<eval>__<variant>__w<sec>.parquet   one file per finished block
  - window_predictions.parquet                       concatenation of all blocks
  - leakage_before_after.csv                         acc/F1/detection summary

Usage:
  python -m classifier.dump_window_predictions                       # everything
  python -m classifier.dump_window_predictions --evals binary_cv     # one eval
  python -m classifier.dump_window_predictions --variants orig none  # subset
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
import xgboost as xgb

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols, compute_sample_weights,
)
from classifier.twostage import TwoStageClassifier
from classifier.leave_one_strategy_out import STRATEGY_FAMILIES

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
BLOCK_DIR = RESULTS_DIR / "window_preds"
BLOCK_DIR.mkdir(parents=True, exist_ok=True)

VARIANTS = ["orig", "none", "causal"]
META_COLS = ["run_id", "workload_label", "threeway_label", "gpu_name",
             "window_start", "window_end"]
THREEWAY = np.array(["other", "ml_training", "ml_inference"])


def _meta(df_win: pd.DataFrame, idx=None) -> pd.DataFrame:
    """Extract the per-window metadata columns (optionally a row subset)."""
    sub = df_win if idx is None else df_win.iloc[idx]
    out = pd.DataFrame(index=range(len(sub)))
    for c in META_COLS:
        out[c] = sub[c].values if c in sub.columns else None
    return out


def _proba_frame(clf_classes, proba: np.ndarray) -> dict:
    """Map a (n, k) probability matrix onto p_train/p_infer/p_other columns."""
    cls = list(clf_classes)
    col = {name: proba[:, cls.index(name)] if name in cls else np.full(len(proba), np.nan)
           for name in THREEWAY}
    return {"p_other": col["other"], "p_train": col["ml_training"], "p_infer": col["ml_inference"]}


# ── Window construction (cached per (variant, window_sec)) ────────────────────

_WIN_CACHE: dict = {}


def get_windows(df_raw, variant, window_sec, stride_sec):
    key = (variant, window_sec)
    if key not in _WIN_CACHE:
        log.info(f"  windowing variant={variant} w={window_sec}s stride={stride_sec}s ...")
        _WIN_CACHE[key] = sliding_windows(df_raw, window_sec, stride_sec, run_level_mode=variant)
    return _WIN_CACHE[key]


# ── Evaluation blocks ─────────────────────────────────────────────────────────

def run_binary_cv(df_win, feature_cols, variant, window_sec):
    """RandomForest binary CV, out-of-fold per-window predictions."""
    X = df_win[feature_cols].fillna(0).values.astype(np.float32)
    y = (df_win["threeway_label"].values == "ml_training").astype(int)
    groups = df_win["run_id"].values
    run_first = df_win.groupby("run_id")["threeway_label"].first()
    n_splits = min(5, int((run_first == "ml_training").astype(int).value_counts().min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups)):
        clf = RandomForestClassifier(
            n_estimators=400, max_depth=None, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(X[tr], y[tr])
        proba = clf.predict_proba(X[te])
        p_train = proba[:, list(clf.classes_).index(1)]
        pred = clf.predict(X[te])
        m = _meta(df_win, te)
        m["y_true"] = np.where(y[te] == 1, "ml_training", "non_training")
        m["y_pred"] = np.where(pred == 1, "ml_training", "non_training")
        m["p_train"] = p_train
        m["p_infer"] = np.nan
        m["p_other"] = np.nan
        m["fold"] = fold_idx
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    out["eval_name"] = "binary_cv_rf"
    return out


def run_threeway_cv(df_win, feature_cols, variant, window_sec):
    """XGBoost three-way CV, out-of-fold per-window predictions."""
    X = df_win[feature_cols].fillna(0).values.astype(np.float32)
    y = df_win["threeway_label"].values
    groups = df_win["run_id"].values
    le = LabelEncoder().fit(y)
    y_enc = le.transform(y)
    run_first = df_win.groupby("run_id")["threeway_label"].first()
    n_splits = min(5, int(run_first.value_counts().min()))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups)):
        clf = xgb.XGBClassifier(
            n_estimators=500, max_depth=6, learning_rate=0.05,
            subsample=0.8, colsample_bytree=0.8,
            eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
        clf.fit(X[tr], y_enc[tr], sample_weight=compute_sample_weights(y_enc[tr]))
        proba = clf.predict_proba(X[te])
        pred = le.inverse_transform(clf.predict(X[te]))
        m = _meta(df_win, te)
        m["y_true"] = y[te]
        m["y_pred"] = pred
        for k, v in _proba_frame(le.classes_, proba).items():
            m[k] = v
        m["fold"] = fold_idx
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    out["eval_name"] = "threeway_cv_xgb"
    return out


def run_twostage_cv(df_win, feature_cols, variant, window_sec):
    """TwoStageClassifier (RF backend) CV, out-of-fold per-window predictions."""
    X = df_win[feature_cols].fillna(0).values.astype(np.float32)
    y = df_win["threeway_label"].values
    groups = df_win["run_id"].values
    run_first = df_win.groupby("run_id")["threeway_label"].first()
    n_splits = max(2, min(5, int(run_first.value_counts().min())))
    cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

    rows = []
    for fold_idx, (tr, te) in enumerate(cv.split(X, y, groups)):
        clf = TwoStageClassifier().fit(X[tr], y[tr], groups=groups[tr], feature_cols=feature_cols)
        proba = clf.predict_proba(X[te])
        pred = clf.predict(X[te])
        m = _meta(df_win, te)
        m["y_true"] = y[te]
        m["y_pred"] = pred
        for k, v in _proba_frame(clf.classes_, proba).items():
            m[k] = v
        m["fold"] = fold_idx
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    out["eval_name"] = "twostage_cv_rf"
    return out


def _loo(df_win, feature_cols, backend_kind):
    """Shared leave-one-strategy-out loop. backend_kind in {twostage, threeway}."""
    all_adv = set()
    for labels in STRATEGY_FAMILIES.values():
        all_adv.update(labels)
    clean_mask = ~df_win["workload_label"].isin(all_adv)
    le = LabelEncoder().fit(sorted(df_win.loc[clean_mask, "threeway_label"].unique()))

    rows = []
    for family, held_labels in STRATEGY_FAMILIES.items():
        other_adv = all_adv - set(held_labels)
        train_mask = clean_mask | df_win["workload_label"].isin(other_adv)
        test_mask = df_win["workload_label"].isin(held_labels)
        if test_mask.sum() == 0:
            continue

        Xtr = df_win.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
        ytr = df_win.loc[train_mask, "threeway_label"].values
        gtr = df_win.loc[train_mask, "run_id"].values
        df_te = df_win.loc[test_mask]
        Xte = df_te[feature_cols].fillna(0).values.astype(np.float32)

        if backend_kind == "twostage":
            clf = TwoStageClassifier(backend="random_forest").fit(
                Xtr, ytr, groups=gtr, feature_cols=feature_cols)
            proba = clf.predict_proba(Xte)
            pred = clf.predict(Xte)
            pf = _proba_frame(clf.classes_, proba)
        else:
            clf = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0)
            clf.fit(Xtr, le.transform(ytr), sample_weight=compute_sample_weights(le.transform(ytr)))
            proba = clf.predict_proba(Xte)
            pred = le.inverse_transform(clf.predict(Xte))
            pf = _proba_frame(le.classes_, proba)

        m = _meta(df_te.reset_index(drop=True))
        m["y_true"] = df_te["threeway_label"].values
        m["y_pred"] = pred
        for k, v in pf.items():
            m[k] = v
        m["fold"] = -1
        m["held_out_family"] = family
        rows.append(m)
    out = pd.concat(rows, ignore_index=True)
    out["eval_name"] = f"loo_{backend_kind}"
    return out


EVALS = {
    "binary_cv":    (run_binary_cv,   [(30, 15), (15, 7)]),
    "threeway_cv":  (run_threeway_cv, [(30, 15)]),
    "twostage_cv":  (run_twostage_cv, [(30, 15)]),
    "loo_twostage": (lambda d, f, v, w: _loo(d, f, "twostage"), [(60, 30)]),
    "loo_threeway": (lambda d, f, v, w: _loo(d, f, "threeway"), [(60, 30)]),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--evals", nargs="+", default=list(EVALS), choices=list(EVALS))
    ap.add_argument("--variants", nargs="+", default=VARIANTS, choices=VARIANTS)
    ap.add_argument("--force", action="store_true", help="recompute finished blocks")
    args = ap.parse_args()

    log.info("Loading all telemetry data (once)...")
    df_raw = load_all_data()
    log.info(f"Loaded {len(df_raw)} rows, {df_raw['run_id'].nunique()} runs")

    for eval_name in args.evals:
        fn, window_specs = EVALS[eval_name]
        for window_sec, stride_sec in window_specs:
            for variant in args.variants:
                block_path = BLOCK_DIR / f"{eval_name}__{variant}__w{window_sec}.parquet"
                if block_path.exists() and not args.force:
                    log.info(f"[skip] {block_path.name} exists")
                    continue
                log.info(f"[run ] {eval_name}  variant={variant}  w={window_sec}s")
                df_win = get_windows(df_raw, variant, window_sec, stride_sec)
                feature_cols = get_feature_cols(df_win)
                block = fn(df_win, feature_cols, variant, window_sec)
                block["feature_variant"] = variant
                block["window_sec"] = window_sec
                block.to_parquet(block_path, index=False)
                log.info(f"[save] {block_path.name}  ({len(block)} rows, "
                         f"{len(feature_cols)} features)")
            # free the window cache for this window_sec once all variants done
            for v in args.variants:
                _WIN_CACHE.pop((v, window_sec), None)

    # Concatenate all finished blocks and build the summary.
    blocks = sorted(BLOCK_DIR.glob("*.parquet"))
    if not blocks:
        log.warning("No blocks produced.")
        return
    allpred = pd.concat([pd.read_parquet(b) for b in blocks], ignore_index=True)
    allpred.to_parquet(RESULTS_DIR / "window_predictions.parquet", index=False)
    log.info(f"Wrote window_predictions.parquet ({len(allpred)} rows)")

    build_summary(allpred)


def build_summary(allpred: pd.DataFrame):
    """Accuracy/F1 for CV evals; whitebox detection for LOO — pivoted by variant."""
    summ = []
    for (eval_name, window_sec), g in allpred.groupby(["eval_name", "window_sec"]):
        for variant, gv in g.groupby("feature_variant"):
            row = {"eval_name": eval_name, "window_sec": window_sec, "variant": variant,
                   "n_windows": len(gv)}
            if eval_name.startswith("loo"):
                # detection = fraction of true-training windows predicted training
                for fam, gf in gv.groupby("held_out_family"):
                    tt = gf[gf["y_true"] == "ml_training"]
                    row[f"det[{fam}]"] = (tt["y_pred"] == "ml_training").mean() if len(tt) else np.nan
            else:
                yt, yp = gv["y_true"].values, gv["y_pred"].values
                row["accuracy"] = accuracy_score(yt, yp)
                row["f1_macro"] = f1_score(yt, yp, average="macro", zero_division=0)
            summ.append(row)
    summ_df = pd.DataFrame(summ)
    out = RESULTS_DIR / "leakage_before_after.csv"
    summ_df.to_csv(out, index=False)
    log.info(f"Wrote {out}")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        log.info("\n" + summ_df.to_string(index=False))


if __name__ == "__main__":
    main()
