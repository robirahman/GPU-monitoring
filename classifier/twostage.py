"""
Two-Stage GPU Workload Classifier (Round 5b)
=============================================
Two parallel binary XGBoost classifiers, each answering a different question:

  Stage 1 — "Is this workload training?"
           labels: ml_training  vs  not-training (= ml_inference ∪ other)
           features: full feature set
           This is the *primary* defender detector. `predict_proba_training(X)`
           returns P(training) directly.

  Stage 2 — "If this is ML activity, is it training or inference?"
           labels: ml_training  vs  ml_inference
                   (trained only on rows where true label is training or inference)
           features: level-features (absolute memory/power/clock means,
                     percentiles, min, max) are DROPPED — the decision must
                     rest on shape/periodicity/FFT/autocorrelation only.
           This is the specialist that catches train↔infer flips that memory
           footprint alone can't separate (the Round 5 WB-O oracle flip).

Wire interface (sklearn-compatible):
    clf = TwoStageClassifier()
    clf.fit(X, y, groups=run_ids)
    p_train = clf.predict_proba_training(X)            # shape (n,), the primary P(train)
    p_train_given_ml = clf.predict_proba_train_vs_infer(X)   # shape (n,) in [0,1]
    proba = clf.predict_proba(X)                       # shape (n, 3) for legacy callers
    clf.classes_                                       # ['other','ml_training','ml_inference']

The 3-class `predict_proba` decomposes as:
    P(train) = P1(train)                               # direct
    P(infer) = P1(train) * (1 - P2) / P2               # derived from Bayes
    P(other) = 1 - P(train) - P(infer)
Both stages are trained independently, so the combination is a post-hoc
reconciliation, not a calibrated joint distribution.

Preserves the contract consumed by `classifier/classifier_oracle.py`:
`predict_proba` returns a 3-column array and the ml_training column is at a
stable index resolved via `classes_`.

Usage:
    python -m classifier.twostage --train
    python classifier/twostage.py --train
"""

from __future__ import annotations

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from classifier.threeway_improved import (
    load_all_data,
    sliding_windows,
    get_feature_cols,
    compute_sample_weights,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Feature names whose *level* (mean / min / max / percentile / median) shifts
# sharply between workload types AND between GPU classes. Dropped for Stage 2
# so the train/inference decision rests on periodicity and shape features
# only. Keep derivative/variance/shape terms (`_std`, `_cv`, `_skew`,
# `_kurt`, FFT, autocorrelation, slope) on the same signals — those
# generalise across GPU classes (see Round 5 notes §"Implications for
# classifier feature engineering").
LEVEL_STATS = ("mean", "min", "max", "p25", "p50", "p75", "p95")
LEVEL_SIGNALS = (
    "mem_used_mb",
    "power_draw_w",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "temperature_c",
    "pcie_tx_mbps",
    "pcie_rx_mbps",
    "gpu_utilization_pct",
    "mem_utilization_pct",
)

LEVEL_FEATURE_NAMES = {
    f"{sig}_{stat}" for sig in LEVEL_SIGNALS for stat in LEVEL_STATS
}
# Also drop the `_range` term which is max - min (a level-scaled statistic)
# and the raw PCIe total mean, while keeping `_cv`, `_iqr` (shape).
LEVEL_FEATURE_NAMES |= {f"{sig}_range" for sig in LEVEL_SIGNALS}
LEVEL_FEATURE_NAMES |= {"pcie_total_mean"}


def stage2_feature_mask(feature_cols: list[str]) -> np.ndarray:
    """Boolean mask selecting the shape/periodicity-only feature subset."""
    return np.array([c not in LEVEL_FEATURE_NAMES for c in feature_cols], dtype=bool)


class TwoStageClassifier:
    """Two parallel binary XGBoost classifiers.

    Stage 1 decides ml_training vs NOT-training (= ml_inference ∪ other),
    using the full feature set. This is the primary defender detector.
    Stage 2 decides ml_training vs ml_inference, trained only on rows where
    the true label ∈ {ml_training, ml_inference}, and only on shape /
    periodicity features (level features — absolute memory, power, clock
    means/percentiles — are dropped so the decision cannot lean on
    absolute magnitudes).

    `predict_proba_training(X)` and `predict_proba_train_vs_infer(X)` are
    the clean per-stage outputs. `predict_proba(X)` composes them into a
    3-class distribution for legacy callers (e.g. `classifier_oracle.py`).
    """

    classes_ = np.array(["other", "ml_training", "ml_inference"])

    def __init__(
        self,
        stage1_n_estimators: int = 500,
        stage1_max_depth: int = 6,
        stage2_n_estimators: int = 500,
        stage2_max_depth: int = 5,
        learning_rate: float = 0.05,
        beta: float = 0.9999,
        random_state: int = 42,
    ):
        self.stage1_n_estimators = stage1_n_estimators
        self.stage1_max_depth = stage1_max_depth
        self.stage2_n_estimators = stage2_n_estimators
        self.stage2_max_depth = stage2_max_depth
        self.learning_rate = learning_rate
        self.beta = beta
        self.random_state = random_state

        self._feature_cols: list[str] | None = None
        self._stage2_mask: np.ndarray | None = None
        self._stage1 = None
        self._stage2 = None

    # ------------------------------------------------------------------
    def fit(self, X, y, groups=None, feature_cols: list[str] | None = None):
        import xgboost as xgb

        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if feature_cols is not None:
            self._feature_cols = list(feature_cols)
            self._stage2_mask = stage2_feature_mask(self._feature_cols)
        else:
            # Fall back: don't filter stage 2 features if we weren't told the names.
            self._stage2_mask = np.ones(X.shape[1], dtype=bool)

        # --- Stage 1: ml_training vs NOT-training (inference + other) ---
        y_s1 = np.where(y == "ml_training", 1, 0).astype(np.int32)
        w_s1 = compute_sample_weights(y_s1, groups=groups, beta=self.beta)
        self._stage1 = xgb.XGBClassifier(
            n_estimators=self.stage1_n_estimators,
            max_depth=self.stage1_max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self._stage1.fit(X, y_s1, sample_weight=w_s1)
        log.info(f"Stage 1 fit: {(y_s1 == 1).sum()} training / {(y_s1 == 0).sum()} not-training")

        # --- Stage 2: ml_training vs ml_inference (on ml rows only) ---
        is_ml = y != "other"
        X_s2 = X[is_ml][:, self._stage2_mask]
        y_s2 = np.where(y[is_ml] == "ml_training", 1, 0).astype(np.int32)
        g_s2 = np.asarray(groups)[is_ml] if groups is not None else None
        w_s2 = compute_sample_weights(y_s2, groups=g_s2, beta=self.beta)
        self._stage2 = xgb.XGBClassifier(
            n_estimators=self.stage2_n_estimators,
            max_depth=self.stage2_max_depth,
            learning_rate=self.learning_rate,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=self.random_state,
            n_jobs=-1,
            verbosity=0,
        )
        self._stage2.fit(X_s2, y_s2, sample_weight=w_s2)
        log.info(
            f"Stage 2 fit: {(y_s2 == 1).sum()} train / {(y_s2 == 0).sum()} infer "
            f"on {int(self._stage2_mask.sum())} shape features "
            f"({int((~self._stage2_mask).sum())} level features dropped)"
        )
        return self

    # ------------------------------------------------------------------
    def _assert_fit(self):
        if self._stage1 is None or self._stage2 is None:
            raise RuntimeError("TwoStageClassifier must be fit before calling predict_proba")

    def predict_proba_training(self, X) -> np.ndarray:
        """P(training) from Stage 1 — the primary defender signal."""
        self._assert_fit()
        X = np.asarray(X, dtype=np.float32)
        return self._stage1.predict_proba(X)[:, 1]

    def predict_proba_train_vs_infer(self, X) -> np.ndarray:
        """P(training | workload is training or inference) from Stage 2.

        Defined on the ML-activity subset. Applied to any row the caller
        passes — useful for cross-checks even on rows Stage 1 thinks are
        'not-training', but callers should be aware the stage 2 classifier
        has never seen 'other' rows during training.
        """
        self._assert_fit()
        X = np.asarray(X, dtype=np.float32)
        X_s2 = X[:, self._stage2_mask]
        return self._stage2.predict_proba(X_s2)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        """Legacy 3-class output [P(other), P(training), P(inference)].

        Derived from the two stages via Bayes:
            P(train) = P1(train)
            P(infer) = P1(train) * (1 - P2(train|ml)) / P2(train|ml)
            P(other) = 1 - P(train) - P(infer)
        Clamped to [0, 1] and renormalized row-wise for numerical safety.
        """
        self._assert_fit()
        X = np.asarray(X, dtype=np.float32)

        p_train = self._stage1.predict_proba(X)[:, 1]
        X_s2 = X[:, self._stage2_mask]
        p2 = self._stage2.predict_proba(X_s2)[:, 1]
        p2 = np.clip(p2, 1e-6, 1.0 - 1e-6)

        # P(infer) derivation from P(train) / P(train∪infer) = P2
        p_infer = p_train * (1.0 - p2) / p2
        p_other = 1.0 - p_train - p_infer

        # Numerical clamp + row-wise renorm.
        stacked = np.stack([p_other, p_train, p_infer], axis=1)
        stacked = np.clip(stacked, 0.0, None)
        row_sums = stacked.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums > 0, row_sums, 1.0)
        proba = stacked / row_sums
        return proba

    def predict(self, X) -> np.ndarray:
        proba = self.predict_proba(X)
        return self.classes_[np.argmax(proba, axis=1)]

    @property
    def feature_cols(self) -> list[str] | None:
        return self._feature_cols


# ── Training pipeline ────────────────────────────────────────────────────────


def train_and_evaluate_twostage(window_secs=(15, 30, 60), stride_ratio: float = 0.5) -> pd.DataFrame:
    """Train TwoStageClassifier at multiple window sizes and cross-validate.

    Matches `threeway_improved.train_and_evaluate` cadence so results are
    directly comparable. Saves `classifier/results/twostage_results.csv`.
    """
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import classification_report, accuracy_score, f1_score

    log.info("Loading all telemetry data...")
    df_raw = load_all_data()

    rows = []
    for win_sec in window_secs:
        stride_sec = win_sec * stride_ratio
        log.info(f"\n{'='*60}\nWindow: {win_sec}s, stride: {stride_sec}s\n{'='*60}")

        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
        if df_win.empty:
            log.warning(f"No windows at {win_sec}s")
            continue

        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        groups = df_win["run_id"].values

        log.info(f"Windows: {len(df_win)} | Features: {len(feature_cols)}")
        log.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        unique_classes = np.unique(y)
        if len(unique_classes) < 3:
            log.warning(f"Only {len(unique_classes)} classes found, skipping")
            continue

        run_class_counts = df_win.groupby("run_id")["threeway_label"].first().value_counts()
        n_splits = min(5, int(run_class_counts.min()))
        if n_splits < 2:
            n_splits = 2
        log.info(f"Using {n_splits}-fold cross-validation")

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        accs, f1s = [], []
        all_y_true, all_y_pred = [], []

        for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
            clf = TwoStageClassifier()
            clf.fit(X[tr_idx], y[tr_idx], groups=groups[tr_idx], feature_cols=feature_cols)
            y_pred = clf.predict(X[te_idx])
            acc = accuracy_score(y[te_idx], y_pred)
            f1 = f1_score(y[te_idx], y_pred, average="macro", zero_division=0)
            accs.append(acc)
            f1s.append(f1)
            all_y_true.extend(y[te_idx])
            all_y_pred.extend(y_pred)

        mean_acc = float(np.mean(accs))
        std_acc = float(np.std(accs))
        mean_f1 = float(np.mean(f1s))
        log.info(f"  [TwoStage {win_sec}s] acc={mean_acc:.4f}±{std_acc:.4f}  f1_macro={mean_f1:.4f}")
        log.info("\n" + classification_report(all_y_true, all_y_pred, zero_division=0))

        rows.append({
            "classifier": "TwoStage",
            "window_sec": win_sec,
            "acc_mean": mean_acc,
            "acc_std": std_acc,
            "f1_macro": mean_f1,
        })

    results_df = pd.DataFrame(rows)
    out = PROJECT_ROOT / "classifier" / "results" / "twostage_results.csv"
    out.parent.mkdir(exist_ok=True)
    results_df.to_csv(out, index=False)
    log.info(f"\nResults saved to {out}")
    log.info("\n" + results_df.to_string(index=False))
    return results_df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train + cross-validate")
    ap.add_argument("--windows", nargs="+", type=int, default=[15, 30, 60])
    args = ap.parse_args()
    if args.train:
        train_and_evaluate_twostage(window_secs=tuple(args.windows))
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
