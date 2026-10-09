"""
Two-Stage GPU Workload Classifier
=================================
A pipeline of two binary XGBoost classifiers that decompose the governance
classification task along its natural decision tree:

  Stage 1 — "Is this workload ML?"
           labels: ml_training ∪ ml_inference  vs  other
           features: full feature set
           The ML-vs-non-ML gate. Level signals (peak memory, sustained
           tensor-core utilization, power draw) carry most of the ML-vs-other
           information, so the full feature set is retained here.

  Stage 2 — "Given ML activity, is it training or inference?"
           labels: ml_training  vs  ml_inference
                   (trained only on rows where true label is training or inference)
           features: level-features (absolute memory/power/clock means,
                     percentiles, min, max, range) are DROPPED — the decision
                     rests on shape/periodicity/FFT/autocorrelation only.
           This restriction is the Round 5 WB-O defense: an adversary who
           spoofs memory levels to make training look like inference cannot
           drag the Stage 2 decision, because Stage 2 has never seen the
           levels.

Governance P(train) is the probabilistic composition
    P(train) = P_stage1(ML) × P_stage2(train | ML)
    P(infer) = P_stage1(ML) × (1 − P_stage2(train | ML))
    P(other) = 1 − P_stage1(ML)
with an optional **Stage 2 rescue rule** that reclassifies borderline
Stage-1-negative rows when Stage 2 is highly confident they're training.
The rescue fires when P_stage2(train|ML) > `rescue_tau` AND
P_stage1(ML) > `rescue_floor`: a fraction of the "other" probability mass
(proportional to Stage 2's confidence above `rescue_tau`) moves into
P(train). Defaults `rescue_tau=0.7` / `rescue_floor=0.1` recover held-out
low-utilization training (e.g., B_low_util LOO) without affecting
confident non-ML rows. Set `rescue_tau=1.0` to disable. Each stage is
still an independently calibratable binary head.

Default backend: `random_forest` (Appendix G hyperparameters — 400 trees,
unlimited depth, min-leaf=2, sqrt features). RandomForest is the
report's headline classifier for the three-way baseline and — under
matched LOO evaluation — produces fewer catastrophic single-variant
failures than XGBoost on the Round-5 whitebox families (see
`classifier/results/twostage_backends.md`). Pass `backend="xgboost"`
to reproduce the earlier XGBoost-backed pipeline.

Wire interface (sklearn-compatible):
    clf = TwoStageClassifier()                         # RF backend
    clf = TwoStageClassifier(backend="xgboost")        # XGB backend
    clf.fit(X, y, groups=run_ids, feature_cols=feature_cols)
    p_ml = clf.predict_proba_ml(X)                     # shape (n,), P(ML)
    p_train_given_ml = clf.predict_proba_train_vs_infer(X)   # shape (n,) in [0,1]
    p_train = clf.predict_proba_training(X)            # shape (n,), composed P(train)
    proba = clf.predict_proba(X)                       # shape (n, 3)
    clf.classes_                                       # ['other','ml_training','ml_inference']

Preserves the consumer contract: `predict_proba` returns a 3-column array
and `ml_training` is at a stable index resolved via `classes_`.

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
    workload_group,
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


# Level-stat suffixes, for signals not enumerated in LEVEL_SIGNALS. Mirrors
# multinode_features._LEVEL_SUFFIXES.
_LEVEL_SUFFIXES = ("_mean", "_min", "_max", "_p25", "_p50", "_p75", "_p95", "_range")
# Structure/shape prefixes that must never be treated as levels even though some
# carry a level-looking suffix (cross-rank correlation, lag, cadence).
_STRUCTURE_PREFIXES = ("xr_corr", "xr_lag", "xr_fft", "xr_dip", "xr_active",
                       "xr_n_ranks", "acf")
# Ratios and differences are dimensionless: mem_util_per_gpu_util_p95 carries a
# level suffix but is a RATIO, and it is one of the Round-6 DRAM features that
# the disguised-training evidence rests on. The existing explicit set already
# keeps power_per_util / power_per_mem_util for the same reason, so treating
# _per_ / _minus_ as structure matches current behaviour rather than changing it.
_STRUCTURE_INFIXES = ("_per_", "_minus_")


def stage2_feature_mask(feature_cols: list[str]) -> np.ndarray:
    """Boolean mask selecting the shape/periodicity-only feature subset.

    LEVEL_FEATURE_NAMES is an explicit set built from the nine NVIDIA
    RAW_SIGNALS, so a level statistic on any signal outside that list was being
    handed to Stage 2 as if it were a shape feature -- which would reopen the
    level-spoofing hole Stage 2 exists to close as soon as TPU-native columns
    (hlo_exec_us_p90, hbm_bw_util_pct, tensorcore_util_pct, dcn_collective_lat_us_*)
    enter the vocabulary. The suffix rule closes that generally.
    """
    keep = []
    for c in feature_cols:
        is_level = c in LEVEL_FEATURE_NAMES
        if (not is_level and not c.startswith(_STRUCTURE_PREFIXES)
                and not any(k in c for k in _STRUCTURE_INFIXES)):
            is_level = c.endswith(_LEVEL_SUFFIXES)
        keep.append(not is_level)
    return np.array(keep, dtype=bool)


class TwoStageClassifier:
    """Pipeline of two binary XGBoost classifiers.

    Stage 1 decides ML (ml_training ∪ ml_inference) vs non-ML (other),
    using the full feature set — the ML-vs-other boundary leans heavily on
    level signals (peak memory, sustained utilization, power draw) that
    shouldn't be dropped.

    Stage 2 decides ml_training vs ml_inference, trained only on rows where
    the true label ∈ {ml_training, ml_inference}, and only on shape /
    periodicity features (level features — absolute memory, power, clock
    means/percentiles, range — are dropped so the decision cannot lean on
    absolute magnitudes; this is the WB-O defense).

    Governance composition is exact:
        P(train) = P_stage1(ML) × P_stage2(train | ML)
        P(infer) = P_stage1(ML) × (1 − P_stage2(train | ML))
        P(other) = 1 − P_stage1(ML)
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
        rescue_tau: float = 0.7,
        rescue_floor: float = 0.1,
        backend: str = "random_forest",
        rf_n_estimators: int = 400,
        rf_max_depth: int | None = None,
        rf_min_samples_leaf: int = 2,
        rf_max_features="sqrt",
    ):
        self.stage1_n_estimators = stage1_n_estimators
        self.stage1_max_depth = stage1_max_depth
        self.stage2_n_estimators = stage2_n_estimators
        self.stage2_max_depth = stage2_max_depth
        self.learning_rate = learning_rate
        self.beta = beta
        self.random_state = random_state
        self.rescue_tau = rescue_tau
        self.rescue_floor = rescue_floor
        if backend not in ("xgboost", "random_forest"):
            raise ValueError(f"backend must be 'xgboost' or 'random_forest', got {backend!r}")
        self.backend = backend
        # The RF path used to hardcode these, so max_depth/n_estimators passed to
        # __init__ were silently ignored. Unlimited depth with min_samples_leaf=2
        # is a memorisation machine, which is what a run-grouped CV could not see:
        # 11.6 points of its accuracy came from workload recognition. Defaults
        # below reproduce the old behaviour exactly.
        self.rf_n_estimators = rf_n_estimators
        self.rf_max_depth = rf_max_depth
        self.rf_min_samples_leaf = rf_min_samples_leaf
        self.rf_max_features = rf_max_features

        self._feature_cols: list[str] | None = None
        self._stage2_mask: np.ndarray | None = None
        self._stage1 = None
        self._stage2 = None

    def _make_head(self, n_estimators: int, max_depth: int):
        """Instantiate a single binary classifier for one pipeline stage."""
        if self.backend == "xgboost":
            import xgboost as xgb
            return xgb.XGBClassifier(
                n_estimators=n_estimators,
                max_depth=max_depth,
                learning_rate=self.learning_rate,
                subsample=0.8,
                colsample_bytree=0.8,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=self.random_state,
                n_jobs=-1,
                verbosity=0,
            )
        # random_forest — Appendix G hyperparameters (depth unlimited, sqrt
        # features, min-leaf=2). max_depth/n_estimators/lr from the XGB path
        # are ignored because RF ignores learning rate and RF trees need
        # unlimited depth to match the Appendix G baseline.
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            n_estimators=self.rf_n_estimators,
            max_depth=self.rf_max_depth,
            min_samples_leaf=self.rf_min_samples_leaf,
            max_features=self.rf_max_features,
            random_state=self.random_state,
            n_jobs=-1,
        )

    # ------------------------------------------------------------------
    def fit(self, X, y, groups=None, feature_cols: list[str] | None = None,
            train_active=None):
        """Fit the two-stage head.

        ``train_active`` (optional) is per-window ground truth from the STEP
        logs: 1.0 = a real training step occurred in this window, 0.0 = none,
        NaN = no per-window truth for this run. Where present it overrides the
        run-level label for the Stage-2 (train-vs-infer) target, so the
        inference-only windows of a diluted/PPO run are no longer taught as
        training. Stage 1 (ML vs other) is unaffected — an inference window of a
        training run is still ML.
        """
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y)
        if feature_cols is not None:
            self._feature_cols = list(feature_cols)
            self._stage2_mask = stage2_feature_mask(self._feature_cols)
        else:
            # Fall back: don't filter stage 2 features if we weren't told the names.
            self._stage2_mask = np.ones(X.shape[1], dtype=bool)

        # --- Stage 1: ML (ml_training ∪ ml_inference) vs non-ML (other) ---
        y_s1 = np.where(y != "other", 1, 0).astype(np.int32)
        w_s1 = compute_sample_weights(y_s1, groups=groups, beta=self.beta)
        self._stage1 = self._make_head(self.stage1_n_estimators, self.stage1_max_depth)
        self._stage1.fit(X, y_s1, sample_weight=w_s1)
        log.info(f"Stage 1 ({self.backend}) fit: {(y_s1 == 1).sum()} ML / {(y_s1 == 0).sum()} non-ML")

        # --- Stage 2: ml_training vs ml_inference (on ml rows only) ---
        is_ml = y != "other"
        X_s2 = X[is_ml][:, self._stage2_mask]
        y_s2 = np.where(y[is_ml] == "ml_training", 1, 0).astype(np.int32)
        if train_active is not None:
            # Per-window truth overrides the run-level label where available.
            ta = np.asarray(train_active, dtype=float)[is_ml]
            known = ~np.isnan(ta)
            y_s2[known] = (ta[known] > 0.5).astype(np.int32)
            if known.any():
                log.info(f"Stage 2: {int(known.sum())} windows use per-window "
                         f"train/infer truth (vs run-level label)")
        g_s2 = np.asarray(groups)[is_ml] if groups is not None else None
        w_s2 = compute_sample_weights(y_s2, groups=g_s2, beta=self.beta)
        self._stage2 = self._make_head(self.stage2_n_estimators, self.stage2_max_depth)
        self._stage2.fit(X_s2, y_s2, sample_weight=w_s2)
        log.info(
            f"Stage 2 ({self.backend}) fit: {(y_s2 == 1).sum()} train / {(y_s2 == 0).sum()} infer "
            f"on {int(self._stage2_mask.sum())} shape features "
            f"({int((~self._stage2_mask).sum())} level features dropped)"
        )
        return self

    # ------------------------------------------------------------------
    def _assert_fit(self):
        if self._stage1 is None or self._stage2 is None:
            raise RuntimeError("TwoStageClassifier must be fit before calling predict_proba")

    def predict_proba_ml(self, X) -> np.ndarray:
        """P(ML) from Stage 1 — the front-gate probability."""
        self._assert_fit()
        X = np.asarray(X, dtype=np.float32)
        return self._stage1.predict_proba(X)[:, 1]

    def predict_proba_train_vs_infer(self, X) -> np.ndarray:
        """P(training | ML) from Stage 2.

        Defined on the ML-activity subset. Callable on any row, but Stage 2
        has never seen 'other' rows during training, so the value is only
        operationally meaningful after Stage 1 has admitted the row.
        """
        self._assert_fit()
        X = np.asarray(X, dtype=np.float32)
        X_s2 = X[:, self._stage2_mask]
        return self._stage2.predict_proba(X_s2)[:, 1]

    def _apply_rescue(self, p_ml, p_t_given_ml, p_train, p_infer, p_other):
        """Move "other" probability mass into "training" when Stage 2 is
        highly confident this is training AND Stage 1 is not confidently
        non-ML. Preserves the sum-to-1 invariant.
        """
        if self.rescue_tau >= 1.0:
            return p_train, p_infer, p_other
        denom = max(1.0 - self.rescue_tau, 1e-6)
        rescue_strength = np.clip((p_t_given_ml - self.rescue_tau) / denom, 0.0, 1.0)
        gate = (p_ml > self.rescue_floor).astype(np.float32)
        rescue_strength = rescue_strength * gate
        rescue_mass = rescue_strength * p_other
        return p_train + rescue_mass, p_infer, p_other - rescue_mass

    def predict_proba_training(self, X) -> np.ndarray:
        """Composed governance P(train), with rescue applied."""
        return self.predict_proba(X)[:, 1]

    def predict_proba(self, X) -> np.ndarray:
        """3-class output [P(other), P(training), P(inference)].

        Pipeline composition:
            P(train) = P_stage1(ML) × P_stage2(train | ML)
            P(infer) = P_stage1(ML) × (1 − P_stage2(train | ML))
            P(other) = 1 − P_stage1(ML)
        Plus Stage 2 rescue (see `_apply_rescue`): when Stage 2 is highly
        confident training-shape AND Stage 1 isn't confidently non-ML,
        mass migrates from P(other) to P(train). Sum-to-1 preserved.
        """
        self._assert_fit()
        X = np.asarray(X, dtype=np.float32)

        p_ml = self._stage1.predict_proba(X)[:, 1]
        X_s2 = X[:, self._stage2_mask]
        p_t_given_ml = self._stage2.predict_proba(X_s2)[:, 1]

        p_train = p_ml * p_t_given_ml
        p_infer = p_ml * (1.0 - p_t_given_ml)
        p_other = 1.0 - p_ml

        p_train, p_infer, p_other = self._apply_rescue(
            p_ml, p_t_given_ml, p_train, p_infer, p_other
        )

        return np.stack([p_other, p_train, p_infer], axis=1)

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

        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec,
                                 run_level_mode="causal")
        if df_win.empty:
            log.warning(f"No windows at {win_sec}s")
            continue

        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        # Report BOTH groupings. The GAP between them is the quantity of
        # interest: it is how much of the score comes from recognising a
        # workload already seen rather than generalising to a new one. Selection
        # is on the workload-grouped number -- that is the deployment question.
        grouping = {
            "run": df_win["run_id"].values,
            "workload": workload_group(df_win["workload_label"]),
        }

        log.info(f"Windows: {len(df_win)} | Features: {len(feature_cols)}")
        log.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        unique_classes = np.unique(y)
        if len(unique_classes) < 3:
            log.warning(f"Only {len(unique_classes)} classes found, skipping")
            continue

        scored = {}
        for gname, groups in grouping.items():
            gser = pd.Series(groups)
            per_class_groups = (df_win.assign(_g=gser.values)
                                .groupby("threeway_label")["_g"].nunique())
            n_splits = max(2, min(5, int(per_class_groups.min())))
            log.info(f"Using {n_splits}-fold CV grouped by {gname} "
                     f"({gser.nunique()} groups; smallest class has "
                     f"{int(per_class_groups.min())})")
            cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
            accs, f1s, all_y_true, all_y_pred = [], [], [], []
            for tr_idx, te_idx in cv.split(X, y, groups):
                clf = TwoStageClassifier()
                clf.fit(X[tr_idx], y[tr_idx], groups=groups[tr_idx],
                        feature_cols=feature_cols)
                y_pred = clf.predict(X[te_idx])
                accs.append(accuracy_score(y[te_idx], y_pred))
                f1s.append(f1_score(y[te_idx], y_pred, average="macro", zero_division=0))
                all_y_true.extend(y[te_idx]); all_y_pred.extend(y_pred)
            scored[gname] = (float(np.mean(accs)), float(np.std(accs)), float(np.mean(f1s)))
            log.info(f"  [TwoStage {win_sec}s / {gname}-grouped] "
                     f"acc={scored[gname][0]:.4f}±{scored[gname][1]:.4f} "
                     f"f1_macro={scored[gname][2]:.4f}")
            log.info("\n" + classification_report(all_y_true, all_y_pred, zero_division=0))

        mean_acc, std_acc, mean_f1 = scored["workload"]
        run_acc = scored["run"][0]
        log.info(f"  IDENTITY GAP at {win_sec}s: run-grouped {run_acc:.4f} vs "
                 f"workload-grouped {mean_acc:.4f} "
                 f"(={run_acc - mean_acc:+.4f} from workload recognition)")

        rows.append({
            "acc_run_grouped": run_acc,
            "identity_gap": run_acc - mean_acc,
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
