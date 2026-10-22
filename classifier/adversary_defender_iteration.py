"""
Adversary-Defender Iteration (All GPUs)
=======================================

Round 1 (Baseline):  Train on clean data → evaluate on adversarial
Round 2 (Hardened):  Retrain with adversarial examples → re-evaluate
Round 3 (Features):  Add evasion-resistant features → re-evaluate

Includes single-GPU adversarial (A-N), composite sweep, and DDP adversarial
strategies from datacenter runs (H200, B200, A100).

Produces (depending on --classifier):
  - classifier/results/adv_defender_rounds.csv          (threeway, default)
  - classifier/results/adv_defender_rounds_twostage.csv (twostage)
  - classifier/results/evasion_tax_5080.csv             (threeway only; composite sweep)

Usage:
    python -m classifier.adversary_defender_iteration
    python -m classifier.adversary_defender_iteration --classifier twostage
"""

import argparse
import logging
import warnings
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols, extract_enhanced_features,
    compute_sample_weights,
    RAW_SIGNALS, threeway_label, TRAINING_LABELS, INFERENCE_LABELS, OTHER_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# ── Additional evasion-resistant features ────────────────────────────────────

def _physics_features(power, util, mem) -> dict:
    """The five Round-3 physical invariants over one observation prefix.

    Caller decides what the prefix is; this only computes. Keeping it separate
    is what makes the causal slice above auditable.
    """
    if len(power) == 0:
        return dict(cumulative_energy_wh=0.0, power_mem_correlation=0.0,
                    util_bimodal_score=0.0, power_efficiency_ratio=0.0,
                    sustained_compute_frac=0.0)
    feats = {}

    # Energy = integral of power over time. If you did real compute, energy was
    # consumed, and no amount of shaping hides it.
    feats["cumulative_energy_wh"] = float(power.sum()) / 3600.0

    # Training: memory high AND power high, correlated. Pre-allocation plus
    # idle padding gives high memory with low power, which is unnatural.
    if len(power) > 5 and power.std() > 0 and mem.std() > 0:
        corr = np.corrcoef(power, mem)[0, 1]
        feats["power_mem_correlation"] = float(corr) if np.isfinite(corr) else 0.0
    else:
        feats["power_mem_correlation"] = 0.0

    # Idle-padded evasion is bimodal (bursts of compute separated by zero);
    # real training is mostly sustained.
    feats["util_bimodal_score"] = (float((util < 15).mean() * (util > 70).mean())
                                   if len(util) > 10 else 0.0)

    # Watts per % utilisation. Real training is consistent; throttled evasion
    # is artificially low.
    mean_util, mean_power = float(util.mean()), float(power.mean())
    feats["power_efficiency_ratio"] = mean_power / mean_util if mean_util > 1 else mean_power

    # Fraction of samples inside a run of >50% utilisation at least 10 s long.
    if len(util) > 10:
        sustained = streak = 0
        for hot in util > 50:
            streak = streak + 1 if hot else 0
            if streak >= 10:
                sustained += 1
        feats["sustained_compute_frac"] = sustained / max(len(util), 1)
    else:
        feats["sustained_compute_frac"] = 0.0
    return feats


EVASION_RESISTANT_FEATURES = ("cumulative_energy_wh", "power_mem_correlation",
                              "util_bimodal_score", "power_efficiency_ratio",
                              "sustained_compute_frac")


def add_evasion_resistant_features(df_win: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add features that are hard to spoof without reducing real throughput.

    Each window sees its run from the start up to its own ``window_end`` and no
    further, which is what a live monitor has and mirrors
    ``sliding_windows(run_level_mode="causal")``.

    Until 2026-10-22 this stamped the WHOLE run onto every window: ``run_raw``
    was never sliced, so ``cumulative_energy_wh`` was total run energy, a
    quantity only knowable once the run is over. The 2026-08 causal sweep fixed
    ``run_level_features`` and missed this sibling, and ``eval_catchup`` strips
    the fixed plateau features only to re-admit these -- so the leak survived on
    exactly the path that claims to control for it. tests/test_causality.py
    pins the property.
    """
    missing = [c for c in ("run_id", "window_end") if c not in df_win.columns]
    if missing:
        raise ValueError(f"causal evasion-resistant features need {missing} on the window frame")

    values = {name: {} for name in EVASION_RESISTANT_FEATURES}
    for run_id, raw in df_raw.groupby("run_id", sort=False):
        index = df_win.index[df_win["run_id"] == run_id]
        if raw.empty or len(index) == 0:
            continue
        raw = raw.sort_values("ts")
        ts = raw["ts"].to_numpy(dtype=float)
        power = raw["power_draw_w"].to_numpy(dtype=float)
        util = raw["gpu_utilization_pct"].to_numpy(dtype=float)
        mem = raw["mem_used_mb"].to_numpy(dtype=float)
        # One prefix length per window: samples strictly before its end.
        observed = np.searchsorted(ts, df_win.loc[index, "window_end"].to_numpy(dtype=float),
                                   side="left")
        for idx, n_obs in zip(index, observed):
            for name, value in _physics_features(power[:n_obs], util[:n_obs], mem[:n_obs]).items():
                values[name][idx] = value

    for name in EVASION_RESISTANT_FEATURES:
        mapped = pd.Series(df_win.index.map(values[name]), index=df_win.index, dtype="float64")
        df_win[name] = mapped.fillna(0.0)
    return df_win


# ── Core analysis ────────────────────────────────────────────────────────────

def _train_predictor(variant, X, y_str, groups, feature_cols, xgb_params, le):
    """Fit a classifier and return a predict_str callable (X) -> str labels."""
    if variant == "threeway":
        import xgboost as xgb
        y_enc = le.transform(y_str)
        w = compute_sample_weights(y_enc)
        log.info(f"  Class-weighted: {len(y_enc)} windows (all data retained)")
        clf = xgb.XGBClassifier(**xgb_params)
        clf.fit(X, y_enc, sample_weight=w)
        return clf, (lambda X_te: le.inverse_transform(clf.predict(X_te)))
    else:
        from classifier.twostage import TwoStageClassifier
        log.info(f"  Two-stage: {len(y_str)} windows (all data retained)")
        clf = TwoStageClassifier().fit(
            X, y_str, groups=groups, feature_cols=feature_cols,
        )
        return clf, (lambda X_te: clf.predict(X_te))


def run_analysis(variant="threeway", holdout_prefixes=()):
    from sklearn.metrics import accuracy_score, classification_report
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb

    log.info("=" * 70)
    log.info(f"ADVERSARY-DEFENDER ITERATION — ALL GPUs (classifier={variant})")
    log.info("=" * 70)

    # ── Load all data ────────────────────────────────────────────────────
    log.info("Loading all telemetry data...")
    df_raw = load_all_data()
    log.info(f"  Total rows: {len(df_raw):,}")
    log.info(f"  Unique workloads: {df_raw['workload_label'].nunique()}")
    log.info(f"  Unique runs: {df_raw['run_id'].nunique()}")

    # Categorize runs
    all_labels = df_raw["workload_label"].unique()
    composite_labels = sorted(l for l in all_labels if "adversarial_composite" in str(l))
    strategy_labels = sorted(l for l in all_labels
                             if "adversarial" in str(l) and "composite" not in str(l))
    adversarial_labels = set(composite_labels) | set(strategy_labels)
    clean_labels = set(all_labels) - adversarial_labels

    log.info(f"  Clean workloads: {len(clean_labels)}")
    log.info(f"  Adversarial strategies: {strategy_labels}")
    log.info(f"  Composite sweep labels: {composite_labels}")

    # ── Extract windows ──────────────────────────────────────────────────
    WIN_SEC = 60
    STRIDE = 30
    log.info(f"\nExtracting {WIN_SEC}s windows (stride {STRIDE}s)...")

    df_win = sliding_windows(df_raw, window_sec=WIN_SEC, stride_sec=STRIDE)
    if df_win.empty:
        log.error("No windows generated!")
        return

    feature_cols = get_feature_cols(df_win)
    log.info(f"  Windows: {len(df_win)} | Features: {len(feature_cols)}")

    # ── Split clean vs adversarial ───────────────────────────────────────
    clean_mask = ~df_win["workload_label"].isin(adversarial_labels)
    strategy_mask = df_win["workload_label"].isin(strategy_labels)
    composite_mask = df_win["workload_label"].isin(composite_labels)

    df_clean = df_win[clean_mask].copy()
    df_strategies = df_win[strategy_mask].copy()
    df_composite = df_win[composite_mask].copy()

    log.info(f"  Clean windows: {len(df_clean)}")
    log.info(f"  Strategy A/B/D windows: {len(df_strategies)}")
    log.info(f"  Composite windows: {len(df_composite)}")

    # ── Round 1: Baseline classifier (clean data only) ───────────────────
    log.info("\n" + "=" * 70)
    log.info("ROUND 1: BASELINE (trained on clean data only)")
    log.info("=" * 70)

    X_clean = df_clean[feature_cols].fillna(0).values.astype(np.float32)
    y_clean = df_clean["threeway_label"].values
    groups_clean = df_clean["run_id"].values

    le = LabelEncoder()
    le.fit(sorted(set(y_clean)))

    xgb_params = dict(
        n_estimators=500, max_depth=6, learning_rate=0.05,
        subsample=0.8, colsample_bytree=0.8,
        eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
    )

    # Cross-validated clean accuracy (threeway only — CV is identical across
    # variants so we skip it for the two-stage rerun)
    if variant == "threeway":
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        y_enc = le.transform(y_clean)
        cv_accs = []
        for tr_idx, te_idx in cv.split(X_clean, y_enc, groups_clean):
            w_fold = compute_sample_weights(y_enc[tr_idx])
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(X_clean[tr_idx], y_enc[tr_idx], sample_weight=w_fold)
            cv_accs.append(accuracy_score(y_enc[te_idx], clf.predict(X_clean[te_idx])))
        log.info(f"  Clean 5-fold CV accuracy: {np.mean(cv_accs):.4f} ± {np.std(cv_accs):.4f}")

    # Train final baseline on all clean data
    clf_baseline, predict_baseline = _train_predictor(
        variant, X_clean, y_clean, groups_clean, feature_cols, xgb_params, le,
    )

    # Evaluate baseline on adversarial strategies
    results = []
    for label in sorted(set(strategy_labels) | set(composite_labels)):
        mask = df_win["workload_label"] == label
        if mask.sum() == 0:
            continue
        X_adv = df_win.loc[mask, feature_cols].fillna(0).values.astype(np.float32)
        y_pred = predict_baseline(X_adv)
        det_rate = (y_pred == "ml_training").mean()
        n_win = len(X_adv)

        # Parse idle fraction for composite
        idle_frac = None
        if "composite" in label:
            try:
                idle_frac = int(label.split("_")[-1]) / 100.0
            except ValueError:
                pass

        results.append({
            "round": "R1_baseline",
            "workload": label,
            "idle_fraction": idle_frac,
            "detection_rate": det_rate,
            "n_windows": n_win,
            "classified_training": (y_pred == "ml_training").sum(),
            "classified_inference": (y_pred == "ml_inference").sum(),
            "classified_other": (y_pred == "other").sum(),
        })
        tag = f"idle={idle_frac:.0%}" if idle_frac is not None else "strategy"
        log.info(f"  {label:42s} [{tag:>10s}]  detected={det_rate:5.1%}  (n={n_win})")

    # ── Round 2: Hardened classifier (retrained with adversarial examples) ─
    log.info("\n" + "=" * 70)
    log.info("ROUND 2: HARDENED (retrained including adversarial examples)")
    log.info("=" * 70)

    # Include strategy A/B/D AND composite 0-50% as training examples
    # (these are still training, just disguised)
    # Exclude composite 70-95% to test whether the hardened classifier
    # generalizes to unseen evasion levels
    hardening_labels = set(strategy_labels)
    for label in composite_labels:
        try:
            frac = int(label.split("_")[-1])
            if frac <= 50:
                hardening_labels.add(label)
        except ValueError:
            pass

    # Hold out chosen families from the hardened TRAINING set (they are still
    # evaluated below) — measures whether the defender generalizes to an evasion
    # family it has never trained on. E.g. --holdout-prefixes adversarial_mm_
    # keeps every memory-minimized evader out of Round-2/3 training.
    if holdout_prefixes:
        held = {l for l in hardening_labels if any(l.startswith(p) for p in holdout_prefixes)}
        hardening_labels -= held
        log.info(f"  Holding {len(held)} families out of hardened training: {sorted(held)}")

    hardening_mask = df_win["workload_label"].isin(hardening_labels)
    df_hardened_train = pd.concat([df_clean, df_win[hardening_mask]])

    X_hard = df_hardened_train[feature_cols].fillna(0).values.astype(np.float32)
    y_hard = df_hardened_train["threeway_label"].values
    groups_hard = df_hardened_train["run_id"].values

    log.info(f"  Hardened training set: {len(X_hard)} windows "
             f"({len(df_clean)} clean + {hardening_mask.sum()} adversarial)")

    # Cross-validated hardened accuracy (threeway only)
    if variant == "threeway":
        y_hard_enc = le.transform(y_hard)
        cv_accs_hard = []
        cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        for tr_idx, te_idx in cv.split(X_hard, y_hard_enc, groups_hard):
            w_fold = compute_sample_weights(y_hard_enc[tr_idx])
            clf = xgb.XGBClassifier(**xgb_params)
            clf.fit(X_hard[tr_idx], y_hard_enc[tr_idx], sample_weight=w_fold)
            cv_accs_hard.append(accuracy_score(y_hard_enc[te_idx], clf.predict(X_hard[te_idx])))
        log.info(f"  Hardened 5-fold CV accuracy: {np.mean(cv_accs_hard):.4f} ± {np.std(cv_accs_hard):.4f}")

    # Train final hardened classifier
    clf_hardened, predict_hardened = _train_predictor(
        variant, X_hard, y_hard, groups_hard, feature_cols, xgb_params, le,
    )

    # Evaluate hardened on ALL adversarial (including held-out composite 70-95%)
    for label in sorted(set(strategy_labels) | set(composite_labels)):
        mask = df_win["workload_label"] == label
        if mask.sum() == 0:
            continue
        X_adv = df_win.loc[mask, feature_cols].fillna(0).values.astype(np.float32)
        y_pred = predict_hardened(X_adv)
        det_rate = (y_pred == "ml_training").mean()
        n_win = len(X_adv)

        idle_frac = None
        if "composite" in label:
            try:
                idle_frac = int(label.split("_")[-1]) / 100.0
            except ValueError:
                pass

        held_out = "composite" in label and idle_frac is not None and idle_frac > 0.50
        results.append({
            "round": "R2_hardened",
            "workload": label,
            "idle_fraction": idle_frac,
            "detection_rate": det_rate,
            "n_windows": n_win,
            "classified_training": (y_pred == "ml_training").sum(),
            "classified_inference": (y_pred == "ml_inference").sum(),
            "classified_other": (y_pred == "other").sum(),
        })
        marker = " *HELD-OUT*" if held_out else ""
        tag = f"idle={idle_frac:.0%}" if idle_frac is not None else "strategy"
        log.info(f"  {label:42s} [{tag:>10s}]  detected={det_rate:5.1%}  (n={n_win}){marker}")

    # ── Round 3: Evasion-resistant features ──────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("ROUND 3: EVASION-RESISTANT FEATURES")
    log.info("=" * 70)

    # Add new features to all windows
    df_win_r3 = add_evasion_resistant_features(df_win.copy(), df_raw)
    new_feat_cols = ["cumulative_energy_wh", "power_mem_correlation",
                     "util_bimodal_score", "power_efficiency_ratio",
                     "sustained_compute_frac"]
    existing_new = [c for c in new_feat_cols if c in df_win_r3.columns]
    r3_feature_cols = feature_cols + existing_new
    log.info(f"  Added {len(existing_new)} evasion-resistant features: {existing_new}")
    log.info(f"  Total features: {len(r3_feature_cols)}")

    # Train on clean + adversarial (same as Round 2) with new features (balanced)
    df_r3_train = pd.concat([
        df_win_r3[clean_mask | hardening_mask]
    ])
    X_r3 = df_r3_train[r3_feature_cols].fillna(0).values.astype(np.float32)
    y_r3 = df_r3_train["threeway_label"].values
    groups_r3 = df_r3_train["run_id"].values

    clf_r3, predict_r3 = _train_predictor(
        variant, X_r3, y_r3, groups_r3, r3_feature_cols, xgb_params, le,
    )

    # Feature importance (threeway only — two-stage has split feature sets per stage)
    if variant == "threeway":
        importance = clf_r3.feature_importances_
        feat_imp = sorted(zip(r3_feature_cols, importance), key=lambda x: -x[1])
        log.info("\n  Top 15 features (Round 3 classifier):")
        for feat, imp in feat_imp[:15]:
            marker = " <-- NEW" if feat in existing_new else ""
            log.info(f"    {feat:45s} importance={imp:.4f}{marker}")

    # Evaluate Round 3 on all adversarial
    for label in sorted(set(strategy_labels) | set(composite_labels)):
        mask = df_win_r3["workload_label"] == label
        if mask.sum() == 0:
            continue
        X_adv = df_win_r3.loc[mask, r3_feature_cols].fillna(0).values.astype(np.float32)
        y_pred = predict_r3(X_adv)
        det_rate = (y_pred == "ml_training").mean()
        n_win = len(X_adv)

        idle_frac = None
        if "composite" in label:
            try:
                idle_frac = int(label.split("_")[-1]) / 100.0
            except ValueError:
                pass

        results.append({
            "round": "R3_evasion_resistant",
            "workload": label,
            "idle_fraction": idle_frac,
            "detection_rate": det_rate,
            "n_windows": n_win,
            "classified_training": (y_pred == "ml_training").sum(),
            "classified_inference": (y_pred == "ml_inference").sum(),
            "classified_other": (y_pred == "other").sum(),
        })
        tag = f"idle={idle_frac:.0%}" if idle_frac is not None else "strategy"
        log.info(f"  {label:42s} [{tag:>10s}]  detected={det_rate:5.1%}  (n={n_win})")

    # ── Save results ─────────────────────────────────────────────────────
    df_results = pd.DataFrame(results)
    out_name = "adv_defender_rounds.csv" if variant == "threeway" else "adv_defender_rounds_twostage.csv"
    out_path = RESULTS_DIR / out_name
    df_results.to_csv(out_path, index=False)
    log.info(f"\nSaved: {out_path}")

    # ── Evasion tax curve (composite only, all 3 rounds) ─────────────────
    log.info("\n" + "=" * 70)
    log.info("EVASION TAX CURVE — COMPOSITE SWEEP (RTX 5080)")
    log.info("=" * 70)

    composite_results = df_results[df_results["idle_fraction"].notna()].copy()
    if not composite_results.empty and variant == "threeway":
        composite_results.to_csv(RESULTS_DIR / "evasion_tax_5080.csv", index=False)
        log.info(f"Saved: {RESULTS_DIR / 'evasion_tax_5080.csv'}")

        # Summary table
        log.info(f"\n{'Idle%':>6s}  {'R1 Baseline':>12s}  {'R2 Hardened':>12s}  {'R3 +Features':>12s}")
        log.info("-" * 50)
        for frac in sorted(composite_results["idle_fraction"].unique()):
            row = {}
            for r in ["R1_baseline", "R2_hardened", "R3_evasion_resistant"]:
                sub = composite_results[(composite_results["idle_fraction"] == frac) &
                                        (composite_results["round"] == r)]
                row[r] = sub["detection_rate"].values[0] if len(sub) > 0 else float("nan")

            log.info(f"  {frac:4.0%}    {row.get('R1_baseline', float('nan')):10.1%}"
                     f"    {row.get('R2_hardened', float('nan')):10.1%}"
                     f"    {row.get('R3_evasion_resistant', float('nan')):10.1%}")

    # ── Evasion tax bounds ───────────────────────────────────────────────
    log.info("\n" + "=" * 70)
    log.info("EVASION TAX BOUNDS")
    log.info("=" * 70)

    for round_name in ["R1_baseline", "R2_hardened", "R3_evasion_resistant"]:
        sub = composite_results[composite_results["round"] == round_name].sort_values("idle_fraction")
        if sub.empty:
            continue

        for threshold in [0.50, 0.20]:
            below = sub[sub["detection_rate"] < threshold]
            if not below.empty:
                min_idle = below["idle_fraction"].min()
                log.info(f"  {round_name:25s}: <{threshold:.0%} detection requires "
                         f"≥{min_idle:.0%} idle (≥{min_idle:.0%} throughput loss)")
            else:
                best = sub.loc[sub["detection_rate"].idxmin()]
                log.info(f"  {round_name:25s}: NEVER reaches <{threshold:.0%} detection "
                         f"(best: {best['detection_rate']:.1%} at {best['idle_fraction']:.0%} idle)")

    log.info("\n" + "=" * 70)
    log.info("ANALYSIS COMPLETE")
    log.info("=" * 70)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["threeway", "twostage"], default="threeway")
    ap.add_argument("--holdout-prefixes", type=str, default="",
                    help="comma-separated label prefixes to hold out of hardened training")
    args = ap.parse_args()
    prefixes = tuple(p.strip() for p in args.holdout_prefixes.split(",") if p.strip())
    run_analysis(variant=args.classifier, holdout_prefixes=prefixes)
