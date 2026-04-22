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

def add_evasion_resistant_features(df_win: pd.DataFrame, df_raw: pd.DataFrame) -> pd.DataFrame:
    """Add features that are hard to spoof without reducing real throughput.

    These exploit physical invariants:
    - Energy = integral of power over time (can't be hidden)
    - Memory-power correlation (training: high mem + high power; idle padding
      creates high mem + LOW power which is unnatural)
    - Compute intensity = power / utilization (training has consistent ratio;
      evasion creates artificial idle gaps that change the ratio)
    """
    new_feats = {}

    for run_id in df_win["run_id"].unique():
        run_raw = df_raw[df_raw["run_id"] == run_id]
        if run_raw.empty:
            continue

        win_mask = df_win["run_id"] == run_id
        for idx in df_win[win_mask].index:
            # Get the raw data for this window's time range
            # (approximation: use whole-run features as window-level proxies)
            power = run_raw["power_draw_w"].values
            util = run_raw["gpu_utilization_pct"].values
            mem = run_raw["mem_used_mb"].values

            # Feature 1: Cumulative energy (Watt-seconds)
            # Can't be hidden — if you did real compute, energy was consumed
            new_feats.setdefault("cumulative_energy_wh", {})[idx] = power.sum() / 3600.0

            # Feature 2: Power-memory correlation
            # Training: memory high AND power high (correlated)
            # Evasion with pre-alloc + idle: memory high but power low (uncorrelated)
            if len(power) > 5 and power.std() > 0 and mem.std() > 0:
                corr = np.corrcoef(power, mem[:len(power)])[0, 1]
                new_feats.setdefault("power_mem_correlation", {})[idx] = corr if np.isfinite(corr) else 0.0
            else:
                new_feats.setdefault("power_mem_correlation", {})[idx] = 0.0

            # Feature 3: Bimodal utilization score
            # Idle-padded evasion creates bimodal util distribution (high compute + zero idle)
            # Real training is mostly sustained high utilization
            if len(util) > 10:
                low_frac = (util < 15).mean()
                high_frac = (util > 70).mean()
                bimodal = low_frac * high_frac  # High when both extremes present
                new_feats.setdefault("util_bimodal_score", {})[idx] = bimodal
            else:
                new_feats.setdefault("util_bimodal_score", {})[idx] = 0.0

            # Feature 4: Power efficiency ratio
            # Watts per % utilization — real training is consistent,
            # evasion with throttling is artificially low
            mean_util = util.mean()
            mean_power = power.mean()
            if mean_util > 1:
                new_feats.setdefault("power_efficiency_ratio", {})[idx] = mean_power / mean_util
            else:
                new_feats.setdefault("power_efficiency_ratio", {})[idx] = mean_power

            # Feature 5: Sustained compute fraction
            # Fraction of time at >50% util for >10 consecutive seconds
            # Training is sustained; evasion is bursty
            if len(util) > 10:
                sustained = 0
                streak = 0
                for u in util:
                    if u > 50:
                        streak += 1
                        if streak >= 10:
                            sustained += 1
                    else:
                        streak = 0
                new_feats.setdefault("sustained_compute_frac", {})[idx] = sustained / max(len(util), 1)
            else:
                new_feats.setdefault("sustained_compute_frac", {})[idx] = 0.0

    # Add new features to dataframe
    for feat_name, feat_dict in new_feats.items():
        df_win[feat_name] = df_win.index.map(feat_dict).fillna(0.0)

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


def run_analysis(variant="threeway"):
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
    args = ap.parse_args()
    run_analysis(variant=args.classifier)
