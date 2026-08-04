"""
Leave-One-Strategy-Out Evaluation
==================================

For each adversarial strategy family, train the hardened classifier on
clean data + ALL OTHER adversarial families, then test on the held-out
family. This tests whether the classifier generalizes to truly unseen
evasion approaches (not just unseen parameters of a known approach).

If detection stays high on held-out strategies, the classifier is
learning generalizable features of "disguised training" rather than
memorizing specific evasion signatures.

Produces (depending on --classifier):
  - classifier/results/leave_one_out.csv          (threeway, default)
  - classifier/results/leave_one_out_twostage.csv (twostage)

Usage:
    python -m classifier.leave_one_strategy_out
    python -m classifier.leave_one_strategy_out --classifier twostage
"""

import argparse
import logging
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

from classifier.threeway_improved import (
    load_all_data, sliding_windows, get_feature_cols,
    compute_sample_weights,
    TRAINING_LABELS,
)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Strategy families — group variants of the same approach
STRATEGY_FAMILIES = {
    "A_util_modulation": ["adversarial_A_util_modulation"],
    "B_low_util": ["adversarial_B_low_util"],
    "D_temporal": ["adversarial_D_temporal_disruption"],
    "E_memory_minimal": ["adversarial_E_memory_minimal"],
    "F_interleave": ["adversarial_F_interleave_30", "adversarial_F_interleave_50"],
    "G_clock_throttled": ["adversarial_G_clock_throttled"],
    "H_mimicry": ["adversarial_H_mimicry_mining", "adversarial_H_mimicry_cufft"],
    "I_stochastic": ["adversarial_I_stochastic_30"],
    "J_pid": ["adversarial_J_pid_75"],
    "K_online_learning": ["adversarial_K_online_learning"],
    "L_diluted": [
        "adversarial_L_diluted_2", "adversarial_L_diluted_5",
        "adversarial_L_diluted_10", "adversarial_L_diluted_20",
    ],
    "composite": [
        "adversarial_composite_00", "adversarial_composite_10",
        "adversarial_composite_20", "adversarial_composite_30",
        "adversarial_composite_40", "adversarial_composite_50",
        "adversarial_composite_60", "adversarial_composite_70",
        "adversarial_composite_80", "adversarial_composite_90",
        "adversarial_composite_95",
    ],
    "M_composite_memmin": [
        "adversarial_M_composite_memmin_10",
        "adversarial_M_composite_memmin_20",
        "adversarial_M_composite_memmin_30",
        "adversarial_M_composite_memmin_40",
    ],
    "N_grad_accum": [
        "adversarial_N_grad_accum_4",
        "adversarial_N_grad_accum_8",
        "adversarial_N_grad_accum_16",
    ],
    # DDP adversarial families (datacenter multi-GPU evasion)
    "K_ddp": ["adversarial_K_ddp"],
    "K_ddp_accum": ["adversarial_K_ddp_accum_4", "adversarial_K_ddp_accum_8"],
    "L_ddp": ["adversarial_L_ddp_5", "adversarial_L_ddp_10"],
    "L_ddp_stagger": ["adversarial_L_ddp_stagger_5"],
    # Round 5: White-box evasion at relevant model scale
    "whitebox_full": [
        "whitebox_diluted_N5", "whitebox_diluted_N10",
        "whitebox_diluted_N20", "whitebox_diluted_N50",
        "whitebox_diluted_N5_ckpt", "whitebox_diluted_N10_ckpt",
        "whitebox_diluted_N20_ckpt",
        "whitebox_oracle",
    ],
    "whitebox_lora": [
        "whitebox_lora_N5", "whitebox_lora_N10", "whitebox_lora_N20",
    ],
    # Round 6: symmetric-GEMM backprop disguise. L4 (the attack, fused SGD) and
    # L5 (AdamW positive control) are registered as SEPARATE held-out families so
    # the disguised-training detection rate is reported apart from the easy
    # AdamW control. Rep suffixes are enumerated because test_mask uses exact
    # isin(); labels that were not collected simply contribute 0 test windows.
    # The symgemm inference controls (L0/L1/L3) are deliberately NOT a family and
    # are excluded from training below (see the symgemm_ startswith filter), so
    # they leak into neither train nor test -- matching symgemm_eval.py's protocol.
    "symgemm_L4_disguise": [
        "symgemm_L4_sgdfused_rep1", "symgemm_L4_sgdfused_rep2",
        "symgemm_L4_sgdfused_rep3", "symgemm_L4_sgdfused_rep4",
        "symgemm_L4_sgdfused_m8192_rep1", "symgemm_L4_sgdfused_m8192_rep2",
        "symgemm_L4_sgdfused_7b_rep1",
    ],
    "symgemm_L5_adam": [
        "symgemm_L5_adam_rep1", "symgemm_L5_adam_rep2",
        "symgemm_L5_adam_rep3", "symgemm_L5_adam_rep4",
    ],
    # Round 6 / Area Chair 1uob: PPO-shaped training (R rollout generations then
    # one fused-SGD policy update) is the one regime the corpus otherwise lacks.
    # Held out as one family so the detection rate reflects an unseen-RL monitor.
    # It is a DILUTED trainer (~3-11% duty cycle), so the window-level truth in
    # the eval matters: most windows are rollout (inference), only the update
    # steps are training. decode (symgemm_decode_*) is NOT a family -- it is the
    # inference control, excluded from clean training by the symgemm_ filter.
    # DISGUISED PPO only. The undisguised variant (ppo_rlhf_*) is a legitimate
    # RL training regime in the baseline corpus and must NOT be held out here --
    # it belongs in the training pool so the monitor is calibrated on RL.
    "symgemm_ppo_fused": [
        "symgemm_ppo_fused_r8_rep1", "symgemm_ppo_fused_r8_rep2",
        "symgemm_ppo_fused_r8_rep3", "symgemm_ppo_fused_r32_rep1",
        "symgemm_ppo_fused_r32_rep2", "symgemm_ppo_fused_r32_rep3",
    ],
}


def main():
    from sklearn.metrics import accuracy_score
    from sklearn.preprocessing import LabelEncoder
    import xgboost as xgb
    from classifier.twostage import TwoStageClassifier

    ap = argparse.ArgumentParser()
    ap.add_argument("--classifier", choices=["threeway", "twostage"], default="threeway")
    ap.add_argument("--backend", choices=["xgboost", "random_forest"], default="random_forest",
                    help="Classifier backend for --classifier twostage. Ignored for threeway.")
    args = ap.parse_args()
    variant = args.classifier
    backend = args.backend

    log.info("=" * 70)
    log.info(f"LEAVE-ONE-STRATEGY-OUT EVALUATION  (classifier={variant})")
    log.info("=" * 70)

    # Load and window all data. Attach per-window train/infer ground truth from
    # the STEP logs (for the runs that have them) so a diluted/PPO family is
    # scored by real per-window activity, not its run-level label.
    log.info("Loading data...")
    df_raw = load_all_data()
    from classifier.window_labels import build_step_log_map
    step_log_map = build_step_log_map("logs/*.log", df_raw["run_id"].astype(str).unique())
    log.info(f"  STEP-log window truth available for {len(step_log_map)} runs")
    df_win = sliding_windows(df_raw, window_sec=60, stride_sec=30,
                             run_level_mode="causal", step_log_map=step_log_map)
    has_window_truth = "window_train_active" in df_win.columns
    feature_cols = get_feature_cols(df_win)
    log.info(f"  Windows: {len(df_win)} | Features: {len(feature_cols)}")

    # Identify all adversarial labels
    all_adv_labels = set()
    for labels in STRATEGY_FAMILIES.values():
        all_adv_labels.update(labels)

    # Exclude EVERY symgemm_* run (including the L0/L1/L3 inference controls,
    # which are not a registered family) from the clean training pool, so a
    # held-out symgemm family is scored by a classifier that has never seen any
    # Round-6 program -- the same no-leakage protocol as symgemm_eval.py.
    all_adv_labels.update(
        l for l in df_win["workload_label"].astype(str).unique()
        if l.startswith("symgemm_")
    )

    clean_mask = ~df_win["workload_label"].isin(all_adv_labels)
    df_clean = df_win[clean_mask]
    log.info(f"  Clean windows: {len(df_clean)}")

    le = LabelEncoder()
    le.fit(sorted(df_clean["threeway_label"].unique()))

    results = []

    for held_out_family, held_out_labels in STRATEGY_FAMILIES.items():
        # Training set: clean + all adversarial EXCEPT the held-out family
        other_adv_labels = all_adv_labels - set(held_out_labels)
        train_mask = clean_mask | df_win["workload_label"].isin(other_adv_labels)
        test_mask = df_win["workload_label"].isin(held_out_labels)

        n_train = train_mask.sum()
        n_test = test_mask.sum()
        if n_test == 0:
            log.info(f"  {held_out_family:25s}: no test data, skipping")
            continue

        X_train = df_win.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_train = df_win.loc[train_mask, "threeway_label"].values
        groups_train = df_win.loc[train_mask, "run_id"].values
        X_test = df_win.loc[test_mask, feature_cols].fillna(0).values.astype(np.float32)

        if variant == "threeway":
            y_train_enc = le.transform(y_train)
            w_train = compute_sample_weights(y_train_enc)
            clf = xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                eval_metric="mlogloss", random_state=42, n_jobs=-1, verbosity=0,
            )
            clf.fit(X_train, y_train_enc, sample_weight=w_train)
            predict_str = lambda X: le.inverse_transform(clf.predict(X))
        else:  # twostage
            train_active = (df_win.loc[train_mask, "window_train_active"].values
                            if has_window_truth else None)
            clf = TwoStageClassifier(backend=backend).fit(
                X_train, y_train, groups=groups_train, feature_cols=feature_cols,
                train_active=train_active,
            )
            predict_str = lambda X: clf.predict(X)

        y_pred = predict_str(X_test)
        det_rate = (y_pred == "ml_training").mean()
        n_as_inference = (y_pred == "ml_inference").sum()
        n_as_other = (y_pred == "other").sum()
        n_as_training = (y_pred == "ml_training").sum()

        # Activity-conditioned detection (STEP-log truth, where available):
        # detection over windows that ACTUALLY trained, and specificity on the
        # inference-only windows a diluted evader legitimately produces. This is
        # the correct denominator — a window with zero training steps is not a
        # missed training window.
        det_active_str = ""
        if has_window_truth:
            wta_test = df_win.loc[test_mask, "window_train_active"].values
            active = wta_test == 1.0
            inactive = wta_test == 0.0
            if active.any():
                det_active = (y_pred[active] == "ml_training").mean()
                det_active_str = (f"  | train-active windows: {int(active.sum())}, "
                                  f"detected={det_active:5.1%}")
                if inactive.any():
                    spec = (y_pred[inactive] != "ml_training").mean()
                    det_active_str += (f"; inference-only: {int(inactive.sum())}, "
                                       f"correct-non-train={spec:5.1%}")

        log.info(f"  {held_out_family:25s}: detected={det_rate:5.1%}  "
                 f"(n={n_test}, train_on={n_train})  "
                 f"[train={n_as_training}, infer={n_as_inference}, other={n_as_other}]"
                 f"{det_active_str}")

        # Per-label breakdown within family
        for label in held_out_labels:
            label_mask = df_win["workload_label"] == label
            if label_mask.sum() == 0:
                continue
            X_l = df_win.loc[label_mask, feature_cols].fillna(0).values.astype(np.float32)
            y_l = predict_str(X_l)
            det_l = (y_l == "ml_training").mean()

            results.append({
                "held_out_family": held_out_family,
                "held_out_label": label,
                "detection_rate": det_l,
                "n_windows": len(X_l),
                "n_training_windows_used": n_train,
                "classified_training": (y_l == "ml_training").sum(),
                "classified_inference": (y_l == "ml_inference").sum(),
                "classified_other": (y_l == "other").sum(),
            })

            if det_l < 1.0:
                log.info(f"    {label:45s} {det_l:5.1%}  "
                         f"(train={int((y_l=='ml_training').sum())}, "
                         f"infer={int((y_l=='ml_inference').sum())}, "
                         f"other={int((y_l=='other').sum())})")

    # Save results. Canonical "twostage" filename holds the default
    # backend (random_forest); non-default backends get a suffix so
    # comparison runs don't overwrite the deployed artifact's CSV.
    df_results = pd.DataFrame(results)
    if variant == "threeway":
        out_name = "leave_one_out.csv"
    elif backend == "random_forest":
        out_name = "leave_one_out_twostage.csv"
    else:
        out_name = f"leave_one_out_twostage_{backend}.csv"
    out_path = RESULTS_DIR / out_name
    df_results.to_csv(out_path, index=False)
    log.info(f"\nSaved: {out_path}")

    # Summary
    log.info(f"\n{'='*70}")
    log.info("SUMMARY")
    log.info(f"{'='*70}")
    log.info(f"{'Family':25s} {'Detection':>10s} {'Windows':>8s}  Notes")
    log.info("-" * 60)
    for family in STRATEGY_FAMILIES:
        sub = df_results[df_results["held_out_family"] == family]
        if sub.empty:
            continue
        det = sub["detection_rate"].mean()
        n = sub["n_windows"].sum()
        note = "EVADES" if det < 0.5 else "partially evades" if det < 0.9 else ""
        log.info(f"  {family:25s} {det:8.1%}  {n:6d}    {note}")

    # Overall
    overall_det = df_results["detection_rate"].mean()
    n_families = len(df_results["held_out_family"].unique())
    evading = df_results[df_results["detection_rate"] < 0.5]["held_out_family"].nunique()
    log.info(f"\n  Overall mean detection: {overall_det:.1%}")
    log.info(f"  Families that evade (<50% detection): {evading}/{n_families}")

    log.info(f"\n{'='*70}")
    log.info("LOO EVALUATION COMPLETE")
    log.info(f"{'='*70}")


if __name__ == "__main__":
    main()
