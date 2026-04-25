"""
Catch-up Evaluation — new attacks vs. the previous round's classifier
=====================================================================

For each adversarial round N, simulate the defender's state at the end
of round N-1 and score it against round N's newly-introduced attack
families. Answers "how much damage does a novel attack class do before
the defender reacts?"

Simulation per round (classifier = end-of-prior-round state):

  Test families (round N) | Trained on                     | Features
  ------------------------|--------------------------------|----------------------
  R1: A, B, D             | clean only                     | standard
  R2: E-J                 | clean + R1 families            | standard
  R3: K, L                | clean + R1-R2 families         | standard
  R4: M, N, composite, DDP| clean + R1-R3 families         | standard + physics (R3)
  R5: WB-*                | clean + R1-R4 families         | standard + physics (no plateau)

Feature sets reflect the defender's response timeline per § 4.4:
  - Physics features (cumulative energy, power-memory correlation, etc.)
    were added as the Round 3 defender response.
  - Memory-plateau features (first_30s_mem_delta, time_to_mem_plateau)
    were added as the Round 5 defender response.

Note on benign pool: the clean training corpus is held constant across
rounds. A stricter simulation would exclude Round 5b benign additions
(matched-size LLM inference, matched-pair LoRA training) from the R5
catch-up training set; we do not. The R5 catch-up number should
therefore be read as an *upper bound* on pre-plateau-feature detection
of white-box attacks — the true historical classifier would also have
had a smaller benign pool.

Output: classifier/results/catchup_evasion.csv

Usage:
    python -m classifier.eval_catchup
    python -m classifier.eval_catchup --backend xgboost
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

from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
from classifier.adversary_defender_iteration import add_evasion_resistant_features
from classifier.twostage import TwoStageClassifier

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
RESULTS_DIR.mkdir(exist_ok=True)


# Round 5 defender addition — not available to classifiers before R5.
PLATEAU_FEATS = {
    "first_30s_mem_delta_mb",
    "first_30s_mem_delta_frac",
    "first_30s_mem_ratio",
    "time_to_mem_plateau_samples",
}

# Round 3 defender addition — available to classifiers from R4 onward.
PHYSICS_FEATS = [
    "cumulative_energy_wh",
    "power_mem_correlation",
    "util_bimodal_score",
    "power_efficiency_ratio",
    "sustained_compute_frac",
]

ROUND_FAMILIES: dict[int, list[str]] = {
    1: [
        "adversarial_A_util_modulation",
        "adversarial_B_low_util",
        "adversarial_D_temporal_disruption",
    ],
    2: [
        "adversarial_E_memory_minimal",
        "adversarial_F_interleave_30", "adversarial_F_interleave_50",
        "adversarial_G_clock_throttled",
        "adversarial_H_mimicry_cufft", "adversarial_H_mimicry_mining",
        "adversarial_I_stochastic_30",
        "adversarial_J_pid_75",
    ],
    3: [
        "adversarial_K_online_learning",
        "adversarial_L_diluted_2", "adversarial_L_diluted_5",
        "adversarial_L_diluted_10", "adversarial_L_diluted_20",
    ],
    4: [
        "adversarial_M_composite_memmin_10", "adversarial_M_composite_memmin_20",
        "adversarial_M_composite_memmin_30", "adversarial_M_composite_memmin_40",
        "adversarial_N_grad_accum_4", "adversarial_N_grad_accum_8",
        "adversarial_N_grad_accum_16",
        "adversarial_composite_00", "adversarial_composite_10",
        "adversarial_composite_20", "adversarial_composite_30",
        "adversarial_composite_40", "adversarial_composite_50",
        "adversarial_composite_60", "adversarial_composite_70",
        "adversarial_composite_80", "adversarial_composite_90",
        "adversarial_composite_95",
        "adversarial_K_ddp", "adversarial_K_ddp_accum_4", "adversarial_K_ddp_accum_8",
        "adversarial_L_ddp_5", "adversarial_L_ddp_10",
        "adversarial_L_ddp_stagger_5",
    ],
    5: [
        "whitebox_diluted_N5", "whitebox_diluted_N10",
        "whitebox_diluted_N20", "whitebox_diluted_N50",
        "whitebox_diluted_N5_ckpt", "whitebox_diluted_N10_ckpt",
        "whitebox_diluted_N20_ckpt",
        "whitebox_oracle",
        "whitebox_lora_N5", "whitebox_lora_N10", "whitebox_lora_N20",
    ],
}


def features_for_round(round_n: int, standard_cols: list[str],
                       physics_cols: list[str]) -> list[str]:
    """Feature set available to the classifier being scored on round N attacks.

    standard_cols has already had plateau features removed (those were
    added in R5, so no pre-R5 classifier has them). Physics features
    join the mix from R4 onward (added as the R3 defender response).
    """
    if round_n <= 3:
        return standard_cols
    return standard_cols + physics_cols


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", choices=["random_forest", "xgboost"], default="random_forest")
    ap.add_argument("--window-sec", type=int, default=60)
    ap.add_argument("--stride-sec", type=int, default=30)
    args = ap.parse_args()

    log.info("=" * 70)
    log.info(f"CATCH-UP EVALUATION  (two-stage backend={args.backend})")
    log.info("=" * 70)

    log.info("Loading telemetry data...")
    df_raw = load_all_data()
    log.info(f"  Total rows: {len(df_raw):,}")

    log.info(f"Extracting {args.window_sec}s windows (stride {args.stride_sec}s)...")
    df_win = sliding_windows(df_raw, window_sec=args.window_sec, stride_sec=args.stride_sec)
    log.info(f"  Windows: {len(df_win)}")

    base_cols = get_feature_cols(df_win)
    log.info("Adding Round-3 physics features (cumulative_energy, power_mem_correlation, ...)")
    df_win = add_evasion_resistant_features(df_win, df_raw)

    standard_cols = [c for c in base_cols if c not in PLATEAU_FEATS]
    physics_present = [c for c in PHYSICS_FEATS if c in df_win.columns]
    log.info(
        f"  base features: {len(base_cols)}  |  "
        f"standard (minus plateau): {len(standard_cols)}  |  "
        f"physics present: {len(physics_present)}"
    )

    all_adv_labels = {l for labels in ROUND_FAMILIES.values() for l in labels}
    clean_mask = ~df_win["workload_label"].isin(all_adv_labels)
    log.info(f"  Clean windows: {clean_mask.sum()}  Adversarial windows: {(~clean_mask).sum()}")

    results: list[dict] = []
    for round_n in sorted(ROUND_FAMILIES):
        test_labels = [l for l in ROUND_FAMILIES[round_n]
                       if (df_win["workload_label"] == l).any()]
        missing = [l for l in ROUND_FAMILIES[round_n] if l not in test_labels]
        if missing:
            log.info(f"[R{round_n}] not present in data (skipping): {missing}")
        if not test_labels:
            log.info(f"[R{round_n}] no test families present; skipping round")
            continue

        prior_labels = {l for prior in range(1, round_n) for l in ROUND_FAMILIES[prior]}
        train_prior_mask = df_win["workload_label"].isin(prior_labels)
        train_mask = clean_mask | train_prior_mask

        feature_cols = features_for_round(round_n, standard_cols, physics_present)

        log.info(
            f"\n[R{round_n}] end-of-R{round_n-1} classifier: "
            f"train_windows={int(train_mask.sum())} "
            f"(clean={int(clean_mask.sum())}, prior_adv={int(train_prior_mask.sum())}) "
            f"features={len(feature_cols)} "
            f"test_families={len(test_labels)}"
        )

        X_tr = df_win.loc[train_mask, feature_cols].fillna(0).values.astype(np.float32)
        y_tr = df_win.loc[train_mask, "threeway_label"].values
        g_tr = df_win.loc[train_mask, "run_id"].values

        clf = TwoStageClassifier(backend=args.backend).fit(
            X_tr, y_tr, groups=g_tr, feature_cols=feature_cols,
        )

        round_dets: list[float] = []
        for label in sorted(test_labels):
            lab_mask = df_win["workload_label"] == label
            X_te = df_win.loc[lab_mask, feature_cols].fillna(0).values.astype(np.float32)
            if len(X_te) == 0:
                continue
            y_pred = clf.predict(X_te)
            det = float((y_pred == "ml_training").mean())
            round_dets.append(det)
            results.append({
                "round": round_n,
                "family": label,
                "detection_rate": det,
                "n_windows": int(len(X_te)),
                "classified_training": int((y_pred == "ml_training").sum()),
                "classified_inference": int((y_pred == "ml_inference").sum()),
                "classified_other": int((y_pred == "other").sum()),
                "n_train_windows_used": int(train_mask.sum()),
                "n_features_used": int(len(feature_cols)),
            })
            log.info(f"  {label:45s} det={det:5.1%}  (n={len(X_te)})")

        if round_dets:
            log.info(
                f"  [R{round_n}] min={min(round_dets):.1%} "
                f"mean={sum(round_dets)/len(round_dets):.1%} "
                f"max={max(round_dets):.1%}"
            )

    df_out = pd.DataFrame(results)
    out_path = RESULTS_DIR / "catchup_evasion.csv"
    df_out.to_csv(out_path, index=False)
    log.info(f"\nSaved: {out_path}")

    log.info("\n" + "=" * 70)
    log.info("PER-ROUND CATCH-UP SUMMARY (family-level aggregates)")
    log.info("=" * 70)
    log.info(f"{'Round':>6s}  {'n_fam':>5s}  {'min':>6s}  {'mean':>6s}  {'max':>6s}")
    for r in sorted(df_out["round"].unique()):
        sub = df_out[df_out["round"] == r]
        log.info(
            f"  R{r:<4d}  {len(sub):>5d}  "
            f"{sub['detection_rate'].min():>5.1%}  "
            f"{sub['detection_rate'].mean():>5.1%}  "
            f"{sub['detection_rate'].max():>5.1%}"
        )


if __name__ == "__main__":
    main()
