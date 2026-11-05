# RandomForest vs XGBoost, job-level holdout

XGBoost is the reported classifier for the camera-ready. RandomForest results
are kept alongside for the appendix and for the comparison below.

The switch rests on a paired test, not on comparing two means: over 10
fold-assignment seeds XGBoost beats RandomForest in **10/10 seeds**, mean
**+1.34 pp** (95% CI +1.10 to +1.57), paired t = 10.98, p < 0.0001, Wilcoxon
p = 0.002. A single fold assignment is not enough to rank them — on seed 42 the
gap was 1.31 pp with the entire difference inside one fold.

The advantage is tail risk rather than a uniform edge. XGBoost wins 33 of 50
individual folds, most narrowly; only 12 folds differ by more than a point. In
those folds RandomForest collapses and XGBoost does not, and the cause is the
same every time: a handful of very large multi-GPU jobs
(`llm_infer_deepseek_v3_tp8` at 1,272 windows, `fsdp_70b_nvlink` at 1,920) that
move between folds as single units under job-level holdout.


## Table 6 — 10 fold-assignment seeds x 5 folds

| classifier | accuracy | macro-F1 | worst fold |
|---|---|---|---|
| XGBoost | 99.02% ± 0.71 | 98.999 | 96.39% |
| RandomForest | 97.69% ± 2.96 | 97.635 | 87.43% |
| SVM-RBF | 96.34% ± 2.20 | 96.242 | 90.94% |
| LogisticRegression | 89.86% ± 1.33 | 89.642 | 86.66% |

## Headline

| holdout | RandomForest | XGBoost | FPR RF → XGB |
|---|---|---|---|
| job_id | 97.39% ± 3.79 | 98.70% ± 1.23 | 3.16% → 1.01% |
| run_id | 99.22% ± 0.82 | 99.36% ± 0.60 | 0.67% → 0.49% |
| workload | 86.83% ± 12.80 | 90.32% ± 8.58 | 27.51% → 20.20% |

## Table 5

| Round | baseline RF → XGB | prev.-round RF → XGB | in-dist RF → XGB | unseen RF → XGB |
|---|---|---|---|---|
| 1 | 16.8–87.3% → 30.0–86.0% | 16.8–87.3% → 30.0–86.0% | 97.8–100.0% → 99.2–100.0% | 28.9–94.1% → 30.0–92.8% |
| 2 | 9.5–47.1% → 14.7–76.9% | 26.3–98.3% → 36.1–97.4% | 95.8–99.9% → 96.0–100.0% | 91.9–100.0% → 92.4–100.0% |
| 3 | 89.1–99.3% → 88.8–98.7% | 89.4–100.0% → 86.8–99.5% | 90.7–100.0% → 90.1–99.9% | 88.8–100.0% → 87.4–99.8% |
| 4 (multi-GPU) | 76.5–99.8% → 98.3–99.8% | 87.7–99.9% → 99.3–100.0% | 99.8–100.0% → 100.0% | 100.0% → 100.0% |
| 4 (single-GPU) | 0.0–100.0% → 10.5–100.0% | 86.2–100.0% → 81.5–100.0% | 100.0% → 100.0% | 92.2–100.0% → 84.6–100.0% |
| 5 | 21.7–94.6% → 49.6–97.1% | 48.7–99.4% → 52.2–89.8% | 87.1–99.6% → 91.8–100.0% | 48.7–99.7% → 56.5–91.1% |
| 6 | 0.3–83.4% → 12.0–99.8% | 2.7–93.0% → 9.1–99.4% | 89.5–100.0% → 87.4–100.0% | 5.1–76.4% → 11.7–96.2% |

Baseline row: 97.43% ± 1.03 → **98.42% ± 0.14**.


## Table 7 — evasion families

| round | family | baseline RF → XGB | in-dist RF → XGB | unseen RF → XGB |
|---|---|---|---|---|
| 1 | `A_util_modulation` | 30.0 → 30.5 | 98.8 → 99.2 | 33.7 → 34.2 |
| 1 | `B_low_util` | 16.8 → 30.0 | 97.8 → 99.9 | 28.9 → 30.0 |
| 1 | `D_temporal_disruption` | 87.3 → 86.0 | 100.0 → 100.0 | 94.1 → 92.8 |
| 2 | `E_memory_minimal` | 45.1 → 76.9 | 99.4 → 100.0 | 97.7 → 100.0 |
| 2 | `F_interleave` | 9.5 → 14.7 | 99.8 → 100.0 | 100.0 → 100.0 |
| 2 | `G_clock_throttled` | 17.1 → 17.1 | 99.1 → 99.1 | 99.1 → 99.1 |
| 2 | `H_mimicry` | 33.2 → 57.5 | 99.9 → 99.9 | 96.7 → 99.8 |
| 2 | `I_stochastic` | 47.1 → 45.9 | 95.8 → 96.0 | 91.9 → 92.4 |
| 2 | `J_pid` | 39.0 → 63.2 | 99.5 → 99.9 | 100.0 → 100.0 |
| 3 | `K_online_learning` | 89.1 → 88.8 | 90.7 → 90.1 | 88.8 → 87.4 |
| 3 | `L_diluted` | 99.3 → 98.7 | 100.0 → 99.9 | 100.0 → 99.8 |
| 4 | `K_ddp` | 99.8 → 99.8 | 100.0 → 100.0 | 100.0 → 100.0 |
| 4 | `K_ddp_accum` | 99.8 → 99.3 | 100.0 → 100.0 | 100.0 → 100.0 |
| 4 | `L_ddp` | 98.9 → 98.3 | 99.8 → 100.0 | 100.0 → 100.0 |
| 4 | `L_ddp_stagger` | 76.5 → 99.6 | 100.0 → 100.0 | 100.0 → 100.0 |
| 4 | `M_composite_memmin` | 0.0 → 10.5 | 100.0 → 100.0 | 100.0 → 100.0 |
| 4 | `N_grad_accum` | 100.0 → 100.0 | 100.0 → 100.0 | 100.0 → 100.0 |
| 4 | `O_composite_idle_pad` | 50.4 → 64.1 | 100.0 → 100.0 | 92.2 → 84.6 |
| 5 | `whitebox_full` | 94.6 → 97.1 | 99.6 → 100.0 | 99.7 → 91.1 |
| 5 | `whitebox_lora` | 21.7 → 49.6 | 87.1 → 91.8 | 48.7 → 56.5 |
| 6 | `symgemm_L4_fused_sgd` | 0.3 → 12.0 | 89.5 → 87.4 | 5.1 → 11.7 |
| 6 | `symgemm_ppo_fused` | 83.4 → 99.8 | 100.0 → 100.0 | 76.4 → 96.2 |

## Table 4 — per GPU

| gpu | CV RF → XGB | LOGO RF → XGB |
|---|---|---|
| A100 SXM4 40GB | 100.00 → 99.91 | 95.33 → 99.12 |
| A100 SXM4 80GB | 99.95 → 99.96 | 96.25 → 98.05 |
| B200 | 98.40 → 98.54 | 80.80 → 91.43 |
| H100 80GB | 99.13 → 98.97 | 99.35 → 97.97 |
| H100 NVL | 94.84 → 91.64 | 87.72 → 87.37 |
| H200 / H200 NVL | 97.41 → 98.68 | 85.11 → 86.40 |
| RTX 4050 Laptop | 97.36 → 94.74 | 60.35 → 66.75 |
| RTX 5080 | 98.03 → 98.01 | 79.00 → 77.59 |
| RTX 5090 | 96.36 → 96.09 | 64.77 → 80.48 |

LOGO mean **83.19% → 87.24%**. The published column is 99.80–100%, so neither classifier reproduces it; see PPO_FUSED_EPOCH_CONFOUND.md and the doc comments on Table 4.


## Figure 3 — window sweep

| window | RandomForest | XGBoost |
|---|---|---|
| 5s | 98.17% ± 0.84 | 98.02% ± 1.04 |
| 10s | 98.68% ± 0.79 | 98.90% ± 0.67 |
| 15s | 99.30% ± 0.26 | 98.88% ± 1.05 |
| 20s | 99.18% ± 0.50 | 99.18% ± 0.35 |
| 30s | 97.39% ± 3.79 | 98.70% ± 1.23 |
| 45s | 97.14% ± 2.66 | 97.85% ± 1.36 |
| 60s | 97.45% ± 2.53 | 98.40% ± 0.49 |
| 90s | 97.53% ± 3.05 | 98.64% ± 0.55 |
| 120s | 97.44% ± 3.17 | 97.91% ± 1.59 |

XGBoost's sweep spans 1.3 pp against RandomForest's 2.2 pp and loses most of the
post-30 s decline, so "accuracy is stable across window size" becomes defensible.
Peak is 20 s under both; "plateau from 30 s" is not supported by either.

## Reproducing

Every script funnels its fits through one factory, switched by an environment
variable, so both columns come from identical code paths:

    PAPER_MODEL=xgb PYTHONPATH=. python3 scripts/paper_table5_armsrace.py \
        --controls-round 0 --ppo-rlhf-round 0 --tag xgb_seeds --seeds 42 7 1234 2026 99

`PAPER_NJOBS` caps thread count per process for running shards in parallel.
`scripts/paper_headline_cache.py` computes the headline from the cached window
frame in seconds rather than rebuilding windows from raw telemetry.

## Caveat

`paper_numbers.json` (raw-telemetry path) gives RandomForest 97.31% ± 3.92 at
job level; the cache path gives 97.39% ± 3.79. Both tables above use the cache
path, so RF and XGBoost are strictly comparable, but which path is canonical for
the paper's headline is unresolved.
