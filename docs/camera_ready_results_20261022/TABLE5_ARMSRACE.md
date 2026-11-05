# Table 5 — sequential arms race, scope-corrected, repeated CV

Corpus: single-node only (<=8 GPUs/job, one host); `symgemm_L3_graddiscard` excluded as
an instrument; every non-adversarial family in `corpus_0` (27,514 windows), so only
evasion families are round-indexed. 30 s / 15 s windows, RF 400 trees,
class_weight=balanced, random_state=42, 5 folds.

**The in-distribution column is repeated over 5 fold-assignment seeds**
([42, 7, 1234, 2026, 99]) and reported as mean, with the per-cell sd in the JSON. Only the
fold assignment varies; the forest stays at random_state=42, so the spread is
fold-assignment variance alone. A single seed is not a measurement: 10 of the 22
families are one workload cell, so under workload grouping the family lands
entirely in one test fold and the "in-distribution" detector never trained on it.
`baseline det.`, `prev.-round det.` and `hardened unseen` do not cross-validate and
are identical under all three protocols.

## Baseline row (no evasion anywhere)

| holdout | accuracy | recall | specificity | groups |
|---|---|---|---|---|
| job_id | 97.43% ± 1.03 | 97.54% ± 1.49 | 97.34% ± 0.85 | 877 |
| run_id | 99.35% ± 0.18 | 99.14% ± 0.25 | 99.52% ± 0.38 | 1111 |
| workload | 87.32% ± 3.24 | 87.90% ± 5.32 | 86.84% ± 2.91 | 127 |

Published value: **97.6%** (defined as held-out non-adversarial *workloads*, so the workload row is the comparable one).


## in-dist holdout = `job_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | max sd | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 16.8-87.3% | 16.8-87.3% | 97.8-100.0% | ±3.9 | 28.9-94.1% | 100% / 59-100% |
| 2 | 6 | 1,455 | 9.5-47.1% | 26.3-98.3% | 95.8-99.9% | ±0.8 | 91.9-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.3% | 89.4-100.0% | 90.7-100.0% | ±2.0 | 88.8-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 76.5-99.8% | 87.7-99.9% | 99.8-100.0% | ±0.5 | 100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 86.2-100.0% | 100.0% | ±0.0 | 92.2-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 21.7-94.6% | 48.7-99.4% | 87.1-99.6% | ±9.6 | 48.7-99.7% | 100% / 43-87% |
| 6 | 2 | 1,525 | 0.3-83.4% | 2.7-93.0% | 89.5-100.0% | ±5.1 | 5.1-76.4% | - / - |

## in-dist holdout = `run_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | max sd | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 16.8-87.3% | 16.8-87.3% | 96.4-99.8% | ±4.5 | 28.9-94.1% | 100% / 59-100% |
| 2 | 6 | 1,455 | 9.5-47.1% | 26.3-98.3% | 96.7-100.0% | ±0.5 | 91.9-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.3% | 89.4-100.0% | 89.8-100.0% | ±1.0 | 88.8-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 76.5-99.8% | 87.7-99.9% | 100.0% | ±0.0 | 100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 86.2-100.0% | 100.0% | ±0.0 | 92.2-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 21.7-94.6% | 48.7-99.4% | 93.9-99.7% | ±4.6 | 48.7-99.7% | 100% / 43-87% |
| 6 | 2 | 1,525 | 0.3-83.4% | 2.7-93.0% | 86.4-100.0% | ±8.4 | 5.1-76.4% | - / - |

## in-dist holdout = `workload`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | max sd | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 16.8-87.3% | 16.8-87.3% | 23.1-82.5% | ±13.0 | 28.9-94.1% | 100% / 59-100% |
| 2 | 6 | 1,455 | 9.5-47.1% | 26.3-98.3% | 77.7-100.0% | ±20.6 | 91.9-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.3% | 89.4-100.0% | 88.4-99.9% | ±3.9 | 88.8-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 76.5-99.8% | 87.7-99.9% | 99.7-100.0% | ±0.7 | 100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 86.2-100.0% | 99.8-100.0% | ±0.3 | 92.2-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 21.7-94.6% | 48.7-99.4% | 86.6-99.6% | ±9.3 | 48.7-99.7% | 100% / 43-87% |
| 6 | 2 | 1,525 | 0.3-83.4% | 2.7-93.0% | 53.2-99.9% | ±29.9 | 5.1-76.4% | - / - |

## Per family

Control rows scored on non-training windows report a FALSE POSITIVE rate. in-dist cells are mean ± sd over the seeds.

| round | family | kind | windows | cells | runs | baseline | prev-round | unseen | in-dist job_id | in-dist run_id | in-dist workload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A_util_modulation | evasion | 190 | 1 | 10 | 30.0% | 30.0% | 33.7% | 98.8% ± 0.2 | 98.7% ± 0.3 | 30.1% ± 0.7 |
| 1 | B_low_util | evasion | 190 | 1 | 10 | 16.8% | 16.8% | 28.9% | 97.8% ± 3.9 | 96.4% ± 4.5 | 23.1% ± 13.0 |
| 1 | D_temporal_disruption | evasion | 236 | 1 | 12 | 87.3% | 87.3% | 94.1% | 100.0% ± 0.0 | 99.8% ± 0.4 | 82.5% ± 8.6 |
| 2 | E_memory_minimal | evasion | 173 | 1 | 9 | 45.1% | 70.5% | 97.7% | 99.4% ± 0.0 | 99.4% ± 0.0 | 77.7% ± 20.6 |
| 2 | F_interleave | evasion | 346 | 2 | 18 | 9.5% | 26.3% | 100.0% | 99.8% ± 0.2 | 99.9% ± 0.1 | 100.0% ± 0.0 |
| 2 | G_clock_throttled | evasion | 117 | 1 | 6 | 17.1% | 98.3% | 99.1% | 99.1% ± 0.0 | 99.1% ± 0.0 | 99.1% ± 0.0 |
| 2 | H_mimicry | evasion | 419 | 2 | 22 | 33.2% | 45.8% | 96.7% | 99.9% ± 0.2 | 100.0% ± 0.0 | 99.2% ± 1.7 |
| 2 | I_stochastic | evasion | 172 | 1 | 9 | 47.1% | 55.8% | 91.9% | 95.8% ± 0.8 | 96.7% ± 0.5 | 93.4% ± 1.7 |
| 2 | J_pid | evasion | 228 | 1 | 12 | 39.0% | 48.7% | 100.0% | 99.5% ± 0.6 | 99.7% ± 0.4 | 99.6% ± 0.6 |
| 3 | K_online_learning | evasion | 348 | 1 | 18 | 89.1% | 89.4% | 88.8% | 90.7% ± 2.0 | 89.8% ± 1.0 | 88.4% ± 3.9 |
| 3 | L_diluted | evasion | 983 | 4 | 51 | 99.3% | 100.0% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.9% ± 0.1 |
| 4 | K_ddp | evasion | 592 | 1 | 24 | 99.8% | 99.8% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | K_ddp_accum | evasion | 1,016 | 2 | 48 | 99.8% | 99.9% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.1 |
| 4 | L_ddp | evasion | 1,128 | 2 | 48 | 98.9% | 98.8% | 100.0% | 99.8% ± 0.5 | 100.0% ± 0.0 | 99.8% ± 0.5 |
| 4 | L_ddp_stagger | evasion | 480 | 1 | 24 | 76.5% | 87.7% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.7% ± 0.7 |
| 4 | M_composite_memmin | evasion | 76 | 4 | 4 | 0.0% | 100.0% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | N_grad_accum | evasion | 57 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | O_composite_idle_pad | evasion | 448 | 11 | 64 | 50.4% | 86.2% | 92.2% | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.8% ± 0.3 |
| 5 | whitebox_full | evasion | 313 | 8 | 16 | 94.6% | 99.4% | 99.7% | 99.6% ± 0.1 | 99.7% ± 0.1 | 99.6% ± 0.4 |
| 5 | whitebox_lora | evasion | 115 | 3 | 6 | 21.7% | 48.7% | 48.7% | 87.1% ± 9.6 | 93.9% ± 4.6 | 86.6% ± 9.3 |
| 6 | symgemm_L4_fused_sgd | evasion | 702 | 6 | 12 | 0.3% | 2.7% | 5.1% | 89.5% ± 5.1 | 86.4% ± 8.4 | 53.2% ± 29.9 |
| 6 | symgemm_ppo_fused | evasion | 823 | 7 | 21 | 83.4% | 93.0% | 76.4% | 100.0% ± 0.1 | 100.0% ± 0.0 | 99.9% ± 0.1 |
| 0 | bestofn_rm | control FP | 240 | 2 | 6 | 0.0% | 0.0% | - | 0.0% ± 0.0 | 0.0% ± 0.0 | 71.9% ± 23.8 |
| 0 | ppo_rlhf | control | 348 | 2 | 12 | 100.0% | 100.0% | - | 99.1% ± 0.2 | 98.9% ± 0.1 | 98.3% ± 0.7 |
| 0 | ppo_rlhf_accum | control | 240 | 2 | 6 | 100.0% | 100.0% | - | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 0 | rl_distributed | control | 640 | 1 | 2 | 100.0% | 100.0% | - | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 0 | symgemm_L5_adam | control | 156 | 1 | 4 | 100.0% | 100.0% | - | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 0 | symgemm_negatives | control FP | 1,039 | 10 | 21 | 0.0% | 0.0% | - | 4.0% ± 1.4 | 3.4% ± 1.9 | 22.5% ± 7.9 |
| 0 | whitebox_inference | control FP | 42 | 1 | 2 | 0.0% | 0.0% | - | 6.7% ± 1.1 | 7.1% ± 0.0 | 13.8% ± 4.6 |
