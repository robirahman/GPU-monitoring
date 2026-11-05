# Table 5 — sequential arms race, scope-corrected, repeated CV

Corpus: single-node only (<=8 GPUs/job, one host); `symgemm_L3_graddiscard` excluded as
an instrument; every non-adversarial family in `corpus_0` (25,922 windows), so only
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
| job_id | 97.07% ± 1.44 | 96.88% ± 2.98 | 97.21% ± 1.39 | 857 |
| run_id | 99.27% ± 0.14 | 99.12% ± 0.24 | 99.39% ± 0.26 | 1091 |
| workload | 87.07% ± 4.92 | 89.78% ± 4.53 | 84.92% ± 6.28 | 118 |

Published value: **97.6%** (defined as held-out non-adversarial *workloads*, so the workload row is the comparable one).


## in-dist holdout = `job_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | max sd | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-91.9% | 9.5-91.9% | 98.0-100.0% | ±4.5 | 21.1-87.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 4.3-55.8% | 6.4-65.8% | 96.2-100.0% | ±0.7 | 82.7-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.7% | 87.6-99.9% | 89.7-100.0% | ±1.1 | 87.9-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 72.9-99.8% | 86.9-100.0% | 99.6-100.0% | ±0.6 | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 100.0% | ±0.0 | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 20.9-79.9% | 52.2-94.9% | 86.4-99.7% | ±6.5 | 48.7-96.5% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-1.2% | 1.5-7.3% | 88.1-99.9% | ±8.3 | 1.3-2.6% | - / - |

## in-dist holdout = `run_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | max sd | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-91.9% | 9.5-91.9% | 93.9-100.0% | ±10.1 | 21.1-87.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 4.3-55.8% | 6.4-65.8% | 94.5-100.0% | ±0.7 | 82.7-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.7% | 87.6-99.9% | 89.4-100.0% | ±0.4 | 87.9-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 72.9-99.8% | 86.9-100.0% | 99.9-100.0% | ±0.1 | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 100.0% | ±0.0 | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 20.9-79.9% | 52.2-94.9% | 92.2-99.0% | ±4.0 | 48.7-96.5% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-1.2% | 1.5-7.3% | 81.0-99.7% | ±2.8 | 1.3-2.6% | - / - |

## in-dist holdout = `workload`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | max sd | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-91.9% | 9.5-91.9% | 18.9-75.1% | ±22.9 | 21.1-87.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 4.3-55.8% | 6.4-65.8% | 62.5-99.9% | ±10.1 | 82.7-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.7% | 87.6-99.9% | 88.4-100.0% | ±1.1 | 87.9-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 72.9-99.8% | 86.9-100.0% | 99.8-100.0% | ±0.4 | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 99.4-100.0% | ±0.6 | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 20.9-79.9% | 52.2-94.9% | 93.0-98.2% | ±4.9 | 48.7-96.5% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-1.2% | 1.5-7.3% | 59.7-66.9% | ±53.3 | 1.3-2.6% | - / - |

## Per family

Control rows scored on non-training windows report a FALSE POSITIVE rate. in-dist cells are mean ± sd over the seeds.

| round | family | kind | windows | cells | runs | baseline | prev-round | unseen | in-dist job_id | in-dist run_id | in-dist workload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A_util_modulation | evasion | 190 | 1 | 10 | 30.0% | 30.0% | 30.0% | 98.6% ± 0.3 | 93.9% ± 10.1 | 29.6% ± 0.7 |
| 1 | B_low_util | evasion | 190 | 1 | 10 | 9.5% | 9.5% | 21.1% | 98.0% ± 4.5 | 99.6% ± 0.4 | 18.9% ± 11.6 |
| 1 | D_temporal_disruption | evasion | 236 | 1 | 12 | 91.9% | 91.9% | 87.3% | 100.0% ± 0.0 | 100.0% ± 0.0 | 75.1% ± 22.9 |
| 2 | E_memory_minimal | evasion | 173 | 1 | 9 | 43.9% | 54.9% | 82.7% | 99.4% ± 0.0 | 99.4% ± 0.0 | 62.5% ± 10.1 |
| 2 | F_interleave | evasion | 346 | 2 | 18 | 4.3% | 6.4% | 100.0% | 99.9% ± 0.1 | 99.9% ± 0.2 | 99.9% ± 0.1 |
| 2 | G_clock_throttled | evasion | 117 | 1 | 6 | 15.4% | 65.8% | 99.1% | 99.1% ± 0.0 | 99.1% ± 0.0 | 99.1% ± 0.0 |
| 2 | H_mimicry | evasion | 419 | 2 | 22 | 8.6% | 15.0% | 96.4% | 100.0% ± 0.1 | 100.0% ± 0.1 | 99.9% ± 0.1 |
| 2 | I_stochastic | evasion | 172 | 1 | 9 | 55.8% | 54.1% | 92.4% | 96.2% ± 0.7 | 94.5% ± 0.7 | 90.1% ± 4.3 |
| 2 | J_pid | evasion | 228 | 1 | 12 | 4.8% | 36.4% | 100.0% | 99.7% ± 0.4 | 99.7% ± 0.4 | 99.7% ± 0.4 |
| 3 | K_online_learning | evasion | 348 | 1 | 18 | 89.1% | 87.6% | 87.9% | 89.7% ± 1.1 | 89.4% ± 0.4 | 88.4% ± 1.1 |
| 3 | L_diluted | evasion | 983 | 4 | 51 | 99.7% | 99.9% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | K_ddp | evasion | 592 | 1 | 24 | 99.8% | 99.8% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | K_ddp_accum | evasion | 1,016 | 2 | 48 | 99.8% | 100.0% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | L_ddp | evasion | 1,128 | 2 | 48 | 98.5% | 98.8% | 99.7% | 99.6% ± 0.6 | 99.9% ± 0.1 | 99.8% ± 0.4 |
| 4 | L_ddp_stagger | evasion | 480 | 1 | 24 | 72.9% | 86.9% | 100.0% | 100.0% ± 0.1 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | M_composite_memmin | evasion | 76 | 4 | 4 | 0.0% | 100.0% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | N_grad_accum | evasion | 57 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 4 | O_composite_idle_pad | evasion | 448 | 11 | 64 | 56.7% | 92.0% | 92.9% | 100.0% ± 0.0 | 100.0% ± 0.0 | 99.4% ± 0.6 |
| 5 | whitebox_full | evasion | 313 | 8 | 16 | 79.9% | 94.9% | 96.5% | 99.7% ± 0.0 | 99.0% ± 1.4 | 98.2% ± 2.0 |
| 5 | whitebox_lora | evasion | 115 | 3 | 6 | 20.9% | 52.2% | 48.7% | 86.4% ± 6.5 | 92.2% ± 4.0 | 93.0% ± 4.9 |
| 6 | symgemm_L4_fused_sgd | evasion | 585 | 5 | 9 | 1.2% | 1.5% | 2.6% | 88.1% ± 8.3 | 81.0% ± 2.8 | 66.9% ± 33.8 |
| 6 | symgemm_ppo_fused | evasion | 234 | 2 | 6 | 0.0% | 7.3% | 1.3% | 99.9% ± 0.2 | 99.7% ± 0.4 | 59.7% ± 53.3 |
| 0 | ppo_rlhf | control | 348 | 2 | 12 | 100.0% | 100.0% | - | 98.7% ± 0.8 | 99.0% ± 0.2 | 98.3% ± 1.9 |
| 0 | symgemm_L5_adam | control | 156 | 1 | 4 | 100.0% | 100.0% | - | 100.0% ± 0.0 | 100.0% ± 0.0 | 100.0% ± 0.0 |
| 0 | symgemm_negatives | control FP | 567 | 6 | 15 | 0.0% | 0.0% | - | 8.1% ± 0.4 | 7.7% ± 3.8 | 43.5% ± 30.9 |
| 0 | whitebox_inference | control FP | 42 | 1 | 2 | 0.0% | 0.0% | - | 7.6% ± 1.1 | 8.6% ± 3.2 | 8.6% ± 1.3 |
