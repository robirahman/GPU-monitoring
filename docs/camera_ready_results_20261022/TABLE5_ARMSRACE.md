# Table 5 — sequential arms race, scope-corrected

30 s / 15 s windows, RF 400 trees, class_weight=balanced, random_state=42, 5 folds.
Single-node only (<=8 GPUs per job, one host): `s16_train_ppodist_r4` (16 GPUs,
2 hosts, 648 windows) is the only job in the corpus that broke the rule and is
excluded. Tier A was already compliant, so the headline, Tables 1/2/4 and
Figures 1/3 are unaffected.

## Procedure

    corpus_0 = the benign corpus: every non-adversarial workload, controls included
    for N in 1..6:
        baseline det.     fit on corpus_0        -> predict round-N evasion families
        prev.-round det.  fit on corpus_{N-1}    -> predict round-N evasion families
        corpus_N = corpus_{N-1} + round-N EVASION families
        hardened in-dist. grouped 5-fold CV over corpus_N, out-of-fold on round N
        hardened unseen   per family f: fit on corpus_N minus f -> predict f

Only the in-distribution column cross-validates, so only it depends on the
holdout key; the other three are identical under all three protocols.

Primary variant follows Table 5's own caption -- "Baseline classifiers are
trained only on non-adversarial data" -- so only evasion families are
round-indexed. Controls (`symgemm_negatives`, `whitebox_inference`,
`symgemm_L5_adam`, `ppo_rlhf`) sit in corpus_0 throughout.

`symgemm_L3_graddiscard` is excluded from the corpus entirely: it computes
the weight gradient and discards it, so it is an instrument, not a workload
anyone organically runs. L0 and L1 are kept, being sustained batched forward
passes -- what any batch-inference, scoring or perplexity-eval job is.

**Caution on the workload-holdout in-distribution column.** 10 of the 22
families consist of a single workload cell, so under workload grouping the
whole family lands in one test fold and the "in-distribution" detector never
trained on it. For those families that column is not in-distribution at all,
and it is a single-fold draw with no averaging: `D_temporal_disruption` reads
96.6%, 80.5% and 3.0% across the three variants below on data that is
otherwise identical.


# primary — organic controls in Round 0, L3 excluded as an instrument  (`_final.json`)

Baseline row (no evasion): job_id 97.40%, run_id 99.41%, workload 83.15% accuracy over 25,922 windows (published 97.6%, workload holdout).


## in-dist holdout = `job_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-91.9% | 9.5-91.9% | 90.0-100.0% | 21.1-87.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 4.3-55.8% | 6.4-65.8% | 97.1-100.0% | 82.7-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.7% | 87.6-99.9% | 89.1-100.0% | 87.9-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 72.9-99.8% | 86.9-100.0% | 98.9-100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 100.0% | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 20.9-79.9% | 52.2-94.9% | 86.1-99.7% | 48.7-96.5% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-1.2% | 1.5-7.3% | 86.7-100.0% | 1.3-2.6% | - / - |

## in-dist holdout = `run_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-91.9% | 9.5-91.9% | 98.4-100.0% | 21.1-87.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 4.3-55.8% | 6.4-65.8% | 94.2-100.0% | 82.7-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.7% | 87.6-99.9% | 89.1-100.0% | 87.9-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 72.9-99.8% | 86.9-100.0% | 100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 100.0% | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 20.9-79.9% | 52.2-94.9% | 94.8-96.5% | 48.7-96.5% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-1.2% | 1.5-7.3% | 77.3-100.0% | 1.3-2.6% | - / - |

## in-dist holdout = `workload`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-91.9% | 9.5-91.9% | 30.0-54.7% | 21.1-87.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 4.3-55.8% | 6.4-65.8% | 46.2-100.0% | 82.7-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.7% | 87.6-99.9% | 89.1-100.0% | 87.9-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 72.9-99.8% | 86.9-100.0% | 99.1-100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 100.0% | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 20.9-79.9% | 52.2-94.9% | 95.8-98.3% | 48.7-96.5% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-1.2% | 1.5-7.3% | 2.1-69.7% | 1.3-2.6% | - / - |

## per family

Control rows scored on non-training windows report a FALSE POSITIVE rate, not detection.

| round | family | scored class | windows | cells | runs | baseline clean only | prev round | unseen | in dist job id | in dist run id | in dist workload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A_util_modulation | evasion | 190 | 1 | 10 | 30.0% | 30.0% | 30.0% | 98.9% | 98.4% | 30.0% |
| 1 | B_low_util | evasion | 190 | 1 | 10 | 9.5% | 9.5% | 21.1% | 90.0% | 99.5% | 30.0% |
| 1 | D_temporal_disruption | evasion | 236 | 1 | 12 | 91.9% | 91.9% | 87.3% | 100.0% | 100.0% | 54.7% |
| 2 | E_memory_minimal | evasion | 173 | 1 | 9 | 43.9% | 54.9% | 82.7% | 99.4% | 99.4% | 46.2% |
| 2 | F_interleave | evasion | 346 | 2 | 18 | 4.3% | 6.4% | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | G_clock_throttled | evasion | 117 | 1 | 6 | 15.4% | 65.8% | 99.1% | 99.1% | 99.1% | 99.1% |
| 2 | H_mimicry | evasion | 419 | 2 | 22 | 8.6% | 15.0% | 96.4% | 100.0% | 100.0% | 100.0% |
| 2 | I_stochastic | evasion | 172 | 1 | 9 | 55.8% | 54.1% | 92.4% | 97.1% | 94.2% | 89.0% |
| 2 | J_pid | evasion | 228 | 1 | 12 | 4.8% | 36.4% | 100.0% | 100.0% | 100.0% | 100.0% |
| 3 | K_online_learning | evasion | 348 | 1 | 18 | 89.1% | 87.6% | 87.9% | 89.1% | 89.1% | 89.1% |
| 3 | L_diluted | evasion | 983 | 4 | 51 | 99.7% | 99.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp | evasion | 592 | 1 | 24 | 99.8% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp_accum | evasion | 1,016 | 2 | 48 | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | L_ddp | evasion | 1,128 | 2 | 48 | 98.5% | 98.8% | 99.7% | 98.9% | 100.0% | 99.1% |
| 4 | L_ddp_stagger | evasion | 480 | 1 | 24 | 72.9% | 86.9% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | M_composite_memmin | evasion | 76 | 4 | 4 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | N_grad_accum | evasion | 57 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | O_composite_idle_pad | evasion | 448 | 11 | 64 | 56.7% | 92.0% | 92.9% | 100.0% | 100.0% | 100.0% |
| 5 | whitebox_full | evasion | 313 | 8 | 16 | 79.9% | 94.9% | 96.5% | 99.7% | 96.5% | 95.8% |
| 5 | whitebox_lora | evasion | 115 | 3 | 6 | 20.9% | 52.2% | 48.7% | 86.1% | 94.8% | 98.3% |
| 6 | symgemm_L4_fused_sgd | evasion | 585 | 5 | 9 | 1.2% | 1.5% | 2.6% | 86.7% | 77.3% | 69.7% |
| 6 | symgemm_ppo_fused | evasion | 234 | 2 | 6 | 0.0% | 7.3% | 1.3% | 100.0% | 100.0% | 2.1% |
| 0 | ppo_rlhf | control | 348 | 2 | 12 | 100.0% | 100.0% | - | 99.4% | 98.9% | 99.7% |
| 0 | symgemm_L5_adam | control | 156 | 1 | 4 | 100.0% | 100.0% | - | 100.0% | 100.0% | 100.0% |
| 0 | symgemm_negatives | control FP | 567 | 6 | 15 | 0.0% | 0.0% | - | 8.1% | 8.1% | 34.7% |
| 0 | whitebox_inference | control FP | 42 | 1 | 2 | 0.0% | 0.0% | - | 9.5% | 7.1% | 9.5% |

# sensitivity — L3 kept in Round 0  (`_ctrl0.json`)

Baseline row (no evasion): job_id 96.42%, run_id 99.36%, workload 90.74% accuracy over 26,078 windows (published 97.6%, workload holdout).


## in-dist holdout = `job_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-93.6% | 9.5-93.6% | 98.9-100.0% | 25.8-93.2% | 100% / 59-100% |
| 2 | 6 | 1,455 | 2.3-47.7% | 6.6-83.8% | 97.1-100.0% | 83.8-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.3% | 87.6-100.0% | 89.4-100.0% | 87.6-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 75.6-99.8% | 84.6-100.0% | 98.9-100.0% | 99.9-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 93.3-100.0% | 100.0% | 94.0-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 15.7-81.8% | 48.7-96.2% | 80.0-99.7% | 53.0-98.4% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-0.7% | 1.9-28.2% | 86.2-100.0% | 2.1-3.4% | - / - |

## in-dist holdout = `run_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-93.6% | 9.5-93.6% | 86.3-100.0% | 25.8-93.2% | 100% / 59-100% |
| 2 | 6 | 1,455 | 2.3-47.7% | 6.6-83.8% | 94.2-100.0% | 83.8-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.3% | 87.6-100.0% | 89.7-100.0% | 87.6-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 75.6-99.8% | 84.6-100.0% | 99.8-100.0% | 99.9-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 93.3-100.0% | 100.0% | 94.0-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 15.7-81.8% | 48.7-96.2% | 87.0-99.0% | 53.0-98.4% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-0.7% | 1.9-28.2% | 92.8-100.0% | 2.1-3.4% | - / - |

## in-dist holdout = `workload`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 9.5-93.6% | 9.5-93.6% | 0.0-31.1% | 25.8-93.2% | 100% / 59-100% |
| 2 | 6 | 1,455 | 2.3-47.7% | 6.6-83.8% | 64.2-100.0% | 83.8-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.3% | 87.6-100.0% | 86.5-100.0% | 87.6-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 75.6-99.8% | 84.6-100.0% | 100.0% | 99.9-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 93.3-100.0% | 99.6-100.0% | 94.0-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 15.7-81.8% | 48.7-96.2% | 93.9-97.4% | 53.0-98.4% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-0.7% | 1.9-28.2% | 82.6-97.4% | 2.1-3.4% | - / - |

## per family

Control rows scored on non-training windows report a FALSE POSITIVE rate, not detection.

| round | family | scored class | windows | cells | runs | baseline clean only | prev round | unseen | in dist job id | in dist run id | in dist workload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A_util_modulation | evasion | 190 | 1 | 10 | 30.0% | 30.0% | 30.0% | 98.9% | 86.3% | 31.1% |
| 1 | B_low_util | evasion | 190 | 1 | 10 | 9.5% | 9.5% | 25.8% | 100.0% | 100.0% | 0.0% |
| 1 | D_temporal_disruption | evasion | 236 | 1 | 12 | 93.6% | 93.6% | 93.2% | 100.0% | 100.0% | 3.0% |
| 2 | E_memory_minimal | evasion | 173 | 1 | 9 | 43.9% | 59.5% | 83.8% | 99.4% | 99.4% | 64.2% |
| 2 | F_interleave | evasion | 346 | 2 | 18 | 2.3% | 6.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| 2 | G_clock_throttled | evasion | 117 | 1 | 6 | 17.1% | 83.8% | 99.1% | 99.1% | 99.1% | 98.3% |
| 2 | H_mimicry | evasion | 419 | 2 | 22 | 7.6% | 19.1% | 96.4% | 100.0% | 100.0% | 96.7% |
| 2 | I_stochastic | evasion | 172 | 1 | 9 | 47.7% | 58.1% | 93.0% | 97.1% | 94.2% | 91.9% |
| 2 | J_pid | evasion | 228 | 1 | 12 | 22.4% | 50.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 3 | K_online_learning | evasion | 348 | 1 | 18 | 89.1% | 87.6% | 87.6% | 89.4% | 89.7% | 86.5% |
| 3 | L_diluted | evasion | 983 | 4 | 51 | 99.3% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp | evasion | 592 | 1 | 24 | 99.8% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp_accum | evasion | 1,016 | 2 | 48 | 99.8% | 100.0% | 100.0% | 99.8% | 100.0% | 100.0% |
| 4 | L_ddp | evasion | 1,128 | 2 | 48 | 98.3% | 98.6% | 99.9% | 98.9% | 99.8% | 100.0% |
| 4 | L_ddp_stagger | evasion | 480 | 1 | 24 | 75.6% | 84.6% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | M_composite_memmin | evasion | 76 | 4 | 4 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | N_grad_accum | evasion | 57 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | O_composite_idle_pad | evasion | 448 | 11 | 64 | 56.5% | 93.3% | 94.0% | 100.0% | 100.0% | 99.6% |
| 5 | whitebox_full | evasion | 313 | 8 | 16 | 81.8% | 96.2% | 98.4% | 99.7% | 99.0% | 93.9% |
| 5 | whitebox_lora | evasion | 115 | 3 | 6 | 15.7% | 48.7% | 53.0% | 80.0% | 87.0% | 97.4% |
| 6 | symgemm_L4_fused_sgd | evasion | 585 | 5 | 9 | 0.7% | 1.9% | 3.4% | 86.2% | 92.8% | 82.6% |
| 6 | symgemm_ppo_fused | evasion | 234 | 2 | 6 | 0.0% | 28.2% | 2.1% | 100.0% | 100.0% | 97.4% |
| 0 | ppo_rlhf | control | 348 | 2 | 12 | 100.0% | 100.0% | - | 98.9% | 99.1% | 96.0% |
| 0 | symgemm_L5_adam | control | 156 | 1 | 4 | 100.0% | 100.0% | - | 100.0% | 100.0% | 100.0% |
| 0 | symgemm_negatives | control FP | 723 | 7 | 19 | 0.0% | 0.0% | - | 7.1% | 28.6% | 55.6% |
| 0 | whitebox_inference | control FP | 42 | 1 | 2 | 0.0% | 0.0% | - | 11.9% | 7.1% | 7.1% |

# sensitivity — controls at their own round  (`_ppo6.json`)

Baseline row (no evasion): job_id 98.68%, run_id 99.36%, workload 88.48% accuracy over 24,809 windows (published 97.6%, workload holdout).


## in-dist holdout = `job_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 4.2-83.5% | 4.2-83.5% | 98.4-100.0% | 23.2-83.9% | 100% / 59-100% |
| 2 | 6 | 1,455 | 2.6-53.5% | 8.4-58.4% | 95.3-100.0% | 90.2-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 87.4-99.5% | 85.3-100.0% | 89.1-100.0% | 85.6-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 69.8-99.8% | 74.0-100.0% | 99.9-100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 100.0% | 94.0-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 40.9-49.5% | 68.7-92.3% | 93.9-95.7% | 56.5-93.0% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-7.5% | 0.4-4.4% | 86.2-100.0% | 2.1-3.4% | - / - |

## in-dist holdout = `run_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 4.2-83.5% | 4.2-83.5% | 98.4-100.0% | 23.2-83.9% | 100% / 59-100% |
| 2 | 6 | 1,455 | 2.6-53.5% | 8.4-58.4% | 95.3-99.8% | 90.2-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 87.4-99.5% | 85.3-100.0% | 89.1-100.0% | 85.6-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 69.8-99.8% | 74.0-100.0% | 100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 100.0% | 94.0-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 40.9-49.5% | 68.7-92.3% | 76.5-94.2% | 56.5-93.0% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-7.5% | 0.4-4.4% | 92.8-100.0% | 2.1-3.4% | - / - |

## in-dist holdout = `workload`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 4.2-83.5% | 4.2-83.5% | 28.4-96.6% | 23.2-83.9% | 100% / 59-100% |
| 2 | 6 | 1,455 | 2.6-53.5% | 8.4-58.4% | 60.7-100.0% | 90.2-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 87.4-99.5% | 85.3-100.0% | 94.3-100.0% | 85.6-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 69.8-99.8% | 74.0-100.0% | 99.7-100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 92.0-100.0% | 99.8-100.0% | 94.0-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 40.9-49.5% | 68.7-92.3% | 92.2-99.7% | 56.5-93.0% | 100% / 43-87% |
| 6 | 2 | 819 | 0.0-7.5% | 0.4-4.4% | 82.6-97.4% | 2.1-3.4% | - / - |

## per family

Control rows scored on non-training windows report a FALSE POSITIVE rate, not detection.

| round | family | scored class | windows | cells | runs | baseline clean only | prev round | unseen | in dist job id | in dist run id | in dist workload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A_util_modulation | evasion | 190 | 1 | 10 | 30.0% | 30.0% | 29.5% | 98.4% | 98.4% | 28.4% |
| 1 | B_low_util | evasion | 190 | 1 | 10 | 4.2% | 4.2% | 23.2% | 98.4% | 100.0% | 30.0% |
| 1 | D_temporal_disruption | evasion | 236 | 1 | 12 | 83.5% | 83.5% | 83.9% | 100.0% | 100.0% | 96.6% |
| 2 | E_memory_minimal | evasion | 173 | 1 | 9 | 43.9% | 58.4% | 90.2% | 99.4% | 99.4% | 60.7% |
| 2 | F_interleave | evasion | 346 | 2 | 18 | 2.6% | 8.4% | 100.0% | 100.0% | 99.7% | 100.0% |
| 2 | G_clock_throttled | evasion | 117 | 1 | 6 | 14.5% | 58.1% | 99.1% | 99.1% | 99.1% | 99.1% |
| 2 | H_mimicry | evasion | 419 | 2 | 22 | 8.4% | 14.3% | 95.2% | 100.0% | 99.8% | 85.7% |
| 2 | I_stochastic | evasion | 172 | 1 | 9 | 53.5% | 55.2% | 92.4% | 95.3% | 95.3% | 92.4% |
| 2 | J_pid | evasion | 228 | 1 | 12 | 3.1% | 39.9% | 100.0% | 100.0% | 98.7% | 99.6% |
| 3 | K_online_learning | evasion | 348 | 1 | 18 | 87.4% | 85.3% | 85.6% | 89.1% | 89.1% | 94.3% |
| 3 | L_diluted | evasion | 983 | 4 | 51 | 99.5% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp | evasion | 592 | 1 | 24 | 99.8% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp_accum | evasion | 1,016 | 2 | 48 | 99.6% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | L_ddp | evasion | 1,128 | 2 | 48 | 96.8% | 97.9% | 99.7% | 99.9% | 100.0% | 99.7% |
| 4 | L_ddp_stagger | evasion | 480 | 1 | 24 | 69.8% | 74.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | M_composite_memmin | evasion | 76 | 4 | 4 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | N_grad_accum | evasion | 57 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | O_composite_idle_pad | evasion | 448 | 11 | 64 | 57.8% | 92.0% | 94.0% | 100.0% | 100.0% | 99.8% |
| 5 | whitebox_full | evasion | 313 | 8 | 16 | 49.5% | 92.3% | 93.0% | 93.9% | 94.2% | 99.7% |
| 5 | whitebox_lora | evasion | 115 | 3 | 6 | 40.9% | 68.7% | 56.5% | 95.7% | 76.5% | 92.2% |
| 6 | symgemm_L4_fused_sgd | evasion | 585 | 5 | 9 | 7.5% | 4.4% | 3.4% | 86.2% | 92.8% | 82.6% |
| 6 | symgemm_ppo_fused | evasion | 234 | 2 | 6 | 0.0% | 0.4% | 2.1% | 100.0% | 100.0% | 97.4% |
| 6 | ppo_rlhf | control | 348 | 2 | 12 | 15.2% | 28.2% | - | 98.9% | 99.1% | 96.0% |
| 6 | symgemm_L5_adam | control | 156 | 1 | 4 | 100.0% | 100.0% | - | 100.0% | 100.0% | 100.0% |

# sensitivity — only PPO-RLHF moved to Round 0  (`_ppo0.json`)

Baseline row (no evasion): job_id 97.06%, run_id 99.31%, workload 89.12% accuracy over 25,157 windows (published 97.6%, workload holdout).


## in-dist holdout = `job_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 7.4-92.4% | 7.4-92.4% | 98.4-100.0% | 25.8-95.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 3.2-52.3% | 5.2-96.6% | 94.8-100.0% | 88.4-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.4% | 86.5-100.0% | 89.4-100.0% | 87.1-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 71.2-99.8% | 80.0-100.0% | 99.8-100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 94.0-100.0% | 100.0% | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 29.6-83.1% | 67.0-95.5% | 93.9-99.7% | 57.4-94.9% | 100% / 43-87% |
| 6 | 2 | 819 | 0.9-16.2% | 5.5-91.5% | 86.2-100.0% | 2.1-3.4% | - / - |

## in-dist holdout = `run_id`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 7.4-92.4% | 7.4-92.4% | 90.5-100.0% | 25.8-95.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 3.2-52.3% | 5.2-96.6% | 97.1-100.0% | 88.4-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.4% | 86.5-100.0% | 89.1-100.0% | 87.1-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 71.2-99.8% | 80.0-100.0% | 99.9-100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 94.0-100.0% | 100.0% | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 29.6-83.1% | 67.0-95.5% | 90.4-99.7% | 57.4-94.9% | 100% / 43-87% |
| 6 | 2 | 819 | 0.9-16.2% | 5.5-91.5% | 92.8-100.0% | 2.1-3.4% | - / - |

## in-dist holdout = `workload`

| Round | fam | windows | Baseline det. | Prev.-round det. | Hardened (in-dist.) | Hardened (unseen) | pub. in-dist / unseen |
|---|---|---|---|---|---|---|---|
| 1 | 3 | 616 | 7.4-92.4% | 7.4-92.4% | 16.3-80.5% | 25.8-95.3% | 100% / 59-100% |
| 2 | 6 | 1,455 | 3.2-52.3% | 5.2-96.6% | 96.0-100.0% | 88.4-100.0% | 100% / 80-100% |
| 3 | 2 | 1,331 | 89.1-99.4% | 86.5-100.0% | 91.4-100.0% | 87.1-100.0% | 100% / 87-100% |
| 4 (multi-GPU) | 4 | 3,216 | 71.2-99.8% | 80.0-100.0% | 100.0% | 99.7-100.0% | 100% / 100% |
| 4 (single-GPU) | 3 | 581 | 0.0-100.0% | 94.0-100.0% | 99.8-100.0% | 92.9-100.0% | 100% / 98-100% |
| 5 | 2 | 428 | 29.6-83.1% | 67.0-95.5% | 96.5-99.4% | 57.4-94.9% | 100% / 43-87% |
| 6 | 2 | 819 | 0.9-16.2% | 5.5-91.5% | 82.6-97.4% | 2.1-3.4% | - / - |

## per family

Control rows scored on non-training windows report a FALSE POSITIVE rate, not detection.

| round | family | scored class | windows | cells | runs | baseline clean only | prev round | unseen | in dist job id | in dist run id | in dist workload |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | A_util_modulation | evasion | 190 | 1 | 10 | 29.5% | 29.5% | 29.5% | 98.4% | 98.4% | 44.7% |
| 1 | B_low_util | evasion | 190 | 1 | 10 | 7.4% | 7.4% | 25.8% | 100.0% | 90.5% | 16.3% |
| 1 | D_temporal_disruption | evasion | 236 | 1 | 12 | 92.4% | 92.4% | 95.3% | 100.0% | 100.0% | 80.5% |
| 2 | E_memory_minimal | evasion | 173 | 1 | 9 | 43.9% | 65.3% | 88.4% | 98.8% | 99.4% | 96.0% |
| 2 | F_interleave | evasion | 346 | 2 | 18 | 3.2% | 5.2% | 99.7% | 100.0% | 99.7% | 100.0% |
| 2 | G_clock_throttled | evasion | 117 | 1 | 6 | 15.4% | 96.6% | 99.1% | 99.1% | 99.1% | 99.1% |
| 2 | H_mimicry | evasion | 419 | 2 | 22 | 7.4% | 16.2% | 96.9% | 100.0% | 100.0% | 100.0% |
| 2 | I_stochastic | evasion | 172 | 1 | 9 | 52.3% | 53.5% | 93.0% | 94.8% | 97.1% | 98.3% |
| 2 | J_pid | evasion | 228 | 1 | 12 | 22.8% | 36.8% | 100.0% | 100.0% | 100.0% | 99.6% |
| 3 | K_online_learning | evasion | 348 | 1 | 18 | 89.1% | 86.5% | 87.1% | 89.4% | 89.1% | 91.4% |
| 3 | L_diluted | evasion | 983 | 4 | 51 | 99.4% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp | evasion | 592 | 1 | 24 | 99.8% | 99.8% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | K_ddp_accum | evasion | 1,016 | 2 | 48 | 99.7% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | L_ddp | evasion | 1,128 | 2 | 48 | 97.3% | 98.2% | 99.7% | 99.8% | 99.9% | 100.0% |
| 4 | L_ddp_stagger | evasion | 480 | 1 | 24 | 71.2% | 80.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | M_composite_memmin | evasion | 76 | 4 | 4 | 0.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | N_grad_accum | evasion | 57 | 3 | 3 | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% | 100.0% |
| 4 | O_composite_idle_pad | evasion | 448 | 11 | 64 | 58.7% | 94.0% | 92.9% | 100.0% | 100.0% | 99.8% |
| 5 | whitebox_full | evasion | 313 | 8 | 16 | 83.1% | 95.5% | 94.9% | 99.7% | 99.7% | 99.4% |
| 5 | whitebox_lora | evasion | 115 | 3 | 6 | 29.6% | 67.0% | 57.4% | 93.9% | 90.4% | 96.5% |
| 6 | symgemm_L4_fused_sgd | evasion | 585 | 5 | 9 | 16.2% | 5.5% | 3.4% | 86.2% | 92.8% | 82.6% |
| 6 | symgemm_ppo_fused | evasion | 234 | 2 | 6 | 0.9% | 91.5% | 2.1% | 100.0% | 100.0% | 97.4% |
| 6 | symgemm_L5_adam | control | 156 | 1 | 4 | 100.0% | 100.0% | - | 100.0% | 100.0% | 100.0% |
| 0 | ppo_rlhf | control | 348 | 2 | 12 | 100.0% | 100.0% | - | 98.9% | 99.1% | 96.0% |
