# Whole-corpus leave-one-family-out detection

Reference numbers for writing. **Not in the paper** — Table 7's LOO column in the
current draft is the published one; these are the regenerated equivalents on the
final 2026-11-05 corpus, kept here to refer to.

Regenerate with:

```
PYTHONPATH=. python3 scripts/paper_table7_loo.py                    # RandomForest
PAPER_MODEL=xgb PYTHONPATH=. python3 scripts/paper_table7_loo.py    # XGBoost
```

Source data: `paper_table7_loo_xgb.json`, `paper_table7_loo_rf.json`. 22 evasion families over 36,666 windows, 182 features, 30 s / 15 s.

## What this measures, and what it is not

Fit on **everything except family f** — every other evasion family, every round,
all the benign data — then predict f. It answers the deployment question: given
the finished corpus, would a new instance of this family be caught?

It is **not** the `unseen` column of `paper_table5_armsrace.py`, which fits on
`corpus_N` minus f, where `corpus_N` holds only rounds 1..N. A Round 1 family
there is scored by a detector that has seen nothing past Round 1, so the two
diverge sharply on early rounds and coincide exactly on Round 6 (the last round,
where `corpus_6` *is* the whole corpus). That coincidence is a free check that
the two scripts agree: both give 11.68% and 96.23%.

**The holdout key does not apply.** The family is held out whole and predicted in
one shot; there is no cross-validation, so there is no grouping to choose. That
holds only because no group straddles two families, which the script asserts on
every run rather than assuming:

```
groups spanning >1 family:  job_id 0  run_id 0  cell 0
```

Job-level and workload-family holdout therefore give the identical number.

## LOO mean

| | XGBoost | RandomForest | published |
|---|---|---|---|
| 22 evasion families | **89.86%** | 87.81% | — |
| the published 20 (no Round 6) | **93.45%** | 92.51% | 92.17% |
| window-weighted, 22 families | 89.76% | 87.54% | — |

The published LOO column essentially reproduces. The 22-family mean is lower only
because Round 6 joins it.

## Per family

| family | round | windows | cells | published | XGBoost | RandomForest | arms-race `unseen` |
|---|---|---|---|---|---|---|---|
| `symgemm_L4_fused_sgd` | 6 | 702 | 6 | — | **11.68%** | 5.13% | 11.7% |
| `whitebox_lora` | 5 | 115 | 3 | 42.6% | **49.57%** | 40.87% | 56.5% |
| `B_low_util` | 1 | 190 | 1 | 59.1% | **70.00%** | 38.42% | 30.0% |
| `O_composite_idle_pad` | 4 | 448 | 11 | 97.6% | **82.14%** | 94.64% | 84.6% |
| `whitebox_full` | 5 | 313 | 8 | 87.4% | **84.03%** | 95.21% | 91.1% |
| `I_stochastic` | 2 | 172 | 1 | 100.0% | **91.86%** | 93.60% | 92.4% |
| `K_online_learning` | 3 | 348 | 1 | 100.0% | **94.54%** | 94.54% | 87.4% |
| `symgemm_ppo_fused` | 6 | 823 | 7 | — | **96.23%** | 76.43% | 96.2% |
| `A_util_modulation` | 1 | 190 | 1 | 93.9% | **98.42%** | 98.95% | 34.2% |
| `D_temporal_disruption` | 1 | 236 | 1 | 100.0% | **99.15%** | 100.00% | 92.8% |
| `H_mimicry` | 2 | 419 | 2 | 99.1% | **99.28%** | 94.99% | 99.8% |
| `E_memory_minimal` | 2 | 173 | 1 | 98.2% | **100.00%** | 100.00% | 100.0% |
| `F_interleave` | 2 | 346 | 2 | 100.0% | **100.00%** | 100.00% | 100.0% |
| `G_clock_throttled` | 2 | 117 | 1 | 98.2% | **100.00%** | 99.15% | 99.1% |
| `J_pid` | 2 | 228 | 1 | 80.4% | **100.00%** | 100.00% | 100.0% |
| `K_ddp` | 4 | 592 | 1 | 100.0% | **100.00%** | 100.00% | 100.0% |
| `K_ddp_accum` | 4 | 1,016 | 2 | 100.0% | **100.00%** | 100.00% | 100.0% |
| `L_ddp` | 4 | 1,128 | 2 | 99.6% | **100.00%** | 99.82% | 100.0% |
| `L_ddp_stagger` | 4 | 480 | 1 | 100.0% | **100.00%** | 100.00% | 100.0% |
| `L_diluted` | 3 | 983 | 4 | 87.3% | **100.00%** | 100.00% | 99.8% |
| `M_composite_memmin` | 4 | 76 | 4 | 100.0% | **100.00%** | 100.00% | 100.0% |
| `N_grad_accum` | 4 | 57 | 3 | 100.0% | **100.00%** | 100.00% | 100.0% |

## Controls

Scored on the class each family actually carries. The four training-labelled
controls give a detection rate, where a miss is a false negative on an honest
job; the three non-training ones give a false-positive rate on a benign family
the detector has never seen.

| family | scored as | windows | XGBoost | RandomForest |
|---|---|---|---|---|
| `bestofn_rm` | false positive | 240 | **50.83%** | 98.75% |
| `ppo_rlhf` | detection | 348 | **51.72%** | 72.13% |
| `ppo_rlhf_accum` | detection | 240 | **100.00%** | 100.00% |
| `rl_distributed` | detection | 640 | **100.00%** | 100.00% |
| `symgemm_L5_adam` | detection | 156 | **100.00%** | 100.00% |
| `symgemm_negatives` | false positive | 1,039 | **58.81%** | 89.03% |
| `whitebox_inference` | false positive | 42 | **7.14%** | 7.14% |

Two things here are not in the paper:

- **Unseen benign families draw false positives about half the time.**
  `bestofn_rm` (best-of-n rejection sampling) and `symgemm_negatives` (the
  FLOP-matched symmetric-GEMM inference controls) are organic inference, and both
  are called training at 50.83% and 58.81% under XGBoost, 98.75% and 89.03% under
  RandomForest. This is a specificity limitation, and a second argument for the
  XGBoost switch in `80439b8e`, which rested on accuracy alone.
- **Honest PPO-RLHF training is missed 48% of the time** (51.72% XGB, 72.13% RF)
  when held out as a family. A false negative on a production workload nobody is
  hiding is arguably more policy-relevant than any evasion number in Table 7.

## Per-cell, where the family mean hides a spread

**`O_composite_idle_pad`** (detection)

| cell | XGBoost |
|---|---|
| `adversarial_composite_80` | 7.14% |
| `adversarial_composite_10` | 42.86% |
| `adversarial_composite_90` | 69.84% |
| `adversarial_composite_95` | 74.60% |
| `adversarial_composite_20` | 78.57% |
| `adversarial_composite_00` | 84.13% |
| `adversarial_composite_70` | 88.89% |
| `adversarial_composite_40` | 92.86% |
| `adversarial_composite_50` | 96.83% |
| `adversarial_composite_30` | 98.41% |
| `adversarial_composite_60` | 100.00% |

**`symgemm_L4_fused_sgd`** (detection)

| cell | XGBoost |
|---|---|
| `symgemm_L4_sgdfused` | 0.00% |
| `symgemm_L4_sgdfused_nov` | 0.00% |
| `symgemm_L4_sgdfused_short7b` | 0.00% |
| `symgemm_L4_sgdfused_long1b` | 0.42% |
| `symgemm_L4_sgdfused_7b` | 35.79% |
| `symgemm_L4_sgdfused_m8192` | 60.26% |

**`symgemm_ppo_fused`** (detection)

| cell | XGBoost |
|---|---|
| `symgemm_ppo_fused_r8` | 73.50% |
| `symgemm_ppo_fused_nov_r32` | 100.00% |
| `symgemm_ppo_fused_nov_r8` | 100.00% |
| `symgemm_ppo_fused_r16` | 100.00% |
| `symgemm_ppo_fused_r32` | 100.00% |
| `symgemm_ppo_fused_r4` | 100.00% |
| `symgemm_ppo_fused_r64` | 100.00% |

**`whitebox_full`** (detection)

| cell | XGBoost |
|---|---|
| `whitebox_oracle` | 72.09% |
| `whitebox_diluted_N10_ckpt` | 73.68% |
| `whitebox_diluted_N5_ckpt` | 74.36% |
| `whitebox_diluted_N10` | 79.49% |
| `whitebox_diluted_N5` | 87.18% |
| `whitebox_diluted_N20_ckpt` | 92.11% |
| `whitebox_diluted_N20` | 97.37% |
| `whitebox_diluted_N50` | 97.44% |

**`whitebox_lora`** (detection)

| cell | XGBoost |
|---|---|
| `whitebox_lora_N20` | 13.16% |
| `whitebox_lora_N10` | 64.10% |
| `whitebox_lora_N5` | 71.05% |

**`bestofn_rm`** (false positive)

| cell | XGBoost |
|---|---|
| `bestofn_rm_n8` | 1.67% |
| `bestofn_rm_n32` | 100.00% |

**`symgemm_negatives`** (false positive)

| cell | XGBoost |
|---|---|
| `symgemm_idle` | 0.00% |
| `symgemm_L1_fwd3_nov` | 5.98% |
| `symgemm_L1_fwd3_long1b` | 12.92% |
| `symgemm_L0_fwd1` | 53.21% |
| `symgemm_L1_fwd3` | 89.10% |
| `symgemm_L1_fwd3_7b` | 100.00% |
| `symgemm_L1_fwd3_7bmatch` | 100.00% |
| `symgemm_L1_fwd3_m8192` | 100.00% |
| `symgemm_L1_fwd3_short7b` | 100.00% |
| `symgemm_decode_g256` | 100.00% |

`symgemm_L4_fused_sgd` is the one to watch: the base cell, the November
replicate and `short7b` are all 0.00%, and only the `7b` and `m8192` scale
variants are partly caught. At the scale the paper reports, a detector trained on
the entire rest of the corpus does not see this attack at all.

