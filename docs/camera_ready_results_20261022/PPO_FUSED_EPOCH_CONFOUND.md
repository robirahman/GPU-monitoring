# A collection-epoch effect, scoped to the `generate()` path

A `transformers` version change between the August and November collections
moves detection of an identical workload by up to 100 points — but only for
cells that spend their time in `model.generate()`. The GEMM ladder that Round 6
rests on is unaffected, and Round 6's claim has now been replicated across both
epochs.

## The effect

`symgemm_ppo_fused_r8`, same program, same rollout count, same GPU model:

| | collected | baseline det. | mean util | mean power |
|---|---|---|---|---|
| `symgemm_ppo_fused_r8` | 2026-08-03 | **0.0%** | 35.3% | 107 W |
| `symgemm_ppo_fused_nov_r8` | 2026-11-05 | **100.0%** | 25.5% | 70 W |
| `symgemm_ppo_fused_r32` | 2026-08-03 | 81.2% | 35.8% | 103 W |
| `symgemm_ppo_fused_nov_r32` | 2026-11-05 | 99.2% | 25.8% | 69 W |

Driver 580.126.16 → 580.178.04, and the node was pinned to transformers 4.57.6
after the August run (`requirements_v2.txt:53` names 5.20.2, which disables
PyTorch under torch 2.5.1).

## What it does NOT affect

`symgemm_L4_sgdfused`, re-collected for exactly this test, with its matched
FLOP-equivalent control:

| cell | epoch | baseline | prev.-round | unseen | util | W |
|---|---|---|---|---|---|---|
| `symgemm_L4_sgdfused` | Aug | 0.0% | 0.0% | 0.0% | 97.5% | 387 |
| `symgemm_L4_sgdfused_nov` | Nov | 0.0% | 0.9% | 0.0% | 96.0% | 346 |
| `symgemm_L1_fwd3` (control) | Aug | 0.0% FP | 0.0% FP | — | 98.5% | 391 |
| `symgemm_L1_fwd3_nov` (control) | Nov | 0.0% FP | 0.0% FP | — | 97.8% | 348 |

Zero to zero, on both the attack and its control.

## The mechanism, and the boundary it draws

| cell type | time spent in | Aug → Nov util | detection |
|---|---|---|---|
| rollout / decode | `model.generate()` | 35% → 25% | 0% → 100% |
| GEMM ladder | `torch.matmul` loop | 97.5% → 96.0% | 0% → 0% |

The library change altered the generation path. L4 saturates the card at ~97%
in both epochs because it never calls it.

  affected    symgemm_ppo_fused, symgemm_decode_g256, ppo_rlhf, bestofn_rm
  unaffected  symgemm_L0, L1, L3, L4, L5

## Consequences

1. **Round 6's headline stands.** `symgemm_L4_fused_sgd` reads 0.3% baseline
   detection across six cells spanning both epochs, against a matched control
   at 0% false positives in both. The symmetric-GEMM disguise is real and now
   has a cross-epoch replication, which the single August collection did not.

2. **`symgemm_ppo_fused` cannot be reported as an evasion success.** With the
   environment held constant, rollout count does essentially nothing: r4 94.9,
   nov_r8 100.0, r16 99.2, nov_r32 99.2, r64 100.0, while update density falls
   16x (74 to 5 optimiser steps per 600 s) and mean utilisation moves 0.4
   points. The original 0.0% was the August `generate()` path, not the disguise.
   Temporal dilution of the policy update does not register at 1 Hz.

3. **Open editorial choice.** The family's Round 6 row currently pools seven
   cells across two software stacks. Keeping the August pair preserves the data
   and shows the spread; dropping it makes the row single-environment. Both are
   defensible; the table is generated with all seven so the spread is visible.

## What this does not tell us

Only the symgemm family was re-collected. The corpus spans 2026-03-09 to
2026-11-05 and rounds 1-4 were re-collected into late October, so other cells
may carry their own environment shifts. Nothing here shows that they do. The
cheap check for any suspect cell is the one run here: re-collect it and compare,
which costs about 20 minutes of single-GPU time per cell.
