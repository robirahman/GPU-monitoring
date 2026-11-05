# The symgemm_ppo_fused rollout sweep is confounded by collection epoch

Status: **the Round 6 `symgemm_ppo_fused` row of
`paper_table5_armsrace_final_seeds.json` must not be used.** A matched
re-collection is in flight; this note records why.

## What was intended

`symgemm_ppo_fused` had two workload cells (r8, r32) against five CV folds, so
its workload-holdout number was bimodal: either both cells land in the same
test fold and the family reads its leave-one-family-out value (~1.3%), or they
split and it reads ~98%. Five fold-assignment seeds gave 2.1, 99.6, 0.4, 97.0,
99.1 — mean 59.7%, sd 53.3, and no seed anywhere near the mean. Three more
cells (r4, r16, r64) were collected to make every fold split partial.

## What actually happened

Per-cell baseline detection is **not monotone in rollout count**:

| cell | collected | baseline det. | prev.-round | unseen |
|---|---|---|---|---|
| r4   | 2026-11-05 |  98.3% | 100.0% | 100.0% |
| r8   | 2026-08-03 |   0.0% |  57.3% |  18.8% |
| r16  | 2026-11-05 | 100.0% | 100.0% | 100.0% |
| r32  | 2026-08-03 |  84.6% |  93.2% |  80.3% |
| r64  | 2026-11-05 | 100.0% | 100.0% | 100.0% |

If temporal dilution were the mechanism, detection would fall as rollout count
rises. It does not. The split is by collection date, and the raw telemetry
says why:

| collected | cells | mean util | mean power | SM clock | PCIe rx |
|---|---|---|---|---|---|
| 2026-08-03 | r8, r32      | 35.2% | 105 W | 1171 MHz | 80 MB/s |
| 2026-11-05 | r4, r16, r64 | 25.4% |  69 W | 1152 MHz | 52 MB/s |

The three November cells sit at 25.2%, 25.4% and 25.5% mean utilisation — a
16x change in rollout count moves utilisation by 0.3 points. The collection
epoch moves it by ten points and power by 35 W. Driver differs too
(580.126.16 vs 580.178.04), and the node was pinned to transformers 4.57.6
after the August run, which changes the `generate()` path the rollout phase
spends most of its time in.

So the sweep did not widen the family along the axis it is named for. The
rollout parameter has no detectable effect on 1 Hz telemetry at this scale;
what the detector separates is the software stack.

## Consequence

The family's baseline detection reads 76.7% with five cells where it read
0.0% with two. That change is an artifact. The seed spread did collapse — the
bimodality is gone — but the family is now heterogeneous for the wrong reason,
and 76.7% would misreport Round 6 as far more detectable than it is.

`symgemm_L4_fused_sgd` is unaffected: 0.0-5.9% per cell at baseline across all
five of its cells, which span both epochs. The Round 6 headline rests on L4
and still holds.

## Fix in flight

r8 and r32 re-collected on the November node as
`symgemm_ppo_fused_nov_r8` / `_nov_r32`, 3 reps each. That gives five cells in
one environment (nov_r8, nov_r32, r4, r16, r64) so the rollout axis is
isolated, and keeps the August r8/r32 as a matched cross-epoch pair — which
measures the batch effect itself, a reproducibility number the paper does not
currently have.

## Generalisation worth checking before camera-ready

If a driver bump and a transformers pin move mean power by a third on an
identical workload, then any corpus cell re-collected between April and
November may carry the same shift. The corpus spans 2026-03-09 to 2026-11-05
and rounds 1-4 were re-collected into late October. Nothing here shows that
has distorted another number, but nothing rules it out either, and the
cross-epoch pair above is the cheapest instrument for measuring it.
