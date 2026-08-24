# MN-I re-run with a saturating filler: two retractions and a real result

Detector: persisted `classifier/models/honest_detector.joblib`, fit 2026-08-21,
28269 honest windows, 174 features. Every cell below is **out-of-sample** -- the
artifact predates the runs and was not refit.

## 1. Telemetry realism (the thing that invalidated the first ladder)

Per-GPU means over the full 600 s run:

| cell | mem | util | power |
|---|---|---|---|
| honest `cufft_benchmark` (reference) | 1845 MB | 91.7% | 509 W |
| honest non-ML, all (reference) | 1794 MB | 73.1% | 300 W |
| honest training (reference) | 15544 MB | 78.1% | 386 W |
| `nonml_sat` (dummy buffer, no model) | 5078 MB | 96.8% | 538 W |
| `advtrain_satfiller_only` (real model) | 5213 MB | 96.6% | 541 W |
| `nonml_himem` (honest, no model) | 18653 MB | 96.6% | 539 W |
| `nonml_lowpower` (honest, duty 0.3) | 5635 MB | 38.7% | 285 W |

The OLD filler ran at 55% / 160 W -- outside *both* classes. The new one lands on
`cufft_benchmark`. `nonml_sat` vs `satfiller_only` differ by 135 MB, 0.2 points of
utilisation and 3 W, so that pair isolates the apparatus and nothing else.

## 2. p_train (S16 / S8 / S1)

| cell | new filler | old filler |
|---|---|---|
| `nonml_sat` (no model) | 0.054 / 0.058 / 0.095 | -- |
| `advtrain_satfiller_only` (real model, F=0) | **0.093 / 0.066 / 0.081** | 0.490 / 0.564 / 0.436 |
| `nonml_himem` (18.6 GB, honest) | 0.069 / 0.065 / 0.091 | -- |
| `nonml_lowpower` (honest, duty 0.3) | **0.323** (S16+S8) | -- |
| `advtrain_satint_f05` | 0.320 / 0.271 / 0.132 | -- |
| `advtrain_satint_f20` | 0.470 / 0.389 / 0.249 | -- |
| `advtrain_satint_f50` | 0.777 / 0.799 / 0.376 | 0.837 / 0.882 / 0.848 |

Honest H100 reference on the same detector: non-ML 0.001, training 0.804.

## RETRACTION 1 -- the apparatus is not the detection channel

`nonml_sat` 0.054 vs `advtrain_satfiller_only` 0.093 at matched telemetry. Swapping a
real 1.3B sharded transformer for a dead buffer moves p_train by ~0.04. The earlier
claim attributed ~0.49 to the apparatus. Confirms the `nonml_hpc` control.

## RETRACTION 2 -- memory footprint is not the channel either

`nonml_himem` holds 18653 MB, *above* honest training's 15544 MB, is genuinely
non-ML, and scores 0.069. The "memory is the discriminator" reading of the
nonml_hpc result was wrong. Both old anomalies are explained by extrapolation: the
old filler sat in a region no class populated, so the classifier returned noise.

## RESULT -- a competent filler evades, and the ladder is a cost curve

F=0 with a saturating filler scores 0.066-0.093 (~5% of windows over 0.10) versus
0.436-0.564 at 100% with the broken filler. "Caught at every train fraction" was an
artifact. The detector responds to *actual training compute*: p_train rises
monotonically in F at every scope.

## THE BLOCKER -- tau=0.1 is unusable, and F=0.05 is not separable

`nonml_lowpower` is HONEST and scores 0.323, **above** `satint_f05` at 0.295.
Window-alert rates, S16+S8 pooled:

| tau | lowpower (HONEST = FP) | satint_f05 | satint_f20 | satint_f50 |
|---|---|---|---|---|
| 0.10 | 100.0% | 97.5% | 100.0% | 100.0% |
| 0.35 | 17.5% | 10.0% | 98.8% | 100.0% |
| 0.40 | 2.5% | 3.8% | 60.0% | 100.0% |
| 0.50 | 0.0% | 0.0% | 16.2% | 100.0% |

No threshold separates F=0.05 from an honest half-idle HPC job -- the honest job
alerts *harder*. tau=0.1, used throughout the earlier ladder, is 100% FP on honest
work.

What survives: at tau ~0.40-0.50 the detector catches F >= 0.2 at <=2.5% FP. The
defensible claim is a **cost**: evading requires capping real training near 5% duty,
a ~20x slowdown. That is stronger than "always caught" because it holds against a
competent disguise.

Caveat: this FP estimate rests on ONE honest workload at 6 runs. 18 further honest
partial-utilisation cells (duty x allocation x kernel-mix) are collecting under
stamp 20260825fp to calibrate tau properly.

## Excluded

`s1_nonml_comm` from the earlier stage measured a Python spin loop (all_reduce is a
no-op at world_size=1, one STEP line per iteration, 1.44 GB log). Fixed by
slot-bounding; exclude that cell from any earlier table.
