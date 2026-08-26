# Improving workload-grouped accuracy: +0.04 macro-F1, and it does not transfer

Baseline (30 s windows, workload-grouped 5-fold): **acc 0.8470 +- 0.060, macro-F1
0.7592**. Run-grouped for the same model is 0.9628, so 11.6 points of accuracy and
19.7 of macro-F1 were workload recognition.

## Metric 1 -- workload-grouped 3-class

| arm | acc | macro-F1 | vs base |
|---|---|---|---|
| baseline (sqrt, leaf2, depth None) | 0.8470 +- .060 | 0.7592 | -- |
| leaf20 | 0.8359 | 0.7573 | -0.0019 |
| leaf50 | 0.8264 | 0.7603 | +0.0011 |
| depth12 leaf10 | 0.8357 | 0.7578 | -0.0014 |
| depth8 leaf20 | 0.7708 | 0.7164 | -0.0428 |
| maxfeat0.3 leaf2 | 0.8466 | 0.7723 | +0.0131 |
| maxfeat0.3 leaf10 | 0.8470 +- .052 | 0.7791 | +0.0199 |
| maxfeat0.5 leaf10 | 0.8414 | 0.7747 | +0.0155 |
| shape-only baseline | 0.8474 | 0.7746 | +0.0154 |
| **shape-only maxfeat0.3 leaf10** | **0.8516 +- .050** | **0.7996** | **+0.0404** |
| xgb depth6 | 0.8557 +- .048 | 0.7863 | +0.0271 |
| xgb depth4 | 0.8526 | 0.7908 | +0.0316 |

Capacity reduction does NOT help -- every leaf/depth cap is flat-to-worse, and
depth8 is much worse. The 84.7% ceiling was not mainly an overfitting problem.

What helped was `max_features` going the OPPOSITE way: more candidates per split,
not fewer. Default `sqrt` offers 13 of 174; at 0.3 a split sees ~52. The audit
found 80% of top features are workload-identity ones, so with 13 random candidates
a split is often forced onto whichever identity feature is locally available.
Variance falls too (.060 -> .050), consistent with that mechanism. Note also that
the RF path previously IGNORED max_depth/n_estimators passed to __init__ -- it
hardcoded unlimited depth at min_samples_leaf=2, so earlier tuning attempts were
no-ops. Those knobs are now real (defaults unchanged).

## Metric 2 -- detection operating point (run-level, k=3 of 5, held out by workload)

The metrics DIVERGE, so a config must be judged on the one the claim rests on.

| config | best low-FPR point | F=0.2 at ~4-5% FPR |
|---|---|---|
| **current: all-174 sqrt leaf2** | **tau=0.80: 0.6% FPR, F=0.5 100%** | 0% |
| all-174 maxfeat0.3 leaf10 | tau=0.70: 2.4% FPR, F=0.5 100% | 50% (13.3% FPR) |
| shape-only maxfeat0.3 leaf10 | tau=0.70: 3.0% FPR, F=0.5 100% | **50% (4.2% FPR)** |
| xgb depth4/depth6 | none: 7.3-20% FPR at every tau | 50% (18-20% FPR) |

**The arm that wins metric 1 loses metric 2 in the high-precision regime.** At
tau=0.80 shape-only maxfeat0.3 gets F=0.5 run-TPR 0% where the current config gets
100% at 0.6% FPR. xgboost is unusable here: its probabilities saturate, so raising
tau to 0.90 still leaves 8.5% FPR.

Neither dominates. There is a Pareto frontier:
* lowest false alarms for half-duty training -> keep the current config (0.6% FPR,
  100% TPR at F=0.5, nothing at F=0.2);
* any F=0.2 sensitivity at all -> shape-only maxfeat0.3 leaf10 detects 50% of F=0.2
  runs at 4.2% FPR, where the current config detects 0% at 4.8%.

## Why this is coherent

The high-precision detection point depends on absolute level features -- exactly
the ones that fingerprint workloads. Dropping them improves generalisation to
unseen workloads (metric 1) and destroys the low-FPR detection point (metric 2).
That also explains the narrow usable tau band: the best operating point rests on
features that do not generalise across workloads.

Caveat: `max_features=0.3` was chosen on the same CV it is scored on, so +0.0404 is
mildly optimistic; a clean estimate needs nested CV. The metric-2 numbers use a
separate held-out-by-workload split, so the divergence itself is not an artifact of
that selection.

---

## AMENDMENT (corpus quarantine): most of the tuning gain was a contaminated baseline

Nine legacy TPU parquets with a pre-rewrite schema were entering the corpus as
ALL-ZERO feature vectors carrying real class labels (43 of 54,071 windows: 21
`ml_training`, 21 `ml_inference`, 1 inside `idle`). See
`gcs_c_quarantine/legacy_tpu_schema/README.md`. Re-measured on the clean corpus
(54,028 windows):

| | contaminated | clean |
|---|---|---|
| baseline acc | 0.8470 +- .060 | **0.8488 +- .045** |
| baseline macro-F1 | 0.7592 | **0.7661** |
| winner (shape-only mf0.3 leaf10) acc | 0.8516 +- .050 | **0.8396 +- .050** |
| winner macro-F1 | 0.7996 | **0.7794** |
| **winner gain** | **+0.0404** | **+0.0133** |

**The tuning result largely evaporates.** Most of the apparent +0.0404 was the
baseline being artificially depressed by 43 all-zero windows. Clean-to-clean the
winner gains +0.0133 macro-F1 and *loses* 0.0092 accuracy. Combined with the
already-established finding that this config gives up the high-precision detection
point (F=0.5 run-TPR 0% at tau=0.80 vs 100% for the current config), there is now
**no case for switching configs**. Keep `all-174 / sqrt / leaf2`.

The mechanism claim still stands on its own evidence: `max_features` helping while
every capacity reduction hurt, and the audit's 80% identity/decision feature overlap.
The effect size is just much smaller than reported.

Only the winning arm was re-measured. The other arms in the table above (maxfeat0.3
on all-174, maxfeat0.5, shape-only baseline, xgboost) are still contaminated-baseline
numbers and their deltas are inflated by roughly the same ~0.007 baseline shift.
Their relative ordering is unverified on clean data.

## What did NOT change

The operating point -- the number the surviving claim rests on -- is **identical**:

| tau | FPR_run | F=0 | F=0.05 | F=0.2 | F=0.5 |
|---|---|---|---|---|---|
| 0.70 | 1.8% | 0% | 0% | 0% | 100% |
| **0.80** | **0.6%** | 0% | 0% | 0% | **100%** |
| 0.85 | 0.0% | 0% | 0% | 0% | 0% |

Worst held-out honest workloads shift only marginally (`s8_nonml_lp_d50hash`
0.582 -> 0.574, `mining_ethash_proxy` 0.389 -> 0.407, `blender_bmw` 0.367 -> 0.290),
so the memory-bound-duty-cycle false-positive finding is unaffected.
