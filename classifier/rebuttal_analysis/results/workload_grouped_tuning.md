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
