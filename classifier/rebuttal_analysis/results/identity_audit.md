# Do the features encode workload identity? Yes. Does removing them help? No.

## The features are a workload fingerprint

RandomForest predicting WORKLOAD IDENTITY from the 174 window features, with the
**run** held out (so identity must come from the features, not from having seen
that exact run):

| population | workloads | identity accuracy | majority baseline | ratio |
|---|---|---|---|---|
| all honest | 119 | **88.9%** | 4.8% | 18.4x |
| honest non-ML only (where FPR lives) | 16 | **97.5%** | 25.2% | 3.9x |

Top-K feature overlap between "name the workload" and "decide ML vs non-ML":
**80%** at K=10 and K=25, 70% at K=50. Both are dominated by absolute memory
statistics -- `mem_used_mb_{max,p75,p95,p50,mean,p25,min}`,
`first_30s_mem_delta_mb`, `time_to_mem_plateau_samples`.

So a run-grouped CV score is substantially a recognition score: with ~2100 runs
over ~170 workloads, every held-out run has same-workload siblings in the training
fold. Model selection now groups by workload (`workload_group()`); the CV reports
both groupings and their gap.

## But the implied fix does not work

If the FPR were caused by identity leakage through level features, dropping them
should help. It does not. Three arms, held-out-by-workload FPR on substantive busy
honest non-ML vs ladder TPR (window-level):

| arm | FPR@0.10 | FPR@0.50 | F=0.5 TPR@0.80 |
|---|---|---|---|
| A all-174, stage-2 defence OFF | 53.9% | 4.6% | 0.0% |
| B all-174, stage-2 defence ON (real model) | 52.2% | 3.9% | **55.0%** |
| C shape-only input, level dropped from BOTH stages | 54.8% | 3.7% | 36.2% |

Arm C is worse than B. Absolute level features carry genuine ML-vs-non-ML signal,
not merely identity, and the two are entangled in the same columns. The identity
leak is real and worth documenting, but removing level features is not the remedy.
Hypothesis tested, not supported.

## Correction to the previous commit

The calibration behind `heldout_calibration.md` called `clf.fit(X, y)` without
`feature_cols`, hitting TwoStageClassifier's documented fallback ("don't filter
stage 2 features if we weren't told the names"). That silently disabled the WB-O
level-feature defence, so those numbers describe arm A, an unprotected model.

The qualitative conclusion is unchanged -- only F=0.5 is detectable, evasion cost
~2x -- but the achievable operating point is much better than reported, because
arm A hid the entire high-precision regime (0% TPR at tau=0.80, vs 55% for the
real model).

## Corrected operating point, run-level, k=3 of m=5

| tau | FPR_run | FPR_win | F=0 | F=0.05 | F=0.2 | F=0.5 |
|---|---|---|---|---|---|---|
| 0.50 | 4.8% | 3.9% | 0% | 0% | 0% | 100% |
| 0.70 | 1.8% | 1.3% | 0% | 0% | 0% | 100% |
| **0.80** | **0.6%** | **0.1%** | 0% | 0% | 0% | **100%** |
| 0.85 | 0.0% | 0.0% | 0% | 0% | 0% | 0% |

**tau=0.80 detects every F=0.5 run at a 0.6% held-out false-alarm rate.** The only
honest workload that false-alarms is `mining_ethash_proxy`, on 3% of its runs. At
0.85 both rates collapse to zero, so the usable band is narrow but real.

F=0.2 and F=0.05 remain 0% everywhere in this band: the earlier finding stands, and
the evasion cost is still ~2x rather than the ~20x originally claimed.
