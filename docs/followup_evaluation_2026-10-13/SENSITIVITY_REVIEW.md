# Reviewed sensitivity results

The complete mechanism evaluation and overall interpretation are now available in [the main review](REVIEW.md).

Both complete sensitivity bundles passed identity, family/job split, exhaustive coverage, and metric-recomputation checks against the frozen cache. Backbone accuracy is 68.97% (macro F1 0.686); STEP-only accuracy is 73.13% (macro F1 0.605). These are different protocols/populations, not a paired causal comparison.

The fixed 0.5 threshold is not suitable for an operational robustness claim: backbone holdouts alert on 49 of 273 entirely benign jobs (17.95%), with 175 alert episodes over 46.175 observed benign job-hours (3.790/hour). STEP-only holdouts alert on 35 of 145 benign jobs (24.14%), with 91 episodes over 25.425 hours (3.579/hour). Calibration on separate jobs and long benign recordings is a priority.

STEP-only pooled accuracy hides poor inference recognition: only 304/1,268 inference windows are correctly classified (23.97%). Its class mixture differs from the full corpus, explaining why a higher accuracy need not mean better three-class discrimination.

Existing adversarial training-window recall is 58.22% under the shared-backbone holdout and 74.00% under STEP-only mechanism holdouts. STEP-only detects at least one training window in every one of the 91 attack jobs, but misses about 26% of their training windows. This is not an unseen adaptive-attack guarantee. Four attack mechanism families are represented.

SWARM training has zero held-out training recall in both sensitivity evaluations, based on only two jobs/78 windows. Backbone-holdout MoE recall is 2.52% (three jobs/119 windows). Matched repetitions and independent implementations of these mechanisms deserve priority; repetitions improve evidence and uncertainty, not automatically the family's own leave-family-out score.

The smallest training-share stratum contains only 26 windows from 24 jobs. Backbone recall there is 26.92%, STEP-only recall 42.31%; this stratum includes all STEP-labelled workloads and must not be presented as an attack-only result or a throughput/learning-utility guarantee.

The two recovered preempted A100 captures are retained in the corpus and excluded only from this already-frozen evaluation snapshot. They should enter a separately versioned subsequent evaluation.
