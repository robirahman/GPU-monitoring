# Review and implications for the follow-up paper

The corrected exhaustive mechanism holdout scores **71.14% accuracy**, macro F1
**0.704**, and **69.23% training-window recall** at the fixed 0.5 threshold, across
19,298 windows from 479 physical jobs and all 43 eligible mechanism families.
The backbone and STEP-only sensitivities score 68.97% and 73.13% accuracy,
respectively. These results provide a reproducible baseline and expose substantial
classification and operational weaknesses.

## The campaign addition does not demonstrate an accuracy improvement

On the same 18,131 old-job test windows, adding the October 1/3 campaigns to the
training folds changes pooled accuracy from **75.44% to 70.88%** (−4.56 percentage
points). The equally weighted paired-family change is **−0.57 points**, with a
95% family-bootstrap interval **[−5.11, +3.62]** across 34 paired families. The
window-weighted and family-weighted estimates answer different questions; the
latter interval includes both improvement and deterioration. Neither supports an
improvement claim for this fixed model. This is an October campaign ablation,
not a direct comparison against paper v1. It does not isolate a universal effect
of collecting more data or estimate uncertainty across independently trained seeds.

The 26 newly included jobs contain 1,167 test windows, overwhelmingly training;
75.15% accuracy there accompanies macro F1 0.399. Report this class imbalance.
Reconsider training weights and model choices using fresh development/calibration
splits, then reserve a new final test. Do not tune on these published outer folds
and continue calling them untouched test results.

## Robustness and false alerts remain limiting

Existing held-out adversarial workloads have **74.75% training-window recall**.
Their high any-window job recall does not establish continuous detection, learning
utility, or robustness to fresh adaptive attacks. Only four eligible attack
mechanism groups are represented. The detailed report separately counts jobs
alerted during an actual training-labelled window.

At threshold 0.5, the main evaluation alerts on **49/273 benign jobs (17.95%)**,
with **175 alert episodes over 46.175 observed benign job-hours (3.790/hour)**.
These are held-out family predictions on short recordings, not a deployment rate
from long stationary benign traffic. Calibration and the actual temporal alert
rule need separate jobs and longer recordings. Among 100 onset-eligible training
jobs, three are missed; 70 other STEP-backed onsets are censored. The 0–30 second
median delay bounds apply only to detected eligible jobs and omit transport and
computation latency.

## The pooled score hides specific failures

- SWARM training recall is **0%** (two jobs, 78 windows), MoE **2.52%** (three jobs,
  119 windows), and LoRA **12.87%** (eight jobs, 334 windows). Repeat these families
  with independent seeds and days, verify useful progress, and add independent
  implementations. Repetitions improve the evidence; they do not automatically
  improve a family's own leave-family-out score because that family is withheld.
- Type-confusion attack training recall is **49.74%**. Collect fresh adversarial
  combinations against a frozen defender with recorded query budgets and useful
  learning/throughput constraints.
- ResNet inference and recommendation inference have zero three-class accuracy;
  sharded inference accuracy is below 1%. SDXL inference triggers training alerts
  on **93.41%** of its windows. Add matched inference/training/zero-update controls
  and production serving recordings rather than relying on pooled accuracy alone.
- The STEP-only sensitivity recognizes just **304/1,268 inference windows
  (23.97%)** despite its higher total accuracy. TPU v6e likewise has zero inference
  recall (0/388) in the main hardware stratum. Hardware strata have different
  workload mixes and are not hardware-transfer tests.

## Next work, in priority order

1. Establish independent development, calibration and final-test job/campaign
   groups, including correlated retry/attack-control grouping. Specify a false-alert
   budget and collect enough long benign serving/HPC exposure to evaluate it.
2. Repeat the failure families above and retain interrupted valid intervals. Add
   strong activity/progress ground truth and matched controls; quantify class,
   hardware, implementation and campaign confounding.
3. Compare models, weighting, GPU-only/NIC features and temporal aggregation using
   development data. Run missing-sensor/rank/clock tests and independent hardware,
   implementation and chronological holdouts with the chosen detector frozen.
4. Develop new adaptive attacks, measure useful learning and cost, and reserve
   unseen final attacks. Add job/campaign-aware uncertainty and model-seed repeats.
5. Run a separately versioned evaluation incorporating the two recovered
   preempted A100 intervals. Their admission remains valid; this frozen snapshot
   intentionally predates it.

The full numeric report, all per-family rows, saved predictions and split manifests
are linked from [README.md](README.md). The checklist marks only completed
milestones; calibration, adaptive robustness and the broader collection program
remain open.
