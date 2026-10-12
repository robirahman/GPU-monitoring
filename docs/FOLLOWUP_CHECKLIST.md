# Multi node workload classification follow up checklist

Improve classification of useful training across nodes, implementations and hardware,
including adaptive evasion at a defensible false alert rate. This is the execution
checklist for the October 12 workload audit and its recommendations. Check an item
only when its linked artifact or result satisfies the stated completion condition.
Implementation, collection and scientific validation are separate milestones.

Baseline: [paper v1](https://arxiv.org/html/2606.19262v1), June 17, 2026.
Current accounting: 529 configurations, 2,889 reconstructed jobs, 5,592,907 samples
and 1,538.917 recorded accelerator hours, including known failures. Excluding the
21 reviewed failed jobs leaves 1,490.944 hours. These are device timestamp spans,
not useful training time. Historical job identities remain provisional.

## First execution sequence

- [x] Reconcile the main inventory with the completed four host TPU recovery and
  recovered NIC sidecars. Evidence: [recovery decisions](workload_review_2026-10-12/recovery/README.md).
- [x] Publish the [updated inventory snapshot](workload_review_2026-10-12/current/summary.json)
  and this checklist with their accounting and provenance.
- [x] Run the existing label, identity, ingestion, recovery and evaluation CPU
  regressions; record failures and fixes before rebuilding caches.
- [x] Check analysis input availability without hydrating the home filesystem;
  produce an exact missing object manifest for accelerator telemetry, NICs and logs.
- [ ] Restore the required analysis objects to durable external storage and verify
  their hashes. Do not require 20 GB of scope/waveform data for an NVML/NIC experiment.
- [ ] Rebuild content bound per stream and pooled job caches with the recovered
  logs and NIC sidecars. Record unresolved labels and sample/rank eligibility.
- [ ] Run the corrected exhaustive family holdouts and paired campaign ablation.
  Keep identical old job test windows in the full and ablated arms.
- [ ] Repeat the most weakly sampled existing families before interpreting their
  apparent generalization. Start with the campaign specified below.
- [x] Implement and test observed job hour false alert accounting and bounded
  onset-to-alert delay in the exhaustive evaluator. Real corpus results remain pending.
- [x] Generate the deterministic [repetition plan](followup_execution_2026-10-12/repetition_plan/summary.json):
  25 configurations, five fresh launches each, three day slots. No VMs launched.

## Corpus accounting and recovery

- [ ] Maintain one reproducible inventory linking raw objects, derived intervals,
  jobs, ranks, labels, completion evidence, quarantine and classifier admission.
  Keep accelerator, NIC and physical sensor hours separate.
- [ ] Reconcile current live GCS generations again after new collection; retain
  object generation, checksum, source URI and any transformation provenance.
- [ ] Resolve the three historical TPU LFS targets whose cloud copies are only
  pointers, or explicitly retain them as unquantified missing data.
- [ ] Adjudicate 35 completed older H100 interleaving/oracle jobs (44.678 raw
  accelerator hours): recover phase labels from the executed source version.
  Keep the documented weak filler separate from competent saturated attacks.
- [ ] Reassess low utilization TPU exclusions using completed device work and
  progress, without filtering on the detector's own input signal alone.
- [ ] Keep smoke tests, failed jobs, recording tails and derived predictions out
  of main corpus totals. Preserve failed prefixes for a separately defined test.
- [ ] Retain the eight short legacy JSON captures as supplemental evidence;
  do not count them as complete 30 second windows.
- [ ] Verify the known truncated waveform against the upstream instrument data;
  quarantine or replace it with documented provenance.
- [ ] Validate durable backup restoration, including LFS payloads rather than
  just pointers. Temporary `/tmp` caches are not a backup.

## Repetition and collection design

- [ ] Repeat local SGD, streaming/decoupled DiLoCo, SWARM, MoE, TP8/TP16,
  IMPALA, Ape X and distributed PPO. Twenty four of the 25 configurations in
  these groups currently have only one reconstructed execution.
- [ ] Target at least five independent launches per important configuration,
  spread over at least three collection days; record model/data/scheduler seeds.
  This is a starting design, not a guarantee of statistical precision.
- [ ] Use 30 to 60 minute recordings for priority cells and longer runs where
  needed to observe repeated slow synchronization and learning cycles.
- [ ] Add multi hour benign serving/HPC recordings to measure operational false
  alerts; retain startup, model load, warmup, idle and active phase annotations.
- [ ] Pair training, disguised training, inference and zero update controls on
  matched hardware, model scale, precision, context, batch and memory footprint.
- [ ] Randomize execution order while keeping both nodes on exactly the same
  distributed schedule. Separate repetitions from smoke tests and resumes.
- [ ] Extend selected difficult cases to four/eight GPU hosts and distinct
  interconnects/sites; do not substitute TPU host count for GPU scale coverage.

## Additional learning workloads

- [ ] Collect production dense and MoE training with combined DP/TP/PP and
  context parallelism, using an implementation independent of the custom probes.
- [ ] Cover communication overlap, activation recomputation/checkpointing, CPU
  optimizer/parameter offload, precision and memory efficient fine tuning.
- [ ] Collect real PPO/GRPO with cached rollouts, reward scoring and sparse
  learners; include learner placement and actor/learner ratio sweeps.
- [ ] Keep existing synthetic RL distinct: repeated prefill and synthetic token
  rewards are mechanism probes, not evidence of realistic RLHF learning utility.
- [ ] Implement and collect zeroth order fine tuning such as MeZO, with matched
  forward only controls, verified parameter updates and held out task improvement.
- [ ] Collect distributed recommendation and GNN training, not just sharded
  recommendation inference.
- [ ] Collect production diffusion/video and multimodal training in addition to
  the custom UNet and contrastive probes.
- [ ] Validate useful learning on real data for representative synthetic training
  families; distinguish same batch on every host from distinct data throughput.

References: [Megatron parallelisms](https://docs.nvidia.com/nemo/megatron-bridge/latest/parallelisms.html),
[verl GRPO](https://github.com/verl-project/verl/blob/main/docs/algo/grpo.md),
[MeZO](https://arxiv.org/abs/2305.17333),
[MLPerf Training](https://mlcommons.org/benchmarks/training/).

## Additional inference and non learning workloads

- [ ] Collect disaggregated prefill/decode with KV transfer between hosts.
- [ ] Collect speculative decoding, multi adapter serving and realistic mixed
  request lengths, rates and concurrency, with explicit request timing.
- [ ] Collect production distributed HPC applications, GPU ETL/data processing
  and vector search; retain synthetic communication and memory controls separately.
- [ ] Collect mixed tenant jobs and concurrent training/inference/non ML activity,
  varying the training share in both time and accelerator count.
- [ ] Keep independent replicas, sharded computation and communicating training
  as separate topology strata even when they use the same number of hosts.

Reference: [vLLM disaggregated prefill](https://docs.vllm.ai/en/latest/features/disagg_prefill/).

## Ground truth and useful adversarial work

- [ ] Define job objective, actual window activity, and useful learning separately.
  Learning includes parameter updates without backpropagation; a backward pass
  whose gradients are discarded is not sufficient evidence of parameter learning.
- [ ] Record per rank activity start/end, completed optimizer updates, useful
  tokens/examples, validation progress, checkpoint hashes and completion status.
- [ ] Retain unknown/unresolved labels; do not synthesize historical request or
  STEP events that were never recorded.
- [ ] Report throughput, elapsed time, energy and cost against matched honest
  training, alongside evasion success. Do not equate one minus logged training
  time fraction with the full utility/cost penalty.
- [ ] Distinguish compute from synchronization stalls inside logged training
  intervals; measure repeated useful progress after warmup.
- [ ] Treat oracle attacks that stop training after startup as utility failures,
  not sustained useful evasion. Require all worker completion/progress evidence.

## Adaptive attacks

- [ ] State sensor trust and attacker capabilities: scheduler, model, optimizer,
  rank placement, software, communication, query access and any telemetry tampering.
- [ ] Freeze and hash the defender before attack development. Save attack query
  budgets, search seeds, objective, cost constraints and every trial's provenance.
- [ ] Search combinations of dilution, scheduling, memory shaping, communication
  suppression/padding and sparse learner placement against that exact defender.
- [ ] Attack different window lengths and the actual temporal alert rule; include
  synchronization phase offsets and sparse intermittent updates.
- [ ] Reserve unseen final adaptive attacks from defender training/tuning; report
  separate known attack and unknown attack performance.
- [ ] Plot detection against preserved useful throughput, learning progress and
  energy/cost rather than reporting only pooled attack accuracy.

Reference: [adaptive evaluation](https://arxiv.org/abs/2002.08347).

## Generalization and operational evaluation

- [ ] Rerun corrected exhaustive mechanism family holdouts. The October 5 random
  family holdouts are historical; their changing test/model comparison is not a
  valid data addition effect. [Correction](workload_audit_2026-10-11.md).
- [ ] Run implementation/backbone holdouts, hardware/site holdouts, and a final
  chronological campaign holdout with no test based model or threshold selection.
- [ ] Keep every rank/window from a physical job and related attack/control
  siblings in the correct split; group correlated repetitions/campaigns explicitly.
- [ ] Compare GPU only, GPU plus NIC and richer instrumentation on identical
  eligible test windows and training populations, with split/feature provenance.
- [ ] Report training recall at prespecified false alert operating points, false
  alerts per benign job hour, job false positive rate, detection delay and misses.
- [ ] Select/calibrate thresholds on separate calibration jobs and freeze them
  before the final test. Record calibration failures rather than relaxing targets.
- [ ] Evaluate the actual aggregation rule on long benign runs; distinguish
  per window positives, alert episodes and jobs ever alerted.
- [ ] Report worst family/hardware/attack results, label source strata and
  confidence intervals resampled by independent job/campaign rather than windows.
- [ ] Test missing ranks, dropped samples, clock offsets, missing NICs and changed
  topology; preserve unavailable channels as missing rather than measured zero.
- [ ] Compare per rank and pooled job detectors, preserving sparse learner
  evidence; report observed host coverage and expected topology explicitly.
- [ ] Keep cross vendor signal semantics and normalization explicit; validate
  transfer separately from within vendor accuracy.

## Execution record

2026-10-12: local checkout `a98ff308`; historical telemetry payloads absent from
the local LFS cache, recovered files available. GCloud VM discovery reports no
active account. Shell GitHub DNS is unavailable; the connected GitHub API can
read the repository. Implementation and CPU validation can proceed locally.
Cloud authentication has been requested; do not paste credentials into this file.

Initial implementation and attempted execution are recorded in
[the execution report](followup_execution_2026-10-12/README.md). The full evaluation
stopped at input preflight: 10,048 missing LFS objects / 420,089,953 bytes.
The checklist remains open for collection, useful-learning validation, calibrated
operating points and real corpus results; passing CPU fixtures does not complete them.
