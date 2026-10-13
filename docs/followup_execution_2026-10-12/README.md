# Follow up execution record

The first implementation adds exact analysis input auditing, a reproducible
evaluation runner, a repetition planner, and operational metrics in the corrected
exhaustive family evaluator. No new classification accuracy is claimed.

## Implemented and exercised

- `scripts/check_analysis_inputs.py` shares the cache input selection policy,
  hashes available payloads and emits all-input and missing-object CSVs. It excludes
  scope waveforms, quarantine and unrelated sensor files. Missing/corrupt payloads
  stop evaluation before a feature cache is built.
- `scripts/run_followup_evaluation.py` runs preflight, pooled 30-second/15-second
  windows, paired mechanism-family evaluation, backbone holdout and STEP-only
  evaluation. It pins the initial threshold at 0.5, records each phase and stops
  on failure. This is not a calibrated deployment operating point.
- `classifier/operational_metrics.py` reports benign observed job hours, false
  alert episodes per job hour, jobs alerted and onset-to-alert delay bounds.
  Overlapping windows count once toward exposure; unobserved gaps count zero.
  Decisions occur at window end, excluding computation/transport delay. Delay
  requires a STEP-backed negative followed by a STEP-backed training window;
  program-only onsets are excluded and left/gap-censored onsets are counted.
  Misses are separate from detected-only delay quantiles. Threshold selection,
  deployment calibration and other alert rules remain open checklist items.
- `scripts/plan_followup_repetitions.py` generated 125 planned launches for 25
  H100 configurations, five fresh launches each, over three day slots. Each is
  30 minutes on two eight-GPU hosts: 62.5 nominal pair-hours / 1,000 GPU-hours.
  The plan includes no startup/failure overhead or dollar estimate. Uniform
  workload seed support remains pending; a schedule seed is not a training seed.

## Actual execution outcome

The evaluation runner was invoked with `--run`. It stopped at preflight with
exit code 2: **10,048 unique LFS objects / 420,089,953 bytes** are missing,
affecting 12,340 selected paths. The complete selected inputs contain 14,716
paths / 667,994,709 unique bytes including logs. The 21 GB full LFS archive is
not required for this telemetry/NIC evaluation.

[Input summary](input_summary.json) and [execution state](evaluation_attempt.json)
record the attempt. Exact path/object manifests were generated in
`/tmp/spar-followup-20261012/inputs/`; regenerate them with the command below
if this temporary directory disappears. No classifier fitting or GPU collection
was performed by this blocked attempt.

The existing audit remediation, multinode review, telemetry recovery and work-log
collection suites, plus the new follow-up suite, pass **66 tests**. These cover
actual cache chunking on fixtures, exhaustive/paired model fitting on fixtures,
input corruption/missing data, overlap/gap accounting, censored onset and
repetition identity. [Validation record](validation.json).

GCloud VM discovery failed because no account was active. GitHub shell DNS is
also unavailable in this sandbox; commits are published through the connected
GitHub API. Local `.git` is read-only, so remote commits do not advance the local
branch reference in this session. The local edited files match the published files.

## Resume on a prepared analysis VM

Use a current checkout with the historical telemetry available in an external
LFS object cache. Choose durable paths sized for raw/feature caches and results;
do not materialize the corpus in the Cloud Shell home filesystem.

```sh
python3 scripts/check_analysis_inputs.py \
  --objects /durable/lfs/objects --out /durable/followup/inputs

python3 scripts/run_followup_evaluation.py \
  --objects /durable/lfs/objects --work /durable/followup/analysis --workers 2 --run
```

The runner rechecks inputs itself and records commands/status in `execution.json`.
Run it in a persistent terminal/service on the analysis VM and sync results to
durable object storage. Existing local files are used directly; the objects path
is only needed for unresolved pointers. Restore missing payloads using the emitted
OID/path manifest before retrying. Do not bypass preflight to score a partial corpus.

## Collection plan status

The [CSV](repetition_plan/repetitions.csv) is a plan, not an automatic scheduler.
Every `day_slot` must map to a distinct actual date and both nodes must use the
same ordered plan. Before running a bounded batch: select dedicated idle nodes,
smoke the workloads, wire/record independent seeds, preserve per-worker completion
and progress, and use durable sync plus peer-gated shutdown. The legacy queue
reaps existing compute processes, so it must not be launched on shared busy nodes.
Multi-day scheduling and uniform seed control remain incomplete.

Regenerate the plan from the inventory:

```sh
python3 scripts/plan_followup_repetitions.py \
  --out /durable/followup/repetition_plan --launches 5 --days 3 --duration 1800
```

## Authenticated collection follow-up

Google Cloud login is now active. Cloud API access from this restricted agent
session fails DNS resolution, so no VM was started. The [collection runbook](collection_runbook.md)
documents the implemented seeded pilot, runtime caps, checkpoint fixes and
verified-output protocol. The repetition plan now includes workload seeds.
Live smoke and collection remain pending; earlier input/evaluation records above
are unchanged.

## Live execution after network access was enabled

Cloud API, SSH and object storage access now work. All 14,716 selected inputs
(667,994,709 unique bytes) are restored and SHA256-verified on `spar-analysis-0`.
The first cache build hit its 24 GiB memory limit; avoidable dataframe copies
were removed and the evaluation relaunched in a separate directory.
The first executed A100 smoke completed TP8 on all 16 workers but was rejected
for missing final NIC files; a SIGTERM handling fix is pushed and the next
smoke passed. [Live execution record](live_execution.json) and
[verified input summary](verified_analysis_inputs.json) supersede the earlier
access/input-blocked status above. No new classifier results are claimed.

The repaired A100 smoke is now [verified](verified_a100_smoke.json): 1,776 GPU
samples and 222 NIC samples across 18 streams, with all 16 workers reporting
progress. These smoke samples stay out of the scientific corpus. The 30-minute
collection started at 23:48 UTC on October 12, but Google Cloud preempted one
Spot node at 23:51:50 UTC. The interrupted run received no completed-repetition credit;
[failure evidence](failed_a100_collection.json) records the interruption and
preserved artifacts. Both hosts stopped before a fresh smoke-gated retry started
at 01:14 UTC on October 13. That retry passed its smoke gate and launched
collection at 01:27 UTC, but a second Spot preemption interrupted it at 01:53 UTC.
Both GPU hosts are now stopped. [Retry evidence](failed_a100_retry.json) records
the partial artifacts and missing completion evidence; this retry also received no
completed-repetition credit. Both runs' valid observed intervals have since been
[recovered into the corpus](preempted_recovery.md): 23,530 GPU samples, 6.52844
accelerator-hours and 94 complete 30-second windows across 16 GPUs and two hosts.
Preemption status is retained; run completion is not required for interval admission.

The analysis retry passed the memory failure
point, then found mixed numeric/`N/A` sensor values at parquet serialization;
normalization is covered by a new round-trip regression and a third isolated
analysis attempt is running. At 02:54 UTC on October 13, the feature cache had
completed with 75,841 windows from 2,561 jobs, and paired mechanism-family
evaluation was running. Final accuracy and robustness results remain pending.
Current paths and status are in the live record.

The [duration/capacity review](duration_capacity_review.json) records why no
replacement is queued solely to reach 30 minutes for this TP8 pilot. Its recovered
23-minute interval already provides 90 complete windows. Future experiments that
require a specific uninterrupted horizon should use standard provisioning with
bounded runtime; the project has sufficient standard A100 quota, but this is not
proof of physical capacity. The full-length pilot milestone is still incomplete.
