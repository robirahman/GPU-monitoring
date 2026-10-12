# Workload audit corrections — 2026-10-11

The October 5 results are historical random family holdouts, not exhaustive
LOFO. Their full/ablation comparison changes both the test population and the
selected model. Do not use that difference as evidence of a data-addition effect.
The corrected implementation requires rebuilt caches; no replacement accuracy
claim is made until those experiments are run.

## Completed inventory

The final scan verified all 13,675 LFS objects (21,121,462,961 bytes) against
their SHA256 hashes. It inspected 16,208 tracked Parquet paths, eight legacy
telemetry JSON files, and 422 raw-waveform paths, deduplicating identical objects
and overlapping stream timestamps. The saved report is
`/home/robi/workload-audit-2026-10-11/inventory.html`; raw objects remain in
`/tmp/gpu-workload-audit/lfs/objects`.

Outside quarantine there are 9,913 accelerator streams, 5,590,675 samples,
529 canonical workload cells and 2,888 reconstructed jobs. Their timestamp
spans total 1,538.298 accelerator-hours (1,538.285 hours after capping gaps).
These totals include unverified and failed collections: they measure available
telemetry, not a validated training set. The quality manifest excludes 21
explicitly failed jobs, accounting for 47.973 accelerator-hours. Most legacy
jobs lack completion logs; the report keeps that uncertainty visible.

The corrected router changes 237 accelerator streams outside quarantine. No
labels remain in the unreviewed fallback bucket. Legacy dataset-scale device
indices and JSON timestamps are now retained, and sidecars and derived feature
tables are reported separately.

One waveform is unreadable even though its committed LFS hash is correct:
`data/h200_nvl_scope/scale/llm_train_qwen25_1p5b_lora_scope_df63136575f7_20260826_063836_raw.npy`.
Its header declares 24,880,573 int16 samples (49,761,274 bytes including the
header), but the stored file contains only 7,572,480 bytes. Restoring it needs
the original capture; downloading the same LFS object cannot repair it. The
inventory records the error and preserves the original bytes.

Validation passes the 75 CPU tests under `tests/` that need no network access
and all 51 collection-cell routing checks. An end-to-end feature check on job `f554104816ef` read all 16 GPUs on
two hosts, joined NIC telemetry, and produced three windows with full expected
rank coverage. The physical loader read 7,912 samples across 65 runs, including
all 21 corrected aliases. Full-corpus model training, exhaustive LOFO and the
paired ablation still need to be run from rebuilt caches; historical accuracy
numbers have not been replaced.

## Ground truth and input policy

`classifier/workload_labels.py` owns the versioned taxonomy used by software and
physical-sensor classifiers. It preserves raw names and supplies objective,
control/attack membership, and routing evidence separately. The 21 collected
physical-training aliases (`dil_v*`, `ev2b_*`, `evasion2_*`, `s2_*`, `s3_*`) now
route to training. Explicit inference/non-ML collection intent takes precedence
over embedded model names such as GPT, BERT, or DDP. A job that never takes a
gradient step is not training: `advtrain_filler_only` and
`advtrain_satfiller_only` (MN-I at F=0) and every `stalled_*` cell (model and
optimizer resident, optimizer never steps) route to `other` and carry a control
tag. That is the job's objective (`job_label`); window targets are the work
observed in each window (below).

Physical hardening no longer patches only some labels locally or drops correctly
labeled probes to conceal inconsistent routing. Older physical scores need
revalidation. The new inventory lists corrected and unresolved routes explicitly;
`unreviewed_other` means that the old fallback needs source review, not that its
non-ML interpretation has been scientifically established.

`classifier/data_sources.py` resolves pointers directly into `SPAR_LFS_OBJECTS`.
Missing/incomplete payloads raise errors. Accelerator ingestion excludes every
quarantine directory variant and NIC, scope, DCGM, BMC, wall-power and physical
sidecars. Raw bytes are preserved. A reviewed `data/quality_exclusions.json` can
exclude entire failed jobs using log-backed identities without moving raw files.
The previously omitted Week 7 raw directories are now included; cached feature
tables are kept separate. Legacy descriptive EC labels follow their actual
programs (phantom/mining-like inference, head-only training and memory-only idle).

New session ingestion uses workload completion/progress evidence before the
utilization screen. Quiet successful sparse learners stay eligible. Quiet runs
without supporting evidence are quarantined as uncertain; explicit failures and
partial files remain separate. Low utilization alone is not a negative label.
Completed actor/non-root workers do not need their own STEP lines; their lack of
local progress logging is recorded without discarding the collection. Evidence
is matched per file in every launcher mode: a `--parallel` replica by its own
status line (its recorded run_id never appears in the stage log), and a
telemetry-only stream by its job head's exit code, since the collector exits 0
whether or not the remote workload ran.

The inventory's log scan reads plain `.log` launcher output in multi-GPU mode
only; gzipped logs and parallel/telemetry-only stages are not parsed. Checked
against the tracked logs: no failed job outside the 21 exclusions has an
accepted accelerator stream that can be windowed. The only failed parallel
replicas in the corpus (20260820 `inferrep_qwen7b`, vLLM import errors) left
3-row stubs, which `sliding_windows` drops.

## Identity, observation, and features

Streams include host, device, and workload segment. Different workloads sharing
one run/device are no longer stitched together. Per-stream windows keep the
run-level `run_id` and add `stream_id` and `job_id`; the three-way CV folds by
`job_id`, so the ranks of one execution never straddle a train/test split. Versioned identities survive
chunking, and independent S1/S8 executions remain separated by host; each
replica of a replicated S1 cell is its own one-rank job. Legacy jobs (one
run_id per collector) chain streams of one cell and GPU type whose starts follow
within 20 s, so a replicated S16 rep whose nodes started either side of a fixed
bucket edge stays one two-host job. These identities remain provisional.

Multinode windows pool per-rank features using mean, maximum and dispersion.
Pairwise lag summaries use all rank pairs. Neither block chooses an arbitrary
lexicographically first GPU. The maximum block preserves sparse learner signals.
Declared device counts and sample/time coverage govern window eligibility;
degraded windows require explicit diagnostic opt-in. Unknown expected widths
remain unknown, not “100% complete.” Measured host counts determine multinode
membership; an S16 filename alone cannot establish two observed hosts.

Unavailable channels remain NaN, including absent NIC sidecars; measured zero is
distinct. Fold-local median imputation replaces the old global zero fill in the
family evaluators. Metadata and numeric targets are excluded from features.
Hardware and attack/control strata are reported separately; this does not replace
a dedicated hardware-transfer experiment or remove every possible platform cue.

## Per-window labels

Window targets describe the work done inside each window, not the job's name.
Every workload writes one `STEP ts=… kind=…` line per slot of work into its
launcher log, and a line reports the work done since the previous line from the
same process in the same log (one execution). A window is training if any part
of it overlaps a gradient step (`kind=train`); otherwise inference if it overlaps
a forward pass (`kind=infer`); otherwise other (filler, non-ML compute, a stall,
launch and model load, or nothing logged). A long step stays training for its
whole duration, including local SGD's 9–12-minute parameter syncs over a shaped
link, which end in a logged training step.

- `classifier/window_labels.py` reads `logs/**/*.log[.gz]` and joins lines to
  jobs by run_id. An s1/s8 job takes only its own node's log; a multi-node job
  takes every node's lines, since rank 0 alone often logs for all of them. Log
  files are never bridged: a smoke run can reuse its cell's run_id hours later.
- Pooled multinode windows use the whole job: training if any rank trained.
  Per-GPU windows use the GPU's own rank when the job logs rank by rank
  (distributed RL: the learner trains while actors serve rollouts), and the
  job's timeline otherwise.
- Three programs logged non-ML filler slots as `kind=infer`: GPU MN-I and the
  single-GPU interleaving strategies F and I. Their own log output identifies
  them, and those slots count as other. GPU MN-I now writes `kind=filler`, as
  its TPU port already did.
- Without STEP evidence, a program that can only do one kind of work (an
  inference server, a non-ML benchmark, a plain trainer, a stall or filler
  control) labels its windows with that kind (`label_source=program`; its launch
  and load windows are not separated). A mixed program (every disguise,
  distributed RL, single-GPU PPO, dilution sweeps above N=1) is `unresolved`:
  kept in the caches, never fitted or scored.
- Before the October 12 log import, coverage in the tracked logs, by multi-node job: 114 jobs had STEP evidence
  (all on 16-GPU clusters), 936 took a program label and 119 were unresolved
  (the earlier count included 104 TPU disguises whose launcher logs were in GCS).
  No multi-node inference job had committed STEP evidence. The October 12
  recovery below supersedes these log-coverage counts; the saved inventory
  identifies 75 multi-host and 29 single-host jobs within the 104 TPU targets.
- Cache columns: `threeway_label` (window work), `job_label` (job objective),
  `label_source` (steps, program, unresolved), `label_train_s`,
  `label_infer_s`, `label_other_s` (seconds of logged work in the window) and
  `label_train_frac`, the share of the window's wall time spent training. One
  minus that share is the evasion tax the window paid. None of them is a
  feature. The cache fingerprint covers the logs.

Checked against the tracked logs, per execution: every honest trainer's
windows are 100% training between its first and last step; every non-ML,
stalled and F=0 filler cell is 100% other; MN-I interleaves are 57% training.
End to end on job `1871ed7bad7a` (s16 MN-I, F=0.2, 16 GPUs on two hosts), 38 of
40 pooled windows hold some training (0.6–12.9 s of 30) and 2 are pure filler:
"any gradient step" labels most windows of a spread-out disguise training.
Their `label_train_frac` records how thin the training was.

### Launcher-log recovery and vLLM timing — October 12

The saved stream inventory independently identifies 104 mixed-work TPU campaign
jobs whose launcher IDs occur in tracked `data/tpu` paths. Before the GCS export,
none had a temporally matching committed worker log. The completed export has
950 logs and no download failures. Import verified every original and compressed
SHA-256 and retained 642 STEP logs (6,074,373 compressed bytes): 618 TPU worker
logs with 720,773 STEP records and 24 Hugging Face inference logs with 5,900
records. The manifest records the other 308 GPU logs as having no STEP evidence.

**All 104 TPU disguise jobs now have STEP intervals overlapping their recorded
telemetry.** The logging worker indices cover every observed host for each job:
29 jobs have one host, 63 have two and 12 have four. Across 4,293 span-probe
windows (30 seconds, 15-second stride), 3,920 contain training and 373 are other;
none is unresolved. These probes include startup/filler spans and do not certify
the sample/rank eligibility of a rebuilt feature cache. A separate older TPU
trace uses a non-campaign run ID and remains outside this set.

The [per-job results](work_log_coverage_2026-10-12/tpu_jobs.csv),
[window probes](work_log_coverage_2026-10-12/tpu_span_windows.csv) and
[provenance](work_log_coverage_2026-10-12/summary.json) record worker indices,
labels, inventory hash, log hashes and the exact analysis-code hashes. The
recorded `commit` is the checkout parent at analysis time; `code_sha256` captures
the additional parser/audit changes included with these results.

Recover the logs from an authenticated Cloud Shell:

```bash
python3 scripts/gcp/fetch_work_logs.py --out /tmp/gpu-step-logs
python3 scripts/gcp/import_work_logs.py --export /tmp/gpu-step-logs
python3 scripts/audit_work_log_coverage.py \
  --streams /home/robi/workload-audit-2026-10-11/streams.csv \
  --out /tmp/gpu-step-logs/coverage
```

The resumable export downloads TPU worker logs for tracked run IDs and GPU
inference launcher logs. It records source URIs, hashes, lengths, run IDs, STEP
kinds and epoch bounds. Import verifies the original bytes and compressed bytes
before adding logs under `logs/gcs/<bucket>/<original-object-path>.gz`. The
manifest lists logs without STEP evidence without converting them into ground
truth. Session/run/worker paths prevent different captures from overwriting one
another. The coverage report probes 30-second windows at 15-second stride over
recorded spans; actual feature windows additionally require sample/rank coverage.
Rebuild caches after importing: log content is part of their fingerprint.

Recovered GPU launcher logs can start after the `Run ID:` header. The parser now
uses the canonical `_node<N>_<hex-run-id>.log[.gz]` filename when that header is
absent and rejects a conflicting header. This preserves the original log bytes
while making recovered HF STEP evidence available to both feature caches.
The 24 HF logs cover 12 run IDs; six of those identify jobs in the accepted
saved inventory, and all six have temporally overlapping STEP intervals. The
remaining logs are retained as evidence without asserting a telemetry match.
All 71 relevant CPU tests pass, including headerless-log joins and conflicting
identity rejection; scanning the existing launcher logs found no conflicts.

Both vLLM programs now emit pending and completed STEP events. Offline
`llm.generate()` reports batch start/end epochs and output-token counts; async
serving reports each request's start/end, prompt/output tokens and completion
status. Explicit `start_ts` avoids filling startup and idle gaps or serializing
overlapping requests. These are generation-call/request wall-time envelopes,
including scheduling, queue and transport time, not per-kernel activity. The
controller leaves device/rank attribution unspecified.

Failed or unfinished requests contribute unresolved intervals. A window with a
confirmed training step remains training, while unresolved activity prevents an
otherwise inference/idle window from becoming a definite negative; its training
fraction remains unknown. Legacy STEP logs retain their adjacent-slot semantics.
These events apply to new collections: historical vLLM logs cannot supply STEP
lines that were never emitted, and no events are synthesized from aggregate
throughput prints.

## Evaluation

`classifier/rebuttal_analysis/paired_family_lofo.py` holds out every family once.
Targets are the per-window work labels above. Unresolved windows are excluded
and counted, metrics are reported per label source, and `--steps-only`
restricts fitting and scoring to windows labeled from STEP evidence. To measure
the evasion tax, every prediction carries its window's training share, each
fold reports the mean share of its training windows, `by_train_fraction`
reports window-level detection by training share (0%, 0–5%, 5–20%, 20–50%,
50–100%), and `by_job_train_share` reports job-level detection by each job's
overall training share.
Each eligible window receives exactly one full-model outer prediction. Missing
classes or ambiguous identities fail visibly. Estimator settings and threshold
are fixed before outer testing; no test fold selects either.

Honest/adversarial pipeline siblings share a family, as do async-serving schedule
variants, memory-footprint variants, accumulation variants and interleaving/filler
controls. `--family-level backbone` adds a stricter shared-transformer holdout to
the mechanism-level test. These are complementary generalization claims. A fold
that removes the only remaining training family is not estimable and is rejected.

For the campaign ablation, both models test the same old-job window IDs. Only
their training-job pool differs. New-job generalization is reported separately.
Zero-effect exclusions fail. Outputs contain predictions, explicit split IDs,
input/cache/code provenance, binary training recall/FPR, three-way confusion,
an explicit any-window job alert rule, and a paired family bootstrap interval.
The any-window rule's job false-positive rate is reported rather than inferred
from window accuracy. It is not a calibrated production operating point.

`mn_binary_lofo.py` uses this same engine for honest training-vs-rest all/shape
comparisons. `multinode_family_holdout.py` remains available under its accurate
random-family-holdout description and reports families never tested.

## Reproduce

Use the external objects directly; do not hydrate HOME:

```bash
export SPAR_LFS_OBJECTS=/tmp/gpu-workload-audit/lfs/objects
python3 scripts/audit_data_inventory.py \
  --lfs "$SPAR_LFS_OBJECTS" \
  --out /tmp/gpu-workload-audit/inventory-2026-10-11
```

The inventory refuses to finalize until every tracked LFS object is present,
then verifies SHA256 hashes. It reads identity/timestamp columns and waveform
headers, unions duplicate timestamps, and separates accelerator-hours from
sidecars and raw waveform bytes. It exports workload descriptions, hardware,
observed/expected topology, durations, samples, log evidence, label corrections,
quarantine and schema errors. Timestamp span and gap-capped coverage are both
reported; neither is useful optimizer time.

On an analysis machine with sufficient RAM and disk space:

```bash
export SPAR_WORK=/tmp/spar-analysis
PYTHONPATH=. python3 scripts/build_mn_window_cache.py "$SPAR_WORK/windows.parquet" 30 15 4
PYTHONPATH=. python3 -m classifier.rebuttal_analysis.paired_family_lofo \
  --windows "$SPAR_WORK/windows.parquet" \
  --exclude-campaigns 20261001,20261003 \
  --out "$SPAR_WORK/paired_lofo.json"
```

Both cache builders bind reuse to input object hashes, current code (including
uncommitted changes), taxonomy versions and configuration, and verify the cached
artifact hash. Unversioned caches are rejected. The cloud runner now follows
`main` and writes the paired result under a new dated destination.

`/tmp` is temporary storage. Persist the completed inventory and objects to your
durable analysis volume before the Cloud Shell instance is discarded.
