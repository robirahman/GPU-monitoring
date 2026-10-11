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

Validation passed 43 CPU regression tests and all 51 collection-cell routing
checks. An end-to-end feature check on job `f554104816ef` read all 16 GPUs on
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
tag. Per-window STEP truth remains a separate field.

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
chunking, and independent S1/S8 executions remain separated by host. Legacy job
reconstruction still uses a 20-second bucket; these identities are provisional.

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

## Evaluation

`classifier/rebuttal_analysis/paired_family_lofo.py` holds out every family once.
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
