# NIC and TPU recovery — 2026-10-12

Recovered the audited NIC checkpoints and both STEP-backed TPU candidates from
the local, checksum-verified GCS download cache. Sources are exclusively from
`gs://focus-surfer-494820-g0-telemetry`. Sigil is outside this project and recovery.

## Results

| Recovered data | Main corpus | Preserved separately |
| --- | --- | --- |
| NIC checkpoints | 215 host streams | 252 host streams without an eligible accelerator pairing |
| Additional finalized NIC files | 10 host streams | 17 host streams without an eligible accelerator pairing |
| TPU `ad5e56af3e3b`, `tpuTMS4_advtrain_typeconf_a32_rep2` | Four hosts/chips; 2,232 samples; 0.61896 accelerator-hours | Original untrimmed telemetry and completion markers |
| TPU `cdc1669c2763`, `tpuTMS4_advtrain_typeconf_a8_rep2` | None: job failed | Four hosts/chips; 2,271 training-interval samples; 0.62980 accelerator-hours, plus original untrimmed telemetry and markers |

The **225 accepted NIC streams** contain **133,611 samples**, covering
**37.05643 host-hours** across **176 jobs** in the prior accelerator inventory.
Each accepted sidecar spans approximately 9–12 minutes. These are additional
communication measurements for existing accelerator runs, not additional
accelerator-hours or new workload classes. Of the recovered sidecars, 66 have
fewer than 600 samples and would have been lost under the older ingestion cutoff.

All **494 candidate NIC streams**, comprising **214,210 samples** across
498 distinct source payloads, are preserved as original bytes. The 269 streams
without an accepted pairing remain outside the training loader. They include
unpaired captures and captures associated with new/unreviewed, smoke, quarantined,
or explicitly excluded accelerator data; presence of a NIC file alone does not
establish a successful accelerator workload.

Both TPU jobs ran the approximately **1.34-billion-parameter MN-T adversarial
training workload on four Google TPU v6e chips across four hosts**, using SGD,
tensor-core filler (`tc_fill=0.9`), and device-to-host trickle traffic
(`d2h_trickle=256`). The `a32` and `a8` variants synchronize parameters every
32 and 8 steps respectively, shaping training telemetry to resemble inference.
The implementation uses the same batch on every host; these runs support
telemetry-classification analysis, not learning-efficiency or distinct-data
throughput claims. See [the workload implementation](../../../workloads/tpu_adversarial_typeconf.py).

- **`a32` completed:** all four worker markers report `rc=0`; every worker logged
  4,444 training steps. Rank 0 reports 585.0 seconds of workload runtime. The
  recovered telemetry spans about **9.3 minutes per host**, after removing startup
  and post-training samples using each worker's first and last STEP timestamps.
- **`a8` failed late:** worker 0 reports `rc=-6`, workers 1000 and 2000 report
  `rc=124`, and worker 3000 reports `rc=0`. The recovered prefix spans about
  **9.5 minutes per host**; three hosts logged 2,247 training steps and one logged
  2,248. The entire job remains quarantined despite genuine training progress.
  Its long idle/hang tails are retained only in the original raw files.

The previous GCS audit classified both TPU jobs as recovery candidates based on
temporally matched STEP evidence. This recovery adds the completion-marker check
and establishes that only `a32` qualifies for the main corpus.

## Locations and provenance

- [Recovery manifest](../../../data/gcs_recovery_20261012/manifest.json): source
  bucket/object/generation, source SHA-256, original-byte paths, derived-file
  hashes, row counts, time bounds, GPU pairings, completion status, and policies.
- `data/gcs_recovery_20261012/nic/`: the 225 accepted NIC sidecars.
- `data/gcs_recovery_20261012/tpu/ad5e56af3e3b/`: completed TPU training intervals.
- `data/gcs_recovery_20261012/_quarantine_failed/tpu/cdc1669c2763/`: recovered
  training intervals from the failed TPU job.
- `data/gcs_recovery_20261012/_quarantine_raw/`: all 522 distinct original source
  payloads, including NIC/TPU telemetry, eight worker logs and eight status markers.
- `logs/gcs/focus-surfer-494820-g0-telemetry/tpu/gap_20260903_221548_ms4/runs/`:
  eight compressed, checksum-verified worker logs imported into the standard
  STEP-log index. [Log manifest](../../../logs/gcs/manifest.json).

The recovery occupies about **38.4 MB** including provenance and compressed logs
in the project checkout. It does not depend on `/tmp` for subsequent classification
of the recovered files. Reproducing recovery from scratch uses the original audit
inventory and download cache. The existing `.parquet` Git LFS rules apply to
these files. The recovery script itself performs no Git commit or cloud upload.

## Admission and transformation rules

NIC files are paired by exact run ID, hostname, canonical workload, and overlapping
time interval against the prior nonquarantined accelerator inventory. The current
21-job quality-exclusion list is applied. Samples are clipped to the paired
accelerator recording spans; this pairing is not itself proof of activity at every
timestamp. Existing STEP-based labeling still determines work labels.

The recovery checks conflicting duplicate timestamps, preserves valid zero traffic,
masks invalid/negative counters, and requires at least 30 seconds of contiguous
usable traffic measurements. It also checks the production NIC join key
`(run_id, hostname, round(timestamp))`; there were no duplicate keys or invalid
metric values in the admitted intervals. Existing NIC recording spans are excluded
with a conservative guard around rounded timestamps. The inherited `*_mbps`
column names contain **MiB/s**, as in the collector; units were not rescaled.

TPU intervals include only samples between the first and last training STEP on
their own worker. Every admitted or recovered-prefix sample has live TPU metrics.
The slice-error field is unavailable (`-1`) in these recordings, so it cannot
establish absence of hardware faults. The declared four-chip topology from the
rank-0 launcher log is stored as `expected_ranks=4` in each derived TPU file.
Collector workers 0/1000/2000/3000 map to JAX processes 0/1/2/3 for these four
one-host slices. All-worker completion is required for main-corpus admission.

Raw files and failed-job derivatives use the repository's quarantine path
convention and are excluded from the normal accelerator and NIC loaders.

## Validation and reproduction

The [focused validation result](validation.json) records:

- SHA-256/size checks passed for all **755 data/evidence output files**.
- All **1,585 referenced existing accelerator files** still match the prior audit
  content hashes, including LFS pointers.
- The actual NIC merge was exercised on **nine cached GPU streams / 5,409 samples**;
  **4,889 samples** acquired recovered NIC measurements, with no row multiplication.
- The production loader admitted exactly **one TPU job, four streams and 2,232
  samples**. Quarantined raw files and the failed job were excluded.
- Production windowing produced **35 complete 30-second windows**, each with
  four hosts/four ranks, full rank coverage, and STEP-backed `ml_training` labels.
- Six recovery regression tests and 13 existing work-log collection tests passed.
- A repeat dry run reproduced the saved output hashes and manifest without changes.

The full historical LFS corpus was not materialized, and the classifier was not
retrained. New data and logs are included in the existing cache-provenance input
selection, so the next full cache build must incorporate them; this recovery does
not establish an accuracy improvement by itself.

From the repository root:

```sh
# Check the recovery plan and hashes; --apply installs missing outputs.
python3 scripts/recover_audited_telemetry.py

# Focused production-loader/merge/window validation; needs the audit cache
# only for the original GPU streams used to test NIC attachment.
python3 docs/workload_review_2026-10-12/recovery/validate_recovery.py

python3 -m unittest discover -s tests -p test_telemetry_recovery.py
python3 -m unittest discover -s tests -p test_work_log_collection.py
```

The [recovery script](../../../scripts/recover_audited_telemetry.py) validates
the complete plan before importing files, refuses conflicting existing output,
and retains the original GCS bytes alongside every derived telemetry file.
