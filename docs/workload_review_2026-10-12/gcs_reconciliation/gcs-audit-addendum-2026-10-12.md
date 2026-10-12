# GCS reconciliation of the workload audit — 12 October 2026

The authenticated GCS check found additional raw captures missing from the earlier stream inventory. They require a separate disposition: cloud storage includes successful runs, failed attempts, smoke tests, untrimmed recording tails, derived results, and Git LFS pointers.

The original audit remains available as [a Markdown file in Google Drive](https://drive.google.com/file/d/1kENAsxZlisdSuRQNIfTp27ayc2fUD1ju/view?usp=drivesdk). This addendum supplements it; it does not silently change the original dataset totals or import captures into the classifier.

## Additional captures

Hours below are the sum of each accelerator stream's timestamp span, **not elapsed experiment time or active training time**. A sixteen-GPU recording lasting ten minutes contributes about 2.67 accelerator-hours. Streams and collector run IDs are not necessarily independent experiments.

| Additional group | Hardware and scope | Streams | Samples | Raw accelerator-hours | Disposition |
|---|---|---:|---:|---:|---|
| Earlier H100 adversarial campaign | H100 80 GB HBM3; S1, S8, and two-host S16 | 284 | 164,718 | 45.682 | Includes completed attacks, failed launches, and one unconfirmed communication capture |
| Additional TPU attempts | v5e and v6e; includes sixteen-chip and four-host configurations | 109 | 26,317 | 7.285 | Most were explicitly excluded during ingestion; two four-host runs are recovery candidates |
| Smoke tests | Mainly H100; also A100 and TPU | 847 | 97,386 | 26.821 | Diagnostic captures; excluded from the proposed main dataset |
| **Total absent from the earlier accelerator-stream inventory** | | **1,240** | **288,421** | **79.788** | **Do not add this entire total to the paper's dataset** |

The original main-corpus figure remains **1,538.298 accelerator-hours**, with its original qualifications. The new 79.788 hours are raw evidence requiring adjudication. They are not 79.788 hours of additional validated training.

### H100: useful older adversarial experiments

These recordings date to August 21–22, 2026. The `runs/20260820/` bucket prefix is a collection destination, not the timestamp of every recording beneath it. Most complete recordings last approximately 10–10.5 minutes per GPU, including collector overhead.

* `s{1,8,16}_advtrain_interleave_f{02,05,10,20,50}` alternates genuine FSDP training with non-ML filler, with requested training fractions of 2%, 5%, 10%, 20%, and 50%. These older configurations are distinct from the saturated-filler variants already counted in the main corpus.
* `s{1,8,16}_advtrain_filler_only` supplies the corresponding zero-training control. Its correct objective is `other`.
* `s{1,8,16}_advtrain_oracle` and the `oracle_tau{50,70,90}` variants query a classifier to gate training. Their utility varies sharply with threshold and detector warmup.
* One extra `s8_nonml_comm` recording has only one observed accelerator stream and no matched workload STEP/completion evidence in the downloaded logs. Its 0.171 hours remain unconfirmed.

There are 31 collector run IDs, corresponding to **39 provisional jobs** after treating each host's S1/S8 execution separately. S8 executes an independent eight-GPU job on each host; two such recordings must not be presented as a sixteen-GPU distributed job.

**The older filler has a documented limitation.** Commit `47506f3c` (August 24) explains that these original non-ML filler kernels were too small and kernel-launch bound, and retracts the earlier interpretation that detecting them demonstrated robust discrimination of training. The `satint`/`satfiller_only` labels were deliberately introduced for the corrected implementation. Keep the recovered older interleaving captures as historical controls or ablations, separately from competent saturated-filler attacks; their recovery does not establish stronger adversarial coverage by itself.

Execution logs identify **35 completed jobs**, covering 259 streams and **44.678 raw accelerator-hours**. Three S16 oracle threshold attempts failed with `RendezvousTimeoutError`, had no STEP evidence, and contributed 24 streams / **0.833 hours**. Completion alone does not establish correct window labels or a successful attack.

For example, the S16 oracle run `7f43162f62e3` reports 3,789 total steps but only **three training steps and 393,216 useful tokens over 600.5 seconds**. All three training steps occurred at the beginning; subsequent progress reports show no additional training. This is a useful example of an attack that suppresses its own training, not evidence of sustained successful evasion. S8/S1 threshold sweeps also record warmup allowances separately in some logs; these should be excluded from claims of steady-state evasion.

The older MN-I logs label non-training filler intervals as `kind=infer`, whereas the current workload code emits `kind=filler` for those intervals. Recover their labels using the executed strategy and source version, rather than interpreting the old STEP string literally as ML inference.

### TPU: distinguish deliberate exclusions from missed ingestion

The 109 additional TPU streams divide into:

* **97 streams / 5.151 raw hours** explicitly excluded by existing ingestion manifests for an idle span, described there as “compile-then-hang.” They include the sixteen-chip v5e interleaving attempts and one single-chip v6e attempt.
* **Four streams / 0.174 hours** excluded because the worker logs contain no STEP lines.
* **Eight streams / 1.959 raw hours** from two four-host, four-chip v6e runs absent from the earlier inventory and ingestion manifests: `tpuTMS4_advtrain_typeconf_a8`, run `cdc1669c2763`, and `tpuTMS4_advtrain_typeconf_a32`, run `ad5e56af3e3b`.

For the latter two, each host has STEP timestamps overlapping its telemetry. Restricting samples to the corresponding first-to-last STEP interval gives **4,503 samples and 1.249 accelerator-hours** across eight streams. This is a defensible recovery candidate, subject to phase labeling and the project's normal acceptance checks. The longer raw spans include uneven recording tails.

The ingestion filter uses observed utilization to reject low-utilization runs. Some rejected runs nevertheless emit training STEP records. For an adversarial-classification paper, reassess these exclusions using independently verified completed device work and progress: filtering on the detector's own input signal can remove precisely the low-duty-cycle behavior the study needs to measure. The manifest's reason is evidence of a past decision, not an independent proof that no useful work occurred.

### Other cloud evidence

* **494 additional NIC streams / 214,210 samples** were found: 27 streams / 10,281 samples in finalized Parquet files, plus **467 streams / 203,929 samples in readable `.parquet.partial` checkpoints**. The bucket contains 1,085 checkpoint objects; all distinct additional checkpoint payloads were examined, along with the analysis bundle's checkpoint members, with no read errors. **223 of the additional checkpoint streams overlap accelerator captures already in the main corpus.** Others accompany newly found captures, quarantined runs, or smoke tests; 47 have no matched accelerator span in this comparison. These are network sidecars, not additional accelerator-hours. Join them to the corresponding host/run/time interval and check counter quality before claiming complete network-channel coverage. Details are in [additional_nic_checkpoints.csv](additional_nic_checkpoints.csv).
* **1,357 already represented TPU streams** have longer raw versions in GCS. Their extra spans total **8.458 hours**, with 29,946 additional samples. The median extension is approximately five seconds; some extend for minutes. These are consistent with the ingestion code's removal of recording/hang tails and are not new workload configurations.
* The September 30 analysis bundle contains 12,227 files, including **11,789 Parquet files whose SHA-256 matches the earlier inventory**. Its other **21 Parquet files are derived classifier predictions**, not additional raw workload recordings.
* The WAVE backup is byte-for-byte identical to the existing local archive. Its Nsight Compute CSVs do not add new one-hertz telemetry. The six April Nsight Compute `.ncu-rep` payloads in GCS are the profiles already documented in `data/ncu_profiles_README.md`.
* The Sigil project and its home-directory archive are excluded from this paper audit at the user's instruction.

## Storage and completeness

The scope is all 26,592 current live objects (11,284,450,489 bytes) in `gs://focus-surfer-494820-g0-telemetry`. The other two visible buckets belong to the separate Sigil project and are excluded at the user's instruction. Historical generations and soft-deleted objects are outside this check. The final object totals and checksum results are recorded in [summary.json](summary.json).

All 17,176 directly stored telemetry-bucket Parquet-named objects are reconciled in [cloud_parquet_reconciliation.csv](cloud_parquet_reconciliation.csv). The check uses object metadata, generation-pinned downloads with checksums, SHA-256 comparisons against the earlier inventory, and run/device/host/timestamp comparisons for different payloads. Archive members are checked separately. Listing a bucket is not equivalent to reading every byte of unrelated game data or deployment packages.

**3,618 Parquet-named objects are actually Git LFS pointer text.** Of these, 3,612 refer to objects represented in the earlier inventory. The remaining six are two copies each of three old pointers: `tpu_idle.parquet`, `tpu_jax_inference.parquet`, and `tpu_jax_training.parquet`. Their target payloads have not been located in the examined telemetry payloads or analysis bundle. They are unquantified references, not additional measured workloads.

The current checkout resides on a 4.8 GB home filesystem. No GCS filesystem mount is visible in this environment; these buckets are object storage accessed through GCS. The previous audit's external LFS cache is absent in this session. Large downloads for this reconciliation were placed under `/tmp/gcs-workload-check-2026-10-12`, where space is available. Those temporary downloads should not be treated as a permanent backup. A GCS copy containing LFS pointers does not by itself preserve the referenced data.

## What changes for the follow-up paper

1. **Recover and adjudicate existing data before repeating collection.** Prioritize the missing NIC checkpoints paired with existing accelerator recordings, and the two four-host TPU type-confusion runs. These could expand network-channel coverage and training repetitions. Preserve the older H100 interleaving/oracle experiments as historical controls and utility/failure analyses, accounting for the documented weak filler. Use the corrected saturated implementation for new claims about attack robustness.
2. **Measure adversarial utility explicitly.** Report completed updates, useful tokens, training progress, wall time, and warmup behavior alongside classification/attack success. A detector fooled only after the attacker stops training should not count as a successful useful attack.
3. **Use phase-correct ground truth and independent acceptance criteria.** Separate training, inference, filler, initialization, teardown, and stalls. Preserve the source commit and executed command because historical STEP semantics changed. Do not discard low-utilization attacks solely because their counters resemble idle behavior.
4. **Add a durable object/run manifest.** Store bucket/key/generation, payload hash, source commit, command, hardware topology, worker IDs, status, STEP interval, and exclusion reason. Explicitly distinguish raw payloads, LFS pointers, derived results, duplicates, and smoke tests. Keep all ranks/hosts and all copies of an experiment in the same evaluation split.

The earlier recommendations on repetition, held-out workload families and hardware, multi-node inference/communication controls, and utility-constrained adaptive attacks still apply. No classifier inputs, model parameters, or accepted-corpus totals were changed by this reconciliation.

The review tables are [new_workloads.csv](new_workloads.csv), [new_capture_dispositions.csv](new_capture_dispositions.csv), [h100_job_evidence.csv](h100_job_evidence.csv), and [execution_log_evidence.csv](execution_log_evidence.csv). They provide the measured details behind this addendum.
