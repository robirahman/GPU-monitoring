# Spot-preempted A100 capture recovery

Preemption ends a run; it does not erase the work already observed. The original
captures were preserved in GCS. The earlier blanket exclusion from the corpus
was too broad and is superseded by interval admission below. No source object
was deleted or overwritten.

| Run | Observed training interval | GPU samples | GPU-hours | 30-second windows |
| --- | ---: | ---: | ---: | ---: |
| `4c6fe5ccea7a` | 88.45 s | 1,408 | 0.38673 | 4 |
| `93a4645f6a42` | 1,383.08 s | 22,122 | 6.14171 | 90 |
| Total | Two interrupted captures | 23,530 | 6.52844 | 94 |

Each capture used two hosts with eight NVIDIA A100 SXM4 40GB GPUs per host.
The workload is synchronous TP8/DP2 training with random tokens. Saved rank-0
STEP events follow the gradient collectives and optimizer step, providing work
evidence for this synchronous workload. This does not establish useful learning
or justify applying the same rank-0 rule to asynchronous workloads.

The windows use a 30-second length and 15-second stride; overlapping windows do
not increase the reported GPU-hours. Hours sum each admitted device's timestamp
span. The common interval is bounded by the first and last saved STEP, all 16
GPU recordings and the cloud preemption event. This retains less than the full
roughly 26-minute second launch because startup and unsaved/unobserved tails
cannot be labeled from duration alone. All original samples remain in GCS.

## Corpus inputs and provenance

- [Manifest](../../data/preempted_recovery_20261013/manifest.json): 74 source
  objects with exact GCS generation, MD5, CRC32C, SHA256 and size; transformations,
  original/admitted time bounds, hardware identity, seed and derived file hashes.
- `data/gcp/followup_a100_20261012T233445Z/` and
  `data/gcp/followup_a100_20261013T011424Z/`: 32 admitted GPU parquets and four
  NIC parquets under the standard corpus input paths, stored through Git LFS.
- Four original launcher logs in `logs/followup_a100_*__spar-a100-*__*.log`:
  standard STEP-index inputs tied to the original run IDs.
- [Validation](../../data/preempted_recovery_20261013/validation.json): production
  loader retains every admitted sample; NIC join adds no rows and matches all
  samples. All 94 windows have two hosts, 16/16 ranks and STEP-derived training
  labels with 100% observed training time.

Final files and rolling checkpoints are merged by physical identity and timestamp,
with exact duplicates removed. Conflicting samples, invalid timestamps, missing
ranks, incompatible UUIDs and gaps over 2.5 seconds stop this recovery. Missing
sensor readings are not filled with zero. The minimum admitted interval is 30
seconds; shorter captures remain useful raw evidence but do not furnish these
30-second classification windows.

Both jobs remain `preempted`, with `run_completed=false` and zero completed
30-minute repetition credit. Their original run IDs are retained. Both attempts
use seed 8123401 and share a `repetition_group`; they are not independent seeded
repetitions. Keep that group together in any split or resampling that could
otherwise separate retries. Completion, censoring and grouping fields are
metadata, not predictive features. The existing family holdouts keep both TP
captures in the same family.

The current long-running analysis uses its existing frozen input snapshot; these
new inputs belong in the next content-bound cache build. Corpus admission does
not claim an accuracy improvement. Earlier historical whole-job exclusions need
their own interval review; this recovery does not automatically relabel them.

## Reproduce the transformation

Download the generation-pinned sources listed in the manifest to external storage,
verify their size/MD5/SHA256, and add a `path` field to each source record pointing
to that download. Save the combined source records as `sources.json`. Then run:

```sh
python3 scripts/recover_preempted_followup.py \
  --sources /external/recovery/sources.json \
  --evidence docs/followup_execution_2026-10-12/failed_a100_collection.json \
             docs/followup_execution_2026-10-12/failed_a100_retry.json \
  --out /external/recovery/corpus
```

The command works offline and refuses to overwrite different existing output
bytes. It preserves raw source identities, clips verified observations and emits
ordinary classifier inputs. The focused recovery, collection, audit and multi-node
regression suites passed 73 tests, including missing ranks, truncated overlap,
conflicting checkpoints and mismatched STEP evidence.
