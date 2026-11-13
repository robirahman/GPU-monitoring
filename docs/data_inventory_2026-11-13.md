# Data inventory, 2026-11-13

Every dataset collected for this project, and which part of it the accepted
NeurIPS paper (arXiv:2606.19262) already reports. The question this answers is
reuse: data outside the published set can carry a new paper on its own, data
inside it has to cite the existing one.

Measured against **`main` @ `c9feeee4`**, not a working tree. The checkout on
`spar-analysis-0` sits on `neurips-camera-ready`, which is missing the whole
2,737-file `data/h200_followup` tree; anything inventoried from that branch
undercounts the four-sensor campaign to zero.

Stream-level figures come from `docs/camera_ready_manifest_20261022/`
(regenerated 2026-11-05, so the 2026-11-05 H100 ppodist run is in). Re-run
`PYTHONPATH=. python3 scripts/camera_ready_manifest.py` rather than trusting
these numbers after the next collection. File sizes resolve LFS pointers to
their payload bytes, so they are real volume, not 130-byte pointers.

## Read these three before quoting anything below

* **Files are not streams.** 12,688 live accelerator parquets hold 10,359
  distinct streams. The same stream is written into more than one directory and
  `load_all_data` collapses them on `(ts, run_id, gpu_uuid)`; the manifest keeps
  the longest copy. File-level counts overstate the corpus by about 18%.
* **Accelerator-hours are not wall-clock.** A 16-GPU job running one hour
  contributes 16 accelerator-hours. The corpus is 1,595.9 accelerator-hours and
  379.1 wall-clock hours.
* **The published/unpublished boundary is campaign-exact, not file-exact.** See
  [Where the boundary is soft](#where-the-boundary-is-soft).

---

## 1. Everything collected

**24,539 data files, 22.9 GB, 2026-03-09 to 2026-11-05.**

### By sensor modality

| Modality | Files | Size | What it is |
| :-- | --: | --: | :-- |
| GPU NVML telemetry | 11,425 | 0.40 GB | 9 signals at 1 Hz — the paper's channel |
| GPU NVML, quarantined | 2,714 | 0.08 GB | failed or partial collection, preserved |
| Oscilloscope raw `.npy` | 419 | 16.29 GB | PicoScope 5464E current clamp, 19.53 MS/s |
| Oscilloscope parquet | 458 | 3.99 GB | decimated to 976.6 kS/s, 80 dB stopband |
| Run and launcher logs | 1,699 | 1.42 GB | `STEP` lines = per-window ground truth |
| Compressed logs (`.gz`) | 1,357 | 0.17 GB | archived launcher output |
| Event logs `events.jsonl.gz` | 210 | 0.29 GB | fused four-sensor records (H200 follow-up) |
| TPU telemetry | 1,361 | 0.09 GB | v5e and v6e |
| NIC counters | 847 | 0.05 GB | inter-node traffic on multi-node runs |
| DCGM counters | 341 | 0.01 GB | tensor-core activity, NVLink throughput |
| BMC / IPMI | 271 | <0.01 GB | chassis-level power |
| Physical sensors | 65 | <0.01 GB | acoustic + power-rail current, RTX 3090 |
| Wall power (Tapo P115) | 49 | <0.01 GB | smart plug, RTX 5080 desktop |
| Resume markers, derived CSV/JSON | 3,195 | 0.02 GB | bookkeeping, not measurements |

Deduplicated, the accelerator telemetry is **10,359 streams, 3,115 physical
jobs, 540 workload cells, 1,595.9 accelerator-hours, 5,796,976 samples**, over
16 hardware models.

### By campaign

| Campaign | Streams | Jobs | accel-h | Hardware | Dates |
| :-- | --: | --: | --: | :-- | :-- |
| `data/gcp/20260923` | 1,390 | 87 | 247.1 | A100 40/80GB, 2x8 | Sep 23–24 |
| `data/tpu` | 1,357 | 319 | 244.7 | TPU v5e, v6e | Aug 31–Sep 4 |
| `data/gcp/20260820` | 1,737 | 1,048 | 241.1 | mixed (archive drop) | Mar 9–Aug 20 |
| `data/gcp/20260904` | 1,245 | 95 | 203.9 | A100 40GB | Sep 4–21 |
| `data/` root | 1,171 | 754 | 109.6 | A100 80GB, H200, B200, 5080 | Mar 9–Nov 5 |
| `data/gcp/20260929,0926,1001,1003,0921` | 1,296 | 83 | 238.0 | A100/H100, 2x8 | Sep 21–Oct 3 |
| `data/gcp/20260908,0909` | 835 | 60 | 155.2 | H100 80GB | Sep 7–9 |
| `data/gcp/20260824sat,0825fp,0822ctl` | 543 | 80 | 93.1 | H100 80GB | Aug 22–25 |
| `data/h200_nvl_scope/**` | 456 | 456 | 35.1 | H200 NVL + oscilloscope | Aug 24–Sep 27 |
| `data/h200_followup/**` | 210 runs | — | 81.9 | H200, four sensors at once | Oct 21–25 |
| `data/gcp/20261023b,20261023c,20261026a` | 189 | 69 | 16.9 | A100 80GB | Oct 23–26 |
| `data/gcp/followup_a100_*` | 32 | 4 | 6.6 | A100 40GB | Oct 12–13 |
| `week4`, `week5`, `week7` | 82 | 58 | 0.4 | H100, B200 | Mar 16–Apr 7 |
| `data/gcp/20261105rl8` | 16 | 2 | 2.8 | H100 80GB x8 | Nov 5 |

`data/h200_followup` is counted in runs, not streams: it stores fused
`events.jsonl.gz` rather than per-device parquets, so it never enters the
stream manifest.

---

## 2. What is not in the camera-ready paper

The paper's footprint is tiers `A_camera_ready` and `B_rebuttal` restricted to
the nine NVIDIA classes of Table 1 (`PAPER_NINE_GPUS` in
`classifier/data_sources.py`; H200 and H200 NVL merge into one class).
Everything else is unpublished.

| | Streams | Jobs | Cells | accel-h | Samples |
| :-- | --: | --: | --: | --: | --: |
| In the paper | 1,807 | 1,306 | 215 | 207.6 | 756,911 |
| **Not in the paper** | **8,552** | **1,809** | **421** | **1,388.3** | **5,040,065** |

**87% of the accelerator-hours ever collected are unpublished.**

| Unpublished block | Streams | accel-h | Why it is out |
| :-- | --: | --: | :-- |
| Two-node 16-GPU (`s16`) | 5,321 | 939.4 | Limitation 1: "exclusively single-node… up to only 8 GPUs" |
| TPU v5e / v6e | 1,361 | 245.3 | Limitation 4: "we only tested on NVIDIA GPUs" |
| One-node 8-GPU follow-ups (`s8`) | 623 | 105.1 | in scope for the threat model, collected after submission |
| H200 NVL + oscilloscope | 456 | 35.1 | Limitation 5: "sampled at ~1 Hz" |
| AMD RX 9070 XT and MI300X VF | 406 | 30.4 | collected in time, deliberately kept out of Table 1 |
| Single-GPU cluster baselines (`s1`) | 91 | 12.0 | companion controls to `s8`/`s16` |
| RTX 3090 physical-sensor day | 66 | 2.2 | acoustic/current channel, GPU not in Table 1 |
| October A100 re-collections | 165 | 14.6 | post-acceptance |

Wholly additional to that table, because it is event-log rather than parquet:

* **`data/h200_followup` — 210 runs, 81.9 GPU-hours, 2026-10-21 to 10-25.**
  NVML, DCGM, BMC and oscilloscope recorded simultaneously on one H200, with a
  pre-registered calibration/test partition (long normal operation, attack,
  attack-matched-ordinary, cover-only control, burst-in-operation, burst
  control, training timing). 209 of 210 runs completed. The cleanest standalone
  dataset in the repo.

Also unpublished, and useful as negative or QA material: 2,714 quarantined
accelerator parquets, the 124 jobs excluded by `data/quality_exclusions.json`
(`nonzero_workload_exit` and similar), 847 NIC counter files, 341 DCGM, 271 BMC.

---

## 3. Where the boundary is soft

**The split is certain per campaign, not per file.** The camera-ready prose
claims 1,566 runs, 194 workloads, 164 GPU-hours, 601,112 readings. Tier A alone
on paper hardware measures 1,629 runs / 167 cells / 145.9 h / 534,799 readings;
tier A plus tier B measures 1,744 / 215 / 207.6 h / 756,911. The paper's figure
sits between the two because it was computed before the current deduplication
and quality-exclusion policy. Treat "published" as accurate to about ±15% at
the file level.

**270 streams inside tier A were collected after the paper** — 207 jobs, 45
cells, 28.6 accelerator-hours, promoted by `data/tier_a_recollections.json` as
substitutes for runs whose published configuration starved its dataloader. The
telemetry is new but it stands in for a published workload definition: reuse
cites the paper for the workload, not for the data.

**The oscilloscope raw data is mostly off-repo.** The 210 `h200_followup` run
manifests report 172.6 GB of `bytes_on_disk` between them; the repo holds about
1.4 MB of `events.jsonl.gz` per run. The waveforms went to Drive via
`h200_campaign/upload_to_drive.sh`. Budget for retrieving ~170 GB if a new
paper needs them. The earlier `data/h200_nvl_scope` campaign is the opposite —
its 16.3 GB of raw `.npy` is committed in full.

---

## 4. Strongest candidates for a new paper

1. **Multi-node.** 939 accelerator-hours over 5,321 streams, with NIC counters
   alongside, and the existing paper names single-node scope as its first
   limitation. Largest and most directly motivated block.
2. **Four-sensor fusion (`h200_followup`).** Already partitioned for a
   pre-registered evaluation, which is hard to retrofit. Needs the Drive pull.
3. **Cross-architecture (TPU).** 245 accelerator-hours, but
   `docs/telemetry_signal_matrix.md` records that TPUs expose only 3 of the
   paper's 9 signals — power, temperature and every clock read back `-1`. That
   makes it a methods contribution about signal availability, not a clean
   transfer result.
4. **Sub-second sensing.** The two oscilloscope campaigns answer Limitation 5
   directly, at 976 kS/s against the paper's 1 Hz.

## Reproducing this

```
PYTHONPATH=. python3 scripts/camera_ready_manifest.py     # stream manifest + tiers
git ls-tree -r -l main                                    # file inventory; resolve
                                                          # <300-byte blobs as LFS pointers
```

The published/unpublished predicate is
`tier in {A_camera_ready, B_rebuttal} and paper_gpu_class(gpu_name) is not None`,
using `classifier/data_sources.paper_gpu_class`.
