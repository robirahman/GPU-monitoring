# Physical-Sensor Workload Detection & Adversarial Red-Teaming

**Hardware:** SAIGE testbed `node1` — 3× NVIDIA GeForce RTX 3090 (Ampere), AMD EPYC 7702, driver 595.71.05.
**Date:** 2026-06-24.
**Scope:** Extends the Phase-1 NVML workload-classification work (arXiv:2606.19262) to a new,
*physical* observation channel: acoustic + power-rail current sensing. We (1) gained programmatic
access to the testbed's physical sensors, (2) trained a training-vs-other classifier from physical
measurements *only* (no NVML), (3) extended the leave-one-GPU-out (LOGO) analysis to the 3090, and
(4) ran one round of adversarial red-teaming against the physical classifier.

This document is written for replication and as source material for the follow-up paper.

**Merge provenance:** The June 24 branch's cross-GPU results are archived as
[`cross_gpu_transfer_binary_rf.csv`](../classifier/results/physical_sensor_20260624/cross_gpu_transfer_binary_rf.csv)
and the matching [PNG](../report/figures/physical_sensor_20260624/cross_gpu_heatmap_binary_rf.png)
/ [PDF](../report/figures/physical_sensor_20260624/cross_gpu_heatmap_binary_rf.pdf).
They use a different corpus from the existing `main` report results, which remain
at their original paths. Neither set was regenerated during the merge. The shared
feature selector now excludes `binary_label` to prevent target leakage when the
heatmap is regenerated; archived scores have not been revalidated with that fix.

---

## 1. The physical sensing setup

The testbed is instrumented by a DAQ host (the "workstation") exposed through the SAIGE dashboard
(`https://gpu.felkru.com`). Instruments:

| Instrument | Detail | Channels |
|---|---|---|
| **UltraMic 200K** | USB acoustic mic, 200 kHz | coil-whine / acoustic |
| **2× PicoScope 2204A** | USB oscilloscopes, 100 kHz, current clamps on power rails | `GPU0`, `GPU1`, `GPU2`, `MOBO` |
| NVML power | per-GPU power/util readout | (software; excluded from the *physical* classifier) |

A "lazy" sensor-streamer samples only while a client polls; it idles out ~12 s after polling stops.

### 1.1 Programmatic access

Live snapshot API (authenticated with the dashboard session cookie):

```
GET https://gpu.felkru.com/sensors/api/snapshot   # one JSON snapshot; keeps capture warm
GET https://gpu.felkru.com/sensors/api/health     # channel_map + sensor availability
```

Snapshot schema (scalars used by the classifier, plus downsampled `wave[]` arrays we currently drop):

```json
{ "ts": <epoch>,
  "mic":   {"rms":.., "dbfs":.., "peak":.., "dominant_hz":.., "rate_hz":200000, "wave":[...]},
  "scopes":[{"label":"GPU0","rms_mv":.., "peak_mv":.., "range_mv":100, "wave":[...]}, ...],
  "power": {"total_w":.., "gpus":[{"index":0,"power_w":..,"util_pct":..}, ...]} }
```

`channel_map` (from `/sensors/api/health`): the clamps map to `GPU0/GPU1/GPU2/MOBO`.
Recorded full-rate captures are also written by the DAQ orchestrator to `/raid` (mounted into testbed
containers) — empty in our container, so we used the live API.

### 1.2 Logger

`scripts/sensor_logger.py` — `SensorLogger` polls the snapshot API at 1 Hz and writes a timestamped,
`run_id`-keyed parquet that mirrors the NVML `TelemetryCollector` schema (auto re-login on cookie
expiry). Credentials read from `/root/temp_login.txt` (`username … password …`) or `SAIGE_USER`/`SAIGE_PASS`.

```bash
python scripts/sensor_logger.py --health
python scripts/sensor_logger.py --duration 60 --output sensor_logs/test.parquet
```

**Validation (ground truth):** a CUDA matmul burn on `cuda:0` drove the node1 `GPU0` current clamp
0 → ~5 mV and back; `cufft_benchmark` drove it 0 → ~31 mV; the snapshot's NVML power matched our
`pynvml` reading to the decimal. The instruments observe *our* container's workloads → this container
is node1.

### 1.3 Paired collection

`scripts/run_workload_with_sensors.py` runs a workload while logging **NVML telemetry → `data/`** and
**physical sensors → `sensor_logs/`** under one shared `run_id`.

> ⚠️ The mic and MOBO rail are **node-shared**. Run **one workload at a time** (target GPU 0, others
> idle) or the physical labels are contaminated. `scripts/collect_3090_suite.sh` does this sequentially.

```bash
python scripts/run_workload_with_sensors.py --workload bert_sst2 --gpu-index 0 --no-dcgm --timeout 220
bash scripts/collect_3090_suite.sh        # full suite, 2 reps, sequential
```

> Sensor parquets live in `sensor_logs/`, **not** `data/` — the NVML classifier's `load_all_data()`
> recursively globs `data/` and would ingest them as all-zero telemetry.

---

## 2. Dataset collected on the 3090

`collect_3090_suite.sh` collected **2 reps each** of 18 workloads (one at a time on GPU 0):
training (resnet/mlp/gpt2/bert ±AMP), inference (resnet50), HPC (cufft, nbody), mining (ethash proxy),
rendering, idle, and the adversarial families K/L/composite/H/J. ~40 runs, ~5.8k 1-Hz paired samples.
(GPT-2/BERT use the workloads' synthetic-text fallback — `datasets` 5.0.0 dropped the script-based
`wikitext`/`glue` loaders — which does not change the GPU compute signature.)

### 2.1 Discriminative physical structure

Per-second `GPU0` current-clamp RMS, summarized over a run:

| Workload | class | rms_mean (mV) | rms **cv** | rms **FFT-peak** | util |
|---|---|---|---|---|---|
| idle | other | 4.1 | 0.56 | 0.17 | 0% |
| mining (ethash) | other | 1.8 | 0.56 | 0.05 | 17% |
| **cufft** | other | **31.4** | **0.19** | **0.09** | 95% |
| resnet (train) | training | 34.8 | 0.25 | 0.16 | 90% |
| gpt2 (train) | training | 22.5 | 0.32 | 0.27 | 90% |
| bert (train) | training | 20.6 | 0.36 | 0.30 | 83% |

**Key insight (drives everything below):** training and steady HPC (cufft) *both* draw high current.
The features that separate them are **burstiness (cv)** and **periodicity (FFT peak)** of the
per-second RMS — both higher for training (the forward→backward→optimizer step cadence). The current
clamp measures rail *ripple*, not raw watts: a compute-bound GEMM draws high power but *low, steady*
ripple; memory-bound work (cufft, training) draws high ripple.

---

## 3. The physical-sensor classifier

`classifier/physical_sensor_classifier.py`. **Physical features only — no NVML, not even the
snapshot's NVML power.** Channels: `GPU0` & `MOBO` current clamps (rms/peak/range mV) + mic
(rms/dbfs/peak/dominant_hz). 13 stats per channel (mean/std/percentiles/iqr/cv/skew/kurt) + temporal
(acf1, acf5, FFT period/peak on the GPU0-clamp RMS and mic RMS) = **99 features/window**. RandomForest
(400 trees), StratifiedGroupKFold (grouped by `run_id`).

```bash
python classifier/physical_sensor_classifier.py   # -> classifier/results/physical_binary_results.csv
```

### 3.1 Results — training vs other (binary)

| Window | NVML (163 feat) | **Physical only (99 feat)** |
|---|---|---|
| 15 s | 100.0% | 99.2% |
| 30 s | 99.4% | **100.0%** |
| 60 s | 99.3% | **100.0%** |

**Training is separable from other workloads using physical measurements alone**, on par with NVML.

> Caveat: this 3090 set is small (40 runs, ~13 workloads/class), so 100% is optimistic vs. the
> Phase-1 9-GPU / 20-adversarial-family diversity (98.2%). Grouped-by-run CV prevents same-run leakage,
> but two reps of a workload are near-identical. Treat as a strong proof-of-concept, not a final number.

---

## 4. Cross-GPU LOGO extended to the 3090

`regen_crossgpu_heatmap_binary_rf.py` (NVML; auto-discovers GPUs — added `RTX 3090` to `GPU_ORDER`).
The repo's other-GPU telemetry is git-LFS; run `git lfs install && git lfs pull` first (≈55 MB, 1816
objects) or only the 3090 appears.

**Generalization onto the Ampere 3090 (train on row GPU → test on 3090):**

| Train GPU → test 3090 | Acc |
|---|---|
| H200, B200, RTX 5080 | **1.00** |
| H100 NVL | 0.96 |
| A100 80GB | 0.87 |
| A100 40GB | 0.77 |
| RTX 4050 | 0.62 |
| H100 80GB | 0.55 |
| RTX 5090 | 0.43 |

The strongest modern datacenter sources (H200/B200) and the RTX 5080 transfer to the older Ampere 3090
essentially perfectly; transfer is **not** universal (RTX 5090 → 3090 = 0.43), so per-architecture
validation still matters. (Caveat: the 3090 set used a single-GPU protocol + synthetic text, a mild
confound vs. the multi-node Phase-1 runs.)

---

## 5. Privileged profiling tiers on this hardware

We installed Nsight Compute 2026.2, Nsight Systems 2026.1, DCGM 4.6, CUDA CLI tools 12.4. Functional
status on the 3090 container (root, but no `CAP_SYS_ADMIN`; driver `RmProfilingAdminOnly=1`):

| Capability | Status | Reason |
|---|---|---|
| NVML (Tier 1) | works | — |
| `nsys` tracing, `nvcc`, `cuda-gdb`, `compute-sanitizer`, CUPTI **activity/trace** | works | not counter-gated |
| `ncu` HW-counter metrics (Tier 3) | **`ERR_NVGPUCTRPERM`** | needs `CAP_SYS_ADMIN` / `RmProfilingAdminOnly=0` |
| DCGM profiling fields (tensor_active, fp16 — Tier 2) | **`-36` unsupported** | GeForce product limit (datacenter-only) |

So the Phase-1 privileged feature sources are both unavailable on a GeForce 3090 — NCU is a fixable
permission wall (relaunch container `--cap-add SYS_ADMIN`, or profile on the host); DCGM DCP metrics
need a datacenter card.

---

## 6. Adversarial red-teaming round 1 (vs. the physical classifier)

Goal: a workload that *really trains* but is classified **non-training** by the physical classifier.
Evaluation: train the classifier on the original 40 runs (attack runs held out as unseen), predict the
attack run's windows; report % windows flagged "training" and mean P(train). All attacks are real
ResNet-18 training and also report **samples/s** (throughput vs. a plain-training baseline).

Workloads: `workloads/adversarial_physical_stealth.py` (ballast masking + dilution) and
`workloads/adversarial_physical_evasion2.py` (throughput-preserving strategies).

| Strategy | Mechanism | Throughput | Verdict (30s / 15s) |
|---|---|---|---|
| baseline | plain training | 1.0× | CAUGHT (P≈0.94) |
| gradaccum | accumulate grads → kill optimizer-step spike | 1.1× | **CAUGHT** (0.96) |
| aperiodic | randomize batch size each step | ~1.0× | **CAUGHT** (0.90) |
| gapfill | throttled gap-filling GEMM ballast | 1.0× | **CAUGHT** (0.89) |
| cofill | continuous bg-thread GEMM ballast | 105× slower | CAUGHT (borderline 0.58) |
| combo | gradaccum + cofill | 112× slower | barely EVADED (0.59) |
| **ballast-dilution** | dominant steady GEMM, training diluted + jittered | **~50× slower** | **EVADED (P=0.27)** |

### 6.1 What worked, and why

**Only `ballast-dilution` reliably evaded** (`adversarial_physical_stealth.py`): a large compute-bound
GEMM ballast pins the GPU at high power but *low, steady* current ripple (a sensor blind spot — the
clamp sees ripple, not watts), while real training is diluted to a small, **jittered** fraction
(removing the FFT periodicity peak: 0.06 vs. training's 0.16–0.30). The resulting signature
(high power/util, low RMS, no periodicity) is out-of-distribution and the RF maps it to "other".

### 6.2 What did NOT work (the important negative result)

**Every throughput-preserving reshaping was caught.** Training's current-draw burstiness and
periodicity are *physical consequences* of forward→backward→optimizer and survive software tricks:
gradient accumulation removes the spike in software but the GPU still computes in bursts; batch
randomization smears the *software* cadence but not the per-step current transients; throttled
gap-filling *added* its own burst/sleep cadence (FFT peak rose to 0.36). Flattening the signal
requires a **dominant** steady load → diluting training → a throughput tax.

### 6.3 Headline finding

There is an inherent **"camouflage tax"** on evading physical detection. Unlike NVML (spoofable in
software at modest cost — Phase-1 saw 43–87% evasion), the physical current/acoustic channel cannot
be cheaply faked: the current draw *is* the computation. In this round the throughput–evasion frontier
sat at **~50×+ slowdown**; no strategy evaded below that. This is direct evidence for the
hardware-enabled-mechanism (HEM) thesis: physical sensing is materially harder to evade than software
telemetry.

### 6.4 Reproduce

```bash
# collect attack runs (one at a time on GPU0)
python scripts/run_workload_with_sensors.py --workload stealth_dilution --gpu-index 0 --no-dcgm --timeout 80 -- \
  python workloads/adversarial_physical_stealth.py --duration 70 --ballast-mode gemm \
    --ballast-size 8192 --ballast-iters 2 --train-every 12
python scripts/run_workload_with_sensors.py --workload ev2_combo --gpu-index 0 --no-dcgm --timeout 80 -- \
  python workloads/adversarial_physical_evasion2.py --mode combo --duration 70
# evaluate: see scratchpad/eval_evasions.py (trains on original 40 runs, scores held-out attack runs)
```

---

## 7. Hardening (adversarial training) + red-team round 2

`classifier/harden_physical_classifier.py`. We adversarial-trained the classifier by adding the
round-1 evasion runs (force-labeled `ml_training`, since they *are* training) to the training set,
then held out **two fresh, unseen dilution variants** (`dil_v2`, `dil_v3` — different seeds/ballast
sizes) plus a full control panel, and compared the un-hardened ("baseline") vs hardened model.

**Hardening result (30 s windows):**

| held-out run | truth | baseline | hardened |
|---|---|---|---|
| `dil_v2`, `dil_v3` (unseen attacks) | training | **0% (EVADE)** | **100% (CATCH)** |
| resnet, gpt2 | training | 100% | 100% |
| cufft, nbody, mining, rendering, idle | other | 0% | **0% (no false positives)** |

Adversarial training catches **unseen** dilution attacks at **zero false-positive cost** — cufft (the
nearest "other", since the GEMM ballast resembles steady compute) stays correctly classified.

**Rounds 2–3 — attack the hardened model by mimicking cufft.** The hardened model now owns the
"low-RMS + low-periodicity + high-power" region (round-1's GEMM signature), so we moved to a different
"other" region: **mimic cufft** with a continuous memory-bound cuFFT ballast (background thread/stream)
+ diluted training, aiming for cufft's high-RMS / low-cv signature.

- **Round 2** (`adversarial_physical_stealth2.py`): 2D 512×512 FFT ballast — only reached RMS ~14.
- **Round 3** (`adversarial_physical_stealth3.py`): the *exact* cufft loop (512×512×64 complex64
  `fftn`/`ifftn`, mirroring `scientific_hpc.run_cufft`) as the ballast, sweeping dilution rate and
  batch size (incl. tiny batch 8).

| run | rms | cv | fft | baseline | hardened |
|---|---|---|---|---|---|
| `s2_fb16/24` (2D FFT) | ~14 | 0.33 | 0.17 | 0% EVADE | **100% CATCH** |
| `s3_tp1/2/3` (exact cufft, batch 128) | **~31** | 0.37 | 0.30 | 0% EVADE | **100% CATCH** |
| `s3_tiny8` (exact cufft, batch 8) | **30.7** | 0.36 | 0.31 | 0% EVADE | **100% CATCH** |
| cufft (reference) | 31.2 | **0.19** | **0.10** | — | 0% EVADE |

**All cufft-mimicry attacks were caught.** Round 3 successfully matched cufft's RMS magnitude (~31)
but **could not match its cv (0.19) or periodicity (0.10)** — they stayed at ~0.36 / ~0.30. The reason
is structural and is the key finding of this round:

> On a single GPU, real training and the cover workload **time-share** the device. Whenever a training
> step runs, it displaces cover compute, so the total rail current is **modulated** (the steady cufft
> draw dips and the training transient adds) — producing an anti-correlated per-second swing that
> raises cv/periodicity above the cover's baseline. Tiny batches and heavier dilution did not fix it.
> Injecting real computation into a steady cover **necessarily** perturbs the physical signal, and the
> (hardened) classifier detects the perturbation.

This strengthens §6.3: the physical channel is robust not just because of a "camouflage tax" but
because **the act of hiding modulates the very signal being measured**. The only round that evaded
(round-1 GEMM dilution) exploited a specific blind spot (compute-bound GEMM's intrinsically low
current ripple), which adversarial training then closed.

### 7.1 Next steps

- Attack vectors not yet tried (likely needed to beat the hardened model): **multi-GPU offloading**
  (train on a GPU whose clamp the defender doesn't read), a cover workload whose current is large and
  variable enough to *absorb* training swings, hardware-level power conditioning, or driving training
  current down with `--cap` clock/power limits (needs permissions).
- Re-harden on rounds 2–3 and iterate, as Phase-1 did for NVML (5 rounds).
- Add the dropped high-rate `wave[]` arrays (200 kHz mic, 100 kHz scope) as features — a far richer
  signature than the 1-Hz scalars, likely raising the attacker's cost further.

### 7.2 Multi-GPU coverage (IN PROGRESS — code ready, data collection pending GPU cleanup)

**Vulnerability:** the original physical classifier only used the **GPU0** clamp, so an adversary
trivially evades it by training on **GPU1/GPU2** — those clamps weren't watched at all.

**Fix (committed):** `classifier/physical_sensor_classifier.py` now uses **every** GPU clamp
(`GPU0/1/2`) plus node-level aggregates — `anyGPU_rms_mv` (max over clamps), `sumGPU_rms_mv`,
`nGPU_active` — and runs the temporal features on `anyGPU_rms_mv`. This makes detection
permutation-invariant to *which* GPU trains, and covers multi-GPU jobs. See `GPU_CLAMPS`,
`augment_aggregates()`, `TEMPORAL`. Supporting pieces: `scripts/run_workload_with_sensors.py --cuda-devices`
(target GPU1, GPU2, or `0,1,2`) and `workloads/multigpu_training.py` (DataParallel).

**To resume (once the GPU is healthy):**
1. `export SAIGE_USER=... SAIGE_PASS=...`
2. `bash scripts/collect_multigpu.sh` — collects training on GPU1, GPU2, GPU0+1+2, GPU1+2, plus
   non-training (cufft/idle) on GPU1/GPU2.
3. `python classifier/eval_multigpu_demo.py` — shows the OLD GPU0-only model being evaded by
   GPU1/GPU2/multi-GPU training vs. the NEW all-clamp model catching it (with no new false positives).

**⚠️ Incident / blocker:** the first multi-GPU collection was launched in a bare SSH session that
dropped, which killed the orchestrator and **orphaned the workload's timeout parent**. The training
process wedged in the NVIDIA UVM driver (`uvm_gpu_retain_by_uuid`, **D-state, unkillable from
userspace**) on **GPU1**; `nvidia-smi` then hung on GPU1/GPU2 (GPU0 stayed responsive). **No `mg_*`
data was saved.** Clear it host-side before resuming: `sudo nvidia-smi --gpu-reset -i 1` (or reload
`nvidia_uvm` / reboot node1). **Run long collections under `nohup`/`tmux` on the host**, never a bare
SSH session, so a dropped connection can't orphan a GPU workload again.
- Add the dropped high-rate `wave[]` arrays (200 kHz mic spectrum, 100 kHz scope) as features — likely
  a much stronger acoustic/electrical signature than the 1-Hz scalars.
- Multi-GPU concurrent collection needs per-GPU isolation for the shared mic/MOBO rail.

## File index

| Path | Purpose |
|---|---|
| `scripts/sensor_logger.py` | Poll snapshot API → timestamped physical-sensor parquet |
| `scripts/run_workload_with_sensors.py` | Run workload + dual NVML/physical logging (shared run_id) |
| `scripts/collect_3090_suite.sh` | Sequential 3090 suite collection (2 reps) |
| `classifier/physical_sensor_classifier.py` | Physical-only binary classifier (99 features) |
| `classifier/harden_physical_classifier.py` | Adversarial-training + retest (hardening experiment) |
| `workloads/adversarial_physical_stealth.py` | Round-1 attack: ballast-masking + dilution |
| `workloads/adversarial_physical_evasion2.py` | Round-1 throughput-preserving attacks (gradaccum/aperiodic/cofill/combo/gapfill) |
| `workloads/adversarial_physical_stealth2.py` | Round-2 attack: cufft-mimicry (2D FFT ballast) |
| `workloads/adversarial_physical_stealth3.py` | Round-3 attack: exact-cufft-loop ballast + diluted/tiny-batch training |
| `regen_crossgpu_heatmap_binary_rf.py` | LOGO heatmap (3090 added to `GPU_ORDER`) |
