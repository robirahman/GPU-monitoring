# Workload-shaping evasion vs. the NVML classifier — results

**Question.** The `llm.c` side-channel patch adds a *workload-shaping* evader
(`-Sw 1`): idle `nanosleep` gaps between training micro-steps reshape the power
and utilization cadence toward request-driven inference, while the arithmetic is
bit-identical. Does this evade the paper's NVML-only training-vs-rest classifier
on the H200, and if so, do DCGM (Tier-2 profiling) or the oscilloscope
(Rogowski dI/dt) catch it?

**Answer: the exploit fails. Shaped training stays fully detected by NVML alone
(0% evasion) at every cadence tested — DCGM and the oscilloscope are not needed
because there is no residual evasion to catch.**

## Setup

- Evader: patched `train_gpt2cu` (GPT-2 124M, tinyshakespeare), one unshaped
  control (`baseline_train`, `-Sw 0`) and five shaping cadences
  (`-Sg` 20/35/50/70/100 ms, jitter ≈ 40%), 3 reps each. Shaping drives NVML
  GPU-utilization from 95% (unshaped) down to 27% (g100) — squarely into the
  inference/idle range. Loss trajectories are **bit-identical** unshaped vs
  shaped: only timing changes.
- Every run captured on all three tiers simultaneously (NVML 1 Hz + Rogowski
  scope ~1 kHz envelope + DCGM profiling ~5 Hz), sharing a run_id.
- Classifier: RandomForest training-vs-rest (`classifier/threeway_improved.py`
  features), 30 s / 15 s windows.

## Result 1 — headline: the paper's NVML classifier, shaped held out

Trained on the full corpus with the shaped family held out; scored per cadence.

| cadence | NVML util | detection | **evasion** |
|---|---|---|---|
| g20 … g100 | 65% → 27% | 1.000 | **0.000** |

All cadences detected 100%. (Caveat: unshaped `baseline_train` is itself only
~20–40% detected by the corpus classifier, because `llm.c` allocates all memory
up-front and lacks the PyTorch growing-allocation startup signature — it is
partly out-of-distribution. The node-local test below removes that confound.)

## Result 2 — node-local sensor ablation (the clean test)

Classifier trained on an on-node balanced set where every run carries all three
sensors (llm.c baseline + PyTorch training as positives; resnet50 inference,
cuFFT, n-body, idle as negatives), shaped held out. Identical runs/windows
across tiers — only the feature columns change.

| cadence (util%) | NVML | +DCGM | +scope | ALL |
|---|---|---|---|---|
| g20 (65%) | 0.000 | 0.000 | 0.000 | 0.000 |
| g35 (52%) | 0.000 | 0.000 | 0.000 | 0.000 |
| g50 (43%) | 0.000 | 0.000 | 0.000 | 0.000 |
| g70 (37%) | 0.000 | 0.000 | 0.000 | 0.000 |
| g100 (27%) | 0.000 | 0.000 | 0.000 | 0.000 |

NVML alone already gives zero evasion; the extra sensors have nothing left to
add against this evader.

## Why NVML catches it — memory footprint, not cadence

Top NVML features (node-local): `mem_used_mb_{max,p95,p50,p75}`,
`first_30s_mem_delta_{mb,frac}` (the optimizer-state startup), and
`mem_util_minus_gpu_util`. **The discriminator is the persistent memory
footprint, not the utilization cadence the evader reshapes.** GPT-2 training
holds weights + gradients + Adam m/v + master weights resident; the shaping gaps
are idle sleeps that never free memory, so `mem_used_mb` is invariant across
cadence (see table below). This is a stronger form of the paper's
"compulsory-DRAM-traffic" argument: the *static allocation* is visible in plain
NVML `mem_used_mb`, before any traffic or profiling counter is consulted.

The evader's own cadence does not help it: shaped training's utilization
periodicity is statistically indistinguishable from real inference's
(`gpu_util_fft_period` ≈ 11 for both `shaped_g50` and `qwen2.5-7B` inference), so
the periodic gaps neither create nor remove a training signature — the memory
footprint decides.

Generalization holds: with the classifier trained on the **corpus only (never
having seen any llm.c run)**, shaped llm.c training is still detected 100% at
every cadence — it is not memorizing llm.c, it generalizes from PyTorch
training's memory signature.

## Per-cadence signatures (node-local means)

| label | util % | **mem_MB** | dram | ripple_A | edge_Hz |
|---|---|---|---|---|---|
| baseline_train (unshaped) | 98 | **12826** | 0.363 | 10.39 | 213 |
| shaped_g20 | 67 | **12826** | 0.252 | 7.74 | 103 |
| shaped_g35 | 53 | **12826** | 0.203 | 6.32 | 83 |
| shaped_g50 | 44 | **12826** | 0.166 | 5.39 | 69 |
| shaped_g70 | 38 | **12826** | 0.137 | 4.52 | 56 |
| shaped_g100 | 28 | **12826** | 0.111 | 3.65 | 44 |
| *resnet50_inference (rest)* | 51 | *4843* | 0.251 | 4.14 | 89 |
| *cuFFT (rest)* | 99 | *2005* | 0.699 | 4.79 | 376 |
| *idle (rest)* | 0 | *613* | 0.000 | 0.40 | 259 |

`mem_used_mb` is **identical (12826 MB) across all six cadences** and 2.6–6×
above every non-training class — the invariant that shaping cannot touch, and
the reason NVML alone gives zero evasion. Note that at *matched* utilization
(shaped_g50 vs resnet50 inference, both ~50%): memory separates them cleanly
(12826 vs 4843 MB), while **DCGM `dram_active` is inverted** (0.166 < 0.251) and
**scope ripple barely differs** (5.4 vs 4.1 A) — so the two physical tiers do not
add discrimination here; NVML memory carries it.

On the scope, both ripple amplitude and my envelope edge-rate *fall* with
shaping (they track the active duty cycle), rather than rising as a naïve
gap-counting argument might suggest — the envelope metric is dominated by
compute-phase micro-fluctuation density, which drops as the GPU idles in each
gap; the added gap-boundary transitions (~10 Hz) are subdominant. Either way the
scope is not needed: NVML memory settles it.

## Interpretation and limits

- **This is a negative result for the evader and a positive result for the
  defender**, consistent with `SIDECHANNEL.md`'s own limitation #1: `-Sw 1`
  "does not make this inference … every FLOP and every weight update is
  identical." It only tests whether the detector keys on cadence — it does not;
  it keys on memory.
- **Where the extra sensors *would* matter**: this evader does not attack the
  feature the classifier actually uses (memory). An evader that *also* minimizes
  the resident footprint — gradient checkpointing + tiny batch + optimizer-state
  offload (the `E_memory_minimal` / `N_grad_accum` families) — is the case where
  NVML memory features weaken and DCGM `dram_active` shape or the scope ripple
  could carry the detection. Combining shaping **with** memory-minimization is
  the natural next experiment; the three-tier capture pipeline built here is
  ready for it.
- The node-local positive class is small (24 training windows); the full-corpus
  generalization test (thousands of training windows) corroborates the result.

---

# Follow-up: attacking the memory feature directly

Shaping fails because it never touches memory. The natural next evader attacks
memory itself, via a tiny micro-batch (`-b 1`) with gradient accumulation (`-d`,
mathematically the same training) plus activation recompute (`-r 2`) — collapsing
the resident footprint from 12.8 GB to ~3.1 GB — and a new `-Sm <MiB>` flag that
holds a fixed padding allocation to *spoof* a chosen footprint. Three new
families (×3 reps, all three sensors):

| evader | mem | util | result (full corpus, NVML) |
|---|---|---|---|
| `mm_min_noshape` (b1, no gaps) | 3.1 GB | 97% | **evasion 1.000 — EVADES** |
| `mm_min_shaped` (b1 + gaps) | 3.1 GB | 9% | evasion 0.000 — caught |
| `mm_spoof_infer_shaped` (b1 + pad to 4.8 GB + gaps) | 4.8 GB | 9% | evasion 0.000 — caught |

**Memory minimization is a real evasion.** `mm_min_noshape` — the *same* GPT-2
training, just at micro-batch 1 — evades the deployed training-vs-rest
classifier completely (evasion 1.000 on both the node-local and the full
29k-window corpus). It defeats the "training ⇒ large memory footprint"
heuristic: at 3.1 GB and 97% utilization it looks like a dense HPC kernel
(cuFFT/n-body), and the classifier calls it "rest".

**But it is a coverage gap, not a sensor limit — and NVML closes it, not
DCGM/scope.** A classifier *trained* to separate `mm_min` from HPC catches it at
**100% recall using NVML alone**; DCGM and the scope add nothing, because plain
NVML already separates the two:

| workload | util | mem_MB | dram | ripple_A | sm_act |
|---|---|---|---|---|---|
| `mm_min_noshape` (hidden training) | 98% | 3718 | 0.141 | 8.12 | 0.579 |
| cuFFT (honest HPC) | 99% | 2005 | 0.699 | 4.79 | 0.941 |
| n-body (honest HPC) | 99% | 2061 | 0.431 | 5.41 | 0.930 |

Hidden training sits at *higher* memory, *lower* DRAM-active and *lower* SM-active
than the HPC it mimics (a matmul-bound GPT-2 step is less DRAM- and
occupancy-saturated than cuFFT). The scope ripple also separates them (8.1 vs
4.8 A), but it is redundant. **So the fix is the paper's own adversary–defender
iteration — add tiny-batch/grad-accumulation training to the classifier's
training set — not new hardware.**

Two secondary findings: (1) *shaping the memory-minimized evader down to 9%
utilization backfires* — it becomes anomalous enough (low util + low mem + a
weak periodic training structure) that the full corpus catches it. (2) *Padding
the footprint up with `-Sm` to mimic an inference server also backfires* — the
(4.8 GB, 9% util) point matches no honest workload and is caught. The evader's
best move is the plain `mm_min_noshape`: look like HPC, don't over-engineer.

**Bottom line.** Timing shaping does not evade; memory-footprint minimization
does, against the *current* model — but it is repairable within NVML by
covering tiny-batch training, so it does not, on this evidence, motivate adding
DCGM or an oscilloscope to the monitor. The case that would motivate them is a
memory-minimized evader that NVML genuinely cannot separate from a benign
workload; we did not find one here (the DRAM-activity and SM-occupancy shapes,
still in NVML's reach via `mem_utilization_pct`, kept training separable).

---

# Harder case: flattening the DRAM / occupancy shape

The remaining NVML tell separating tiny-batch training from HPC was the
DRAM-activity shape: bare `mm_min` sits at `mem_utilization` ≈ 18% (compute-bound
matmul), while cuFFT/n-body are 55–90% (bandwidth-bound). Two attacks on that
tell:

- **`mm_freq`** — gradient-accumulation 1 (`-d 1024`), so a full weight update
  fires every micro-step, making the compulsory weight-write DRAM traffic
  near-continuous. This raised `dcgm_dram_active` from 0.14 to **0.22** (and
  `mem_utilization` ~18→29%) with no second process.
- **`mm_covered`** — a concurrent steady DRAM-bandwidth cover kernel
  (`scripts/cover_kernel.py`) to pin aggregate DRAM/occupancy flat and high.
  *Engineering limitation:* on a single GPU the cover and the training serialize
  on the default CUDA stream, so a steady cover starves the tiny-batch training
  (runs did not complete). Making this work needs MPS or stream-priority
  isolation; not pursued further here.

**Result: flattening the DRAM shape does not produce a harder case.** `mm_freq`
still evades the deployed classifier (evasion 1.000 — it still reads as HPC), and
is still separable from HPC using **NVML alone** (recall 1.000); DCGM and scope
remain redundant.

| evader | util | mem_MB | dram | full-corpus evasion | NVML recall vs HPC |
|---|---|---|---|---|---|
| `mm_min_noshape` | 98% | 3718 | 0.14 | 1.000 (evades) | 1.000 (separable) |
| `mm_freq` | 98% | 3710 | 0.22 | 1.000 (evades) | 1.000 (separable) |

The reason flattening one feature does not help: **NVML exposes many correlated
tells, and the evader must flatten all of them at once.** Raising DRAM activity
leaves the resident footprint (3.7 GB vs HPC's 2.0 GB), the PCIe traffic, and the
power/occupancy shape still separating training from HPC. Each is tied to the
real computation, so driving one toward a benign target does not move the others.
That is *why* NVML stays sufficient here, and why we still did not find a
memory-minimized evader that only a physical sensor could catch.

**Overall conclusion of the study.** Against this classifier: (1) timing shaping
fails; (2) memory-footprint minimization is a real evasion of the *deployed*
model but is NVML-repairable (label tiny-batch training); (3) attempts to flatten
the residual DRAM/occupancy shape do not create a case requiring DCGM or the
oscilloscope. On this evidence the physical sensors are not yet justified as
additions to the monitor — the open door for them remains a hypothetical evader
that simultaneously flattens *every* NVML feature (footprint, DRAM, PCIe,
occupancy, power) while still training, which we were not able to construct
without co-execution isolation (MPS) we did not have time to set up.

---

# Week 2 follow-ups (forward research, not the paper)

## Raw oscilloscope waveform (Workstream A)

New `classifier/scope_raw_spectrum.py` mines the 1 MS/s raw current snippets the
envelope discards. Finding: the **current-ripple amplitude** (`raw_amps_rms`)
separates high-utilization memory-minimized training (mm_min/mm_freq ≈ 8.7 A)
from HPC (cuFFT/n-body ≈ 2.7–4.1 A) and inference (≈ 4.1 A) — a signal orthogonal
to NVML memory/util/power (training draws *lower* mean power than cuFFT yet higher
AC ripple). The specific optimizer-step spectral line is NOT a clean tell: GPU VRM
/ clock harmonics give cuFFT and n-body strong periodic lines too. So amplitude,
not periodicity, is the scope's discriminator — and it is the signal to test
against a fully-NVML-flattened evader (Workstream D). (Also fixed a latent bug:
`load_all_data` was ingesting `_scope_`/`_dcgm_` sidecar parquets as garbage runs.)

## Defender iteration on the memory-min evaders (Workstream B)

`classifier/adversary_defender_iteration.py` (added `--holdout-prefixes`), 60 s
windows, grouped CV. Detection rate (= 1 − evasion):

| evader | R1 clean | R2 hardened (family IN train) | R2 hardened (family HELD OUT) | R3 +physics, held out |
|---|---|---|---|---|
| mm_min_noshape (98% util) | 0% | 100% | **0%** | **42%** |
| mm_freq (98% util) | 40% | 100% | 40% | 40% |
| mm_min_shaped (9% util) | 33% | 100% | 100% | 100% |
| mm_spoof_infer_shaped (9% util) | 0% | 100% | 100% | 100% |

Reading: (1) the deployed (clean) monitor is evaded by memory minimization; (2) it
is fully repairable **if the defender trains on tiny-batch training specifically**
(R2 family-in-train → 100%); (3) but high-util memory minimization is a genuinely
*new* mode — it does **not** generalize for free from the other adversarial
families (R2 held-out mm_min_noshape still 0%), unlike the low-util variants which
look like the existing composite/interleave evaders and are caught by
generalization; (4) the physical-invariant features (energy integral, power-mem
correlation, sustained-compute) partially close it (0→42%) without the family in
training. So covering this evasion needs either the family in the training set or
better physics features — it is not caught by the existing monitor as-is.

## Scale bound: memory-min evasion is confined to sub-~1B models (Workstream C)

Memory-minimized training across a model-size ladder, all triple-sensor, each x3
reps (`scripts/run_scale_sweep.py`, `data/h200_nvl_scope/scale/`). Resident
`mem_used` (already minimized where possible: full-FT uses grad-ckpt + batch 1;
LoRA at batch 1–2):

| workload | kind | mem_used | vs benign band? |
|---|---|---|---|
| idle | rest | 0.6 GB | — |
| cuFFT | rest (HPC) | 2.0 GB | benign |
| GPT-2 124M `mm_min` | **TRAINING** | **3.1 GB** | **IN band → evades** |
| TinyLlama-1.1B inference | rest | 3.5 GB | benign |
| resnet50 inference | rest | 4.8 GB | benign (band top) |
| Qwen-1.5B LoRA | TRAINING | 6.8 GB | above → caught |
| Qwen-3B LoRA | TRAINING | 8.8 GB | above → caught |
| **TinyLlama-1.1B full-FT** | TRAINING | **12.2 GB** | above → caught |
| Mistral-7B LoRA | TRAINING | 15.2 GB | above → caught |

The benign (non-training) band tops out at ~4.8 GB. Only the 124M model's training
footprint (3.1 GB) fits inside it — every training run ≥ 1.5B params, LoRA or
full-FT, sits at 6.8–15.2 GB, well above, and is caught by NVML `mem_used` alone.

**Decisive contrast — same model, TinyLlama-1.1B:** inference = 3.5 GB (benign,
hides trivially) vs full fine-tune = 12.2 GB (3.5× the band). The ~8.7 GB
difference is the optimizer state (fp32 master + Adam m + v + gradients), the
irreducible resident tell that memory-minimization cannot remove for real
fine-tuning.

**So the memory-minimization evasion is a small-model artifact.** It works for
toy models (GPT-2-124M) whose entire training state fits within an inference-sized
footprint, and fails for anything at a scale worth governing: full fine-tuning of
even a 1.1B model, or LoRA of a ≥1.5B model, cannot bring its resident footprint
into the benign band, so plain NVML memory catches it. (Caveat: optimizer-state
CPU-offload or 8-bit optimizers could lower the full-FT floor somewhat — a
worthwhile follow-up — but they trade the memory tell for a PCIe-traffic or
compute tell, and don't apply to the weight-resident LoRA floor.)

## Co-execution cover under CUDA MPS (Workstream D)

The remaining "flatten every NVML feature" attack: run a steady DRAM-bandwidth
cover kernel concurrently with tiny-batch training so the aggregate NVML signature
reads as HPC (`scripts/cover_kernel.py`, `covered_train_mps.sh`, CUDA MPS daemon).

- **The signature does flatten.** With a saturating cover (MPS, intensity 0.6) the
  combined GPU reads util 100% / mem_utilization 100% / mem_used 4.8 GB — visually
  identical to cuFFT at an inference-sized footprint.
- **But the training is starved to a standstill.** Under the same cover the
  tiny-batch trainer made **0 optimizer steps in 90 s** (killed at timeout, never
  past init). MPS shares SM slots concurrently but does **not** partition DRAM
  bandwidth, so the cover — which must saturate DRAM to flatten `mem_utilization`
  — leaves no bandwidth for the training's weight/activation traffic.
- **This is the fundamental tension:** flattening the DRAM/util signal requires
  saturating the very resource the training needs, so a cover strong enough to
  hide the training's signature also prevents it from computing. A cover weak
  enough to let training run only partially raises the signal — and by
  Workstream A the current-ripple amplitude still separates the (now-running)
  training from genuine HPC. So a co-execution cover is not a clean NVML evasion.
  (Unresolved refinement, blocked this session by a degraded Tailscale link:
  MPS `CUDA_MPS_ACTIVE_THREAD_PERCENTAGE` partitioning + a light cover, to map the
  partial-flatten / training-throughput trade-off precisely. Scaffolding —
  `covered_train_mps.sh` — is committed and runnable when the link recovers.)

## Week-2 synthesis

Across all four workstreams, no evasion both hides from NVML and survives:
- **Timing shaping** — caught by NVML memory (footprint invariant to cadence).
- **Memory minimization** — evades the *deployed* model, but (B) is repairable and
  a genuinely new mode, and (C) is a **small-model artifact**: any training ≥1.5B
  (LoRA or full-FT) can't bring its footprint into the benign band.
- **Co-execution cover** — (D) can flatten the NVML signature but starves the
  training; and (A) the current-ripple amplitude is a physical signal orthogonal
  to NVML that flags training even at HPC-like utilization — the first concrete
  case for the oscilloscope, to be nailed down against a running MPS evader when
  the hardware/link allow.
