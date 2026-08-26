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
