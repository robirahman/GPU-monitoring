# Workloads and Evasion Strategies

This document describes every workload in the repository: what it does, what
telemetry signature it produces, and (for adversarial workloads) what classifier
feature it targets.

All workloads are run for a configurable duration (typically 300s) while a
separate telemetry collector polls NVML at 1 Hz, recording 9 signals: GPU
utilization, memory utilization, memory used, power draw, temperature, SM clock,
memory clock, PCIe TX, and PCIe RX. The classifier extracts 159 features from
sliding windows (5-60s) over these signals.

---

## 1. Baseline and Clean Workloads

### 1.1 Idle (`workloads/idle.py`)

Sleeps for the configured duration. Produces near-zero utilization, low power,
low temperature, and idle clocks. Used as a calibration baseline.

### 1.2 ML Training (`workloads/pytorch_training.py`)

Standard supervised training: ResNet-18 or a simple MLP on CIFAR-10 with SGD.
Configurable batch size, learning rate, epochs, and AMP (mixed precision).

**Telemetry signature:** High sustained GPU utilization (90-100%), elevated
memory from model + optimizer + activations, steady power draw, periodic
batch-level patterns from the training loop.

Variants registered: `pytorch_resnet_cifar10`, `pytorch_resnet_cifar10_amp`,
`pytorch_mlp_cifar10`.

### 1.3 GPT-2 Fine-Tuning (`workloads/gpt2_finetune.py`)

Fine-tunes GPT-2 124M on WikiText-2 using HuggingFace Trainer. 400 steps,
configurable batch size and AMP.

**Telemetry signature:** Similar to PyTorch training but with higher memory
usage (transformer activations scale with sequence length) and more variable
utilization (autoregressive attention is less uniform than convolution).

Variants: `gpt2_wikitext2`, `gpt2_wikitext2_amp`.

### 1.4 BERT Fine-Tuning (`workloads/bert_finetune.py`)

Fine-tunes BERT-base on SST-2 sentiment classification. 300 steps.

**Telemetry signature:** Similar to GPT-2 but encoder-only (no autoregressive
decode phase). Steady high utilization.

Variants: `bert_sst2`, `bert_sst2_amp`.

### 1.5 ML Inference (`workloads/inference.py`)

ResNet-50 batch inference on CIFAR-10 test set, forward-only under
`torch.no_grad()`. Loops for the configured duration.

**Telemetry signature:** Lower memory than training (no gradients, no optimizer
states), lower power (no backward pass doubles the FLOPS), and lower power
coefficient of variation (CV). The absence of optimizer state memory and
backward-pass compute is the fundamental signal the classifier exploits.

Variant: `resnet50_inference`.

### 1.6 Modern LLM Inference (`workloads/modern_inference.py`)

Autoregressive text generation with real LLMs:
- **DeepSeek-R1** (1.5B): memory-bandwidth-bound decode, growing KV cache,
  bursty utilization as each token is generated sequentially.
- **Qwen 2.5** (0.5B): similar decode pattern, smaller model.
- **Whisper large-v3-turbo**: encoder-decoder for audio transcription, static
  memory footprint, compute-bound encoder phase then decode phase.

Variants: `deepseek_r1_inference`, `qwen25_inference`, `whisper_inference`.

### 1.7 Scientific HPC (`workloads/scientific_hpc.py`)

Two GPU-compute workloads that are not ML:
- **cuFFT:** Repeated large 3D FFT (512x512x64 complex). Memory-bandwidth
  bound. High SM utilization but no tensor-core use, distinctive memory clock
  pattern.
- **N-body:** Gravitational O(N^2) pairwise force computation on 8K-16K
  particles. Compute-bound FP32. High SM utilization, moderate memory.

Variants: `cufft_benchmark`, `nbody_sim`.

### 1.8 Crypto Mining Proxy (`workloads/mining_proxy.py`)

Simulates Ethash (Ethereum-style) memory-hard hashing using PyTorch:
- Allocates a 1+ GB DAG (directed acyclic graph) tensor
- Performs 64 sequential random gather reads per hash (memory-bound)
- XOR mixing between reads (light compute)

**Telemetry signature:** Very high memory bandwidth utilization, moderate GPU
compute utilization (~75%), distinctive pattern of irregular memory access that
differs from ML's regular tensor operations.

Variant: `mining_ethash_proxy`.

### 1.9 Rendering Proxy (`workloads/rendering_proxy.py`)

Simulates Monte Carlo path tracing (like Blender Cycles):
- Per-ray random walk with hemisphere sampling
- Irregular memory access patterns (per-ray divergence)
- Mixed FP32 compute: dot products, sqrt, transcendentals
- No tensor-core use (no large GEMMs)

**Telemetry signature:** Bursty utilization (frame-by-frame rendering), moderate
power, no large contiguous memory allocations. Very different from ML's regular
matmul-dominated compute.

Variant: `rendering_proxy`.

### 1.10 Blender Render (`workloads/blender_render.py`)

Actual Blender Cycles GPU rendering of the BMW benchmark scene. CUDA backend.

Variant: `blender_bmw`.

### 1.11 FFmpeg NVENC (`workloads/ffmpeg_nvenc.py`)

Hardware video encoding via NVIDIA NVENC. Generates a synthetic test video and
encodes it. Uses the dedicated video encoder hardware, not the CUDA cores, so
GPU compute utilization is low while encoder utilization is high.

Variant: `ffmpeg_nvenc`.

### 1.12 Ethash CUDA (`workloads/ethash_cuda.py`)

Custom CUDA kernel implementation of Ethash mining. More realistic than the
PyTorch proxy but requires CUDA compilation.

Variant: `ethash_cuda`.

### 1.13 GROMACS (`workloads/gromacs_sim.py`)

GROMACS molecular dynamics simulation (ADH benchmark). GPU-accelerated
biomolecular simulation, representative of real HPC workloads.

Variant: `gromacs_adh`.

---

## 2. Distributed Training Workloads

### 2.1 DDP Training (`workloads/pytorch_training_ddp.py`)

Standard PyTorch DistributedDataParallel training. ResNet-18 or ResNet-50 on
CIFAR-10 across multiple GPUs. Uses NCCL backend with allreduce after each
backward pass.

**Telemetry signature:** Same as single-GPU training on each GPU, plus periodic
NVLink/PCIe communication bursts from gradient allreduce. The PCIe TX/RX
features show distinctive periodic spikes that are absent in single-GPU
workloads.

Variants: `ddp_resnet18_cifar10`, `ddp_resnet50_cifar10`,
`ddp_resnet18_cifar10_amp`.

### 2.2 FSDP Training (`workloads/fsdp_training.py`)

Fully Sharded Data Parallel for NVLink stress testing. Supports 7B-70B
parameter models. AllGather/ReduceScatter every forward/backward step.

**Telemetry signature:** Extreme NVLink utilization. Memory per-GPU is lower
than DDP (sharded parameters) but communication is constant, creating a
distinctive sustained high PCIe/NVLink pattern.

Variants: `fsdp_7b_nvlink` through `fsdp_70b_nvlink`.

---

## 3. Adversarial Evasion Strategies

All adversarial strategies perform **real training** (gradients flow, loss
decreases) while modifying the workload to evade the telemetry classifier. The
throughput penalty from evasion is the "evasion tax."

### Overview

| Round | Strategy | File | Evasion Target | Model |
|-------|----------|------|----------------|-------|
| 1 | A: Util Modulation | `adversarial_training.py` | Utilization features | ResNet-18 |
| 1 | B: Low-Util Training | `adversarial_training.py` | Utilization features | ResNet-18 |
| 1 | D: Temporal Disruption | `adversarial_training.py` | Autocorrelation/FFT features | ResNet-18 |
| 1 | Composite | `adversarial_composite.py` | All features simultaneously | ResNet-18 |
| 2 | E: Memory-Minimal | `adversarial_advanced.py` | Memory features | ResNet-18 |
| 2 | F: Workload Interleave | `adversarial_advanced.py` | Window-average features | ResNet-18 |
| 2 | G: Clock-Throttled | `adversarial_advanced.py` | Power/clock/temp features | ResNet-18 |
| 2 | H: Mimicry Training | `adversarial_round2.py` | All features (match target) | ResNet-18 |
| 2 | I: Stochastic Interleave | `adversarial_round2.py` | Autocorrelation features | ResNet-18 |
| 2 | J: PID Thermal-Matched | `adversarial_round2.py` | Utilization/power features | ResNet-18 |
| 3 | K: Online Learning | `adversarial_round3.py` | Training/inference boundary | ResNet-50 |
| 3 | L: Diluted Training | `adversarial_round3.py` | Training/inference boundary | ResNet-50 |
| 4 | M: Composite+MemMin | `adversarial_round4.py` | Memory + power features | ResNet-18 |
| 4 | N: Grad Accum Dilution | `adversarial_round4.py` | Mode-switching artifacts | ResNet-50 |
| 5 | WB-I: Inference Baseline | `adversarial_whitebox.py` | Control (no evasion) | TinyLlama-1.1B |
| 5 | WB-D: Diluted Full FT | `adversarial_whitebox.py` | Memory + compute features | TinyLlama-1.1B |
| 5 | WB-L: Diluted LoRA | `adversarial_whitebox.py` | Memory features | TinyLlama-1.1B |
| 5 | WB-O: Oracle-Guided | `adversarial_whitebox.py` | All features (adaptive) | TinyLlama-1.1B |
| DDP | K-DDP: Online Learning | `adversarial_ddp.py` | Training/inference boundary | ResNet-50 |
| DDP | L-DDP: Diluted Training | `adversarial_ddp.py` | Training/inference boundary | ResNet-50 |
| DDP | L-stagger: Staggered | `adversarial_ddp.py` | NVLink periodicity | ResNet-50 |
| DDP | K-accum: Grad Accum | `adversarial_ddp.py` | NVLink traffic frequency | ResNet-50 |

---

### Round 1: Basic Disguise

These were the first adversarial strategies, designed before seeing the
classifier's features. They modify training's telemetry at a surface level.

#### Strategy A: Utilization Modulation (`adversarial_training.py`)

**Idea:** Insert `torch.cuda.synchronize()` + `time.sleep(random)` between
training batches to create a bursty utilization pattern resembling rendering.

**Mechanism:**
1. Normal ResNet-18 training on CIFAR-10 (batch 256, SGD)
2. After each batch, synchronize GPU and sleep for a random interval
   (default 0.1-0.5s)
3. The sleep gaps lower mean utilization, increase utilization CV, break
   power autocorrelation, and create an irregular duty cycle

**What it targets:** Mean GPU utilization, utilization CV, power ACF1.

**Why it fails:** The classifier learns that Strategy A has a distinctive
bimodal utilization distribution (100% during compute, 0% during sleep) that
doesn't match any real workload. Real rendering has varied utilization within
frames, not binary on/off.

#### Strategy B: Low-Utilization Training (`adversarial_training.py`)

**Idea:** Make training look like inference by using a tiny batch size and
CPU-bottlenecking the data pipeline.

**Mechanism:**
1. Batch size 2 (vs 256 for normal training)
2. Heavy CPU-side augmentation (9 transforms including random crop, color
   jitter, gaussian blur, random erasing)
3. Single DataLoader worker, no pinned memory
4. The GPU sees brief compute bursts separated by long data-loading waits

**What it targets:** Mean GPU utilization (aims for <20%), power draw.

**Why it fails:** The power variability pattern is distinctive. Real inference
has steady moderate utilization; Strategy B has tiny spikes separated by
CPU-wait gaps, creating a unique power CV signature.

#### Strategy D: Temporal Disruption (`adversarial_training.py`)

**Idea:** Destroy all periodicity in the training loop by randomizing
everything.

**Mechanism:**
1. Each step uses a random batch size from {4, 8, 16, 32, 64, 128, 256, 512}
2. 30% chance of injecting a random CUDA matmul (64-2048 dim, 1-20 iterations)
   between steps
3. 20% chance of a random micro-sleep (0-50ms)
4. All data pre-loaded into a single tensor, random-indexed per batch

**What it targets:** FFT periodicity, autocorrelation at all lags, regular
memory allocation patterns.

**Why it fails:** While periodicity is destroyed, the aggregate statistics
(mean utilization, mean power, memory level) still betray training. The random
CUDA noise also creates a distinctive memory pattern.

#### Composite Strategy (`adversarial_composite.py`)

**Idea:** Combine all known evasion techniques with a parameterized idle
fraction to map the evasion tax curve.

**Mechanism:** Four techniques applied simultaneously:
1. **Memory pre-allocation:** Allocate a large buffer upfront (default 4 GB)
   to flatten the memory ramp, spoofing `mem_slope` and `mem_slope_ratio`
2. **Idle padding:** Sleep for fraction F of total time between batches,
   reducing time-averaged power/utilization
3. **Random timing noise:** Vary sleep durations +/-50%, occasionally inject
   dummy CUDA matmul kernels (10% chance)
4. **Gradient accumulation:** Only call `optimizer.step()` every N batches
   (default 4) to reduce periodic optimizer memory spikes

**Key parameter:** `--idle-fraction F` (0.0 to 0.95)
- F=0.0: full throughput, no evasion
- F=0.5: 50% throughput loss
- F=0.95: 95% throughput loss

Sweeping F produces the **evasion tax curve**: the throughput cost to achieve
a given detection evasion rate.

---

### Round 2: Counter-Hardened Strategies

Designed after analyzing which classifier features betrayed Round 1. These
strategies directly address the physical signals that exposed earlier evasion.

#### Strategy E: Memory-Minimal Training (`adversarial_advanced.py`)

**Idea:** Make training's memory footprint match inference.

**Mechanism:**
1. **Gradient checkpointing:** `checkpoint_sequential` on ResNet-18 segments.
   Recomputes activations during backward pass instead of storing them. Trades
   compute for memory (3-5x slower).
2. **Tiny batch size:** 4 (minimizes activation memory)
3. **SGD with no momentum:** No momentum buffers to store
4. **Aggressive GC:** `torch.cuda.empty_cache()` + `gc.collect()` after every
   step. Immediately deletes all intermediates.
5. **`zero_grad(set_to_none=True)`:** Deallocates gradient tensors instead of
   zeroing them.

**What it targets:** Memory-used features (mean, max, slope), memory
utilization.

**Result:** Memory stays ~500-1500 MB, close to inference. But the aggressive
GC creates a distinctive memory oscillation pattern (memory clock CV=0.91 vs
0.01 for normal workloads).

#### Strategy F: Workload Interleaving (`adversarial_advanced.py`)

**Idea:** Fill idle time with genuine non-training compute so the 60s window
features are a weighted average of training and non-training.

**Mechanism:**
1. Fixed cycle length (default 10s)
2. Spend `train_frac` (default 30%) of each cycle training ResNet-18 (batch 64,
   Adam)
3. Spend the remainder doing actual non-ML work:
   - Even iterations: mining-like memory-hard random access (512 MB DAG)
   - Odd iterations: cuFFT (8x 512x512 complex)
4. `empty_cache()` between phases to prevent memory accumulation

**What it targets:** Window-average features (mean power, mean utilization,
memory mean). The non-training compute fills in "normal" feature values.

**Result:** Features within a window are indeed averaged. But the fixed 10s
cycle creates periodic autocorrelation (acf10, acf20) that the classifier
learns as a distinctive signature.

#### Strategy G: Clock-Throttled Stealth Training (`adversarial_advanced.py`)

**Idea:** Force the GPU into a low-power regime where training telemetry
matches idle/light compute.

**Mechanism:**
1. Set power limit via `nvidia-smi -pl` (default 100W, vs typical 300W TDP)
2. Lock SM clocks to 210-600 MHz via `nvidia-smi --lock-gpu-clocks`
3. Small batch size (32) to reduce per-step power
4. 10ms sleep between steps for additional power reduction
5. Always restores original power/clock settings on exit (even on crash)

**What it targets:** Power draw, SM clock frequency, temperature features.

**Result:** Power/clock/temperature successfully match light compute. But GPU
utilization is too uniformly ~22% (real idle/inference has more variation), and
the memory footprint still shows training-level allocation.

#### Strategy H: Mimicry Training (`adversarial_round2.py`)

**Idea:** Don't just disguise training — actively match a specific target
workload's telemetry profile using feedback control.

**Mechanism:**
1. Define target profiles from measured clean runs:
   - Mining: 75% util, 140W, util range 60-90%
   - cuFFT: 83% util, 165W, util range 70-95%
2. On each iteration, randomly decide (with probability `train_probability`)
   whether to do a training step or a mimicry step
3. **Mimicry steps** run the target workload's actual compute:
   - Mining: memory-hard DAG random access (512 MB)
   - cuFFT: 3D FFT forward+inverse
   - N-body: pairwise gravitational force computation
4. **Feedback loop:** Read GPU utilization via pynvml every 30s. If utilization
   is too high vs target, reduce `train_probability` by 0.05. If too low,
   increase it. (PID-like adjustment.)
5. Use random per-step decisions (not periodic phases) to avoid autocorrelation

**What it targets:** All aggregate features simultaneously — by matching a real
workload, all features should fall within that workload's distribution.

**Key insight:** Avoids creating a distinctive evasion signature by matching an
existing category rather than creating a new pattern.

#### Strategy I: Stochastic Interleaving (`adversarial_round2.py`)

**Idea:** Fix Strategy F's periodic autocorrelation by randomizing phase
lengths and varying the non-training workload.

**Mechanism:**
1. Each phase has a random length drawn from an exponential distribution
   (mean=5s, clipped to 1-15s)
2. Phase type (train vs other) chosen randomly per phase based on `train_frac`
3. Non-training phases randomly choose between mining, cuFFT, or idle
4. No fixed cycle length — destroys autocorrelation at all lags

**What it targets:** Autocorrelation features (acf5, acf10, acf20) that
betrayed Strategy F.

#### Strategy J: PID Thermal-Matched Training (`adversarial_round2.py`)

**Idea:** Use a proper PID controller on live GPU telemetry to precisely match
a target utilization profile.

**Mechanism:**
1. Target utilization (default 75%, matching mining)
2. Read GPU utilization via pynvml every 500ms
3. PID controller (Kp=0.0005, Ki=0.00005, Kd=0.0001) adjusts a sleep
   duration between training batches
4. If utilization too high: increase sleep → utilization drops
5. If utilization too low: decrease sleep → utilization rises
6. Operates at normal GPU clocks (unlike Strategy G), just modulates the
   compute duty cycle

**What it targets:** GPU utilization mean and variance, power draw profile.

**Difference from G:** Strategy G locks clocks low, creating abnormal clock
features. Strategy J keeps clocks normal but uses duty-cycle modulation,
producing normal clock/memory features with controlled utilization.

---

### Round 3: Training-as-Inference

These are the strongest single-GPU evasion strategies. Instead of disguising
training as a non-ML workload, they minimize the *delta between training and
inference* for the same model.

#### Strategy K: Online Learning Camouflage (`adversarial_round3.py`)

**Idea:** The minimum possible difference from inference is adding
`loss.backward() + optimizer.step()` to each forward pass.

**Mechanism:**
1. Structurally identical to `inference.py`: ResNet-50, CIFAR-10 test set,
   batch 256, same data pipeline
2. **Only difference:** removes `torch.no_grad()`, adds `loss.backward()` +
   `optimizer.step()` after each forward pass
3. SGD with lr=0.001, **no momentum** (zero persistent optimizer state)
4. `optimizer.zero_grad(set_to_none=True)` immediately frees gradient memory

**What the classifier must detect:** The backward pass approximately doubles
the FLOPS per batch, which shows up as ~2x higher sustained power draw and
higher GPU utilization than pure inference. But with zero optimizer state, the
memory footprint is close to inference.

**Key principle:** This is the theoretical minimum delta. Any training must
do at least a backward pass, which at minimum doubles compute.

#### Strategy L: Diluted Training (`adversarial_round3.py`)

**Idea:** Only train on 1 out of every N batches. The rest are pure inference.
This is the parameterized version of Strategy K.

**Mechanism:**
1. Same setup as K: ResNet-50, CIFAR-10 test set, batch 256
2. Loop over batches. On batch index divisible by N:
   - `model.train()`, forward + backward + optimizer step
3. On all other batches:
   - `model.eval()`, `torch.no_grad()`, forward only
4. `empty_cache()` after each training batch to prevent memory fragmentation
   from mode switching

**Key parameter:** `--dilution N`
- N=2: 50% of batches train (strongest training, easiest to detect)
- N=5: 20% of batches train
- N=10: 10% of batches train
- N=20: 5% of batches train (hardest to detect, slowest learning)

**What the classifier sees:** At N=20, 95% of the telemetry windows are pure
inference. The training batches create brief power/utilization spikes that may
or may not fall within a given 60s window. Detection depends heavily on window
size and whether a training batch occurs within it.

**Weakness:** The `model.train()`/`model.eval()` mode switching creates
utilization transitions that the classifier can detect. Strategy N (Round 4)
addresses this.

---

### Round 4: LOO Weakness Exploitation

Designed after leave-one-strategy-out analysis revealed specific classifier
gaps.

#### Strategy M: Composite + Memory Minimal (`adversarial_round4.py`)

**Idea:** Combine the strongest elements from two partially-successful
strategies: composite's idle padding with Strategy E's memory minimization.

**Mechanism:** Applies all of the following simultaneously:
1. **Memory pre-allocation** (from composite): allocate configurable buffer
   upfront to flatten memory slope
2. **Gradient checkpointing** (from E): recompute activations during backward,
   using `checkpoint_sequential` on ResNet-18 segments
3. **Tiny batch size 4** (from E): minimize activation memory
4. **SGD no momentum** (from E): zero optimizer state
5. **Aggressive GC** (from E): `empty_cache()` + `gc.collect()` after every step
6. **Idle padding with random timing** (from composite): sleep between batches
   with +/-50% jitter
7. **Occasional dummy CUDA kernels** (from composite): 10% chance per step

**Key parameter:** `--idle-fraction` (0.0-0.95)

**What it targets:** Both memory features (via checkpointing + pre-alloc) AND
power/utilization features (via idle padding) simultaneously.

#### Strategy N: Gradient Accumulation Dilution (`adversarial_round4.py`)

**Idea:** Strategy L was partially betrayed by mode-switching artifacts between
`model.train()` and `model.eval()`. Strategy N stays in train mode permanently.

**Mechanism:**
1. ResNet-50, batch 128, always in `model.train()` mode
2. Every batch does `forward + loss.backward()` (gradients accumulate)
3. Only every N-th batch calls `optimizer.step() + zero_grad()`
4. Loss is scaled by `1/N` for correct gradient magnitude
5. No mode switching, no `empty_cache()`, no utilization transitions

**Key difference from L:** Strategy L alternates between train and eval modes.
Strategy N always runs backward passes — the only periodic signal is the
optimizer step every N batches, which is a much subtler memory/power spike than
the train/eval transition in L.

**Tradeoff:** Every batch pays the backward-pass compute cost (2-3x FLOPS of
forward), so the throughput is lower than Strategy L for the same effective
training rate. But the telemetry is more uniform.

---

### Round 5: White-Box Evasion at Scale

**Key difference from Rounds 1-4:** All previous rounds used ResNet-18/50
(~11-25M parameters) on CIFAR-10. Round 5 uses models sized to fill GPU
memory (1.1B+ parameters), which changes the detection problem fundamentally.
Additionally, the adversary has full access to the classifier internals
(features, model, thresholds).

All Round 5 workloads are in `workloads/adversarial_whitebox.py` with the
classifier oracle in `classifier/classifier_oracle.py`.

#### Strategy WB-I: Inference Baseline (`adversarial_whitebox.py --strategy WB-I`)

**Purpose:** Control group. Pure inference with the same large model used by
the evasion strategies.

**Mechanism:**
1. Load TinyLlama-1.1B (or other model matching GPU VRAM) in FP16
2. `model.eval()`, generate synthetic random-token batches
3. Forward-only under `torch.no_grad()`
4. Log throughput and memory

This provides ground truth for what inference telemetry looks like at 1B+ scale.

#### Strategy WB-D: Diluted Full Fine-Tuning (`adversarial_whitebox.py --strategy WB-D`)

**Idea:** Diluted training (like Strategy L) but with a full-scale LLM and
multiple stealth techniques informed by white-box classifier analysis.

**Mechanism:**
1. Load TinyLlama-1.1B in FP16, create AdamW optimizer (lr=1e-5)
2. **Pre-allocate optimizer states:** Run one dummy backward+step at startup.
   This materializes all AdamW momentum and variance buffers immediately,
   preventing the memory growth signal that normally betrays training init.
3. **Never call `model.train()`:** Stay in eval mode, use
   `torch.enable_grad()` manually for training batches. Avoids batch norm
   mode switching.
4. **Gradient checkpointing** (optional, `--grad-ckpt`): Bounds activation
   memory to approximately inference level.
5. **No `torch.cuda.empty_cache()`:** Avoids the distinctive memory compaction
   pattern.
6. Loop: every N-th batch does `forward + backward + optimizer.step()`, all
   others do forward-only under `torch.no_grad()`.

**Dilution parameter:** `--dilution N` (5, 10, 20, 50)

**Model sizing:** At ~16 bytes/param for FP16 AdamW training, a 1.1B model
needs ~17.6 GB total (model weights + optimizer states + gradients). On a 16 GB
GPU, this requires gradient checkpointing to fit.

**What makes this harder to detect than Strategy L:**
- At 1.1B params, inference already uses ~2.2 GB and saturates GPU utilization
  during forward passes. The gap between training and inference utilization is
  much narrower than for ResNet-50.
- Pre-allocated optimizer states eliminate the memory growth signature.
- Gradient checkpointing bounds activation memory.
- The remaining signal is purely the backward-pass compute, which at high
  dilution (N=50) occurs in only 2% of batches.

#### Strategy WB-L: Diluted LoRA Fine-Tuning (`adversarial_whitebox.py --strategy WB-L`)

**Idea:** Instead of full fine-tuning (which requires optimizer states for all
parameters), use LoRA to train only a tiny fraction of parameters. This makes
the memory signature nearly identical to inference.

**Mechanism:**
1. Load TinyLlama-1.1B in FP16
2. **Apply LoRA:** rank-16 adapters on `q_proj` and `v_proj` attention
   matrices via the `peft` library. Adds ~10M trainable params out of 1.1B
   total (~0.9%)
3. AdamW optimizer on LoRA params only. Optimizer state overhead: ~160 MB
   (vs ~13 GB for full fine-tuning)
4. Same diluted loop as WB-D: every N-th batch trains, rest are inference
5. Same stealth techniques: pre-allocate optimizer, eval mode, optional
   grad checkpointing

**Key difference from WB-D:** The optimizer state memory is ~80x smaller. The
memory footprint is within ~160 MB of pure inference — essentially
indistinguishable on a 16 GB GPU. The only remaining signal is the backward
pass compute.

**Tradeoff:** LoRA only updates attention projections, so the training
capability is limited compared to full fine-tuning. But for the purpose of
demonstrating evasion, it trains real parameters and produces real gradient
updates.

#### Strategy WB-O: Oracle-Guided Training (`adversarial_whitebox.py --strategy WB-O`)

**Idea:** The strongest possible white-box attack. Query the actual classifier
in real time and only train when the classifier says it's safe.

**Mechanism:**
1. Same setup as WB-D (full fine-tuning, TinyLlama-1.1B, AdamW)
2. Initialize `ClassifierOracle` from `classifier/classifier_oracle.py`:
   - Trains the exact same XGBoost classifier from existing data
   - Starts a background thread polling pynvml at 1 Hz
   - Maintains a rolling buffer of telemetry samples
3. On each candidate training batch (every N-th):
   - Query `oracle.query_p_training(window_sec=15)` — returns P(ml_training)
     for the last 15 seconds of telemetry
   - Only train if P(training) < threshold (default 0.3)
   - If the classifier thinks the current window looks like training, skip
     the training batch
4. Wait 5s at startup for telemetry buffer warmup

**The ClassifierOracle** (`classifier/classifier_oracle.py`):
- Trains the same balanced XGBoost (500 trees, max_depth=6, lr=0.05) from
  the same data
- Reads 9 NVML signals at 1 Hz via pynvml
- On query: takes the last N seconds of telemetry, computes the same 159
  features using `extract_enhanced_features()` from `threeway_improved.py`,
  runs `predict_proba()`, returns P(ml_training)

**Why this is the strongest attack:** The adversary knows every feature (all
159), every tree split, every threshold. It can predict exactly when the
classifier would flag the current window and avoid training during those
moments. If the oracle-guided workload still gets detected at >50%, that's a
strong result for the monitoring approach.

---

### DDP Adversarial Strategies

These extend single-GPU strategies to distributed training, adding NVLink/PCIe
communication evasion.

#### Strategy K-DDP: Online Learning, Distributed (`adversarial_ddp.py --strategy K`)

**Mechanism:** Same as single-GPU Strategy K but with DDP:
- All ranks do forward + backward + allreduce + optimizer step on every batch
- SGD with no momentum, ResNet-50, test set
- The allreduce after each backward pass creates periodic NVLink bursts

**What the classifier sees:** Per-GPU telemetry looks like Strategy K (training
disguised as inference) plus periodic PCIe TX/RX spikes from allreduce.

#### Strategy L-DDP: Diluted Training, Synchronized (`adversarial_ddp.py --strategy L`)

**Mechanism:** Same as single-GPU Strategy L but with DDP:
- Every N-th batch: all ranks train together (forward + backward + allreduce)
- Other batches: all ranks do inference (no communication)
- Synchronized: all ranks agree on which batches are training batches

**What it targets:** Same as Strategy L, with the added challenge that the
allreduce communication is an additional detectable signal.

#### Strategy L-stagger: Staggered Diluted Training (`adversarial_ddp.py --strategy L-stagger`)

**Idea:** Spread the allreduce communication across time by having each rank
train on different batches.

**Mechanism:**
1. Rank r trains on batch indices where `batch_idx % (dilution * world_size) == r`
2. Uses `model.no_sync()` to suppress DDP allreduce on non-training batches
3. Gradients are applied locally (no inter-rank synchronization)
4. The allreduce signal is spread across time steps rather than synchronized

**What it targets:** NVLink/PCIe periodicity. A detector looking for correlated
multi-GPU activity sees noise instead of periodic synchronized bursts.

**Tradeoff:** Without synchronized allreduce, model parameters diverge between
ranks. This produces a different kind of training (local SGD) with potentially
worse convergence.

#### Strategy K-accum: Gradient Accumulation (`adversarial_ddp.py --strategy K-accum`)

**Idea:** Reduce the frequency of allreduce by accumulating gradients.

**Mechanism:**
1. Every batch does forward + backward, but uses `model.no_sync()` to suppress
   allreduce
2. Every M-th batch: normal DDP backward (triggers allreduce) + optimizer step
3. M=4: allreduce happens 25% as often as normal DDP
4. M=8: allreduce happens 12.5% as often

**What it targets:** NVLink traffic frequency and volume. Normal DDP has an
allreduce after every batch. K-accum reduces this by a factor of M, making the
communication pattern less distinctive.

---

## 4. Dilution Sweep

Separate from the adversarial strategies, the dilution sweep
(`classifier/eval_dilution_sweep.py`) runs a controlled experiment with a
dedicated set of workloads (`dilution_sweep_inference`, `dilution_sweep_N1`
through `dilution_sweep_N20`). These are ResNet-50 workloads identical to
Strategy L but collected specifically for the dilution analysis, sweeping N
from 1 (every batch trains) through 20 (1 in 20 batches trains) at window
sizes of 5s, 15s, 30s, and 60s.

---

## 5. Feature Extraction and Classification

The classifier (`classifier/threeway_improved.py`) extracts features from
sliding windows of telemetry:

- **13 statistics per signal:** mean, std, min, max, p25, p50, p75, p95, IQR,
  range, CV, skew, kurtosis
- **9 signals:** gpu_util, mem_util, mem_used, power, temp, sm_clock,
  mem_clock, pcie_tx, pcie_rx
- **Cross-signal features:** correlations between key signal pairs
- **Temporal features:** power ACF at lags 1/5/10/20

Total: ~159 features per window.

The classifier is an XGBoost model (500 trees, max_depth=6, lr=0.05) trained
on balanced data (1/3 each: ml_training, ml_inference, other). It classifies
each window into one of these three classes.

---

## 6. Key Results Summary

With the balanced classifier:
- **DDP strategies** (K-DDP, L-DDP, K-accum, L-stagger) are reliably detected
  at 98-100%, likely because the allreduce communication pattern is an
  unforgeable signal.
- **Single-GPU diluted training** (Strategy L at high N, Strategy N) partially
  evades: 4 of 18 families evade at <50% detection.
- **Overall LOO mean detection:** 68.3% (honest balanced evaluation).
- **White-box strategies** (WB-D, WB-L, WB-O) are implemented but awaiting
  data collection.
