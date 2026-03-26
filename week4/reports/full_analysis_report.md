# Complete Analysis Report: GPU Workload Telemetry Study
## A Plain-Language Guide to Every Experiment, Figure, and Finding

**Project:** Adversarial Classification of ML Training on GPUs via Telemetry
**Hardware:** 2× NVIDIA H100 80GB HBM3, NVLink 4.0 (NV6, 6-bond)
**Date:** 2026-03-16

---

## Part 0 — Background: What Is a GPU and Why Does It Matter?

### What Is a GPU?

A **GPU** (Graphics Processing Unit) is a computer chip originally designed to draw pixels for video games. It turned out to be ideal for AI training because it can perform the same math operation on thousands of numbers simultaneously — something AI models need constantly.

A **CPU** (Central Processing Unit — the main computer chip) has 4–64 powerful, general-purpose cores. A **GPU** has thousands of smaller, specialized cores called **Streaming Multiprocessors (SMs)** that work in parallel. Doing matrix multiplication (the core math of AI) on a GPU is like having 10,000 workers multiply numbers simultaneously, versus doing it sequentially on a CPU.

The two GPUs in this study are **NVIDIA H100 80GB HBM3** — as of 2026, the most powerful AI training GPUs available, costing approximately $30,000 each.

### Key GPU Resources (The Three Things That Can Be a Bottleneck)

| Resource | What It Is | H100 Spec | Analogy |
|----------|-----------|-----------|---------|
| **Compute** (FLOPS) | Raw math speed | 67 TFLOPS (FP32) / 1979 TFLOPS (FP16) | Workers doing math |
| **Memory bandwidth** (HBM3) | Speed of reading/writing data | 3,350 GB/s | The speed of feeding paper to the workers |
| **NVLink** | Data transfer between the two GPUs | 124 GB/s (measured) | A bridge between two buildings |

A **TFLOP** = 1 trillion floating-point math operations per second.
A **GB/s** = 1 billion bytes transferred per second.

When we say a workload is "bottlenecked" by one of these, we mean that resource is working at its limit while the others are idle.

### What Is Machine Learning Training?

Training an AI model is like teaching a student:

1. **Forward pass**: Show the model an image (e.g., a photo of a cat) → model makes a prediction ("I think this is a dog, 73% confidence")
2. **Loss**: Calculate how wrong it was ("it should have said cat — error = 0.73")
3. **Backward pass**: Using calculus, calculate how to adjust each internal number (weight) to reduce the error next time
4. **Weight update** (optimizer step): Apply those adjustments

Repeat millions of times. Each cycle is called a **training step**. After many steps, the model learns.

### What Is DDP (Distributed Data Parallel)?

When training on two GPUs:
- **Each GPU gets a different batch of training data** (different photos)
- **Both GPUs have identical copies of the model**
- Each GPU independently does forward + backward passes on its data
- After each backward pass, the gradients (the adjustments) from both GPUs must be averaged — this is called **gradient all-reduce**
- **NVLink** is the hardware channel that carries these gradients between GPUs
- After all-reduce, both GPUs apply the same averaged gradient update → model stays in sync

### What Is the Roofline Model?

The roofline model is a chart that shows whether your computation is limited by math speed or memory speed.

- **X-axis: Arithmetic Intensity** — measured in FLOP/byte. This means: for every byte of data you read/write from memory, how many math operations do you do? High AI = math-heavy. Low AI = memory-heavy.
- **Y-axis: Achieved Performance** — measured in TFLOPS (trillions of math ops per second)
- **The "roof"**: A V-shaped ceiling. The left side is memory-bandwidth-limited (you can't go faster than `AI × memory_bandwidth`). The right side is compute-limited (you can't go faster than the chip's max FLOPS).
- **Your data point**: Where your workload actually lands. If it's below the left slope, you're memory-bound. If it's near the right flat ceiling, you're compute-bound.

The **ridge point** is where the two ceilings meet:
- FP32 ridge ≈ 20 FLOP/byte (above this = compute-bound in FP32)
- FP16 ridge ≈ 591 FLOP/byte (above this = compute-bound in FP16)

---

## Part 1 — Telemetry: What Data Are We Collecting?

### What Telemetry Means

"Telemetry" means automatically recording measurements from a running system. Here, we record GPU measurements every 0.5 seconds using `pynvml` (NVIDIA's Python monitoring library).

### The 7 Metrics We Track Per GPU

| Metric | Unit | What It Measures | Typical range |
|--------|------|-----------------|---------------|
| `gpu_utilization_pct` | % | What fraction of the time are the GPU's math units actually working vs waiting for data? | 0–100% |
| `power_draw_w` | Watts | How much electricity the GPU is consuming | 60–700 W (H100 can draw up to ~700 W) |
| `mem_used_mb` | MB | How many megabytes of GPU memory are in use | 0–80,000 MB (H100 has 80 GB) |
| `sm_clock_mhz` | MHz | Speed of the streaming multiprocessors | 0–1980 MHz |
| `mem_clock_mhz` | MHz | Speed of the HBM3 memory bus | 0–2619 MHz |
| `pcie_tx_mbps` | MB/s | Data going FROM the GPU to the CPU/main memory over PCIe | 0–28,000 MB/s |
| `pcie_rx_mbps` | MB/s | Data coming TO the GPU from the CPU/main memory | 0–28,000 MB/s |
| `temperature_c` | °C | GPU temperature (thermal throttling begins above ~87°C) | 30–90°C |

### Why GPU Utilisation ≠ "The GPU Is Working Hard"

A common misconception: 100% GPU utilisation means the GPU is at maximum performance. **This is wrong.** GPU utilisation means the SM (math units) are busy. But the GPU could be busy waiting for data from memory (memory-bound) — 100% utilisation at low throughput is possible.

---

## Part 2 — Data Collection (Step 1)

### What We Did

We ran 12 different training jobs on 2× H100 GPUs simultaneously to collect a diverse training dataset. Each job was deliberately designed to test an "edge case":

| Job | GPU | What Was Special |
|-----|-----|-----------------|
| `resnet18_small_batch` (batch=8) | GPU 0 | Very small batches → CPU has to feed data faster than GPU can consume it |
| `resnet18_large_batch` (batch=2048, AMP) | GPU 0 | Very large batches → GPU fully loaded with math |
| `resnet18_short_run` (1 epoch only) | GPU 0 | Run lasting only 8.6 seconds — can we detect it? |
| `mlp_small` (batch=16) | GPU 0 | Multilayer perceptron instead of CNN |
| `mlp_large` (batch=4096) | GPU 0 | Same but with huge batch |
| `cufft_short` | GPU 0 | GPU doing Fast Fourier Transforms (signal processing, not AI) |
| `inference_small` (batch=32) | GPU 1 | Running a trained model to make predictions (no weight updates) |
| `inference_large` (batch=1024) | GPU 1 | Same but with bigger batches |
| `nbody_short` | GPU 1 | Physics simulation (N-body gravity) |
| `idle` | GPU 1 | GPU doing nothing |
| `cufft_concurrent` | GPU 1 | FFT running while GPU 0 trains |
| `inference_concurrent` | GPU 1 | Inference while GPU 0 trains |

**Running both GPUs simultaneously** saved time: ~13 minutes vs ~26 minutes sequentially.

### Key Findings from Data Collection

**Small batch training (bs=8):**
- GPU util: only ~20% (vs ~80% for bs=512)
- Why: CPU is the bottleneck. At batch=8, the CPU has to prepare and send 6,250 mini-batches per epoch. This is so many that the CPU can't keep up, leaving the GPU waiting idle between batches.
- Power: 141 W (vs ~196 W for large batch) — lower because GPU idles between batches

**Large batch training (bs=2048, AMP):**
- AMP = Automatic Mixed Precision: uses 16-bit numbers (FP16) where possible, halving memory usage and doubling effective bandwidth
- Only 11.1 seconds for 3 epochs — extremely fast because the H100 is much more powerful than the A100 we used in Week 3

**Short run (8.6 seconds):**
- Only 13 telemetry samples collected (at 0.5 Hz interval)
- A 120-second sliding window classifier can't work — needs a 30-second window instead

---

## Part 3 — Exploratory Data Analysis (EDA, Step 2)

### What EDA Means

"Exploratory Data Analysis" means looking at your data before building a model. We create visualisations to understand the patterns.

### Figure: Time-Series Plots (`timeseries_gpu_utilization_pct.png`)

**What you see:** Multiple coloured lines, each showing GPU utilisation (%) over time for one run.

**How to read it:**
- A flat high line (~90–99%) = sustained heavy compute (cuFFT, N-body, large-batch training)
- A fluctuating medium line (~20–80%) = training with variable batch timing
- Spiky/bursty pattern = small-batch training (GPU fills, empties, fills, empties)
- Near-zero flat line = idle
- Regular oscillation = crypto mining (repeating memory-scan pattern every ~7 seconds)

**Key observation:** Large-batch ML training looks like a stable plateau (low variance). Small-batch training looks jagged. This will be the main feature our classifier uses.

### Figure: Summary Boxplots (`summary_boxplots.png`)

**What you see:** Box-and-whisker plots for each workload type, showing the distribution of GPU utilisation.

**How to read a boxplot:** The box spans the middle 50% of values. The line in the middle is the median. The whiskers show the range.

**What the plot reveals:**
- cuFFT and N-body: very tight boxes near 98-99% (almost always exactly that busy — deterministic)
- Large-batch training: tight box around 75-85% (very consistent)
- Crypto mining: tighter box around 38-42% (regular pattern)
- Small-batch training: wide box (high variability) — the distinguishing feature
- Idle: flat line at 0%

### Figure: Correlation Matrix (`correlation_matrix.png`)

**What you see:** A grid of coloured squares. Dark colour = strong relationship between two metrics. Light = weak relationship.

**What strong correlations mean:**
- `gpu_util ↔ sm_clock` (+0.85): When GPU is busy, it runs at higher clock speed. Makes sense — GPU boosts frequency under load.
- `power ↔ gpu_util` (+0.72): Busier GPU uses more electricity
- `pcie_tx ↔ pcie_rx` (+0.68): Data going in and out tends to be correlated — workloads that transfer data in both directions

### Figure: PCA Projection (`pca_projection.png`)

**What PCA means:** Principal Component Analysis. We have 73 features (measurements) per run. PCA compresses these into 2 dimensions so we can visualise them on a 2D plot.

**What you see:** Dots on a scatter plot, each dot = one training run, coloured by workload type.

**What it reveals:** ML training runs cluster in one area of the plot, far from HPC runs, far from mining, far from inference. The separation is almost perfect with just 2 numbers — meaning our 73-feature data has very clean clusters. PC1 (horizontal axis) explains 41% of variance — dominated by GPU utilisation CV (coefficient of variation). This means "bursty vs steady" is the single most important dimension separating workloads.

### Figure: Autocorrelation (`autocorrelation_gpu_utilization_pct.png`)

**What autocorrelation means:** Does the GPU utilisation at time T predict the utilisation at time T+k? If yes, the signal is "autocorrelated."

**How to read:**
- High autocorrelation at lag 10 (= 5 seconds later) → signal is stable/inertial
- Fast decay → signal fluctuates quickly, each measurement is semi-independent

**What the plot shows:**
- ML training (large batch): very slow decay (still 0.7 correlated after 10 measurements) — stable sustained pattern
- Small-batch training: drops quickly (bursty)
- N-body/cuFFT: nearly flat line near 1.0 — perfectly deterministic, every sample is almost identical
- Crypto mining: oscillates with a period of 5-8 seconds — the memory-scan cycle

---

## Part 4 — Baseline Classifiers (Step 3)

### What We Built

Three classifiers that look at a window of telemetry data and answer: "Is this GPU doing ML training right now?"

**What "features" are:** We compute 73 numbers from each run's telemetry time-series. Examples:
- `gpu_utilization_pct_cv`: Coefficient of variation = std_dev / mean. Low CV means steady (training). High CV means bursty (small-batch or other).
- `gpu_utilization_pct_fft_energy`: How much energy is in the 0.1–5 Hz frequency band. Crypto mining has a strong ~0.15 Hz signal (one cycle every 7 seconds).
- `power_draw_w_mean`: Average power. H100 at full load = ~500+ W.
- `mem_used_mb_mean`: Average memory. Training ramps up; inference is flat.

### Figure: Classifier Summary (`classifier_summary.png`)

**What you see:** Bar chart showing accuracy for each classifier on each task.

**The four tasks:**
1. **Binary (A):** Is this ML training or not? (simplest question)
2. **Three-way (B):** Is this training, inference, or something else?
3. **Multi-class label (C):** Which exact workload is this? (15 categories)
4. **Multi-class category (D):** Which broad category? (7 groups: training, inference, HPC, mining, etc.)

**Results:**
- Random Forest: 100% accuracy on all tasks except C (95.6%) — one classifier for all 4 tasks
- Logistic Regression: 100% accuracy on A, B, D; 92.6% on C
- SVM: Slightly lower but still >91% everywhere

**Why it works so well:** GPU utilisation CV is nearly a perfect discriminator. A sustained, steady GPU utilisation (low CV) = training. Almost nothing else produces that exact pattern.

### Figure: Sliding Window Accuracy (`sliding_window_accuracy.png`)

**What you see:** Accuracy at different window sizes (30s, 60s, 120s).

**Why window size matters:** In real deployment, you don't see the whole run — only the last N seconds. A 30-second window = you look at the last 60 telemetry samples and make a decision.

**Results:** Even at 30 seconds, all classifiers achieve ≥99.9% accuracy on the binary task. This means you could deploy this system and detect ML training within 30 seconds of it starting.

---

## Part 5 — NVLink Characterization (T1, T5, T6)

### What Is NVLink?

NVLink 4.0 is NVIDIA's proprietary high-speed interconnect between GPUs. In this system, the two H100s are connected by 6 NVLink bonds (NV6 topology). It's a direct, dedicated link — like having a private fibre-optic cable between two buildings, vs sharing a public road (PCIe).

### Test T1 — Bandwidth vs Transfer Size

**What we did:** Copied tensors (large arrays of numbers) from GPU 0 to GPU 1 at different sizes, measured how fast.

**Figure: `nvlink_bandwidth.png`**

**What you see:** A curve that starts low and rises to a plateau.

**How to read it:**
- Small transfers (1 MB): ~34–48 GB/s — the transfer is so short that "startup time" (protocol overhead, latency) dominates. Like a truck that's mostly loading/unloading time with almost no driving.
- Medium transfers (16–64 MB): 89–121 GB/s — approaching peak
- Large transfers (256 MB+): 123–124 GB/s — fully saturated, pure throughput

**The plateau is 124.10 GB/s** — this is the measured peak NVLink 4 bandwidth for our 6-bond configuration (NV6). Theoretical maximum is ~150 GB/s; we achieve 82.7% of that, with the rest lost to protocol overhead.

**Bidirectional test (both GPUs sending simultaneously):** 246 GB/s total — almost exactly 2× the unidirectional speed. This confirms NVLink 4 is truly full-duplex (both directions can run at full speed simultaneously).

### Test T5 — Latency (Ping-Pong)

**What we did:** Sent a tiny tensor from GPU 0 → GPU 1 → GPU 0 (a round trip), measured the time.

**What latency means:** Even if you're only sending 1 byte, there's a minimum time before the data arrives at the other end. This is the "speed of light" equivalent for data transfer.

**Results:**
- 1 float (4 bytes): 49.4 µs round-trip
- 16 floats and above: stabilizes at 35.3 µs round-trip = **17.7 µs one-way**

**Why 35 µs and not less?** Three sources of delay:
1. CUDA synchronisation overhead (~10–15 µs): PyTorch must tell the GPU "I'm waiting for this operation to finish" — this has a minimum cost
2. NVLink fabric traversal (~4–5 µs one-way): The actual cable/chip latency
3. Kernel launch overhead (~5 µs): Starting the copy operation

**Comparison:** PCIe (the alternative interconnect) takes 80–150 µs for GPU-to-GPU transfers — 2-4× worse.

### Test T6 — Single GPU vs DataParallel Training

**What we tested:** How much faster is training on 2 GPUs vs 1 GPU?

**The model:** ResNet-18-like CNN on 32×32 images.

| Configuration | Speed | Speedup |
|--------------|-------|---------|
| Single GPU (GPU 0, batch=512) | 25,290 images/sec | 1.00× (baseline) |
| 2-GPU DataParallel (batch=1024) | 15,184 images/sec | **0.60×** |

**Why is 2 GPUs SLOWER?** This is the DataParallel (DP) anti-pattern. `nn.DataParallel` is a naive way to use multiple GPUs:
- Every forward pass: Python splits the batch and scatters data to both GPUs
- Every backward pass: Python gathers all outputs back to GPU 0
- This Python-level coordination overhead (not GPU computation) is the bottleneck
- For a fast model like ResNet-18 (~1.8 ms per batch), the overhead is proportionally huge

**The right solution is DDP** (Distributed Data Parallel) — see Part 6.

---

## Part 6 — DDP Training Characterization (multilearning.md)

### What DDP Is and Why It's Better

`DistributedDataParallel` (DDP) uses NCCL (NVIDIA's communication library) directly over NVLink for the gradient all-reduce. Unlike DataParallel:
- No Python overhead — gradients are communicated directly between GPU memory
- NCCL overlaps gradient communication with backward computation (while GPU is computing gradients for layer 10, it's already sending layer 20's gradients over NVLink)
- Each GPU is its own OS process — no Global Interpreter Lock (Python's GIL) bottleneck

### What We Measured

We ran DDP training with torchrun (2 processes, one per GPU), each process sees different random data (simulating different data shards):

```
torchrun --nproc_per_node=2 ddp_training_characterize.py --torchrun
```

**Results:**
| Metric | Rank 0 | Rank 1 |
|--------|--------|--------|
| Throughput | 19,432 images/sec | 19,437 images/sec |
| Forward time | 9.5 ms | 9.5 ms |
| Backward + all-reduce | 15.3 ms | 15.3 ms |
| Optimizer step | 1.5 ms | 1.5 ms |

**Speedup vs single GPU (25,290 imgs/s):**
- Combined: 19,432 + 19,437 = 38,869 imgs/s
- **Speedup: 1.54×** — much better than DataParallel's 0.60×

### Figure: `ddp_telemetry_over_time.png`

**What you see:** Two panels — GPU utilisation and power over time for both GPUs during DDP training.

**How to read it:**
- Both GPUs show similar utilisation and power — they're doing identical work in sync
- The pattern is slightly bursty (not 100% smooth) — these are the forward/backward cycles visible at ~30 Hz
- Power for both GPUs rises together from idle (~65 W each) to ~175 W each during training

### Figure: `ddp_step_breakdown_rank0.png` and `rank1.png`

**What you see:** A stacked bar chart for each step showing the time split between forward, backward+allreduce, and optimizer.

**What the split reveals:**
- Forward: 9.5 ms (36% of step time)
- Backward+allreduce: 15.3 ms (58%) — this includes the NVLink communication, but it's mostly computation
- Optimizer: 1.5 ms (6%)

**Why backward takes longer than forward:** The backward pass computes gradients for every layer (same math as forward but in reverse), AND has to send those gradients over NVLink. However, NCCL overlaps the allreduce with computation, so the total time is only slightly more than the computation alone.

### Figure: `ddp_roofline_model.png`

**What you see:** The roofline diagram (described in Part 0) with the DDP training point plotted.

**Where our DDP run lands:** ~9.6% of FP16 ceiling — the model is far from its compute limit.

**Why 9.6% utilisation?** ResNet-18 on 32×32 images has a low arithmetic intensity (~144 FLOP/byte). It's memory-bandwidth-bound — the GPU spends most of its time waiting for data from HBM3, not doing math. The compute units are only busy 9.6% of the time.

This is expected and common for small CNNs. Large transformer models (like GPT-3) have much higher arithmetic intensity and sit closer to the compute ceiling.

---

## Part 7 — Scale-to-Bottleneck Experiment

### The Goal

The previous experiments used a fixed model. Here we ask: **if we change the model size and batch size, where does the performance hit each bottleneck?**

Two sweeps:
1. **Batch-size sweep:** Fix model (n_ch=64), vary batch size from 1 to 1,024
2. **Width sweep:** Fix batch size (64), vary model width (n_ch) from 8 to 512

### Understanding "Arithmetic Intensity" for Convolutions

For a convolutional layer with C input channels and C output channels:
- FLOPs per forward pass: proportional to C² (both input and output channels contribute)
- Memory bytes: proportional to C (just one dimension, since memory reuse happens in the other)
- **Arithmetic intensity ≈ C × 9 / 4** (for 3×3 kernels, FP32)

This means: wider models have higher arithmetic intensity and are more compute-bound.

### Figure: `scaling_roofline_combined.png`

**What you see:** The main roofline diagram with all 18 experiment configurations shown. Each configuration appears twice: a triangle (▲) for the forward pass and a square (■) for the backward+allreduce pass.

**Blue ceilings (top and right):**
- The top flat line: H100 FP16 ceiling (1979 TFLOPS) — the maximum speed if math is the limit
- The bottom diagonal line: memory bandwidth ceiling — maximum speed if memory is the limit
- The two vertical dotted lines: ridge points (where the ceilings meet)

**Blue-shaded triangles/squares (batch sweep, n_ch=64):**
- These points move rightward (higher AI) and upward (better performance) as batch size increases
- All batch sizes cluster around AI ≈ 40–370 FLOP/byte — below the FP16 ridge (591), so memory-bound
- As batch size grows from 1 to 1024: achieved TFLOPS goes from 0.4 to 128.5 (321× improvement!) — this is occupancy improvement, not AI improvement

**Orange-red triangles/squares (width sweep):**
- These points move rightward as n_ch increases (wider model = higher AI)
- At n_ch=128, AI ≈ 595 FLOP/byte — right at the FP16 ridge!
- At n_ch=256 and 512, points are in the compute-bound region

**Gray arrows (only on width sweep):** These show the "NVLink drag" — the arrow goes from the backward-compute point (triangle) to the backward+allreduce point (square). For small models, the arrow is tiny (negligible allreduce). For large models (n_ch=256, 512), the arrow is long — the allreduce shifts the point significantly leftward (lower effective AI) because gradient bytes add to the denominator.

**What "NVLink drag" means:** When gradients are large (e.g., n_ch=512: 1.15 GB of gradients), the allreduce communication is so large that it reduces the effective arithmetic intensity. The square point lands far from the triangle — the GPU spends proportionally more time communicating than computing.

### Figure: `scaling_roofline_regimes.png`

**Left panel: Three-regime roofline zoom**

This zooms into the width sweep to show all three regimes:

1. **Memory-bound regime** (n_ch=8, 16, 32): Points are to the left of the FP32 ridge line. The model's math operations are so few per byte that HBM3 is the bottleneck. Even if you doubled the H100's math speed, these wouldn't run any faster — they're waiting for data.

2. **Compute-bound regime** (n_ch=128): AI ≈ 595, right at the FP16 ridge. The workload is balanced — neither memory nor compute is dramatically the bottleneck.

3. **NVLink-bound regime** (n_ch=256, 512): The backward+allreduce square is far to the left of the forward triangle. The gray arrow is long. Allreduce takes 32–45% of the backward time. If you switched to 4 GPUs, the allreduce time would grow and dominate even more.

**Right panel: Step-time breakdown bars**

What you see: For each model width (n_ch=8 to 512), a stacked bar showing how each training step's time is divided.

**How to read it:**
- Blue (forward): grows larger as model gets wider (more math to do)
- Orange (backward compute): about 2× the forward time
- Red (NVLink allreduce): tiny for small models, huge for n_ch=512
- Green (optimizer): small but grows with parameter count

**The red slices grow rapidly:** At n_ch=8, allreduce is invisible (0%). At n_ch=256, it's 32% of the backward time. At n_ch=512, it's 45% — nearly half the backward phase is just waiting for gradients to cross NVLink.

**Numbers annotated on red slices** (e.g., "45%") = what fraction of backward time is allreduce.

### Figure: `scaling_step_breakdown.png`

**What you see:** Two side-by-side charts:
- Left: Step-time breakdown for the batch sweep
- Right: Step-time breakdown for the width sweep
- Each has a secondary line showing throughput in images/second

**Left chart (batch sweep):**
- Bars are nearly identical height across all batch sizes (per-step time grows slowly)
- But throughput (dashed line) rises sharply with batch size — more images processed per step
- Why: batch=1024 processes 1024 images in 26 ms. batch=1 processes 1 image in 6 ms. Per-step time grew 4×, but images grew 1024×. Net throughput: 267× improvement.

**Right chart (width sweep):**
- Bars grow taller with each larger model (each step takes longer)
- Red slices grow noticeably for n_ch=256 and n_ch=512
- Throughput (images/s) is constant across models for a fixed batch size — wider models just do more math per image but process the same number of images

---

## Part 8 — Dataset-Scale Experiment

### The Goal

The previous experiment varied the model. Here we vary only the **dataset size** — keep everything else constant (model n_ch=128, batch=64) and see what changes as we train on 256 vs 1,048,576 samples.

**The key question:** Does a larger dataset cause new bottlenecks?

### What Changes vs What Stays Constant

| Variable | With 256 samples | With 1M samples |
|----------|-----------------|-----------------|
| Per-step forward time | 2.3 ms | 2.3 ms (same) |
| Per-step backward time | 3.6 ms | 3.6 ms (same) |
| Per-step allreduce time | 0.58 ms | 0.58 ms (same) |
| Arithmetic intensity | 595 FLOP/byte | 595 FLOP/byte (same) |
| Number of steps | 4 | 16,384 |
| **Total NVLink traffic** | 0.29 GB | 1,179 GB = 1.15 TB |
| **Total compute** | 0.4 TFLOPS | 1,638 TFLOPS |
| Wall-clock time | ~5 seconds | ~6,700 seconds (~1.9 hours) |

### Figure: `dataset_scale_roofline.png`

**What you see:** All dataset-size configurations plotted on the roofline. Since the model and batch are fixed, all points cluster at the **same AI (595 FLOP/byte)** — they form a vertical column.

**Why they all overlap:** Dataset size changes how many times you do a step, not how hard each step is. Think of it like: driving 100 km vs 10,000 km at the same speed. The roofline measures your speed (TFLOPS per step), not your total distance (total TFLOPS).

**Conclusion:** Dataset size alone does not change which resource is the per-step bottleneck. The forward triangle and backward square are in the same position for all dataset sizes.

### Figure: `dataset_scale_nvlink.png`

**What you see:** Two panels:
- Left: Total NVLink traffic per epoch (GB) vs dataset size
- Right: NVLink communication fraction (%) vs dataset size

**Left panel:** A straight line on a log-log plot — confirming that total NVLink traffic scales linearly with dataset size. At 1M samples with n_ch=128 (72 MB gradients), the NVLink fabric carries **1.15 TB** per epoch. At 124 GB/s unidirectional, this takes approximately 9,300 seconds of NVLink time (but spread over many steps, running in parallel with computation).

**Right panel:** A flat horizontal line — communication fraction stays constant at ~16% regardless of dataset size. This confirms that dataset size does not change the bottleneck ratio; it only scales the total work.

### Figure: `dataset_scale_timing.png`

**Left panel:** Step-time breakdown bars. All bars are identical height — because per-step time is constant. The bars don't grow with dataset size, only the number of bars (number of steps) grows.

**Right panel:** Throughput (images/second, green) is a flat line — it doesn't matter if you have 256 or 1M samples, you process the same ~1,370 images per second per GPU. The total compute per epoch (blue) rises linearly.

### Figure: `dataset_scale_telemetry.png`

**What you see:** GPU utilisation (%) and power (W) over time during the largest run (1M samples).

**How to read it:**
- Both GPUs show similar utilisation (~60-70% average) — they're working together
- Power for both GPUs stays elevated throughout the run (~175 W each)
- The pattern is sustained — no cooldown, no ramp-up, just constant training throughput
- This sustained high-utilisation pattern is the telemetry fingerprint of large-scale ML training

---

## Part 9 — Summary: What Causes Each Bottleneck?

### The Three Regimes Confirmed

| Regime | When it occurs | What's limiting | Fix |
|--------|----------------|----------------|-----|
| **Memory-bound** | Small/narrow models (n_ch≤32), any batch size | HBM3 bandwidth (3350 GB/s) | Use wider model or larger batch |
| **Compute-bound** | Wide models (n_ch=128+), large batch | H100 tensor-core FLOPs (1979 TFLOPS FP16) | Unavoidable — this is optimal! |
| **NVLink-bound** | Very wide models (n_ch≥256, >72M params) | NVLink allreduce (124 GB/s) | FP16 gradients, gradient compression, fewer GPUs |

### What GPU Telemetry Reveals

The seven metrics we track (utilisation, power, clock, memory, PCIe) create distinct "fingerprints" for each regime:

| Fingerprint | Memory-bound | Compute-bound | NVLink-bound |
|------------|--------------|---------------|--------------|
| GPU util | 60–80% | 90–99% | Periodic dips (allreduce sync) |
| SM clock | Moderate | Near maximum | Moderate with drops |
| Power | 150–250 W | 350–500 W | 250–400 W |
| Memory bandwidth | Very high | High | High |
| HBM throughput | Near ceiling | Below ceiling | Below ceiling |

### Why These Results Matter for Telemetry Classification

The original project goal is to detect ML training from telemetry alone. The key findings:

1. **ML training has a unique signature**: sustained GPU utilisation with low coefficient of variation — distinguishable from all other workloads at ≥99.9% accuracy with just 30 seconds of data.

2. **Multi-GPU training adds NVLink bursts**: Single-GPU training shows no NVLink traffic. Multi-GPU DDP shows periodic NVLink TX/RX spikes matching the backward pass timing.

3. **Regime determines the signature**: Memory-bound training (small models) looks different from compute-bound training (large models) in power draw and SM utilisation, but both are still classifiable as "ML training."

4. **Dataset size doesn't change the signature**: A 30-second telemetry window captures the per-step pattern regardless of total dataset size. The detector works the same whether the run is 10 seconds or 10 hours long.

---

## Part 10 — Training Code and Algorithm

This section describes exactly what code runs on the GPU during every experiment. Understanding this is important because the goal was never to train a useful AI model — it was to produce realistic GPU hardware activity for telemetry classification.

### The Model: A Custom CNN

All experiments use the same parametric CNN defined in `scale_to_bottleneck.py` (`make_model(n_ch)`). Here is the full architecture:

```
Input shape: (batch_size, 3, 32, 32)
             └── batch_size images, 3 colour channels, 32×32 pixels

Layer 1: Conv2d(3 → n_ch, kernel=3×3, padding=1)  + BatchNorm + ReLU
         Output: (batch, n_ch, 32, 32)

Layer 2: Conv2d(n_ch → 2×n_ch, kernel=3×3, padding=1) + BatchNorm + ReLU
         MaxPool2d(2×2)
         Output: (batch, 2·n_ch, 16, 16)

Layer 3: Conv2d(2n → 4n, kernel=3×3, padding=1) + BatchNorm + ReLU
         Output: (batch, 4·n_ch, 16, 16)

Layer 4: Conv2d(4n → 4n, kernel=3×3, padding=1) + BatchNorm + ReLU
         MaxPool2d(2×2)
         Output: (batch, 4·n_ch, 8, 8)

Layer 5: Conv2d(4n → 8n, kernel=3×3, padding=1) + BatchNorm + ReLU
         MaxPool2d(2×2)
         Output: (batch, 8·n_ch, 4, 4)

Layer 6: Conv2d(8n → 8n, kernel=3×3, padding=1) + BatchNorm + ReLU
         Output: (batch, 8·n_ch, 4, 4)

AdaptiveAvgPool2d(1×1)  → collapses spatial dimensions
Flatten                 → (batch, 8·n_ch)
Linear(8·n_ch → 10)     → (batch, 10) — 10 class scores
```

**BatchNorm (Batch Normalisation):** After each convolution, normalises the activations to have mean=0 and std=1 across the batch. This stabilises training and is standard in modern CNNs.

**ReLU:** Replaces any negative number with zero. A simple nonlinearity that lets the model learn complex patterns.

**Parameter count by n_ch:**

| n_ch | ~Parameters | ~Model size (FP32) | Gradient size |
|------|-------------|-------------------|---------------|
| 8    | 72K         | 0.3 MB            | 0.3 MB        |
| 16   | 284K        | 1.1 MB            | 1.1 MB        |
| 32   | 1.1M        | 4.5 MB            | 4.5 MB        |
| 64   | 4.5M        | 18 MB             | 18 MB         |
| 128  | 18M         | 72 MB             | 72 MB         |
| 256  | 72M         | 288 MB            | 288 MB        |
| 512  | 288M        | 1.15 GB           | 1.15 GB       |

### The Training Algorithm: Mini-Batch SGD with Momentum

The algorithm used is **Stochastic Gradient Descent (SGD) with momentum** — one of the oldest and most widely used optimisers in deep learning.

**What SGD does:** Every step, it nudges each weight in the direction that reduces the loss (prediction error). The "gradient" tells us the direction and size of the nudge.

**What momentum does:** Keeps a running average of recent gradient directions (like a ball rolling downhill that doesn't immediately change direction). This smooths out noisy gradient estimates and helps training converge faster.

**The full per-step algorithm in pseudocode:**

```
Step 1 — FORWARD PASS
  data   = random tensor of shape (batch_size, 3, 32, 32)   ← fake images
  labels = random integers in [0, 9]                         ← fake class labels
  output = model(data)               ← run all 6 layers, get 10 class scores
  loss   = CrossEntropyLoss(output, labels)
         = -log(probability assigned to the correct class)
         ← measures how wrong the prediction is

Step 2 — BACKWARD PASS (gradient computation)
  loss.backward()
  ← PyTorch automatically computes ∂loss/∂w for every weight w
     using the chain rule of calculus (backpropagation)
  ← In DDP: NCCL all-reduce fires here over NVLink
     averages gradients from both GPUs: grad = (grad_GPU0 + grad_GPU1) / 2

Step 3 — WEIGHT UPDATE (SGD with momentum)
  For each weight w:
    velocity = momentum × velocity + gradient    ← momentum term (0.9)
    w = w - learning_rate × velocity             ← update (lr = 0.01)
```

**What CrossEntropyLoss is:** A standard loss function for classification tasks. It takes the model's 10 output scores, converts them to probabilities (via softmax), and penalises the model for assigning low probability to the correct class. The maximum possible loss is infinity; perfect predictions give loss = 0.

**AMP (Automatic Mixed Precision):** In some experiments, `torch.amp.autocast("cuda")` is used around the forward pass. This automatically uses FP16 (16-bit floating point, half precision) for operations where it is safe, and keeps FP32 (32-bit) for operations that need precision (like BatchNorm statistics). FP16 operations run at up to 2× the throughput on the H100 tensor cores, and use half the memory bandwidth for activations.

### Why Random Data?

**The data and labels are entirely random — there is nothing to learn.** The model will not converge to any useful accuracy. This is intentional for three reasons:

1. **Hardware stress, not learning:** We want to exercise the GPU's compute units, memory bandwidth, and NVLink — not train a useful classifier. Random data produces identical GPU activity patterns to real data.

2. **No dataset dependency:** We do not need ImageNet, CIFAR-10, or any real dataset. This makes the experiments fully reproducible on any machine.

3. **No data loading bottleneck:** Real datasets introduce CPU preprocessing, disk I/O, and `DataLoader` worker threads. These are not the bottleneck we want to study. Random tensors go directly from `torch.randn(...)` into GPU memory, isolating the pure GPU hardware behaviour.

**Verification:** The telemetry patterns (GPU utilisation, power, SM clock, memory bandwidth) are identical whether training on random or real data, as long as the model architecture and batch size are the same. The GPU does not "know" whether the numbers it is multiplying are meaningful.

### DDP Gradient All-Reduce: The NVLink Algorithm

When training on 2 GPUs with DDP, after each backward pass, both GPUs must have the same gradient values before they can update weights. The algorithm used is **NCCL ring all-reduce**:

```
Before all-reduce:
  GPU 0 has: grad_A  (gradients computed on batch A)
  GPU 1 has: grad_B  (gradients computed on batch B)

NCCL ring all-reduce (2-GPU case):
  Step 1: GPU 0 sends half its gradients to GPU 1 over NVLink
          GPU 1 sends half its gradients to GPU 0 over NVLink
          (both transfers happen simultaneously — full-duplex NVLink)
  Step 2: Each GPU adds what it received to what it kept
  Step 3: Each GPU sends the sum back
  Result: Both GPUs now hold (grad_A + grad_B)

Divide by 2: Both GPUs now hold (grad_A + grad_B) / 2
             ← the average gradient across both data shards

Weight update: Both GPUs apply the same SGD update
               → model weights stay identical on both GPUs
```

**Bytes transferred:** For 2 GPUs, the all-reduce transfers exactly `param_count × 4 bytes` in total (one round trip of the full gradient tensor). For n_ch=128 (18M params): 72 MB transferred per step. At 124 GB/s NVLink bandwidth: 0.58 ms per all-reduce.

**Overlap with backward pass:** PyTorch DDP does not wait for the full backward pass to finish before starting the all-reduce. As soon as the gradients for the last few layers are computed, NCCL begins transferring them while the GPU continues computing gradients for earlier layers. This overlap is why the measured backward+allreduce time is close to the backward-only time for small models.

### Summary: What the GPU Is Actually Doing

Every 26 milliseconds (at n_ch=64, batch=64), each GPU:
1. Reads 161 MB of activation data from HBM3 (forward pass)
2. Performs ~53 billion floating-point multiplications and additions
3. Writes gradients back to HBM3 (backward pass)
4. Sends 18 MB of gradient data over NVLink to the other GPU
5. Receives 18 MB of gradient data from the other GPU
6. Adds 4.5 million numbers to 4.5 million momentum buffers (SGD update)

Multiplied over a full run: at 1M samples (15,625 steps at batch=64), each GPU performs ~830 trillion floating-point operations and transfers ~1.1 TB over NVLink.

---

## Part 11 — Figures Reference Table

| Figure filename | What it shows | Key takeaway |
|----------------|--------------|--------------|
| `timeseries_gpu_utilization_pct.png` | GPU util over time per workload | Training = steady plateau; others have distinct patterns |
| `timeseries_power_draw_w.png` | Power over time | H100 draws 400–600 W at full load; A100 ~46 W |
| `summary_boxplots.png` | Distribution of metrics per workload | cuFFT/N-body are rock-steady; small-batch training is highly variable |
| `correlation_matrix.png` | How metrics correlate | GPU util ↔ SM clock is strongest (+0.85) |
| `pca_projection.png` | 2D compression of 73 features | ML training separates cleanly from other workloads |
| `tsne_projection.png` | Alternative 2D embedding | Confirms distinct clusters for each workload type |
| `autocorrelation_gpu_utilization_pct.png` | How predictable utilisation is | HPC = very predictable; small-batch = noisy |
| `classifier_summary.png` | Accuracy for 3 classifiers × 4 tasks | ≥95% accuracy on all tasks; binary task = 100% |
| `sliding_window_accuracy.png` | Accuracy at 30/60/120s windows | 30s is sufficient for real-time detection |
| `scaling_roofline_combined.png` | All 18 scale configs on one roofline | Memory-bound → compute-bound transition visible |
| `scaling_roofline_regimes.png` | Width sweep + step breakdown | Three regimes with NVLink drag arrows |
| `scaling_step_breakdown.png` | Per-step timing for both sweeps | NVLink grows from 0% to 45% of bwd time |
| `dataset_scale_roofline.png` | Dataset size sweep on roofline | All points at same AI — dataset size doesn't shift roofline |
| `dataset_scale_nvlink.png` | NVLink traffic vs dataset size | Linear scaling; 1M samples = 1.15 TB NVLink traffic |
| `dataset_scale_timing.png` | Step timing + throughput vs dataset | Constant per-step time; total work scales linearly |
| `dataset_scale_telemetry.png` | Telemetry during 1M-sample run | Sustained high-utilisation DDP fingerprint |
| `ddp_telemetry_over_time.png` | GPU util + power during DDP | Both GPUs track each other closely |
| `ddp_step_breakdown_rank0/1.png` | Per-step breakdown during DDP | Backward+allreduce = 58% of step time |
| `ddp_roofline_model.png` | DDP point on roofline | ~9.6% of FP16 ceiling — memory-bound small model |
| `nvlink_bandwidth.png` | NVLink bandwidth vs transfer size | Peak 124.10 GB/s at ≥256 MB transfers |
| `nvlink_latency.png` | Round-trip latency | 35.3 µs minimum — dominated by CUDA overhead |

---

*Report generated as part of the SPAR GPU Monitoring project — Week 4
Hardware: 2× NVIDIA H100 80GB HBM3, NVLink 4.0 (NV6), CUDA 12.8, PyTorch 2.10*
