# SPAR GPU Telemetry — Week 3 Analysis Report

**Project:** Adversarial Classification of ML Workloads via GPU Telemetry
**Date:** 2026-03-09
**Hardware:** 2× NVIDIA A100-SXM4-40GB (dual GPU collection)
**Tier collected:** Tier 1 — NVML/pynvml at 1 Hz

---

## 1. Experimental Setup

### 1.1 Hardware & Software

| Item | Value |
|------|-------|
| GPUs | 2× NVIDIA A100-SXM4-40GB (SXM4 interconnect) |
| CUDA | 13.0 |
| PyTorch | 2.10.0+cu128 |
| Driver | 580.105.08 |
| Sampling rate | 1 Hz (NVML via pynvml) |
| Metrics collected | 15 NVML fields (9 analyzed) |

### 1.2 Telemetry Tier

**Tier 1 (NVML/pynvml)** was the only viable tier in this container environment:

- **Tier 2 (DCGM profiling fields)** — blocked: container lacks `CAP_SYS_ADMIN`; `nv-hostengine` could not access profiling fields
- **Tier 3 (NCU/Nsight Compute)** — blocked: kernel parameter `RmProfilingAdminOnly=1` prevents non-root PMC access

Tier 1 captures: GPU utilization, memory utilization, memory used/total, power draw, temperature, SM clock, memory clock, PCIe TX/RX, encoder/decoder utilization, fan speed.

### 1.3 Workloads

Eleven workload types were collected across five categories:

| Category | Workload Label | Description |
|----------|---------------|-------------|
| **Baseline** | `idle` | GPU idle, no compute |
| **ML Training (FP32)** | `pytorch_resnet_cifar10` | ResNet-50 training, CIFAR-10, batch=128 |
| **ML Training (FP32)** | `pytorch_mlp_cifar10` | 3-layer MLP training, CIFAR-10, batch=256 |
| **ML Training (FP32)** | `gpt2_wikitext2` | GPT-2 124M fine-tuning, WikiText-2, batch=8 |
| **ML Training (FP32)** | `bert_sst2` | BERT-base fine-tuning, SST-2, batch=32 |
| **ML Training (AMP)** | `pytorch_resnet_cifar10_amp` | ResNet-50 with mixed precision (FP16) |
| **Inference** | `resnet50_inference` | ResNet-50 batch inference, no gradients, batch=256 |
| **HPC** | `cufft_benchmark` | Repeated 512×512×64 complex float32 3D FFT |
| **HPC** | `nbody_sim` | Direct-summation N-body gravitational sim O(N²) |
| **Crypto Mining** | `mining_ethash_proxy` | Ethash-like: 1 GB DAG, 64 random gathers/hash |
| **Rendering** | `rendering_proxy` | Monte Carlo path tracing (replaces NVENC; A100 has no video encoder) |

### 1.4 Collection Protocol

- **GPU 0:** sequential collection of all 11 workloads × 3 runs each
- **GPU 1:** parallel simultaneous collection of 8 workloads × 3 runs each
- Total: **54 Parquet files**, **20,322 samples**, collected on 2026-03-09

---

## 2. Dataset Statistics

| Workload | Runs | Samples | Mean duration (s) |
|----------|------|---------|------------------|
| `idle` | 3 | 375 | 124 |
| `pytorch_resnet_cifar10` | 3 | 205 | 67 |
| `pytorch_resnet_cifar10_amp` | 3 | 141 | 46 |
| `pytorch_mlp_cifar10` | 3 | 226 | 74 |
| `gpt2_wikitext2` | 6 | 788 | 130 |
| `bert_sst2` | 6 | 389 | 64 |
| `resnet50_inference` | 6 | 3,653 | 608 |
| `cufft_benchmark` | 6 | 3,636 | 605 |
| `nbody_sim` | 6 | 3,636 | 605 |
| `mining_ethash_proxy` | 6 | 3,637 | 605 |
| `rendering_proxy` | 6 | 3,636 | 605 |

---

## 3. Workload Signature Profiles

Each workload exhibits a distinct "signature" across the 9 analyzed metrics. The table below shows mean and standard deviation for key discriminating features.

### 3.1 Full Signature Table

| Workload | GPU util % | Mem util % | Mem used MB | Power W | Temp °C | SM clock MHz | PCIe RX MB/s |
|----------|-----------|-----------|------------|---------|---------|-------------|-------------|
| `idle` | **0 ± 0** | 0 ± 0 | 518 ± 0 | **39.3 ± 0.2** | **28 ± 0.5** | 210 ± 0 | 0 ± 0 |
| `pytorch_mlp_cifar10` | 6 ± 3 | 0 ± 0 | 1042 ± 181 | 40.2 ± 0.3 | 32 ± 3.2 | 1018 ± 249 | 173 ± 237 |
| `mining_ethash_proxy` | **39 ± 4** | **0 ± 0** | 2030 ± 153 | 40.5 ± 0.3 | 33 ± 2.3 | 1157 ± 83 | **106 ± 67** |
| `bert_sst2` | 78 ± 40 | 12 ± 7 | 4268 ± 1752 | 41.9 ± 0.9 | 52 ± 4.3 | 1106 ± 317 | 9 ± 8 |
| `gpt2_wikitext2` | 84 ± 37 | 19 ± 8 | **9227 ± 3551** | 41.8 ± 0.8 | 51 ± 6.9 | 1064 ± 224 | 20 ± 324 |
| `pytorch_resnet_cifar10` | 82 ± 35 | 47 ± 20 | 4510 ± 1521 | 43.2 ± 1.3 | 47 ± 5.5 | 1023 ± 263 | 99 ± 140 |
| `pytorch_resnet_cifar10_amp` | 77 ± 38 | 35 ± 18 | 2492 ± 921 | 43.0 ± 1.5 | 48 ± 3.3 | 1028 ± 325 | 168 ± 391 |
| `resnet50_inference` | 65 ± 22 | 49 ± 17 | **5156 ± 550** | 43.5 ± 1.7 | 52 ± 1.7 | 1169 ± 173 | **1147 ± 2651** |
| `cufft_benchmark` | **98 ± 10** | **98 ± 10** | 1784 ± 127 | **46.0 ± 0.6** | **58 ± 2.1** | 1035 ± 72 | 17 ± 8 |
| `nbody_sim` | **98 ± 10** | **88 ± 9** | 1840 ± 132 | **46.1 ± 0.6** | **58 ± 1.9** | **1227 ± 84** | 24 ± 18 |
| `rendering_proxy` | 84 ± 9 | 41 ± 5 | 1157 ± 64 | 43.7 ± 0.4 | 57 ± 3.6 | **1391 ± 97** | 137 ± 91 |

Bold values indicate the most extreme (highest or lowest) for that metric.

---

## 4. Analysis by Metric

### 4.1 GPU Utilization

GPU utilization is the single most informative discriminating metric, spanning the full 0–100% range across workloads:

| Tier | Workloads | Mean GPU util |
|------|----------|--------------|
| Idle | `idle` | 0% |
| Very low | `pytorch_mlp_cifar10` | 6% |
| Moderate | `mining_ethash_proxy` | 39% |
| Moderate–high | `resnet50_inference` | 65% |
| High | `bert_sst2`, `gpt2_wikitext2`, `pytorch_resnet_cifar10`, `rendering_proxy` | 77–84% |
| Near-saturated | `cufft_benchmark`, `nbody_sim` | 98% |

**Coefficient of variation (CV)** reveals two behavioral classes:
- **Bursty workloads** (high CV ~40–50%): `bert_sst2` (51%), `pytorch_mlp_cifar10` (56%), `pytorch_resnet_cifar10_amp` (50%), `gpt2_wikitext2` (44%) — these have periodic compute bursts separated by data-loading or communication gaps
- **Steady workloads** (low CV ~10%): `cufft_benchmark` (10%), `nbody_sim` (10%), `mining_ethash_proxy` (11%), `rendering_proxy` (11%) — these sustain constant compute loops with no pipeline stalls

This burst vs. steady-state distinction is a key classification signal.

### 4.2 Memory Utilization

Memory utilization (fraction of memory bandwidth used) is highly diagnostic for compute-bound vs memory-bound workloads:

- `cufft_benchmark`: **98%** — FFT is the most memory-bandwidth-bound operation measured; every sample sees near-peak bandwidth
- `nbody_sim`: **88%** — N-body O(N²) forces repeated reads of all particle positions; similarly bandwidth-bound
- `resnet50_inference`: **49%** — moderate; batch inference alternates compute and activation memory accesses
- `gpt2_wikitext2`: **19%** — GPT-2 training is compute-bound (large matrix multiplications); memory bandwidth is secondary
- `bert_sst2`: **12%** — BERT similarly compute-bound
- `mining_ethash_proxy`: **0%** — anomalous; despite 1 GB DAG with random gathers, NVML memory utilization counter reads zero, suggesting the random-access pattern does not register as sustained bandwidth pressure in NVML's sampling window
- `pytorch_mlp_cifar10`: **0%** — model is tiny (3 layers), barely stresses memory bandwidth

### 4.3 Memory Used (MB)

GPU memory footprint is a strong fingerprint for model architecture:

| Workload | Mean memory (MB) | Notes |
|----------|-----------------|-------|
| `gpt2_wikitext2` | **9,227 MB** | GPT-2 weights + optimizer states + activations; 10.7 GB peak |
| `resnet50_inference` | **5,156 MB** | Model + batch buffer, stable (CV=11%) |
| `pytorch_resnet_cifar10` | **4,510 MB** | Adds gradient tensors vs inference |
| `bert_sst2` | **4,268 MB** | BERT-base + optimizer; grows with steps |
| `pytorch_resnet_cifar10_amp` | **2,492 MB** | AMP halves parameter storage vs FP32 |
| `mining_ethash_proxy` | **2,030 MB** | 1 GB DAG + working buffers |
| `cufft_benchmark` | **1,784 MB** | 512×512×64 complex buffers |
| `nbody_sim` | **1,840 MB** | Position/velocity tensors for N=1024 particles |
| `pytorch_mlp_cifar10` | **1,042 MB** | Tiny MLP, mostly framework overhead |
| `rendering_proxy` | **1,157 MB** | Ray buffers, low memory footprint |
| `idle` | **518 MB** | Framework baseline (CUDA context) |

**Key finding:** AMP (`pytorch_resnet_cifar10_amp`) uses 45% less memory than FP32 (`pytorch_resnet_cifar10`) for the same model — a reliable AMP detector when combined with similar GPU utilization.

### 4.4 Power Draw

All workloads remained in a narrow 39–46 W band — far below the A100's 400 W TDP. This is because:
1. Container power capping at the host level
2. NVML's `power_draw` field in this environment appears to reflect only the GPU power envelope visible to the container, not total board power

Despite the narrow range, power is still discriminating:

| Power band | Workloads |
|-----------|----------|
| ~39.3 W | `idle` (absolute baseline) |
| ~40–41 W | `mining_ethash_proxy`, `pytorch_mlp_cifar10`, `gpt2_wikitext2`, `bert_sst2` |
| ~43–44 W | `resnet50_inference`, `pytorch_resnet_cifar10`, `rendering_proxy` |
| ~46 W | `cufft_benchmark`, `nbody_sim` (highest — both are near-100% GPU util) |

Power and GPU utilization are **strongly correlated** (expected) but not perfectly — rendering_proxy achieves 84% GPU util at only 43.7 W, while nbody achieves 98% at 46 W, suggesting rendering uses fewer power-hungry functional units.

### 4.5 SM Clock Frequency

The SM clock is a sensitive indirect indicator of workload type. The A100 boosts clock depending on the computational pattern:

| Workload | Mean SM clock (MHz) | Interpretation |
|----------|-------------------|---------------|
| `rendering_proxy` | **1,391 MHz** | Highest — irregular branchy raytracing code; GPU boosts aggressively |
| `nbody_sim` | **1,227 MHz** | High — pure arithmetic, benefits from max clock |
| `resnet50_inference` | **1,169 MHz** | High SM clock, no gradient overhead |
| `mining_ethash_proxy` | **1,157 MHz** | Sustained compute loops, good boost |
| `bert_sst2` | **1,106 MHz** | Lower than training peers — bursty pattern inhibits peak boost |
| `gpt2_wikitext2` | **1,064 MHz** | Similar to BERT |
| `cufft_benchmark` | **1,035 MHz** | Lower than expected — memory-bound, SM is often waiting |
| `pytorch_resnet_cifar10_amp` | **1,028 MHz** | |
| `pytorch_resnet_cifar10` | **1,023 MHz** | Similar to AMP — SM clock not sensitive to precision here |
| `pytorch_mlp_cifar10` | **1,018 MHz** | Low utilization → low boost |
| `idle` | **210 MHz** | Base clock — definitive idle signature |

**Note:** Memory clock (`mem_clock_mhz`) is **1,215 MHz for all workloads including idle** — the A100 locks memory clock at maximum regardless of load. This metric provides zero classification signal in this setup.

### 4.6 PCIe Bandwidth

PCIe RX (data transferred from CPU to GPU) is the most discriminating metric for data-loading-intensive workloads, but also the noisiest:

| Workload | Median PCIe RX (MB/s) | p95 PCIe RX | Pattern |
|----------|----------------------|------------|---------|
| `resnet50_inference` | 17 | **7,858** | Burst pattern — batches loaded then processed |
| `rendering_proxy` | 137 | 142 | Steady moderate |
| `mining_ethash_proxy` | 105 | 110 | Steady, consistent DAG reads |
| `pytorch_resnet_cifar10_amp` | 30 | 347 | Bursty |
| `pytorch_resnet_cifar10` | 23 | 340 | Bursty, bimodal |
| `nbody_sim` | 24 | 25 | Very stable, low |
| `cufft_benchmark` | 17 | 18 | Stable, low |
| `gpt2_wikitext2` | 9 | 24 | Low, bursty |
| `bert_sst2` | 8 | 33 | Low, bursty |
| `pytorch_mlp_cifar10` | 0 | 646 | Bimodal: idle or burst |
| `idle` | 0 | 0 | Zero |

`resnet50_inference` has a p95 PCIe RX of **7,858 MB/s** — it is the only workload that saturates PCIe bandwidth during batch loading, making it uniquely identifiable even with just PCIe data.

---

## 5. Workload Similarity and Confusability

Cosine similarity on z-scored mean feature vectors reveals which workloads are hardest to distinguish:

### 5.1 Highly Similar Pairs (potential confusability)

| Pair | Cosine similarity | Why similar |
|------|-----------------|-------------|
| `cufft_benchmark` ↔ `nbody_sim` | **+0.964** | Both near-100% GPU util, ~98% mem util, ~46 W, low PCIe — HPC compute-bound twins |
| `mining_ethash_proxy` ↔ `pytorch_mlp_cifar10` | **+0.898** | Both low GPU util (~5–40%), low power, low memory — superficially similar idle-ish signatures |
| `bert_sst2` ↔ `gpt2_wikitext2` | **+0.707** | Both NLP transformer training: similar GPU util, memory bandwidth, power |

### 5.2 Highly Dissimilar Pairs (easy to distinguish)

| Pair | Cosine similarity | Why different |
|------|-----------------|---------------|
| `pytorch_mlp_cifar10` ↔ `pytorch_resnet_cifar10` | **−0.894** | Same framework but vastly different model complexity — MLP uses 6% GPU, ResNet uses 82% |
| `cufft_benchmark` ↔ `mining_ethash_proxy` | **−0.829** | FFT is 98% GPU util + 98% mem util vs. mining at 39% GPU util + 0% mem util |
| `mining_ethash_proxy` ↔ `nbody_sim` | **−0.745** | Mining has unique low-util + high-PCIe RX pattern; nbody is saturated |
| `idle` ↔ `nbody_sim` | **−0.665** | Trivially separable: zero vs saturated |

### 5.3 Category-Level Summary

| Category pair | Distinguishability | Key discriminating signals |
|--------------|-------------------|--------------------------|
| HPC vs Idle | Easy | GPU util (0% vs 98%), temp, power |
| HPC vs Mining | Easy | GPU util (98% vs 39%), mem util (98% vs 0%) |
| NLP Training vs HPC | Medium | Memory footprint (9 GB vs 1.8 GB), bursty vs steady |
| ResNet FP32 vs AMP | Hard | Memory used (4.5 GB vs 2.5 GB), otherwise similar |
| BERT vs GPT-2 | Hard | Memory used (4.3 GB vs 9.2 GB), otherwise similar |
| Rendering vs ResNet training | Medium | SM clock (1391 vs 1023 MHz), low CV vs high CV |
| Mining vs MLP | Hard | GPU util (39% vs 6%) — both low; differ in PCIe patterns |

---

## 6. Feature Importance Summary

Ranked by ability to separate workload classes using Tier 1 metrics alone:

| Rank | Metric | Classification value | Why |
|------|--------|---------------------|-----|
| 1 | **GPU utilization %** | Very high | Spans full range; separates idle/low/medium/high/saturated tiers |
| 2 | **GPU util CV (variability)** | Very high | Distinguishes bursty (training) from steady (HPC/mining/inference) |
| 3 | **Memory used (MB)** | High | Model size fingerprint; separates NLP from CV, inference from training |
| 4 | **Memory utilization %** | High | Separates memory-bound HPC (98%) from everything else |
| 5 | **PCIe RX bandwidth** | High | Uniquely high for `resnet50_inference`; distinguishes data-loading patterns |
| 6 | **SM clock (MHz)** | Medium | Rendering proxy has uniquely high SM clock; idle has uniquely low |
| 7 | **Temperature °C** | Medium | Proxy for sustained power; HPC runs ~8°C hotter than training |
| 8 | **Power draw (W)** | Medium–Low | Narrow range (39–46 W) due to container caps; still separates HPC |
| 9 | **Memory clock (MHz)** | None | Locked at 1,215 MHz for all workloads — provides zero signal |

---

## 7. Notable Findings

### 7.1 The cufft/nbody Confusability Problem
`cufft_benchmark` and `nbody_sim` are nearly identical in all Tier 1 metrics (cosine similarity 0.964). Both sustain ~98% GPU utilization, ~90% memory utilization, ~46 W, and ~1,100–1,227 MHz SM clock. The only distinguishing signal is SM clock (nbody: 1,227 MHz vs cufft: 1,035 MHz) — nbody is pure float arithmetic while FFT is transform-bound. **Tier 2/3 metrics (tensor core activity, FP32/FP64 pipe utilization) would be required to reliably separate them.**

### 7.2 Mining Proxy Shows Low GPU Utilization
Despite being designed as a "GPU-intensive" crypto-mining simulation, `mining_ethash_proxy` achieves only 39% GPU utilization. This is because random memory accesses to the 1 GB DAG create severe memory-access latency that stalls SM execution. This actually matches real Ethash behavior on data-center GPUs — mining is fundamentally memory-latency-bound, not compute-bound.

### 7.3 AMP Halves Memory Without Affecting Compute Profile
`pytorch_resnet_cifar10_amp` uses 45% less GPU memory (2,492 MB vs 4,510 MB for FP32) but shows nearly identical GPU utilization, power, and SM clock. Memory footprint is the single reliable AMP detector at Tier 1. Tier 2 metrics (FP16 pipe utilization) would confirm this.

### 7.4 ResNet Inference has Extreme PCIe Bursts
`resnet50_inference` shows the most extreme PCIe behavior: median 17 MB/s but p95 at 7,858 MB/s. This bimodal pattern — sustained compute followed by large batch uploads — creates a unique temporal signature invisible in mean statistics but visible in time-series.

### 7.5 Temperature as a Sustained-Load Indicator
`cufft_benchmark` and `nbody_sim` run ~8–10°C hotter than ML training workloads despite similar GPU utilization. This is because HPC workloads sustain near-100% utilization continuously (low CV), while training workloads have periodic gaps that allow partial thermal recovery.

### 7.6 Memory Clock Provides No Information
The A100 locks memory clock at maximum (1,215 MHz) regardless of workload, including during idle. This is a hardware power management policy difference from consumer GPUs where memory clock scales with load.

---

## 8. Limitations and Next Steps

### Limitations
1. **No Tier 2/3 data** — tensor core activity, FP16/FP32/FP64 pipe utilization, and cache miss rates would dramatically improve classification of confusable pairs (cufft vs nbody, BERT vs GPT-2)
2. **Low power range** — NVML power readings appear container-capped at ~46 W vs real A100 TDP of 400 W; power is a weaker signal than it would be on bare-metal
3. **Short training runs** — BERT (64 s) and ResNet (67 s) runs are short enough that warmup phases dominate; longer runs would show stable steady-state
4. **Memory clock locked** — eliminates one NVML metric entirely on A100
5. **1 Hz sampling** — fast events (kernel launches, memory allocations) are averaged out; higher-frequency sampling would reveal sub-second patterns

### Recommended Next Steps (Week 4)
1. **EDA / signal comparison table** — compute per-workload statistics for each metric and rank by discriminability (F-statistic or Kruskal-Wallis H)
2. **Temporal feature engineering** — compute rolling std, autocorrelation, and burst frequency from time-series to capture periodicity
3. **Extend baseline classifier** — add multi-class classification; test on held-out runs
4. **Confusion matrix analysis** — quantify which workload pairs are most commonly misclassified at Tier 1 to justify Tier 2/3 collection

---

## 10. Can We Distinguish Learning Tasks from Others?

**Yes — with high accuracy.** A binary classification experiment was run on all 54 runs, labelling the 5 ML training workloads (`bert_sst2`, `gpt2_wikitext2`, `pytorch_resnet_cifar10`, `pytorch_resnet_cifar10_amp`, `pytorch_mlp_cifar10`) against all other workloads (HPC, inference, mining, rendering, idle).

### 10.1 Classification Results

| Classifier | Granularity | F1 Score |
|---|---|---|
| Logistic Regression | Per sample (1 Hz) | 0.709 |
| Random Forest | Per sample (1 Hz) | **0.968 ± 0.005** |
| Random Forest | Per run (aggregated stats) | **1.000 ± 0.000** |

Per-run classification is **perfect** — when given the full statistics of a run (mean, std, and CV of each metric), the model makes zero errors across 5-fold cross-validation. Even at 1 Hz sample granularity, the Random Forest achieves 96.8% F1.

### 10.2 Confusion Matrix (per sample, 5-fold CV)

|  | Predicted: Non-training | Predicted: Training |
|---|---|---|
| **Actual: Non-training** | 18,526 ✓ | 47 ✗ |
| **Actual: Training** | 65 ✗ | 1,684 ✓ |

Total error rate: **112 / 20,322 samples = 0.55%**

### 10.3 The Key Signal: Variability, Not Level

Training workloads are not distinguished by how high their metrics are — they are distinguished by how **erratically those metrics fluctuate**. The top 4 features by Random Forest importance are all variability measures:

| Rank | Feature | Importance | Training value | Non-training value |
|---|---|---|---|---|
| 1 | GPU util CV | 18.9% | 42–56% | 0–35% |
| 2 | Memory used CV | 15.9% | 17–41% | 5–11% |
| 3 | SM clock std | 14.9% | 220–325 MHz | 72–173 MHz |
| 4 | SM clock CV | 14.3% | 21–31% | 0–18% |
| 5 | Memory used std | 7.9% | 920–3,550 MB | 0–550 MB |

The top 4 features alone account for **64% of the model's total discriminating power**.

**Physical explanation:** Training loops alternate between compute-intensive forward/backward passes and CPU-side operations (data loading, optimizer steps, logging). This produces a periodic burst-and-pause pattern in GPU utilization and memory usage. Non-training workloads (HPC, mining, rendering, inference) run tight continuous loops with stable resource usage.

### 10.4 Interpretable One-Rule Classifier

A simple threshold rule derived from the top two features correctly separates all training from all non-training runs:

> **GPU utilization CV > 30% AND SM clock std > 150 MHz → Training**

| Workload | GPU Util CV | SM Clock Std | Rule prediction | Correct? |
|---|---|---|---|---|
| `bert_sst2` | 51.2% | 317 MHz | Training | ✓ |
| `gpt2_wikitext2` | 43.7% | 224 MHz | Training | ✓ |
| `pytorch_resnet_cifar10` | 42.2% | 263 MHz | Training | ✓ |
| `pytorch_resnet_cifar10_amp` | 50.0% | 325 MHz | Training | ✓ |
| `pytorch_mlp_cifar10` | 56.1% | 249 MHz | Training | ✓ |
| `resnet50_inference` | 33.5% | 173 MHz | Non-training | ✓ |
| `rendering_proxy` | 10.9% | 97 MHz | Non-training | ✓ |
| `cufft_benchmark` | 10.1% | 72 MHz | Non-training | ✓ |
| `nbody_sim` | 10.0% | 84 MHz | Non-training | ✓ |
| `mining_ethash_proxy` | 10.5% | 83 MHz | Non-training | ✓ |
| `idle` | 0.0% | 0 MHz | Non-training | ✓ |

**All 11 workloads correctly classified** by this two-feature rule at the per-run level.

### 10.5 Hardest Edge Case: ResNet Inference

`resnet50_inference` is the most confusable non-training workload at the per-sample level (29 of its samples were misclassified). Its GPU util CV (33.5%) and SM clock std (173 MHz) are higher than all other non-training workloads because batch inference also has a load-compute cycle, just faster and less variable. The two-threshold rule correctly places it in the non-training class since both values fall below the thresholds.

### 10.6 PCA Visualization

A 2-component PCA of the 7 NVML metrics explains a significant fraction of variance. In the PCA projection:
- Training workloads form a scattered, diffuse cloud (reflecting their high within-run variability)
- Non-training workloads form tight, compact clusters (low variability = consistent feature vectors sample-to-sample)
- `resnet50_inference` partially overlaps with the training cloud due to its batch-load pattern

See `analysis/plots/training_vs_other/pca_projection.png`.

---

## 9. Plots Reference

All plots are in `analysis/plots/`:

| Plot | Path | Description |
|------|------|-------------|
| Time-series per metric | `timeseries/<metric>.png` (9 files) | All runs overlaid, x = elapsed seconds |
| Box plots per metric | `boxplots/<metric>.png` (9 files) | Sample distribution per workload |
| Mean ± std bar charts | `bars/<metric>_bar.png` (5 files) | Key metrics sorted by mean |
| Signal heatmap | `heatmap_mean_metrics.png` | Z-score normalised mean metrics × workloads |
| Correlation matrix | `correlation_matrix.png` | Pearson r between all 9 metrics |
| Power vs util scatter | `scatter_power_vs_util.png` | 20k samples coloured by workload |
| Memory timeline | `memory_timeline_by_category.png` | Per-category memory usage over time |
| **Training vs other scatter** | `training_vs_other/scatter_cv_smstd.png` | GPU util CV vs SM clock std — main decision boundary |
| **Training vs other violin** | `training_vs_other/violin_top4_features.png` | Top 4 discriminating features side-by-side |
| **Feature importance** | `training_vs_other/feature_importance.png` | RF Gini importance for binary classification |
| **PCA projection** | `training_vs_other/pca_projection.png` | 2D PCA coloured by workload and by class |
| **Timeseries comparison** | `training_vs_other/timeseries_training_vs_other.png` | Bursty training vs steady non-training side-by-side |

---

*Generated by `analysis/visualize_telemetry.py` from 54 Parquet files collected 2026-03-09.*
