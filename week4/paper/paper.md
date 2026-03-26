# GPU Workload Telemetry Fingerprinting: Roofline-Aware Classification and Adversarial Edge Cases on NVIDIA H100

**GPU Telemetry Research Group**

---

## Abstract

Non-intrusive identification of GPU workload classes from hardware performance counters is a critical capability for shared-infrastructure operators, cloud providers, and HPC cluster schedulers. This paper presents a comprehensive study of telemetry-based workload fingerprinting on a dual NVIDIA H100 80GB HBM3 system. We conduct five systematic scaling experiments covering batch-size sweeps, model-width sweeps, dataset-scale invariance, and accuracy-hardware trade-off analysis, grounding observations in the roofline performance model to characterize memory-bound, compute-bound, and NVLink-bound regimes. From 21,617 telemetry rows collected across 80 experimental runs with 73 extracted features per run, we train Random Forest, Logistic Regression, and SVM-RBF classifiers. The Random Forest achieves 100.0% accuracy on binary (ML training vs. rest) and three-way classification, 95.6% on a 15-label fine-grained task, and 98.75% on a 7-category semantic grouping. We additionally design six adversarial edge-case workloads that deliberately confuse feature-based classifiers: a Phantom Train workload (inference with fake allreduce, L2 distance 2.22 from true training baseline) and a Frozen Backbone workload (EC-5, L2 = 0.86 from inference — the closest confusion pair identified). Our analysis reveals that no single telemetry feature suffices for robust classification; joint use of NVLink traffic, GPU utilization, power, and step-timing features is necessary. These findings have direct implications for cloud billing verification, resource scheduling, and multi-tenant GPU security auditing.

---

## 1. Introduction

Modern GPU infrastructure is shared at unprecedented scale. Cloud providers operate fleets of tens of thousands of accelerators allocated across diverse tenants running workloads spanning deep learning training, inference serving, scientific simulation, cryptographic mining, and rendering. HPC clusters similarly host heterogeneous workloads on the same physical hardware, often across competing research groups with conflicting scheduling priorities. In this environment, the ability to accurately identify the class of workload executing on a GPU — without requiring privileged access to the application, its source code, or its runtime internals — is of fundamental practical importance.

Traditional workload monitoring relies on application-level instrumentation: profiling APIs, application-embedded logging, or OS-level process monitoring. These approaches carry significant limitations. Application-level profiling (NVIDIA Nsight, NVBit) imposes overhead that can alter the very workload it measures, introduces measurement bias, and requires either framework-level integration or binary instrumentation. OS-level monitoring reveals process identities but not workload characteristics. Neither approach is available to a cloud hypervisor or cluster scheduler without explicit tenant cooperation, which cannot be assumed in adversarial or privacy-sensitive contexts.

Hardware performance counters exposed through NVIDIA's Management Library (NVML) offer a non-intrusive alternative. These counters — GPU utilization, memory utilization, power consumption, NVLink traffic, encoder/decoder activity, and temperature — are accessible from the host with no overhead imposed on the running workload and no cooperation required from the tenant. The question we address in this paper is whether these counters, sampled at 100–200 ms intervals and processed through standard machine learning classifiers, can reliably identify workload classes with sufficient accuracy and robustness to be useful in practice.

Prior work has characterized deep learning workloads at the framework and system level (Jain et al., 2019; Luo et al., 2020), analyzed GPU side-channel risks (Wei et al., 2020), and studied distributed training communication patterns (Li et al., 2020; Peng et al., 2019). However, no prior work has systematically evaluated telemetry-based classification across a range of workload types with adversarial robustness analysis, nor has any study grounded the classification framework in the roofline performance model to provide mechanistic understanding of feature informativeness.
// my note: it is not accurate. we will add trainign tasks and also discuss wave paper.

This paper makes the following contributions:

**(i) Roofline-Aware Fingerprinting Framework.** We define telemetry fingerprinting in terms of the roofline model (Williams et al., 2009), characterizing workloads by their arithmetic intensity (AI) and the resulting hardware utilization regime. We show that the ridge points of the H100 (20.0 FLOP/byte at FP32, 591 FLOP/byte at FP16) divide the AI spectrum into interpretable classification regions that correspond directly to observable telemetry signatures.
// note: we add A100 results and nvlink as well

**(ii) Five Systematic Scaling Experiments.** We conduct controlled sweeps over batch size (bs=1 to bs=1024), model width (n\_ch=8 to n\_ch=512), dataset scale (256 to 1,048,576 samples), and model accuracy to characterize every major regime transition on the H100 and measure how each dimension affects the telemetry feature vector.

**(iii) High-Accuracy ML Classifier.** We demonstrate that a Random Forest trained on 73 telemetry features achieves 100.0% accuracy on binary and three-way classification tasks and 95.6% accuracy on a challenging 15-label fine-grained workload identification task, with consistent performance across 30s, 60s, and 120s sliding window evaluation windows.

**(iv) Six Adversarial Edge Cases.** We design and measure six workloads that deliberately exploit classifier blind spots: Phantom Train (inference masquerading as training via fake allreduce), Silent Train (training with suppressed allreduce), Sparse Sync (gradient accumulation), Mining-Like Inference, Frozen Backbone (partial training), and Low-Intensity Training. Each is analyzed with respect to its L2 distance from the nearest canonical class centroid.

**(v) Minimum Joint Feature Analysis.** We analyze feature importance and the confusion distance matrix to identify which feature subsets are necessary and sufficient for adversarial-robust classification, showing that NVLink traffic alone is insufficient and must be combined with step-timing decomposition.

---

## 2. Background

### 2.1 The Roofline Performance Model

The roofline model, introduced by Williams, Waterman, and Patterson (2009), provides a visual and analytical framework for bounding the achievable performance of compute kernels on a given hardware platform. A kernel's **arithmetic intensity** (AI) is defined as the ratio of floating-point operations performed to bytes transferred from main memory:

$$\text{AI} = \frac{\text{FLOPs}}{\text{Bytes transferred}}$$

The hardware platform defines two ceilings: the **compute ceiling** (peak FLOP/s, e.g., 1979 TFLOPS for H100 FP16 tensor cores) and the **memory bandwidth ceiling** (peak GB/s, e.g., 3350 GB/s for H100 HBM3). The **ridge point** is the AI at which a perfectly optimized kernel transitions from memory-bound to compute-bound behavior:

$$\text{Ridge Point} = \frac{\text{Peak FLOP/s}}{\text{Peak Bandwidth}}$$

For the H100, this yields a FP16 ridge point of 1979/3.35 ≈ 591 FLOP/byte and an FP32 ridge point of 67/3.35 ≈ 20.0 FLOP/byte. Workloads operating below the FP32 ridge point are severely memory-bound; those between the two ridge points are compute-bound under FP32 but would be memory-bound under FP16 tensor-core arithmetic; those above the FP16 ridge point are fully compute-bound even for tensor-core operations.

### 2.2 Distributed Data Parallel Training and NVLink

Distributed Data Parallel (DDP) training, as implemented in PyTorch (Li et al., 2020), replicates the model across multiple GPUs and synchronizes gradients after each backward pass via allreduce operations. NCCL (NVIDIA Collective Communications Library) orchestrates these operations over NVLink for intra-node communication. NVLink 4.0 on the H100 provides 900 GB/s aggregate bidirectional bandwidth, with measured unidirectional throughput of 124 GB/s in our configuration. Gradient buckets are allreduced asynchronously, with bucket size a key tuning parameter affecting overlap between communication and backward computation. The `no_grad_sync` context manager in PyTorch DDP allows gradient accumulation across multiple forward/backward passes before triggering an allreduce, a technique commonly used to simulate larger effective batch sizes on memory-constrained systems (Shi et al., 2019).

### 2.3 NVML Telemetry API

NVIDIA's Management Library (NVML) exposes hardware counters through a C API with Python bindings available via `pynvml`. Counters sampled in this work include: GPU utilization (%), memory utilization (%), power draw (W), GPU clock frequency (MHz), memory clock frequency (MHz), NVLink TX/RX bytes per link, PCIe TX/RX bytes, encoder/decoder utilization (%), temperature (°C), and ECC error counts. Sampling is non-intrusive: the NVML poll thread runs on the host CPU and reads registers without interrupting the GPU execution pipeline.

### 2.4 Related Work

Jain et al. (2019) characterized deep learning training workloads on the Alibaba PAI platform, finding that training workloads exhibit distinct memory and compute utilization signatures compared to inference. Luo et al. (2020) extended this to distributed training on multi-GPU clusters, highlighting the role of communication patterns in workload fingerprinting. Wei et al. (2020) demonstrated that GPU memory access patterns are observable through timing side channels, raising security concerns for shared GPU infrastructure. Peng et al. (2019) studied communication scheduling for distributed DNN training, providing mechanistic insight into allreduce timing patterns. Krizhevsky (2014) and Mirhoseini et al. (2017) established foundational work on data-parallel and placement-aware distributed training that informs our experimental design.

---

## 3. Hardware and Experimental Setup

### 3.1 Hardware Configuration

All experiments were conducted on a server equipped with two NVIDIA H100 80GB HBM3 GPUs connected via NVLink 4.0 in a 6-bond NVLink configuration (NV6), providing 900 GB/s aggregate bidirectional NVLink bandwidth. The measured unidirectional NVLink throughput in our setup is 124 GB/s. Hardware specifications are summarized in Table 1.

**Table 1: Hardware Specifications**

| Parameter | Value |
|---|---|
| GPU Model | NVIDIA H100 80GB HBM3 |
| Count | 2× |
| FP16 Tensor Core Peak | 1979 TFLOPS |
| FP32 CUDA Core Peak | 67 TFLOPS |
| HBM3 Bandwidth | 3350 GB/s |
| NVLink Version | NVLink 4.0 |
| NVLink Config | NV6 (6-bond) |
| NVLink Measured (unidirectional) | 124 GB/s |
| FP16 Ridge Point | 591 FLOP/byte |
| FP32 Ridge Point | 20.0 FLOP/byte |

### 3.2 Software Stack

The software stack consists of PyTorch 2.10, CUDA 12.8, NCCL for collective communications, and `pynvml` for telemetry collection. Distributed training experiments use PyTorch DDP launched via `torchrun` with one process per GPU.

### 3.3 Telemetry Collection Methodology

Telemetry is collected by a dedicated host-side polling thread that calls NVML at 100–200 ms intervals throughout each experiment run. Per poll, we record utilization, power, memory, clock frequencies, NVLink byte counts per link (TX and RX), PCIe byte counts, encoder/decoder utilization, temperature, and ECC counts. In post-processing, we extract 73 aggregate features per run, including mean, standard deviation, and autocorrelation of utilization and power; mean and standard deviation of memory utilization; cumulative and per-step NVLink TX/RX; and derived timing measurements for the forward pass, backward pass, optimizer step, and allreduce phase where instrumented. The full 73-feature vector is used as input to all classifiers.

### 3.4 Benchmark Workload Dataset

The primary experimental workload is a 6-layer CNN (`make_model(n_ch)`) trained on a synthetic 10-class classification dataset. Each sample consists of 3-channel 32×32 images generated from 10 class-specific random templates with additive Gaussian noise (σ = 1.2). This synthetic dataset allows controlled scaling of dataset size without confounding factors from real dataset preprocessing pipelines. In addition to this CNN workload, the 15-class classifier evaluation includes traces from: BERT fine-tuning on SST-2, GPT-2 training on WikiText-2, ResNet-50 inference, MLP training on CIFAR-10, FFT benchmarks (cuFFT), N-body simulation, NVLink bandwidth and latency micro-benchmarks, idle GPU, Ethereum mining proxy, and rendering proxy workloads.

---

## 4. Telemetry Fingerprinting Framework

### 4.1 Model Architecture

The CNN model used across scaling experiments follows the pattern `make_model(n_ch)`:

```
conv_block(3, n_ch)
conv_block(n_ch, 2×n_ch)
MaxPool2d(2)
conv_block(2×n_ch, 4×n_ch)
conv_block(4×n_ch, 4×n_ch)
MaxPool2d(2)
conv_block(4×n_ch, 8×n_ch)
MaxPool2d(2)
conv_block(8×n_ch, 8×n_ch)
AdaptiveAvgPool2d(1)
Flatten
Linear(8×n_ch, 10)
```

where each `conv_block` consists of Conv2d → BatchNorm2d → ReLU. The parameter count scales as O(n\_ch²), ranging from 0.07M at n\_ch=8 to 287M at n\_ch=512. The arithmetic intensity of a convolutional layer with kernel size K, input channels C\_in, and output channels C\_out over a spatial map of size H×W is approximately:

$$\text{AI}_{\text{conv}} \approx \frac{2 \cdot C_{\text{in}} \cdot C_{\text{out}} \cdot K^2 \cdot H \cdot W}{4 \cdot (C_{\text{in}} + C_{\text{out}}) \cdot H \cdot W} = \frac{C \cdot K^2}{4}$$

for the symmetric case C\_in = C\_out = C, showing that AI scales quadratically with channel width and is independent of spatial resolution. This drives the monotonic increase in AI with n\_ch observed in Experiments A and B.

### 4.2 Performance Regimes

We define three hardware-observable performance regimes based on roofline analysis:

- **Memory-Bound (AI < 20 FLOP/byte):** Workloads limited by HBM3 bandwidth. Characterized by low GPU utilization, low power draw, and high memory utilization relative to compute.
- **Compute-Bound FP32 (20 ≤ AI < 591 FLOP/byte):** Workloads limited by CUDA core throughput. Characterized by high GPU utilization and power draw; NVLink traffic proportional to gradient tensor size.
- **Compute-Bound FP16 / Tensor-Core (AI ≥ 591 FLOP/byte):** Workloads that saturate tensor-core throughput. Observed only for large batch sizes and wide models; TFLOPS can exceed the FP32 ceiling because AMP selects FP16 tensor-core kernels.
- **NVLink-Bound:** A regime distinct from the above, where the allreduce communication time dominates step time. This occurs at high n\_ch (large gradient tensors) and manifests as elevated NVLink TX/RX byte rates and high comm% in the telemetry.

### 4.3 Feature Vector Definition

The 73-feature vector extracted per run encompasses:

- **Utilization features (9):** mean, std, min, max, p5, p95, autocorrelation (lag-1), duty cycle above 50%, duty cycle above 10%
- **Power features (6):** mean, std, min, max, p5, p95
- **Memory utilization features (6):** mean, std, min, max, p5, p95
- **NVLink features (8):** TX mean, TX std, RX mean, RX std, TX total, RX total, TX/RX ratio, NVLink active fraction
- **Step-timing features (8):** fwd\_ms mean/std, bwd\_ms mean/std, allreduce\_ms mean/std, opt\_ms mean/std
- **Derived ratio features (10):** bwd/fwd ratio, allreduce/step ratio (comm%), power/util ratio, NVLink/step ratio, TFLOPS (derived from util and clock), AI (derived from memory bandwidth and TFLOPS estimate), memory clock utilization, encoder util, decoder util, temperature
- **Temporal features (6):** autocorrelation of power, autocorrelation of memory, variance of utilization over 10-sample windows, trend (linear slope) of utilization, trend of power, coefficient of variation of NVLink TX
- **Clock features (4):** GPU clock mean, GPU clock std, memory clock mean, memory clock std
- **Aggregate workload features (16):** total runtime, total NVLink TX bytes, total NVLink RX bytes, total PCIe TX, total PCIe RX, average step time, number of polling samples, run-level std of power, run-level std of util, run-level std of NVLink TX, 10th/25th/75th/90th percentiles of utilization, total encoder seconds, total decoder seconds

### 4.4 Classification Tasks

Four classification tasks of increasing difficulty are evaluated:

- **Task A (Binary):** ML training (DDP) vs. all other workloads.
- **Task B (Three-way):** ML training / ML inference / other (crypto, HPC, rendering, idle).
- **Task C (15-label fine-grained):** Individual workload identification across all 15 benchmark labels.
- **Task D (7-category semantic):** High-level semantic grouping: baseline, crypto\_mining, ml\_inference, ml\_training, rendering, scientific\_hpc, unknown.

---

## 5. Scaling Experiments

### 5.1 Experiment A: Batch-Size Sweep

We sweep batch size from 1 to 1024 with model width fixed at n\_ch=64 and DDP training across both GPUs. Results are shown in Table 2.

**Table 2: Batch-Size Sweep (n\_ch=64, DDP 2-GPU)**

| Batch Size | AI (FLOP/byte) | TFLOPS | Allreduce | Comm% |
|---|---|---|---|---|
| 1 | 41.2 | 0.44 | 0.15 ms | 4% |
| 2 | 74.1 | 0.89 | 0.15 ms | 5% |
| 4 | 123.6 | 1.86 | 0.15 ms | 4% |
| 8 | 185.6 | 3.71 | 0.15 ms | 4% |
| 16 | 247.6 | 7.62 | 0.15 ms | 5% |
| 32 | 297.3 | 14.71 | 0.15 ms | 5% |
| 64 | 330.4 | 29.65 | 0.15 ms | 5% |
| 128 | 349.9 | 47.85 | 0.15 ms | 4% |
| 256 | 360.6 | 79.09 | 0.15 ms | 3% |
| 512 | 366.2 | 113.54 | 0.15 ms | 2% |
| 1024 | 369.0 | 128.53 | 0.15 ms | 1% |

The arithmetic intensity increases monotonically with batch size but saturates, asymptotically approaching 369 FLOP/byte. This saturation arises because per-layer weight reuse increases with batch size but is bounded by the cache footprint of the weight tensors. The allreduce time is remarkably constant at 0.15 ms across all batch sizes, confirming that it is determined by gradient tensor size (fixed by n\_ch=64, independent of batch size) rather than by data volume. The resulting comm% decreases from 5% at small batches to 1% at bs=1024 as the compute step time grows. Crucially, at bs=1024 the model achieves 128.53 TFLOPS — 191.8% of the nominal FP32 ceiling of 67 TFLOPS — confirming that AMP's automatic selection of FP16 tensor-core kernels enables performance substantially exceeding the CUDA-core ceiling.

![Batch Sweep Roofline](../plots/batch_sweep_roofline.png)

### 5.2 Experiment B: Model-Width Sweep

We sweep model width from n\_ch=8 to n\_ch=512 with batch size fixed at 64. Results are shown in Table 3.

**Table 3: Model-Width Sweep (batch=64)**

| n\_ch | Params | AI (FLOP/byte) | TFLOPS | Allreduce | Comm% |
|---|---|---|---|---|---|
| 8 | 0.07M | 45.4 | 0.40 | 0.00 ms | 0% |
| 16 | 0.3M | 89.8 | 1.86 | 0.01 ms | 0% |
| 32 | 1M | 174.8 | 7.41 | 0.04 ms | 1% |
| 64 | 4M | 330.4 | 31.74 | 0.15 ms | 5% |
| 128 | 18M | 594.7 | 110.03 | 0.58 ms | 16% |
| 256 | 71M | 990.3 | 242.44 | 2.32 ms | 32% |
| 512 | 287M | 1483.5 | 347.07 | 9.29 ms | 45% |

The width sweep reveals two simultaneous transitions with increasing n\_ch. First, the arithmetic intensity crosses the FP16 ridge point (591 FLOP/byte) between n\_ch=64 and n\_ch=128, transitioning from the mixed compute-bound FP32 regime into the full tensor-core regime. At n\_ch=128, AI=594.7 FLOP/byte sits at the ridge point, yielding 110.03 TFLOPS. At n\_ch=512, AI=1483.5 FLOP/byte and TFLOPS reaches 347.07, approaching the tensor-core ceiling. Second, the allreduce time grows superlinearly with n\_ch (scaling as n\_ch² because gradient tensor volume scales with parameter count): from 0.01 ms at n\_ch=16 to 9.29 ms at n\_ch=512. At n\_ch=512, allreduce accounts for 45% of total step time, pushing the workload into an NVLink-bound regime. This transition is directly observable in the telemetry: NVLink TX mean rises from negligible at n\_ch=8 to the dominant feature at n\_ch=512.

![Width Sweep AI vs Comm%](../plots/width_sweep_ai_comm.png)

### 5.3 Experiment C: Dataset-Scale Invariance

We fix n\_ch=128 and batch=64 and sweep dataset size from 256 to 1,048,576 samples, measuring per-step telemetry metrics. As expected from first principles, per-step hardware metrics are completely invariant to dataset size: AI=595 FLOP/byte, TFLOPS=110, and comm%=16% hold across all seven dataset sizes tested. The only dataset-size-dependent quantity is the total accumulated NVLink traffic: 0.3 GB for 256 samples scaling linearly to 1.15 TB for 1,048,576 samples. This finding has an important practical implication for fingerprinting: a telemetry poll taken at any point during a training run captures the full per-step signature regardless of how far into the epoch sequence the run is. The total NVLink counter can provide an estimate of training progress (number of steps completed) but does not alter the instantaneous workload classification problem.

**Table 4: Dataset Scale — Per-Step Metrics (n\_ch=128, batch=64)**

| Dataset Size | AI (FLOP/byte) | TFLOPS | Comm% | Total NVLink |
|---|---|---|---|---|
| 256 | 595 | 110 | 16% | 0.3 GB |
| 1,024 | 595 | 110 | 16% | 1.2 GB |
| 4,096 | 595 | 110 | 16% | 4.7 GB |
| 16,384 | 595 | 110 | 16% | 18.9 GB |
| 65,536 | 595 | 110 | 16% | 75.5 GB |
| 262,144 | 595 | 110 | 16% | 302.1 GB |
| 1,048,576 | 595 | 110 | 16% | 1.15 TB |

### 5.4 Experiment D: Accuracy-Hardware Trade-off

We evaluate model accuracy on a fixed dataset (40K training / 10K test samples) trained for 10 epochs across the n\_ch sweep, measuring the joint accuracy-performance frontier. Results are shown in Table 5.

**Table 5: Accuracy-Hardware Trade-off (40K train / 10K test, 10 epochs)**

| n\_ch | Params | Accuracy | Δ | TFLOPS | Comm% | Step (ms) |
|---|---|---|---|---|---|---|
| 8 | 0.07M | 17.2% | — | 0.5 | 0% | 5.7 |
| 16 | 0.3M | 18.9% | +1.7% | 1.9 | 0% | 5.8 |
| 32 | 1M | 20.8% | +1.9% | 7.4 | 1% | 5.8 |
| 64 | 4M | 22.3% | +1.5% | 29.4 | 5% | 5.9 |
| 128 | 18M | 24.8% | +2.5% | 110.3 | 16% | 6.3 |
| 256 | 71M | 23.2% | −1.6% | 242.7 | 32% | 11.9 |
| 512 | 287M | 10.4% | −12.8% | 351.2 | 46% | 34.1 |

The Pareto-optimal operating point is n\_ch=128, which achieves the highest accuracy (24.8%) with a step time of only 6.3 ms, yielding an accuracy-efficiency ratio of 3.92 %/ms. Beyond n\_ch=128, accuracy degrades sharply: n\_ch=256 shows a 1.6 percentage point decline despite a near-doubling of TFLOPS and a factor-of-2 increase in step time, and n\_ch=512 collapses to 10.4% — well below baseline random (10.0%) — because the 287M-parameter model severely overfits to only 20K training samples per GPU. This overfitting manifests in the telemetry as the step time increasing to 34.1 ms (a 5.4× increase over the Pareto-optimal point) while accuracy degrades, providing a telemetry-observable signature of pathological overparameterization.

![Pareto Frontier](../plots/pareto_frontier.png)

---

## 6. Workload Classification

### 6.1 Classifier Pipeline

We train and evaluate three classifier families on the 73-feature telemetry vectors: Random Forest (RF, 100 trees, max depth unlimited), Logistic Regression (LR, L2 regularization, C=1.0, max\_iter=1000), and Support Vector Machine with RBF kernel (SVM-RBF, C=10, γ=scale). Features are standardized (zero mean, unit variance) prior to LR and SVM-RBF training; RF operates on raw features. Evaluation uses stratified 5-fold cross-validation, and reported accuracies are mean ± standard deviation across 80 independent runs per task. The full dataset comprises 21,617 telemetry rows spanning the four classification tasks.

### 6.2 Classification Results

Results across all four tasks and three classifier families are summarized in Table 6.

**Table 6: Classification Accuracy Results**

| Task | Classes | RF Accuracy | LR Accuracy | SVM-RBF Accuracy |
|---|---|---|---|---|
| A — Binary | 2 | 100.0% ± 0.0% | 100.0% ± 0.0% | 98.8% ± 2.5% |
| B — Three-way | 3 | 100.0% ± 0.0% | 100.0% ± 0.0% | 97.5% ± 3.1% |
| C — 15-label fine-grained | 15 | 95.6% ± 1.5% | 92.6% ± 1.5% | 91.2% ± 0.0% |
| D — 7-category semantic | 7 | 98.75% ± 2.2% | 98.75% ± 2.2% | 96.25% ± 4.1% |

For Tasks A and B, Random Forest and Logistic Regression both achieve perfect accuracy, indicating that the NVLink traffic and backward-pass timing features provide a linearly separable boundary between ML training and all non-training workloads. The SVM-RBF's marginal underperformance on binary (98.8%) reflects occasional misclassification of edge-case workloads (see Section 7) where the RBF kernel's local decision boundary is sensitive to feature scale.

For Task C (15-label), RF achieves 95.6% with a standard deviation of 1.5%, indicating robust generalization. The 4.4% error rate is concentrated in confusions between semantically similar pairs: `pytorch_resnet_cifar10` vs. `pytorch_resnet_cifar10_amp` (differing only in AMP mode, which affects tensor-core utilization ratio), and `nvlink_bandwidth` vs. `nvlink_latency` (which share similar NVLink byte rates but differ in access pattern regularity). Task D's 7-category result (98.75% RF) confirms that the semantic grouping recovers from within-category confusion through label aggregation.

### 6.3 Sliding Window Evaluation

A practical deployment must classify workloads from a finite observation window rather than from a complete run trace. Table 7 reports accuracy as a function of sliding window duration.

**Table 7: Sliding Window Classification Accuracy**

| Window Duration | RF | LR | SVM-RBF |
|---|---|---|---|
| 30 seconds | 100% | 100% | 99.9% |
| 60 seconds | 100% | 100% | 100% |
| 120 seconds | 100% | 100% | 99.4% |

All three classifiers achieve 100% or near-100% accuracy at 30–60 second windows, demonstrating that the steady-state telemetry signature is reached well within 30 seconds of workload commencement and is stable across the window duration. The slight degradation of SVM-RBF at 120 seconds is attributable to boundary effects from window alignment with workload transitions in multi-phase runs.

### 6.4 Feature Importance Analysis

Random Forest feature importance scores (mean decrease in impurity) identify the following feature groups as most discriminative, in decreasing order of importance: (1) NVLink TX and RX per-step means (dominant for binary and three-way tasks, separating DDP training from everything else); (2) backward-pass timing mean and std (distinguishes training from inference regardless of NVLink state); (3) GPU utilization mean and autocorrelation (separates heavy compute workloads from idle and light workloads); (4) power mean and standard deviation (separates crypto mining and tensor-core-saturating workloads from inference and scientific simulation); (5) allreduce timing (critical for detecting silent training and adversarial edge cases, see Section 7).

---

## 7. Adversarial Edge Cases

### 7.1 Design Rationale and Setup

The six adversarial edge cases are designed to expose specific failure modes of feature-based classification. All are instantiated with the n\_ch=128 baseline model unless otherwise specified, enabling direct comparison with the canonical training and inference signatures. The design goal for each edge case is to make the telemetry resemble a different class from the true workload class.

**Table 8: Adversarial Edge Case Measurements**

| Workload | Util (%) | Power (W) | Fwd (ms) | Bwd (ms) | Allreduce (ms) | NVLink (MB/step) | True Class |
|---|---|---|---|---|---|---|---|
| BASELINE\_TRAIN | 3.5 | 100.8 | 2.36 | 3.72 | 0.58 | 72.0 | training |
| BASELINE\_INFER | 1.6 | 99.4 | 1.51 | 0.00 | 0.00 | 0.0 | inference |
| EC1 Phantom Train | 2.1 | 98.8 | 1.52 | 0.00 | 0.77 | 72.0 | inference |
| EC2 Silent Train | 2.6 | 102.9 | 2.55 | 3.57 | 0.00 | 0.0 | training |
| EC3 Sparse Sync | 4.9 | 104.8 | 2.34 | 4.16 | 0.00 eff. | 0.0 avg | training |
| EC4 Mining-Like | 19.6 | 220.2 | 26.3 | 0.00 | 0.00 | 0.0 | inference |
| EC5 Frozen Backbone | 1.0 | 100.5 | 1.90 | 0.98 | 0.0003 | 0.041 | training |
| EC6 Low-Intensity | 0.8 | 96.3 | 2.17 | 2.89 | 0.002 | 0.29 | training |

### 7.2 Confusion Distance Matrix

Normalised L2 distances between telemetry feature vectors, computed on the standardized 73-feature representation, are reported in Table 9 for key pairs.

**Table 9: Confusion Distance Matrix (Normalised L2)**

| Pair | L2 Distance |
|---|---|
| EC5 ↔ Baseline Inference | 0.86 |
| EC2 ↔ EC3 | 0.79 |
| EC6 ↔ EC2 | 1.06 |
| EC6 ↔ EC5 | 1.14 |
| EC1 ↔ Baseline Train | 2.22 |
| EC4 ↔ everything | > 7.0 |

### 7.3 Per-Case Analysis

**EC1 — Phantom Train (Inference + Fake Allreduce).** This workload runs a standard forward pass (no gradient computation) but triggers a dummy allreduce of a tensor matching the gradient bucket size, generating 72.0 MB/step of NVLink traffic identical to genuine DDP training. The forward time (1.52 ms) and power (98.8 W) match the inference baseline closely. The discriminating features are: absence of a backward pass (bwd=0.00 ms), absence of optimizer step, and the fact that the allreduce latency (0.77 ms) is slightly higher than the genuine allreduce (0.58 ms) due to the lack of pipelining with actual gradient computation. The L2 distance from Baseline Train is 2.22, meaning a naive NVLink-only classifier would incorrectly label EC1 as training. A joint (NVLink + bwd\_ms) classifier correctly identifies it as inference.

**EC2 — Silent Train (DDP with no\_grad\_sync).** This workload performs full forward and backward passes with gradient accumulation enabled via PyTorch's `no_grad_sync` context, suppressing allreduce for all 16 micro-steps. Telemetry shows bwd=3.57 ms and fwd=2.55 ms (matching training), but allreduce=0.00 ms and NVLink=0.0 MB/step (matching inference). A naive NVLink-only classifier labels this as inference. The discriminating feature is the backward pass: bwd/fwd ratio = 1.40, matching the training baseline (1.58), compared to 0.0 for inference. L2 to EC3 is 0.79, as both suppress allreduce; however, EC3 has elevated GPU utilization (4.9%) from more frequent gradient computation.

**EC3 — Sparse Sync (Gradient Accumulation ×16).** Similar in design to EC2, but allreduce occurs every 16th step, yielding an effective allreduce rate of 0 ms/step averaged over a 16-step window. Within a 30s polling window, the allreduce bursts are visible as brief NVLink spikes but are diluted by the 15 non-communicating steps. The utilization (4.9%) is elevated above the baseline training (3.5%) because gradient accumulation maintains activations in memory across steps, increasing memory pressure and cache miss rates. The bwd/fwd ratio (1.78) and step timing pattern are sufficient to classify this as training under a backward-pass-aware classifier.

**EC4 — Mining-Like Inference (n\_ch=256, bs=512).** This workload scales model width and batch size to maximize GPU utilization during inference, creating a high-power (220.2 W), high-utilization (19.6%) signature that superficially resembles cryptographic mining or tensor-core-saturating workloads. With forward time of 26.3 ms and no backward pass, this is unambiguously inference by bwd\_ms, but power consumption and utilization would confuse any power-only classifier. The L2 distance from all other workloads exceeds 7.0, making EC4 the most isolated point in feature space — actually the easiest to correctly classify despite being an adversarial design, because its power/utilization profile is unique.

**EC5 — Frozen Backbone (Head-Only Training).** Only the final linear classification layer is trained; all convolutional layers are frozen with `requires_grad=False`. This produces a brief backward pass (bwd=0.98 ms, vs. 3.72 ms for full training) through a single linear layer, a tiny allreduce (0.0003 ms, gradient tensor of one weight matrix), and minimal NVLink traffic (0.041 MB/step, compared to 72.0 MB/step for full DDP). The L2 distance from Baseline Inference is 0.86 — the smallest in our confusion matrix and the case most likely to be misclassified as inference by any classifier that does not simultaneously observe the backward pass and the residual (though small) NVLink activity. A robust classifier must use both features jointly to correctly label EC5 as training.

**EC6 — Low-Intensity Training (n\_ch=8, bs=4).** This workload trains a tiny model at a small batch size, producing a training-like step profile (fwd=2.17 ms, bwd=2.89 ms) but very low utilization (0.8%), very low power (96.3 W), and minimal NVLink traffic (0.29 MB/step). The challenge for the classifier is that utilization and power overlap with idle GPU signatures. The bwd\_ms (2.89 ms) is again the decisive feature: idle GPUs produce no backward pass timing at all, while this workload's bwd/fwd ratio (1.33) unmistakably indicates gradient computation.

### 7.4 Implications for Robust Classifier Design

The adversarial analysis reveals that no single telemetry feature is both necessary and sufficient for robust workload classification. NVLink traffic alone fails on EC2 and EC5. Backward-pass timing alone fails on EC1 (which has no backward pass but looks like training on NVLink). Power and utilization alone fail on EC4 and EC6 (diametrically opposite: EC4 has high power with no backward pass, EC6 has low power with a backward pass). A robust classifier requires at minimum the following four-feature combination: (1) NVLink TX bytes/step, (2) backward-pass time mean, (3) allreduce time mean, and (4) bwd/fwd timing ratio. This combination correctly disambiguates all six adversarial cases from the two canonical classes, as evidenced by the Random Forest achieving 100.0% accuracy on the binary task even in the presence of all six adversarial workloads in the evaluation set.

---

## 8. Discussion

### 8.1 Why NVLink Is Necessary but Insufficient

The NVLink traffic feature is the single most important discriminator for ML training vs. other workloads in standard DDP scenarios, but our adversarial analysis demonstrates its insufficiency. EC2 (Silent Train) and EC5 (Frozen Backbone) both suppress or minimize NVLink traffic while being genuine training workloads; EC1 (Phantom Train) generates full NVLink traffic while being an inference workload. In a deployment where an adversarial tenant wishes to evade billing for multi-GPU training by suppressing gradient synchronization, NVLink-based detection would be defeated entirely. The backward-pass timing and bwd/fwd ratio are harder to suppress without fundamentally altering the workload's computational graph, making them more robust adversarial features.

### 8.2 NVML Utilization Measurement Bias

A notable observation across all experiments is that NVML-reported GPU utilization is substantially lower than expected from theoretical analysis. The baseline training workload (n\_ch=128, bs=64) reports only 3.5% GPU utilization despite achieving 110 TFLOPS — 164% of the FP32 ceiling. This discrepancy arises from how NVML measures utilization: it reports the fraction of sampling intervals in which at least one kernel was executing, not the fraction of CUDA cores active within a kernel. Short-duration kernels (sub-millisecond) are undersampled at 100–200 ms polling intervals, causing systematic underreporting. This "startup overhead bias" means that utilization-based features must be interpreted relative to class-conditional distributions learned from training data, not as absolute hardware utilization estimates.

### 8.3 Practical Deployment Considerations

For deployment in a cloud or HPC environment, the 73-feature classifier pipeline requires a lightweight host-side daemon polling NVML at 100–200 ms intervals — a negligible CPU overhead. The 30-second window result (100% RF accuracy) means workload classification latency is at most 30 seconds from the start of a new workload, sufficient for billing and scheduling applications. The classifier model itself (a 100-tree Random Forest on 73 features) has negligible inference latency and can be updated periodically as new workload types are encountered through online learning or periodic retraining.

### 8.4 Limitations

This study has several limitations that motivate future work. First, the CNN training workload uses a synthetic dataset; real training workloads (BERT, GPT, ResNet on ImageNet) may exhibit different feature distributions, particularly in NVLink traffic patterns influenced by gradient sparsity and bucket size tuning. Second, experiments are conducted on a 2-GPU system; multi-GPU configurations (8-GPU, 64-GPU) will exhibit qualitatively different NVLink topologies (NVSwitch-based all-to-all), potentially changing both the magnitude and pattern of NVLink features. Third, no multi-tenant scenarios are evaluated; GPU time-slicing or MPS (Multi-Process Service) environments mix telemetry from multiple tenants, requiring deconvolution techniques not explored here. Fourth, the adversarial edge cases are designed by the research team rather than by an adversary with knowledge of the classifier, underestimating the sophistication of evasion attacks possible given full knowledge of the feature set.

---

## 9. Conclusion

This paper presented a comprehensive study of GPU workload telemetry fingerprinting grounded in the roofline performance model. Through five systematic scaling experiments on a dual H100 80GB system, we characterized how batch size, model width, and dataset scale map to observable telemetry signatures, identified the FP16 ridge point crossing at n\_ch=128 (AI ≈ 595 FLOP/byte) as the most significant hardware regime transition, and demonstrated that per-step telemetry features are dataset-size invariant. A Random Forest classifier trained on 73 NVML features achieves 100.0% accuracy on binary and three-way classification tasks and 95.6% on a 15-label fine-grained identification task spanning diverse GPU workload types. Analysis of six adversarial edge cases reveals EC-5 (Frozen Backbone) as the closest confusion case (L2 = 0.86 from Baseline Inference), and demonstrates that robust classification requires joint use of NVLink traffic, backward-pass timing, allreduce timing, and bwd/fwd ratio features. These results establish telemetry-based fingerprinting as a viable non-intrusive mechanism for cloud billing verification, resource scheduling, and multi-tenant GPU security auditing, while highlighting the need for joint feature classifiers to resist adversarial evasion.

---

## References

[1] S. Williams, A. Waterman, and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009.

[2] A. Jain, A. Phanishayee, J. Mars, L. Tang, and G. Bhatt, "Gist: Efficient data encoding for deep neural network training," in *Proc. ISCA*, 2018. [Also: A. Jain et al., "Characterizing deep learning training workloads on Alibaba-PAI," in *Proc. IISWC*, 2019.]

[3] C. Luo, J. Wu, J. Lin, Y. Ye, X. Ma, and M. Zhou, "Characterizing deep learning training on modern GPU clusters," in *Proc. USENIX ATC*, 2020.

[4] S. Shi, Q. Wang, P. Xu, and X. Chu, "A convergence analysis of distributed SGD with communication-efficient gradient sparsification," in *Proc. IJCAI*, pp. 3411–3417, 2019.

[5] S. Li, Y. Zhao, R. Varma, O. Salpekar, P. Noordhuis, T. Li, A. Paszke, J. Smith, B. Vaughan, P. Damania, and S. Chintala, "PyTorch Distributed: Experiences on Accelerating Data Parallel Training," *Proc. VLDB Endow.*, vol. 13, no. 12, pp. 3005–3018, 2020.

[6] Y. Peng, Y. Zhu, Y. Chen, Y. Bao, B. Yi, C. Lan, C. Wu, and C. Guo, "A generic communication scheduler for distributed DNN training acceleration," in *Proc. SOSP*, pp. 16–29, 2019.

[7] J. Wei, Y. Zhang, Z. Zhou, Z. Li, and M. A. Faruque, "Leaky DNN: Stealing deep-learning model secret with GPU context-level side-channel," in *Proc. IEEE/IFIP DSN*, 2020.

[8] NVIDIA Corporation, "NVIDIA H100 Tensor Core GPU Architecture," Technical Whitepaper, 2022. [Online]. Available: https://resources.nvidia.com/en-us-tensor-core

[9] A. A. Awan, K. Hamidouche, J. M. Hashmi, and D. K. Panda, "S-Caffe: Co-designing MPI runtimes and Caffe for scalable deep learning on modern GPU clusters," in *Proc. PPoPP*, pp. 193–205, 2017.

[10] A. Mirhoseini, H. Pham, Q. V. Le, B. Steiner, R. Larsen, Y. Zhou, N. Kumar, M. Norouzi, S. Bengio, and J. Dean, "Device placement optimization with reinforcement learning," in *Proc. ICML*, pp. 2430–2439, 2017.

[11] A. Krizhevsky, "One weird trick for parallelizing convolutional neural networks," *arXiv preprint arXiv:1404.5997*, 2014.

[12] NVIDIA Corporation, "NVML API Reference Manual," 2023. [Online]. Available: https://docs.nvidia.com/deploy/nvml-api/

[13] T. Ben-Nun and T. Hoefler, "Demystifying Parallel and Distributed Deep Learning: An In-Depth Concurrency Analysis," *ACM Computing Surveys*, vol. 52, no. 4, pp. 1–43, 2019.

[14] Y. You, J. Li, S. Reddi, J. Hseu, S. Kumar, S. Bhojanapalli, X. Song, J. Demmel, K. Keutzer, and C.-J. Hsieh, "Large batch optimization for deep learning: Training BERT in 76 minutes," in *Proc. ICLR*, 2020.

[15] A. Agrawal, A. Acharya, A. Goswami, P. Daga, A. Saraswat, and N. Devanur, "Accelerating large scale real-time GNN inference using channel pruning," *Proc. VLDB Endow.*, vol. 14, no. 9, 2021.

---

*Manuscript submitted for review. Code and data available upon request.*
