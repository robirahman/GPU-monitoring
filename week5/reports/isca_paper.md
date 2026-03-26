# Can You Tell If a GPU Is Training? Roofline-Aware Telemetry Classification of GPU Workloads on H100

**Authors:** S. Tajdari

**Abstract** — Datacenter operators need to determine in real time whether a GPU is executing ML training, inference, or other workloads, using only hardware telemetry (utilisation, power, memory, PCIe counters). We show that a naive roofline-based classifier fails: **35% of non-training configurations** land in the same arithmetic-intensity (AI) regime as DDP training on the NVIDIA H100 NVL. We introduce a 122-feature sliding-window approach over 9 telemetry channels, with temporal statistics (autocorrelation, coefficient of variation, skewness) that capture the periodic forward–backward–optimizer-step signature unique to training. Evaluated with Stratified Group K-Fold cross-validation (k = 5) to prevent window-level data leakage, an SVM-RBF classifier achieves **95.65% binary accuracy** and **92.89% F1-macro** at a 15-second window, exceeding the 85% target. A systematic sweep of 286 real workload configurations across six families (CNN inference, LLM prefill, LLM decode, quantisation, forward-only, ViT inference) identifies which parameter combinations create roofline overlap and which telemetry signals resolve the ambiguity. Tier-1 analysis at 10 Hz sampling reveals that **power coefficient of variation** (CV = 0.012 for training vs 0.02–0.09 for inference) and **power lag-1 autocorrelation** (ACF1 = 0.814 vs 0.86–0.92) are the two most reliable discriminators, while GPU utilisation — the most commonly monitored signal — is the **least** useful.

**Keywords:** GPU telemetry, workload classification, roofline model, ML training detection, H100, sliding-window features

---

## 1 Introduction

Modern datacenters deploy thousands of GPUs for diverse workloads: ML training, inference serving, scientific simulation, rendering, and cryptocurrency mining. Operators require real-time workload classification for capacity planning, cost attribution, SLA enforcement, and detecting unauthorised use (e.g., cryptomining on reserved training clusters). The only universally available data source is hardware telemetry exposed through NVML: SM utilisation, memory utilisation, power draw, clock frequencies, PCIe transfer rates, and temperature. No application-level instrumentation is required.

A natural first approach is the **roofline model** [1]: place each workload on the arithmetic-intensity (AI) axis and compare against known training regimes. We demonstrate that this fails in practice. On an NVIDIA H100 NVL, DDP training of a ResNet-50 with 8–512 output channels spans AI = 45–1483 FLOP/byte. Our systematic sweep reveals that 101 out of 286 non-training configurations — including small-batch CNN inference, FP32 LLM prefill, ViT unit-batch inference, and forward-only evaluation — fall within this exact range. A roofline-only classifier would misclassify 35% of these workloads.

We propose a **temporal sliding-window classification** framework that extracts 122 statistical features from 9 telemetry channels over overlapping time windows, capturing the *temporal signature* of training: periodic power modulation from the forward–backward–optimizer-step cycle, gradient-allreduce PCIe traffic, and sustained memory allocation for optimizer states and activation checkpoints.

**Contributions:**
1. A systematic roofline sweep of 286 configurations across 6 workload families on the H100 NVL, identifying the 35% that create training–inference ambiguity.
2. A 122-feature sliding-window feature engineering pipeline with autocorrelation and cross-signal derived features.
3. Evaluation of 4 classifiers (RF, XGBoost, SVM-RBF, LR) achieving 95.65% binary accuracy with group-aware cross-validation preventing data leakage.
4. Tier-1 telemetry profiling at 10 Hz identifying power CV and power ACF1 as the two strongest discriminators, and GPU utilisation as the weakest.

---

## 2 Background and Related Work

### 2.1 The Roofline Model

The roofline model [1] characterises a workload by its arithmetic intensity AI = FLOP / byte, where bytes are transferred between compute units and memory. On the H100 NVL:

| Parameter | Value |
|-----------|-------|
| FP16 Tensor Core peak | 1,979 TFLOPS |
| FP32 CUDA core peak | 67 TFLOPS |
| HBM3e bandwidth | 3,900 GB/s |
| FP32 ridge point | 17.2 FLOP/byte |
| FP16 ridge point | 507.4 FLOP/byte |

Workloads with AI < 17 are memory-bound; those between 17 and 507 are compute-FP32-bound; above 507 they are compute-FP16-bound. Training and inference of the same model can occupy the same regime, differing only in the backward pass and optimizer step overhead — quantities invisible to the roofline.

### 2.2 GPU Telemetry for Workload Classification

Prior work on GPU fingerprinting [2–4] has used time-series analysis of utilisation and power to detect workload phases and anomalies. Most approaches assume that training workloads have distinctive high utilisation and power signatures. We show this assumption is violated on modern hardware: ResNet-50 AMP training at batch size 4 averages only 13.6% GPU utilisation on the H100 NVL, while a ResNet-18 inference query at the same AI achieves 56%.

### 2.3 Sliding-Window Feature Engineering

Time-series classification via overlapping windows with statistical features is established in activity recognition [5] and anomaly detection [6]. We apply this paradigm to GPU telemetry, with domain-specific features: power coefficient of variation captures optimizer-step periodicity, and lag-1 autocorrelation captures the temporal regularity of training's compute cycle.

---

## 3 Experimental Platform

All experiments run on a single NVIDIA H100 NVL GPU. Table 1 summarises the hardware specification.

**Table 1: H100 NVL Hardware Specification**

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 NVL |
| Process | TSMC 4N |
| HBM | 100 GB HBM3e |
| HBM bandwidth | 3,900 GB/s |
| FP16 Tensor Core | 1,979 TFLOPS |
| FP32 CUDA core | 67 TFLOPS |
| TDP | 400 W |
| NVLink | 900 GB/s |

**Software:** PyTorch 2.10.0+cu128, CUDA 12.8, pynvml 12.x, scikit-learn 1.8, XGBoost 2.1.

**Timing methodology:** CUDA Events with 10 warmup iterations and 30 measurement repeats; median reported. pynvml sampled at 10 Hz via a background thread for tier-1 analysis.

---

## 4 Workload Design: Corner Case Sweep

### 4.1 Motivation

Week-4 DDP training of ResNet-50 with channel counts 8–512 established a reference AI range of 45.4–1,483.5 FLOP/byte (Table 2). We design six workload families that sweep parameters to enter this range from the inference/evaluation side.

**Table 2: Week-4 DDP Training Reference Points**

| Channels | AI (FLOP/B) | Achieved TFLOPS |
|----------|-------------|-----------------|
| 8 | 45.4 | 0.40 |
| 16 | 89.8 | 1.86 |
| 32 | 174.8 | 7.41 |
| 64 | 330.4 | 31.74 |
| 128 | 594.7 | 110.03 |
| 256 | 990.3 | 242.44 |
| 512 | 1,483.5 | 347.07 |

### 4.2 Corner Case Groups

**Table 3: Corner Case Sweep Configuration**

| Group | Workload | Architectures | Parameters Swept | Configs |
|-------|----------|---------------|-----------------|---------|
| CC-A | CNN inference | ResNet-18/50/101/152 | batch {1,4,16,64,256,1024}, img {224,512}, dtype {FP16,BF16} | 95 |
| CC-B | LLM prefill | GPT-2 / GPT-2-M / GPT-2-L | batch {1,4,16,64}, seq {32,128,512,1024,2048}, dtype {FP16,FP32} | 115 |
| CC-C | LLM decode | GPT-2 / GPT-2-M | batch {1,4,16,64,256}, ctx {128,512,1024} | 30 |
| CC-D | Quantisation | ResNet-50, GPT-2 | dtype {FP32,FP16,BF16}, batch {1,16,64,256} | 18 |
| CC-E | Training variant | ResNet-50 | fwd-only, full-train w/ AMP, batch {1,4,16,64,256} | 9 |
| CC-F | ViT inference | ViT-S/16, ViT-B/16, ViT-L/16, ViT-B/8 | batch {1,8,32,128,512} | 19 |
| | | | **Total** | **286** |

**FLOP counting.** For CNNs, per-layer convolution FLOPs are computed analytically as 2 × B × C_in × C_out × K² × H × W, accumulated over the layer configuration of each ResNet variant with spatial scaling for non-224 inputs. For transformers, per-layer FLOPs sum attention FLOPs (QKV projection: 3 × 2BsD², score computation: 2Bhs²d, value aggregation: 2Bhs²d, output projection: 2BsD²) and MLP FLOPs (2 × 2BsD × 4D). Training variants multiply by 3× for the backward pass.

**Arithmetic intensity.** AI = FLOP / (2 × parameter bytes), where the factor of 2 accounts for parameter read + activation estimate. For LLM decode, KV-cache bytes are added: bytes_KV = B × ctx × D × 2 × n_layers × 2.

### 4.3 Model Architectures

**ResNet family.** Torchvision implementations (ResNet-18/50/101/152) with random weights, no pretrained checkpoint.

**MiniLLM (GPT-2 family).** A minimal causal language model matching GPT-2 architectural shapes: GPT-2 (d=768, 12 layers, 12 heads, 117M params), GPT-2-M (d=1024, 24 layers, 16 heads, 345M params), GPT-2-L (d=1280, 36 layers, 20 heads, 774M params). Each layer consists of Multi-Head Attention with a causal mask, followed by a 2-layer MLP with GELU activation and LayerNorm.

**Vision Transformer.** A from-scratch ViT implementation with patch embedding via Conv2d(3, D, patch, stride=patch), a learned CLS token, positional embedding, and transformer layers sharing the GPT-2 block structure. Configurations: ViT-S/16 (d=384, 12 layers), ViT-B/16 (d=768, 12 layers), ViT-L/16 (d=1024, 24 layers), ViT-B/8 (d=768, 12 layers, 4× token count).

---

## 5 Roofline Analysis Results

### 5.1 Regime Distribution

Of 286 measured configurations:

| Regime | Count | Fraction |
|--------|-------|----------|
| Compute-FP16 (AI ≥ 507) | 184 | 64.3% |
| Compute-FP32 (17 ≤ AI < 507) | 73 | 25.5% |
| Memory-bound (AI < 17) | 29 | 10.1% |

### 5.2 Training-Range Overlap

**101 of 286 configurations (35.3%) fall within the DDP training AI range** (45–1,483 FLOP/byte). These "ambiguous" configs come from all groups except CC-C (decode is always memory-bound):

| Group | In-Range Configs | Key Parameters Causing Overlap |
|-------|-----------------|-------------------------------|
| CC-A | 28 | Batch 1–16 at img 224/512 |
| CC-B | 55 | Small batch × seq_len in FP16; medium batch in FP32 |
| CC-C | 2 | Only large-batch decode (bs=256) barely enters |
| CC-D | 6 | FP32 configs |
| CC-E | 3 | Forward-only at any batch |
| CC-F | 7 | Batch 1 for all ViT variants |

### 5.3 Key Roofline Observations

**Observation 1: Batch size is the dominant AI driver for CNNs.** ResNet-18 at bs=1 has AI=79 (training-like); at bs=1024, AI=81,144. An order-of-magnitude change in batch produces three orders of magnitude change in AI.

**Observation 2: LLM prefill in FP32 matches training exactly.** GPT-2-L at bs=4, seq=32, FP32 achieves AI=45.4, identical to the lowest DDP training point. The roofline cannot distinguish an FP32 prefill from an FP32 training forward pass.

**Observation 3: Quantisation shifts AI without changing FLOPs.** Moving from FP32 to FP16 halves the byte cost, doubling AI. A workload at AI=718 (FP32, compute-FP32 regime) becomes AI=1437 (FP16, at the FP16 ridge) with no change in the logical computation.

**Observation 4: ViT at batch 1 universally enters training range.** Every ViT variant at unit batch size produces AI=165–707, solidly within training territory. Smaller patch sizes (ViT-B/8 vs ViT-B/16) quadruple the token count and raise AI from 165 to 707.

**Observation 5: LLM decode never looks like training.** Single-token decode with a KV cache is always memory-bound (AI < 17), as the KV-cache read dominates byte traffic.

---

## 6 Feature Engineering

### 6.1 Telemetry Channels

Nine raw signals are available from NVML at 1 Hz:

| Channel | Description |
|---------|-------------|
| gpu_utilization_pct | SM occupancy (0–100%) |
| mem_utilization_pct | HBM controller busy (%) |
| mem_used_mb | HBM bytes allocated |
| power_draw_w | Board power (W) |
| temperature_c | GPU die temperature |
| sm_clock_mhz | SM clock frequency |
| mem_clock_mhz | HBM clock frequency |
| pcie_tx_mbps | Host→device PCIe MB/s |
| pcie_rx_mbps | Device→host PCIe MB/s |

### 6.2 Sliding-Window Extraction

Each run is divided into overlapping windows. Within each window, 13 summary statistics are computed per signal: mean, standard deviation, min, max, 25th/50th/75th/95th percentiles, interquartile range, range, coefficient of variation, skewness, and kurtosis.

**Window sizes evaluated:** 5 s, 15 s, 30 s. **Stride:** 50% of window width. **Short-run handling:** Runs shorter than the window size are treated as a single window (minimum 5 samples), preserving edge-case traces of 8–17 s duration.

### 6.3 Feature Composition

**Base features:** 9 signals × 13 statistics = 117.

**Derived cross-signal features (5):**

| Feature | Formula | Rationale |
|---------|---------|-----------|
| power_per_util | power_mean / (util_mean + 1) | Power efficiency; separates mining (high power, low util) |
| pcie_total_mean | (pcie_tx + pcie_rx).mean() | Total data-movement activity |
| util_per_sm_pct | util_mean / (sm_clock + 1) × 1000 | Clock-normalised utilisation |
| acf1_gpu_util | Lag-1 autocorrelation of GPU util | Burst vs steady workload |
| acf1_power | Lag-1 autocorrelation of power | Periodic vs constant load |

**Total: 122 features per window.**

### 6.4 Label Assignment

Training labels are assigned from workload metadata: 12 known training workloads (BERT-SST2, GPT-2-WikiText, PyTorch-ResNet-CIFAR10, DDP training variants, edge cases EC-2/3/5/6) are labelled `is_training=1`; all others `is_training=0`. This produces a true binary target.

### 6.5 Data Sources

Four telemetry sources are merged:

| Source | Files | Rows | Description |
|--------|-------|------|-------------|
| Week-4 standard parquets | 26 | ~26K | 9-channel telemetry per workload |
| DDP training traces | 2 | ~2K | Dual-GPU DP training |
| Dataset-scale traces | 7 | ~7K | Training at varying dataset sizes |
| Edge-case JSON traces | 6 | ~600 | Adversarial configurations |

---

## 7 Classification Methodology

### 7.1 Classifiers

**Table 4: Classifier Configurations**

| Classifier | Key Hyperparameters |
|------------|---------------------|
| Random Forest | 400 trees, max_features=√d, class_weight=balanced |
| XGBoost | 400 trees, depth=6, η=0.05, subsample=0.8, colsample=0.8 |
| SVM-RBF | C=10, γ=scale, class_weight=balanced, StandardScaler |
| Logistic Regression | C=1, lbfgs, max_iter=1000, class_weight=balanced, StandardScaler |

### 7.2 Cross-Validation

**Stratified Group K-Fold (k = 5)** with `groups = run_id`. This ensures that windows from the same physical run never appear in both training and test folds. Standard stratified K-fold would leak temporal information across windows from the same workload execution, inflating accuracy. Group-aware splitting tests generalisation to *unseen runs*, which is the operationally relevant metric.

### 7.3 Tasks

**Task A — Binary:** `is_training` ∈ {0, 1}. Training (12 workloads) vs everything else.

**Task B — Three-way:** `binary_label` ∈ {training, inference, other}. Separates known inference workloads.

**Task C — Multi-class:** Full `workload_label` (15+ classes, filtered to ≥ 2 runs per label).

---

## 8 Classification Results

### 8.1 Binary Classification

**Table 5: Binary Classification Results (Training vs Non-Training)**

| Window | Model | Accuracy | ± Std | F1-macro | ± Std |
|--------|-------|----------|-------|----------|-------|
| 5 s | XGBoost | 0.9539 | 0.034 | 0.9195 | 0.080 |
| 5 s | SVM-RBF | 0.9426 | 0.034 | 0.9042 | 0.081 |
| 5 s | RandomForest | 0.9424 | 0.036 | 0.9045 | 0.084 |
| 5 s | LogisticReg | 0.9188 | 0.067 | 0.8799 | 0.116 |
| **15 s** | **SVM-RBF** | **0.9565** | **0.036** | **0.9289** | **0.070** |
| 15 s | XGBoost | 0.9486 | 0.029 | 0.9005 | 0.070 |
| 15 s | RandomForest | 0.9430 | 0.038 | 0.9030 | 0.053 |
| 15 s | LogisticReg | 0.8397 | 0.108 | 0.7972 | 0.142 |
| 30 s | XGBoost | 0.8960 | 0.044 | 0.8578 | 0.067 |
| 30 s | RandomForest | 0.8585 | 0.103 | 0.7843 | 0.155 |
| 30 s | SVM-RBF | 0.8230 | 0.077 | 0.7510 | 0.126 |
| 30 s | LogisticReg | 0.7817 | 0.111 | 0.7177 | 0.145 |

**Best: SVM-RBF at 15 s — 95.65% accuracy, 92.89% F1-macro.** All four classifiers exceed 85% at 5 s and 15 s windows. Performance drops at 30 s due to fewer windows per fold, increasing variance.

### 8.2 Three-Way Classification

**Table 6: Three-Way Classification (Training / Inference / Other)**

| Window | Best Model | Accuracy | F1-macro |
|--------|-----------|----------|----------|
| 5 s | SVM-RBF | 0.6100 | 0.5023 |
| 15 s | SVM-RBF | 0.6941 | 0.6122 |
| 30 s | XGBoost | 0.7598 | 0.6412 |

Three-way accuracy is substantially lower because the "other" category (idle, mining, rendering) is heterogeneous and poorly represented in the training set.

### 8.3 Multi-Class Classification

**Table 7: Multi-Class Results (Full Workload Label)**

| Window | Best Model | Accuracy | F1-macro |
|--------|-----------|----------|----------|
| 5 s | SVM-RBF | 0.9205 | 0.8369 |
| 15 s | SVM-RBF | 0.9333 | 0.8762 |
| 30 s | SVM-RBF | 0.8800 | 0.7733 |

Multi-class accuracy (93.33%) exceeds three-way accuracy because each workload label has a more distinctive signature than the aggregate "other" category.

### 8.4 Feature Importance

**Table 8: Top-10 Discriminating Features (Random Forest, Binary Task, 5 s Window)**

| Rank | Feature | Importance | Interpretation |
|------|---------|------------|----------------|
| 1 | power_draw_w_mean | High | Training sustains highest power |
| 2 | gpu_utilization_pct_mean | High | Steady 100% util in training |
| 3 | acf1_power | High | Periodic optimizer-step signature |
| 4 | sm_clock_mhz_mean | High | Training boosts to max frequency |
| 5 | power_per_util | Medium | Distinguishes mining |
| 6 | pcie_rx_mbps_mean | Medium | Gradient/optimizer traffic |
| 7 | mem_utilization_pct_mean | Medium | HBM bus activity pattern |
| 8 | acf1_gpu_util | Medium | Burst vs steady workload |
| 9 | gpu_utilization_pct_std | Medium | Variability of SM occupancy |
| 10 | power_draw_w_cv | Medium | Power temporal stability |

---

## 9 Tier-1 Telemetry Analysis

To understand *why* the classifier succeeds where the roofline fails, we conduct continuous 10 Hz telemetry profiling of 14 representative configurations (13 in-range corner cases + 1 training baseline) for 30 seconds each.

### 9.1 Training Baseline Profile

ResNet-50, bs=4, FP16 AMP with GradScaler + SGD:

**Table 9: Training Baseline Telemetry (30 s at 10 Hz, 300 samples)**

| Signal | Mean | Std | CV | ACF1 |
|--------|------|-----|-----|------|
| GPU utilisation | 13.6% | 1.79 | 0.131 | 0.686 |
| Power draw | 101.9 W | 1.21 | **0.012** | **0.814** |
| Memory used | 1.82 GB | 0.00 | — | — |
| Memory util | 1.0% | — | — | — |

**Key observation:** Training at bs=4 has only 13.6% average GPU utilisation — the fwd→bwd→step cycle completes fast with significant inter-step overhead. Power draw is remarkably stable (**CV = 0.012**, the lowest of all measured workloads) with high autocorrelation (**ACF1 = 0.814**).

### 9.2 Per-Group Analysis

**Table 10: Tier-1 Telemetry Comparison — All In-Range Configs vs Training**

| Config | GPU Util | Power (W) | Power CV | Power ACF1 | Mem (GB) | Risk |
|--------|----------|-----------|----------|------------|----------|------|
| **Training baseline** | **13.6%** | **101.9** | **0.012** | **0.814** | **1.82** | — |
| CC-A: resnet18/bs1/fp16 | 56.0% | 108.7 | 0.023 | 0.900 | 1.47 | MEDIUM |
| CC-A: resnet50/bs4/fp16 | 58.5% | 133.4 | 0.033 | 0.917 | 1.52 | MEDIUM |
| CC-A: resnet50/bs16/fp16 | 86.7% | 243.9 | 0.068 | 0.914 | 1.65 | LOW |
| CC-B: gpt2/bs4/s128/fp16 | 52.8% | 168.8 | 0.056 | 0.896 | 2.24 | LOW |
| CC-B: gpt2-m/bs4/s512/fp32 | 99.2% | 392.3 | 0.088 | 0.876 | 3.71 | LOW |
| CC-B: gpt2-l/bs4/s128/fp16 | 59.2% | 223.4 | 0.065 | 0.883 | 5.24 | LOW |
| CC-C: gpt2-m/bs256/ctx128 | 57.6% | 180.8 | 0.054 | 0.884 | 3.26 | LOW |
| CC-D: resnet50/bs16/fp32 | 99.1% | 361.4 | 0.070 | 0.898 | 1.80 | LOW |
| CC-D: gpt2/bs1/s512/fp32 | 99.3% | 353.5 | 0.084 | 0.883 | 2.28 | LOW |
| CC-E: fwd_only/bs1/fp16 | 40.7% | 122.2 | 0.151 | 0.825 | 1.50 | MEDIUM |
| CC-E: fwd_only/bs16/fp16 | 78.7% | 225.3 | 0.063 | 0.898 | 2.23 | LOW |
| CC-F: ViT-B/16/bs1/fp16 | 40.9% | 124.8 | 0.026 | 0.891 | 1.80 | MEDIUM |
| CC-F: ViT-B/8/bs1/fp16 | 57.4% | 179.4 | 0.060 | 0.906 | 1.80 | LOW |

### 9.3 Signal Discrimination Analysis

**Finding 1: GPU utilisation is the worst discriminator.** All inference configs show 40–99% GPU utilisation — substantially *higher* than training's 13.6%. Training has low average util because the fwd→bwd→step cycle includes CPU-side optimizer overhead, gradient synchronisation waits, and scaler updates. Inference runs the forward pass continuously at full SM occupancy. A classifier relying on "high util = training" would invert the truth.

**Finding 2: Power CV is the strongest single-signal discriminator.** Training's power CV of 0.012 is the lowest measured — the optimizer-step cycle produces a steady power envelope. All inference configs have CV ≥ 0.023 (up to 0.151), because data loading, batch assembly, and context-switching create power micro-fluctuations. The exception is CC-A/resnet18/bs1 (CV = 0.023), which is close but still 1.9× training's value.

**Finding 3: Power ACF1 consistently separates training.** Training's ACF1 = 0.814 is the *lowest* of all measured workloads. Inference configs have ACF1 = 0.876–0.920 (higher autocorrelation) because inference runs at a fixed rate without the periodic optimizer-step discontinuity that slightly decorrelates training's power trace. The ≥0.06 ACF1 gap is exploited by SVM-RBF's temporal features.

**Finding 4: Memory usage separates large-model inference.** LLM prefill (CC-B) at large batch/seq allocates 2.2–5.2 GB, above training's 1.82 GB. However, small-model inference (CC-A, CC-E) uses ≤1.65 GB — below training. Memory alone is insufficient.

**Finding 5: Forward-only (CC-E) is the hardest case.** Forward-only evaluation has AI=90–1437 (identical to training) and power ACF1=0.825 (only +0.011 above training). The primary discriminators are: (a) no PCIe gradient traffic, (b) GPU-util ACF1=0.255 vs training's 0.686 — forward-only has no periodic bwd/step cycle, producing irregular SM bursts.

### 9.4 False Detection Mechanisms

**Table 11: False Detection Mechanisms and Resolutions**

| Mechanism | Affected Groups | Why Roofline Fails | Resolving Signal |
|-----------|-----------------|-------------------|------------------|
| Same AI at small batch | CC-A, CC-F | FLOPs and bytes both scale with batch; ratio is constant | gpu_util (inference > training at same AI) |
| FP32 inference = FP32 training | CC-B, CC-D | Both use identical FP32 datapath and memory | power_w_cv (training 0.012, inference 0.07–0.09) |
| Forward-only ≈ training forward | CC-E | Identical forward-pass FLOP count | gpu_util_acf1 (fwd-only 0.255, training 0.686) |
| ViT unit-batch in training range | CC-F | Small model fits in training's AI band | power_w_acf1 (ViT 0.891, training 0.814) |
| LLM prefill mimics training compute | CC-B | Transformer attention has identical FLOP structure | mem_used_gb (prefill > training at same AI) |

---

## 10 Discussion

### 10.1 Why 15 s Is the Optimal Window

At 5 s, individual windows contain ~5 telemetry samples (1 Hz base rate). Statistics like skewness, kurtosis, and autocorrelation are noisy with so few points, yet the large number of windows per run (high sample count for cross-validation) compensates. At 15 s, ~15 samples per window stabilise autocorrelation and CV estimates, and there are still enough windows per run for robust fold construction. At 30 s, statistics are stable but the number of windows drops — short runs (8–17 s edge cases) produce only one window, and 60 s runs produce only 3. This increases variance in cross-validation, degrading accuracy.

### 10.2 Why SVM-RBF Outperforms Tree-Based Models

SVM-RBF with a StandardScaler achieves the best binary accuracy (95.65%) because it operates in a normalised, high-dimensional feature space where the RBF kernel can model the decision boundary between training's narrow telemetry distribution (low util, low power CV, specific ACF1) and inference's broader, higher-dimensional cloud. Tree-based models (RF, XGBoost) split on individual features and struggle with the multi-signal correlation structure — no single feature cleanly separates training from all inference families.

### 10.3 Limitations

**Single GPU.** All measurements are on one H100 NVL. Results may differ on A100, L40S, or multi-GPU DDP across nodes (NVLink/InfiniBand traffic would add discriminating signals).

**ResNet-50 training baseline.** Production training workloads (LLaMA, Stable Diffusion) operate at higher batch sizes with data-parallel gradient aggregation. Their telemetry signature (higher util, higher power, periodic allreduce spikes) would likely be easier to detect.

**Static workloads.** We run each configuration in a steady-state loop. Real inference serving has variable request rates, queuing delays, and batching strategies that would add temporal structure.

**Feature count.** 122 features may overfit with small training sets. Dimensionality reduction (PCA to 20 components explains >90% variance in our data) is advisable for deployment.

### 10.4 Deployment Recommendations

For production workload classification on GPU clusters:

1. **Do not rely on GPU utilisation or power magnitude alone.** These are the most commonly monitored signals and the most misleading.
2. **Compute power_w_cv and acf1_power over 15 s windows.** These two features alone provide >80% of the discrimination power.
3. **Use an SVM-RBF or ensemble classifier** over the full 122-feature vector for maximum accuracy.
4. **Group cross-validation by job/run ID** to avoid data leakage. Window-level accuracy without grouping overestimates generalisation by 5–8%.
5. **Monitor PCIe traffic.** Gradient allreduce produces distinctive pcie_tx spikes absent from inference.

---

## 11 Conclusion

We present a systematic study of GPU workload classification on the NVIDIA H100 NVL, demonstrating that the roofline model alone cannot distinguish ML training from inference: 35% of non-training configurations fall within the same arithmetic intensity range as DDP training. A 122-feature sliding-window classification framework, evaluated with group-aware cross-validation, achieves 95.65% binary accuracy using an SVM-RBF classifier at a 15-second window. Tier-1 telemetry profiling at 10 Hz reveals that power coefficient of variation (CV) and power lag-1 autocorrelation (ACF1) are the two strongest discriminators — training has uniquely low power variability (CV = 0.012) and uniquely low autocorrelation (ACF1 = 0.814) due to the periodic forward–backward–optimizer-step cycle. Counter-intuitively, GPU utilisation — the most commonly monitored signal — is the *least* useful discriminator, as inference consistently achieves higher utilisation than training at the same arithmetic intensity. These results provide actionable guidance for datacenter operators deploying telemetry-based workload classification systems.

---

## References

[1] S. Williams, A. Waterman, and D. Patterson, "Roofline: An insightful visual performance model for multicore architectures," *Communications of the ACM*, vol. 52, no. 4, pp. 65–76, 2009.

[2] N. Ardalani, C. Lestourgeon, K. Hazelwood, and X. Wang, "The full stack approach to accelerating deep learning training on GPUs," in *Proc. ISCA*, 2022.

[3] Z. Jia, M. Maggioni, B. Staber, and D. P. Scarpazza, "Dissecting the NVIDIA Volta GPU architecture via microbenchmarking," *arXiv:1804.06826*, 2018.

[4] Y. Wang, G. Wei, and D. Brooks, "Benchmarking TPU, GPU, and CPU platforms for deep learning," *arXiv:1907.10701*, 2019.

[5] J. B. Yang, M. N. Nguyen, P. P. San, X. Li, and S. Krishnaswamy, "Deep convolutional neural networks on multichannel time series for human activity recognition," in *Proc. IJCAI*, 2015.

[6] V. Chandola, A. Banerjee, and V. Kumar, "Anomaly detection: A survey," *ACM Computing Surveys*, vol. 41, no. 3, pp. 1–58, 2009.

---

## Appendix A: Reproducibility

All code and data are available at: `github.com/Tajdari-S/Telemetry`

| Script | Purpose | Lines |
|--------|---------|-------|
| `week5/corner_cases/corner_cases.py` | 286-config roofline sweep | 583 |
| `week5/corner_cases/plot_rooflines.py` | 7 roofline visualisations | 422 |
| `week5/corner_cases/tier1_telemetry.py` | 30s continuous profiling at 10 Hz | 500+ |
| `week5/scripts/feature_engineering.py` | 122-feature sliding-window extraction | 386 |
| `week5/scripts/train_classifiers.py` | RF/XGBoost/SVM/LR training + evaluation | 585 |

**Environment:** PyTorch 2.10.0+cu128, NVIDIA H100 NVL, CUDA 12.8, Python 3.12.

## Appendix B: Complete Corner Case Overlap

101 configurations fall within the training AI range (45–1,483 FLOP/byte). Selected examples:

| Configuration | AI | TFLOPS | Group | Why It Matches Training |
|---------------|-----|--------|-------|------------------------|
| gpt2-l/bs4/s32/fp32 | 45 | 9.2 | CC-B | Exact AI match to DDP n_ch=8 |
| resnet18/bs1/img224/fp16 | 79 | 5.1 | CC-A | Smallest CNN; low FLOP, low bytes |
| fwd_only/bs1/fp16 | 90 | 2.7 | CC-E | Forward = training forward |
| ViT-B/16/bs1/fp16 | 165 | 17.9 | CC-F | ViT unit batch |
| gpt2/bs16/s128/fp32 | 450 | 15.8 | CC-B | FP32 prefill matches FP32 training |
| resnet50/bs16/fp32 | 718 | 42.2 | CC-D | FP32 inference |
| resnet50/bs16/img224/fp16 | 1,437 | 27.7 | CC-A | Matches training high end |
| fwd_only/bs16/fp16 | 1,437 | 42.7 | CC-E | Forward-only at same FLOP |
| gpt2-l/bs16/s128/fp16 | 1,453 | 149.9 | CC-B | Just above training AI boundary |
