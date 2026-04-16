# Week 8 Experimental Results: Local Hardware + Adversarial Robustness

**Date:** April 10–11, 2026
**Hardware:** NVIDIA GeForce RTX 5080 (Blackwell, 16 GB), RTX 4050 Laptop (Ada Lovelace, 6 GB)
**Total new data:** 170+ parquet files across 2 local GPUs

---

## 1. Cross-GPU Generalization (5 Architectures)

### Setup

We trained and evaluated the three-way classifier (ml_training / ml_inference / other) across 5 GPU types spanning 4 NVIDIA architecture generations:

| GPU | Architecture | Generation | VRAM | Context |
|-----|-------------|------------|------|---------|
| A100 SXM4 40GB | Ampere | 2020 | 40 GB HBM2e | Cloud datacenter |
| H100 80GB HBM3 | Hopper | 2022 | 80 GB HBM3 | Cloud datacenter |
| H100 NVL | Hopper | 2022 | 94 GB HBM3 | Cloud datacenter (NVLink) |
| RTX 5080 | Blackwell (GB203) | 2025 | 16 GB GDDR7 | Local desktop |
| RTX 4050 Laptop | Ada Lovelace (AD107) | 2023 | 6 GB GDDR6 | Local laptop |

30-second sliding windows, XGBoost classifier, 159 features from Tier 1 NVML telemetry.

### Cross-GPU Transfer Matrix

| Train → Test | RTX 4050 | H100 NVL | H100 80GB | RTX 5080 | A100 |
|--------------|----------|----------|-----------|----------|------|
| **RTX 4050** | **100%** | 65.7% | 56.5% | 54.4% | 65.5% |
| **H100 NVL** | 72.4% | **100%** | 84.0% | 40.0% | 88.4% |
| **H100 80GB** | 59.9% | 72.7% | **100%** | 40.5% | 77.8% |
| **RTX 5080** | 55.8% | 65.6% | 77.7% | **100%** | 72.6% |
| **A100** | 60.3% | 77.1% | 93.5% | 56.3% | **100%** |
| **Pooled (all)** | | | | | **97.3%** |

### Key Findings

1. **Same-GPU accuracy: 100%** across all 5 GPU types. The classifier works perfectly when trained and tested on the same hardware.

2. **Cross-GPU transfer: 40–94%** with significant domain shift. The worst transfers involve the RTX 5080, which has very different clock frequencies and PCIe bandwidth characteristics from datacenter GPUs. The best cross-GPU transfer is A100 → H100 80GB (93.5%), as expected for GPUs in the same datacenter family.

3. **Pooled classifier: 97.3%** accuracy with 5-fold grouped cross-validation. Training on diverse hardware eliminates the domain shift problem. This is the practical deployment recommendation: train the monitoring classifier on data from multiple GPU architectures.

4. **RTX 4050 (Ada Lovelace) as holdout:** When no 4050 data is in the training set, other GPUs achieve 56–72% accuracy on 4050 workloads. The 6 GB VRAM and 60W TDP create a distinct telemetry regime (lower power, thermal throttling, smaller batches). Including even a small amount of 4050 data in training would resolve this.

### Feature Stability

The most architecture-stable features (low CV of means across 5 GPUs):
- Temperature percentiles (CV ≈ 0.08) — physics-constrained
- GPU utilization fft_period (CV ≈ 0.10) — workload structure, not hardware
- GPU utilization max/p95 (CV ≈ 0.14)

The most architecture-unstable features:
- PCIe throughput (CV > 1.7) — PCIe 5.0 x16 on RTX 5080 vs PCIe 4.0 on others
- Memory clock statistics (CV > 1.6) — GDDR7 vs HBM2e vs HBM3
- SM clock skew (CV > 2.3) — different boost clock behaviors

**Implication for robust classifiers:** Rely on utilization patterns, temperature dynamics, and memory growth rates rather than absolute clock/bandwidth values.

### Detection Latency

| Window size | Accuracy (pooled, 5 GPUs) | F1 |
|-------------|---------------------------|-----|
| **5 seconds** | **97.4%** | 0.964 |
| 10 seconds | 97.2% | 0.965 |
| 30 seconds | 97.1% | 0.965 |
| 60 seconds | 96.9% | 0.965 |
| 120 seconds | 97.0% | 0.952 |

Training can be detected within **5 seconds** of starting with 97.4% accuracy. Detection performance is remarkably stable across window sizes, with even sub-10-second windows achieving the same accuracy as 2-minute windows. This enables real-time GPU monitoring.

---

## 2. Leave-One-Strategy-Out Evaluation

### Motivation

In our adversary-defender iteration, the hardened classifier (trained on clean data + adversarial examples) achieved 100% detection across all 16 evasion strategies. However, this was partly because the classifier had seen examples of every strategy during training.

The leave-one-strategy-out (LOO) evaluation answers a more rigorous question: **if a completely new evasion technique appears that the defender has never seen, how likely is detection?**

### Method

For each of 12 strategy families, we:
1. Remove all examples of that family from the training set
2. Train the hardened classifier on clean data + all OTHER adversarial families
3. Test detection rate on the held-out family

### Results (class-balanced classifier)

*Note: Results below use class-balanced training (downsampled to equal class
counts per run_id). Previous results used imbalanced training that inflated
detection rates by biasing the classifier toward predicting "training."*

| Held-out Family | Detection Rate | Verdict |
|-----------------|---------------|---------|
| F: Interleave (30/50%) | **100%** | Fully detected unseen |
| K: Online learning | **100%** | Fully detected unseen |
| K_ddp: DDP online learning | **100%** | Fully detected unseen |
| K_ddp_accum: DDP grad accum | **100%** | Fully detected unseen |
| L_ddp_stagger: DDP staggered | **100%** | Fully detected unseen |
| N: Gradient accumulation | **100%** | Fully detected unseen |
| L_ddp: DDP diluted | **98.8%** | Near-perfect |
| M: Composite + memory-min | 94.4% | Strong |
| I: Stochastic interleave | 89.1% | Strong |
| A: Utilization modulation | 86.4% | Strong |
| G: Clock-throttled | 80.4% | Moderate |
| L: Diluted training | 78.3% | Moderate (N=10: 55.6%) |
| D: Temporal disruption | 68.2% | Partial evasion |
| J: PID util-match | 57.1% | Partial evasion |
| H: Mimicry (mining/cuFFT) | **43.3%** | **Evades** (cuFFT: 37.5%) |
| E: Memory-minimal | **39.3%** | **Evades** |
| Composite (idle sweep) | **29.0%** | **Evades** (0% at idle=40/60/80%) |
| B: Low-utilization | **16.7%** | **Evades** |

**Overall mean detection on unseen strategies: 68.3%**
**Families evading at <50%: 4/18** (B, E, H, composite)
**DDP families: 98.8-100%** (governance-relevant case remains robust)

### Key Findings

1. **DDP strategies remain robust at 98.8-100%.** All 4 multi-GPU distributed
   evasion families are detected even when completely unseen, confirming that
   NVLink/AllReduce signatures are very difficult to suppress. This is the
   governance-relevant case for frontier model training.

2. **4 of 18 families evade at <50% with balanced training.** The class-balanced
   classifier honestly reveals which strategies succeed:
   - B_low_util (16.7%): tiny batch training is classified as "other"
   - Composite idle-padding (29.0%): high idle fractions effectively evade
   - E_memory_minimal (39.3%): gradient checkpointing + aggressive GC
   - H_mimicry (43.3%): cuFFT mimicry (37.5%) is the hardest single strategy

3. **Previous results (81.6% mean, 0/12 evading) were inflated by class
   imbalance.** The training set had 766 training runs vs 49 inference runs.
   Without balanced resampling, the classifier predicted "training" for 75-100%
   of inference windows, inflating detection rates.

4. **Diluted training partially evades at low dilution.**
   - L_diluted_2 (50% training): 76.8% detected
   - L_diluted_5 (20% training): 81.0% detected
   - L_diluted_10 (10% training): 55.6% detected
   - L_diluted_20 (5% training): 100% detected (batch size memory effect)

5. **Evasion-resistant features (R3) substantially recover detection.** The
   physics-based features (cumulative energy, power-memory correlation) bring
   H_mimicry_cufft from 37.5% LOO to 98.2% R3, and E_memory_minimal from
   39.3% to 98.2%. This confirms the "involuntary witness" hypothesis.

### Evasion Tax Bound (class-balanced)

The controlled dilution sweep (ResNet-50, batch 128, RTX 5080) with balanced
classifiers reveals the honest evasion bound:

| Dilution N | Train% | LOO 5s | LOO 15s | Hardened 5s | Cost |
|-----------|--------|--------|---------|-------------|------|
| 1 (full)  | 100%   | 98.4%  | 100%    | 100%        | 1x   |
| 2         | 50%    | 98.4%  | 100%    | 100%        | 2x   |
| 3         | 33%    | 65.6%  | 96.7%   | 95.1%       | 3x   |
| 5         | 20%    | 86.9%  | 100%    | 100%        | 5x   |
| 10        | 10%    | 67.2%  | 86.7%   | 83.6%       | 10x  |
| 20        | 5%     | 32.8%  | 100%    | 75.4%       | 20x  |

**Key findings:**
- At 5-second windows, N=20 dilution evades the LOO classifier at 67% (32.8%
  detection). But extending observation to 15 seconds recovers 100% detection.
- The hardened classifier maintains 75%+ detection even at N=20, 5s windows.
- At 15-second windows, all dilution levels are detected at 87-100% (LOO) or
  90-100% (hardened).

**Evasion tax:** An adversary diluting to N=10 (10x cost) still faces 67-87%
detection depending on window size. Sustained covert training over hours or
days would accumulate enough positive detections to trigger investigation.

---

## 3. Summary of All Adversarial Strategies (3 Rounds)

### Round 1 (A–D): Basic Evasion Techniques
- **A (Util modulation):** Sleep between batches. 92.6% detected (naive), 100% (hardened).
- **B (Low-util):** Tiny batch size. 50% → 100%.
- **D (Temporal disruption):** Random batch sizes + CUDA noise. 11.1% → 100%.

### Round 1.5 (Composite): Parameterized Idle Sweep
- Pre-allocation + idle padding + timing noise + gradient accumulation.
- Sweep 0–95% idle fraction on RTX 5080, 3 reps each.
- Naive: 0–60% detection depending on idle fraction. Hardened: 100%.

### Round 2 (E–J): Targeted Counter-Strategies
- **E (Memory-minimal):** Gradient checkpointing + GC. 44.4% → 100%. LOO: 70.4%.
- **F (Interleave 30/50%):** Fixed-period training + mining/cuFFT. 0% → 100%. LOO: 100%.
- **G (Clock-throttled):** Locked low clocks + power limit. 0% → 100%. LOO: 100%.
- **H (Mimicry mining/cuFFT):** Feedback-controlled target matching. 0% → 100%. LOO: 90.7%.
- **I (Stochastic interleave):** Random phase lengths. 77.8% → 100%. LOO: 100%.
- **J (PID util-match):** PID controller targeting 75% util. 0% → 100%. LOO: 100%.

### Round 3 (K–L): Strongest-Possible Attacks
- **K (Online learning):** Inference + backward pass per batch. 0% → 100%. LOO: 100%.
- **L (Diluted):** Train on 1/N batches. LOO results: N=5 evades at 96.3%, N=10 at 88.9%.

### The Arms Race Outcome (class-balanced)

| Round | Attacker innovation | Baseline det. | R2 Hardened | R3 +Features | LOO det. |
|-------|---------------------|---------------|-------------|--------------|----------|
| 1 | Noise/padding (A,B,D) | 0-50% | 5-61% | 38-61% | 17-86% |
| 2 | Mimicry/PID (E-J) | 0-60% | 13-45% | 82-100% | 39-100% |
| 3 | Training-as-inference (K,L) | 57-100% | 84-100% | 100% | 56-100% |
| 4 | Composite+grad accum (M,N) | 100% | 100% | 100% | 78-100% |
| DDP | Multi-GPU distributed | 98-100% | 99-100% | 99-100% | **98-100%** |

The defender's advantage is strongest for multi-GPU distributed training
(98-100% across all conditions). Single-GPU evasion is a genuine challenge:
the balanced hardened classifier detects Round 2 strategies at only 13-45%,
but evasion-resistant physics-based features (R3) recover detection to 82-100%.
The LOO evaluation reveals that 4/18 strategy families evade at <50%, but
the physically-grounded features substantially narrow this gap.
