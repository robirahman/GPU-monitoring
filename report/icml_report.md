# The Evasion Tax: Why GPU Telemetry Monitoring Favors Defenders in AI Compute Governance

**Robi Rahman** (MIRI) and **Sabiha Tajdari** (UVA)

TAIGR Workshop @ ICML 2026 — 8 pages + appendices — Deadline: April 24, 2026

---

## Working Abstract (~150 words)

Hardware-enabled monitoring of GPU workloads is a cornerstone of proposals for
AI compute governance. But if adversaries can defeat monitoring mechanisms, such
schemes are unworkable. We evaluate the adversarial robustness of GPU workload
classification using only zero-overhead Tier 1 NVML telemetry. Across N rounds
of adversary-defender iteration with [TBD: final count] evasion strategies on
5 GPU architectures (A100, H100 80GB, H100 NVL, RTX 5080, RTX 4050), we find
that (1) a pooled XGBoost classifier achieves 97.3% three-way accuracy and
detects training within 5 seconds; (2) hardened classifiers retrained with
adversarial examples achieve 100% detection against all known attacks; and
(3) the strongest unseen evasion—diluted training—requires sacrificing at least
[TBD]% of training throughput, multiplying training costs by [TBD]x. Using
published cost estimates for frontier models, we show this evasion tax renders
covert large-scale training economically unviable, supporting the feasibility
of telemetry-based compute governance.

---

## 1. Introduction (~1 page)

**Opening:** AI compute governance proposals increasingly depend on the ability
to verify what workloads run on regulated hardware [Baker et al. 2025,
Aarne et al. 2024]. If monitoring can be systematically evaded, the regulatory
scheme collapses. We ask: *how robust is GPU telemetry monitoring against a
sophisticated adversary, and what is the minimum cost of evasion?*

**Key claims (paragraph each):**

- Zero-overhead GPU telemetry (NVML Tier 1) is sufficient to classify workloads
  as training, inference, or other with 97%+ accuracy across 5 GPU architectures.

- Through N rounds of adversary-defender iteration, we designed [TBD] evasion
  strategies. Every strategy is defeated once the defender retrains with examples.
  Leave-one-strategy-out evaluation shows 81.6% detection on *completely unseen*
  attack types.

- The strongest unseen evasion (diluted training) imposes an **evasion tax**:
  to avoid detection, an adversary must reduce training throughput by [TBD]%,
  multiplying wall-clock time and cost proportionally. Using cost estimates from
  Rahman (2024), this makes covert frontier training economically unviable.

**Contributions (bulleted list):**
1. A classifier achieving 97.3% pooled accuracy across 5 GPU architectures with
   5-second detection latency, using only zero-overhead telemetry.
2. Systematic adversarial evaluation: [TBD] evasion strategies across N rounds,
   with leave-one-strategy-out generalization testing.
3. Quantification of the **evasion tax**: the minimum throughput sacrifice
   required to evade detection, with economic cost estimates for frontier models.
4. [TBD: wall-power validation / datacenter adversarial results / HEM recommendations]

---

## 2. Background & Threat Model (~0.75 page)

### 2.1 GPU Telemetry Tiers

- **Tier 1 (NVML/pynvml):** GPU utilization, memory usage, power draw,
  temperature, SM/memory clocks, PCIe bandwidth. Polled at 1 Hz. Zero runtime
  overhead. Always available on all NVIDIA GPUs.
- **Tier 2 (DCGM):** Datacenter-only. Adds tensor core activity, NVLink
  throughput, ECC errors. Requires driver configuration.
- **Tier 3 (Nsight Compute):** Per-kernel PMC profiling. 1,200-5,300% overhead.
  Impractical for continuous monitoring.
- Our classifier uses **Tier 1 only** — the minimum viable signal.

### 2.2 Threat Models

**Threat model A — Cloud tenant:** A user renting GPU time attempts to disguise
ML training as inference or non-ML workloads to evade a cloud provider's
monitoring. The adversary controls user-space software but not the host OS,
hypervisor, or hardware. The defender (cloud provider or regulator) has access
to hardware-level telemetry.

**Threat model B — Datacenter operator:** A datacenter operator attempts to hide
unauthorized AI training from a regulator. The adversary controls the full
software stack and can modify workload scheduling, but cannot tamper with
hardware-level monitoring (assumes tamper-resistant HEM per Aarne et al. 2024).

**Discussion:** Threat model B is strictly harder — the operator can deploy any
software-level evasion and also control scheduling. Our evasion strategies
primarily target model A but are also relevant to B, since the fundamental
constraint (backward pass energy overhead) is physical, not software.

### 2.3 Classification Task

Three-way: **ml_training** / **ml_inference** / **other** (HPC, mining,
rendering, idle). Sliding windows of 5-60 seconds over 1 Hz telemetry.
159 statistical features per window.

---

## 3. Related Work (~0.5 page)

- **Compute governance:** Shavit (2023) proposes compute monitoring for
  verifying rules on large-scale training — the foundational paper for this
  line of work. Sastry, Heim et al. (2024) provide a comprehensive framework
  for compute as a governance lever. Baker et al. (2025) identify six
  verification layers for international AI agreements; our work provides
  empirical evidence for one layer (built-in monitoring). O'Gara et al. (2025)
  and the RAND workshop report (Kulp, Siu & Heim 2025) evaluate HEM options
  including workload attestation. Petrie (2024) proposes firmware-based
  offline licensing for export controls.

- **Workload fingerprinting:** WAVE [Xu et al. 2026] identifies inference
  workloads via 24 Nsight Compute PMCs (87-93% accuracy) but requires Tier 3
  profiling with 1,200-5,300% overhead. Bhatele et al. (IPDPS 2026) characterize
  production GPU workloads using DCGM counters at scale. We achieve comparable
  accuracy using only Tier 1 (zero overhead).

- **GPU telemetry for AI governance:** Monfared et al. (2026) study timing
  and memory observables for detecting training, even when host and device are
  untrusted — the most directly related work. Their focus is on what signals
  remain informative under a strong adversary model; ours is on quantifying the
  *cost* of evasion through iterative adversary-defender refinement across
  multiple GPU architectures. The approaches are complementary.
  Chen et al. (2025, "Reveal") propose unsupervised anomaly detection from
  hardware telemetry across 30+ ML models.

- **GPU kernel validation:** ShadowScope [Jackowski et al. 2025] validates GPU
  kernel behavior via CUPTI counters. Targets kernel-level compliance rather
  than workload-level classification.

- **Physical side channels:** Maia et al. (USENIX Security 2022) show magnetic
  flux from GPU power cables reveals neural network topology. Arefin &
  Serwadda (CODASPY 2025) demonstrate USB/HDMI port power measurement can infer
  GPU activities. Latif et al. (2024) empirically measure power signatures of
  training on 8-GPU H100 nodes. These establish that physical signals carry
  workload-identifying information; we extend this to software-accessible
  telemetry for governance.

- **Differential Architecture** (anonymous, ISCA 2026) proposes hardware
  mechanisms to limit performance of targeted applications — a complementary
  enforcement approach to our detection approach.

---

## 4. Method (~1.5 pages)

### 4.1 Data Collection

**Hardware:** 5 GPU types spanning 4 architecture generations:

| GPU              | Arch      | VRAM    | Context            |
|------------------|-----------|---------|--------------------|
| A100 SXM4 40GB   | Ampere    | 40 GB   | Cloud datacenter   |
| H100 80GB HBM3   | Hopper    | 80 GB   | Cloud datacenter   |
| H100 NVL          | Hopper    | 94 GB   | Cloud datacenter   |
| RTX 5080          | Blackwell | 16 GB   | Local desktop      |
| RTX 4050 Laptop   | Ada       | 6 GB    | Local laptop       |

**Workloads:** [TBD: final count] workload types across 5 categories:
- ML training: ResNet-18/50, MLP, BERT, GPT-2 (FP32 + AMP)
- ML inference: ResNet-50 batch inference, LLM generation (DeepSeek, Qwen, Whisper)
- Scientific HPC: 3D cuFFT, N-body simulation
- Crypto mining: Ethash proxy
- Other: Monte Carlo rendering, idle baseline

**Telemetry:** 9 raw NVML signals at 1 Hz. [TBD: + Tier 5 wall power from
Tapo P115 smart plug on RTX 5080.]

**Scale:** [TBD] total runs producing [TBD] parquet files, ~[TBD] hours of
GPU telemetry.

### 4.2 Feature Extraction

159 features from sliding windows:
- **Base statistics** (9 signals x 13 stats = 117): mean, std, min, max,
  percentiles (25/50/75/95), IQR, range, CV, skew, kurtosis
- **Autocorrelation** (3 signals x 5 lags = 15): lags 1/2/5/10/20 for
  utilization, power, memory
- **Cross-signal:** power/utilization ratio, PCIe total, utilization per SM%
- **Training-specific:** memory slope, epoch periodicity via FFT
- **Inference-specific:** duty cycle, idle fraction, utilization transitions

### 4.3 Classifier

XGBoost (n_estimators=500, max_depth=6, lr=0.05). Evaluated with stratified
group k-fold CV (5 folds, grouped by run_id to prevent leakage).

### 4.4 Adversarial Evaluation Protocol

**Iterative refinement:** N rounds of adversary-defender co-evolution.
Each round: (1) design new evasion strategies targeting the current classifier's
weaknesses; (2) retrain a hardened classifier incorporating the new adversarial
examples.

**[TBD: final count] evasion strategies** spanning:
- Utilization manipulation (sleep injection, clock throttling, PID control)
- Memory footprint reduction (gradient checkpointing, tiny batches)
- Workload mimicry (interleaving training with mining/HPC, feedback-controlled
  telemetry matching)
- Training-as-inference (online learning, diluted training at varying ratios)
- [TBD: datacenter-specific strategies if applicable]

**Table 1** (or appendix): Full strategy catalog with description, round
introduced, and technique category.

**Leave-one-strategy-out (LOO):** For each strategy family, train the classifier
on clean data + ALL OTHER adversarial families, test on the held-out family.
This tests generalization to *truly unseen* evasion techniques.

---

## 5. Results (~2.5 pages)

### 5.1 Baseline Classification

- **Same-GPU accuracy:** 100% across all 5 GPU types.
- **Pooled accuracy (5 GPUs):** 97.3% three-way (5-fold grouped CV).
- **Detection latency:** 97.4% at 5-second windows — performance is flat
  from 5s to 120s.

**Figure 1: Cross-GPU transfer heatmap.** 5x5 matrix showing accuracy when
training on one GPU and testing on another. Diagonal = 100%. Off-diagonal =
40-94%. Bottom row: pooled classifier at 97.3%.

*Key insight:* Single-GPU classifiers fail across architectures (clock speeds,
memory types, PCIe generations differ). Pooled training on diverse hardware
eliminates domain shift.

**Figure 3: Feature importance** (top 15 XGBoost features by gain). Shows
what the classifier actually relies on.

**Feature stability analysis** (brief): Temperature and utilization features
transfer best (CV < 0.15); clock and PCIe features are architecture-specific
(CV > 1.6).

**False positive rate:** Brief note that training-class precision is [TBD]%
(F1 = 0.96+); false positives are rare. Full confusion matrix and discussion
of governance implications in Appendix E.

### 5.2 Adversary-Defender Iteration

**Table 2: Arms race summary**

| Round | Attacker capability     | Baseline det. | Hardened det. | LOO det. |
|-------|------------------------|---------------|---------------|----------|
| 1     | Noise/padding          | 11-93%        | 100%          | 94-98%   |
| 2     | Targeted mimicry       | 0%            | 100%          | 70-100%  |
| 3     | Training-as-inference  | 0-4%          | 100%          | 4-100%   |
| [N]   | [TBD]                  | [TBD]         | [TBD]         | [TBD]    |

- Every evasion strategy is defeated after one round of adversarial retraining.
- LOO evaluation: **81.6% mean detection on completely unseen strategies.**
  7/12 strategy families detected at >94% even when never seen in training.
- [TBD: datacenter adversarial results]

**Figure 4: LOO detection rate per strategy family** (bar chart, sorted by
detection rate). Visually highlights which attacks generalize and which don't.

### 5.3 The Evasion Tax

**Figure 2: Dilution curve.** X-axis: training throughput fraction (1/N).
Y-axis: detection rate (LOO). Second y-axis or annotations: estimated training
cost multiplier using Rahman (2024) methodology.

| Dilution N | Throughput | LOO Detection | Cost multiplier |
|-----------|-----------|---------------|-----------------|
| 1 (full)  | 100%      | ~95%          | 1x              |
| 2         | 50%       | [TBD]%        | 2x              |
| 5         | 20%       | [TBD]%        | 5x              |
| 10        | 10%       | [TBD]%        | 10x             |
| 20        | 5%        | [TBD]%        | 20x             |

[TBD: controlled batch-size sweep — re-run at consistent batch size]

**Economic argument:** At the evasion threshold of [TBD]x throughput loss,
covert training of a frontier model costing $X would cost $[TBD]X. Using
published compute cost estimates [Rahman 2024], training [specific model] at
1/5 throughput would require [TBD] additional GPU-hours, costing an additional
$[TBD]M. This cost premium, combined with the extended wall-clock time
(increasing the window of detection), makes covert frontier training
economically unviable.

### 5.4 Wall-Power Validation [TBD — pending data collection]

- Correlation between NVML power_draw_w and Tapo P115 wall power
- System idle baseline: ~65W
- GPU power vs wall power offset: PSU efficiency + CPU/RAM overhead
- [If interesting: wall power as additional discriminative feature]

---

## 6. Discussion (~1 page)

### 6.1 The Involuntary Witness

The GPU microarchitecture acts as an involuntary witness: the backward pass
requires ~2x the FLOPs of the forward pass, creating a physical energy
signature that cannot be eliminated without eliminating training itself.
An adversary can redistribute this signature over time (dilution) but cannot
remove it — only amortize it across more wall-clock time.

### 6.2 Threat Model Comparison

[Discussion of tenant vs. operator: which evasion strategies are available
to each? Is the multi-GPU datacenter setting easier or harder to evade?
NVLink utilization as an additional signal for distributed training.]

[TBD: Placeholder for datacenter-specific discussion once those experiments
are done. Hypothesis: defender is even more favored because multi-GPU
training produces distinctive NVLink/interconnect patterns.]

### 6.3 Implications for Hardware-Enabled Mechanisms

Based on our feature analysis, a minimal HEM for training detection should
monitor:
- GPU utilization (most stable cross-architecture feature)
- Power draw (physical constraint on training)
- Memory usage and growth rate (training has distinctive allocation pattern)
- Temperature dynamics (physics-constrained, architecture-stable)
- [If multi-GPU: NVLink/interconnect utilization]

Minimum viable sampling rate: 1 Hz (our results hold at 1 Hz Tier 1).
Minimum detection window: 5 seconds (97.4% accuracy).

### 6.4 Limitations

- **Single-node only:** All evasion experiments run on single GPUs.
  Multi-node distributed training (data-parallel, pipeline-parallel) may have
  different evasion characteristics. [TBD: partially addressed if datacenter
  experiments are done.]
- **Batch-size confound:** [TBD: addressed by controlled re-run, or noted
  as limitation if not completed before deadline.]
- **Kernel-level evasion:** An adversary with access to custom CUDA kernels
  could potentially manipulate individual kernel behavior. We test only
  framework-level (PyTorch) evasion.
- **Classifier knowledge:** Our adversary does not have white-box access to
  the classifier. A white-box adversary could craft more targeted evasions.
- **Workload diversity:** We test [TBD] workload types. Real datacenters run
  a wider variety; the classifier may need retraining for new workload classes.

---

## 7. Conclusion (~0.5 page)

Zero-overhead GPU telemetry monitoring is a viable mechanism for AI compute
governance. Defenders are structurally favored in the adversarial arms race:
(1) every evasion we designed is defeated with one round of retraining;
(2) the classifier generalizes to most unseen attack types; and (3) the
minimum evasion tax—[TBD]x throughput loss—makes covert frontier training
economically unviable. These results provide empirical support for
telemetry-based verification in compute governance frameworks.

---

## Appendices

### A. Full Adversarial Strategy Catalog

Detailed description of each evasion strategy: motivation, implementation,
telemetry signature, detection rate across all evaluation modes.

[Table with columns: ID, Name, Round, Technique, Baseline Det., Hardened Det.,
LOO Det., Key feature exploited]

### B. Feature List

Full 159-feature specification with extraction formulas.

### C. Cross-GPU Feature Stability

Per-feature CV across 5 architectures, sorted by stability.

### D. Wall-Power Validation

Scatter plot: Tapo P115 wall power vs NVML power_draw_w for all workloads.
Discussion of PSU efficiency, system overhead (CPU/RAM), and whether NVML
is a trustworthy proxy for actual power consumption.

### E. False Positive Analysis

3x3 confusion matrix for the pooled classifier (5 GPUs, grouped CV).
Discussion: what types of workloads get misclassified as training? What are
the governance implications if a legitimate inference or HPC workload is
falsely flagged? How should a monitoring system handle disputed classifications?

---

## Figures & Tables Summary

### Main text figures
| # | Type      | Content                                          | Section |
|---|-----------|--------------------------------------------------|---------|
| 1 | Heatmap   | Cross-GPU transfer matrix (5x5 + pooled row)     | 5.1     |
| 2 | Plot      | Dilution curve + economic cost annotations        | 5.3     |
| 3 | Bar chart | Feature importance (top 15 XGBoost features)      | 5.1     |
| 4 | Bar chart | LOO detection rate per strategy family             | 5.2     |
| 5 | 4-panel   | Telemetry traces: training / inference / evasion / mining | 4 or 5  |

### Appendix figures
| # | Type      | Content                                          | Section |
|---|-----------|--------------------------------------------------|---------|
| A1| Scatter   | Wall power (Tapo P115) vs NVML power_draw_w       | App. D  |

### Tables
| # | Content                                          | Section |
|---|--------------------------------------------------|---------|
| 1 | Hardware + workload summary (5 GPUs, N workloads) | 4.1     |
| 2 | Arms race summary (rounds x detection rates)     | 5.2     |
| 3 | Evasion tax (dilution x detection x cost)         | 5.3     |

---

## Key References

**Compute governance:**
- Shavit (2023) — "What does it take to catch a Chinchilla?" Compute monitoring for AI governance [2303.11341]
- Sastry, Heim et al. (2024) — Computing Power and the Governance of AI [2402.08797]
- Baker, Kulp, Marks, Brundage, Heim (2025) — Verifying International Agreements on AI [2507.15916]
- Aarne, Fist, Withers (2024) — Secure, Governable Chips
- O'Gara, Kulp et al. (2025) — Hardware-Enabled Mechanisms for Verifying Responsible AI Development [2505.03742]
- Kulp, Siu & Heim (2025) — Insights from a Workshop on HEMs (RAND CF-A3056-1)
- Petrie (2024) — Firmware-Based Offline Licensing for AI Chip Export Controls [2404.18308]

**Workload fingerprinting & telemetry:**
- Xu et al. (2026) — WAVE: Architecture Observation for Model Oversight (ASPLOS)
- Jackowski, Copik, Hoefler (2025) — ShadowScope: GPU Kernel Validation [2509.00300]
- Monfared, Ganji, Holcomb & Tajik (2026) — Timing and Memory Telemetry on GPUs for AI Governance [2602.09369]
- Bhatele et al. (2026) — Characterizing Production GPU Workloads Using System-wide Telemetry (IPDPS)
- Chen et al. (2025) — Reveal: Detecting Anomalies in ML Infrastructure via Hardware Telemetry [2510.26008]

**Physical side channels:**
- Maia et al. (2022) — Snooping the GPU via Magnetic Side Channel (USENIX Security)
- Arefin & Serwadda (2025) — Exploiting HDMI and USB Ports for GPU Side-Channel Insights (CODASPY) [2410.02539]
- Latif et al. (2024) — Empirical Measurements of AI Training Power Demand [2412.08602]

**Other:**
- Rahman (2024) — AI training costs [2405.21015]
- Anonymous (2026) — Differential Architecture (ISCA, under review)

---

## TODO Before Submission (April 24)

### Experiments
- [ ] More adversary-defender rounds on local hardware (Robi)
- [ ] Datacenter adversarial experiments on H100 (Sabiha?)
- [ ] Controlled batch-size dilution sweep (consistent batch size across all N)
- [ ] Wall-power analysis (data collection in progress now)

### Analysis & figures
- [ ] Generate cross-GPU heatmap (Figure 1)
- [ ] Generate dilution curve with cost annotations (Figure 2)
- [ ] Generate feature importance bar chart (Figure 3)
- [ ] Generate LOO detection bar chart (Figure 4)
- [ ] Select representative telemetry traces for Figure 5
- [ ] Wall power vs NVML correlation scatter (Appendix)
- [ ] Compute economic cost estimates using Rahman (2024) methodology

### Writing
- [ ] Fill all [TBD] placeholders with final numbers
- [ ] Write actual prose (this is currently an outline)
- [ ] Format for ICML workshop style (LaTeX template)
- [ ] Proofread and polish
