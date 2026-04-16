# The Evasion Tax: Why GPU Telemetry Monitoring Favors Defenders in AI Compute Governance

**Robi Rahman** (MIRI) and **Sabiha Tajdari** (UVA)

TAIGR Workshop @ ICML 2026 — 8 pages + appendices — Deadline: April 24, 2026

---

## Working Abstract (~150 words)

Hardware-enabled monitoring of GPU workloads is a cornerstone of proposals for
AI compute governance. But if adversaries can defeat monitoring mechanisms, such
schemes are unworkable. We evaluate the adversarial robustness of GPU workload
classification using only zero-overhead, privacy-preserving Tier 1 NVML
telemetry — content-agnostic signals that observe physical effects of computation
without accessing model weights, training data, or hyperparameters. Across 4
rounds of adversary-defender iteration with 18 evasion strategy families —
including 4 multi-GPU distributed evasion strategies — on 7 GPU types spanning
4 architecture generations (A100, H100, H200, B200, RTX 5080, RTX 4050), we find
that (1) a pooled classifier achieves 97.9% three-way accuracy across all GPUs,
detecting training within 5 seconds; (2) all 4 multi-GPU distributed evasion
strategies are detected at 98.8-100% even in leave-one-strategy-out evaluation,
with hardened classifiers achieving 95-100% detection on composite evasion at
all idle fractions; and (3) single-GPU evasion partially succeeds against a
balanced classifier (4/18 families evade at <50% in LOO), but imposes a
substantial throughput cost and is defeated by evasion-resistant physics-based
features. Using published cost estimates for frontier models, we show the
evasion tax renders covert large-scale training economically unviable,
supporting the feasibility of telemetry-based compute governance.

---

## 1. Introduction (~1 page)

**Opening:** AI compute governance proposals increasingly depend on the ability
to verify what workloads run on regulated hardware [Baker et al. 2025,
Aarne et al. 2024]. If monitoring can be systematically evaded, the regulatory
scheme collapses. We ask: *how robust is GPU telemetry monitoring against a
sophisticated adversary, and what is the minimum cost of evasion?*

**Key claims (paragraph each):**

- Zero-overhead GPU telemetry (NVML Tier 1) is sufficient to classify workloads
  as training vs non-training with 97.9% three-way accuracy across 7 GPU types
  spanning consumer and datacenter hardware (Ampere through Blackwell). Crucially,
  this telemetry is content-agnostic and privacy-preserving: it observes physical
  effects of computation (power, temperature, utilization) without accessing model
  weights, training data, or hyperparameters [Shavit 2023, O'Gara et al. 2025].
  Static architectural heuristics like the roofline model [Williams et al. 2009]
  are insufficient: 35.3% of non-training configurations on H100 NVL overlap with
  training in arithmetic-intensity space [Tajdari, midterm]. Temporal, multi-signal
  telemetry resolves this ambiguity.

- Through 4 rounds of adversary-defender iteration, we designed 18 evasion
  strategy families including 4 multi-GPU distributed evasion strategies tested
  on 8-GPU datacenter nodes. Leave-one-strategy-out evaluation with a
  class-balanced classifier shows that **multi-GPU distributed evasion is
  detected at 98.8-100%** even when completely unseen, while single-GPU evasion
  is harder: 4/18 families partially evade (<50% LOO detection), with an overall
  mean of 68.3%. Physics-based evasion-resistant features (cumulative energy,
  power-memory correlation) recover detection for most single-GPU strategies.

- The strongest unseen evasion (diluted training, composite idle-padding)
  imposes an **evasion tax**: with a balanced hardened classifier, detection
  never drops below 50% on the composite sweep, and evasion-resistant features
  bring detection to 82-100% for most strategies. Using cost estimates from
  Rahman (2024), the throughput sacrifice required for partial evasion renders
  covert frontier training economically unviable. Critically, multi-GPU
  distributed evasion — the governance-relevant case — remains detected at
  99-100% across all conditions.

**Contributions (bulleted list):**
1. A privacy-preserving, content-agnostic classifier achieving 97.9% three-way
   pooled accuracy across 7 GPU types (4 architecture generations) with
   5-second detection latency (97.8%), using only zero-overhead Tier 1 telemetry.
2. Systematic adversarial evaluation: 18 evasion strategy families across 4
   rounds, including 4 multi-GPU distributed evasion strategies on 8-GPU
   datacenter nodes. Class-balanced LOO generalization: 68.3% mean, with DDP
   strategies at 98.8-100%. 4/18 families partially evade single-GPU detection.
3. Quantification of the **evasion tax**: with balanced hardened classifier,
   detection never drops below 50% on composite sweep; evasion-resistant
   physics-based features recover 82-100% for most strategies. Economic cost
   estimates show covert frontier training is unviable.
4. Cross-architecture generalization: B200→A100 transfer at 91.6% accuracy,
   demonstrating that classifiers trained on newer hardware detect training on
   older hardware without retraining. Wall-power validation via smart plug
   confirms NVML is a trustworthy proxy for actual power consumption.

---

## 2. Background & Threat Model (~0.75 page)

### 2.1 GPU Telemetry Tiers

- **Tier 1 (NVML/pynvml):** GPU utilization, memory usage, power draw,
  temperature, SM/memory clocks, PCIe bandwidth. Polled at 1 Hz. Zero runtime
  overhead. Always available on all NVIDIA GPUs.
- **Tier 2 (DCGM):** Datacenter-only. Adds tensor core activity, NVLink
  throughput, ECC errors. Requires driver configuration.
- **Tier 3 (Nsight Compute):** Per-kernel PMC profiling. 1,200-5,300% overhead.
  Impractical for continuous monitoring. High-overhead monitoring is especially
  problematic in production AI training datacenters, which already face severe
  power-management constraints — large synchronized training jobs produce rapid
  power swings, and the monitoring layer itself can worsen efficiency and thermal
  pressures [Choukse et al. 2025].
- Our classifier uses **Tier 1 only** — the minimum viable signal. This is also
  the most deployable: content-agnostic (no access to model weights, data, or
  code), privacy-preserving, and zero-overhead.

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

Primary task: **binary** — **ml_training** vs **non-training** (inference, HPC,
mining, rendering, idle). This directly matches the governance question: is
this GPU being used for ML training? Secondary: three-way (**ml_training** /
**ml_inference** / **other**) to characterize finer distinctions.
Sliding windows of 5-60 seconds over 1 Hz telemetry. 159 statistical features
per window.

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

- **Security-oriented workload detection:** Gangwal et al. (CANS 2020) show that
  Hardware Performance Counters can detect covert cryptomining using RF/SVM
  classifiers. They note an adversarial limitation: evasion is possible through
  restricted mining or signature neutralization, but at a profitability cost —
  a direct precedent for our "evasion tax" framing.

- **GPU telemetry for reliability:** Esposito et al. (2025) combine NVML
  telemetry and performance counters to estimate GPU stress and reliability
  across workloads. Their results confirm that telemetry reveals meaningful
  workload differences, but the target is reliability estimation rather than
  governance-oriented training detection.

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

**Hardware:** 7 GPU types spanning 4 architecture generations:

| GPU              | Arch      | VRAM    | Context            | GPUs |
|------------------|-----------|---------|--------------------|------|
| RTX 4050 Laptop   | Ada       | 6 GB    | Local laptop       | 1    |
| RTX 5080          | Blackwell | 16 GB   | Local desktop      | 1    |
| A100 SXM4 80GB    | Ampere    | 80 GB   | Cloud 8-GPU node   | 8    |
| H100 80GB HBM3    | Hopper    | 80 GB   | Cloud datacenter   | 8    |
| H100 NVL           | Hopper    | 94 GB   | Cloud datacenter   | 2    |
| H200 SXM 141GB    | Hopper    | 141 GB  | Cloud 8-GPU node   | 8    |
| B200 SXM 192GB    | Blackwell | 192 GB  | Cloud 8-GPU node   | 8    |

**Workloads:** 107 unique workload configurations across 6 categories:
- ML training: ResNet-18/50, MLP, BERT, GPT-2 (FP32 + AMP); DDP (8-GPU);
  FSDP sharded training (7B-70B LLaMA-architecture models)
- ML inference: ResNet-50 batch inference, LLM generation (DeepSeek, Qwen, Whisper)
- Scientific HPC: 3D cuFFT, N-body simulation
- Crypto mining: Ethash proxy
- Other: Monte Carlo rendering, ffmpeg NVENC, idle baseline
- Adversarial evasion: 18 strategy families (Rounds 1-4) including 4 multi-GPU
  distributed evasion strategies

**Telemetry:** 9 raw NVML signals at 1 Hz. Wall power from Tapo P115 smart
plug on RTX 5080 (35 workloads with simultaneous NVML + wall power).

**Scale:** 987 runs producing 746 parquet files, 302,252 telemetry rows across
~83 hours of GPU telemetry.

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

**Design rationale:** Detection can be conceptualized as a two-stage pipeline:
(1) a structural/roofline layer that identifies workloads occupying a
training-like operating region in arithmetic-intensity space, and (2) a temporal
telemetry layer that resolves ambiguity within that region using time-domain
features [cf. Williams et al. 2009 for the roofline model]. Our XGBoost
classifier incorporates both layers simultaneously: its feature set includes
static level features (mean power, mean utilization — capturing roofline-relevant
information) alongside temporal and cross-signal features (autocorrelation, CV,
periodicity — resolving the ambiguity that static summaries cannot).

### 4.4 Adversarial Evaluation Protocol

**Iterative refinement:** N rounds of adversary-defender co-evolution.
Each round: (1) design new evasion strategies targeting the current classifier's
weaknesses; (2) retrain a hardened classifier incorporating the new adversarial
examples.

**18 evasion strategy families** spanning:
- Utilization manipulation (sleep injection, clock throttling, PID control)
- Memory footprint reduction (gradient checkpointing, tiny batches)
- Workload mimicry (interleaving training with mining/HPC, feedback-controlled
  telemetry matching)
- Training-as-inference (online learning, diluted training at varying ratios,
  gradient accumulation)
- Multi-GPU distributed evasion (DDP online learning, DDP diluted training,
  staggered dilution to spread NVLink signatures, gradient accumulation across
  ranks)

**Table 1** (or appendix): Full strategy catalog with description, round
introduced, and technique category.

**Leave-one-strategy-out (LOO):** For each strategy family, train the classifier
on clean data + ALL OTHER adversarial families, test on the held-out family.
This tests generalization to *truly unseen* evasion techniques.

---

## 5. Results (~2.5 pages)

### 5.1 Baseline Classification

- **Same-GPU accuracy:** 100% across all 7 GPU types (both binary and three-way).
- **Pooled three-way accuracy (7 GPUs):** 97.9%, F1=0.955 (5-fold grouped CV,
  18,056 windows).
- **Detection latency:** 97.8% at 5-second windows — performance is flat
  from 5s to 120s (97.8-98.5% across all window sizes).

**Figure 1: Cross-GPU transfer heatmap.** 7x7 matrix showing accuracy when
training on one GPU and testing on another. Diagonal = 100%. Notable
cross-architecture transfers: B200→A100 91.6%, B200→H200 98.5%, H200→A100
86.6%. Bottom row: pooled classifier at 97.9%.

*Key insight:* Single-GPU classifiers fail across architectures (clock speeds,
memory types, PCIe generations differ). Pooled training on diverse hardware
eliminates domain shift.

**Figure 3: Feature importance** (top 15 XGBoost features by gain). Shows
what the classifier actually relies on.

**Feature stability analysis** (brief): Temperature and utilization features
transfer best (CV < 0.15); clock and PCIe features are architecture-specific
(CV > 1.6).

**Counterintuitive finding:** GPU utilization is a surprisingly weak indicator
of training — inference can exhibit *higher* average utilization than training
at the same arithmetic intensity. Power coefficient of variation and lag-1
power autocorrelation are far more reliable discriminators, because they capture
the periodic-but-stable temporal structure of the training loop (repeated
forward/backward/optimizer phases). This is why temporal features are essential
and static utilization thresholds are insufficient.

**False positive rate:** Brief note that training-class precision is [TBD]%
(F1 = 0.96+); false positives are rare. Full confusion matrix and discussion
of governance implications in Appendix E.

### 5.2 Adversary-Defender Iteration

**Table 2: Arms race summary**

| Round | Attacker capability              | Baseline det. | Hardened det. | R3 +Features | LOO det. |
|-------|----------------------------------|---------------|---------------|--------------|----------|
| 1     | Noise/padding (A,B,D)            | 0-50%         | 5-61%         | 38-61%       | 17-86%   |
| 2     | Targeted mimicry (E-J)           | 0-60%         | 13-45%        | 82-100%      | 39-100%  |
| 3     | Training-as-inference (K,L)      | 57-100%       | 84-100%       | 100%         | 56-100%  |
| 4     | Composite + grad accum (M,N)     | 100%          | 100%          | 100%         | 78-100%  |
| DDP   | Multi-GPU distributed (K/L DDP)  | 98-100%       | 99-100%       | 99-100%      | 98-100%  |

All classifiers use class-balanced training (downsampled to equal class counts).

- LOO evaluation: **68.3% mean detection on completely unseen strategies.**
  4/18 strategy families evade at <50% detection: composite (29%), H_mimicry
  (43%), E_memory_minimal (39%), B_low_util (17%). 10/18 families detected at
  >78%. Evasion-resistant features (R3) substantially improve detection for
  strategies that exploit the R2 classifier's weaknesses.
- **Datacenter adversarial results:** All 4 DDP evasion families (K_ddp,
  K_ddp_accum, L_ddp, L_ddp_stagger) detected at **98.8-100% in LOO
  evaluation** even with balanced training. Multi-GPU evasion is substantially
  harder to evade than single-GPU evasion — the additional NVLink/AllReduce
  communication patterns provide a strong, hard-to-suppress signal.

**Figure 4: LOO detection rate per strategy family** (bar chart, 18 families
sorted by detection rate). Visually highlights that only H_mimicry partially
evades; all other families including DDP strategies are detected at >98%.

### 5.3 The Evasion Tax

**Figure 2: Dilution curve.** X-axis: training throughput fraction (1/N).
Y-axis: detection rate (LOO). Second y-axis or annotations: estimated training
cost multiplier using Rahman (2024) methodology.

**Table 3a: Controlled dilution sweep** (ResNet-50, batch 128, RTX 5080,
class-balanced classifiers):

| Dilution N | Throughput | LOO 5s  | LOO 15s | Hardened 5s | Cost mult. |
|-----------|-----------|---------|---------|-------------|------------|
| 1 (full)  | 100%      | 98.4%   | 100%    | 100%        | 1x         |
| 2         | 50%       | 98.4%   | 100%    | 100%        | 2x         |
| 3         | 33%       | 65.6%   | 96.7%   | 95.1%       | 3x         |
| 5         | 20%       | 86.9%   | 100%    | 100%        | 5x         |
| 10        | 10%       | 67.2%   | 86.7%   | 83.6%       | 10x        |
| 20        | 5%        | 32.8%   | 100%    | 75.4%       | 20x        |

**Table 3b: Composite idle-padding sweep** (balanced hardened R2 classifier):

| Idle fraction | R2 Hardened | R3 +Features |
|---------------|-------------|--------------|
| 0%            | 95.2%       | 76.2%        |
| 10-50%        | 95-100%     | 52-100%      |
| 60%           | 100%        | 0%           |
| 70%           | 95.2%       | 57.1%        |
| 80%           | 50.0%       | 66.7%        |
| 90%           | 81.0%       | 81.0%        |
| 95%           | 100%        | 38.1%        |

The balanced hardened classifier (R2) **never drops below 50%** on the
composite sweep — evasion is partial at best.

**Economic argument:** The evasion tax is no longer infinite but remains
substantial. At 5-second windows, the LOO classifier detects N=20 dilution
(5% training throughput, 20x cost) at only 32.8%, but extending observation
to 15 seconds recovers 100% detection. The balanced hardened classifier
maintains >50% detection at all composite idle fractions, meaning an adversary
cannot reliably evade — each 60-second observation window has at least a
coin-flip chance of triggering an alert, making sustained covert training
infeasible over the hours-to-weeks timescale of real training runs.

Using published compute cost estimates [Rahman 2024], even partial evasion
requiring >60% idle fraction imposes a >2.5x cost multiplier. Frontier
training power demand is growing at >2x per year, with individual training
runs projected to require 4-16 GW by 2030 [You & Owen, EPRI 2025]. This cost
premium, combined with the extended wall-clock time (increasing the window of
detection), makes covert frontier training economically unviable.

Critically, **multi-GPU distributed training — the governance-relevant case
for frontier models — is detected at 98-100%** across all conditions,
including balanced LOO evaluation. The single-GPU evasion results are
relevant for smaller-scale unauthorized training but do not undermine the
core governance use case.

### 5.4 Wall-Power Validation

- Tapo P115 smart plug wall power collected simultaneously with NVML for 35
  workloads on RTX 5080 (Tier 5 data).
- System idle baseline: ~65W wall power at GPU idle.
- Strong correlation between NVML power_draw_w and wall power; offset is
  consistent (~65W from PSU efficiency + CPU/RAM overhead).
- NVML is a trustworthy proxy for actual power consumption — hardware-level
  tampering would be needed to spoof it.

---

## 6. Discussion (~1 page)

### 6.1 The Involuntary Witness

The GPU microarchitecture acts as an involuntary witness: the backward pass
requires ~2x the FLOPs of the forward pass, creating a physical energy
signature that cannot be eliminated without eliminating training itself.
An adversary can redistribute this signature over time (dilution) but cannot
remove it — only amortize it across more wall-clock time.

### 6.2 Threat Model Comparison

Our datacenter experiments (8-GPU A100, H200, B200 nodes) directly test the
operator threat model. We designed 4 multi-GPU evasion strategies: DDP online
learning, DDP diluted training, staggered dilution (spreading NVLink traffic
across time), and gradient accumulation across ranks. All are detected at
99.6-100% in leave-one-strategy-out evaluation. Multi-GPU evasion is no harder
to detect than single-GPU evasion — in fact, the additional NVLink/AllReduce
communication patterns may make distributed training *easier* to identify.
FSDP training at 7B-70B scale produces heavy AllGather + ReduceScatter NVLink
traffic on every forward/backward step, creating distinctive per-GPU telemetry
signatures even when individual GPUs show moderate utilization.

### 6.3 Privacy-Preserving Monitoring

Tier 1 telemetry is content-agnostic: it observes the physical and architectural
effects of computation (power draw, temperature, utilization, memory pressure)
without accessing model weights, training data, hyperparameters, or application
logic. This addresses a central deployment concern in shared cloud settings,
where governance mechanisms that require inspecting proprietary information face
legal, operational, and competitive objections [Shavit 2023, O'Gara et al. 2025].
Our classifier demonstrates that the governance-relevant question ("is this GPU
training a model?") can be answered without any access to the semantic content
being processed — a necessary property for real-world adoption.

### 6.4 Layered Defense and Cross-Layer Security

If telemetry is collected and checked at multiple layers — device-level (NVML),
host-level (OS counters), and infrastructure-level (network, power) — an
adversary must evade all layers simultaneously [O'Gara et al. 2025]. The evasion
cost is multiplicative across layers: our single-layer Tier 1 results represent
a *lower bound* on detection capability. Combined with tamper-resistant hardware
(HEM per Aarne et al. 2024), software-level evasion alone cannot defeat
monitoring. This structural advantage compounds the evasion tax: even if an
adversary discovers a strategy that reduces detection in one layer, the remaining
layers provide independent verification.

### 6.5 Implications for Hardware-Enabled Mechanisms

Based on our feature analysis, a minimal HEM for training detection should
monitor:
- GPU utilization (most stable cross-architecture feature)
- Power draw (physical constraint on training)
- Memory usage and growth rate (training has distinctive allocation pattern)
- Temperature dynamics (physics-constrained, architecture-stable)
- [If multi-GPU: NVLink/interconnect utilization]

Minimum viable sampling rate: 1 Hz (our results hold at 1 Hz Tier 1).
Minimum detection window: 5 seconds (97.4% accuracy).

### 6.6 Limitations

- **Single-node multi-GPU only:** We test DDP and FSDP evasion on 8-GPU
  nodes (A100, H200, B200). Multi-node distributed training (pipeline-parallel
  across machines) may have different evasion characteristics due to network
  (rather than NVLink) communication patterns.
- **Kernel-level evasion:** An adversary with access to custom CUDA kernels
  could potentially manipulate individual kernel behavior. We test only
  framework-level (PyTorch) evasion.
- **Model scale:** All evasion experiments use small models (ResNet-18/50,
  25M params). Evasion dynamics may differ for relevant-scale models (1-70B
  params) where training and inference telemetry signatures converge.
- **Workload diversity:** We test [TBD] workload types. Real datacenters run
  a wider variety; the classifier may need retraining for new workload classes.

---

## 7. Conclusion (~0.5 page)

Zero-overhead GPU telemetry monitoring is a viable mechanism for AI compute
governance. Defenders are structurally favored for the governance-relevant case:
(1) **multi-GPU distributed evasion is detected at 98-100%** across all
conditions including balanced LOO evaluation on 8-GPU datacenter nodes —
the NVLink/AllReduce signatures of distributed training are extremely difficult
to suppress; (2) single-GPU evasion partially succeeds against a balanced
classifier (4/18 families below 50% LOO), but evasion-resistant physics-based
features recover detection to 82-100% for most strategies; (3) the balanced
hardened classifier never drops below 50% detection on composite idle-padding
at any throughput level, imposing a substantial and unavoidable evasion tax.
These results hold across 7 GPU types spanning 4 architecture generations
from consumer laptops to datacenter B200 nodes, providing empirical support
for telemetry-based verification in compute governance frameworks.

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

### F. Roofline Overlap Analysis

Static architectural heuristics are insufficient for training detection.
286-configuration corner-case sweep on H100 NVL [Tajdari, midterm report],
spanning 6 workload families:
- **CNN inference** (ResNet-18/50/101/152, 95 configs): enters training band
  via batch-size scaling
- **LLM prefill** (GPT-2/M/L, 115 configs): FP32 prefill nearly exactly
  matches training arithmetic intensity
- **LLM decode** (GPT-2/M, 30 configs): contrasting case — KV-cache traffic
  keeps it memory-bound
- **Quantization variants** (ResNet-50/GPT-2, 18 configs): shifts arithmetic
  intensity without changing task semantics
- **Forward-only execution** (ResNet-50, 9 configs): closest semantic neighbor
  to training — preserves forward compute while removing backward/optimizer
- **ViT inference** (ViT-S/B/L, 19 configs): enters training range at unit batch

Result: 101/286 (35.3%) non-training configurations fall inside the training
arithmetic-intensity range [Williams et al. 2009]. This motivates the temporal
telemetry approach used in the main paper.

Figure A2: Roofline plot showing training reference band and 286 corner-case
configurations, colored by workload family.

### G. Alternative Classifiers [if time permits]

Evaluate SVM-RBF, Random Forest, and Logistic Regression alongside XGBoost
on both binary and three-way tasks. On Sabiha's hard H100 NVL corner-case
benchmark, SVM-RBF at 15-second windows achieves 95.65% binary accuracy and
92.89% macro-F1 — comparable to or exceeding XGBoost on the hardest cases.
Scripts: `week7/scripts/train_classifiers.py`.

### H. Window Size vs Accuracy

Classification accuracy across 5s / 15s / 30s windows for binary and three-way
tasks. 15-second windows provide the best overall tradeoff; 5-second windows
already achieve strong results (97.4% three-way), showing that detection latency
can be kept low without meaningful accuracy loss.

---

## Figures & Tables Summary

### Main text figures
| # | Type      | Content                                          | Section |
|---|-----------|--------------------------------------------------|---------|
| 1 | Heatmap   | Cross-GPU transfer matrix (7x7 + pooled row)     | 5.1     |
| 2 | Plot      | Dilution curve + economic cost annotations        | 5.3     |
| 3 | Bar chart | Feature importance (top 15 XGBoost features)      | 5.1     |
| 4 | Bar chart | LOO detection rate per 18 strategy families        | 5.2     |
| 5 | 4-panel   | Telemetry traces: training / inference / evasion / mining | 4 or 5  |

### Appendix figures
| # | Type      | Content                                          | Section |
|---|-----------|--------------------------------------------------|---------|
| A1| Scatter   | Wall power (Tapo P115) vs NVML power_draw_w       | App. D  |
| A2| Scatter   | Roofline plot: training band + 286 corner cases   | App. F  |
| A3| Bar chart | Classifier comparison (XGB/SVM/RF/LR) on binary   | App. G  |
| A4| Line plot | Accuracy vs window size (5s/15s/30s)              | App. H  |

### Tables
| # | Content                                          | Section |
|---|--------------------------------------------------|---------|
| 1 | Hardware + workload summary (7 GPUs, 107 workloads) | 4.1     |
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

**Security-oriented workload detection:**
- Gangwal et al. (2020) — Detecting Covert Cryptomining Using HPC (CANS, Springer LNCS)

**GPU telemetry & reliability:**
- Esposito et al. (2025) — GPU Under Pressure: Estimating Stress via Telemetry [2511.05067]

**Datacenter power & economics:**
- Choukse et al. (2025) — Power Stabilization for AI Training Datacenters [2508.14318]
- You & Owen (2025) — Scaling Intelligence: Exponential Growth of AI's Power Needs (EPRI+Epoch)

**Performance modeling:**
- Williams, Waterman & Patterson (2009) — Roofline: Visual Performance Model (CACM)

**Other:**
- Cottier, Rahman, et al. (2024) — AI training costs [2405.21015]
- Anonymous (2026) — Differential Architecture (ISCA, under review)

---

## TODO Before Submission (April 24)

### Experiments
- [ ] More adversary-defender rounds on local hardware (Robi)
- [ ] Datacenter adversarial experiments on A100/H100/B100
- [ ] Controlled batch-size dilution sweep (consistent batch size across all N)
- [ ] Wall-power analysis (Robi)

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
