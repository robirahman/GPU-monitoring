# The Evasion Tax: Why GPU Telemetry Monitoring Favors Defenders in AI Compute Governance

**Robi Rahman** (MIRI) and **Sabiha Tajdari** (UVA)

TAIGR Workshop @ ICML 2026 — 2 pages + appendices — Deadline: April 24, 2026

---

## Abstract (~150 words)

Hardware-enabled monitoring of GPU workloads is a cornerstone of proposals for
AI compute governance. But if adversaries can defeat monitoring, such schemes
are unworkable. We evaluate the adversarial robustness of GPU workload
classification using only zero-overhead, privacy-preserving, content-agnostic
NVML counters — utilization, power, temperature, and memory sampled at 1 Hz —
which observe physical effects of computation without accessing model weights,
training data, or hyperparameters. Across 4 rounds of adversary-defender
iteration with 18 evasion strategy families — including multi-GPU distributed
evasion on 8-GPU nodes — on 7 GPU types spanning 4 architecture generations
(Ampere through Blackwell), we find that (1) a pooled classifier achieves
97.9% three-way accuracy with 5-second detection; (2) hardened classifiers
achieve 100% detection against all known attacks; and (3) leave-one-strategy-out
evaluation detects 98.8% of completely unseen evasion families. We quantify
the evasion tax: any throughput sacrifice sufficient to reduce detection renders
covert frontier training economically unviable.

---

## 1. Introduction (~250 words)

**Opening:** AI compute governance proposals depend on verifying what workloads
run on regulated hardware [Baker et al. 2025, Shavit 2023]. If monitoring can
be systematically evaded, the regulatory scheme collapses. We ask: *how robust
is GPU telemetry monitoring against a sophisticated adversary, and what is the
minimum cost of evasion?*

**Claim 1 — Detection works:** Zero-overhead NVML counters (utilization, power,
temperature, memory, clocks, PCIe bandwidth — sampled at 1 Hz with no runtime
cost) are sufficient to classify workloads as training vs non-training with
97.9% three-way accuracy across 7 GPU types spanning 4 architecture generations,
with detection in 5 seconds. This telemetry is content-agnostic and
privacy-preserving: it reveals no model weights, training data, or
hyperparameters [Shavit 2023, O'Gara et al. 2025].

**Claim 2 — Adversarial robustness:** Through 4 rounds of adversary-defender
iteration, we designed 18 evasion strategy families including 4 multi-GPU
distributed strategies on 8-GPU datacenter nodes. Every strategy is defeated
after one round of retraining. Leave-one-strategy-out evaluation achieves
98.8% mean detection on completely unseen strategies.

**Claim 3 — The evasion tax:** After one round of defender hardening, no
evasion strategy at any dilution level (0-95% idle) drops below 100% detection.
Even against the unhardened baseline, >80% idle fraction is needed to evade —
a >5x cost multiplier that renders covert frontier training economically
unviable [Cottier, Rahman et al. 2024].

**Contributions:**
1. Privacy-preserving classifier: 97.9% pooled accuracy, 7 GPUs, 5s latency.
2. 18 evasion families across 4 rounds; LOO generalization: 98.8%.
3. Evasion tax quantification: 100% hardened detection at all dilution levels.
4. Cross-architecture transfer (B200→A100 at 91.6%) without retraining.

---

## 2. Method (~200 words)

**Hardware:** 7 GPU types spanning 4 architecture generations:

| GPU             | Arch      | VRAM   | GPUs |
|-----------------|-----------|--------|------|
| RTX 4050 Laptop | Ada       | 6 GB   | 1    |
| RTX 5080        | Blackwell | 16 GB  | 1    |
| A100 SXM4       | Ampere    | 80 GB  | 8    |
| H100 80GB       | Hopper    | 80 GB  | 8    |
| H100 NVL        | Hopper    | 94 GB  | 2    |
| H200 SXM        | Hopper    | 141 GB | 8    |
| B200 SXM        | Blackwell | 192 GB | 8    |

**Workloads:** 107 unique configurations: ML training (ResNet, BERT, GPT-2,
FSDP 7B-70B LLaMA), ML inference (LLM generation, batch classification),
scientific HPC (cuFFT, N-body), crypto mining, rendering/encoding, and 18
adversarial evasion strategy families. 987 runs, ~83 hours of telemetry.

**Features:** 159 statistical features from sliding windows over 9 NVML signals
at 1 Hz: per-signal statistics (mean, std, percentiles, CV, skew, kurtosis),
autocorrelation at multiple lags, cross-signal ratios, and training-specific
temporal features (memory slope, epoch periodicity).

**Classifier:** XGBoost (500 trees, depth 6). Stratified group k-fold CV
(5 folds, grouped by run to prevent leakage).

**Adversarial protocol:** 4 rounds of adversary-defender co-evolution. 18
evasion families spanning: utilization manipulation, memory reduction, workload
mimicry, training-as-inference, and multi-GPU distributed evasion.
Leave-one-strategy-out (LOO): train on all other families, test on the
held-out family.

---

## 3. Results (~300 words)

### 3.1 Baseline Classification

Same-GPU accuracy: 100% across all 7 GPU types. Pooled three-way accuracy
(7 GPUs): 97.9%, F1=0.955. Detection latency: 97.8% at 5-second windows —
performance is flat from 5s to 120s. Cross-architecture transfer without
retraining: B200→A100 91.6%, B200→H200 98.5%.

### 3.2 Adversarial Robustness

**Table 1: Arms race summary**

| Round | Attacker capability           | Baseline | Hardened | LOO     |
|-------|-------------------------------|----------|----------|---------|
| 1     | Noise/padding                 | 41-97%   | 100%     | 100%    |
| 2     | Targeted mimicry              | 13-64%   | 100%     | 89-100% |
| 3     | Training-as-inference         | 89-100%  | 100%     | 99-100% |
| 4     | Composite + gradient accum.   | 100%     | 100%     | 100%    |
| DDP   | Multi-GPU distributed (8-GPU) | 99-100%  | 100%     | 100%    |

Every evasion strategy is defeated after one round of retraining. LOO
evaluation: **98.8% mean detection on completely unseen strategies.** 0/18
families evade at <50%; 16/18 detected at >98%. All 4 DDP evasion families
detected at 99.6-100% in LOO — multi-GPU evasion is no harder to detect than
single-GPU evasion.

[**Figure 1:** LOO detection rate per 18 strategy families (bar chart)]

### 3.3 The Evasion Tax

**Table 2: Dilution curve**

| Dilution | Throughput | LOO Det. | Hardened | Cost mult. |
|----------|-----------|----------|----------|------------|
| 1 (full) | 100%      | ~99%     | 100%     | 1x         |
| 2        | 50%       | 100%     | 100%     | 2x         |
| 5        | 20%       | 100%     | 100%     | 5x         |
| 10       | 10%       | 95.2%    | 100%     | 10x        |
| 20       | 5%        | 100%     | 100%     | 20x        |

After defender hardening, detection remains 100% at all idle fractions (0-95%).
No throughput sacrifice enables evasion. Even against the unhardened baseline,
>80% idle is needed to drop below 50% detection — a >5x cost multiplier. Using
published frontier training cost estimates [Cottier, Rahman et al. 2024], this
renders covert large-scale training economically unviable — especially as
frontier power demand grows at >2x/year [You & Owen 2025].

[**Figure 2:** Dilution curve with cost multiplier annotations]

---

## 4. Discussion & Conclusion (~100 words)

The GPU acts as an involuntary witness: the backward pass requires ~2x the
FLOPs of the forward pass, creating a physical energy signature that cannot
be eliminated without eliminating training itself. An adversary can dilute this
signature over time but cannot remove it. Our telemetry is content-agnostic —
it reveals whether a GPU is training, not what it is training — addressing
privacy concerns for deployment in shared cloud environments [O'Gara et al.
2025]. These single-layer results are a lower bound: layered monitoring across
device, host, and infrastructure levels compounds the evasion cost
multiplicatively.

**Limitations:** Single-node multi-GPU only (no multi-node pipeline parallel);
framework-level evasion only (no custom CUDA kernels); no white-box adversary.

**Conclusion:** Zero-overhead GPU telemetry monitoring is viable for compute
governance. Defenders are structurally favored: every evasion is defeated with
retraining, unseen attacks are detected at 98.8%, and the evasion tax makes
covert frontier training economically unviable.

---

## Appendices

### A. Related Work
Full related work section (see 8-page version Section 3).

### B. Threat Models
Cloud tenant (Threat Model A) vs datacenter operator (Threat Model B)
distinction; discussion of why our evasion strategies are relevant to both.

### C. Wall-Power Validation
Tapo P115 smart plug vs NVML power_draw_w correlation for 35 workloads on
RTX 5080. NVML as trustworthy proxy for actual power consumption.

### D. Roofline Overlap Analysis
286-configuration corner-case sweep on H100 NVL: 35.3% of non-training configs
overlap with training in arithmetic-intensity space, motivating temporal
telemetry over static heuristics.

### E. Full Adversarial Strategy Catalog
18 strategy families: ID, name, round, technique, baseline/hardened/LOO
detection rates, key feature exploited.

### F. Cross-GPU Transfer Heatmap
7x7 accuracy matrix (train on one GPU, test on another) + pooled row.

### G. Feature List & Stability
159-feature specification. Per-feature CV across architectures.

### H. False Positive Analysis
3x3 confusion matrix. Governance implications of misclassification.

### I. Window Size vs Accuracy
5s / 15s / 30s window comparison for binary and three-way tasks.

---

## References

- Baker, Kulp, Marks, Brundage & Heim (2025) — Verifying International Agreements on AI [2507.15916]
- Cottier, Rahman et al. (2024) — Rising AI Training Costs [2405.21015]
- O'Gara, Kulp et al. (2025) — Hardware-Enabled Mechanisms [2505.03742]
- Shavit (2023) — What Does It Take to Catch a Chinchilla? [2303.11341]
- You & Owen (2025) — Scaling Intelligence (EPRI+Epoch)
