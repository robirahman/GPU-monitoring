# Classifying GPU Workloads via Always-Available Telemetry for Compute Governance

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
rounds of adversary-defender iteration plus a white-box Round 5 and a
defender-tier Round 5b with dense LLM inference at matched model sizes (Qwen
2.5 7B/14B/32B FP16), we evaluate 18 evasion strategy families — including 4
multi-GPU data-parallel (DDP) distributed strategies plus 2 Round-5
whitebox families — on 9 GPU classes spanning
4 architecture generations (A100-40GB/80GB, H100 80GB, H100 NVL, H200, B200,
RTX 5080, RTX 5090, RTX 4050 Laptop). We find that (1) a pooled classifier
achieves 98.4% three-way accuracy (RandomForest, 30 s windows; 5-fold
cross-validation grouped by run to prevent leakage) with detection in 5 seconds
(97.8%); (2) leave-one-strategy-out (LOO) evaluation — holding an entire
evasion family out of training — detects unseen strategies at a 93.9%
unweighted family mean across the 18 R1–R4 strategy families (95.3%
for the deployed two-stage pipeline; § 2.3), with all 4 DDP families at
98.4–100% and the remaining single-family misses concentrated in
Round-5 whitebox LoRA fine-tuning (46–35% family mean); and
(3) a hardened classifier closes the remaining gaps — no evasion strategy
reduces detection below the economic break-even threshold. Using published
cost estimates for frontier models, we show the evasion tax renders covert
large-scale training economically unviable, supporting the feasibility of
telemetry-based compute governance.

---

## 1. Introduction (~1 page)

**Opening:** AI compute governance proposals [Shavit 2023, Baker et al. 2025,
Aarne et al. 2024] and enacted policy — notably U.S. Executive Order 14110
(2023), which established a 10²⁶-FLOP compute-reporting threshold for
frontier model training — increasingly depend on the ability to verify what
workloads run on regulated hardware. If monitoring can be systematically
evaded, the regulatory scheme collapses. *The paper's central question is
therefore one of adversarial robustness, not classification capability: given
that a workload classifier exists, how robust is it against a sophisticated
evasion adversary, and what is the minimum cost of evasion?* We build the
classifier as a prerequisite, then spend most of our effort attacking it.

**Key claims (paragraph each):**

- Zero-overhead GPU telemetry (NVML Tier 1) is sufficient to classify workloads
  as training vs non-training with 98.4% three-way accuracy across 9 GPU
  classes spanning consumer and datacenter hardware (Ampere through Blackwell).
  Crucially, this telemetry is content-agnostic and privacy-preserving: it
  observes physical effects of computation (power, temperature, utilization)
  without accessing model weights, training data, or hyperparameters
  [Shavit 2023, O'Gara et al. 2025]. Static architectural heuristics like the
  roofline model [Williams et al. 2009] are insufficient: in a
  286-configuration corner-case sweep on H100 NVL (detailed in Appendix F),
  35.3% of non-training configurations overlap with training in
  arithmetic-intensity space. Temporal, multi-signal telemetry resolves this
  ambiguity.

- Through 4 rounds of adversary-defender iteration, plus a white-box Round 5
  and a defender-tier Round 5b targeting the inference/training boundary at
  matched LLM sizes, we designed 18 evasion strategy families including 4
  multi-GPU distributed strategies tested on 8-GPU datacenter nodes, plus 2
  Round-5 whitebox families (`whitebox_full`, `whitebox_lora`) evaluated
  under the same LOO protocol.
  Leave-one-strategy-out evaluation with a class-balanced classifier shows
  an **unweighted family mean of 93.9%** (three-way) / **95.3%**
  (deployed two-stage, § 2.3) on the 18 R1–R4 strategy families, with
  all 4 DDP families detected at 98.4–100%. Single-family misses
  concentrate in Round-5 whitebox LoRA (`whitebox_lora` at 46.3%
  three-way / 35.2% two-stage) where dilution + low trainable-parameter
  footprint weakens the level signal. Physics-based evasion-resistant
  features (cumulative energy, power-memory correlation, pre-allocation
  memory-plateau features) close the remaining gap for the hardest
  single-GPU strategies in the in-distribution hardened state.

- The strongest unseen evasion (diluted training, composite idle-padding)
  imposes an **evasion tax**: with a balanced hardened classifier, detection
  never falls below the economic break-even threshold on the composite sweep,
  and evasion-resistant features push detection to 95-100% for most
  strategies. Using cost estimates from Cottier, Rahman et al. (2024), the
  throughput sacrifice required for partial evasion renders covert frontier
  training economically unviable. Critically, multi-GPU distributed evasion — the
  governance-relevant case — remains detected at 99-100% across all
  conditions.

**Contributions (bulleted list):**
1. A privacy-preserving, content-agnostic classifier achieving 98.4% three-way
   pooled accuracy (RF, 30 s, 5-fold grouped CV) across 9 GPU classes (4
   architecture generations) with 5-second detection latency (97.8%), using
   only zero-overhead Tier 1 telemetry.
2. Systematic adversarial evaluation: 18 evasion strategy families across 4
   adversary-defender rounds plus a white-box Round 5 and defender-tier Round
   5b (dense LLM inference at matched sizes), including 4 multi-GPU
   distributed strategies on 8-GPU datacenter nodes, plus 2 Round-5 whitebox
   families (20 families total under LOO). Class-balanced LOO
   generalization: **93.9% (three-way) / 95.3% (two-stage) unweighted family
   mean** on the 18 R1–R4 strategy families, with DDP strategies at
   98.4–100% and the one sub-50% outlier being Round-5 `whitebox_lora`
   (heavy-dilution LoRA fine-tuning).
3. Quantification of the **evasion tax**: with a balanced hardened classifier,
   detection on the composite sweep remains above economic break-even at all
   idle fractions; evasion-resistant physics-based features recover 95-100%
   for most strategies. Economic cost estimates show covert frontier training
   is unviable.
4. Cross-architecture generalization: datacenter-to-datacenter transfer stays
   ≥90% — e.g. B200→H200 94.2%, H200→A100-80GB 98.1%, H100 NVL→H100-80GB
   90.9%, B200→A100-40GB 90.9% — demonstrating that classifiers trained on
   newer datacenter hardware detect training on older datacenter hardware
   without retraining. Cross-consumer and consumer↔datacenter transfer is
   weaker (the RTX 4050 Laptop is a clear outlier), motivating pooled
   training across hardware classes. Wall-power validation via smart plug
   confirms NVML is a trustworthy proxy for actual power consumption.

---

## 2. Background & Threat Model (~0.75 page)

### 2.1 Telemetry Signals

Our classifier operates on nine always-available NVML counters sampled at 1 Hz:

- GPU utilization (%)
- Memory utilization (%)
- Memory used (MB)
- Power draw (W)
- Temperature (°C)
- SM clock (MHz)
- Memory clock (MHz)
- PCIe TX bandwidth (MB/s)
- PCIe RX bandwidth (MB/s)

These counters are available on every NVIDIA GPU from Pascal onwards with zero
runtime overhead — polling them adds no measurable load to the monitored
workload, no kernel-level instrumentation, and no special driver configuration.
Reading them requires no access to application code, model weights, or training
data: the signals are purely physical/architectural.

We deliberately restrict ourselves to this always-available signal set, in
preference to richer alternatives such as DCGM (which adds tensor-core
activity, NVLink throughput, and ECC counters — datacenter-only, requires
persistent daemon) or kernel-level profiling via Nsight Compute (per-kernel
PMC reads at 1,200-5,300% overhead; impractical for production datacenters
already under power-management stress [Choukse et al. 2025]). This makes our
result a *lower bound*: anything that works at this minimal observational
budget works with more information, so a real deployment that can access
DCGM or layered host/wall-level power monitoring starts from at least the
detection we demonstrate here and only gets stronger.

### 2.2 Threat Models

**Assumptions common to both models.** Throughout, we assume that (i) the
adversary cannot disable NVML polling or tamper with the returned counter
values at the hardware level, and (ii) the adversary cannot inject synthetic
telemetry into the monitor's ingestion channel — i.e., cannot "spoof" a
different workload's signature to the verifier. Both assumptions describe
mechanisms the software stack alone cannot enforce; they are the minimum
hardware-enabled-mechanism (HEM) guarantees our results depend on
[Aarne et al. 2024, O'Gara et al. 2025]. § 6.5 discusses the HEM
requirements in more detail as a limitation and a direction for deployment.

**Threat model A — Cloud tenant.** A user renting GPU time attempts to
disguise ML training as inference or non-ML workloads to evade a cloud
provider's monitoring. The adversary controls user-space software but not
the host OS, hypervisor, or hardware. This threat model underpins recent
*compute-structuring* defences: Seferis & Fist (2025) argue that
structured-evasion attacks — e.g., splitting a training run across multiple
cloud providers to dodge per-provider reporting thresholds — are detectable
*if* each cloud host can classify the workloads running on its own GPUs.
Our classifier is precisely that building block. If tenants can disguise
training as another workload on a single host, the compute-structuring
defence collapses; our adversarial evaluation is a direct test of its
feasibility.

**Threat model B — Datacenter operator.** A datacenter operator attempts to
hide unauthorized AI training from an external regulator. The adversary
controls the full software stack and can modify workload scheduling, but
(per the HEM assumptions above) cannot tamper with hardware-level telemetry
or spoof signals. This is the governance-relevant case for frontier-model
compute oversight. Our multi-GPU DDP evasion experiments (K_ddp, L_ddp,
L_ddp_stagger, etc.; § 5.2) on 8-GPU datacenter nodes directly test it.

**Relationship.** Model B is strictly harder than Model A: the operator can
deploy any framework-level evasion *and* control scheduling. The fundamental
physical constraint behind both models — that the backward pass requires
~2× the FLOPs of the forward pass and therefore leaves a matching energy
signature (§ 6.1) — is the same. Evasion strategies designed against Model A
transfer to Model B because the constraint is physical, not software.

### 2.3 Classification Task

**The governance question is binary** — a regulator asking "is this GPU
training a model?" needs a training-versus-not-training decision, not a
fine-grained workload taxonomy. The operational form we deploy is a
**two-stage pipeline** that factors the decision along its natural tree:

- **Stage 1 — "Is this workload ML?"** A binary XGBoost head on the full
  166-feature window classifies `ml_training ∪ ml_inference` versus
  `other`. The ML-vs-non-ML boundary leans heavily on level signals
  (peak memory footprint, sustained tensor-core utilization, power draw),
  so the full feature set is retained.
- **Stage 2 — "Given ML activity, is it training or inference?"** A
  binary XGBoost head trained only on rows where the true label is
  `ml_training` or `ml_inference`, with level features dropped
  (absolute memory/power/clock means, percentiles, min, max, range). The
  train-vs-infer decision rests on shape, periodicity, FFT, and
  autocorrelation terms only. This restriction is the Round 5 WB-O
  defense (§ 4.4): an adversary who spoofs memory levels to disguise
  training as inference cannot drag Stage 2, because Stage 2 has never
  seen the levels.

Governance P(train) is the exact probabilistic composition
P(train) = P_stage1(ML) × P_stage2(train | ML), with
P(infer) = P_stage1(ML) × (1 − P_stage2(train | ML)) and
P(other) = 1 − P_stage1(ML); these sum to 1 by construction. Each stage
is independently calibratable — operators can tune the ML-gate
permissiveness (Stage 1 threshold) and the train-vs-infer cutoff
(Stage 2 threshold) separately. Implementation: `classifier/twostage.py`.

**We evaluate the strictly harder three-way task** (`ml_training` /
`ml_inference` / `other`) as the headline benchmark. A classifier that
correctly distinguishes all three categories necessarily solves the
binary governance question, so three-way accuracy is a conservative
lower bound on the binary-governance accuracy we would report. Three-way
is also the form in which the adversarial rounds and the
leave-one-strategy-out evaluation were run, so it is the form in which
our headline numbers (§ 5.1-5.3) are stated. Pooled 5-fold CV on the
refactored two-stage pipeline tracks the strictly-harder three-way
(XGBoost) number within 0.6-0.9 pp across 15-60 s windows — three-way
97.98% vs two-stage 97.14% at 30 s (Δ = −0.84 pp;
`classifier/results/twostage_vs_threeway.md`). On the Round-5 oracle
regression test (`whitebox_inference` held out at training time), the
two-stage's median P(train) on the held-out 20 inference windows is
0.040 (vs 0.056 for three-way), well below the governance threshold.
The two-stage classifier is the deployed artefact; the three-way
classifier is the evaluation vehicle.

Sliding windows of 5-60 seconds over 1 Hz telemetry. 166 features per window
(159 statistical base features + 7 window-level); detection-latency analysis
(Appendix H) spans 5-120 s, with 30 s as the operating point for headline
accuracy.

---

## 3. Related Work (~0.5 page)

- **GPU telemetry for AI governance (most directly related).** Monfared et
  al. (2026) study timing and memory observables for detecting training when
  host and device are untrusted, constructing their own derived signals from
  primitives that survive a strong tampering adversary. Their question is
  *what signals remain informative when NVML cannot be relied upon.* Ours is
  the complementary question: *given that trusted NVML is available (the
  deployment most compute-governance proposals assume, backed by HEM),
  what is the cost of evading it?* Their work establishes a lower floor on
  the signal set; ours establishes an upper bound on what an adversary can
  hide inside that signal set. The two frame different regions of the same
  design space and are directly complementary.
  Chen et al. (2025, "Reveal") propose unsupervised anomaly detection from
  hardware telemetry across 30+ ML models — relevant for flagging novel
  workloads rather than classifying known ones.

- **Compute governance:** Shavit (2023) proposes compute monitoring for
  verifying rules on large-scale training — the foundational paper for this
  line of work. Sastry, Heim et al. (2024) provide a comprehensive framework
  for compute as a governance lever. Heim & Koessler (2024) analyse FLOP
  thresholds as a regulatory instrument — the 10²⁶-FLOP line in U.S. EO
  14110 is a direct operationalisation of this. Baker et al. (2025) identify
  six verification layers for international AI agreements; our work provides
  empirical evidence for one layer (built-in monitoring). O'Gara et al.
  (2025) and the RAND workshop report (Kulp, Siu & Heim 2025) evaluate HEM
  options including workload attestation. Petrie (2024) proposes
  firmware-based offline licensing for export controls. Seferis & Fist
  (2025) argue that compute-structuring evasion (splitting training across
  providers) is detectable conditional on per-host workload classification;
  we supply and stress-test exactly that per-host classifier.

- **Workload fingerprinting:** WAVE [Xu et al. 2026] accomplishes a related
  task — inference-workload identification — with 87-93% accuracy using
  24 Nsight Compute per-kernel counters (1,200-5,300% overhead). We
  accomplish a similar-granularity classification using only always-available
  NVML physical counters at zero runtime overhead; the tasks are not
  identical (WAVE separates inference families; we separate
  training/inference/other), so the comparison is directional rather than
  head-to-head. Bhatele et al. (IPDPS 2026) characterise production GPU
  workloads using DCGM counters at scale.

- **GPU kernel validation:** ShadowScope [Jackowski et al. 2025] validates
  GPU kernel behavior via CUPTI counters. Targets kernel-level compliance
  rather than workload-level classification.

- **Security-oriented workload detection:** Gangwal et al. (CANS 2020) show
  that Hardware Performance Counters can detect covert cryptomining using
  RF/SVM classifiers. They note an adversarial limitation: evasion is
  possible through restricted mining or signature neutralization, but at a
  profitability cost — a direct precedent for our "evasion tax" framing.

- **GPU telemetry for reliability:** Esposito et al. (2025) combine NVML
  telemetry and performance counters to estimate GPU stress and reliability
  across workloads. Their results confirm that telemetry reveals meaningful
  workload differences, but the target is reliability estimation rather than
  governance-oriented training detection.

- **Physical side channels:** Maia et al. (USENIX Security 2022) show
  magnetic flux from GPU power cables reveals neural network topology.
  Arefin & Serwadda (CODASPY 2025) demonstrate USB/HDMI port power
  measurement can infer GPU activities. Latif et al. (2024) empirically
  measure power signatures of training on 8-GPU H100 nodes. These establish
  that physical signals carry workload-identifying information; we extend
  this to software-accessible telemetry for governance.

- **Differential Architecture** (anonymous, ISCA 2026) proposes hardware
  mechanisms to limit performance of targeted applications — a complementary
  enforcement approach to our detection approach.

---

## 4. Method (~1.5 pages)

### 4.1 Data Collection

**Hardware:** 9 GPU classes spanning 4 architecture generations:

| GPU              | Arch      | VRAM    | Context            | GPUs |
|------------------|-----------|---------|--------------------|------|
| RTX 4050 Laptop   | Ada       | 6 GB    | Local laptop       | 1    |
| RTX 5080          | Blackwell | 16 GB   | Local desktop      | 1    |
| RTX 5090          | Blackwell | 32 GB   | Local desktop      | 1    |
| A100 SXM4 40GB    | Ampere    | 40 GB   | Cloud 8-GPU node   | 8    |
| A100 SXM4 80GB    | Ampere    | 80 GB   | Cloud 8-GPU node   | 8    |
| H100 80GB HBM3    | Hopper    | 80 GB   | Cloud datacenter   | 8    |
| H100 NVL          | Hopper    | 94 GB   | Cloud datacenter   | 2    |
| H200              | Hopper    | 141 GB  | Cloud 8-GPU node   | 8    |
| B200              | Blackwell | 192 GB  | Cloud 8-GPU node   | 8    |

NVML distinguishes "H200" and "H200 NVL" only by marketing name — the two SKUs
share identical 141 GB HBM3e and identical telemetry signatures — so we collapse
them into a single H200 class. H100 80GB HBM3 and H100 NVL are kept separate
(distinct 80 GB HBM3 vs 94 GB HBM3 memory, distinct clock/power envelopes).

**Workloads:** 6 workload categories plus adversarial evasion:
- ML training: ResNet-18/50, MLP, BERT, GPT-2 (FP32 + AMP); DDP (8-GPU);
  FSDP sharded training (7B-70B LLaMA-architecture models); LoRA fine-tuning
  on phi-3, smollm2, qwen25-1.5B, mistral-7B.
- ML inference: ResNet-50 batch inference; LLM generation (DeepSeek-V3 TP8,
  Whisper); dense LLM inference at matched training sizes (Qwen 2.5 7B/14B/32B
  FP16, plus 72B FP16/FP8) on A100/H200/B200 8-GPU nodes.
- Scientific HPC: 3D cuFFT, N-body simulation.
- Crypto mining: Ethash proxy.
- Other: Monte Carlo rendering, ffmpeg NVENC, idle baseline.
- Adversarial evasion: 18 strategy families across 5 rounds of
  adversary-defender iteration (catalogued in § 4.4 and Appendix A),
  including 4 multi-GPU distributed evasion strategies.

**Telemetry:** 9 raw NVML signals at 1 Hz (GPU utilization, memory utilization,
memory used, power draw, temperature, SM clock, memory clock, PCIe TX/RX).
Wall power from Tapo P115 smart plug on RTX 5080 (18 workloads with
simultaneous NVML + wall-plug power).

**Scale:** **1,437 parquet files**, **1,424 labeled runs** (867 training,
352 inference, 205 other), **445,330 telemetry rows** across **~124 hours**
of GPU telemetry.

### 4.2 Feature Extraction

166 features per sliding window. Composition:
- **Base statistics** (9 signals × 13 stats = 117): mean, std, min, max,
  percentiles (25/50/75/95), IQR, range, CV, skew, kurtosis.
- **Autocorrelation** (3 signals × 5 lags = 15): lags 1/2/5/10/20 for
  utilization, power, memory.
- **Cross-signal:** power/utilization ratio, PCIe total, utilization per SM%.
- **Training-specific temporal:** memory slope (training monotonically grows
  allocated memory toward a plateau; inference reaches its plateau almost
  instantly), epoch periodicity via FFT of power.
- **Pre-allocation / memory-plateau features** (Round 5 addition):
  `first_30s_mem_delta_mb`, `first_30s_mem_delta_frac`, and
  `time_to_mem_plateau_samples`. These were added after a white-box adversary
  in Round 5 learned to spoof the steady-state memory signature; they capture
  the *transient* memory-allocation profile that a training workload cannot
  suppress without aborting the run.
- **Inference-specific:** duty cycle, idle fraction, utilization transitions.

### 4.3 Classifier

We pick the best-performing of two strong tree-ensemble baselines —
**RandomForest** [Breiman 2001] (n_estimators=400, max_depth=None,
min_samples_leaf=2, max_features="sqrt", class_weight="balanced") and
**XGBoost** (n_estimators=500, max_depth=6, lr=0.05, sample_weights via
effective-number-of-samples class balancing) — evaluated head-to-head at
15/30/60 s windows with stratified-group 5-fold CV (grouped by `run_id`
to prevent within-run leakage). RandomForest wins by a small margin at
every window size; at 30 s it reaches 98.42% ± 0.50% accuracy / 0.982
macro-F1. We report RF 30 s as the headline operating point, and retain
XGBoost only for gain-based feature-importance analysis (Fig 3).
Alternative classifier baselines (SVM-RBF, logistic regression) are
surveyed in Appendix G.

**Design rationale (feature-set layering):** The feature set is designed
along two signal layers: (1) a structural/roofline layer that identifies
workloads occupying a training-like operating region in arithmetic-
intensity space (mean power, mean utilization), and (2) a temporal
telemetry layer that resolves ambiguity within that region using
time-domain features (autocorrelation, CV, periodicity) [cf. Williams
et al. 2009 for the roofline model]. Both classifier heads (RF and XGB)
consume both layers simultaneously, and both Stages of the two-stage
pipeline (§ 2.3) use the same underlying feature vocabulary — Stage 1
retains the full set; Stage 2 drops the level features as the WB-O
defense.

### 4.4 Adversarial Evaluation Protocol

**Iterative refinement:** five rounds of adversary-defender co-evolution.
Each round: (1) the attacker designs new evasion strategies targeting the
current classifier's weaknesses; (2) the defender retrains a hardened
classifier incorporating the new adversarial examples (and, in some rounds,
adds new features or collects expanded benign data). The classifier passes
through three distinct states across the five rounds:

- **baseline** — trained only on non-adversarial data;
- **hardened** — baseline plus all adversarial families known up to that round;
- **evasion-resistant** — hardened plus physics-based features (power–memory
  correlation, cumulative energy, pre-allocation memory-plateau features) that
  capture physical invariants the adversary cannot suppress without aborting
  training.

**Round-by-round protocol:**

| Round | Attacker introduces                                                               | Defender response                                                                   |
|-------|-----------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| 1     | Utilization manipulation: A (util modulation), B (low util), D (temporal disruption) | Retrain → **hardened** classifier                                                   |
| 2     | Memory reduction + mimicry: E (memory minimal), F (interleave), G (clock throttle), H (mimicry cufft/mining), I (stochastic), J (PID control) | Retrain → **hardened** (extended)                                                    |
| 3     | Training-as-inference: K (online learning), L (diluted training at N ∈ {2,5,10,20}) | Add power-memory correlation + cumulative energy features → **evasion-resistant**   |
| 4     | Composite evasion: M (composite memmin), N (gradient accumulation), composite idle-padding sweep (idle fraction 0–95 %), plus 4 multi-GPU DDP variants (K_ddp, K_ddp_accum, L_ddp, L_ddp_stagger) on 8-GPU datacenter nodes | Retrain on multi-GPU data → **evasion-resistant** (extended)                         |
| 5     | White-box (attacker given classifier internals): WB-D memory-profile spoofing, WB-L label-smeared dilution, WB-O optimizer-phase hiding, WB + grad-accum combined | Add pre-allocation memory-plateau features (`first_30s_mem_delta`, `time_to_mem_plateau`); expand benign set with matched-size dense LLM inference (Qwen 2.5 7B/14B/32B on A100/H200/B200) |

This yields **18 evasion strategy families** covering five technique
classes: utilization manipulation, memory reduction, workload mimicry,
training-as-inference (including dilution and gradient accumulation),
multi-GPU distributed evasion, and white-box adversary. The 11-point
composite idle-padding sweep within Round 4 is expanded inline rather than
counted as separate families. Full per-strategy catalogue with description,
round introduced, technique class, and per-strategy detection rates is in
Appendix A.

**Leave-one-strategy-out (LOO) evaluation.** For each strategy family, we
train the classifier on clean data + every *other* adversarial family, then
test on the held-out family. This tests generalization to *truly unseen*
evasion techniques — the governance-relevant case where an adversary
invents a strategy the defender has not yet catalogued.

---

## 5. Results (~2.5 pages)

### 5.1 Baseline Classification

- **Pooled three-way accuracy (9 GPU classes):** 98.42% ± 0.50%, macro-F1 =
  0.982 (RandomForest, 30 s windows, stride 15 s, 5-fold grouped CV,
  26,496 windows across 1,424 runs).
- **Detection latency:** 97.76% at 5-second windows; 98.35% at 30-second;
  plateau across 15-120 s at 98.2-98.5% (Appendix H, Fig H1).
- **Per-GPU 5-fold CV accuracy** (diagonal of the pooled-training matrix):

  | GPU                | Pooled CV accuracy |
  |--------------------|--------------------|
  | A100-SXM4 40GB     | 99.93%             |
  | A100-SXM4 80GB     | 99.94%             |
  | B200               | 99.26%             |
  | H100 NVL           | 99.57%             |
  | H200               | 99.64%             |
  | H100 80GB HBM3     | 97.83%             |
  | RTX 5090           | 97.50%             |
  | RTX 5080           | 93.91%             |
  | RTX 4050 Laptop    | 87.65%             |

  Datacenter GPUs are at or above 99% on every class. The RTX 4050 Laptop
  sits outside the scope of practical compute governance — laptop consumer
  GPUs are not covered by any published compute-reporting threshold or HEM
  proposal [Shavit 2023, O'Gara et al. 2025, Heim & Koessler 2024] and are
  included here only as a diagnostic of how the classifier behaves on a
  thermally- and power-constrained envelope at the extreme low end. The
  consumer-desktop RTX 5080 / 5090 classes (93.9 % / 97.5 %) and all seven
  datacenter classes (≥ 97.8 %) cover the realistic governance deployment
  range.

**Figure 1: Cross-GPU transfer heatmap.** Per-cell value is accuracy when
training on the row GPU and testing on the column GPU. Datacenter-to-
datacenter transfer stays ≥90% (B200→H200 94.2%, H200→A100-80GB 98.1%,
B200→A100-40GB 90.9%, H100 NVL→H100-80GB 90.9%). Cross-consumer and
consumer↔datacenter transfer is weaker: RTX 5080/5090/4050 rows drop below
50% on many datacenter targets, and the RTX 4050 row peaks at only 92% even
on its own GPU. The bottom row of the heatmap is the pooled classifier,
which recovers ≥93.9% on every GPU (≥97.5% excluding the 4050 laptop).

*Key insight:* Within a tier (datacenter or consumer), the classifier
generalises without retraining. Across tiers, domain shift is substantial —
pooled training across diverse hardware is required to maintain accuracy on
heterogeneous fleets. This matches the governance use case, where a
regulator training on public datacenter telemetry can detect training on
similar datacenter GPUs without per-site retraining.

**Figure 3: Feature importance** (top 15 XGBoost features by gain). The two
Round-5 memory-plateau additions — `time_to_mem_plateau_samples` (gain ≈
0.082) and `first_30s_mem_delta_mb` (0.065) — dominate the classifier by a
wide margin. Memory-footprint features (`mem_used_mb` p95 / p75 / max,
`mem_utilization_pct` min / max) fill most of the next tier; PCIe traffic
statistics (`pcie_rx_mbps` IQR / skew, `pcie_total_mean`) and GPU-utilization
statistics (IQR, mean, min, CV, per-SM) round out the top 15.

**The story the classifier tells:** training has a distinctive
memory-allocation *dynamics* signature — memory climbs toward a ceiling
during the first seconds (the backward pass needs gradients, activations,
and optimiser state) and then plateaus; inference allocates its
steady-state footprint almost instantly. This *transient* is what the
top-ranked features encode. Memory *footprint* (how much) is secondary;
PCIe traffic patterns (how the workload moves data) and GPU-utilization
shape (not level) are tertiary.

**Surprises.** Absolute power features (`power_draw_w` statistics,
power autocorrelation) do **not** appear in the top 15 — not because
training has no energy signature (§ 6.1 argues it must), but because
absolute power envelopes vary substantially across the 9 GPU classes in
the pooled training set, so the classifier leans on the more
class-invariant memory-dynamics signal instead. This is itself a
cross-architecture-generalisation finding: the features that transfer
across GPU classes are the ones that describe *how a workload allocates
and moves memory over time*, not how much power it happens to draw on a
particular die.

**False positive rate (confusion matrix, RF 30 s, Fig Appendix E):**
Training-class precision is 0.994 and recall 0.994; inference precision
0.973; other precision 0.991. The dominant residual error is
non-ML ↔ training mis-assignment, not training ↔ inference — consistent
with the governance objective, since inference/training confusion is the
more benign error mode for a "is this GPU training?" alert.

### 5.2 Adversary-Defender Iteration

**Table 2: Arms race summary** — per-family detection rates by classifier
state (class-balanced; `baseline` / `hardened` / `evasion-resistant`
correspond to the three-way classifier states introduced in § 4.4; rates
from `adv_defender_rounds.csv`, `leave_one_out.csv`, and the two-stage
parity runs `adv_defender_rounds_twostage.csv` /
`leave_one_out_twostage.csv`). LOO columns report family-mean detection
over all variants; "3-way" is the three-way XGBoost evaluation vehicle,
"2-stage" is the deployed pipeline (§ 2.3):

| Round | Attacker capability                                                | Baseline det. | Hardened det. | Ev.-resistant | LOO 3-way | LOO 2-stage |
|-------|--------------------------------------------------------------------|---------------|---------------|---------------|-----------|-------------|
| 1     | Utilization manipulation (A, B, D)                                  | 30-95%        | 100%          | 100%          | 61-97%    | 65-92%      |
| 2     | Memory reduction / mimicry (E, F, G, H, I, J)                      | 2-47%         | 100%          | 100%          | 82-100%   | 95-100%     |
| 3     | Training-as-inference (K online, L diluted)                         | 89-100%       | 100%          | 100%          | 93-99%    | 85-100%     |
| 4     | Composite + grad accum (M, N, composite 0–95 % idle)                | 0-100%        | 100%          | 100%          | 93-100%   | 92-100%     |
| 4-DDP | Multi-GPU distributed (K/L DDP + stagger)                           | 98-100%       | 100%          | 100%          | 99-100%   | 100%        |
| 5     | White-box (WB-D/L/O, WB+grad-accum) — classifier internals exposed  | —             | —             | 100%          | 46-94%    | 35-79%      |

All classifiers use class-balanced training (effective-number-of-samples
sample weights). The "Ev.-resistant" column is three-way R3 in-distribution
detection, where the held-out family is *in* training; both classifier
forms achieve 100% in this state on every R1-R4 family. LOO columns are
the generalisation test: the family is held out, and the classifier sees
it for the first time at test time. R1 family-mean drops (below 97% for
three-way on `B_low_util` and `D_temporal`) are a consequence of adding
Round-5b whitebox data to the training pool — when a held-out family is
low-utilization training, the classifier learns to associate low
utilization with the Round-5b LoRA inference signature, hurting
generalisation. Rescue (§ 2.3) recovers most of this on the two-stage
side. Round-5 `whitebox_lora` remains the dominant LOO weakness for
both classifiers (46% / 35% family mean) — low trainable-parameter
footprint + heavy dilution is the current frontier for LOO
generalisation.

**LOO evaluation — 93.9% (three-way) / 95.3% (two-stage) unweighted
family mean** across the 18 R1–R4 strategy families; 91.6% / 91.5% when
the two Round-5 whitebox families are included (20 families total, 51
variant rows). Each family averages its own variant rows; see
Appendix A for the per-family table. Key facts:

- **Single family below 50% detection:** `whitebox_lora` at 46.3%
  three-way / 35.2% two-stage. Every other family lands above the
  coin-flip threshold. Within `whitebox_lora`, the `_N20` variant
  (20× dilution) is the worst variant for both classifiers
  (11.1% three-way / 0.0% two-stage) — a combination of extreme
  dilution and low trainable-parameter footprint that strips the
  training signature almost completely.
- **Families below 90% (three-way):** `B_low_util` (60.6%),
  `D_temporal` (67.9%), `F_interleave` (82.1%), `whitebox_lora`
  (46.3%). **Families below 90% (two-stage):** `B_low_util` (65.2%),
  `L_diluted` (84.5%), `whitebox_full` (79.1%), `whitebox_lora`
  (35.2%). The two-stage rescue rule (§ 2.3) recovers
  `A_util_modulation`, `D_temporal`, `E_memory_minimal`, and
  `composite` vs a no-rescue baseline, at a cost on Round-5
  whitebox families (where Stage 2's shape-only feature restriction
  discards level information that would help on heavy LoRA dilution).
- **Weakest *variant rows* (three-way):** `whitebox_lora_N20` 11.1%,
  `composite_10` 50.0%, `B_low_util` 60.6%, `whitebox_lora_N5` 61.1%,
  `whitebox_diluted_N50` 61.1%. The composite, B_low_util, and
  whitebox_full variants all recover to 100% in the hardened and
  evasion-resistant in-distribution states.
- **Datacenter adversarial results:** All 4 DDP evasion families
  (K_ddp, K_ddp_accum, L_ddp, L_ddp_stagger) are detected at
  **98.4–100% in LOO** for both classifier forms — multi-GPU evasion
  is *harder* to evade than single-GPU evasion, not easier. The
  NVLink/AllReduce communication pattern of distributed training is a
  strong, hard-to-suppress signal that shows up on every rank's PCIe
  and power traces.
- Variant-level mean = 92.0% (three-way) / 88.7% (two-stage) across
  all 51 variants; we report the unweighted family mean as the more
  adversarially conservative number, since DDP variants dominate
  window count and would otherwise mask the long-tail whitebox misses.

**Figure 4: LOO detection rate per strategy family** (bar chart, 20
families sorted by detection rate; error bars show min/max across
variants). Visually highlights `whitebox_lora` as the outlier below
the coin-flip threshold, `B_low_util` / `D_temporal` /
`whitebox_full` as the second-tier sub-90% cluster, and all 4 DDP
families at ≥98.4%.

### 5.3 The Evasion Tax

**Figure 2: Dilution curve** (`dilution_curve.{png,pdf}`). Primary X-axis:
dilution N (1, 2, 3, 5, 10, 20) — the multiplicative cost factor the
adversary pays to stretch a training run over N × its natural wall-clock
time. Secondary top X-axis: training throughput fraction (100 %, 50 %,
33 %, 20 %, 10 %, 5 %). Y-axis: detection rate at 5 s windows. Three
lines: *baseline* classifier (dips sharply at N = 3 and N ≥ 10), *LOO*
with the specific dilution level held out of training (intermediate),
and *hardened* class-balanced classifier (flat at 100 % across all
dilutions). The 50 % threshold is dashed.

**Table 3a: Controlled dilution sweep** (ResNet-50, batch 128, RTX 5080,
class-balanced classifiers; source: `dilution_sweep_eval.csv`):

| Dilution N | Throughput | Baseline 5s | LOO 5s | Hardened 5s | LOO 15s | Cost mult. |
|-----------|-----------|-------------|--------|-------------|---------|------------|
| 1 (full)  | 100%      | 98.4%       | 100%   | 100%        | 100%    | 1x         |
| 2         | 50%       | 100%        | 100%   | 100%        | 100%    | 2x         |
| 3         | 33%       | 67.2%       | 80.3%  | 100%        | 96.7%   | 3x         |
| 5         | 20%       | 93.4%       | 98.4%  | 100%        | 100%    | 5x         |
| 10        | 10%       | 68.9%       | 80.3%  | 100%        | 90.0%   | 10x        |
| 20        | 5%        | 44.3%       | 72.1%  | 100%        | 100%    | 20x        |

**Table 3b: Composite idle-padding sweep** (ResNet-50 training interleaved
with idle padding at the given fraction; source:
`adv_defender_rounds.csv`, composite rows):

| Idle fraction | Baseline | Hardened | Evasion-resistant |
|---------------|----------|----------|-------------------|
| 0%            | 95.2%       | 100%        | 100%                  |
| 10%           | 100%        | 100%        | 100%                  |
| 20%           | 100%        | 100%        | 100%                  |
| 30%           | 90.5%       | 100%        | 100%                  |
| 40%           | 100%        | 100%        | 100%                  |
| 50%           | 66.7%       | 100%        | 100%                  |
| 60%           | 50.0%       | 100%        | 100%                  |
| 70%           | 61.9%       | 100%        | 100%                  |
| 80%           | 0.0%        | 100%        | 100%                  |
| 90%           | 33.3%       | 100%        | 100%                  |
| 95%           | 52.4%       | 100%        | 100%                  |

The baseline classifier has a clear weak spot at 60-90% idle — composite
evasion at 80% idle completely defeats a classifier that has never seen
this strategy. **One round of retraining (hardened classifier) closes this
gap entirely: 100% detection across all idle fractions, including the
80-90% range.** The evasion-resistant feature set (hardened classifier +
power-memory correlation + cumulative energy + first-30 s memory-delta
features) holds 100% detection without any regression.

**Economic argument:** Even before hardening, the economic cost is
prohibitive. Running training at 5% throughput (N=20 dilution, 20× cost
multiplier) achieves only 44% baseline and 72% LOO detection — meaning a
sustained covert run has ≈50-70% chance of being flagged in any given
5-second window, and essentially certain detection over the hours-to-weeks
timescale of a real training run. After hardening, the tax becomes
absolute: there is no dilution or idle-fraction setting at which the
hardened classifier falls below 100%.

Cottier et al. (2024, Epoch AI) report that the cost of training frontier
models has grown at ~2.4× per year since 2016 and project the largest
training runs to exceed **$1 billion by 2027**; recent flagships are
already in the $40-100 M range on a hardware-amortised accounting. Against
this baseline, even a modest 2× cost multiplier — the threshold at which
the hardened classifier's detection exceeds 99 % across the dilution sweep
— moves a frontier run from ~$100 M to ~$200 M and from ~$1 B to ~$2 B,
amounts that are hard to hide in a financial audit [Baker et al. 2025].
Frontier training power demand is growing at a comparable rate, with
individual training runs projected to require 4-16 GW by 2030 [You & Owen,
EPRI 2025]. The wall-clock time expansion that accompanies dilution
(running at 10 % throughput for the same FLOP count means 10× wall-clock
time) further multiplies the monitoring exposure window. Taken together,
the cost premium and extended exposure make covert frontier training
economically unviable under trusted telemetry.

Critically, **multi-GPU distributed training — the governance-relevant case
for frontier models — is detected at 98-100%** across all conditions,
including balanced LOO evaluation on both classifier forms. The
single-GPU evasion results are relevant for smaller-scale unauthorized
training but do not undermine the core governance use case.

### 5.4 Wall-Power Validation

- Tapo P115 smart plug wall power collected simultaneously with NVML for 18
  workloads on RTX 5080 (source: `wall_power_correlation.csv`).
- System idle baseline: ~48 W wall power at GPU idle (~22 W NVML + ~26 W
  system overhead).
- **Strong correlation between NVML `power_draw_w` and wall power**:
  Pearson *r* = 0.90, *R²* = 0.81 (*p* = 4 × 10⁻⁷). Linear fit wall ≈
  1.47 × NVML + 28 W.
- Per-workload overhead (wall − NVML) is 68-80 W for clean ML workloads
  (consistent PSU efficiency + CPU/RAM background), rising to ~218 W for
  `adversarial_K_online_learning` because that strategy deliberately drives
  the CPU to sustained 100% to mask training on the GPU — a side effect
  that is itself an evasion-detection signal, observable at the wall.
- NVML is a trustworthy proxy for actual GPU power consumption; hardware-
  level tampering (not software-level evasion) would be needed to spoof it.
  The wall-level signal is in fact *richer* than the GPU-level signal,
  because it captures CPU-side helper processes that adversarial evasion
  often requires.

---

## 6. Discussion (~1 page)

### 6.1 The Involuntary Witness

The GPU microarchitecture acts as an involuntary witness. The backward pass
of gradient-based training requires roughly 2× the FLOPs of the forward
pass (a standard accounting relation from the neural-scaling-laws
literature [Kaplan et al. 2020]; this is a conceptual prior, not a result
of our experiments), so any training run of a given model leaves a
physical energy signature that cannot be eliminated without eliminating
training itself. An adversary can redistribute this signature over time
(dilution, interleaving) but cannot remove it — only amortise it across
more wall-clock time and a correspondingly larger exposure window.

Our empirical evidence is *consistent* with this framing rather than
demonstrating it directly. The feature-importance analysis (Fig 3) shows
that the classifier leans most heavily on memory-allocation *dynamics* —
time-to-memory-plateau and first-30 s memory-delta — followed by
memory-footprint percentiles and PCIe traffic patterns. This is a
downstream observable of the same physical constraint: training must
allocate space for gradients, activations, and optimiser state, and that
allocation has a distinctive transient profile that inference (which
reaches its steady-state footprint almost immediately) does not share.
Notably, absolute power statistics do not appear in the top 15 features,
because power envelopes vary across GPU classes in our pooled training
set — the classifier has instead learned a more class-invariant,
memory-behaviour-centred discriminator.

### 6.2 Threat Model Comparison

Our datacenter experiments (8-GPU A100, H200, B200 nodes) directly test the
operator threat model. We designed 4 multi-GPU evasion strategies: DDP online
learning, DDP diluted training, staggered dilution (spreading NVLink traffic
across time), and gradient accumulation across ranks. All are detected at
98.4–100% in leave-one-strategy-out evaluation (both classifier forms;
see Table 2). Multi-GPU evasion is no harder
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

If telemetry is collected and checked at multiple layers — device-level
(NVML), host-level (OS counters), and infrastructure-level (network,
power) — an adversary must evade all layers simultaneously [O'Gara et
al. 2025]. The evasion cost is multiplicative across layers: our
always-available-NVML results are a lower bound on what a layered monitor
can achieve. DCGM counters (tensor-core activity, NVLink throughput) and
wall- or rack-level power measurements provide orthogonal signal — our
wall-power validation (§ 5.4, Appendix D) already shows that
adversarial strategies relying on CPU-side helper processes are visible
at the wall even when they suppress GPU-level telemetry. Combined with
tamper-resistant hardware (HEM per Aarne et al. 2024; see § 6.5),
software-level evasion alone cannot defeat a layered monitor. This
structural advantage compounds the evasion tax: even if an adversary
discovers a strategy that reduces detection in one layer, the remaining
layers provide independent verification.

### 6.5 Implications for Hardware-Enabled Mechanisms

Our results pair naturally with a minimal HEM design. The software-level
classifier demonstrates that always-available NVML counters at 1 Hz carry
enough signal to detect training-versus-not at 98 % three-way accuracy
with LOO adversarial robustness — but this result only holds if the
counters themselves are trustworthy. A deployable monitoring system
therefore requires three HEM components in addition to the classifier:

1. **Tamper-resistant NVML read path.** The monitoring agent must read
   counter values from a source the adversary cannot tamper with — e.g.,
   an attested enclave on the GPU firmware boundary (as in Aarne et al.
   2024) or a verified read path through secure-boot-chained firmware
   [Petrie 2024, O'Gara et al. 2025]. Without this, an adversary with
   software privilege simply returns forged values.
2. **Authenticated telemetry channel.** Telemetry in transit from the
   GPU agent to the verifier must be signed with a hardware-rooted key
   so an in-path adversary cannot substitute a "benign" workload's
   telemetry for the actual workload. This extends to rate-limiting and
   nonce-binding to prevent replay of previously-observed benign
   signatures.
3. **Secure boot of the monitoring agent.** The agent itself, its
   classifier weights, and its feature-extraction code must be measured
   and attested so an adversary cannot replace them with a permissive
   version (e.g., a classifier that labels everything as inference).

Signal content that a minimal HEM should expose (from the union of our
feature-importance analysis in Fig 3 and the layered-defence argument in
§ 6.4):
- **Memory-allocation dynamics** (how memory grows during the first seconds
  of a workload, and when it plateaus) — top-ranked features in our
  classifier.
- **Memory usage** statistics (percentiles, min/max) — a secondary-tier
  contributor and a prerequisite for computing the dynamics above.
- **PCIe traffic patterns** — mid-tier contributor; captures
  batch-loading and gradient-sharing signatures that differ between
  training and inference.
- **GPU utilization** statistics (shape, not just mean) — mid-tier
  contributor; most cross-architecture-stable single signal.
- **Power draw** — not in our top-15 features (power envelopes vary too
  much across GPU classes for pooled classification), but retained here
  because it is the cheapest independent physical check at the
  infrastructure layer (wall/rack power; § 5.4, Appendix D) and because
  the backward-pass FLOP overhead (§ 6.1) must manifest in power
  eventually.
- **Temperature dynamics** — physics-constrained and architecture-stable.
- **NVLink / interconnect utilization** — for multi-GPU detection
  (DDP/FSDP signatures).

Minimum viable sampling rate: 1 Hz (our classifier reaches headline
accuracy at 1 Hz). Minimum detection window: 5 seconds (97.76 % three-way
accuracy). Adding DCGM counters (NVLink throughput, tensor-core activity)
is expected to further improve distributed-training detection but is not
required to reach the numbers reported in this paper.

### 6.6 Limitations

- **Single-node multi-GPU only:** We test DDP and FSDP evasion on 8-GPU
  nodes (A100, H200, B200). Multi-node distributed training (pipeline-parallel
  across machines) may have different evasion characteristics due to network
  (rather than NVLink) communication patterns.
- **Kernel-level evasion:** An adversary with access to custom CUDA kernels
  could potentially manipulate individual kernel behavior. We test only
  framework-level (PyTorch) evasion.
- **Telemetry-channel integrity (HEM requirement).** Our entire evaluation
  assumes the adversary cannot (a) disable NVML polling or tamper with the
  returned counter values at the hardware level, and (b) inject synthetic
  telemetry into the verifier's ingestion channel to spoof a different
  workload's signature. Both properties are outside the software stack's
  ability to enforce — they are exactly the guarantees that a hardware-
  enabled mechanism [Aarne et al. 2024, O'Gara et al. 2025] is designed to
  provide. Without tamper-resistant telemetry and an authenticated
  telemetry channel, a sufficiently privileged adversary bypasses software-
  level evasion and simply feeds a forged signal to the monitor. Deployment
  therefore requires *both* our software-level detection capability and a
  minimal HEM that guarantees the signal's integrity; § 6.5 sketches the
  minimal HEM design.
- **Model scale of evasion experiments:** The 18 adversarial *evasion*
  strategies themselves use small models (ResNet-18/50, 25M params), chosen
  so an experiment across 18 strategies × 4 rounds × multiple GPUs fits
  within the project's compute budget. The *defender-tier Round 5b* data,
  by contrast, includes real LLM training and inference at matched dense
  sizes (Qwen 2.5 7B/14B/32B FP16, plus 72B FP16/FP8 inference, and
  FSDP/DeepSeek-V3 training at equivalent scale), which is the regime where
  training and inference telemetry signatures are expected to converge most.
  The headline 98.4% accuracy is computed on this combined dataset. Whether
  an adversary running *evasion* on a 70B-scale model can exploit extra
  degrees of freedom not available at 25M scale remains open.
- **Workload diversity:** We test 6 workload categories (ML training, ML
  inference, HPC, crypto mining, rendering/encoding, idle) plus 18 adversarial
  strategy families. Real datacenters run a wider variety — particularly
  inference-serving traffic with bursty request patterns, specialized
  codecs, and emerging MoE workloads — and the classifier may need
  retraining for new workload classes.

---

## 7. Conclusion (~0.5 page)

Zero-overhead GPU telemetry monitoring is a viable mechanism for AI compute
governance. Defenders are structurally favored for the governance-relevant case:
(1) **multi-GPU distributed evasion is detected at 98-100%** across all
conditions including balanced LOO evaluation on 8-GPU datacenter nodes —
the NVLink/AllReduce signatures of distributed training are extremely difficult
to suppress; (2) single-GPU evasion partially succeeds against a balanced
classifier at a 93.9% (three-way) / 95.3% (two-stage) unweighted family
mean on the 18 R1–R4 strategy families, with Round-5 `whitebox_lora` the
sole family below 50% (LoRA's low trainable-parameter footprint plus
heavy dilution is the open frontier); evasion-resistant physics-based
features (pre-allocation memory-plateau, power-memory correlation,
cumulative energy) close the remaining gaps for the R1–R4 families in
the in-distribution hardened state; (3) one round of hardening brings
composite idle-padding detection to 100% at every idle fraction from 0%
through 95%, imposing an absolute evasion tax. These results hold across
the 8 GPU classes in the governance-relevant deployment range (datacenter
and consumer desktop), covering 4 architecture generations from Ampere
through Blackwell.

**Governance takeaway.** The compute-monitoring proposals of Shavit
(2023), Baker et al. (2025), and O'Gara et al. (2025) — and the policy
instruments (U.S. EO 14110; any successor FLOP-threshold regime
[Heim & Koessler 2024]) that depend on per-GPU workload classification —
are empirically realisable against a realistic software-level evasion
adversary, *conditional* on the two HEM assumptions stated in § 2.2:
tamper-resistant telemetry and an authenticated telemetry channel. With
those hardware guarantees in place, the adversary is left with a
physical-cost trade-off (dilution and idle-padding) whose economics do
not favour them at frontier scale. The open research frontier is not
"can we build a classifier?" — we can — but "can we build the matching
HEM?", which is a hardware-security problem, not a machine-learning one.

---

## Appendices

### A. Full Adversarial Strategy Catalog

Per-family detection rates aggregated from `adv_defender_rounds.csv`,
`leave_one_out.csv`, and the two-stage parity run
`leave_one_out_twostage.csv`. Families ordered by round; `baseline` /
`hardened` / `evasion-resistant` columns report the three-way min–max
range across variants (in-distribution); LOO columns report the
unweighted mean across variants (family mean) and the weakest variant
for each classifier form. DDP variants collected in the multi-GPU block;
Round-5 whitebox families added at the bottom.

| Family | Round | Technique | Baseline | Hardened | Ev.-resistant | LOO 3w mean | LOO 3w worst | LOO 2s mean | LOO 2s worst | Variants |
|---|---|---|---|---|---|---|---|---|---|---|
| `A_util_modulation`     | 1 | Utilization manipulation | 95 %        | 100 % | 100 % | 97.0 %     | 97.0 %             | 92.4 %     | 92.4 %              | 1 |
| `B_low_util`            | 1 | Utilization manipulation | 30 %        | 100 % | 100 % | **60.6 %** | **60.6 %**         | **65.2 %** | **65.2 %**          | 1 |
| `D_temporal_disruption` | 1 | Utilization manipulation | 91 %        | 100 % | 100 % | **67.9 %** | **67.9 %**         | 91.7 %     | 91.7 %              | 1 |
| `E_memory_minimal`      | 2 | Memory reduction         | 23 %        | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 1 |
| `F_interleave`          | 2 | Workload mimicry         | 2-16 %      | 100 % | 100 % | **82.1 %** | **64.3 %** (30)    | 99.1 %     | 98.2 % (50)         | 2 |
| `G_clock_throttled`     | 2 | Utilization manipulation | 30 %        | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 1 |
| `H_mimicry`             | 2 | Workload mimicry         | 29-30 %     | 100 % | 100 % | 99.2 %     | 98.4 % (mining)    | 96.0 %     | 93.8 % (mining)     | 2 |
| `I_stochastic`          | 2 | Utilization manipulation | 47 %        | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 1 |
| `J_pid`                 | 2 | Utilization manipulation | 11 %        | 100 % | 100 % | 100 %      | 100 %              | 94.6 %     | 94.6 %              | 1 |
| `K_online_learning`     | 3 | Training-as-inference    | 100 %       | 100 % | 100 % | 92.9 %     | 92.9 %             | 100 %      | 100 %               | 1 |
| `L_diluted`             | 3 | Training-as-inference    | 89-100 %    | 100 % | 100 % | 99.2 %     | 98.4 % (N=5)       | **84.5 %** | **76.1 %** (N=20)   | 4 |
| `M_composite_memmin`    | 4 | Composite                | 100 %       | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 4 |
| `N_grad_accum`          | 4 | Training-as-inference    | 100 %       | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 3 |
| `composite` (idle-pad)  | 4 | Composite idle-padding   | **0-100 %** | 100 % | 100 % | 92.6 %     | **50.0 %** (10)    | 92.0 %     | **66.7 %** (60)     | 11 |
| `K_ddp`                 | 4 | Multi-GPU DDP            | 100 %       | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 1 |
| `K_ddp_accum`           | 4 | Multi-GPU DDP            | 100 %       | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 2 |
| `L_ddp`                 | 4 | Multi-GPU DDP            | 99-100 %    | 100 % | 100 % | 99.0 %     | 98.4 % (5)         | 99.8 %     | 99.6 % (5)          | 2 |
| `L_ddp_stagger`         | 4 | Multi-GPU DDP            | 100 %       | 100 % | 100 % | 100 %      | 100 %              | 100 %      | 100 %               | 1 |
| `whitebox_full`         | 5 | Whitebox dilution + ckpt | —           | —     | 100 % | 94.4 %     | 61.1 % (N50)       | **79.1 %** | **55.0 %** (oracle) | 8 |
| `whitebox_lora`         | 5 | Whitebox LoRA + dilution | —           | —     | 100 % | **46.3 %** | **11.1 %** (N20)   | **35.2 %** | **0.0 %** (N20)     | 3 |

Bolded cells mark LOO family means or worst variants below 90 %.
Round-5 whitebox families were introduced in Round 5b and added to the
LOO harness after the original three-way LOO run; both classifier forms
struggle with `whitebox_lora` (LoRA's small trainable-parameter footprint
combined with heavy dilution strips the training signature to the noise
floor), which is the primary remaining adversarial-research frontier.
`B_low_util` and `D_temporal` regressed on three-way LOO after the
Round-5b data was added to training pools; the two-stage rescue rule
(§ 2.3) recovers `D_temporal` but not `B_low_util`. All R1–R4 families
are detected at 100 % in the in-distribution hardened and
evasion-resistant states.

### B. Feature List and Signal Examples

**Figure B1** (`telemetry_traces_4panel.{png,pdf}`): illustrative 120-second
1-Hz NVML traces on RTX 5080 for four workload types, showing GPU
utilisation (solid line), power (dashed), and memory (orange fill) on
each panel:

1. **Training — GPT-2 on WikiText-2**: bursty utilisation spiking to
   ~100 %, power to ~300 W, memory stepping up toward ~15 GB. The
   characteristic forward-backward-optimiser rhythm is visible directly
   in the trace, as is the monotonic memory ramp that drives the top
   features in Fig 3.
2. **Inference — Qwen2.5-1.5B**: steady-state ~45 % utilisation, ~100 W
   power, ~5 GB memory. Almost featureless after the initial warm-up —
   the workload reaches its memory plateau in seconds, the opposite of
   training.
3. **Evasion — `H_mimicry` (training disguised as mining)**: ~40 %
   utilisation, ~110 W power, ~2 GB memory. Visually, this panel is
   **difficult to tell apart from the inference panel** by inspection —
   this is exactly why a hand-tuned threshold is insufficient and a
   learned classifier is necessary. The memory-dynamics features in
   Fig 3 (memory-delta in the first 30 s, time-to-plateau) are what
   separate this from true inference.
4. **Non-ML — ethash mining**: flat ~25-30 % utilisation, ~70 W, ~2.5 GB.
   Very distinct from all three ML panels.

**Full 166-feature specification** with extraction formulas: 117 base
statistics (9 NVML signals × 13 stats = mean, std, min, max,
percentiles 25/50/75/95, IQR, range, CV, skew, kurtosis), 15
autocorrelation features (3 signals × 5 lags at 1/2/5/10/20 samples),
cross-signal ratios (power/util, PCIe total, util per SM), training-
specific temporal features (memory slope, FFT epoch periodicity),
pre-allocation memory-plateau features (Round 5 additions:
`first_30s_mem_delta_mb`, `first_30s_mem_delta_frac`,
`time_to_mem_plateau_samples`), inference-specific duty-cycle features,
and 7 window-level features.

### C. Cross-GPU Feature Stability

Per-feature coefficient of variation across the 9 GPU classes, sorted by
stability. Temperature and utilization features are most stable; clock
and PCIe features are architecture-specific and benefit from
pooled-training normalization.

### D. Wall-Power Validation

**Figure D1** (`wall_power_scatter.{png,pdf}`): scatter of Tapo P115 wall
power vs. NVML `power_draw_w` for 18 workloads on the RTX 5080.

- Pearson *r* = 0.899, *R²* = 0.81 (*p* = 4 × 10⁻⁷). Linear fit:
  wall ≈ 1.47 × NVML + 28 W.
- Clean ML workloads (ResNet-50 inference, GPT-2 training, BERT,
  MLP, etc.), HPC (cuFFT, N-body), mining, and rendering all sit within
  ±10 W of the fit line — the NVML signal explains the wall signal well,
  with a consistent ~68-80 W system overhead (PSU efficiency + CPU/RAM
  background).
- Adversarial strategies that rely on CPU-side helper processes — most
  dramatically `adversarial_K_online_learning` (~218 W overhead) and
  `adversarial_L_diluted_5`/`L_diluted_10` (~200 W overhead) — sit
  substantially *above* the fit line. These strategies deliberately
  shift work off the GPU to reduce NVML-visible telemetry, but the extra
  CPU load is visible at the wall.
- NVML is a trustworthy GPU-power proxy; hardware-level (not
  software-level) tampering would be needed to spoof it. The wall signal
  is *richer* than NVML alone, because it exposes exactly the CPU-side
  helper processes adversarial evasion commonly spawns. A
  layered-monitoring system that combines NVML with wall or
  rack-level power will detect evasion that NVML alone misses.

### E. False Positive Analysis

3×3 confusion matrix for the pooled classifier (RF, 30 s, 9 GPU classes,
grouped 5-fold CV), from `confusion_matrix_rf30s.png`. Training-class
precision 0.994 / recall 0.994; inference precision 0.973; other precision
0.991. The dominant residual error is non-ML ↔ training misclassification
(rare), not training ↔ inference. Governance implication: the remaining
error mode is detectable-by-audit — a flagged non-training workload can be
disputed with additional context, while the few missed training runs are
concentrated in the most heavily diluted, short-window cases that are
already uneconomic for an adversary to sustain.

### F. Roofline Overlap Analysis

Static architectural heuristics are insufficient for training detection.
286-configuration corner-case sweep on H100 NVL, spanning 6 workload families:
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

**Figure F1** (`roofline_corner_cases.{png,pdf}`): H100-NVL roofline with
the 286 corner-case inference configurations overlaid on a dense-FP32
training reference sweep. Colour-coded by workload family (CNN inference,
LLM prefill, LLM decode, quantisation variants, forward-only training
variant, ViT inference). The CNN-inference, LLM-prefill, and quantisation
clouds land squarely inside the training AI band, confirming visually
that static AI-based heuristics cannot separate these inference workloads
from training.

### G. Classifier Comparison

Head-to-head evaluation at 30 s windows, 5-fold grouped CV on the full
post-Round-5b dataset (26,496 windows across 1,396 runs with usable
feature vectors). We compare two tree-ensemble baselines — RandomForest
[Breiman 2001] and XGBoost — with two non-ensemble baselines — SVM-RBF
(RBF kernel, standardised features, balanced class weights) and Logistic
Regression (saga solver, L2 regularisation, standardised features).

| Classifier     | Accuracy             | Macro-F1  | CV fit time (5 folds) |
|----------------|----------------------|-----------|-----------------------|
| RandomForest   | **98.42 % ± 0.50 %**  | **0.982** | 66 s                  |
| XGBoost        | 98.13 % ± 0.52 %      | 0.976     | 67 s                  |
| SVM-RBF        | 95.19 % ± 1.19 %      | 0.945     | 103 s                 |
| Logistic Reg.  | 88.27 % ± 0.89 %      | 0.873     | 506 s                 |

The tree ensembles dominate by ~3 pp over SVM-RBF and ~10 pp over
logistic regression, which is consistent with the feature set carrying
non-linear interactions that a kernel-SVM or linear classifier cannot
exploit as cheaply as a tree ensemble. SVM-RBF at 95.2 % is a useful
cross-check that the tree-ensemble headline is not an artefact of the
specific algorithm family. The drop to 88.3 % for logistic regression
quantifies the non-linearity of the problem: a purely linear decision
boundary in the 166-feature space is ~10 pp worse, confirming that the
classifier is exploiting feature interactions (power CV × memory slope,
autocorrelation × FFT periodicity, etc.) not just per-feature
thresholds.

Numbers in the table above come from
`classifier/results/classifier_comparison.csv`, produced by
`regen_classifier_comparison.py` at the project root.

**Figure G1** (`classifier_comparison.{png,pdf}`) shows the same data as
side-by-side accuracy / macro-F1 bars with fold-wise error bars. A prior
evaluation on Sabiha's H100-NVL corner-case benchmark also showed SVM-RBF
at 15 s reaching 95.65 % binary accuracy and 92.89 % macro-F1 on the
hardest corner cases — consistent with the ordering we see here.

### H. Window Size vs Accuracy

Classification accuracy across 5/10/15/20/30/45/60/90/120 s windows
(source: `detection_latency.csv`). Detection is essentially flat from 15 s
onward (98.2-98.5%); 5 s is 97.76% and 10 s is 97.67%, so lowering the
detection window to 5 s costs less than 1 percentage point. A monitoring
system can therefore operate with ≤15 s alert latency without meaningful
accuracy loss. Figure H1: accuracy vs window size (already generated as
`report/figures/accuracy_vs_window.png`).

---

## Figures & Tables Summary

### Main text figures (all present in `report/figures/`)
| #  | Type      | File                                       | Content                                             | Section |
|----|-----------|--------------------------------------------|-----------------------------------------------------|---------|
| 1  | Heatmap   | `cross_gpu_heatmap.{png,pdf}`              | Cross-GPU transfer matrix (9 GPUs + pooled row)      | 5.1     |
| 2  | Plot      | `dilution_curve.{png,pdf}`                 | Dilution curve (baseline / LOO / hardened)           | 5.3     |
| 3  | Bar chart | `feature_importance_top15.{png,pdf}`        | Top 15 XGBoost features by gain                      | 5.1     |
| 4  | Bar chart | `loo_detection_bar.{png,pdf}`               | LOO detection rate per strategy family               | 5.2     |

### Appendix figures
| #  | Type      | File                                       | Content                                             | Appendix |
|----|-----------|--------------------------------------------|-----------------------------------------------------|----------|
| B1 | 4-panel   | `telemetry_traces_4panel.{png,pdf}`        | Telemetry traces: training / inference / evasion / mining | B     |
| D1 | Scatter   | `wall_power_scatter.{png,pdf}`             | Wall power (Tapo P115) vs NVML `power_draw_w`        | D        |
| E1 | Heatmap   | `confusion_matrix_rf30s.{png,pdf}`         | 3×3 confusion matrix (RF 30 s pooled)                | E        |
| F1 | Scatter   | `roofline_corner_cases.{png,pdf}`          | Roofline plot: training band + 286 corner cases      | F        |
| G1 | Bar chart | `classifier_comparison.{png,pdf}`          | Classifier comparison (RF/XGB/SVM/LR)                | G        |
| H1 | Line plot | `accuracy_vs_window.{png,pdf}`             | Accuracy vs window size (5-120 s)                    | H        |

### Tables
| # | Content                                                 | Section |
|---|---------------------------------------------------------|---------|
| 1 | Hardware + workload summary (9 GPU classes, 6 categories) | 4.1     |
| 2 | Arms race summary (rounds × detection rates)             | 5.2     |
| 3a| Controlled dilution sweep                                | 5.3     |
| 3b| Composite idle-padding sweep (baseline/hardened/evasion-resistant) | 5.3 |

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
- Heim & Koessler (2024) — Training Compute Thresholds: Features and Functions in AI Regulation [2405.10799]
- Seferis & Fist (2025) — Detecting compute-structuring evasion across cloud providers. OpenReview: https://openreview.net/forum?id=qseqw1sWzz
- U.S. Executive Order 14110 (2023) — Safe, Secure, and Trustworthy Development and Use of Artificial Intelligence. The White House.

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
- You & Owen (2025) — Scaling Intelligence: The Exponential Growth in AI's Energy Needs. Joint EPRI / Epoch AI report.
- Cottier, Rahman, Fattorini, Maslej & Owen (2024) — The Rising Costs of Training Frontier AI Models [2405.21015]

**Performance modeling:**
- Williams, Waterman & Patterson (2009) — Roofline: Visual Performance Model (CACM)
- Kaplan et al. (2020) — Scaling Laws for Neural Language Models [2001.08361]

**Classifiers:**
- Breiman (2001) — Random Forests (Machine Learning)

**Other:**
- Anonymous (2026) — Differential Architecture. ISCA, under review.

---

## Remaining work before TAIGR submission (April 24)

- [ ] Compress this full-length draft into the 2-page TAIGR outline
      (`report/icml_2page_outline.md`) with updated numbers.
- [ ] LaTeX formatting for workshop submission.
- [ ] Proofread.
