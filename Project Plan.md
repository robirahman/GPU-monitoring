# SPAR GPU Workload Classification: 10-Week Project Plan

## Context

This is a SPAR research project on adversarial classification of ML training workloads on GPUs. The goal is to determine whether on-chip telemetry and side-channel measurements can reliably distinguish ML training from other GPU workloads (scientific HPC, crypto mining, rendering), even when an adversary actively disguises their workload. The deliverables are a technical report, a signal comparison table, a HEM (Hardware Enabled Mechanism) telemetry design proposal, and a prototype classifier. Budget: $1,000 for compute.

**Sharpened research question (post-literature-review):** Two recently published papers have pre-empted parts of the original scope. WAVE (Xu et al., ASPLOS '26) already demonstrates GPU PMC-based fingerprinting of LLM *inference* workloads using Nsight Compute. Differential Architecture (Anonymous, ISCA '26) already characterizes the hardware bottlenecks (compute vs. memory bandwidth vs. cache bandwidth) for all major GPU workload classes. Our novel contributions are therefore: (1) detection of ML *training* specifically, which neither paper addresses; (2) adversarial evasion and robustness, which neither paper addresses; (3) classification using always-available Tier 1 NVML metrics rather than Nsight Compute's 1200-5300% overhead Tier 3 PMCs, framed as a deployability question; and (4) the HEM governance proposal, which synthesizes all three bodies of work.

---

## Hardware & Infrastructure

**Primary compute: Cloud GPUs (Vast.ai + RunPod)**
- Full $1,000 budget allocated to cloud compute (~1,300–1,500 A100-hours on Vast.ai at $0.66–0.78/hr)
- Bare-metal PCIe passthrough on most Vast.ai hosts = full DCGM, pynvml, and Nsight Compute access
- Can install DCGM yourself in the root Docker container to get Tier 2 metrics (tensor core utilization, FP16/FP32/FP64 pipe utilization, SM occupancy, DRAM bandwidth)
- Mining software not platform-blocked on Vast.ai (important for profiling mining workloads)
- Use RunPod Secure Cloud as backup ($1.14-1.39/hr A100) -- more reliable, DCGM pre-installed, but blocks mining software

**Optional: Physical Hardware (Grant-Dependent)**
- If the hardware grant is approved, purchase a used NVIDIA RTX 3090 (~$500–600) and a Kill-A-Watt power meter (~$25) for local experiments
- 24GB VRAM, Ampere architecture, full NVML support
- Gives direct access to: power draw, temperature, GPU/memory utilization, clock speeds, PCIe throughput, fan speed via `nvidia-smi` and `pynvml`
- Advanced per-kernel profiling via Nsight Compute (`ncu`) and CUPTI
- Note: GPM metrics (per-SM pipe utilization, tensor core %) are Hopper-only; use Nsight Compute for kernel-level instruction mix on Ampere
- Kill-A-Watt enables wall-power side-channel measurements (Tier 5)
- Local GPU enables cross-GPU generalization experiments (cloud A100 vs. local 3090) and physical side-channel analysis
- **This hardware is not required for core experiments in Weeks 1–5; it supplements cloud data starting Week 6 if available**

**Why Vast.ai over other providers:**

| Provider | Bare-metal? | DCGM | Nsight | A100 $/hr | Mining OK? |
|----------|------------|------|--------|-----------|------------|
| **Vast.ai** | Yes | Yes | Host-dependent | **$0.66-0.78** | Yes (varies by host) |
| RunPod | Yes | Yes (Secure Cloud) | Yes | $1.14-1.39 | Blocked |
| Lambda | Yes | Yes | Yes | TBD | Not documented |
| CoreWeave | Yes | Yes | Yes | TBD | Not documented |
| GCP | Mostly virtualized | Yes (v2 w/ Ops Agent) | Limited | ~$2.74 | Blocked |
| AWS | Virtualized | Limited | Limited | $2.74 | Blocked |
| Azure | Some bare-metal | Yes (auto-installed) | Limited | Premium | Blocked |

Key insight: **bare-metal cloud access is essential** for this project. Hyperscaler VMs (AWS, most GCP) restrict DCGM advanced metrics and Nsight profiling behind the hypervisor. Vast.ai and RunPod both offer bare-metal PCIe passthrough, which provides the kernel-level access needed for Tiers 1–3. Physical side-channel data (wall power, acoustics) requires local hardware, which is deferred to Weeks 6+ if the grant is approved.

**Budget breakdown:**
| Item | Cost |
|------|------|
| Vast.ai cloud compute | ~$990 |
| RunPod test instance (1-2 hrs) | ~$10 |
| **Total (current budget)** | **~$1,000** |
| *Physical hardware (grant-dependent)* | *~$575 (RTX 3090 ~$550 + Kill-A-Watt ~$25)* |

---

## Software Tools Needed

- **Monitoring:** `nvidia-smi`, `pynvml` (Python NVML bindings), NVIDIA DCGM, Nsight Compute (`ncu`), Nsight Systems (`nsys`)
- **ML frameworks:** PyTorch, Hugging Face Transformers
- **Workloads:** GROMACS (molecular dynamics), Blender (rendering), T-Rex miner or custom CUDA kernels (mining proxy), cuFFT, llama.cpp (inference)
- **Analysis:** Python, pandas, scikit-learn, XGBoost, matplotlib/seaborn
- **Infra:** Git, Docker, Jupyter, LaTeX/Overleaf

---

## Representative Workloads to Profile

| Category | Workloads |
|----------|-----------|
| ML Training | ResNet-50 on CIFAR-10 (FP32 + mixed-precision), GPT-2 124M fine-tuning, BERT fine-tuning, simple MLP |
| ML Inference | llama.cpp 7B generation, batch image classification |
| Scientific HPC | GROMACS MD simulation, cuFFT benchmark, N-body simulation |
| Crypto Mining | T-Rex miner (benchmark mode) or custom Ethash-like CUDA kernel |
| Rendering | Blender Cycles (BMW scene, Classroom scene) |
| Other | Idle GPU, FFmpeg NVENC video encoding |

---

## Metrics to Collect

**Tier 1 - Basic NVML (available on all NVIDIA GPUs including cloud A100, 1 Hz sampling):**
GPU utilization %, memory utilization %, memory used (MB), power draw (W), temperature (C), SM clock (MHz), memory clock (MHz), PCIe TX/RX throughput (MB/s), encoder/decoder utilization %, fan speed %

**Tier 2 - DCGM (available on cloud A100):**
Tensor core active %, FP16/FP32/FP64 pipe utilization, SM occupancy %, DRAM read/write bandwidth, NVLink bandwidth

**Tier 3 - Per-kernel profiling (Nsight Compute):**
Instruction mix (INT, FP16, FP32, FP64, tensor ops), kernel duration, achieved occupancy, memory throughput per kernel

**Tier 4 - Temporal/behavioral patterns (derived):**
Periodicity, memory allocation stability, power draw variance, communication patterns

**Tier 5 - Physical side channels (grant-dependent, deferred to Weeks 6+):**
Wall power draw (Kill-A-Watt), acoustic emissions (stretch goal). Requires local GPU and Kill-A-Watt meter; not available on cloud instances. If the hardware grant is not approved, Tier 5 analysis becomes a documented future-work recommendation.

---

## Week-by-Week Plan

### Week 1: Kickoff, Literature Review, Cloud Platform Setup

**Tasks:**
- Read core papers:
  - Shavit (2023), "What does it take to catch a Chinchilla?" (arXiv:2303.11341)
  - Kulp et al. (2024), "Hardware-Enabled Governance Mechanisms" (RAND)
  - "GPU Under Pressure: Estimating Application's Stress via Telemetry" (arXiv:2511.05067)
  - "Detecting Covert Cryptomining using HPC" (arXiv:1909.00268)
  - **Xu et al., "WAVE: Leveraging Architecture Observation for Privacy-Preserving Model Oversight" (ASPLOS '26)** -- establishes GPU PMC fingerprinting of LLM inference; read alongside the open-source repo at https://github.com/sept-usc/Wave
  - **Anonymous, "Differential Architecture: Limiting Performance of Targeted Applications" (ISCA '26 submission)** -- establishes hardware bottleneck characterization for all major GPU workload classes
- Create accounts on Vast.ai and RunPod; spin up a test instance on each for 1 hour to verify `nvidia-smi dmon`, DCGM access, and bare-metal profiling capabilities
- Identify and bookmark Vast.ai hosts that support Nsight Compute (bare-metal, PCIe passthrough) for Tier 3 collection in Week 3
- Set up Git repository with directory structure: `literature/`, `data/`, `scripts/`, `notebooks/`, `workloads/`, `classifier/`, `report/`

**Checkpoint:** Mentees submit 1-page summary of WAVE and Differential Architecture, explicitly noting what each paper establishes and what remains open. Cloud access verified on both Vast.ai and RunPod. Bare-metal-capable hosts identified.

### Week 2: Environment Setup and Telemetry Pipeline

**Tasks:**
- Set up cloud GPU environment: prepare a Docker image or setup script with NVIDIA drivers (550+), CUDA 12.x, pynvml, Nsight tools, and DCGM. Test on a Vast.ai A100 instance.
- Clone the WAVE repository (https://github.com/sept-usc/Wave); run their example collection script on a cloud instance to understand the 9-metric Nsight Compute PMC collection approach. Adapt their Tier 3 collection code rather than writing it from scratch.
- Build data collection harness (`scripts/collect_telemetry.py`): polls pynvml at 1 Hz, logs to CSV/Parquet with metadata (workload label, GPU model, timestamps). This is the Tier 1 harness -- separate from WAVE's Tier 3 harness.
- Build workload launcher (`scripts/run_workload.py`): starts telemetry, launches workload, stops telemetry, saves labeled data
- Test pipeline end-to-end on a cloud A100: idle GPU + simple PyTorch training
- Complete literature review and annotated bibliography
- Finalize cloud platform choice based on Week 1 testing (Vast.ai primary, RunPod backup)

**Checkpoint:** Both Tier 1 (pynvml) and Tier 3 (adapted WAVE) pipelines produce clean, labeled data files on cloud instances. Literature review draft complete.

### Week 3: Data Collection Round 1

**Tasks:**
- Run all representative workloads on cloud A100 via Vast.ai (10-15 min each, 3 runs minimum per workload)
- Collect Tier 1 metrics via pynvml harness for every run
- Collect Tier 2 metrics via DCGM for every run (tensor core utilization, FP16/FP32/FP64 pipe utilization, SM occupancy, DRAM bandwidth)
- Run Nsight Compute profiles (Tier 3) for 1 run of each workload type (`ncu --set full`) using the adapted WAVE collection script on bare-metal cloud hosts that support it
- Organize data: `{workload}_{gpu}_{run}_{date}.parquet`
- **Focus for training workloads:** collect enough runs to characterize training-specific temporal signals -- epoch periodicity, optimizer state memory growth over time, forward/backward pass asymmetry. These are not covered by WAVE or Differential Architecture.

**Checkpoint:** 15-20+ workload runs complete. Tier 1 and Tier 2 data for all workloads. Nsight Compute profiles collected on bare-metal cloud hosts.

### Week 4: Data Collection Round 2 and Exploratory Analysis

**Tasks:**
- Complete any remaining cloud runs; add edge cases (ML inference, mixed workloads, short training, DataLoader-bottlenecked training)
- Exploratory data analysis (all based on cloud-collected data):
  - Time-series plots of each metric per workload type
  - Summary statistics (mean, std, autocorrelation) per workload
  - PCA/t-SNE visualization of workload feature vectors
- Build draft signal comparison table (Deliverable 4): workload types vs. metrics, with observed values
- Analyze Nsight Compute kernel profiles: instruction mix, GEMM prevalence, kernel repetition patterns
- **Note on bottleneck characterization:** The Differential Architecture paper has already established which hardware bottlenecks dominate each workload class (MatMul/training=compute-bound, FFT/scientific=cache+memory-bound, vector-mult/inference=memory-bandwidth-bound, rendering=compute+RT-bound). Do not spend time re-deriving this from scratch. Instead, verify that your measurements are consistent with their findings, and focus analytical effort on: (a) the training/inference distinction within the ML class, and (b) temporal and behavioral patterns (epoch periodicity, memory growth curves) that static bottleneck analysis does not capture.
- Write 1-page summary: "Which metrics distinguish ML training from inference, and from non-ML workloads, at Tier 1 resolution?"

**Checkpoint:** 30+ runs in dataset (all cloud A100). EDA notebook complete. Draft signal comparison table started. Training/inference distinction characterized.

### Week 5: Feature Engineering and Baseline Classifier

**Tasks:**
- Feature engineering: 60-second sliding windows, compute per metric: mean, std, min, max, CV, skewness, autocorrelation at lags 1/5/10/30s, dominant FFT frequency (~120 features per window)
- **Training-specific features to add** (not in WAVE or Differential Architecture): epoch-level periodicity score (autocorrelation peak at epoch-duration lag), optimizer state memory growth rate (MB/min over the run), compute/memory-bandwidth ratio (to operationalize Differential Architecture's bottleneck model as a feature), ratio of power draw during forward vs. backward pass windows
- Split by workload run (not window) to prevent leakage: 70/15/15 train/val/test
- Train classifiers: Random Forest, XGBoost, SVM, Logistic Regression
- Report accuracy, per-class precision/recall/F1, confusion matrices for: binary (ML training vs. rest), three-way (ML training vs. ML inference vs. non-ML), and full multi-class
- Extract feature importance rankings from tree models
- Error analysis: which workload pairs are most confused? Pay particular attention to training vs. inference confusion.

**Checkpoint:** Target >85% binary accuracy on clean data. Training/inference classifier accuracy reported. Feature importance ranking produced.

### Week 6: Classifier Refinement and WAVE Comparison

**Tasks:**
- Add cross-metric features (power/utilization ratio, memory allocation rate, PCIe burstiness)
- Test window sizes: 30s, 60s, 120s, 300s -- report accuracy vs. window size
- Train time-series models (1D CNN, LSTM) on raw metric sequences; compare with feature-engineered approach
- Test with Tier 2 metrics from cloud A100: how much does tensor core utilization improve accuracy?
- Cross-GPU generalization: *if the hardware grant is approved and a local RTX 3090 is available*, train on cloud A100 data and test on local 3090 data (and vice versa). Otherwise, test generalization across different cloud A100 hosts/configurations.
- **New: WAVE vs. Tier 1 comparison experiment.** Train a parallel classifier using WAVE's 9 Nsight Compute PMC metrics (Tier 3) as features. Compare against the Tier 1 NVML classifier on accuracy and F1. Report the accuracy/overhead tradeoff: "Tier 3 achieves X% accuracy at 1200-5300% runtime overhead; Tier 1 achieves Y% accuracy at ~0% overhead." This is a key novel result framing our deployability contribution.

**Checkpoint:** Target >90% binary, >80% multi-class on clean data. Window size sensitivity and cross-GPU results documented. Tier 1 vs. Tier 3 accuracy/overhead tradeoff quantified. *(If grant approved: cross-GPU generalization between A100 and RTX 3090.)*

### Week 7: Adversarial Workload Design

**Tasks:**
- Design and implement 5 disguise strategies:
  - **A: Utilization modulation** -- insert sleeps/dummy work between training batches to mimic rendering patterns
  - **B: Low-utilization training** -- tiny batch sizes + throttling to resemble inference
  - **C: Instruction mix obfuscation** -- disable mixed-precision + run dummy FP64 kernels alongside training
  - **D: Temporal disruption** -- randomize batch sizes, vary compute intensity to break epoch periodicity
  - **E: PMC signature spoofing** -- dummy CUDA kernels crafted to produce WAVE-like periodic PMC patterns corresponding to a non-training workload (e.g., FFT-like cache/memory traffic ratios). This directly targets the Nsight-based Tier 3 detector; test whether it also fools the Tier 1 NVML classifier.
- Run each adversarial workload 3+ times on cloud A100, collect telemetry with same pipeline (both Tier 1 and Tier 3)
- *If the hardware grant is approved and a local RTX 3090 is available*, also run adversarial workloads on the local GPU for additional cross-hardware data
- Label as `ml_training_adversarial_A/B/C/D/E`

**Checkpoint:** All 5 adversarial strategies implemented. Adversarial telemetry data collected at both Tier 1 and Tier 3 on cloud instances. *(If grant approved: additional local GPU adversarial data collected.)*

### Week 8: Adversarial Robustness Testing

**Tasks:**
- Evaluate Week 6 classifiers (both Tier 1 and Tier 3) on all 5 adversarial strategies: detection rate per strategy per tier
- Analyze successful evasions: which features shifted? By how much?
- Cost-of-evasion analysis: measure training throughput (samples/sec) and total energy with vs. without disguise. Report as a table: strategy, detection rate (Tier 1), detection rate (Tier 3), throughput penalty %, energy overhead %.
- Retrain classifier with adversarial examples in training set (adversarial training)
- Add evasion-resistant features: cumulative FLOP counting, tensor core presence detection, memory allocation stability
- **Two rounds of adversary-defender iteration:** after initial robustification, have one team member design a refined attack against the hardened classifier, then defend again. Document the evolution. This is the primary novel contribution of the project.

**Checkpoint:** Adversarial evaluation table complete (5 strategies x 2 tiers). Cost-of-evasion quantified. Two rounds of adversary-defender iteration documented.

### Week 9: HEM Design Proposal and Code Cleanup

**Tasks:**
- Write HEM telemetry design proposal (Deliverable 5). The proposal now synthesizes three bodies of prior work:
  - **From Differential Architecture:** the bottleneck model establishes *which* hardware resources distinguish workload classes at the architectural level. Use their triple-point analysis to justify which counters are most discriminative (compute FLOPS counters for training/MatMul-heavy workloads, memory bandwidth counters for inference/decode, cache bandwidth for scientific workloads).
  - **From WAVE:** the 9 PMC metrics and the overhead measurements establish the tradeoff between monitoring precision and runtime cost. Use their overhead table (1200-5300% for Nsight; near-zero for DCGM-style periodic reporting) to argue for a tiered monitoring architecture.
  - **From our experiments:** the adversarial testing results establish which metrics are evasion-resistant and which are not. Ground every recommendation in a specific experimental result.
  - **Tiered monitoring proposal:** always-on Tier 1 NVML as a low-cost continuous filter; triggered Tier 2/3 sampling (DCGM-style, not Nsight-style) for flagged sessions. Argue that Nsight-style re-execution profiling is impractical for governance and that hardware-native aggregate counters are the right implementation target.
  - **Must-have metrics:** instruction type counters (tensor/FP16/FP32/FP64), cumulative FLOP counters, memory allocation patterns
  - **Should-have:** power draw time series, SM utilization, PCIe/NVLink volumes
  - For each: cite experimental evidence, evasion difficulty, recommended sampling rate, hardware vs. firmware implementation
  - Address privacy, tamper resistance, false positive considerations
  - *If the hardware grant is approved and physical side-channel data (Tier 5) has been collected*, the HEM proposal's physical side-channel recommendations become empirically grounded. Otherwise, physical side-channel analysis is included as a documented future-work recommendation citing the literature.
- Clean up classifier code into a package with README, requirements, and a live-prediction demo script
- Begin report writing: outline, methodology section, results tables and figures

**Checkpoint:** HEM proposal drafted and grounded in all three papers. Classifier code documented and reproducible. Report outline and methodology section done. *(If grant approved: HEM physical side-channel recommendations grounded in local GPU data.)*

### Week 10: Report and Final Deliverables

**Tasks:**
- Write and polish the full technical report (15-25 pages):
  1. Introduction and research questions -- frame contributions explicitly relative to WAVE and Differential Architecture
  2. Literature review (Deliverable 1) -- include WAVE, Differential Architecture, and prior papers; discuss what each establishes and what gap we fill
  3. Workload characterization (Deliverable 2) -- cite Differential Architecture for bottleneck analysis; contribute training/inference distinction and temporal signatures
  4. Experimental setup
  5. Classification results (Deliverable 3) -- report Tier 1 vs. Tier 3 accuracy/overhead tradeoff as a central result
  6. Signal comparison table (Deliverable 4)
  7. Adversarial testing results -- primary novel contribution; 5 strategies x 2 tiers, two rounds of iteration
  8. HEM design proposal (Deliverable 5) -- synthesizes all three bodies of prior work
  9. Discussion (limitations, future work, policy implications)
  10. Conclusion
- Finalize signal comparison chart as publication-quality table/heatmap
- Final classifier code with pre-trained weights and demo mode (Deliverable 6)
- Prepare 15-20 minute presentation
- Mentor review mid-week, mentees incorporate feedback

**Final deliverables checklist:**
- [ ] Technical report (PDF)
- [ ] Literature review / annotated bibliography
- [ ] Signal comparison table/chart
- [ ] HEM telemetry design proposal
- [ ] Classifier code repository with README and demo
- [ ] Collected dataset with documentation
- [ ] Presentation slides
- [ ] TODO (repo cleanup): Remove PDFs from `literature/` and replace with a `literature/bibliography.md` containing full citations and links. PDFs may be under copyright and should not remain in a public repo.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| Hardware grant not approved | All core experiments run on cloud; physical side-channel analysis (Tier 5) becomes a documented future-work recommendation rather than an empirical result. Cross-GPU generalization tested across cloud host configurations instead of A100 vs. 3090. |
| Cloud budget exhaustion | Monitor spending weekly; prioritize key workloads; use spot/interruptible instances on Vast.ai; reduce run count for low-priority edge cases |
| Cloud platform blocks mining software | Use benchmark mode or write custom CUDA kernels mimicking mining patterns |
| Nsight Compute unavailable on cloud host | Pre-identify bare-metal hosts during Week 1; fall back to DCGM Tier 2 metrics if Nsight access is limited |
| Adversarial strategies too effective | This is a valid research finding -- document fragility and recommend hardened metrics |
| Insufficient data for robust classifier | Augment with varied hyperparameters; use smaller windows for more samples |

---

## Verification

- Telemetry pipeline: run a known workload, verify CSV output matches `nvidia-smi dmon` readings
- Classifier: k-fold cross-validation with per-run splits; report confidence intervals
- Adversarial testing: blind evaluation (classifier never sees adversarial strategy labels during training for the initial test)
- HEM proposal: ground each recommendation in specific experimental results with figure/table references
- End-to-end demo: run classifier live on GPU while running an unlabeled workload, verify correct prediction
