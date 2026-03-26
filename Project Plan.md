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

**Local Hardware (Grant Approved — Purchased March 2026)**
- **GPU:** ASUS PRIME NVIDIA GeForce RTX 5070 Ti OC Edition, 16GB GDDR7 — $1,099 after tax
  - Blackwell architecture (GB203), 5th-gen Tensor Cores (FP8/FP4/FP6/BF16/FP16/TF32)
  - 8,960 CUDA cores, 280 Tensor Cores, 300W TDP
  - NVENC Gen 9 (H.264, HEVC, AV1) — enables FFmpeg encoding workload impossible on A100/H100
  - 12V-2x6 (12VHPWR) power connector; adapter to dual 8-pin included in box
  - PCIe 5.0 x16 (runs at Gen 3 speeds in the Z640 — negligible impact on ML training)
  - **Cross-architecture value:** Blackwell is a third GPU architecture alongside cloud A100 (Ampere) and H100 (Hopper), maximizing cross-GPU generalization experiments
- **Workstation:** HP Z640 Tower, refurbished — $533
  - Intel Xeon E5-2640 v3 (Haswell-EP), 8 cores / 16 threads, 2.60 GHz base / 3.40 GHz boost
  - 32GB DDR4 ECC RAM
  - 512GB SSD + 2TB HDD
  - 925W PSU (sufficient for 300W GPU + ~85W CPU + peripherals)
  - PCIe 3.0 x16 slots
  - Note: Haswell-EP CPU may be a data-loading bottleneck for GPU-bound workloads; mitigate by pre-loading datasets to GPU memory and using small datasets (CIFAR-10, SST-2, WikiText-2)
- **Power measurement:**
  - Kill-A-Watt P4400 (~$30) — manual ground-truth spot checks
  - Shelly Plus Plug US (~$25) — automated wall-power logging via local REST API at ~1 Hz
  - `nvidia-smi` software logging at 10 Hz (GPU-reported board power, free)
- **Total hardware grant spend:** ~$1,687 ($1,099 GPU + $533 workstation + $55 power meters)
- Full NVML, DCGM, and Nsight Compute access (no cloud virtualization restrictions)
- Enables: Tier 5 physical side-channel measurements, FFmpeg NVENC workload, unrestricted Tier 3 profiling, cross-architecture generalization (Ampere/Hopper/Blackwell)
- **Hardware arrives and is set up during Week 6; local experiments begin Week 7**

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

Key insight: **bare-metal cloud access is essential** for this project. Hyperscaler VMs (AWS, most GCP) restrict DCGM advanced metrics and Nsight profiling behind the hypervisor. Vast.ai and RunPod both offer bare-metal PCIe passthrough, which provides the kernel-level access needed for Tiers 1–3. Physical side-channel data (wall power, acoustics) requires local hardware (available from Week 7 onward).

**Budget breakdown:**
| Item | Cost | Source |
|------|------|--------|
| Vast.ai cloud compute | ~$990 | Cloud compute budget |
| RunPod test instance (1-2 hrs) | ~$10 | Cloud compute budget |
| **Cloud subtotal** | **~$1,000** | |
| ASUS PRIME RTX 5070 Ti OC 16GB | $1,099 | Hardware grant |
| HP Z640 Tower (refurb, 32GB, 925W) | $533 | Hardware grant |
| Kill-A-Watt P4400 + Shelly Plus Plug US | ~$55 | Hardware grant |
| **Hardware subtotal** | **~$1,687** | |
| **Total project spend** | **~$2,687** | |

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

**Tier 5 - Physical side channels (local RTX 5070 Ti, Weeks 7+):**
Wall power draw via Shelly Plus Plug US (automated, ~1 Hz via REST API) and Kill-A-Watt P4400 (manual ground truth). GPU-reported board power via `nvidia-smi -lms 100` (10 Hz). Acoustic emissions via USB microphone (stretch goal). Not available on cloud instances; collected on local HP Z640 + RTX 5070 Ti.

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

### Week 6: Hardware Setup + Shipping (1 week)

**Tasks:**
- Order Kill-A-Watt P4400 and Shelly Plus Plug US if not already ordered
- Receive and assemble HP Z640 + RTX 5070 Ti:
  - Install RTX 5070 Ti in PCIe x16 slot (replace Quadro NVS 295)
  - Connect 12VHPWR power via included dual 8-pin adapter to Z640's 925W PSU
  - Install Ubuntu 22.04 or 24.04 (dual-boot or replace Windows 10)
  - Install NVIDIA drivers (560+), CUDA 12.x, cuDNN, pynvml, Nsight Compute, DCGM
  - Install PyTorch 2.x + Hugging Face Transformers + scikit-learn + XGBoost
  - Clone project repository and verify all scripts run
- **Validation runs:** Execute 1 run of each workload category on RTX 5070 Ti to establish local baselines:
  - idle, pytorch_resnet_cifar10, bert_sst2, gpt2_wikitext2, cufft_benchmark, nbody_sim, mining_ethash_proxy, rendering_proxy, resnet50_inference
  - **FFmpeg NVENC** -- first-ever collection of this workload (impossible on A100/H100)
  - **GROMACS** with proper GPU build (NGC container) -- blocked on Vast.ai
- Run `nvml_smoke_test.py` to verify telemetry pipeline
- Test DCGM profiling fields (Tier 2) -- unrestricted on local hardware (blocked on cloud by virtualization)
- Test Nsight Compute (Tier 3) -- unrestricted on local hardware (blocked on cloud by `RmProfilingAdminOnly`)
- Set up Shelly Plus Plug US for automated wall-power logging; write Python polling script
- **WAVE vs. Tier 1 comparison experiment.** Train a parallel classifier using WAVE's 9 Nsight Compute PMC metrics (Tier 3) as features, now possible with unrestricted local Nsight access. Compare against the Tier 1 NVML classifier on accuracy and F1. Report the accuracy/overhead tradeoff: "Tier 3 achieves X% accuracy at 1200-5300% runtime overhead; Tier 1 achieves Y% accuracy at ~0% overhead."

**Checkpoint:** RTX 5070 Ti operational. All workload categories collected locally including FFmpeg NVENC and GROMACS. Tier 2 and Tier 3 access verified. Wall-power logging functional. Tier 1 vs. Tier 3 accuracy/overhead tradeoff quantified.

### Week 7: Cross-GPU Generalization + Classifier Refinement

This is the **highest-value experiment** enabled by the local hardware. The project now has telemetry from three GPU architectures: A100 (Ampere, 3rd-gen Tensor Cores), H100 (Hopper, 4th-gen), and RTX 5070 Ti (Blackwell, 5th-gen).

**Tasks:**
- **Cross-architecture generalization experiments:**
  1. Train on cloud A100/H100 data (Weeks 3-5), test on local RTX 5070 Ti data: does the 95.65% binary accuracy hold on Blackwell?
  2. Train on RTX 5070 Ti data alone, test on cloud data: do local signatures transfer to datacenter GPUs?
  3. Pool all three GPU types and retrain: does the combined classifier improve or degrade?
  4. Feature stability analysis: which of the 122 features are stable across architectures? Power CV and power ACF1 were the best discriminators on A100/H100 -- do they hold on Blackwell?
- **Fill data gaps to strengthen three-way classifier** (currently 69-76% accuracy vs 80% target):
  - FFmpeg NVENC telemetry (new workload class, only possible on local hardware)
  - GROMACS with GPU-enabled build (blocked on Vast.ai)
  - More diverse inference workloads on RTX 5070 Ti (Blackwell tensor core behavior may differ)
- Add cross-metric features (power/utilization ratio, memory allocation rate, PCIe burstiness)
- Test window sizes: 5s, 15s, 30s, 60s, 120s -- report accuracy vs. window size across GPUs
- Train time-series models (1D CNN, LSTM) on raw metric sequences; compare with feature-engineered approach

**Checkpoint:** Cross-GPU generalization results documented (3 architectures). Three-way classifier accuracy improved. Feature stability across architectures analyzed.

### Week 8: Adversarial Workload Design + Tier 5 Physical Side Channels

**Tasks — Adversarial workloads (5 strategies):**
- Design and implement 5 disguise strategies:
  - **A: Utilization modulation** -- insert sleeps/dummy work between training batches to mimic rendering patterns
  - **B: Low-utilization training** -- tiny batch sizes + throttling to resemble inference
  - **C: Instruction mix obfuscation** -- disable mixed-precision + run dummy FP64 kernels alongside training
  - **D: Temporal disruption** -- randomize batch sizes, vary compute intensity to break epoch periodicity
  - **E: PMC signature spoofing** -- dummy CUDA kernels crafted to produce WAVE-like periodic PMC patterns corresponding to a non-training workload. Requires Nsight Compute access (available on local RTX 5070 Ti, blocked on most cloud hosts).
- Run each adversarial workload 3+ times on **both** local RTX 5070 Ti and cloud A100
- Collect Tier 1 (NVML), Tier 3 (Nsight Compute on local), and Tier 5 (wall power via Shelly Plug) for every run
- Label as `ml_training_adversarial_A/B/C/D/E`

**Tasks — Tier 5 physical side-channel experiments:**
- Collect synchronized wall-power time series (via Shelly Plus Plug US REST API at ~1 Hz) during all workload types on RTX 5070 Ti
- Compare wall power vs. `nvidia-smi` reported board power -- measure the discrepancy (PSU efficiency, CPU/motherboard overhead)
- Test whether adversarial strategies A-E that fool NVML Tier 1 classifiers also fool wall-power classifiers
- Key question: **Can an adversary spoof software telemetry while the physical side channel still reveals true training activity?**
- Acoustic emissions measurement (stretch goal): record fan noise patterns with USB microphone during different workloads

**Checkpoint:** All 5 adversarial strategies implemented and collected on both local and cloud. Tier 5 wall-power data collected for all workload types. Software vs. physical side-channel comparison documented.

### Week 9: Adversarial Robustness Testing + HEM Design

**Tasks — Adversarial robustness:**
- Evaluate classifiers (Tier 1, Tier 3, and Tier 5) on all 5 adversarial strategies: detection rate per strategy per tier
- Analyze successful evasions: which features shifted? By how much?
- Cost-of-evasion analysis: measure training throughput (samples/sec) and total energy with vs. without disguise. Report as a table: strategy, detection rate (Tier 1), detection rate (Tier 3), detection rate (Tier 5 wall-power), throughput penalty %, energy overhead %.
- Retrain classifier with adversarial examples in training set (adversarial training)
- Add evasion-resistant features: cumulative FLOP counting, tensor core presence detection, memory allocation stability
- **Two rounds of adversary-defender iteration:** after initial robustification, design a refined attack against the hardened classifier, then defend again. Document the evolution.

**Tasks — HEM design proposal (Deliverable 5):**
- Synthesize three bodies of prior work + our new contributions:
  - **From Differential Architecture:** which hardware resources distinguish workload classes at the architectural level
  - **From WAVE:** the accuracy/overhead tradeoff between Nsight Compute and DCGM-style monitoring
  - **From our experiments:** which metrics are evasion-resistant, which are not; cross-GPU generalization results; physical side-channel validation
- **Tiered monitoring proposal:** always-on Tier 1 NVML as low-cost continuous filter; triggered Tier 2/3 sampling for flagged sessions; physical side-channel (wall power) as tamper-resistant ground truth
- **Must-have metrics:** instruction type counters, cumulative FLOP counters, memory allocation patterns
- **Should-have:** power draw time series, SM utilization, PCIe/NVLink volumes
- **Physical side-channel recommendation:** grounded in empirical Tier 5 data from local RTX 5070 Ti showing wall-power signatures that resist software spoofing
- For each metric: cite experimental evidence, evasion difficulty, recommended sampling rate, hardware vs. firmware implementation
- Address privacy, tamper resistance, false positive considerations

**Checkpoint:** Adversarial evaluation table complete (5 strategies x 3 tiers including wall power). Cost-of-evasion quantified. Two rounds of adversary-defender iteration documented. HEM proposal drafted.

### Week 10: Code Cleanup and Report Preparation

**Tasks:**
- Clean up classifier code into a package with README, requirements, and a live-prediction demo script
- Begin report writing: outline, methodology section, results tables and figures
- Finalize signal comparison chart as publication-quality table/heatmap

**Checkpoint:** Classifier code documented and reproducible. Report outline and methodology section done.

### Week 11: Report and Final Deliverables

**Tasks:**
- Write and polish the full technical report (15-25 pages):
  1. Introduction and research questions -- frame contributions explicitly relative to WAVE and Differential Architecture
  2. Literature review (Deliverable 1) -- WAVE, Differential Architecture, prior work; what each establishes and what gap we fill
  3. Workload characterization (Deliverable 2) -- cite Differential Architecture for bottleneck analysis; contribute training/inference distinction and temporal signatures
  4. Experimental setup -- three GPU architectures (A100, H100, RTX 5070 Ti), cloud + local, Tiers 1-5
  5. Classification results (Deliverable 3) -- Tier 1 vs. Tier 3 accuracy/overhead tradeoff; cross-GPU generalization across 3 architectures
  6. Signal comparison table (Deliverable 4)
  7. Adversarial testing results -- 5 strategies x 3 tiers (Tier 1 NVML, Tier 3 Nsight, Tier 5 wall power), two rounds of iteration
  8. HEM design proposal (Deliverable 5) -- tiered monitoring with physical side-channel validation
  9. Discussion (limitations, future work, policy implications)
  10. Conclusion
- Revise ISCA paper draft (`week5/reports/isca_paper.md`) with cross-GPU and adversarial results
- Final classifier code with pre-trained weights and demo mode (Deliverable 6)
- Prepare 15-20 minute presentation
- Mentor review mid-week, incorporate feedback

**Final deliverables checklist:**
- [ ] Technical report (PDF)
- [ ] Literature review / annotated bibliography
- [ ] Signal comparison table/chart
- [ ] HEM telemetry design proposal (with physical side-channel empirical grounding)
- [ ] Classifier code repository with README and demo
- [ ] Collected dataset with documentation (3 GPU architectures: A100, H100, RTX 5070 Ti)
- [ ] Presentation slides
- [ ] Revised ISCA paper draft
- [ ] TODO (repo cleanup): Remove PDFs from `literature/` and replace with a `literature/bibliography.md` containing full citations and links.

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| RTX 5070 Ti incompatible with HP Z640 (power connectors, BIOS, PCIe) | Test immediately upon receipt; Z640 925W PSU should handle 300W GPU; worst case, purchase a different workstation with remaining grant funds |
| CPU bottleneck (Haswell-EP E5-2640 v3 is 10+ years old) | Use small datasets (CIFAR-10, SST-2, WikiText-2) that fit in GPU memory; pre-load data; CPU bottleneck affects throughput but not telemetry signatures, which is what we're measuring |
| 16GB VRAM insufficient for some workloads | All current workloads (BERT-base, GPT-2 124M, ResNet) fit in 16GB; reduce batch size if needed; monitor with `nvidia-smi` |
| Cloud budget exhaustion | Monitor spending weekly; prioritize key workloads; use spot instances; most cloud work already complete (Weeks 3-5) |
| Cloud platform blocks mining software | Use benchmark mode or custom CUDA kernels mimicking mining patterns |
| Nsight Compute unavailable on cloud host | Now mitigated: unrestricted Nsight access on local RTX 5070 Ti |
| Adversarial strategies too effective | This is a valid research finding -- document fragility and recommend hardened metrics; test wall-power side channel as evasion-resistant fallback |
| Insufficient data for robust classifier | Three GPU architectures now available; augment with varied hyperparameters; use smaller windows for more samples |

---

## Verification

- Telemetry pipeline: run a known workload, verify CSV output matches `nvidia-smi dmon` readings
- Classifier: k-fold cross-validation with per-run splits; report confidence intervals
- Adversarial testing: blind evaluation (classifier never sees adversarial strategy labels during training for the initial test)
- HEM proposal: ground each recommendation in specific experimental results with figure/table references
- End-to-end demo: run classifier live on GPU while running an unlabeled workload, verify correct prediction
