# SPAR GPU Workload Classification: 10-Week Project Plan

## Context

This is a SPAR research project on adversarial classification of ML training workloads on GPUs. The goal is to determine whether on-chip telemetry and side-channel measurements can reliably distinguish ML training from other GPU workloads (scientific HPC, crypto mining, rendering), even when an adversary actively disguises their workload. The deliverables are a technical report, a signal comparison table, a HEM (Hardware Enabled Mechanism) telemetry design proposal, a prototype classifier, and a **workshop paper submitted to TAIGR @ ICML 2026**. Budget: $1,000 for compute.

**Publication target:** [TAIGR @ ICML 2026](https://taigr-workshop.com/) — Second Workshop on Technical AI Governance Research (Seoul, South Korea, July 10–11, 2026). **Submission deadline: April 24, 2026, 23:59 AOE.** Full papers (8 pages) or short papers (2 pages), ICML 2026 LaTeX template, double-blind review by both ML and policy/governance experts, non-archival. Submission via [OpenReview](https://openreview.net/group?id=ICML.cc/2026/Workshop/TAIGR). Our work fits TAIGR pillars: Verification (verifying claims about AI workloads), Access (external monitoring of GPU activity), and Operationalisation (translating governance goals into technical monitoring).

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
- **Infra:** Git, Docker, Jupyter, LaTeX/Overleaf, ICML 2026 LaTeX template (for TAIGR submission)

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

### Week 6: Hardware Setup + Shipping + Paper Scaffold (~March 26 – April 1)

**⚠️ TAIGR deadline: April 24, 2026 (29 days away). Paper writing begins this week in parallel with hardware setup.**

**Tasks — Hardware setup:**
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

**Tasks — TAIGR paper scaffold:**
- Set up ICML 2026 LaTeX template in `paper/` directory (download from ICML 2026 Author Instructions page)
- **Decision: full paper (8 pages) or short paper (2 pages).** Target full paper; fall back to short paper if experiments run behind schedule. Write modularly so the paper can be contracted.
- Write paper skeleton with section headings:
  1. Introduction (research question, why it matters for AI governance)
  2. Related Work (WAVE, Differential Architecture, Shavit 2023, Kulp et al. 2024)
  3. Methodology (telemetry tiers, workload taxonomy, feature engineering, classifier pipeline)
  4. Experimental Setup (3 GPU architectures, cloud + local, workload registry)
  5. Results (cross-GPU generalization, adversarial robustness, Tier 1 vs Tier 3 tradeoff)
  6. Discussion & Policy Implications (HEM proposal, tiered monitoring, governance recommendations)
  7. Conclusion
- **Draft Related Work section** (can be written now from existing literature review)
- **Draft Methodology section** (feature engineering pipeline, classifier setup — already implemented in Weeks 3-5)
- Frame the governance angle: our work provides technical verification tools for AI governance — always-available Tier 1 monitoring as a low-cost compliance mechanism, adversarial robustness as an evasion-cost analysis for policymakers

**Checkpoint:** RTX 5070 Ti operational. All workload categories collected locally including FFmpeg NVENC and GROMACS. Tier 2 and Tier 3 access verified. Wall-power logging functional. Tier 1 vs. Tier 3 accuracy/overhead tradeoff quantified. **Paper LaTeX template set up; Related Work and Methodology sections drafted.**

### Week 7: Cross-GPU Generalization + Paper Results Draft (~April 1 – April 8)

This is the **highest-value experiment** enabled by the local hardware. The project now has telemetry from three GPU architectures: A100 (Ampere, 3rd-gen Tensor Cores), H100 (Hopper, 4th-gen), and RTX 5070 Ti (Blackwell, 5th-gen).

**⚠️ TAIGR deadline: April 24 (17 days away). Cross-GPU results are the paper's core contribution — prioritize these experiments.**

**Tasks — Experiments:**
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

**Tasks — Paper writing:**
- **Draft Experimental Setup section** with GPU specifications table, workload registry, data collection methodology
- **Draft Results section — Part 1:** cross-GPU generalization tables and figures (accuracy per train/test GPU split, feature stability heatmap)
- Generate publication-quality figures: confusion matrices, feature importance plots, cross-GPU accuracy comparison bars
- **Draft Introduction:** frame the governance gap, state contributions (3 GPU architectures, adversarial robustness, Tier 1 vs Tier 3 tradeoff, policy recommendations)

**Checkpoint:** Cross-GPU generalization results documented (3 architectures). Three-way classifier accuracy improved. Feature stability across architectures analyzed. **Paper Introduction, Experimental Setup, and cross-GPU Results drafted.**

### Week 8: Local Hardware Data Collection + Adversarial Experiments (~April 6 – April 10)

**⚠️ TAIGR deadline: April 24 (14 days away). This is the final data-collection week — every experiment must serve a table or figure in the paper.**

**Status entering Week 8:** All 60+ parquet files are A100-only. No local GPU data exists yet. The midterm report classifier (100% binary, 95.6% 15-class) is trained exclusively on A100 data. The paper's core claim — cross-architecture generalization — has zero supporting data. This week fills that gap on two new architectures simultaneously.

**Local hardware available:**
| Machine | GPU | Arch | VRAM | TDP | Unique capabilities |
|---------|-----|------|------|-----|---------------------|
| HP Z640 desktop | RTX 5070 Ti | Blackwell (SM 10.0) | 16 GB GDDR7 | 300W | Tier 5 wall power (Shelly Plug), NVENC Gen 9, unrestricted Nsight, full DCGM |
| Laptop | RTX 4050 Laptop | Ada Lovelace (SM 8.9) | 6 GB GDDR6 | 60W | 3rd consumer architecture, laptop thermal throttling regime |

---

#### Monday April 6: Desktop RTX 5070 Ti — Full Workload Suite (P0)

This is the highest-priority day. The 5070 Ti runs every workload at the same parameters as the A100 baseline, producing directly comparable cross-architecture data.

**Tasks:**
- Verify desktop setup: drivers, CUDA, pynvml, DCGM, project repo cloned, `collect_telemetry.py` smoke test
- Run 3 reps of each workload (same parameters as A100 Week 3 collection):

| Workload | Label | Batch size | Duration/Epochs | Notes |
|----------|-------|-----------|-----------------|-------|
| Idle | `idle` | — | 120s | Baseline |
| ResNet-18 CIFAR-10 FP32 | `pytorch_resnet_cifar10` | 512 | 10 epochs | |
| ResNet-18 CIFAR-10 AMP | `pytorch_resnet_cifar10_amp` | 512 | 10 epochs | |
| MLP CIFAR-10 | `pytorch_mlp_cifar10` | 512 | 20 epochs | |
| GPT-2 124M WikiText-2 FP32 | `gpt2_wikitext2` | 8 | 400 steps | |
| GPT-2 124M WikiText-2 AMP | `gpt2_wikitext2_amp` | 8 | 400 steps | |
| BERT SST-2 FP32 | `bert_sst2` | 32 | 300 steps | |
| BERT SST-2 AMP | `bert_sst2_amp` | 32 | 300 steps | |
| ResNet-50 inference | `resnet50_inference` | 256 | 600s | |
| cuFFT benchmark | `cufft_benchmark` | — | 600s | |
| N-body simulation | `nbody_sim` | 4096 bodies | 600s | |
| Mining proxy | `mining_ethash_proxy` | 1 GB DAG | 600s | |
| Rendering proxy | `rendering_proxy` | 32768 rays | 600s | |
| FFmpeg NVENC | `ffmpeg_nvenc` | — | 300s | **New: impossible on A100** |

- Collect Tier 1 (pynvml) for all runs
- Collect Tier 2 (DCGM) in parallel where possible
- Set up Shelly Plus Plug US wall-power logging; record Tier 5 power for every run
- Save as `{workload}_NVIDIA_GeForce_RTX_5070_Ti_{hash}_{timestamp}.parquet`

**Target:** ~42 parquet files (14 workloads × 3 reps). This is the single biggest data-collection push of the week.

---

#### Tuesday April 7: Laptop RTX 4050 — Cross-Architecture Holdout Set (P0)

The laptop provides Ada Lovelace data — the 3rd architecture for generalization testing. All batch sizes reduced to fit 6 GB VRAM.

**Tasks:**
- Verify laptop setup: pynvml installed, `collect_telemetry.py` runs, CUDA functional
- Run 3 reps of each workload with VRAM-adjusted parameters:

| Workload | Label | Batch size | Duration/Epochs | VRAM adjustment |
|----------|-------|-----------|-----------------|-----------------|
| Idle | `idle` | — | 120s | None |
| ResNet-18 CIFAR-10 FP32 | `pytorch_resnet_cifar10` | 256 | 10 epochs | Halved batch |
| ResNet-18 CIFAR-10 AMP | `pytorch_resnet_cifar10_amp` | 256 | 10 epochs | Halved batch |
| MLP CIFAR-10 | `pytorch_mlp_cifar10` | 512 | 20 epochs | Fits as-is |
| GPT-2 124M WikiText-2 FP32 | `gpt2_wikitext2` | 2 | 400 steps | Reduced from 8 |
| GPT-2 124M WikiText-2 AMP | `gpt2_wikitext2_amp` | 4 | 400 steps | Reduced from 8 |
| BERT SST-2 FP32 | `bert_sst2` | 8 | 300 steps | Reduced from 32 |
| BERT SST-2 AMP | `bert_sst2_amp` | 16 | 300 steps | Reduced from 32 |
| ResNet-50 inference | `resnet50_inference` | 64 | 600s | Reduced from 256 |
| cuFFT benchmark | `cufft_benchmark` | — | 600s | Fits as-is |
| N-body simulation | `nbody_sim` | 4096 bodies | 600s | Fits as-is |
| Mining proxy | `mining_ethash_proxy` | 512 MB DAG | 600s | Halved DAG |
| Rendering proxy | `rendering_proxy` | 32768 rays | 600s | Fits as-is |
| FFmpeg NVENC | `ffmpeg_nvenc` | — | 300s | Fits as-is |

- Collect Tier 1 only (DCGM may not be available on laptop; don't waste time debugging it)
- Save as `{workload}_NVIDIA_GeForce_RTX_4050_Laptop_{hash}_{timestamp}.parquet`
- **Monitor thermal throttling**: log `nvidia-smi` temperature alongside telemetry; note any clock drops during sustained runs — this is itself an interesting data point (laptop thermal signatures differ from desktop/server)

**Target:** ~42 parquet files. Combined with Monday, the dataset grows from 1 architecture to 3.

---

#### Wednesday April 8: Adversarial Workloads on Desktop (P0)

Run on the 5070 Ti where wall-power ground truth is available via Shelly Plug.

**Tasks — Implement and run 5 adversarial disguise strategies (3 reps each):**

| Strategy | Description | What it attacks | Label |
|----------|-------------|-----------------|-------|
| A: Utilization modulation | Insert `time.sleep()` between training batches to mimic rendering's bursty pattern | GPU util CV, periodicity features | `ml_training_adversarial_A` |
| B: Low-utilization training | Batch size 1 + clock throttle to resemble inference | Utilization level, power draw | `ml_training_adversarial_B` |
| C: Instruction mix obfuscation | Disable AMP + run dummy FP64 kernels alongside training | Tensor core features, instruction mix | `ml_training_adversarial_C` |
| D: Temporal disruption | Randomize batch size each step (1–64), vary learning rate wildly | Epoch periodicity, autocorrelation | `ml_training_adversarial_D` |
| E: PMC signature spoofing | Dummy CUDA kernels emitting non-training PMC patterns | Tier 3 Nsight features | `ml_training_adversarial_E` |

- Use ResNet-18 CIFAR-10 as the base training workload for all strategies (fastest iteration)
- Collect **all three tiers simultaneously** for each adversarial run:
  - Tier 1: pynvml (always)
  - Tier 3: Nsight Compute 1 run per strategy (unrestricted on local 5070 Ti)
  - Tier 5: Shelly Plug wall power (key question: does physical power reveal true training even when software telemetry is spoofed?)
- Record training throughput (samples/sec) and total energy for each strategy vs. clean baseline — this becomes the cost-of-evasion table

**Target:** 15 adversarial parquet files + 5 Nsight profiles + 15 wall-power logs. Key paper deliverable: **cost-of-evasion table** (strategy × throughput penalty × energy overhead × detection rate per tier).

---

#### Thursday April 9: Cross-GPU Classifier Evaluation + Tier 5 Analysis (P0)

No new data collection. This is analysis day — the results feed directly into paper tables and figures.

**Tasks — Cross-architecture generalization experiments:**
1. **A100-only → 5070 Ti:** Train on A100 data (Weeks 3–5), test on 5070 Ti data from Monday. Report binary and multi-class accuracy.
2. **A100-only → 4050:** Train on A100 data, test on 4050 laptop data from Tuesday. Report same.
3. **Leave-one-GPU-out:** For each of the 3 GPUs, train on the other two, test on the held-out GPU. 3×2 accuracy matrix.
4. **Pooled classifier:** Train on all 3 GPUs combined, report accuracy via grouped cross-validation (group = GPU type).
5. **Feature stability heatmap:** For top-20 features by importance, compute mean ± std per workload on each GPU. Which features are architecture-invariant? Which break?

**Tasks — Adversarial robustness evaluation:**
6. Evaluate the A100-trained classifier on Wednesday's adversarial runs: detection rate per strategy
7. Evaluate Tier 5 wall-power classifier: train a simple model on clean wall-power data, test on adversarial wall-power data
8. **Key result for paper:** detection rate table (5 strategies × {Tier 1, Tier 5}), plus cost-of-evasion columns

**Tasks — Generate paper figures:**
- Cross-GPU accuracy comparison bar chart (Figure for Section 5)
- Feature stability heatmap across 3 architectures (Figure for Section 5)
- Adversarial detection rate heatmap (Figure for Section 5)
- Cost-of-evasion scatter: throughput penalty (%) vs. detection rate (%)
- Wall power vs. nvidia-smi reported power comparison (Figure for Section 5 or Appendix)

**Target:** All numerical results for the paper's two core tables: (1) cross-GPU generalization table, (2) adversarial robustness table. Figures generated.

---

#### Friday April 10: Adversarial Iteration + Paper Writing (P1)

**Tasks — Second-round adversarial iteration:**
- Retrain the classifier including Wednesday's adversarial examples in the training set (adversarial training)
- Re-evaluate: which strategies still evade the hardened classifier?
- If any strategy still evades: add evasion-resistant features (cumulative memory growth, backward-pass duration ratio) and retrain
- Document the adversary-defender evolution (1 paragraph for the paper)

**Tasks — Begin paper writing for Week 8 results:**
- Draft **Experimental Setup** section: 3 GPU architectures table, workload registry, data collection methodology, batch-size adjustments for VRAM-constrained GPUs
- Draft **Results — Cross-GPU Generalization** subsection using Thursday's tables/figures
- Draft **Results — Adversarial Robustness** subsection using Thursday's tables/figures
- Draft **Results — Physical Side Channel** subsection: wall-power vs. software telemetry, adversarial evasion resistance

**Target:** First complete draft of Results section. Adversary-defender iteration documented.

---

#### Week 8 Checkpoint

By end of Friday April 10:
- [ ] **~84 new parquet files** across 2 local GPUs (42 per GPU), covering all 14 workload types × 3 reps
- [ ] **15 adversarial parquet files** on 5070 Ti with simultaneous Tier 1 + Tier 5 collection
- [ ] **Cross-GPU generalization table** (3 architectures: A100, 5070 Ti, 4050 Laptop) — the paper's central result
- [ ] **Adversarial robustness table** (5 strategies × Tier 1 + Tier 5 detection rate + cost-of-evasion)
- [ ] **Wall-power vs. software-power comparison** showing Tier 5 evasion resistance
- [ ] **Paper Results section** first draft complete with figures
- [ ] Dataset grows from ~60 runs (1 GPU) to **~160 runs (3 GPUs)** — substantial enough for publication

**What's left for Weeks 9–10:** Paper polish (Introduction, Discussion, Conclusion), second adversarial iteration if needed, anonymization, formatting, and submission by April 24.

### Week 9: Paper Finalization + Submit TAIGR (~April 15 – April 24)

**⚠️ TAIGR SUBMISSION DEADLINE: APRIL 24, 2026, 23:59 AOE.**

**Tasks — Final experiments (April 15–17):**
- **Second round of adversary-defender iteration:** refine attack against hardened classifier, then defend again. Document the evolution.
- Add evasion-resistant features: cumulative FLOP counting, tensor core presence detection, memory allocation stability
- Any remaining gap-filling experiments (additional adversarial runs, edge cases)
- Finalize cost-of-evasion table: strategy, detection rate (Tier 1), detection rate (Tier 3), detection rate (Tier 5 wall-power), throughput penalty %, energy overhead %

**Tasks — HEM design proposal:**
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

**Tasks — TAIGR paper finalization (April 17–23):**
- Complete all remaining paper sections; ensure Discussion includes specific policy recommendations (TAIGR requirement)
- Finalize all figures and tables (publication-quality)
- Write Abstract and Conclusion
- Full paper proofread and anonymization check (double-blind: remove author names, institutional references, GitHub URLs)
- Verify page limit: 8 pages max for full paper (or 2 for short paper), excluding references/appendices
- Verify ICML 2026 LaTeX template formatting
- **April 23:** Internal review — circulate to collaborator and mentor for final comments
- **April 24 (before 23:59 AOE):** Submit via [OpenReview](https://openreview.net/group?id=ICML.cc/2026/Workshop/TAIGR)

**Tasks — Policy framing for TAIGR (critical for acceptance):**
- TAIGR requires "authors of more technical-leaning papers should meaningfully interact with policy motivations and implications of their work, and, where suitable, provide specific policy recommendations"
- Include a dedicated subsection on policy recommendations:
  - Mandate Tier 1 NVML telemetry logging on cloud GPU instances as a low-cost governance baseline
  - Define cost-of-evasion thresholds: if adversarial disguise costs >X% throughput, monitoring is economically effective
  - Recommend physical side-channel (wall power) audits as tamper-resistant verification for high-assurance scenarios
  - Discuss privacy implications of GPU telemetry monitoring and appropriate access controls

**Checkpoint:** **TAIGR paper submitted by April 24.** Adversarial evaluation table complete (5 strategies x 3 tiers). Cost-of-evasion quantified. Two rounds of adversary-defender iteration documented. HEM proposal drafted.

### Week 10: Code Cleanup and Report Preparation

**Tasks:**
- Clean up classifier code into a package with README, requirements, and a live-prediction demo script
- Expand TAIGR paper into full technical report (15-25 pages) — the paper provides the core; the report adds extended results, full literature review, and appendices
- Finalize signal comparison chart as publication-quality table/heatmap
- Address any reviewer feedback if TAIGR notifications arrive early (unlikely before May 14)

**Checkpoint:** Classifier code documented and reproducible. Full report draft well underway.

### Week 11: Report and Final Deliverables

**Tasks:**
- Complete and polish the full technical report (15-25 pages):
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
- [ ] **TAIGR @ ICML 2026 workshop paper** (submitted April 24; notification expected May 14-15)
- [ ] Technical report (PDF, 15-25 pages — expanded version of TAIGR paper)
- [ ] Literature review / annotated bibliography
- [ ] Signal comparison table/chart
- [ ] HEM telemetry design proposal (with physical side-channel empirical grounding)
- [ ] Classifier code repository with README and demo
- [ ] Collected dataset with documentation (3 GPU architectures: A100, H100, RTX 5070 Ti)
- [ ] Presentation slides
- [ ] Revised ISCA paper draft
- [ ] TODO (repo cleanup): Remove PDFs from `literature/` and replace with a `literature/bibliography.md` containing full citations and links.

**Post-Week 11 (if TAIGR accepted, notification ~May 14-15):**
- Prepare camera-ready version (9 pages for full paper, 3 for short paper)
- At least one author must attend ICML 2026 in Seoul (July 10-11) to present poster
- Outstanding papers may be invited for lightning talks

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
