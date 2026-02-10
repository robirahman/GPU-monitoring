# SPAR GPU Workload Classification: 10-Week Project Plan

## Context

This is a SPAR research project on adversarial classification of ML training workloads on GPUs. The goal is to determine whether on-chip telemetry and side-channel measurements can reliably distinguish ML training from other GPU workloads (scientific HPC, crypto mining, rendering), even when an adversary actively disguises their workload. The deliverables are a technical report, a signal comparison table, a HEM (Hardware Enabled Mechanism) telemetry design proposal, and a prototype classifier. Budget: $1,000 for compute.

---

## Hardware & Infrastructure

**Local GPU: Buy a used NVIDIA RTX 3090 (~$500-600)**
- 24GB VRAM, Ampere architecture, full NVML support
- Gives direct access to: power draw, temperature, GPU/memory utilization, clock speeds, PCIe throughput, fan speed via `nvidia-smi` and `pynvml`
- Advanced per-kernel profiling via Nsight Compute (`ncu`) and CUPTI
- Note: GPM metrics (per-SM pipe utilization, tensor core %) are Hopper-only; use Nsight Compute for kernel-level instruction mix on Ampere
- Also buy a Kill-A-Watt power meter (~$25) for wall-power side-channel measurements

**Cloud GPU: Use Vast.ai for A100 access (~$0.50-1.00/hr)**
- Remaining ~$400 budget provides 400-800 A100-hours
- Vast.ai gives root Docker containers where you can install DCGM and access Tier 2 metrics (tensor core utilization, FP16/FP32/FP64 pipe utilization, DRAM bandwidth)
- Optionally test RunPod or AWS briefly to compare telemetry access across platforms

**Budget breakdown:**
| Item | Cost |
|------|------|
| Used RTX 3090 | ~$550 |
| Kill-A-Watt meter | ~$25 |
| Vast.ai cloud compute | ~$400 |
| Platform comparison tests | ~$25 |
| **Total** | **~$1,000** |

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

**Tier 1 - Basic NVML (available on RTX 3090, 1 Hz sampling):**
GPU utilization %, memory utilization %, memory used (MB), power draw (W), temperature (C), SM clock (MHz), memory clock (MHz), PCIe TX/RX throughput (MB/s), encoder/decoder utilization %, fan speed %

**Tier 2 - DCGM (available on cloud A100):**
Tensor core active %, FP16/FP32/FP64 pipe utilization, SM occupancy %, DRAM read/write bandwidth, NVLink bandwidth

**Tier 3 - Per-kernel profiling (Nsight Compute):**
Instruction mix (INT, FP16, FP32, FP64, tensor ops), kernel duration, achieved occupancy, memory throughput per kernel

**Tier 4 - Temporal/behavioral patterns (derived):**
Periodicity, memory allocation stability, power draw variance, communication patterns

**Tier 5 - Physical side channels (local GPU only):**
Wall power draw (Kill-A-Watt), acoustic emissions (stretch goal)

---

## Week-by-Week Plan

### Week 1: Kickoff, Literature Review, Hardware Procurement

**Tasks:**
- Read core papers:
  - Shavit (2023), "What does it take to catch a Chinchilla?" (arXiv:2303.11341)
  - Kulp et al. (2024), "Hardware-Enabled Governance Mechanisms" (RAND)
  - "GPU Under Pressure: Estimating Application's Stress via Telemetry" (arXiv:2511.05067)
  - "Detecting Covert Cryptomining using HPC" (arXiv:1909.00268)
- Order RTX 3090 from eBay (verify seller, check return policy)
- Create accounts on Vast.ai; spin up a test instance for 1 hour to verify `nvidia-smi dmon` and DCGM access
- Set up Git repository with directory structure: `literature/`, `data/`, `scripts/`, `notebooks/`, `workloads/`, `classifier/`, `report/`

**Checkpoint:** Mentees submit 1-page summary of 2 most relevant papers. GPU ordered. Cloud access verified.

### Week 2: Environment Setup and Telemetry Pipeline

**Tasks:**
- Install local GPU, NVIDIA drivers (550+), CUDA 12.x, pynvml, Nsight tools
- Build data collection harness (`scripts/collect_telemetry.py`): polls pynvml at 1 Hz, logs to CSV/Parquet with metadata (workload label, GPU model, timestamps)
- Build workload launcher (`scripts/run_workload.py`): starts telemetry, launches workload, stops telemetry, saves labeled data
- Test pipeline end-to-end: idle GPU + simple PyTorch training
- Complete literature review and annotated bibliography
- Finalize cloud platform choice based on Week 1 testing

**Checkpoint:** Pipeline produces clean, labeled data files. Literature review draft complete.

### Week 3: Data Collection Round 1

**Tasks:**
- Run all representative workloads on local RTX 3090 (10-15 min each, 3 runs minimum per workload)
- Collect Tier 1 metrics via pynvml harness for every run
- Record wall power via Kill-A-Watt during each run
- Run Nsight Compute profiles for 1 run of each workload type (`ncu --set full`)
- Run selected workloads (ResNet-50, GPT-2, GROMACS, mining) on cloud A100 with DCGM for Tier 2 metrics
- Organize data: `{workload}_{gpu}_{run}_{date}.parquet`

**Checkpoint:** 15-20+ workload runs complete. Tier 1 data for all, Tier 2 data for key workloads, Nsight profiles collected.

### Week 4: Data Collection Round 2 and Exploratory Analysis

**Tasks:**
- Complete any remaining runs; add edge cases (ML inference, mixed workloads, short training, DataLoader-bottlenecked training)
- Exploratory data analysis:
  - Time-series plots of each metric per workload type
  - Summary statistics (mean, std, autocorrelation) per workload
  - PCA/t-SNE visualization of workload feature vectors
- Build draft signal comparison table (Deliverable 4): workload types vs. metrics, with observed values
- Analyze Nsight Compute kernel profiles: instruction mix, GEMM prevalence, kernel repetition patterns
- Write 1-page summary: "Which metrics look most promising for classification?"

**Checkpoint:** 30+ runs in dataset. EDA notebook complete. Draft signal comparison table started.

### Week 5: Feature Engineering and Baseline Classifier

**Tasks:**
- Feature engineering: 60-second sliding windows, compute per metric: mean, std, min, max, CV, skewness, autocorrelation at lags 1/5/10/30s, dominant FFT frequency (~120 features per window)
- Split by workload run (not window) to prevent leakage: 70/15/15 train/val/test
- Train classifiers: Random Forest, XGBoost, SVM, Logistic Regression
- Report accuracy, per-class precision/recall/F1, confusion matrices for both binary (ML training vs. rest) and multi-class
- Extract feature importance rankings from tree models
- Error analysis: which workload pairs are most confused?

**Checkpoint:** Target >85% binary accuracy on clean data. Feature importance ranking produced.

### Week 6: Classifier Refinement

**Tasks:**
- Add cross-metric features (power/utilization ratio, memory allocation rate, PCIe burstiness)
- Test window sizes: 30s, 60s, 120s, 300s -- report accuracy vs. window size
- Train time-series models (1D CNN, LSTM) on raw metric sequences; compare with feature-engineered approach
- Test with Tier 2 metrics from cloud A100: how much does tensor core utilization improve accuracy?
- Cross-GPU generalization: train on RTX 3090 data, test on A100 data (and vice versa)

**Checkpoint:** Target >90% binary, >80% multi-class on clean data. Window size sensitivity and cross-GPU results documented.

### Week 7: Adversarial Workload Design

**Tasks:**
- Design and implement 3-4 disguise strategies:
  - **A: Utilization modulation** -- insert sleeps/dummy work between training batches to mimic rendering patterns
  - **B: Low-utilization training** -- tiny batch sizes + throttling to resemble inference
  - **C: Instruction mix obfuscation** -- disable mixed-precision + run dummy FP64 kernels alongside training
  - **D: Temporal disruption** -- randomize batch sizes, vary compute intensity to break epoch periodicity
- Run each adversarial workload 3+ times, collect telemetry with same pipeline
- Label as `ml_training_adversarial_A/B/C/D`

**Checkpoint:** 3-4 adversarial strategies implemented. Adversarial telemetry data collected.

### Week 8: Adversarial Robustness Testing

**Tasks:**
- Evaluate Week 6 classifier on adversarial data: detection rate per strategy
- Analyze successful evasions: which features shifted? By how much?
- Cost-of-evasion analysis: measure training throughput (samples/sec) and total energy with vs. without disguise
- Retrain classifier with adversarial examples in training set (adversarial training)
- Add evasion-resistant features: cumulative FLOP counting, tensor core presence detection, memory allocation stability
- If time: one round of adversary-defender iteration (modify attack, re-defend)

**Checkpoint:** Adversarial evaluation table complete. Cost-of-evasion quantified. Robustified classifier tested.

### Week 9: HEM Design Proposal and Code Cleanup

**Tasks:**
- Write HEM telemetry design proposal (Deliverable 5):
  - **Must-have metrics:** instruction type counters (tensor/FP16/FP32/FP64), cumulative FLOP counters, memory allocation patterns
  - **Should-have:** power draw time series, SM utilization, PCIe/NVLink volumes
  - For each: cite experimental evidence, evasion difficulty, recommended sampling rate, hardware vs. firmware implementation
  - Address privacy, tamper resistance, false positive considerations
- Clean up classifier code into a package with README, requirements, and a live-prediction demo script
- Begin report writing: outline, methodology section, results tables and figures

**Checkpoint:** HEM proposal drafted. Classifier code documented and reproducible. Report outline and methodology section done.

### Week 10: Report and Final Deliverables

**Tasks:**
- Write and polish the full technical report (15-25 pages):
  1. Introduction and research questions
  2. Literature review (Deliverable 1)
  3. Workload characterization (Deliverable 2)
  4. Experimental setup
  5. Classification results (Deliverable 3)
  6. Signal comparison table (Deliverable 4)
  7. Adversarial testing results
  8. HEM design proposal (Deliverable 5)
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

---

## Risks and Mitigations

| Risk | Mitigation |
|------|-----------|
| RTX 3090 arrives defective | Buy from seller with return policy; budget 1 week for shipping |
| DCGM limited on consumer GPU | Use Nsight Compute locally for kernel-level data; use cloud A100 for DCGM |
| Cloud platform blocks mining software | Use benchmark mode or write custom CUDA kernels mimicking mining patterns |
| Adversarial strategies too effective | This is a valid research finding -- document fragility and recommend hardened metrics |
| Insufficient data for robust classifier | Augment with varied hyperparameters; use smaller windows for more samples |

---

## Verification

- Telemetry pipeline: run a known workload, verify CSV output matches `nvidia-smi dmon` readings
- Classifier: k-fold cross-validation with per-run splits; report confidence intervals
- Adversarial testing: blind evaluation (classifier never sees adversarial strategy labels during training for the initial test)
- HEM proposal: ground each recommendation in specific experimental results with figure/table references
- End-to-end demo: run classifier live on GPU while running an unlabeled workload, verify correct prediction
