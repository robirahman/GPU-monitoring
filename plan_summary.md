# Literature Update and Plan Changes — Participant Summary

Two papers found during the literature search directly overlap with our original scope. Here is what you need to know and what is changing.

---

## What the Two Papers Already Established

**WAVE (Xu et al., ASPLOS '26 — `literature/Architecture Observation for Model Oversight.pdf`)**
WAVE uses GPU Performance Monitoring Counters (PMCs) collected via Nsight Compute to fingerprint LLM *inference* workloads. From nine metrics (FP16/FP32/FP64 instruction counts, memory transaction counts), they can identify model architecture family (GPT-2 vs. LLaMA vs. Qwen), recover batch size, layer count, token count, and approximate model size with ~7% error. Detection accuracy for model substitution fraud is 87–93%. The code is open-sourced at https://github.com/sept-usc/Wave. **Critical caveat:** overhead is 1,200–5,300% using Nsight Compute's re-execution profiling, which is completely impractical for deployed monitoring.

**Differential Architecture (Anonymous, ISCA '26 — `literature/Differential_Architecture_ISCA_2026 draft.pdf`)**
This paper analytically characterizes the hardware bottlenecks for every major GPU workload class. The key result: LLM *prefill* (MatMul) is exclusively compute-bound; LLM *decode* (vector multiplication) is exclusively memory-bandwidth-bound; scientific computing (FFT) is cache- and memory-bandwidth-bound; rendering (ray tracing) is compute-bound with secondary cache sensitivity. These findings are validated against cycle-accurate simulators at 12% average error. **We do not need to re-derive these bottleneck characteristics from our own experiments** — we cite the paper and verify our measurements are consistent.

---

## What This Means for Our Project

Neither paper addresses ML *training* workloads. Neither paper addresses an adversary actively trying to disguise their workload. Neither paper asks whether *always-available* Tier 1 NVML metrics (the kind from `nvidia-smi` at zero overhead) can substitute for expensive Nsight profiling. These three gaps are now our sharpened research focus.

**The core question is now:** Can always-available GPU telemetry (Tier 1 NVML) detect adversarially disguised ML training, and what does it cost the adversary to evade detection?

---

## Key Changes to the Plan

| Area | Before | After |
|------|--------|-------|
| **Hardware approach** | Buy RTX 3090 + Kill-A-Watt locally; split budget between local and cloud | **Cloud-only for Weeks 1–5.** Full $1,000 budget allocated to cloud compute (~1,300–1,500 A100-hours on Vast.ai). Local GPU deferred to Weeks 6+ if hardware grant is approved. |
| **Budget** | ~$550 local GPU + ~$400 cloud | ~$1,000 cloud compute; physical hardware (~$575) funded separately by grant if approved |
| **Tier 5 (physical side channels)** | Collected via Kill-A-Watt on local GPU | Deferred/grant-dependent; becomes a future-work recommendation if no local hardware |
| Week 1 | Order RTX 3090; set up cloud | Cloud platform setup only (Vast.ai + RunPod); identify bare-metal hosts for Nsight access |
| Week 2 | Install local GPU + drivers; build pipelines | All environment setup on cloud instances; build and test pipelines on cloud A100 |
| Weeks 3–4 | Data collection on local RTX 3090 + cloud A100 | All data collection on cloud A100; no wall-power or local GPU measurements |
| Week 6 | Cross-GPU: train on 3090, test on A100 | Cross-GPU with local 3090 *only if grant approved*; otherwise test across cloud host configurations |
| Week 7 | Adversarial workloads on local GPU | Adversarial workloads on cloud; local GPU data *if grant approved* |
| Week 9 HEM | Physical side-channel recommendations | Physical side-channel recommendations empirically grounded *only if grant approved*; otherwise cited as future work |
| **Risks** | "RTX 3090 arrives defective" | Replaced with "Hardware grant not approved" and "Cloud budget exhaustion" |
| Week 1 reading list | 4 papers | +2 papers (WAVE, Differential Architecture); checkpoint is a summary of those two specifically |
| Week 2 pipeline | Build PMC harness from scratch | Clone and adapt the WAVE repo for Tier 3 collection; build Tier 1 NVML harness ourselves |
| Weeks 3–4 EDA | Re-derive workload bottleneck characteristics | **Skip bottleneck re-derivation; cite Differential Architecture.** Focus EDA on training-specific temporal signals (epoch periodicity, optimizer memory growth, forward/backward asymmetry) and the training vs. inference distinction |
| Week 5 features | Standard windowed statistics | + training-specific features: epoch periodicity score, optimizer state memory growth rate, compute/memory-BW ratio as a feature |
| Week 6 | Classifier refinement only | + new experiment: Tier 1 NVML classifier vs. Tier 3 WAVE-style classifier — quantify the accuracy vs. overhead tradeoff |
| Week 7 | 4 adversarial strategies | 5 strategies: add **Strategy E (PMC signature spoofing)** — dummy kernels crafted to produce WAVE-like PMC signatures while running real training |
| Week 8 | 1 round of adversary-defender iteration | **2 rounds**; evaluate all 5 strategies against both Tier 1 and Tier 3 detectors; this is the primary novel contribution |
| Week 9 HEM proposal | Grounded in our experiments | Also grounded in Differential Architecture (which resources to target) and WAVE (overhead tradeoff, tiered monitoring argument) |
| Report framing | General contribution | Explicitly positioned against WAVE and Differential Architecture in the introduction |
