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

### Local Hardware Update (April 2026)

**Desktop with RTX 5080 loaned for 3 weeks from another researcher.** The original plan deferred local hardware to "if grant approved." A desktop with an RTX 5080 has been loaned starting Week 8, providing local Blackwell-architecture GPU access for cross-GPU and adversarial experiments.

| Item | Notes |
|------|-------|
| NVIDIA GeForce RTX 5080, 16GB GDDR7 | Blackwell architecture — complements cloud B100/B200; enables Tier 5 physical side channels |
| Desktop (Intel Core Ultra 9 285K, 32GB RAM, 2TB NVMe) | Loaned; PCIe 5.0 x16; Windows 11 |
| Kill-A-Watt P4400 + Shelly Plus Plug US | ~$55 — manual + automated wall-power measurement |

**What this enables (Weeks 8+):**
- Cross-architecture generalization: Ampere (A100) vs Hopper (H100) vs **Blackwell (cloud B100/B200 + local RTX 5080)** — three GPU generations
- FFmpeg NVENC video encoding workload (A100/H100 lack NVENC hardware encoder)
- GROMACS with proper GPU build (blocked on Vast.ai)
- Unrestricted Nsight Compute (Tier 3) access on local hardware — blocked by virtualization on cloud. Note: DCGM Tier 2 profiling fields are unavailable on consumer GPUs (RTX 5080) regardless of OS; Tier 2 data from cloud data-center GPUs only.
- Tier 5 physical side channels: automated wall-power logging via Shelly Plug REST API (~1 Hz)
- Adversarial workloads with simultaneous Tier 1 + Tier 3 + Tier 5 measurement

**Timeline:** Loaned desktop available starting Week 8. Local experiments begin immediately. Project extended to 11 weeks to accommodate.

### TAIGR Workshop Paper (Added March 2026)

**New deliverable:** Workshop paper submitted to [TAIGR @ ICML 2026](https://taigr-workshop.com/) (Second Workshop on Technical AI Governance Research, Seoul, July 10–11, 2026).

| Detail | Value |
|--------|-------|
| Submission deadline | **April 24, 2026, 23:59 AOE** |
| Format | Full paper (8 pages) or short paper (2 pages), ICML 2026 LaTeX template |
| Review | Double-blind, by both ML and governance/policy reviewers |
| Proceedings | Non-archival (workshop website only); dual submission OK |
| Notification | ~May 14–15, 2026 |
| Presentation | Poster (in person, Seoul); select papers get lightning talks |

**Why TAIGR is the right venue:** Our work directly addresses three of TAIGR's five governance pillars — Verification (can we verify what workloads run on GPUs?), Access (external monitoring of GPU activity), and Operationalisation (translating AI governance goals into technical monitoring tools). TAIGR requires policy recommendations alongside technical contributions, which aligns with our HEM proposal.

**Paper writing timeline (integrated into Weeks 6–9):**
- **Week 6:** LaTeX template setup; draft Related Work + Methodology (from existing literature review and pipeline)
- **Week 7:** Draft Introduction + Experimental Setup + cross-GPU Results
- **Week 8:** Draft adversarial Results + Discussion/Policy section; ~80% complete
- **Week 9 (April 15–24):** Finalize all sections, generate figures, anonymize, internal review, **submit by April 24**

### Earlier Plan Changes (Post-Literature-Review)

| Area | Before | After |
|------|--------|-------|
| Weeks 1–5 | Split between local + cloud | **Cloud-only** (Vast.ai A100 + H100). Full $1,000 cloud budget. |
| Week 6 | Classifier refinement | **Paper scaffold** + cloud B100/B200 Blackwell data collection begins |
| Week 7 | Adversarial workloads | **Cloud Blackwell experiments** + preliminary cross-GPU analysis + **paper results draft** |
| Week 8 | Adversarial robustness | **Local desktop setup (loaned RTX 5080)** + local/laptop data collection + **adversarial workloads** (5 strategies) + **Tier 5 physical side channels** + **paper ~80% complete** |
| Week 9 | HEM proposal | **Adversarial robustness testing** + HEM proposal + **TAIGR paper finalized and submitted (April 24)** |
| Week 10 | Report | Code cleanup + expand paper into full technical report |
| Week 11 (new) | — | **Final report + deliverables** |
| Tier 5 | Deferred/future work | **Empirically grounded** with Kill-A-Watt + Shelly Plug on local RTX 5080 |
| Cross-GPU | A100 vs 3090 (if grant) | **A100 vs H100 vs B100/B200/RTX 5080** (3 architecture generations: Ampere, Hopper, Blackwell) |
| ISCA paper | Limited to cloud data | **Revised** with cross-GPU generalization and adversarial robustness across 3 tiers |
| Publication | No external submission | **TAIGR @ ICML 2026 workshop paper** (deadline April 24) |
| Week 1 reading list | 4 papers | +2 papers (WAVE, Differential Architecture) |
| Week 2 pipeline | Build PMC harness from scratch | Clone and adapt WAVE repo for Tier 3; build Tier 1 NVML harness ourselves |
| Weeks 3–4 EDA | Re-derive bottleneck characteristics | **Skip re-derivation; cite Differential Architecture.** Focus on training-specific temporal signals |
| Week 5 features | Standard windowed statistics | + training-specific features: epoch periodicity, optimizer memory growth, compute/memory-BW ratio |
| Adversarial strategies | 4 strategies | 5 strategies: add **Strategy E (PMC signature spoofing)** |
| Adversary-defender | 1 round | **2 rounds** against Tier 1, Tier 3, and Tier 5 detectors |
| HEM proposal | Grounded in cloud experiments only | Grounded in Differential Architecture + WAVE + our 3-tier experiments + physical side-channel data |
| Report framing | General contribution | Explicitly positioned against WAVE and Differential Architecture |
