# Literature Review Summary

**Total papers:** 5

## Reading Progress

- unread: 5
- skimmed: 0
- read: 0
- annotated: 0

## Papers by Tag

### governance (3 papers)

- **Hardware-Enabled Governance Mechanisms: Developing Technical Solutions to Manage AI Chip Access** (2024)
  Tim Fist, Alan Hickey
  > Explores hardware-enabled governance mechanisms for AI chips, including firmware controls, cryptographic attestation, and performance monitoring.
  Tags: governance
  Notes: RAND report on hardware governance mechanisms.
  Status: [ ] unread

- **Secure, Governable Chips: Using On-Chip Mechanisms to Manage National Security Risks from AI & Advanced Computing** (2024)
  Onni Aarne, Tim Fist, Caleb Withers
  > Proposes on-chip governance mechanisms including secure boot, remote attestation, and usage logging for AI accelerator chips to enable compute governance.
  Tags: governance, nvidia
  Notes: CNAS policy report. Key reference for governance motivation.
  Status: [ ] unread

- **ShadowScope: From GPU Kernel Validation to Governing AI Chip Usage** (2025)
  Andrzej Jackowski, Marcin Copik, Torsten Hoefler
  > Uses GPU hardware performance counters via CUPTI PMU to validate GPU kernel behavior and distinguish ML training from other workloads.
  Tags: perf-counters, governance, gpu-profiling, ml-training
  Notes: Core reference for this SPAR project. Directly addresses our research question.
  Status: [ ] unread

### gpu-profiling (3 papers)

- **Rendered Insecure: GPU Side Channel Attacks are Practical** (2018) [cited: 180]
  Hoda Naghibijouybari, Ajaya Neupane, Zhiyun Qian et al.
  > Demonstrates practical GPU side-channel attacks exploiting performance counters and resource contention to leak information about GPU computations.
  Tags: side-channel, gpu-profiling
  Notes: Foundational GPU side-channel work. CCS 2018.
  Status: [ ] unread

- **Forecasting GPU Performance for Deep Learning Training and Inference** (2025)
  Seonho Lee, Amar Phanishayee, Divya Mahajan
  > Decomposes deep learning workloads into GPU kernel types (GEMM, convolution, etc.) to forecast performance, revealing distinctive resource utilization patterns.
  Tags: gpu-profiling, ml-training
  Notes: ASPLOS 2025. Useful for understanding ML training kernel signatures.
  Status: [ ] unread

- **ShadowScope: From GPU Kernel Validation to Governing AI Chip Usage** (2025)
  Andrzej Jackowski, Marcin Copik, Torsten Hoefler
  > Uses GPU hardware performance counters via CUPTI PMU to validate GPU kernel behavior and distinguish ML training from other workloads.
  Tags: perf-counters, governance, gpu-profiling, ml-training
  Notes: Core reference for this SPAR project. Directly addresses our research question.
  Status: [ ] unread

### ml-training (2 papers)

- **Forecasting GPU Performance for Deep Learning Training and Inference** (2025)
  Seonho Lee, Amar Phanishayee, Divya Mahajan
  > Decomposes deep learning workloads into GPU kernel types (GEMM, convolution, etc.) to forecast performance, revealing distinctive resource utilization patterns.
  Tags: gpu-profiling, ml-training
  Notes: ASPLOS 2025. Useful for understanding ML training kernel signatures.
  Status: [ ] unread

- **ShadowScope: From GPU Kernel Validation to Governing AI Chip Usage** (2025)
  Andrzej Jackowski, Marcin Copik, Torsten Hoefler
  > Uses GPU hardware performance counters via CUPTI PMU to validate GPU kernel behavior and distinguish ML training from other workloads.
  Tags: perf-counters, governance, gpu-profiling, ml-training
  Notes: Core reference for this SPAR project. Directly addresses our research question.
  Status: [ ] unread

### nvidia (1 papers)

- **Secure, Governable Chips: Using On-Chip Mechanisms to Manage National Security Risks from AI & Advanced Computing** (2024)
  Onni Aarne, Tim Fist, Caleb Withers
  > Proposes on-chip governance mechanisms including secure boot, remote attestation, and usage logging for AI accelerator chips to enable compute governance.
  Tags: governance, nvidia
  Notes: CNAS policy report. Key reference for governance motivation.
  Status: [ ] unread

### perf-counters (1 papers)

- **ShadowScope: From GPU Kernel Validation to Governing AI Chip Usage** (2025)
  Andrzej Jackowski, Marcin Copik, Torsten Hoefler
  > Uses GPU hardware performance counters via CUPTI PMU to validate GPU kernel behavior and distinguish ML training from other workloads.
  Tags: perf-counters, governance, gpu-profiling, ml-training
  Notes: Core reference for this SPAR project. Directly addresses our research question.
  Status: [ ] unread

### side-channel (1 papers)

- **Rendered Insecure: GPU Side Channel Attacks are Practical** (2018) [cited: 180]
  Hoda Naghibijouybari, Ajaya Neupane, Zhiyun Qian et al.
  > Demonstrates practical GPU side-channel attacks exploiting performance counters and resource contention to leak information about GPU computations.
  Tags: side-channel, gpu-profiling
  Notes: Foundational GPU side-channel work. CCS 2018.
  Status: [ ] unread

## Year Distribution

- 2018: 1 papers
- 2024: 2 papers
- 2025: 2 papers
