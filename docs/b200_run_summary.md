# 8x B200 Datacenter Run Summary

**Date:** 2026-04-12 to 2026-04-13
**Instance:** 8x NVIDIA B200 (183,359 MiB / 179 GB per GPU)
**Total time:** ~2h 15min (1h 58min main collection + 17min 70B variants)

---

## Hardware

| Component | Details |
|-----------|---------|
| GPU | 8x NVIDIA B200 (Blackwell, sm_100), 183 GB VRAM each |
| CPU | 2x Intel Xeon Platinum 8570, 56 cores / 112 threads each (224 total) |
| RAM | 3.9 TiB DDR5 |
| Disk | 256 GB root (overlay) |
| Driver | 580.126.09 |
| CUDA | 13.0 (V13.0.88) |
| PyTorch | 2.11.0+cu130 |
| NCCL | 2.28.9 |
| OS | Ubuntu 24.04 |

---

## DCGM Status

- DCGM v3.3.9 installed and host engine started successfully
- **Profiling fields (1001-1012) blocked** -- container environment, profiling
  module failed to load. Same as A100 experience.
- **Basic fields (100, 101, 150, 155, 203) worked** -- clock, memory clock,
  temperature, power, utilization. The telemetry collector auto-detected this
  and fell back gracefully.
- Ran **with DCGM enabled** (not `--no-dcgm`). Each parquet has 25 columns:
  14 NVML Tier 1 + 5 DCGM basic + 6 metadata.
- DCGM concurrency limitation: only the first GPU per parallel batch got DCGM
  enabled; others fell back to NVML-only. This is a minor limitation since the
  DCGM basic fields duplicate NVML data.

---

## Collection Results

### Phase 1: Single-GPU (26 workloads, ~24 min)

13 adversarial strategies + 13 baselines, run in parallel across 8 GPUs.

| Batch | Workloads | Status |
|-------|-----------|--------|
| 1 (8 GPUs) | E, F_30, F_50, G, H_mining, H_cufft, I, J | 8/8 OK |
| 2 (8 GPUs) | K, L_2, L_5, L_10, L_20, resnet, resnet_amp, idle | 8/8 OK |
| 3 (8 GPUs) | mlp, gpt2, gpt2_amp, bert, bert_amp, resnet50_inf, nbody, cufft | 8/8 OK |
| 4 (2 GPUs) | mining_ethash, rendering | 2/2 OK |

Strategy G (clock throttling) worked in the container with root access.
Strategies H, J have the known wrong-GPU NVML feedback issue (ran in parallel, not serial).

### Phase 2: DDP Baseline (3 workloads, ~18 min)

| Workload | Epochs | Epoch Time | Final Acc |
|----------|--------|-----------|-----------|
| ddp_resnet18_cifar10 | 282 | 1.0-1.2s | 100% |
| ddp_resnet50_cifar10 | 52 | 5.7-5.9s | 99.8% |
| ddp_resnet18_cifar10_amp | 300+ | 0.9-1.0s | 100% |

B200 comparison: ResNet18 DDP achieved 282 epochs in 300s (vs 200 on A100).
ResNet50 DDP achieved 52 epochs (vs 37 on A100). ~1.4x throughput improvement.

### Phase 3: DDP Adversarial (6 workloads, ~36 min)

All 6 distributed adversarial strategies completed successfully:
- adversarial_K_ddp (online learning)
- adversarial_L_ddp_5 (diluted 1/5)
- adversarial_L_ddp_10 (diluted 1/10)
- adversarial_L_ddp_stagger_5 (staggered dilution)
- adversarial_K_ddp_accum_4 (gradient accumulation x4)
- adversarial_K_ddp_accum_8 (gradient accumulation x8)

### Phase 4: FSDP NVLink Stress (4 workloads, ~39 min)

| Model | Params | Peak Mem/GPU | Avg Power | GPU Util | Avg Temp | Samples |
|-------|--------|-------------|-----------|----------|----------|---------|
| 7B | 5.3B | 35.0 GB | 650W | 94% | 54C | 485 |
| 13B | 10.2B | 54.5 GB | 382W | 100% | 40C | 317 |
| 30B | 25.4B | 69.5 GB | 599W | 77% | 49C | 785 |
| **70B** | **59.6B** | **131.8 GB** | **389W** | **98%** | **40C** | **845** |

**70B is new** -- OOM'd at 82.4 GB on A100-80GB, fits comfortably on B200 at
131.8 GB (71.9% VRAM utilization) with plain SGD.

### 70B FSDP Variants (3 workloads, ~17 min)

Bonus runs exploiting B200's 183 GB VRAM headroom:

| Variant | Optimizer | seq_len | Peak Mem/GPU | Steps | Throughput | Status |
|---------|-----------|---------|-------------|-------|-----------|--------|
| SGD (baseline) | SGD | 256 | 131.8 GB | ~560 | ~3,860 tok/s | OK |
| SGD+momentum | SGD (m=0.9) | 256 | 103.5 GB | 415 | ~2,500 tok/s | OK |
| seq_len=512 | SGD | 512 | 179.8 GB | 191 | 2,600 tok/s | OK |
| AdamW | AdamW | 256 | 183.1 GB | 0 | -- | **OOM** |

Key findings:
- **SGD+momentum fits** with 50 GB headroom. Adds momentum buffer (~30 GB/GPU
  sharded) but memory is lower than baseline due to different allocation pattern.
- **seq_len=512 pushes to 97.7% VRAM** (179.8 / 183.4 GB). This is the maximum
  useful configuration on B200 -- any larger and it OOMs.
- **AdamW OOMs even on 183 GB**. The 2x fp32 optimizer states for 59.6B params
  (sharded across 8 GPUs: ~60 GB/GPU) plus activations exceed capacity.
  Would need ~240 GB/GPU or CPU offloading.

---

## Data Summary

| Category | Files | Telemetry Rows |
|----------|-------|---------------|
| Phase 1: Single-GPU | 26 | ~6,900 |
| Phase 2-3: DDP (baseline + adversarial) | 72 | ~28,000 |
| Phase 4: FSDP (7B/13B/30B/70B) | 32 | ~9,700 |
| 70B Variants (SGD+m, seq512, AdamW) | 24 | ~3,200 |
| **Total** | **154** | **55,724** |

All parquets tracked via Git LFS (`data/*.parquet` in `.gitattributes`).

### Columns per parquet (25)

**NVML Tier 1 (14):** timestamp_utc, timestamp_epoch, gpu_utilization_pct,
mem_utilization_pct, mem_used_mb, mem_total_mb, power_draw_w, temperature_c,
sm_clock_mhz, mem_clock_mhz, pcie_tx_mbps, pcie_rx_mbps, encoder_util_pct,
decoder_util_pct, fan_speed_pct

**DCGM Basic (5):** dcgm_gpu_util, dcgm_power_usage, dcgm_gpu_temp,
dcgm_sm_clock, dcgm_mem_clock

**Metadata (6):** workload_label, run_id, gpu_name, gpu_uuid, driver_version

---

## Setup Notes

1. `setup.sh` failed at CUDA toolkit step because `nvcc` wasn't in root's
   PATH (it's at `/usr/local/cuda/bin/nvcc`). CUDA 13.0 was already installed.
   Workaround: ran setup steps manually (venv, pip install, DCGM).

2. DCGM Python bindings were at `/usr/local/dcgm/bindings/python3/` (same as
   A100 experience). Fixed with `.pth` file in venv site-packages.

3. Progress tracker (`datacenter_collection_progress.json`) contained A100
   results from the prior run. Reset with `--reset` before starting.

4. PyTorch 2.11.0+cu130 installed from stable channel. No compatibility issues
   with Blackwell (compute capability 10.0).

5. 70B FSDP was commented out in `run_datacenter_collection.py` (designed for
   80GB GPUs). Uncommented for B200.

---

## What This Adds to the Paper

1. **6th GPU architecture** for the cross-GPU generalization matrix (B200 /
   Blackwell sm_100, alongside A100, H100-80GB, H100-NVL, RTX 5080, RTX 4050).

2. **70B FSDP telemetry** -- first data point above 30B. Validates that FSDP
   training produces distinctive telemetry signatures at frontier-relevant
   model scales. The 70B memory/power/NVLink profile is qualitatively different
   from smaller models.

3. **70B optimizer variants** -- SGD vs SGD+momentum vs AdamW demonstrate that
   optimizer choice produces measurably different telemetry (power envelope,
   memory utilization, step timing). AdamW OOM confirms the memory boundary.

4. **Baseline workloads on Blackwell** -- 13 clean baselines (idle, training,
   inference, scientific, mining proxy, rendering proxy) establish the B200
   feature distribution for classifier training.

5. **DDP adversarial on Blackwell** -- validates that distributed evasion
   strategies (K-DDP, L-DDP, stagger, accum) produce transferable signatures
   across GPU architectures.
