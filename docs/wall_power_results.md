# Tier 5 Wall-Power Results: GPU vs System-Level Power on RTX 5080

**Date:** 2026-04-12
**Hardware:** NVIDIA GeForce RTX 5080 (Blackwell), Intel Core Ultra 9 285K, 32GB RAM, Windows 11
**Smart plug:** TP-Link Tapo P115, polled at 1 Hz
**GPU telemetry:** pynvml (NVML) at 1 Hz, synchronized by shared `run_id`

## Data Collection

35 workloads run once each with simultaneous NVML + Tapo P115 wall-power logging:
- 1 idle baseline (120s)
- 7 ML training (ResNet-18/50, MLP, GPT-2, BERT; FP32 and AMP variants)
- 1 ML inference (ResNet-50 batch inference, 600s)
- 2 scientific HPC (cuFFT 3D, N-body simulation)
- 1 crypto mining proxy (Ethash-like)
- 1 rendering proxy (Monte Carlo path tracing)
- 7 adversarial Round 1 (strategies A-G)
- 4 adversarial Round 2 (strategies H-J)
- 5 adversarial Round 3 (strategy K online learning, L diluted at N=2/5/10/20)
- 6 adversarial composite idle-fraction sweep (0-95%)

Total: 5,721 paired per-second GPU + wall-power samples.

## GPU Power vs Wall Power Correlation

| Metric | Value |
|--------|-------|
| Pearson r | 0.83 |
| R² | 0.68 |
| Linear fit | wall = 1.229 * gpu + 64.8 W |
| RMSE | 64.3 W |
| PSU efficiency (1/slope) | ~81% |
| System base load (intercept) | ~65 W |

The moderate R² (0.68 rather than >0.9) reflects that system overhead is **not constant** — it varies significantly with workload type, primarily due to CPU utilization differences.

## Per-Workload Power Breakdown

| Workload | GPU (W) | Wall (W) | Overhead (W) | Category |
|----------|---------|----------|--------------|----------|
| idle | 21.5 | 48.2 | +26.7 | Idle |
| pytorch_mlp_cifar10 | 28.9 | 96.1 | +67.2 | Training |
| mining_ethash_proxy | 59.0 | 131.4 | +72.4 | Mining |
| adversarial_L_diluted_20 | 79.2 | 159.5 | +80.2 | Adversarial |
| bert_sst2_amp | 109.5 | 182.6 | +73.2 | Training |
| adversarial_L_diluted_2 | 127.0 | 258.6 | +131.6 | Adversarial |
| pytorch_resnet_cifar10_amp | 129.1 | 200.5 | +71.4 | Training |
| rendering_proxy | 139.6 | 212.1 | +72.5 | Rendering |
| pytorch_resnet_cifar10 | 141.9 | 211.8 | +69.9 | Training |
| resnet50_inference | 146.2 | 227.1 | +80.8 | Inference |
| bert_sst2 | 148.4 | 219.9 | +71.4 | Training |
| gpt2_wikitext2_amp | 164.9 | 234.7 | +69.8 | Training |
| gpt2_wikitext2 | 190.4 | 260.7 | +70.3 | Training |
| adversarial_L_diluted_5 | 192.6 | 387.7 | +195.0 | Adversarial |
| adversarial_L_diluted_10 | 199.7 | 398.8 | +199.0 | Adversarial |
| nbody_sim | 209.0 | 279.1 | +70.1 | HPC |
| cufft_benchmark | 214.5 | 284.3 | +69.8 | HPC |
| adversarial_K_online_learning | 231.0 | 449.0 | +218.0 | Adversarial |

## Key Finding: Anomalous System Overhead on Adversarial Workloads

System overhead (wall power minus GPU power) falls into two distinct regimes:

**Normal workloads (training, inference, HPC, mining, rendering): 68-81 W overhead.**
This is consistent across all non-adversarial workloads regardless of GPU power level, reflecting a stable base system draw (CPU + RAM + PSU losses + fans).

**Adversarial mode-switching workloads (K, L_diluted at N=2/5/10): 132-218 W overhead.**
These workloads show 2-3x the normal system overhead. The anomaly is specific to strategies that rapidly switch between `model.train()` and `model.eval()` modes and/or perform backward passes interleaved with inference.

| Workload | Training fraction | Overhead (W) | Overhead vs normal |
|----------|-------------------|--------------|-------------------|
| L_diluted_20 | 5% | 80 W | 1.0x (normal) |
| L_diluted_2 | 50% | 132 W | 1.7x |
| L_diluted_5 | 20% | 195 W | 2.6x |
| L_diluted_10 | 10% | 199 W | 2.6x |
| K_online_learning | 100% | 218 W | 2.9x |

Note: L_diluted_20 (5% training) falls within the normal range, while L_diluted_5 and L_diluted_10 (10-20% training) show the highest anomaly. The non-monotonic pattern (L_diluted_2 at 50% has *lower* overhead than L_diluted_5/10) likely reflects different batch sizes and mode-switching frequencies.

### Why this happens

The adversarial mode-switching strategies create heavy CPU-side overhead that GPU telemetry alone does not capture:

1. **`torch.cuda.empty_cache()` calls** after each training batch force GPU memory compaction, which is CPU-coordinated
2. **`model.train()` / `model.eval()` switching** triggers batchnorm state changes and autograd context setup
3. **Gradient computation and optimizer steps** create CPU-side memory management overhead (allocation, deallocation of gradient tensors)
4. **Data pipeline contention** — the backward pass delays data prefetching, causing CPU utilization spikes when the next batch is prepared

### Implications for detection

This is a **new detection signal**: even an adversary who perfectly matches GPU-reported power draw to a benign workload would still leak information through system-level power consumption. Wall-power measurement captures CPU-side overhead from training that is invisible to GPU telemetry.

In a datacenter setting, this suggests that rack-level power monitoring (already standard for capacity planning) could serve as an independent detection channel. The anomalous overhead is large enough (2-3x normal) that it would be detectable even with relatively coarse power monitoring.

### Limitations

- Only one GPU architecture tested (RTX 5080 Blackwell)
- Desktop system with consumer-grade PSU; datacenter PSU efficiency profile may differ
- Tapo P115 reports power at ~1 Hz with limited precision (mW resolution)
- CPU load not independently measured (would need per-core power counters to decompose overhead)
- The non-monotonic overhead pattern (L_diluted_5 > L_diluted_2) warrants further investigation

## Files

- `data/wall_power/` — 35 Tapo P115 wall-power parquets
- `data/*_5080_*_202604{11,12}_*.parquet` — matched NVML telemetry
- `classifier/results/wall_power_correlation.csv` — per-workload GPU vs wall power summary
- `classifier/results/wall_power_fit.csv` — linear fit parameters and correlation stats
- `classifier/energy_bound.py` — analysis code
