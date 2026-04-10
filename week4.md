# Week 3 H100 Collection Report

**Generated:** 2026-03-11 20:47 UTC  
**GPU:** NVIDIA H100 80GB HBM3  
**Total runs collected:** 98  
**Total telemetry samples:** 22,547  
**Collection tier:** Tier 1 (pynvml, 1 Hz, no DCGM)  

---

## Run Inventory

| Workload | Category | Runs | Avg Duration | Samples/Run |
|----------|----------|------|-------------|-------------|
| `bert_sst2` | ML Training (FP32) | 7 | 34s | 34 |
| `bert_sst2_amp` | ML Training (AMP) | 8 | 26s | 26 |
| `blender_bmw` | Other | 3 | 439s | 440 |
| `cufft_benchmark` | Scientific HPC | 6 | 606s | 607 |
| `ffmpeg_nvenc` | Other | 3 | 5s | 6 |
| `gpt2_wikitext2` | ML Training (FP32) | 12 | 35s | 35 |
| `gpt2_wikitext2_amp` | ML Training (AMP) | 14 | 28s | 28 |
| `idle` | Baseline | 11 | 114s | 115 |
| `mining_ethash_proxy` | Crypto Mining | 6 | 606s | 607 |
| `nbody_sim` | Scientific HPC | 6 | 606s | 607 |
| `pytorch_mlp_cifar10` | ML Training (FP32) | 3 | 56s | 56 |
| `pytorch_resnet_cifar10` | ML Training (FP32) | 4 | 37s | 38 |
| `pytorch_resnet_cifar10_amp` | ML Training (AMP) | 3 | 33s | 33 |
| `rendering_proxy` | Rendering | 6 | 606s | 607 |
| `resnet50_inference` | ML Inference | 6 | 609s | 609 |

## Signal Comparison Table

Mean values across all runs per workload. CV = coefficient of variation (std/mean %).

| Workload | Category | GPU Util % (mean) | GPU Util % CV% | Mem Used MB (mean) | Mem Used MB CV% | Power W (mean) | Power W CV% | Temp °C (mean) | Temp °C CV% | SM Clock MHz (mean) | SM Clock MHz CV% | Mem Clock MHz (mean) | Mem Clock MHz CV% |
|----------|----------| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| `bert_sst2` | ML Training (FP32) | 46.5 | 104.8 | 3166.2 | 71.5 | 330.5 | 64.1 | 40.8 | 21.0 | 1570.2 | 45.1 | 2619.0 | 0.0 |
| `bert_sst2_amp` | ML Training (AMP) | 23.9 | 136.9 | 2345.8 | 79.5 | 178.9 | 52.2 | 37.2 | 22.8 | 1363.4 | 53.3 | 2272.9 | 26.7 |
| `blender_bmw` | Other | 94.4 | 23.6 | 3689.0 | 18.9 | 167.3 | 15.3 | 52.4 | 13.8 | 1387.5 | 10.1 | 1215.0 | 0.0 |
| `cufft_benchmark` | Scientific HPC | 97.8 | 11.0 | 1828.3 | 8.0 | 586.4 | 8.6 | 53.0 | 5.4 | 1967.5 | 7.2 | 2619.0 | 0.0 |
| `ffmpeg_nvenc` | Other | 0.0 | 0.0 | 536.8 | 37.6 | 117.2 | 18.2 | 26.4 | 1.9 | 1071.7 | 78.0 | 2619.0 | 0.0 |
| `gpt2_wikitext2` | ML Training (FP32) | 52.3 | 93.9 | 6464.1 | 77.2 | 397.9 | 63.2 | 47.4 | 23.8 | 1619.7 | 41.8 | 2619.0 | 0.0 |
| `gpt2_wikitext2_amp` | ML Training (AMP) | 38.5 | 120.8 | 5030.7 | 93.2 | 230.8 | 68.4 | 40.6 | 19.2 | 1389.2 | 49.6 | 2239.3 | 27.9 |
| `idle` | Baseline | 0.0 | 0.0 | 449.0 | 0.0 | 100.3 | 0.2 | 26.1 | 1.6 | 345.0 | 0.0 | 2619.0 | 0.0 |
| `mining_ethash_proxy` | Crypto Mining | 52.4 | 11.2 | 2075.3 | 7.9 | 151.9 | 3.0 | 30.8 | 6.9 | 1967.6 | 7.2 | 2619.0 | 0.0 |
| `nbody_sim` | Scientific HPC | 97.8 | 10.8 | 1884.4 | 7.9 | 482.1 | 8.1 | 51.8 | 10.8 | 1967.9 | 7.1 | 2619.0 | 0.0 |
| `pytorch_mlp_cifar10` | ML Training (FP32) | 2.0 | 53.0 | 1188.3 | 25.3 | 143.6 | 10.7 | 28.6 | 2.7 | 1787.6 | 29.6 | 2619.0 | 0.0 |
| `pytorch_resnet_cifar10` | ML Training (FP32) | 49.6 | 94.7 | 3069.8 | 73.6 | 329.1 | 60.6 | 36.7 | 22.5 | 1367.1 | 58.1 | 2619.0 | 0.0 |
| `pytorch_resnet_cifar10_amp` | ML Training (AMP) | 52.7 | 67.4 | 2481.9 | 47.6 | 312.4 | 40.1 | 37.3 | 11.9 | 1657.6 | 39.3 | 2619.0 | 0.0 |
| `rendering_proxy` | Rendering | 83.7 | 11.5 | 1203.5 | 6.6 | 296.0 | 6.8 | 38.6 | 5.7 | 1967.0 | 7.4 | 2619.0 | 0.0 |
| `resnet50_inference` | ML Inference | 24.5 | 66.1 | 4629.0 | 11.3 | 257.0 | 9.7 | 36.2 | 6.6 | 1962.2 | 8.6 | 2619.0 | 0.0 |

## Key Findings

### GPU Utilization Coefficient of Variation (CV)

The most discriminating Tier 1 feature, consistent with Week 3 A100 results:

- **ML Training workloads:** mean GPU util CV = **95.9%**
  (high variability from epoch boundaries, data loading pauses, forward/backward pass asymmetry)
- **Non-training workloads (HPC/mining/rendering/inference):** mean GPU util CV = **19.2%**
  (sustained, stable utilization)

Training CV is **5.0×** higher than non-training on H100.

### SM Clock Stability

- **ML Training:** SM clock std = **682.1 MHz** (frequent transitions between idle and compute)
- **Non-training:** SM clock std = **244.9 MHz** (locked at peak during sustained workloads)

### Power Draw

- Peak ML training power: **398 W** (`gpt2_wikitext2`)
- Mean non-training power: **294 W**

### Binary Classification Rule (Tier 1 Only)

Consistent with A100 Week 3 findings, a simple rule identifies ML training:

```
ML_TRAINING if: gpu_utilization CV > 30% AND sm_clock std > 150 MHz
```

This rule requires zero ML — it is a deterministic threshold on two metrics.

## H100 vs A100 Notes

- **GPU:** NVIDIA H100 80GB HBM3 (vs A100 SXM4 40GB in prior runs)
- **Memory bandwidth:** H100 HBM3 ~3.35 TB/s vs A100 ~2.0 TB/s — memory-bound workloads
  (mining, cufft) may show higher throughput and lower utilization %
- **Tensor Core gen:** 4th gen (H100) vs 3rd gen (A100) — AMP training should run faster
- **DCGM profiling fields** (tensor_active, fp16/32 pipes): not collected (Tier 1 only)
  These would further differentiate training from inference via tensor core activity.

## Data Quality

- Columns with >50% NaN (expected for unavailable hardware counters): `dcgm_gpu_util`, `dcgm_power_usage`, `dcgm_gpu_temp`, `dcgm_sm_clock`, `dcgm_mem_clock`
- Short runs (<10 samples): `bert_sst2` (run 627bf5, 9 samples); `ffmpeg_nvenc` (run 6df45e, 6 samples); `ffmpeg_nvenc` (run 80600b, 6 samples); `ffmpeg_nvenc` (run f64d58, 6 samples); `gpt2_wikitext2` (run 8a8e4e, 5 samples); `gpt2_wikitext2` (run b26db1, 5 samples); `gpt2_wikitext2` (run be8fe3, 9 samples); `gpt2_wikitext2` (run d33f8b, 9 samples); `gpt2_wikitext2` (run f8f1a6, 5 samples); `gpt2_wikitext2_amp` (run 072d22, 5 samples); `gpt2_wikitext2_amp` (run 18ba41, 9 samples); `gpt2_wikitext2_amp` (run 3027c4, 9 samples); `gpt2_wikitext2_amp` (run 6a4f06, 5 samples); `gpt2_wikitext2_amp` (run 6c978c, 5 samples); `gpt2_wikitext2_amp` (run 8a3edd, 9 samples); `pytorch_resnet_cifar10` (run 7e3392, 5 samples)

## Next Steps (Week 4)

1. **Exploratory analysis**: time-series plots, PCA/t-SNE, correlation matrix
2. **Baseline classifier**: train Random Forest on per-run aggregate features (mean, std, CV)
   - Evaluate binary (ML vs non-ML), 3-way, and multi-class tasks
   - Target >85% binary accuracy at Tier 1
3. **Edge-case collection**: vary batch sizes, short runs, DataLoader-bottlenecked training
4. **DCGM Tier 2 test**: verify tensor_active / fp16 pipe fields on RunPod (~$1.50)
5. **Cross-GPU comparison**: compare H100 signatures to A100 Week 3 data to assess
   whether a classifier trained on one GPU generalizes to another
