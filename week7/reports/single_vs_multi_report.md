# Week 7 -- Single-GPU vs Multi-GPU Matched Experiment

## Configuration

| Parameter | Value |
|-----------|-------|
| Model | ResNet-18-like (n_ch=128) |
| Parameters | 10,797,706 |
| Batch size | 64 per GPU |
| Epochs | 5 |
| Steps/epoch | 1000 |
| AMP | Enabled |
| Optimizer | SGD (lr=0.1, cosine) |
| Hardware | NVIDIA B200 (Blackwell) |

## Results

| Metric | 1 GPU | 2 GPU (DP) | Speedup |
|--------|-------|------------|---------|
| Throughput (img/s) | 19578 | 12319 | 0.63x |
| Step time (ms) | 3.27 | 10.39 | - |
| GPU util (%) | 66 | 29 / 27 | - |
| Power (W) | 421 | 326 / 320 | - |

## Analysis

The 2-GPU DataParallel configuration achieves **0.63x** throughput over single GPU. 
Sub-linear scaling indicates significant DP overhead. DDP (via torchrun) would improve this.

## Telemetry Differences

Key observable differences between single-GPU and multi-GPU training:
- **GPU utilization:** 2-GPU shows lower per-GPU util due to batch splitting
- **Power draw:** second GPU draws significant power during DP even as replica
- **Memory usage:** both GPUs hold model copies in DP mode
- **SM clock:** both GPUs boost during active computation phases
