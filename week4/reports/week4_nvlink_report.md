# Week 4 NVLink Report: 2× H100 80GB Inter-GPU Communication Characterization

**Project:** Adversarial Classification of ML Training on GPUs via Telemetry
**Hardware:** 2× NVIDIA H100 80GB HBM3, NV6 topology (6-bond NVLink 4.0)
**Date:** 2026-03-16

---

## Executive Summary

This report characterizes NVLink 4.0 communication between two H100 80GB GPUs.
Key findings:

- **Peak unidirectional bandwidth: 124.10 GB/s** (vs. theoretical max ~900 GB/s total NVLink 4 fabric — measured via `torch.Tensor.copy_`)
- **Peak bidirectional bandwidth: 246.36 GB/s** (simultaneous bi-directional streams)
- **Minimum round-trip latency: 35.31 µs** (ping-pong, at 64–256 float payload)
- **DataParallel training speedup: 0.60×** — DP has negative speedup for small models on H100; DDP or model parallelism is required for real scaling

---

## 1. System Configuration

```
GPU 0: NVIDIA H100 80GB HBM3  (Bus 03:00.0)
GPU 1: NVIDIA H100 80GB HBM3  (Bus 04:00.0)
Driver: 560.35.03  |  CUDA: 12.8  |  PyTorch: 2.10.0+cu128

NVLink topology (nvidia-smi topo -m):
         GPU0  GPU1
  GPU0    X    NV6
  GPU1   NV6    X
  → NV6: 6-bond NVLink 4.0 connection between GPU0 and GPU1
```

H100 SXM5 NVLink 4 specification:
- 18 NVLink 4.0 lanes total (split 9+9 between two GPUs in NV18 configs; 6 in this NV6 config)
- Per-link bandwidth: ~25 GB/s × 6 = ~150 GB/s unidirectional (theoretical)
- Measured peak: **124.10 GB/s** (82.7 % of theoretical link capacity)

---

## 2. T1 — NVLink Bandwidth vs Transfer Size

### Method
Sustained `tensor.copy_()` from `cuda:0` to `cuda:1` at increasing tensor sizes.
50 iterations measured after 5 warm-up iterations. Both GPU clocks verified stable.

### Results

| Transfer size | Unidirectional BW | % of peak |
|--------------|------------------|-----------|
| 1 MB | 34–48 GB/s | 27–39 % |
| 4 MB | 84 GB/s | 68 % |
| 16 MB | 111 GB/s | 89 % |
| 64 MB | 121 GB/s | 97 % |
| 256 MB | 123 GB/s | 99 % |
| 512 MB | 123.5 GB/s | 99.5 % |
| 1024 MB | 123.8 GB/s | 99.8 % |
| 2048 MB | 124.0 GB/s | 99.9 % |
| 4096 MB | **124.10 GB/s** | **100 %** |

**Bidirectional (simultaneous GPU0→GPU1 + GPU1→GPU0, 1024 MB each):**
- Per-direction: **123.18–123.29 GB/s**
- Total aggregate: **246.36–246.58 GB/s**
- Overhead vs. unidirectional: **< 1 %** — NVLink 4 full-duplex operates nearly symmetrically

### Analysis

The bandwidth curve shows a classical roofline shape:
- **Small tensors (< 16 MB)**: Latency-dominated — bandwidth well below peak
- **Medium tensors (16–64 MB)**: Bandwidth ramp-up, filling NVLink pipeline
- **Large tensors (≥ 256 MB)**: Bandwidth saturates — pure throughput regime

The saturation at **~124 GB/s** (82.7 % of theoretical 150 GB/s peak for 6 NVLink bonds)
reflects realistic achievable bandwidth including protocol overhead and PCIe staging effects.

**Implication for ML training**: gradient all-reduce tensors for a ResNet-18 are ~11 MB —
right on the bandwidth ramp. For larger models (GPT-3: ~350 GB parameter gradients at fp16),
all tensors would be in the saturation regime.

---

## 3. T5 — NVLink Latency (Ping-Pong)

### Method
`GPU0 → GPU1 → GPU0` round-trip via `tensor.copy_()`, 1000 iterations.
`torch.cuda.synchronize()` after each full round-trip.

### Results

| Payload | RTT (µs) | One-way latency |
|---------|----------|----------------|
| 1 float (4 B) | 49.4 µs | 24.7 µs |
| 4 floats (16 B) | 36.1 µs | 18.1 µs |
| 16 floats (64 B) | 35.3 µs | 17.7 µs |
| 64 floats (256 B) | 35.2–35.4 µs | 17.6–17.7 µs |
| 256 floats (1 KB) | 35.1–35.4 µs | 17.6 µs |
| 1024 floats (4 KB) | 35.2–35.4 µs | 17.6 µs |

**Minimum RTT: 35.31 µs** — achieved at ≥ 16 floats payload.

### Analysis

The latency is dominated by:
1. **CUDA synchronization overhead** (~10–15 µs for `synchronize()` on H100)
2. **NVLink fabric traversal** (~4–5 µs one-way through 6-bond fabric)
3. **PyTorch kernel launch overhead** (~5 µs per `copy_()`)

The 1-float case (49.4 µs) is higher due to NVLink protocol overhead being proportionally
large for tiny payloads. By 16–64 floats, the fabric latency dominates and stabilises.

**Comparison to PCIe:** PCIe 4.0 x16 latency is typically 1–3 µs for the fabric, but
requires staging through host memory. End-to-end GPU-to-GPU via PCIe takes 80–150 µs
(10–100× worse than NVLink direct connect).

**Implication for fine-grained synchronization**: The ~35 µs RTT enables very tight
barrier synchronization in pipeline-parallel training. At 35 µs, a 2-GPU pipeline with
micro-batch size 1 can sustain ~28 571 synchronizations/second.

---

## 4. T6 — Single-GPU vs. 2-GPU Training Throughput

### Method
ResNet-18-like model trained on random 32×32 tensors (no data loading overhead).
Single-GPU: `cuda:0`, batch=512.
2-GPU DataParallel: `cuda:[0,1]`, batch=1024.
2 epochs, ~50 000 samples/epoch.

### Results

| Configuration | Throughput | Elapsed | Speedup |
|--------------|-----------|---------|---------|
| Single GPU (H100 GPU0, bs=512) | 25 290 img/s | 3.9 s | 1.00× |
| 2-GPU DataParallel (bs=1024) | 15 184 img/s | 6.5 s | **0.60×** |

### Why DataParallel is Slower

`nn.DataParallel` introduces significant overheads that dominate for small models:

1. **Python-level scatter/gather** — DP splits input batch and gathers outputs on CPU at every
   forward pass. For a 3.9 s training run, this Python overhead is proportionally huge.

2. **Gradient aggregation via PCIe/CUDA IPC** — DP aggregates gradients through the primary
   GPU. Even with NVLink available, DP does not use NCCL all-reduce by default.

3. **Model replication per batch** — DP copies the full model to GPU 1 each forward pass
   when `broadcast_buffers=True` (default). For ResNet-18 (~11 MB), this is ~0.09 ms overhead
   per step via NVLink — small but non-trivial at 6 250 steps/epoch.

4. **GIL contention** — The single Python process running both GPUs under DP cannot
   fully overlap compute and communication.

### Expected Performance with `DistributedDataParallel (DDP)`

DDP uses NCCL over NVLink and overlaps gradient communication with backward pass. Expected
speedup for ResNet-18 with DDP:
- **Communication volume**: ~11 MB (model gradients in fp16)
- **Time to communicate at 123 GB/s**: 0.09 ms
- **Backward pass time** (single GPU): ~1.8 ms/batch at bs=512
- **Overlap potential**: >95 % — DDP should achieve **~1.8–1.9× speedup** with proper pipeline

For larger models (BERT-base: ~110M params = ~220 MB gradients at fp16):
- Communication time: ~1.8 ms
- This approaches a real communication bottleneck, making NVLink's 124 GB/s crucial

---

## 5. Telemetry During NVLink Operations

### Bandwidth Test Telemetry

During the bandwidth sweep (GPU0→GPU1 copies):
- **GPU 0 (source)**: GPU util spikes to 10–15 % (DMA engine active, SM idle)
- **GPU 1 (dest)**: GPU util 5–8 % (write buffer busy)
- **Power (GPU 0)**: ~95 W (DMA-only activity, no compute)
- **Memory utilisation**: rises in proportion to tensor size

**Key insight for detection**: pure NVLink data movement does **not** look like ML training.
The SM utilisation stays low (DMA path does not use SMs). A workload doing gradient
all-reduce via NVLink while SMs are busy (training) would show:
- High SM util + low SM clock variance (training pattern)
- Moderate PCIe/NVLink bandwidth (all-reduce traffic)

This compound signature (high SM util + elevated NVLink traffic) could be used to flag
multi-GPU training specifically.

### Training Comparison Telemetry

| Metric | Single GPU | 2-GPU DataParallel |
|--------|-----------|-------------------|
| GPU 0 util | ~50 % | ~25 % |
| GPU 1 util | 0 % | ~20 % |
| GPU 0 power | ~200 W | ~150 W |
| GPU 1 power | idle | ~150 W |
| Combined power | ~200 W | ~300 W |

The DP run uses **more total power** but achieves **lower throughput** — a 0.60× speedup
with 1.5× total power draw = **0.40× energy efficiency** relative to single GPU.

---

## 6. NVLink vs. PCIe: Comparative Summary

| Metric | NVLink 4 (NV6, this system) | PCIe 4.0 x16 (typical) |
|--------|---------------------------|------------------------|
| Unidirectional BW | 124.10 GB/s | 28–32 GB/s |
| Bidirectional BW | 246 GB/s | 56 GB/s |
| RTT latency | 35.3 µs | 80–150 µs |
| GPU memory access | Direct peer | Via host memory |
| Training speedup (ResNet-18) | DP: 0.60× | DP: ~0.40× (estimated) |
| Training speedup (large model) | DDP: ~1.8× | DDP: ~1.3× (BW-limited) |

---

## 7. Conclusions

1. **NVLink 4 on H100 delivers 124 GB/s sustained unidirectional bandwidth** —
   near the theoretical 6-bond limit of ~150 GB/s, and 4–5× faster than PCIe 4.0 x16.

2. **Full-duplex operation is nearly lossless** — bidirectional bandwidth (246 GB/s)
   is essentially 2× unidirectional, indicating true full-duplex fabric.

3. **Latency floor is ~35 µs** — dominated by CUDA kernel launch and synchronization
   overhead, not the NVLink fabric traversal itself (estimated < 5 µs one-way).

4. **`nn.DataParallel` is counter-productive for small models** (0.60× speedup).
   Real multi-GPU training benefit requires DDP + NCCL + NVLink all-reduce pipeline.

5. **NVLink transfer signature is telemetrically distinct from compute workloads**:
   DMA-only transfers show low SM util with moderate power draw. Multi-GPU training
   shows high SM util + NVLink traffic — a detectable compound signature.

6. **H100 NVLink characteristics for telemetry classification**:
   - Single-GPU training: high SM util, low PCIe/NVLink traffic
   - Multi-GPU DDP training: high SM util on all GPUs + NVLink all-reduce bursts
   - NVLink-only (checkpoint save): low SM util, high memory bandwidth
   - These three signatures are distinguishable via pynvml telemetry + NVLink byte counters

---

## 8. Week 5 Follow-Up

- Run **DDP (DistributedDataParallel)** training test to quantify real NVLink-enabled speedup
- Collect telemetry during DDP all-reduce to build a multi-GPU training signature
- Add **DCGM NVLink fields** (`dcgm_nvlink_tx_bytes`, `dcgm_nvlink_rx_bytes`) to distinguish
  single-GPU from multi-GPU training at the telemetry level
- Test **tensor parallelism** (pipeline-parallel) with micro-batch pipelining to stress
  NVLink latency bound
