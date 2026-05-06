# AMD / ROCm port — design notes

This document covers the AMD GPU data-collection branch. It describes (1)
the telemetry-field cross-walk between NVIDIA NVML/DCGM and AMD
amdsmi/RDC, (2) which workloads are portable as-is and which needed
replacement, and (3) the known limitations relative to the NVIDIA pipeline.

## Files in this branch

| File | Purpose |
|---|---|
| [scripts/collect_amdsmi_telemetry.py](../scripts/collect_amdsmi_telemetry.py) | Main collector. Library API + CLI. Mirrors `collect_telemetry.py`. |
| [scripts/collect_amdsmi_simple.py](../scripts/collect_amdsmi_simple.py) | Lightweight standalone collector. Mirrors `collect_nvml_telemetry.py`. |
| [scripts/amdsmi_smoke_test.py](../scripts/amdsmi_smoke_test.py) | One-shot sanity check. Mirrors `nvml_smoke_test.py`. |
| [scripts/run_workload_amd.py](../scripts/run_workload_amd.py) | Workload runner with telemetry orchestration. Mirrors `run_workload.py`. |
| [workloads/ffmpeg_vaapi.py](../workloads/ffmpeg_vaapi.py) | VA-API video-encode workload. Replaces `ffmpeg_nvenc.py`. |
| [workloads/registry_amd.py](../workloads/registry_amd.py) | AMD workload registry (subset of `registry.py`, batch sizes for 16 GB). |
| [requirements_amd.txt](../requirements_amd.txt) | Python dependency list. |

## Tier 1 field cross-walk (NVML -> amdsmi)

The collector emits the same column names as the NVIDIA collector wherever
the semantic equivalent exists, so downstream analysis code does not need
a separate code path per vendor.

| NVIDIA column | AMD source | Notes |
|---|---|---|
| `gpu_utilization_pct` | `amdsmi_get_gpu_activity().gfx_activity` | Graphics engine busy %. Direct analogue. |
| `mem_utilization_pct` | `amdsmi_get_gpu_activity().umc_activity` | **Semantic difference**: AMD reports memory-controller busy %; NVML reports % of time memory was being read/written. Trends correlate; absolute values do not. |
| `mem_used_mb` / `mem_total_mb` | `amdsmi_get_gpu_memory_usage(VRAM)` / `_total` | Identical. |
| `power_draw_w` | `amdsmi_get_power_info().current_socket_power` | Falls back to `average_socket_power` if current is unavailable. |
| `temperature_c` | `amdsmi_get_temp_metric(EDGE, CURRENT)` | Edge sensor — closest analogue to NVML's GPU temp. |
| `sm_clock_mhz` | `amdsmi_get_clock_info(GFX).clk` | Renamed only for cross-vendor schema compatibility; AMD has no SMs. |
| `mem_clock_mhz` | `amdsmi_get_clock_info(MEM).clk` | Identical. |
| `pcie_tx_mbps` / `pcie_rx_mbps` | `amdsmi_get_pcie_info().pcie_metric.{pcie_tx,pcie_rx}` | Available in newer ROCm; -1 if unsupported. |
| `encoder_util_pct` | `amdsmi_get_gpu_activity().mm_activity` | **Combined VCN encode+decode** — see "Known divergences" below. |
| `decoder_util_pct` | `amdsmi_get_gpu_activity().mm_activity` | Same value; `mm_combined=True` flag is set. |
| `fan_speed_pct` | `amdsmi_get_gpu_fan_speed` / `_max` | RPM converted to % via the per-GPU max-RPM reading. |

### AMD-only extension columns

These are added on top of the cross-vendor schema:

| Column | Source | Why useful |
|---|---|---|
| `temperature_hotspot_c` | `amdsmi_get_temp_metric(HOTSPOT)` | Junction temp; throttle threshold is on this sensor, not edge. |
| `temperature_vram_c` | `amdsmi_get_temp_metric(VRAM)` | Memory-die temp; dominant signal under heavy bandwidth load. |
| `power_limit_w` | `amdsmi_get_power_info().power_limit` | Helps detect Strategy G analogues (power-capped training). |
| `throttle_status` | `amdsmi_get_gpu_metrics_info().throttle_status` | Bitfield indicating active throttle reasons. |
| `mm_combined` | constant `True` | Flags that the encoder/decoder columns are not separable on this vendor. |

## Tier 2 field cross-walk (DCGM -> RDC)

Tier 2 on AMD is best-effort; RDNA 4 consumer cards typically expose only
the basic activity/clock/power fields, while CDNA datacenter cards
(MI200/MI300) may expose profiling counters analogous to DCGM's tensor /
pipe-active fields. The collector tries both and falls back gracefully.

| DCGM field (NVIDIA) | RDC field (AMD) | Notes |
|---|---|---|
| `dcgm_gpu_util` | `RDC_FI_GPU_BUSY_PERCENT` | Basic, available everywhere. |
| `dcgm_power_usage` | `RDC_FI_POWER_USAGE` | Basic. |
| `dcgm_gpu_temp` | `RDC_FI_GPU_TEMP` | Basic. |
| `dcgm_sm_clock` | `RDC_FI_GPU_CLOCK` | Basic. |
| `dcgm_mem_clock` | `RDC_FI_MEM_CLOCK` | Basic. |
| `dcgm_gr_engine_active` | `RDC_FI_PROF_GUI_UTIL` | Profiling — CDNA usually, RDNA sometimes. |
| `dcgm_sm_occupancy` | `RDC_FI_PROF_OCCUPANCY_PERCENT` | Profiling. |
| `dcgm_fp32_pipe_active` | `RDC_FI_PROF_VALU_PIPE_ISSUE_UTIL` | Profiling. Maps SM FP32 pipe -> AMD VALU pipe. |
| `dcgm_tensor_active` | `RDC_FI_PROF_TENSOR_ACTIVE_PERCENT` | Profiling. AMD MFMA / WMMA equivalent; may not exist on RDNA. |

The RDC collector reports which fields it actually obtained on first
sample (logged at INFO level). Expect a thinner Tier 2 column set on
RDNA 4 than on H100 / A100.

## Workload portability

Most existing workloads in `workloads/` are pure PyTorch and run on ROCm
unchanged because PyTorch ROCm aliases the `torch.cuda` namespace.
`torch.cuda.synchronize()`, `device("cuda")`, `tensor.cuda()`, etc. all
operate on the AMD GPU.

| Workload | Portable as-is? | Notes |
|---|---|---|
| `idle.py` | Yes | No GPU calls. |
| `pytorch_training.py` | Yes | Standard PyTorch. |
| `gpt2_finetune.py`, `bert_finetune.py` | Yes | HuggingFace + PyTorch. |
| `inference.py`, `resnet50_inference.py` | Yes | PyTorch. |
| `cufft_benchmark.py`, `scientific_hpc.py` | Yes | `torch.fft` -> rocFFT. |
| `nbody_sim.py` | Yes | Pure tensor ops. |
| `mining_proxy.py`, `ethash_cuda.py` | Yes | Despite the names, these are pure PyTorch — no CUDA C++. |
| `rendering_proxy.py` | Yes | Pure tensor ops. |
| `modern_inference.py`, `llm_inference.py` | Yes (consumer-tier models) | Restricted in `registry_amd.py` to models that fit 16 GB. |
| `llm_inference_vllm.py` | Conditional | vLLM has a ROCm fork. Not in the AMD registry by default — add if your ROCm/vLLM combination is known-good. |
| `lora_training.py`, `adversarial_*.py` | Yes (single-GPU subset) | Multi-GPU DDP/FSDP entries removed; consumer GPU is single-device. |
| `ffmpeg_nvenc.py` | **No** | Replaced by `ffmpeg_vaapi.py`. |
| `blender_render.py` | **No** | Blender Cycles' HIP backend exists but is brittle on RDNA 4; treated as a separate piece of work, not in this branch. |
| `gromacs_sim.py` | **No** | Requires a ROCm-built GROMACS; not bundled. |

## Known divergences from the NVIDIA pipeline

1. **VCN combined encode/decode.** AMD reports a single multimedia engine
   activity value covering both encode and decode. The two NVENC vs.
   NVDEC columns become a single column with a flag. Workloads that the
   NVIDIA pipeline used to distinguish encode-heavy from decode-heavy
   patterns will need a different feature on AMD (e.g., correlate
   `mm_activity` with the workload label rather than the column name).
2. **`mem_utilization_pct` semantics.** AMD's `umc_activity` is
   memory-controller busy %; NVML's memory-util is % of cycles where
   memory was read/written. Trends agree; absolute values are not
   directly comparable.
3. **Tier 2 thinness on RDNA.** Profiling counters that DCGM exposes on
   datacenter NVIDIA cards (tensor-active, fp32-pipe-active, sm-occupancy)
   are not reliably available on RDNA 4 via RDC. The classifier's Tier 2
   feature surface on AMD will be smaller. This is itself worth reporting
   as a finding.
4. **No NVENC-vs-NVDEC separation in features.** See (1).
5. **No NVLink / NCCL telemetry.** RDC has equivalent fields for Infinity
   Fabric, but only on multi-GPU CDNA systems. Single-GPU consumer setups
   produce no inter-GPU traffic, so this is moot for the local box.
6. **PCIe throughput field availability.** `amdsmi_get_pcie_info()`
   returns tx/rx bytes only on recent ROCm builds. The collector returns
   -1 if unavailable rather than failing.

## Verification sequence

After installing ROCm and the Python deps:

```bash
# 1. Hardware visible to ROCm
rocminfo | grep -E "Name:.*gfx|Marketing Name"
rocm-smi
amd-smi static       # newer unified CLI

# 2. amdsmi Python bindings work
python scripts/amdsmi_smoke_test.py

# 3. Tier 1 collector saves a parquet with sane values
python scripts/collect_amdsmi_telemetry.py --duration 5 --output /tmp/amd_test.parquet --no-rdc
python -c "import pandas as pd; df=pd.read_parquet('/tmp/amd_test.parquet'); print(df.head()); print(df.dtypes)"

# 4. Workload runner end-to-end (idle baseline)
python scripts/run_workload_amd.py --workload idle

# 5. First real workload — ResNet on ROCm
python scripts/run_workload_amd.py --workload pytorch_resnet_cifar10
```

If step 3 shows `gpu_utilization_pct = 0` while a `pytorch_training.py`
process is running, the GPU is not being used — most likely PyTorch was
not built with ROCm support. Reinstall via the ROCm wheel index at
[https://pytorch.org/](https://pytorch.org/).
