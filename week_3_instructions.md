# Week 3: Data Collection Instructions

## Overview

Week 3 is the main data collection campaign. The goal is **30+ labeled workload runs** on a cloud A100 covering all workload categories, with Tier 1 + Tier 2 metrics for every run and Tier 3 (Nsight Compute) profiles for at least 1 run per workload type.

### What was completed in Week 2 (on RTX 5080)

- Telemetry pipeline built and validated (`scripts/collect_telemetry.py`, `scripts/run_workload.py`)
- Tier 1 (pynvml) collection working: 15 metrics at 1 Hz
- Tier 3 NCU script written and ready (`scripts/collect_ncu_metrics.py`)
- 12 Tier 1 runs collected on RTX 5080 (3 runs each: idle, ResNet-18 FP32, ResNet-18 AMP, MLP)
- DCGM installed and tested; profiling fields confirmed to require data-center GPUs

### What RTX 5080 could NOT do (and why you need A100)

| Feature | RTX 5080 limitation | A100 capability |
|---------|---------------------|-----------------|
| DCGM profiling fields (Tier 2) | `DCGMError_ModuleNotLoaded` — profiling module only loads on data-center GPUs | Full support: tensor_active, fp16/32/64 pipes, SM occupancy, DRAM bandwidth |
| Nsight Compute (Tier 3) | `ERR_NVGPUCTRPERM` — host kernel has `RmProfilingAdminOnly=1`, can't change from Docker | Works on bare-metal hosts with profiling enabled |
| VRAM | 16 GB — limits larger model training | 40/80 GB — can run GPT-2 fine-tuning, larger batch sizes |

---

## Step 1: Select a Vast.ai A100 Instance

### Requirements for full Tier 1 + 2 + 3 collection

You need a **bare-metal A100** host on Vast.ai with these properties:

1. **GPU**: NVIDIA A100 (40GB or 80GB SXM or PCIe)
2. **Bare-metal / PCIe passthrough**: Required for Nsight Compute and DCGM profiling
3. **Docker image**: Use `nvidia/cuda:12.4.1-devel-ubuntu22.04` or similar with CUDA dev tools
4. **Disk**: At least 50 GB (for datasets, profiling reports, and Parquet output)
5. **CPU/RAM**: At least 8 vCPUs, 32 GB RAM (for data loading, GROMACS, etc.)

### How to find NCU-capable hosts

On Vast.ai, not all hosts allow Nsight Compute profiling. To find ones that do:

1. Go to https://cloud.vast.ai/create/ and filter for A100 GPUs
2. Sort by price (cheapest first)
3. **Before committing to a long rental**, rent for 1 hour and run this test:

```bash
# Quick NCU capability test (run AFTER setup.sh)
python3 -c "import torch; x=torch.randn(256,256,device='cuda'); y=torch.mm(x,x); print('CUDA OK')"
ncu --set basic python3 -c "import torch; x=torch.randn(256,256,device='cuda'); y=torch.mm(x,x)"
```

If NCU succeeds (prints kernel profiling output), this host supports Tier 3. If you see `ERR_NVGPUCTRPERM`, try a different host.

**Indicators of NCU-capable hosts:**
- Listed as "bare-metal" or "PCIe passthrough" (not virtualized)
- Host has NVIDIA driver 535+ with `RmProfilingAdminOnly=0`
- Some hosts explicitly list "Nsight compatible" in their description

### How to check `RmProfilingAdminOnly` from inside the container

```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
# Need: RmProfilingAdminOnly: 0
# If it shows 1, NCU will NOT work on this host
```

### Fallback if no NCU-capable host is found

If you cannot find an A100 host that allows NCU:
- Collect Tier 1 + Tier 2 on any A100 host (DCGM profiling fields work on all A100s)
- Defer Tier 3 NCU collection to a later week when a compatible host is available
- Document the host selection process in the report

### RunPod as backup

RunPod Secure Cloud ($1.14-1.39/hr A100) is more reliable for DCGM but:
- May also restrict NCU
- Blocks mining software (cannot profile crypto mining workloads)
- Use for Tier 1 + 2 collection if Vast.ai hosts are unreliable

---

## Step 2: Instance Setup

Once you have an A100 instance, run:

```bash
# Clone the repo
git clone https://github.com/robirahman/SPAR-GPU-monitoring.git
cd SPAR-GPU-monitoring

# Run the automated setup
chmod +x setup.sh
sudo ./setup.sh

# Activate the Python environment
source .venv/bin/activate

# Verify all tiers
# Tier 1
python3 scripts/collect_telemetry.py --duration 5 --output /tmp/test.parquet
python3 -c "import pandas as pd; df=pd.read_parquet('/tmp/test.parquet'); print(f'Tier 1 OK: {len(df.columns)} columns')"

# Tier 2 (DCGM profiling fields — should work on A100)
python3 scripts/collect_telemetry.py --duration 5 --output /tmp/test_dcgm.parquet
python3 -c "
import pandas as pd
df = pd.read_parquet('/tmp/test_dcgm.parquet')
dcgm_cols = [c for c in df.columns if c.startswith('dcgm_')]
print(f'DCGM columns: {len(dcgm_cols)}')
print(dcgm_cols)
# Should see: dcgm_tensor_active, dcgm_fp16_pipe_active, etc.
# If you only see dcgm_gpu_util/dcgm_power_usage/etc., profiling fields are not available
"

# Tier 3 (NCU — may or may not work depending on host)
ncu --set basic python3 -c "import torch; x=torch.randn(256,256,device='cuda'); y=torch.mm(x,x)"
```

### Expected DCGM profiling columns on A100

When DCGM profiling fields work, you should see these additional columns in the Parquet output:

| Column | Meaning | Range |
|--------|---------|-------|
| `dcgm_gr_engine_active` | Graphics engine active ratio | 0.0–1.0 |
| `dcgm_sm_active` | SM active ratio | 0.0–1.0 |
| `dcgm_sm_occupancy` | SM occupancy ratio | 0.0–1.0 |
| `dcgm_tensor_active` | Tensor core active ratio | 0.0–1.0 |
| `dcgm_dram_active` | DRAM active ratio | 0.0–1.0 |
| `dcgm_fp64_pipe_active` | FP64 pipe utilization | 0.0–1.0 |
| `dcgm_fp32_pipe_active` | FP32 pipe utilization | 0.0–1.0 |
| `dcgm_fp16_pipe_active` | FP16 pipe utilization | 0.0–1.0 |

These are the key metrics for distinguishing ML training from other workloads. `dcgm_tensor_active` will be high during mixed-precision training and near zero for scientific HPC/mining. `dcgm_fp16_pipe_active` vs `dcgm_fp32_pipe_active` distinguishes AMP from FP32 training.

---

## Step 3: Workloads to Add Before Collection

Before running the full collection, add the remaining Week 3 workloads to `workloads/registry.py`. The following workloads need to be implemented:

### Already implemented (from Week 2)

- `idle` — GPU idle baseline
- `pytorch_resnet_cifar10` — ResNet-18, CIFAR-10, FP32
- `pytorch_resnet_cifar10_amp` — ResNet-18, CIFAR-10, AMP
- `pytorch_mlp_cifar10` — Simple MLP, CIFAR-10, FP32

### ML Training (add to workloads/)

- **GPT-2 124M fine-tuning**: Fine-tune GPT-2 small on a text dataset (e.g., WikiText-2). Use Hugging Face Transformers. This is critical — it's the primary "large model training" signature.
- **BERT fine-tuning**: Fine-tune BERT-base on a text classification task (e.g., SST-2). Use Hugging Face Transformers.

```bash
pip install transformers datasets accelerate
```

### ML Inference (add to workloads/)

- **Batch image classification**: Run ResNet-50 inference on CIFAR-10 test set (no gradient computation)
- **llama.cpp inference** (optional): Run quantized LLaMA-7B generation via llama.cpp. Requires downloading a GGUF model.

### Scientific HPC (add to workloads/)

- **GROMACS MD simulation**: Install GROMACS (`apt-get install gromacs`), run the ADH benchmark or a short lysozyme simulation
- **cuFFT benchmark**: Write a simple CUDA FFT benchmark using `torch.fft` or the cuFFT Python wrapper
- **N-body simulation**: Write a simple gravitational N-body simulation in PyTorch (GPU-accelerated)

### Crypto Mining (add to workloads/)

- **T-Rex miner (benchmark mode)**: Download T-Rex miner, run in benchmark mode (`t-rex -a ethash --benchmark`). Vast.ai allows this on most hosts; RunPod does not.
- **Custom Ethash CUDA kernel** (fallback): If T-Rex is blocked, write a CUDA kernel that mimics Ethash memory-hard hashing patterns.

### Rendering (add to workloads/)

- **Blender Cycles**: Install Blender (`apt-get install blender` or download), render the BMW benchmark scene with CUDA.

```bash
# Example Blender CLI render
blender -b bmw.blend -E CYCLES -o /tmp/render_ -f 1 -- --cycles-device CUDA
```

### Other (add to workloads/)

- **FFmpeg NVENC**: Encode a video using NVIDIA's hardware encoder

```bash
ffmpeg -hwaccel cuda -i input.mp4 -c:v h264_nvenc -preset fast output.mp4
```

---

## Step 4: Run the Full Collection

### Collection protocol

For each workload:
- **Minimum 3 runs** (for statistical robustness)
- **10-15 minutes per run** (except idle: 2 minutes, and short workloads that finish sooner)
- Collect **Tier 1 + Tier 2** on every run (both enabled by default)
- Collect **Tier 3 (NCU)** on **1 run per workload** (adds significant overhead)

### Running the collection

```bash
cd /workspace/SPAR-GPU-monitoring
source .venv/bin/activate

# --- ML Training ---
for run in 1 2 3; do
    python3 scripts/run_workload.py --workload pytorch_resnet_cifar10
    python3 scripts/run_workload.py --workload pytorch_resnet_cifar10_amp
    python3 scripts/run_workload.py --workload pytorch_mlp_cifar10
    # Add GPT-2 and BERT once implemented:
    # python3 scripts/run_workload.py --workload gpt2_wikitext2
    # python3 scripts/run_workload.py --workload bert_sst2
    sleep 10  # cooldown between runs
done

# --- ML Inference ---
for run in 1 2 3; do
    python3 scripts/run_workload.py --workload resnet50_inference
    sleep 10
done

# --- Scientific HPC ---
for run in 1 2 3; do
    python3 scripts/run_workload.py --workload gromacs_adh
    python3 scripts/run_workload.py --workload cufft_benchmark
    python3 scripts/run_workload.py --workload nbody_sim
    sleep 10
done

# --- Crypto Mining ---
for run in 1 2 3; do
    python3 scripts/run_workload.py --workload trex_benchmark
    sleep 10
done

# --- Rendering ---
for run in 1 2 3; do
    python3 scripts/run_workload.py --workload blender_bmw
    sleep 10
done

# --- Other ---
for run in 1 2 3; do
    python3 scripts/run_workload.py --workload idle -- python3 workloads/idle.py --duration 120
    python3 scripts/run_workload.py --workload ffmpeg_nvenc
    sleep 10
done
```

### Tier 3 NCU profiling (1 run per workload)

Run these ONLY on an NCU-capable host. Use short runs to limit overhead:

```bash
# NCU profiling pass (one run each, shorter epochs/durations)
python3 scripts/run_workload.py --workload pytorch_resnet_cifar10 --ncu --timeout 120
python3 scripts/run_workload.py --workload pytorch_resnet_cifar10_amp --ncu --timeout 120
python3 scripts/run_workload.py --workload pytorch_mlp_cifar10 --ncu --timeout 120
# Add other workloads as they're implemented
```

---

## Step 5: Verify Collected Data

After collection, run this verification:

```python
import os
import pandas as pd

data_dir = "data/"
files = sorted([f for f in os.listdir(data_dir) if f.endswith(".parquet")])

print(f"Total files: {len(files)}")
print()

by_workload = {}
for f in files:
    df = pd.read_parquet(os.path.join(data_dir, f))
    label = df["workload_label"].iloc[0]
    by_workload.setdefault(label, []).append({
        "file": f,
        "rows": len(df),
        "duration_s": df.timestamp_epoch.iloc[-1] - df.timestamp_epoch.iloc[0],
        "gpu": df["gpu_name"].iloc[0],
        "dcgm_prof": any(c.startswith("dcgm_tensor") for c in df.columns),
    })

for label, runs in sorted(by_workload.items()):
    print(f"{label}: {len(runs)} runs")
    for r in runs:
        prof = "Tier2-prof" if r["dcgm_prof"] else "Tier1-only"
        print(f"  {r['rows']:4d} rows, {r['duration_s']:5.0f}s, {r['gpu']}, {prof}")
    print()
```

### Checklist

- [ ] At least 30 total runs across all workload categories
- [ ] At least 3 runs per workload type
- [ ] Tier 2 DCGM profiling columns present (dcgm_tensor_active, dcgm_fp16_pipe_active, etc.)
- [ ] At least 1 NCU .ncu-rep + .csv file per workload type (if NCU host available)
- [ ] Data organized as `{workload}_{gpu}_{run_id}_{date}.parquet`
- [ ] ML training workloads show clear signatures: high GPU util, high power, tensor core activity
- [ ] ML inference shows different pattern: lower util, no backward pass, less power
- [ ] Non-ML workloads show distinct profiles: different pipe utilization, no tensor cores

---

## Budget Tracking

| Item | Estimated cost |
|------|---------------|
| A100 instance (Vast.ai, ~$0.70/hr) | ~$5-7 for 8 hours of data collection |
| RunPod backup (if needed) | ~$1-2 for 1 hour test |
| RTX 5080 Week 2 testing | ~$2-3 |
| **Running total** | **~$8-12 of $1,000 budget** |

Most of the budget is preserved for Week 6-8 adversarial experiments, which require longer runs and more diverse configurations.

---

## Key Metrics to Focus On During Collection

From the project plan, focus analytical effort on:

1. **Training vs. inference distinction**: tensor core utilization (high in training, low in inference), memory growth over time (training allocates optimizer states), power draw patterns
2. **Epoch periodicity**: training workloads should show periodic patterns in GPU utilization at epoch boundaries (data loader pauses, validation passes)
3. **Forward/backward pass asymmetry**: training has both forward and backward passes (roughly 2:1 compute ratio); inference has only forward
4. **Memory allocation stability**: training memory grows then stabilizes; inference memory is constant
5. **These temporal patterns are NOT captured by WAVE or Differential Architecture** — they are our novel contribution

---

## Troubleshooting

### DCGM profiling fields show NaN

The host may not support profiling. Check:
```bash
dcgmi dmon -e 1004 -c 5  # Should show tensor core utilization
```
If this returns errors, the DCGM profiling module is not loaded. Try a different A100 host.

### NCU returns ERR_NVGPUCTRPERM

The host's NVIDIA kernel module has profiling restricted:
```bash
cat /proc/driver/nvidia/params | grep RmProfilingAdminOnly
```
If `1`, you cannot use NCU on this host. Try a different Vast.ai host.

### Workload runs out of VRAM

Reduce batch size or use AMP. The A100 40GB should handle all planned workloads. If using A100 80GB, you can increase batch sizes for more realistic training signatures.

### DCGM host engine won't start

```bash
# Kill any existing instance and restart
pkill nv-hostengine
sleep 2
nv-hostengine -d
dcgmi discovery -l  # Should show the GPU
```
