# Week 8: Laptop RTX 4050 Data Collection Instructions

## Overview

This document covers experiments to run on the **RTX 4050 Laptop GPU** (Ada Lovelace, SM 8.9, 6 GB GDDR6, 60W TDP) during Week 8. The laptop provides Ada Lovelace architecture data — a fourth GPU type for cross-architecture generalization testing alongside cloud A100 (Ampere), cloud H100 (Hopper), and cloud B100/B200 + local RTX 5080 (Blackwell).

**Goal:** ~42 parquet files (14 workloads x 3 reps) of Tier 1 telemetry, saved as `{workload}_NVIDIA_GeForce_RTX_4050_Laptop_{hash}_{timestamp}.parquet`.

**Key constraint:** 6 GB VRAM. All batch sizes are reduced from the A100/RTX 5080 defaults to fit. The smaller batch sizes are themselves an interesting data point — they test whether the classifier generalizes across batch-size regimes, not just GPU architectures.

---

## Step 1: Environment Setup

```bash
# Clone the repo
git clone https://github.com/robirahman/SPAR-GPU-monitoring.git
cd SPAR-GPU-monitoring

# Create Python environment
python -m venv .venv
# On Windows:
.venv\Scripts\activate
# On Linux:
# source .venv/bin/activate

# Install dependencies
pip install -r requirements_v2.txt

# Verify CUDA and pynvml
python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0)}')"
python -c "import pynvml; pynvml.nvmlInit(); h=pynvml.nvmlDeviceGetHandleByIndex(0); print(pynvml.nvmlDeviceGetName(h))"
```

### Smoke test

```bash
python scripts/collect_telemetry.py --duration 5 --output /tmp/test_laptop.parquet
python -c "import pandas as pd; df=pd.read_parquet('/tmp/test_laptop.parquet'); print(f'OK: {len(df)} rows, {len(df.columns)} columns')"
```

### Notes on the laptop environment
- **Tier 1 only.** Do not attempt DCGM — it is Linux-only and unavailable on consumer GPUs regardless.
- **Nsight Compute** may or may not work depending on driver permissions. Don't spend time debugging it — NCU data is collected on the desktop RTX 5080.
- **Thermal throttling is expected.** The 60W TDP laptop GPU will throttle under sustained load. This is fine — the thermal throttling signature is itself useful data (laptop thermals differ from desktop/server).

---

## Step 2: Run the Full Collection

### Workload table (VRAM-adjusted for 6 GB)

| # | Workload | Registry label | Batch size | Duration/Epochs | VRAM adjustment vs. A100 default |
|---|----------|----------------|-----------|-----------------|----------------------------------|
| 1 | Idle | `idle` | — | 120s | None |
| 2 | ResNet-18 CIFAR-10 FP32 | `pytorch_resnet_cifar10` | 256 | 10 epochs | Halved from 512 |
| 3 | ResNet-18 CIFAR-10 AMP | `pytorch_resnet_cifar10_amp` | 256 | 10 epochs | Halved from 512 |
| 4 | MLP CIFAR-10 | `pytorch_mlp_cifar10` | 512 | 20 epochs | Fits as-is |
| 5 | GPT-2 124M WikiText-2 FP32 | `gpt2_wikitext2` | 2 | 400 steps | Reduced from 8 |
| 6 | GPT-2 124M WikiText-2 AMP | `gpt2_wikitext2_amp` | 4 | 400 steps | Reduced from 8 |
| 7 | BERT SST-2 FP32 | `bert_sst2` | 8 | 300 steps | Reduced from 32 |
| 8 | BERT SST-2 AMP | `bert_sst2_amp` | 16 | 300 steps | Reduced from 32 |
| 9 | ResNet-50 inference | `resnet50_inference` | 64 | 600s | Reduced from 256 |
| 10 | cuFFT benchmark | `cufft_benchmark` | — | 600s | Fits as-is |
| 11 | N-body simulation | `nbody_sim` | 4096 bodies | 600s | Fits as-is |
| 12 | Mining proxy | `mining_ethash_proxy` | 512 MB DAG | 600s | Halved from 1024 MB |
| 13 | Rendering proxy | `rendering_proxy` | 32768 rays | 600s | Fits as-is |
| 14 | FFmpeg NVENC | `ffmpeg_nvenc` | — | 300s | Fits as-is |

### Running the collection

Because batch sizes differ from the registry defaults, override them on the command line. Run 3 reps of each workload with a 10-second cooldown between runs.

```bash
cd SPAR-GPU-monitoring
# On Windows: .venv\Scripts\activate
# On Linux:   source .venv/bin/activate

# --- Idle ---
for run in 1 2 3; do
    python scripts/run_workload.py --workload idle
    sleep 10
done

# --- ML Training (reduced batch sizes for 6 GB VRAM) ---
for run in 1 2 3; do
    python scripts/run_workload.py --workload pytorch_resnet_cifar10 -- --batch-size 256
    python scripts/run_workload.py --workload pytorch_resnet_cifar10_amp -- --batch-size 256
    python scripts/run_workload.py --workload pytorch_mlp_cifar10
    python scripts/run_workload.py --workload gpt2_wikitext2 -- --batch-size 2
    python scripts/run_workload.py --workload gpt2_wikitext2_amp -- --batch-size 4
    python scripts/run_workload.py --workload bert_sst2 -- --batch-size 8
    python scripts/run_workload.py --workload bert_sst2_amp -- --batch-size 16
    sleep 10
done

# --- ML Inference ---
for run in 1 2 3; do
    python scripts/run_workload.py --workload resnet50_inference -- --batch-size 64
    sleep 10
done

# --- Scientific HPC ---
for run in 1 2 3; do
    python scripts/run_workload.py --workload cufft_benchmark
    python scripts/run_workload.py --workload nbody_sim
    sleep 10
done

# --- Crypto Mining ---
for run in 1 2 3; do
    python scripts/run_workload.py --workload mining_ethash_proxy -- --dag-size-mb 512
    sleep 10
done

# --- Rendering ---
for run in 1 2 3; do
    python scripts/run_workload.py --workload rendering_proxy
    sleep 10
done

# --- Other ---
for run in 1 2 3; do
    python scripts/run_workload.py --workload ffmpeg_nvenc
    sleep 10
done
```

### If a workload runs out of VRAM

If any workload OOMs at the batch sizes above, reduce further:
- GPT-2 FP32: try `--batch-size 1`
- BERT FP32: try `--batch-size 4`
- Mining proxy: try `--dag-size-mb 256`
- ResNet-50 inference: try `--batch-size 32`

Document any batch-size changes in the parquet metadata or a notes file.

---

## Step 3: Monitor Thermal Throttling

The laptop's thermal envelope is a useful data point. During each run, the telemetry pipeline already logs temperature and SM clock. After collection, check for throttling:

```python
import pandas as pd
import os

data_dir = "data/"
files = sorted([f for f in os.listdir(data_dir) if "RTX_4050" in f and f.endswith(".parquet")])

for f in files:
    df = pd.read_parquet(os.path.join(data_dir, f))
    label = df["workload_label"].iloc[0]
    temp_max = df["temperature_gpu"].max()
    clock_min = df["clocks_current_sm"].min()
    clock_max = df["clocks_current_sm"].max()
    clock_drop = clock_max - clock_min
    throttled = "THROTTLED" if clock_drop > 200 else "ok"
    print(f"{label:35s} temp_max={temp_max:3.0f}C  clock={clock_min:.0f}-{clock_max:.0f} MHz  ({throttled})")
```

If sustained workloads (mining, rendering, HPC) show significant clock drops (>200 MHz), note this in the analysis — it's a real-world signal that distinguishes laptop from desktop/server telemetry.

---

## Step 4: Verify Collected Data

After all runs are complete:

```python
import os
import pandas as pd

data_dir = "data/"
files = sorted([f for f in os.listdir(data_dir) if "RTX_4050" in f and f.endswith(".parquet")])

print(f"Total laptop files: {len(files)}")
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
    })

for label, runs in sorted(by_workload.items()):
    print(f"{label}: {len(runs)} runs")
    for r in runs:
        print(f"  {r['rows']:4d} rows, {r['duration_s']:5.0f}s, {r['gpu']}")
    print()
```

### Checklist

- [ ] At least 42 parquet files (14 workloads x 3 reps)
- [ ] All files contain `RTX_4050` in the filename
- [ ] GPU name in parquet metadata matches `NVIDIA GeForce RTX 4050 Laptop GPU` (or similar)
- [ ] ML training workloads completed all epochs/steps (check for early OOM termination)
- [ ] Temperature data present in all files (for throttling analysis)
- [ ] No workload silently fell back to CPU (check `gpu_utilization` > 0 during active phases)

---

## Step 5: Transfer Data

After collection, transfer the parquet files to the main project machine (loaned desktop) for analysis:

```bash
# Option A: Git (if files are small enough)
git add data/*RTX_4050*.parquet
git commit -m "Add RTX 4050 Laptop telemetry data (Week 8)"
git push

# Option B: Direct copy via USB drive or network share
# Copy data/*RTX_4050*.parquet to the desktop's data/ directory
```

---

## What This Data Enables

The laptop RTX 4050 data feeds into Thursday's (April 9) cross-GPU analysis:

1. **A100-only → 4050 transfer test:** Train classifier on A100 data, evaluate on 4050 data. Tests whether datacenter-trained models detect training on low-power consumer hardware.
2. **Leave-one-architecture-out:** Hold out Ada Lovelace data, train on Ampere + Hopper + Blackwell, test on Ada Lovelace.
3. **Pooled classifier:** Include 4050 data in the combined training set; evaluate via grouped cross-validation.
4. **Thermal throttling as a feature:** Laptop clock-drop patterns may be a distinguishing signal — or a confound that the classifier must learn to ignore.

---

## Estimated Time

| Phase | Time |
|-------|------|
| Environment setup + smoke test | ~30 min |
| ML training workloads (7 workloads x 3 reps) | ~4–6 hrs |
| ML inference (3 reps) | ~30 min |
| Scientific HPC (2 workloads x 3 reps) | ~1 hr |
| Mining + rendering + NVENC (3 workloads x 3 reps) | ~1.5 hrs |
| Idle (3 reps) | ~6 min |
| **Total** | **~7–9 hrs** |

The laptop can run overnight for the longer workloads. Start with ML training (the most time-consuming and highest-priority category), then queue the rest.
