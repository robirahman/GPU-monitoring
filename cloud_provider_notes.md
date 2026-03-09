# Cloud Provider Notes for GPU Profiling

## Requirements

This project needs **data-center GPUs** (A100, H100, or newer) for DCGM profiling fields (Tier 2). For Nsight Compute (Tier 3), the host kernel parameter `RmProfilingAdminOnly=0` is required, which is only available on true bare-metal or properly configured hosts. Most cloud providers use virtualization that blocks this.

**Key distinction:**
- **Tier 2 (DCGM profiling fields)**: Works on data-center GPUs with GPU passthrough. Does NOT require bare metal. Should work on most cloud A100/H100 instances.
- **Tier 3 (NCU)**: Requires `RmProfilingAdminOnly=0` in the host kernel. Most cloud providers restrict this even on "bare-metal" instances. True bare-metal or specially configured hosts are needed.

## Provider Comparison

| Provider | Tier 2 (DCGM profiling) | Tier 3 (NCU) | GPU options | $/hr | Notes |
|----------|------------------------|--------------|-------------|------|-------|
| **RunPod Secure Cloud** | Untested (see below) | **No** -- confirmed not bare-metal | A100, H100 | $1.14-1.39 (A100), ~$2-3 (H100) | Blocks mining software; "instant cluster" (16+ GPUs) may allow multi-server monitoring experiments |
| **Massed Compute** | Yes | **Likely yes** -- has Nsight docs, offers bare-metal on request | A100, H100 | Varies (check website) | Most promising for NCU |
| **Nebius** | Likely yes | Reportedly yes | H100 (A100 may be retired) | ~$2/hr (H100) | Explorer Tier at $1.99/hr H100 |
| **Vast.ai** | **Partial** -- basic DCGM fields only, profiling fields blocked by virtualization | **No** -- not bare-metal despite some claims | A100, H100 | ~$0.70+ | Confirmed: NCU does not work; DCGM profiling module reports "not supported as non root" even when running as root |
| **Lambda Labs** | Likely yes | **No** -- staff confirmed unsupported | A100, H100 | ~$1.29 (A100), ~$2.49 (H100) | Virtualization blocks NCU |
| **CoreWeave** | Likely yes | Unknown | A100, H100 | Enterprise pricing | Probably overkill for this project |

Sources:
- Lambda NCU unsupported: https://deeptalk.lambda.ai/t/nvidia-profiling-support/4763
- Massed Compute Nsight docs: https://massedcompute.com/faq-answers/?question=How+do+I+use+NVIDIA+Nsight+Systems+and+NVIDIA+Nsight+Compute+in+a+cloud+environment?
- Nebius pricing: https://nebius.com/prices

## Tested Results (Week 3)

### Vast.ai (2x A100 SXM4 40GB) -- current instance
- **Tier 1 (pynvml)**: Works perfectly. 15 metrics at 1 Hz.
- **Tier 2 (DCGM)**: Partial. Basic fields work (gpu_util, power, temp, sm_clock, mem_clock). Profiling fields (tensor_active, fp16/32/64 pipes, SM occupancy, DRAM bandwidth) are blocked -- DCGM profiling module errors with "not supported when the host engine is running as non root" even as root. This is the virtualization layer blocking it.
- **Tier 3 (NCU)**: Not tested yet, expected to fail (not bare-metal).
- **Collection status**: Running Tier 1 + basic DCGM collection across all 15 workloads (3 runs each).

### RunPod -- contacted
- Confirmed: **not bare-metal**. NCU will not work.
- DCGM profiling fields still untested -- may have same issue as Vast.ai.
- **Instant cluster option** (16+ GPUs): Could be useful for multi-server monitoring experiments in later weeks. Not needed for current data collection.

## Decision

**Current approach:**

### Tier 1 + basic DCGM collection: Vast.ai (in progress)
- Collecting 25 columns per sample across all 15 workloads
- 3 runs per workload = 45 total runs
- This data is sufficient for temporal pattern analysis (the project's novel contribution)

### Tier 2 profiling fields: Need a provider that supports them
- RunPod may work (untested) -- worth a 1-hour test
- Massed Compute likely supports it (data-center GPUs, bare-metal option)
- Budget: ~$2-4 for a 1-hour test

### Tier 3 NCU profiling: Massed Compute (or Nebius)
- Massed Compute offers bare-metal on request and has explicit Nsight documentation
- Do a 1-hour test to confirm NCU works before committing
- Only need ~1 run per workload type (much less time than bulk collection)
- Nebius is the backup if Massed Compute doesn't work out

### Fallback
If no provider supports full Tier 2 or NCU:
- Tier 1 + basic DCGM (25 columns) is already collected on Vast.ai
- The project's novel contribution is temporal patterns from Tier 1+2 -- NCU is supplementary
- Document the provider search in the report

## Workload-Specific Software Requirements

Most workloads are pure Python/PyTorch and run anywhere. Three optional workloads need external software:

### GROMACS (`gromacs_adh`)
The Ubuntu package (`apt install gromacs`) is compiled **without GPU support**. To collect GPU telemetry for molecular dynamics, you need a GPU-enabled build:

- **Option 1 — NGC container**: `nvcr.io/hpc/gromacs:2023.3` has CUDA support built in. Run with `docker run --gpus all ...` or use Singularity/Apptainer on HPC systems.
- **Option 2 — Build from source**: Download from https://manual.gromacs.org and build with `-DGMX_GPU=CUDA`. Requires CUDA toolkit and cmake.
- **Option 3 — Different instance**: Some cloud providers (e.g., Lambda Labs, CoreWeave) offer instances with GPU-accelerated GROMACS pre-installed or available via containers.

Without GPU support, GROMACS runs on CPU only and won't produce a meaningful GPU telemetry signature.

### FFmpeg NVENC (`ffmpeg_nvenc`)
FFmpeg with NVENC requires a GPU that has a **hardware video encoder** (NVENC chip). Data-center GPUs like A100 and H100 **do not have NVENC**. This workload requires:

- A consumer/professional GPU (e.g., RTX 3090, RTX 4090, L40) or an older data-center GPU with NVENC (e.g., T4)
- The `ffmpeg` build must be compiled with `--enable-nvenc` (the Ubuntu package includes this, but the hardware must support it)

### Blender Cycles (`blender_bmw`)
Works on A100/H100 via CUDA. The workload script generates a procedural scene (no download needed) and disables the denoiser for headless compatibility. First run compiles CUDA kernels (~5 min one-time cost). Already collected successfully on Vast.ai A100.

## GPU Selection: A100 vs H100

Either A100 or H100 works for this project. Choose based on price:
- A100 40/80GB: Cheaper (~$1-1.40/hr), sufficient VRAM, full DCGM support
- H100 80GB: More expensive (~$2-3/hr), but may be the only option on some providers (e.g., Nebius)
- Both have tensor cores, DCGM profiling fields, and NCU support (if host allows)
- The telemetry pipeline works identically on both -- just different absolute values for metrics
