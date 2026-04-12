# Datacenter Experiment Plan

**Goal:** Collect adversarial evasion data on 8x A100, 8x H100, and 8x B100
to extend the TAIGR paper beyond local hardware.

**Budget:** ~$60-110 total across 3 instances (~90 min each).

---

## Quick Start (copy-paste on each instance)

```bash
# 1. Clone and setup (~5 min)
git clone https://github.com/<your-repo>/spar-gpu-monitoring.git
cd spar-gpu-monitoring
sudo bash setup.sh
source .venv/bin/activate

# 2. Verify (~1 min)
python -c "import torch; print(torch.cuda.device_count(), 'GPUs'); print('NCCL:', torch.cuda.nccl.version())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 3. Dry run to confirm plan
python scripts/run_datacenter_collection.py --dry-run --num-gpus 8

# 4. Run everything (~60-75 min)
#    A100 or H100 (have existing baselines):
python scripts/run_datacenter_collection.py --num-gpus 8

#    B100 (no existing data — also run clean baselines):
python scripts/run_datacenter_collection.py --num-gpus 8 --include-baselines

# 5. Copy data off the instance before terminating
#    All output goes to data/ (parquets) — rsync or scp this directory.
tar czf spar_data_$(hostname)_$(date +%Y%m%d).tar.gz data/
#    Then scp this tarball to your local machine.
```

---

## What Gets Run

### Phase 1: Single-GPU adversarial (~10 min)
Runs 13 strategies in parallel across 8 GPUs (2 batches of 8, then 5):

| # | Strategy | Notes |
|---|----------|-------|
| 1 | adversarial_E_memory_minimal | Gradient checkpointing |
| 2 | adversarial_F_interleave_30 | Training + mining/cuFFT |
| 3 | adversarial_F_interleave_50 | Training + mining/cuFFT |
| 4 | adversarial_G_clock_throttled | **May fail on managed GPUs** (see Known Issues) |
| 5 | adversarial_H_mimicry_mining | **Reads wrong GPU** (see Known Issues) |
| 6 | adversarial_H_mimicry_cufft | **Reads wrong GPU** (see Known Issues) |
| 7 | adversarial_I_stochastic_30 | Stochastic interleaving |
| 8 | adversarial_J_pid_75 | **Reads wrong GPU** (see Known Issues) |
| 9 | adversarial_K_online_learning | Online learning camouflage |
| 10 | adversarial_L_diluted_2 | 50% training dilution |
| 11 | adversarial_L_diluted_5 | 20% training dilution |
| 12 | adversarial_L_diluted_10 | 10% training dilution |
| 13 | adversarial_L_diluted_20 | 5% training dilution |

### Phase 2: DDP baseline training (~18 min)
Uses all 8 GPUs together via torchrun + NCCL:

- `ddp_resnet18_cifar10` (300s)
- `ddp_resnet50_cifar10` (300s)
- `ddp_resnet18_cifar10_amp` (300s)

### Phase 3: DDP adversarial (~36 min)
Multi-GPU evasion strategies:

- `adversarial_K_ddp` — online learning, distributed
- `adversarial_L_ddp_5` — diluted 1/5, synchronized across ranks
- `adversarial_L_ddp_10` — diluted 1/10, synchronized
- `adversarial_L_ddp_stagger_5` — staggered dilution (each rank trains on
  different batches — **novel DDP evasion**)
- `adversarial_K_ddp_accum_4` — gradient accumulation every 4 steps (hides allreduce)
- `adversarial_K_ddp_accum_8` — gradient accumulation every 8 steps

---

## Known Issues and Workarounds

### 1. Strategies H, J read wrong GPU when run in parallel

`adversarial_round2.py` hardcodes `nvmlDeviceGetHandleByIndex(0)` for the
pynvml feedback loop. When run on GPU 3 via `CUDA_VISIBLE_DEVICES=3`, the
NVML handle still reads GPU 0's utilization/power, so the feedback controller
targets the wrong device.

**Impact:** The evasion strategy won't match the target signal correctly.
The training itself still runs on the right GPU (CUDA sees only one device).
The telemetry data is still valid — it's just that the evasion behavior won't
be as effective as on a single-GPU machine.

**Workaround:** Run H and J manually one at a time on GPU 0:
```bash
python scripts/run_workload.py --workload adversarial_H_mimicry_mining --gpu-index 0 --timeout 360 --no-dcgm
python scripts/run_workload.py --workload adversarial_H_mimicry_cufft --gpu-index 0 --timeout 360 --no-dcgm
python scripts/run_workload.py --workload adversarial_J_pid_75 --gpu-index 0 --timeout 360 --no-dcgm
```
This wastes 7 GPUs for ~15 min but gives correct behavior. Run the other 10
strategies in parallel first, then do H/J serially.

### 2. Strategy G may fail on managed GPUs

`adversarial_advanced.py` Strategy G runs `nvidia-smi -pl <watts>` and
`nvidia-smi --lock-gpu-clocks=210,600`. On datacenter GPUs managed by DCGM
or kubernetes device plugins, these commands may be blocked or require root.

**Workaround:** If it fails, skip it. G is already well-covered on RTX 5080
data. A failure here is not critical for the paper.

### 3. DCGM profiling fields may be blocked on virtualized instances

If the instance is a VM or container (Vast.ai, RunPod), DCGM profiling fields
(tensor_active, fp16/32/64 pipe activity, NVLink) may not be available even
with root access. The telemetry collector falls back to basic NVML fields
automatically.

**To check:** After setup, run:
```bash
dcgmi discovery -l
dcgmi dmon -e 1001,1004,1011 -c 3
```
If the second command shows values (not "N/A"), profiling fields work.
If not, add `--no-dcgm` to all run commands to skip DCGM and avoid error spam.

### 4. PyTorch version

`setup.sh` installs from `requirements.txt` which requests `torch>=2.1.0`.
This will install the latest stable PyTorch, which should support A100/H100
natively. For B100 (Blackwell datacenter), you may need:
```bash
pip install --pre torch torchvision --index-url https://download.pytorch.org/whl/nightly/cu128
```
Check after setup:
```bash
python -c "import torch; print(torch.__version__); print(torch.cuda.get_device_capability())"
```
If capability is (10, 0) or higher on B100 and PyTorch doesn't crash, you're fine.

### 5. torchrun + NCCL verified only on Linux

The DDP scripts (`pytorch_training_ddp.py`, `adversarial_ddp.py`) were
written and import-tested on Windows but `torch.distributed` only works on
Linux with NCCL. They have NOT been end-to-end tested on a multi-GPU
instance yet. If Phase 2/3 fails:

**Fallback:** Phase 1 (single-GPU adversarial) is the critical data.
Skip Phases 2/3 and still get the paper's main result (evasion tax
on datacenter hardware). DDP results are a bonus.

**Debug:** If torchrun hangs, try:
```bash
# Test NCCL directly
torchrun --nproc_per_node=2 -c "
import torch.distributed as dist
dist.init_process_group('nccl')
print(f'Rank {dist.get_rank()} OK')
dist.destroy_process_group()
"
```

### 6. nvidia-ml-py3 vs nvidia-ml-py

`requirements.txt` lists `nvidia-ml-py3`. On some systems this installs a
different (older) package than `nvidia-ml-py`. If pynvml imports fail:
```bash
pip uninstall nvidia-ml-py3 nvidia-ml-py -y
pip install nvidia-ml-py
```

---

## Instance-Specific Notes

### 8x A100 (Ampere)
- Already have 14 baseline workloads + A, B, D adversarial on H100 NVL
- Run standard plan (no `--include-baselines`)
- DCGM profiling likely works on bare-metal
- Expected Phase 1 time: ~10 min

### 8x H100 (Hopper)
- Already have 16-20 baseline workloads
- Run standard plan (no `--include-baselines`)
- DCGM profiling likely works on bare-metal
- Expected Phase 1 time: ~10 min

### 8x B100 (Blackwell datacenter)
- **No existing data** — run with `--include-baselines` to collect clean workloads too
- May need PyTorch nightly for Blackwell support (see Known Issue #4)
- Expected Phase 1 time: ~20 min (26 workloads instead of 13)
- This adds a 6th architecture to the cross-GPU generalization story

---

## Data Transfer

After each instance finishes, copy the data before terminating:

```bash
# Option A: tar + scp
tar czf spar_data_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')_$(date +%Y%m%d).tar.gz data/
scp spar_data_*.tar.gz you@your-machine:~/spar-gpu-monitoring/

# Option B: rsync
rsync -avz data/ you@your-machine:~/spar-gpu-monitoring/data/

# Option C: git (if parquets aren't too large)
git add data/*.parquet
git commit -m "Add datacenter adversarial data: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
git push
```

**Important:** The parquets go into `data/` (same directory as existing data).
The filename encodes the GPU name, so they won't collide with existing files.

---

## After Collection (back on local machine)

```bash
# 1. Merge the tarballs into the repo
tar xzf spar_data_A100_*.tar.gz
tar xzf spar_data_H100_*.tar.gz
tar xzf spar_data_B100_*.tar.gz

# 2. Verify
ls data/*A100*.parquet | wc -l
ls data/*H100*.parquet | wc -l
ls data/*B100*.parquet | wc -l

# 3. Re-run analysis
python -m classifier.cross_gpu_generalization      # Updated 6-GPU matrix
python -m classifier.adversary_defender_iteration   # Retrain with datacenter adversarial
python -m classifier.leave_one_strategy_out         # LOO with datacenter data
python -m classifier.energy_bound                   # Energy profiles per GPU type

# 4. Update paper
# - Cross-GPU heatmap now has 6x6 matrix (add B100)
# - LOO evaluation includes datacenter adversarial strategies
# - Evasion tax bound validated across datacenter hardware
```

---

## Pre-Rental Checklist

### Done
- [x] `workloads/pytorch_training_ddp.py` — DDP training (resnet18/50/mlp)
- [x] `workloads/adversarial_ddp.py` — 4 strategies (K-DDP, L-DDP, L-stagger, K-accum)
- [x] `scripts/run_workload_multi_gpu.py` — per-GPU telemetry orchestrator
- [x] `scripts/run_datacenter_collection.py` — automated 3-phase collection
- [x] `workloads/registry.py` — 9 DDP entries added (54 total workloads)
- [x] Tested imports, model building, arg parsing, GPU forward/backward locally
- [x] Dry-run verified: correct plan for 8 GPUs

### Still TODO before renting
- [ ] **Push to GitHub** so instances can `git clone`
- [ ] Fix strategies H/J for per-GPU NVML device (or accept the workaround above)
- [ ] Fix strategy G for per-GPU `nvidia-smi -i N` (or accept it may fail)

### Judgment calls for on-instance
- If DCGM profiling is blocked: add `--no-dcgm` and continue
- If B100 needs nightly PyTorch: install it (see Known Issue #4)
- If DDP phases fail: Phase 1 alone is sufficient for the paper
- If strategy G fails: skip it, not critical
- Run H/J/G serially on GPU 0 after the parallel batch if you want correct feedback behavior
