# Datacenter Experiment Plan

**Goal:** Collect adversarial evasion data and FSDP NVLink stress data on
8x A100, 8x H100, and 8x B100 to extend the TAIGR paper beyond local hardware.

**Budget:** ~$60-110 total across 3 instances (~90 min each).

---

## Lessons Learned from A100 Run (2026-04-12)

> These notes capture what actually happened on the first 8x A100-SXM4-80GB
> run so we don't repeat mistakes on H100/B100.

1. **Allocate >=100GB root disk.** The default 32GB overlay on the A100 instance
   left only 13GB free after installing dependencies. We had to commit parquets
   to GitHub via LFS and delete them locally between phases to free space.
   On H100/B100, request at least 100GB root disk at instance creation time.

2. **Use `--no-dcgm` by default.** DCGM profiling fields (1001, 1004, 1011)
   were blocked on the A100 instance (containerized environment). Basic DCGM
   monitoring fields (100, 155, 204) worked, but the profiling fields returned
   permission errors even as root. The `nv-hostengine` daemon needed to be
   started manually and initially only listened on a Unix domain socket that
   `dcgmi` couldn't connect to -- restarting it without `-d` fixed this.
   Use `--no-dcgm` to avoid error spam; NVML Tier 1 telemetry is sufficient.

3. **`run_workload_multi_gpu.py` had a bug** (now fixed): it passed `interval=`
   and `label=` to `TelemetryCollector` instead of `interval_sec=` and
   `workload_label=`, and used `c.gpu_name` instead of `c.metadata.gpu_name`.
   All 9 DDP workloads in Phases 2-3 failed on the first attempt. The fix is
   committed. This should not recur.

4. **NCCL teardown warnings are harmless.** Every DDP/FSDP workload produces
   `c10::Error` and `Broken pipe` messages on process group shutdown. These are
   normal -- rank 0 tears down the TCPStore server before other ranks'
   heartbeat monitors disconnect. The workloads still complete successfully.

5. **FSDP memory limits on A100-80GB:**
   - 7B (5.3B params): 13.9 GB/GPU with AdamW -- plenty of headroom
   - 13B (10.2B params): 25.8 GB/GPU with AdamW -- comfortable
   - 30B (25.4B params): 42.1 GB/GPU with SGD (momentum=0.9) -- AdamW OOMs
   - 70B (59.6B params): OOM at 82.4 GB even with plain SGD (no momentum)
   - AdamW needs 2 fp32 state tensors per param; SGD+momentum needs 1; plain SGD needs 0
   - On H100-80GB, expect similar limits. On B100 (if 80GB), same. On H200 (141GB), 70B should fit.

6. **`setup.sh` targets Ubuntu 22.04** but the A100 instance was Ubuntu 24.04.
   The CUDA repo keyring was already configured for 24.04, so `apt-get install
   datacenter-gpu-manager` worked directly. DCGM Python bindings were at
   `/usr/local/dcgm/bindings/python3/` (not `/usr/lib/python3/dist-packages/`
   as `setup.sh` assumes). The venv `.pth` file was fixed manually.

7. **PyTorch installed was 2.11.0+cu130** (CUDA 13.1). Worked perfectly on A100.
   No issues with any workload.

8. **Strategy G (clock throttled) worked** on the A100 instance despite being
   in a container. This may not hold on all providers.

9. **Strategies H, J worked in parallel** but with the known wrong-GPU NVML
   feedback issue. The telemetry data is still valid, just the evasion
   effectiveness is reduced. We did not run the serial workaround.

10. **Git LFS for data transfer works well.** Parquets are tracked via
    `data/*.parquet` in `.gitattributes`. Commit, push, delete local copies,
    continue. Much simpler than scp/rsync for iterative runs.

---

## Quick Start (copy-paste on each instance)

**IMPORTANT:** When creating the instance, allocate **>=100GB root disk space.**

```bash
# 1. Clone and setup (~5 min)
git clone https://github.com/robirahman/SPAR-GPU-monitoring.git
cd SPAR-GPU-monitoring
sudo bash setup.sh
source .venv/bin/activate

# 1b. Fix DCGM Python bindings path if needed
#     (setup.sh assumes /usr/lib/python3/dist-packages/ but DCGM may be elsewhere)
python -c "import pydcgm; print('DCGM OK')" 2>/dev/null || {
    DCGM_PATH=$(find /usr -name "pydcgm.py" -path "*/python*" 2>/dev/null | head -1 | xargs dirname)
    if [ -n "$DCGM_PATH" ]; then
        VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
        echo "$DCGM_PATH" > "$VENV_SITE/dcgm.pth"
        echo "Fixed DCGM path: $DCGM_PATH"
    fi
}

# 1c. Start DCGM host engine (if not running)
if ! pgrep -x nv-hostengine &>/dev/null; then
    nv-hostengine --log-level INFO 2>&1
fi

# 2. Verify (~1 min)
python -c "import torch; print(torch.cuda.device_count(), 'GPUs'); print('NCCL:', torch.cuda.nccl.version())"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

# 2b. Check DCGM (optional -- we use --no-dcgm anyway)
dcgmi discovery -l
dcgmi dmon -e 1001,1004,1011 -c 3  # if this fails, profiling fields are blocked

# 3. Dry run to confirm plan
python scripts/run_datacenter_collection.py --dry-run --num-gpus 8

# 4. Run everything (~90 min for Phases 1-4)
#    A100 or H100 (have existing baselines):
python scripts/run_datacenter_collection.py --num-gpus 8 --no-dcgm

#    B100 (no existing data -- also run clean baselines):
python scripts/run_datacenter_collection.py --num-gpus 8 --no-dcgm --include-baselines

# 5. Commit data to GitHub via LFS and push
git add data/*.parquet data/datacenter_collection_progress.json
git commit -m "Add datacenter data: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
git push
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
| 4 | adversarial_G_clock_throttled | Worked on A100 container; may fail elsewhere |
| 5 | adversarial_H_mimicry_mining | Wrong-GPU NVML feedback (see Known Issues) |
| 6 | adversarial_H_mimicry_cufft | Wrong-GPU NVML feedback (see Known Issues) |
| 7 | adversarial_I_stochastic_30 | Stochastic interleaving |
| 8 | adversarial_J_pid_75 | Wrong-GPU NVML feedback (see Known Issues) |
| 9 | adversarial_K_online_learning | Online learning camouflage |
| 10 | adversarial_L_diluted_2 | 50% training dilution |
| 11 | adversarial_L_diluted_5 | 20% training dilution |
| 12 | adversarial_L_diluted_10 | 10% training dilution |
| 13 | adversarial_L_diluted_20 | 5% training dilution |

**A100 result:** 13/13 OK. ~10 min total.

### Phase 2: DDP baseline training (~18 min)
Uses all 8 GPUs together via torchrun + NCCL:

- `ddp_resnet18_cifar10` (300s) -- 200 epochs on A100
- `ddp_resnet50_cifar10` (300s) -- 37 epochs on A100
- `ddp_resnet18_cifar10_amp` (300s)

**A100 result:** 3/3 OK. ~18 min total. NCCL teardown warnings are normal.

### Phase 3: DDP adversarial (~36 min)
Multi-GPU evasion strategies:

- `adversarial_K_ddp` -- online learning, distributed
- `adversarial_L_ddp_5` -- diluted 1/5, synchronized across ranks
- `adversarial_L_ddp_10` -- diluted 1/10, synchronized
- `adversarial_L_ddp_stagger_5` -- staggered dilution (**novel DDP evasion**)
- `adversarial_K_ddp_accum_4` -- gradient accumulation every 4 steps
- `adversarial_K_ddp_accum_8` -- gradient accumulation every 8 steps

**A100 result:** 6/6 OK. ~36 min total.

### Phase 4: FSDP NVLink stress (~25 min)
Model-sharded training via FSDP FULL_SHARD. Generates heavy AllGather +
ReduceScatter NVLink traffic on every forward/backward step -- orders of
magnitude more interconnect traffic than DDP with ResNet-50.

- `fsdp_7b_nvlink` (300s) -- 5.3B params, AdamW, batch=4, seq_len=1024
- `fsdp_13b_nvlink` (300s) -- 10.2B params, AdamW, batch=2, seq_len=1024
- `fsdp_30b_nvlink` (600s) -- 25.4B params, SGD (momentum=0.9), batch=1, seq_len=512

**A100 result:** 3/3 OK. Performance summary:

| Model | Params | Peak Mem/GPU | Throughput | NVLink/step (est.) |
|-------|--------|-------------|------------|-------------------|
| 7B | 5.3B | 13.9 GB | 13.4K tok/s | ~21 GB |
| 13B | 10.2B | 25.8 GB | 5.2K tok/s | ~41 GB |
| 30B | 25.4B | 42.1 GB | 3.86K tok/s | ~102 GB |

**70B does not fit on 80GB GPUs.** Tested with plain SGD (no momentum),
batch=1, seq_len=256 -- OOM at 82.4GB. Would need H200 (141GB) or CPU
offloading (too slow for useful telemetry: 1 step in 20s vs hundreds of steps).

---

## Known Issues and Workarounds

### 1. Strategies H, J read wrong GPU when run in parallel

`adversarial_round2.py` hardcodes `nvmlDeviceGetHandleByIndex(0)` for the
pynvml feedback loop. When run on GPU 3 via `CUDA_VISIBLE_DEVICES=3`, the
NVML handle still reads GPU 0's utilization/power, so the feedback controller
targets the wrong device.

**Impact:** The evasion strategy won't match the target signal correctly.
The training itself still runs on the right GPU (CUDA sees only one device).
The telemetry data is still valid -- it's just that the evasion behavior won't
be as effective as on a single-GPU machine.

**A100 experience:** We ran H/J in parallel with other strategies. The data
was collected but evasion effectiveness was reduced. Not worth the ~15 min
serial workaround unless you need perfect evasion behavior.

**Workaround (if needed):** Run H and J manually one at a time on GPU 0:
```bash
python scripts/run_workload.py --workload adversarial_H_mimicry_mining --gpu-index 0 --timeout 360 --no-dcgm
python scripts/run_workload.py --workload adversarial_H_mimicry_cufft --gpu-index 0 --timeout 360 --no-dcgm
python scripts/run_workload.py --workload adversarial_J_pid_75 --gpu-index 0 --timeout 360 --no-dcgm
```

### 2. Strategy G may fail on managed GPUs

`adversarial_advanced.py` Strategy G runs `nvidia-smi -pl <watts>` and
`nvidia-smi --lock-gpu-clocks=210,600`. On datacenter GPUs managed by DCGM
or kubernetes device plugins, these commands may be blocked or require root.

**A100 experience:** Strategy G worked fine in the container with root access.

**Workaround:** If it fails, skip it. G is already well-covered on RTX 5080
data. A failure here is not critical for the paper.

### 3. DCGM profiling fields blocked on containerized instances

DCGM profiling fields (1001 tensor_active, 1004 fp16/32/64 pipe, 1011/1012
NVLink bytes) require elevated permissions that are typically blocked in
containers, even with root.

**A100 experience:** `dcgmi dmon -e 1001,1004,1011` returned permission errors.
Basic fields (100 sm_clock, 155 power, 204 mem_util) worked fine. The
nv-hostengine daemon needed manual start: `nv-hostengine --log-level INFO`
(without `-d` which forces domain socket mode that dcgmi can't connect to).

**Workaround:** Use `--no-dcgm` for all run commands. NVML Tier 1 telemetry
(GPU util, memory, power, PCIe TX/RX, clocks, temperature) is collected
regardless and is sufficient for the classifier.

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

**A100 experience:** PyTorch 2.11.0+cu130 installed automatically. No issues.

### 5. NCCL teardown warnings (not a real error)

Every DDP and FSDP workload produces alarming-looking `c10::Error` stack traces
and `Broken pipe` / `TCPStore server has shut down too early` warnings during
process group teardown. **These are harmless.** Rank 0 finishes training and
tears down the TCPStore server before other ranks' heartbeat monitors
disconnect. The workloads complete successfully despite the warnings.

**How to tell if a real error occurred:** Check the collection script's output
for `OK` or `FAIL(N)` status after each workload label. Ignore the c10 stack
traces between workload runs.

### 6. nvidia-ml-py3 vs nvidia-ml-py

`requirements.txt` lists `nvidia-ml-py3`. On some systems this installs a
different (older) package than `nvidia-ml-py`. If pynvml imports fail:
```bash
pip uninstall nvidia-ml-py3 nvidia-ml-py -y
pip install nvidia-ml-py
```

### 7. DCGM Python bindings path varies by installation

`setup.sh` adds `/usr/lib/python3/dist-packages` to the venv's `.pth` file,
but DCGM may install Python bindings elsewhere.

**A100 experience:** Bindings were at `/usr/local/dcgm/bindings/python3/`.

**Fix:** The Quick Start section above includes a step to auto-detect and fix
the path. If `import pydcgm` fails after setup, run:
```bash
DCGM_PATH=$(find /usr -name "pydcgm.py" -path "*/python*" 2>/dev/null | head -1 | xargs dirname)
VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
echo "$DCGM_PATH" > "$VENV_SITE/dcgm.pth"
```

### 8. FSDP 30B needs SGD optimizer (not AdamW)

AdamW maintains 2 fp32 state tensors per parameter (mean + variance). For a
25.4B-param model sharded across 8 GPUs, this is ~25.4GB/GPU for optimizer
alone, which combined with params + gradients + activations exceeds 80GB.

SGD with momentum has only 1 state tensor, fitting comfortably at 42.1GB/GPU.
The `fsdp_training.py` script automatically selects SGD for 30b+ sizes.

### 9. Disk space: 32GB root overlay is insufficient

The A100 instance had a 32GB root overlay with only 13GB free after
dependencies. With ~366 parquet files from Phases 1-3 plus the CIFAR-10
dataset and Python venv, disk filled up. We worked around this by committing
to GitHub LFS and deleting local copies between phases.

**Prevention:** Allocate >=100GB root disk when creating the instance. This
avoids the need for mid-run data evacuation.

---

## Instance-Specific Notes

### 8x A100 (Ampere) -- COMPLETED 2026-04-12
- **All 25 workloads completed successfully** (Phases 1-4)
- Driver 590.48.01, CUDA 13.1, PyTorch 2.11.0, NCCL 2.28.9
- Ubuntu 24.04 (not 22.04 as expected -- no issues)
- DCGM profiling fields blocked; used `--no-dcgm`
- Phase 1: 13/13 OK (~10 min)
- Phase 2: 3/3 OK (~18 min)
- Phase 3: 6/6 OK (~36 min)
- Phase 4: 3/3 OK (~25 min) -- 7B, 13B, 30B
- 70B OOM (does not fit on 80GB GPUs)
- Data pushed to GitHub via LFS

### 8x H100 (Hopper) -- TODO
- Already have 16-20 baseline workloads from previous runs
- Run standard plan with `--no-dcgm` (no `--include-baselines`)
- **Allocate >=100GB root disk**
- H100-80GB has same memory as A100, so expect same FSDP limits
  (7B/13B/30B OK, 70B OOM)
- H100 has higher NVLink bandwidth (900 GB/s vs 600 GB/s on A100),
  so FSDP throughput may be higher
- Expected total time: ~90 min (Phases 1-4)

### 8x B100 (Blackwell datacenter) -- TODO
- **No existing data** -- run with `--include-baselines` to collect clean
  baselines too
- **Allocate >=100GB root disk**
- May need PyTorch nightly for Blackwell support (see Known Issue #4)
- If B100 has 80GB VRAM: same FSDP limits as A100/H100
- If B100 has 192GB VRAM (B200 variant): 70B should fit with AdamW
- Expected total time: ~110 min (Phases 1-4 + baselines)
- This adds a 6th architecture to the cross-GPU generalization story

---

## Data Transfer

Data is tracked by git LFS (`data/*.parquet` in `.gitattributes`). After each
phase or at the end of the run:

```bash
# Commit and push via LFS (preferred)
git add data/*.parquet data/datacenter_collection_progress.json
git commit -m "Add datacenter data: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1)"
git push

# If disk is tight, delete local copies after pushing:
git lfs prune
rm data/*.parquet  # safe -- they're on GitHub

# Fallback: tar + scp
tar czf spar_data_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')_$(date +%Y%m%d).tar.gz data/
scp spar_data_*.tar.gz you@your-machine:~/spar-gpu-monitoring/
```

**Important:** The parquets go into `data/` (same directory as existing data).
The filename encodes the GPU name, so they won't collide with existing files.

---

## After Collection (back on local machine)

```bash
# 1. Pull all data
git pull

# 2. Verify
ls data/*A100*.parquet | wc -l   # expect ~100 (Phase 1-4, 8 GPUs each)
ls data/*H100*.parquet | wc -l
ls data/*B100*.parquet | wc -l
ls data/*fsdp*.parquet | wc -l   # FSDP NVLink stress data

# 3. Re-run analysis
python -m classifier.cross_gpu_generalization      # Updated 6-GPU matrix
python -m classifier.adversary_defender_iteration   # Retrain with datacenter adversarial
python -m classifier.leave_one_strategy_out         # LOO with datacenter data
python -m classifier.energy_bound                   # Energy profiles per GPU type

# 4. Update paper
# - Cross-GPU heatmap now has 6x6 matrix (add B100)
# - LOO evaluation includes datacenter adversarial strategies
# - Evasion tax bound validated across datacenter hardware
# - FSDP interconnect traffic analysis (new section)
```

---

## Pre-Rental Checklist

### Done
- [x] `workloads/pytorch_training_ddp.py` -- DDP training (resnet18/50/mlp)
- [x] `workloads/adversarial_ddp.py` -- 4 strategies (K-DDP, L-DDP, L-stagger, K-accum)
- [x] `workloads/fsdp_training.py` -- FSDP sharded training (1B-70B, NVLink stress)
- [x] `scripts/run_workload_multi_gpu.py` -- per-GPU telemetry orchestrator (bug fixed)
- [x] `scripts/run_datacenter_collection.py` -- automated 4-phase collection
- [x] `workloads/registry.py` -- 13 DDP + FSDP entries added (58 total workloads)
- [x] Tested imports, model building, arg parsing, GPU forward/backward locally
- [x] Dry-run verified: correct plan for 8 GPUs
- [x] **Full run on 8x A100: 25/25 workloads OK**
- [x] Pushed to GitHub with LFS-tracked parquets
- [x] `data/*.parquet` tracked in `.gitattributes` for LFS

### Still TODO before H100/B100 rentals
- [ ] Fix strategies H/J for per-GPU NVML device (or accept the workaround)
- [ ] Fix strategy G for per-GPU `nvidia-smi -i N` (or accept it may fail)
- [ ] Test 70B FSDP on H200 or other >80GB GPU instance (optional)

### Judgment calls for on-instance
- If DCGM profiling is blocked: use `--no-dcgm` (this is now the default recommendation)
- If B100 needs nightly PyTorch: install it (see Known Issue #4)
- If strategy G fails: skip it, not critical
- Strategies H/J: accept reduced evasion effectiveness in parallel runs
- If 30B FSDP OOMs on a different GPU: try `--batch-size 1 --seq-len 256`
- Git LFS is set up -- commit and push data rather than using tar/scp
