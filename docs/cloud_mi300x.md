# Next-Steps Doc: Datacenter AMD Corpus on MI300X / MI325X

This doc is what to do once you've rented a datacenter AMD instance. It
replaces nothing — supplements `setup_amd_datacenter.sh` (turnkey installer)
and the high-level plan in `~/.claude/plans/what-else-should-we-soft-kahn.md`.

## Why this matters

The paper's Figure 1 (cross-GPU transfer heatmap) already shows
**consumer ↔ datacenter NVIDIA transfer is weak**: RTX 5080/5090/4050
rows drop below 50% accuracy on many datacenter targets. We've now
collected a complete RX 9070 XT (consumer AMD, RDNA 4) corpus. Adding
**MI300X / MI325X telemetry** gives us:

1. **Consumer→datacenter AMD transfer** — direct analog to the paper's
   weakest cross-tier finding, on the other vendor. Likely publishable.
2. **Multi-GPU DDP adversarial families** — K_ddp, L_ddp_5/10,
   L_ddp_stagger_5, K_ddp_accum_4/8. These don't run on our 1-GPU local
   box; the paper's Round-4 evader suite includes them and we should
   match.
3. **The full 10×10+ cross-vendor heatmap** with both consumer and
   datacenter tiers on both vendors.

## Provider options

Updated 2026-05 (May 17). Prices are spot-checked from public pages, will
drift. Provisioning availability fluctuates by week — check live.

| Provider | GPUs available | $/hr/GPU | Notes |
|---|---|---|---|
| **RunPod** (Community Cloud) | MI300X 1× and 8× | $2.49–2.99 (1×), $19.92/node (8×) | Fastest to provision. PyTorch ROCm image available. Ubuntu 22.04. |
| **TensorWave** | MI300X, MI325X (1× and 8×) | $2.99 (MI300X), $4.99 (MI325X) | AMD-focused, often has 8× nodes when RunPod is dry. |
| **Hot Aisle** | MI300X 1× and 8× | ~$2.50 (1×) | Newer entrant, AMD-focused. |
| **AMD Developer Cloud** | MI300X | Free hours for accepted developers | Apply at amd.com — slow turnaround but free credits. |
| **Vultr** | MI300X | $2.59 (1×) | Bare-metal option. |
| **Vast.ai** | Sporadic MI300X listings | varies | Untested for ROCm; check the host's reported `rocminfo` before committing. |
| **Massed Compute** | MI300X | varies | Cloud-provider notes (`cloud_provider_notes.md`) flag them as the most NCU-friendly NVIDIA host, may carry to AMD too. |

**My recommendation for the budget envelope:** 1 night on a RunPod MI300X
8× community-cloud pod is $19.92/h × ~7h = $140 and gets us the entire
single-GPU sweep + multi-GPU DDP sweep in one go. If 8× is unavailable,
fall back to 1× MI300X (~$3/h × 7h ≈ $20) for the single-GPU sweep and
re-rent later for DDP.

If we also want **MI325X** for HBM3e contrast: TensorWave is the most
reliable, $4.99/h × 4h ≈ $20 for a single-GPU sweep.

## Provisioning checklist

### RunPod (community cloud, fastest path)

```bash
# Web UI:
# 1. Pods > Deploy > Community Cloud > filter "AMD MI300X"
# 2. Pick an 8x node if available, else 1x.
# 3. Container image: runpod/pytorch:2.4.0-py3.11-rocm6.1.0-ubuntu22.04
#    (or whatever the current rocm6.4+ tag is — image gets refreshed)
# 4. Set disk: 100 GB volume, 50 GB container.
# 5. Expose SSH (port 22).
# 6. Set start command: "sleep infinity" (so pod doesn't autoexit).
# 7. Deploy. Note the ssh command from the dashboard.

# After SSH:
git clone https://github.com/robirahman/SPAR-GPU-monitoring.git
cd SPAR-GPU-monitoring
git checkout amd
git lfs install && git lfs pull   # we need the existing NVIDIA + AMD corpus
                                  # for downstream cross-vendor eval
sudo ./setup_amd_datacenter.sh    # installs ROCm 6.4, venv, PyTorch ROCm,
                                  # amdsmi-py matching libamd-smi version
exec newgrp render                # apply group membership without re-login
source .venv/bin/activate
python scripts/amdsmi_smoke_test.py   # verify telemetry path
```

If the smoke test reports all 8 GPUs with valid power/temp/clock, you're
good. If `amdsmi.amdsmi_get_gpu_activity` raises `UNEXPECTED_DATA` on some
handles, those are probably CPU sockets enumerated alongside the dGPUs
(the AMD library quirk we hit on the local box too); the collector handles
this by skipping them. Confirm via `rocminfo | grep Marketing` — should
list 8× "AMD Instinct MI300X" rows.

### TensorWave (MI325X)

```bash
# Web UI: tensorwave.com > Provision > MI325X
# Ubuntu 22.04 image, ROCm 6.4 pre-installed on most images.
# SSH key uploaded via console.

# After SSH:
git clone https://github.com/robirahman/SPAR-GPU-monitoring.git
cd SPAR-GPU-monitoring && git checkout amd
git lfs install && git lfs pull
# If ROCm is already there, setup_amd_datacenter.sh's apt-install steps no-op.
sudo ./setup_amd_datacenter.sh
exec newgrp render
source .venv/bin/activate
python scripts/amdsmi_smoke_test.py
```

## What to run

### Phase A — single-GPU sweep (~7 h, $20–25 on MI300X 1×)

```bash
nohup scripts/run_amd_corpus.sh --reps 3 --all --gpu-index 0 \
    > data/collection_logs/dc_corpus.out 2>&1 &
# Watcher (commits each parquet)
nohup scripts/commit_amd_parquets.sh --poll-sec 30 \
    > data/collection_logs/dc_watcher.out 2>&1 &
```

Expected:
- ~58 runs total (24 clean × 3 + 18 adversarial × 3 + 5 canonical
  whitebox × 3 + WB-O × 3 + J × 3).
- The `pytorch_resnet_cifar10` FP32 variant will hit the 1200s timeout
  (it's a 10-epoch training that doesn't have a `--duration` cap).
  That's normal — `exit=247` just means timeout-killed, data is fine.
- Larger workloads (`llm_infer_qwen25_3b`, SDXL) fit more comfortably on
  MI300X's 192 GB HBM3 than they did on the consumer 16 GB. Consider
  bumping `--batch-size` and seq-len for those once we know they work.

### Phase B — multi-GPU DDP adversarial (only on 8× nodes, ~2 h, $40)

```bash
nohup scripts/run_amd_datacenter_corpus.sh --reps 3 --skip-single-gpu \
    > data/collection_logs/dc_ddp.out 2>&1 &
```

This runs the 6 DDP families:
- `adversarial_K_ddp` — online learning, every rank every batch
- `adversarial_L_ddp_5` / `_10` — diluted training, ranks synchronized
- `adversarial_L_ddp_stagger_5` — per-rank dilution offsets
- `adversarial_K_ddp_accum_4` / `_8` — gradient-accumulation variants

Each is `torchrun --nproc_per_node=8 workloads/adversarial_ddp.py`. RCCL
should substitute for NCCL transparently. If it doesn't:
- Check `RCCL_DEBUG=INFO` output.
- Try `export NCCL_SOCKET_IFNAME=lo` or `NCCL_IB_DISABLE=1` if InfiniBand
  is finicky.
- Confirm `torch.distributed.is_available()` and `is_nccl_available()`
  return True under ROCm.

### Phase C — wall-power validation (optional, free)

The cloud provider almost certainly won't let you plug a Tapo P115
between PSU and wall. Skip on the rented instance; do this once on the
local box per Phase 0g.

## Pushing the data back

The git-lfs setup in `setup_amd_datacenter.sh` pulls existing LFS objects
on clone. New parquets created during the corpus run will be inline blobs
until we either:
- Run `git lfs migrate import` post-collection (same trick we ran on the
  local box, see commit `add66ce` and `f11fed7` on the amd branch), or
- Pre-configure LFS to track new files automatically.

The cleanest pattern, on the rented instance:

```bash
git lfs install
git lfs track 'data/*.parquet'   # idempotent if already in .gitattributes
# ... run collection, watcher commits parquets as LFS blobs ...
git push origin amd
```

The watcher script `scripts/commit_amd_parquets.sh` invokes `git add`
which goes through the LFS filter when `git lfs install` has been run.
**Confirm with `git lfs ls-files | wc -l` after the first commit** — if
the count goes up by 1, LFS is wired correctly. If not, kill the watcher,
fix LFS, restart.

## Cost summary

| Configuration | $/hr | Duration | Total |
|---|---|---|---|
| RunPod MI300X 1× (single-GPU sweep only) | $2.99 | 7h | **$21** |
| RunPod MI300X 8× (single + DDP) | $19.92 | 7h | **$140** |
| TensorWave MI325X 1× (HBM3e contrast) | $4.99 | 4h | **$20** |
| TensorWave MI300X 8× (DDP fallback) | $25.00 | 2h | **$50** |

**Recommended initial spend:** ~$160 across MI300X 8× + MI325X 1×. If
budget tight, $20 just for MI300X 1× gets the consumer→datacenter
comparison story.

## Risks

- **MI300X 8× community-cloud availability** fluctuates daily. If
  blocked, fall back to MI300X 1× and accept that the DDP corpus is
  pending.
- **ROCm version mismatch.** If RunPod's image has ROCm 6.0 (older),
  `setup_amd_datacenter.sh` upgrades to 6.4 from AMD's repo. That can
  take 15+ min and occasionally breaks PyTorch ROCm wheels. Fall-back:
  use whatever ROCm the image has, install PyTorch via
  `pip install --index-url https://download.pytorch.org/whl/rocm$(rocminfo | grep -oP 'Runtime Ext Version:\s+\K[0-9.]+' | head -1) torch torchvision`.
- **amdsmi-py / libamd-smi version skew.** Same problem we hit locally
  (pip wheel 7.0.2 vs library 7.2.0). Setup script clones AMD's amdsmi
  repo at the tag matching the installed library version. If the tag
  doesn't exist, falls back to mainline.
- **NCCL → RCCL substitution failures** in PyTorch DDP. If
  `torch.distributed.init_process_group(backend='nccl')` fails on ROCm,
  try `backend='rccl'` explicitly, or set `TORCH_DISTRIBUTED_BACKEND=rccl`.

## Once data is collected

Re-run the cross-vendor analyses against the expanded corpus:

```bash
python classifier/eval_cross_vendor.py
python classifier/eval_whitebox_amd.py
python regen_crossgpu_heatmap_binary_rf_amd.py
```

Expected outcomes (predictions):
- **Cross-tier AMD transfer** weak, mirroring NVIDIA: RX 9070 XT →
  MI300X probably 40–70%, MI300X → RX 9070 XT similar.
- **Same-tier cross-vendor** strong: MI300X ↔ A100/H100 likely ≥90%
  (datacenter parts share thermal/power envelopes).
- **WB-L detection on MI300X** uncertain — paper's NVIDIA-only WB-L LOO
  ranges 43–87%. On a hardened cross-vendor classifier trained with
  MI300X data included, may rise back into that range.
