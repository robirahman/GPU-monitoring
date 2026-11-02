# Collecting the missing A100 80GB runs on Vast.ai

**Rent: 8x NVIDIA A100 SXM4 80GB with NVLink.** 2 cells x 3 reps = 6 runs,
about 1.2 h of workload plus setup.

This exists because Google Cloud cannot supply the node. We hold
`NVIDIA_A100_80GB_GPUS = 16` on-demand in five regions, so quota is not the
problem -- every `a2-ultragpu-8g` creation in all eight A100-80GB zones has
returned `ZONE_RESOURCE_POOL_EXHAUSTED` continuously since 2026-10-29, checked
again on 2026-11-02. And A100 80GB is the one family with
`PREEMPTIBLE_NVIDIA_A100_80GB_GPUS = 0`, so there is no Spot or Flex Start
queue to fall back on (unlike B200 and H200, which have 64 each). Support case
77159076 asks for that quota; this doc is the route that does not wait on it.

## Pick the right instance -- this is the part that goes wrong

The corpus contains exactly two A100 device strings: `NVIDIA A100-SXM4-40GB`
and `NVIDIA A100-SXM4-80GB`. **You need SXM4, not PCIe.** A PCIe A100 reports
`NVIDIA A100 80GB PCIe`, and since the corpus keys on `gpu_name` it would land
as a new device rather than repairing the one we are missing.

Both cells are named `_nvlink` and are measuring the interconnect, so this is
not pedantry: on a PCIe box there is no NVLink to measure and the number would
mean something else.

When filtering on Vast:

- **GPU**: A100 SXM4 80GB (Vast usually lists this as `A100 SXM4 80GB`; avoid
  anything saying `PCIE`)
- **Count**: 8x on one machine, not 8 separate machines
- **NVLink / NVSwitch**: required
- **Disk**: 150 GB is ample
- The preflight below refuses to continue if any of this is wrong, so a bad
  pick costs minutes rather than an hour.

## 1. Clone and install

```bash
# LFS skip matters: the corpus is ~12k parquet payloads this box never reads.
GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \
    https://github.com/robirahman/GPU-monitoring.git
cd GPU-monitoring

python3 -m venv venv && . venv/bin/activate
pip install --upgrade pip
pip install -r requirements_v2.txt      # torch 2.5.1+cu121; A100 is sm_80, fine
pip install psutil                      # so the run records CPU, cores and RAM

export STAMP=$(date -u +%Y%m%d)a
mkdir -p data logs
```

No CIFAR needed: `fsdp_training.py` generates its own synthetic batches.

## 2. Preflight -- do not skip

```bash
python - <<'PY'
import sys, torch
assert torch.cuda.is_available(), "CUDA not available"
n = torch.cuda.device_count()
name = torch.cuda.get_device_name(0)
print("devices:", n)
print("name   :", name)
print("torch  :", torch.__version__, "| cuda", torch.version.cuda)
if "A100-SXM4-80GB" not in name:
    sys.exit(f"FAIL: need 'NVIDIA A100-SXM4-80GB', got {name!r}. "
             "A PCIe A100 is a different device in this corpus.")
if n < 8:
    sys.exit(f"FAIL: these cells are torchrun --nproc_per_node=8; found {n} GPUs")
# NVLink must actually be present, not just advertised
import subprocess
topo = subprocess.run(["nvidia-smi", "topo", "-m"], capture_output=True, text=True).stdout
if "NV" not in topo:
    sys.exit("FAIL: nvidia-smi topo -m shows no NVLink between GPUs")
print("nvlink :", "present")
x = torch.randn(4096, 4096, device="cuda"); (x @ x).sum().item()
torch.cuda.synchronize()
print("preflight OK")
PY
```

## 3. Collect

```bash
CELLS="fsdp_30b_nvlink fsdp_70b_nvlink"
for cell in $CELLS; do
  for rep in 1 2 3; do
    log="logs/${STAMP}__$(hostname -s)__${cell}_rep${rep}.log"
    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
    python scripts/run_workload_multi_gpu.py --workload "$cell" --num-gpus 8 \
      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
    echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=${PIPESTATUS[0]}" >> "$log"
  done
done
```

Each run is `--duration 600`, so expect ~10-11 minutes per rep and roughly
1.2 h for all six.

`llm_infer_mistral_7b_q4` is the third A100 cell and is **not** in this list --
it is single-GPU and is being collected on Google Cloud separately. Do not run
it here or you will produce a duplicate.

## 4. Both cells are known to work here

Collected on 2026-11-02 on an 8x A100-SXM4-80GB node, three reps each:

| cell | span | busy | mean util | steps |
| :-- | --: | --: | --: | --: |
| `fsdp_30b_nvlink` | 619-621 s | 0.97 | 96.1-96.3% | 414 |
| `fsdp_70b_nvlink` | 620-622 s | 0.98 | 96.5-96.9% | 189 |

An earlier revision of this file told you to expect `fsdp_70b_nvlink` to OOM,
reasoning that AdamW fp32 moments beside bf16 parameters come to about 560 GB
against 640 GB of HBM, and to record the failure as a hardware limit. That was
wrong: it runs, at 98% busy for the full ten minutes. The corpus's single A100
70b run -- 24 s, zero windows -- was a collection failure like the others, not
evidence about the hardware.

So a failure of either cell here means the setup is wrong, not the GPU.

## 5. Check, then ship

```bash
for f in logs/${STAMP}__*.log; do
  printf "%-54s steps=%-7s %s\n" "$(basename "$f")" \
    "$(grep -c '^STEP ' "$f")" "$(grep -o 'SPAR_CELL_DONE.*' "$f")"
done
grep -h '"verdict"' data/*.health.json | sort | uniq -c
ls data/fsdp_*nvlink*.parquet | wc -l     # expect 8 per successful run
```

A successful 8-GPU run writes **8 parquets**, one per rank. Those filenames
carry a run id and no date (`fsdp_30b_nvlink_node0_gpu0_NVIDIA_A100-SXM4-80GB_<runid>.parquet`),
which has caught me out twice when fetching them later -- mention the commit
hash when you push so they can be pulled by commit rather than by date.

```bash
git add data/*.parquet data/*.health.json logs/${STAMP}__*.log
git commit -m "Collect A100 80GB FSDP re-runs, stamp ${STAMP}"
git push origin neurips-camera-ready
```

Then on the analysis box:

```bash
PYTHONPATH=. python3 scripts/audit_run_health.py --write   # idempotent
```
