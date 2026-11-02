# Collecting the missing H100 NVL runs

**Rent: 1x H100 **NVL** (94GB PCIe).** 1 cells x 3 reps = 3 runs.

NVL specifically, not SXM 80GB: they report different `gpu_name` and would land as different devices.

These replace tier-A jobs that aborted, stopped short, or ran with the GPU idle. Under the policy recorded in `data/tier_a_recollections.json`, a re-collection of the same cell on the same `gpu_name` can be promoted back into tier A, so this is repairing the frozen corpus and not only its coverage claims.

## 1. Clone and install

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \
    https://github.com/robirahman/GPU-monitoring.git
cd GPU-monitoring
python3 -m venv venv && . venv/bin/activate
pip install --upgrade pip
pip install -r requirements_v2.txt
pip install psutil   # so the run records its CPU, cores and RAM
python -c "import torchvision; torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True)"

# Do NOT add requirements_gptq.txt to this venv. On 2026-11-02 it
# pulled torch 2.14.1 over the pinned 2.5.1 and gptqmodel still did
# not import. The quantized cell runs without it -- see the note
# at the end of this file if it ever does not.
export STAMP=$(date -u +%Y%m%d)a
mkdir -p data logs
```

## 2. Preflight

```bash
python - <<'PY'
import sys, torch
want = 'H100 NVL'
forbid = 'H200'
assert torch.cuda.is_available(), 'CUDA not available'
name = torch.cuda.get_device_name(0)
print('device :', name)
print('torch  :', torch.__version__, '| cuda', torch.version.cuda)
print('arch   :', torch.cuda.get_arch_list())
if want not in name or (forbid and forbid in name):
    sys.exit(f'FAIL: need a device named like {want!r}, got {name!r}')
x = torch.randn(2048, 2048, device='cuda'); (x @ x).sum().item()
torch.cuda.synchronize(); print('preflight OK')
PY
```

## 3. The 1 single-GPU cells

```bash
CELLS="pytorch_resnet_cifar10"
for cell in $CELLS; do for rep in 1 2 3; do
  log="logs/${STAMP}__$(hostname -s)__${cell#adversarial_}_rep${rep}.log"
  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
  python scripts/run_workload.py --workload "$cell" \
    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=${PIPESTATUS[0]}" >> "$log"
done; done
```

## 5. Check, then ship

```bash
for f in logs/${STAMP}__*.log; do
  printf "%-52s steps=%-7s %s\n" "$(basename "$f")" \
    "$(grep -c '^STEP ' "$f")" "$(grep -o 'SPAR_CELL_DONE.*' "$f")"
done
grep -h '"verdict"' data/*.health.json | sort | uniq -c

git add data/*.parquet data/*.health.json logs/${STAMP}__*.log
git commit -m "Collect H100 NVL re-runs, stamp ${STAMP}"
git push origin neurips-camera-ready
```

Every health verdict should read `ok`. Not every cell emits STEP lines -- only the instrumented adversarial strategies do -- so `steps=0` on an inference or baseline cell is expected, but `verdict` other than `ok` is not.

