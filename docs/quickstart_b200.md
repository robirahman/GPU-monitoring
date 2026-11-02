# Collecting the missing B200 runs

**Rent: 1x B200.** 4 cells x 3 reps = 12 runs.

Needs `requirements_blackwell.txt` (cu128). The repo's default pin is torch 2.5.1+cu121, which carries no sm_100 kernels.

`adversarial_G_clock_throttled` calls `nvidia-smi -pl` and `--lock-gpu-clocks`, which need privileged access. It was skipped on the last B200 rental for exactly that reason. Take a bare-metal or privileged instance if you want it; otherwise expect it to fail and leave the other three cells unaffected.

These replace tier-A jobs that aborted, stopped short, or ran with the GPU idle. Under the policy recorded in `data/tier_a_recollections.json`, a re-collection of the same cell on the same `gpu_name` can be promoted back into tier A, so this is repairing the frozen corpus and not only its coverage claims.

## 1. Clone and install

```bash
GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \
    https://github.com/robirahman/GPU-monitoring.git
cd GPU-monitoring
python3 -m venv venv && . venv/bin/activate
pip install --upgrade pip
pip install -r requirements_blackwell.txt   # cu128, sm_100
pip install psutil   # so the run records its CPU, cores and RAM
python -c "import torchvision; torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True)"
export STAMP=$(date -u +%Y%m%d)a
mkdir -p data logs
```

## 2. Preflight

```bash
python - <<'PY'
import sys, torch
want = 'B200'
forbid = ''
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

## 3. The 4 single-GPU cells

```bash
CELLS="adversarial_G_clock_throttled cufft_benchmark nbody_sim resnet50_inference"
for cell in $CELLS; do for rep in 1 2 3; do
  log="logs/${STAMP}__$(hostname -s)__${cell#adversarial_}_rep${rep}.log"
  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
  python scripts/run_workload.py --workload "$cell" \
    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=${PIPESTATUS[0]}" >> "$log"
done; done
```

## Already collected on B200 -- do not re-run these

| cell | runs already collected |
| :-- | --: |
| `adversarial_E_memory_minimal` | 3 |
| `adversarial_F_interleave_50` | 3 |
| `adversarial_H_mimicry_cufft` | 3 |
| `adversarial_I_stochastic_30` | 3 |
| `adversarial_J_pid_75` | 3 |
| `adversarial_K_online_learning_mw8` | 3 |
| `adversarial_L_diluted_10_mw8` | 3 |
| `adversarial_L_diluted_20_mw8` | 3 |
| `adversarial_L_diluted_2_mw8` | 3 |
| `adversarial_L_diluted_5_mw8` | 3 |

## Cannot be reproduced on B200

`nvlink_latency`, `nvlink_p2p` have no registry entry, so there is no command to run them. Gaps to state in the paper, not work for this rental.

## 5. Check, then ship

```bash
for f in logs/${STAMP}__*.log; do
  printf "%-52s steps=%-7s %s\n" "$(basename "$f")" \
    "$(grep -c '^STEP ' "$f")" "$(grep -o 'SPAR_CELL_DONE.*' "$f")"
done
grep -h '"verdict"' data/*.health.json | sort | uniq -c

git add data/*.parquet data/*.health.json logs/${STAMP}__*.log
git commit -m "Collect B200 re-runs, stamp ${STAMP}"
git push origin neurips-camera-ready
```

Every health verdict should read `ok`. Not every cell emits STEP lines -- only the instrumented adversarial strategies do -- so `steps=0` on an inference or baseline cell is expected, but `verdict` other than `ok` is not.

