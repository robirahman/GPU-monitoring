# Collecting the missing H200 runs

Everything needed to fill the H200 gap in the camera-ready corpus, on a machine with one H200. **5 cells x 3 reps = 15 runs, about 1.2 h of workload.**

One GPU is enough. None of these cells is a `torchrun` job -- the only reason this cannot run on Google Cloud is that GCP sells H200 exclusively in 8-GPU nodes, and our quota request for one was denied (support case 76886544).

## 0. The stack

H200 is Hopper (sm_90), so the corpus pin in `requirements_v2.txt` -- torch 2.5.1+cu121 -- runs on it unchanged. Use it: matching the stack the rest of the corpus was collected under is worth more than a newer torch.

## 1. Clone

```bash
# LFS skip matters: the corpus is ~12k parquet payloads this box
# never reads, and smudging them is a multi-GB download.
GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \
    https://github.com/robirahman/GPU-monitoring.git
cd GPU-monitoring
```

## 2. Install

```bash
python3 -m venv venv && . venv/bin/activate
pip install --upgrade pip
pip install -r requirements_v2.txt

# CIFAR-10, so the first run does not spend its 300 s downloading
python -c "import torchvision; torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True)"
```

## 3. Preflight -- do not skip

```bash
# Refuses to go further unless the device is the right one AND can run a
# kernel. A Blackwell card with a cu121 wheel imports fine and fails only when
# something touches the GPU, which would otherwise be 40 minutes from now.
python - <<'PY'
import sys, torch
want = "NVIDIA H200"
assert torch.cuda.is_available(), "CUDA not available"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device      : {name}")
print(f"capability  : sm_{cap[0]}{cap[1]}")
print(f"torch       : {torch.__version__}")
print(f"torch cuda  : {torch.version.cuda}")
print(f"arch list   : {torch.cuda.get_arch_list()}")
if want not in name:
    sys.exit(f"FAIL: expected {want!r}, found {name!r}. The corpus keys on "
             f"gpu_name, so this would land as a different device.")
x = torch.randn(2048, 2048, device="cuda")
(x @ x).sum().item()          # the actual test: can it launch a kernel?
torch.cuda.synchronize()
print("preflight OK")
PY
```

## 4. Collect

```bash
export STAMP=$(date -u +%Y%m%d)a
mkdir -p data logs

CELLS="adversarial_K_online_learning_mw8 adversarial_L_diluted_10_mw8 adversarial_L_diluted_20_mw8 adversarial_L_diluted_2_mw8 adversarial_L_diluted_5_mw8"
for cell in $CELLS; do
  for rep in 1 2 3; do
    log="logs/${STAMP}__$(hostname -s)__${cell#adversarial_}_rep${rep}.log"
    # idempotent: a cell with a clean log is skipped, so an
    # interrupted session resumes instead of redoing.
    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
    python scripts/run_workload.py --workload "$cell" \
      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
  done
done
```

`--require-gpu-activity` exits 5 if a run aborts, stops short, or leaves the GPU idle. Every job this file exists to replace was produced by a run that exited cleanly and collected nothing, so let it fail loudly rather than filing another one.

## 5. Check before shipping

```bash
# Every cell here emits STEP lines. A run with steps=0 is the
# exact failure this collection exists to undo -- do not ship it.
for f in logs/${STAMP}__*.log; do
  printf "%-52s steps=%-7s %s\n" "$(basename "$f")" \
    "$(grep -c '^STEP ' "$f")" "$(grep -o 'SPAR_CELL_DONE.*' "$f")"
done

# and the collector's own verdict, written beside each parquet
grep -h '"verdict"' data/*.health.json | sort | uniq -c
```

Expected: 15 logs, every one with a non-zero step count and `exit=0`, and 15 health verdicts all `ok`.

## 6. Ship it back

```bash
git add data/*.parquet data/*.health.json logs/${STAMP}__*.log
git commit -m "Collect H200 re-runs, stamp ${STAMP}"
git push origin neurips-camera-ready
```

If pushing parquets through LFS from a rented box is awkward, put them in the bucket instead and the analysis box will ingest:

```bash
gcloud storage cp data/*.parquet \
  gs://focus-surfer-494820-g0-telemetry/runs/${STAMP}/$(hostname -s)/data/
gcloud storage cp logs/${STAMP}__*.log \
  gs://focus-surfer-494820-g0-telemetry/runs/${STAMP}/$(hostname -s)/logs/
```

## Why five cells are named `_mw8`

`adversarial_L_diluted_*` and `adversarial_K_online_learning` appear as `_mw8` variants. Re-running the originals just reproduces the fault: `num_workers=0` puts `Resize(224)` on one CPU thread and the GPU waits, which is how the archived runs managed about four batches in 324 s with the H200 busy in under 2% of samples. The `_mw8` cells pass `--num-workers 8` and are GPU-bound.

That makes them a **different workload**, which is why they carry a different label. Do not pool them with the `num_workers=0` runs, and do not delete the originals -- those are what the camera-ready measured.

