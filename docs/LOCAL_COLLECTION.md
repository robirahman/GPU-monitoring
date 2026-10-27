# Collecting on a machine that is not in Google Cloud

For the GPUs this project cannot reach: the RTX 4050 laptop, the RTX 5080
desktop, H100 NVL and anything rented elsewhere. Everything here runs from a
clone of `neurips-camera-ready`.

The list it works from is
[`recollection_list.csv`](camera_ready_results_20261022/recollection_list.csv),
which `scripts/audit_run_health.py` regenerates. 46 tier-A adversarial jobs are
unusable: 19 `aborted` (stopped seconds in with the GPU never active),
12 `truncated` (stopped early with the GPU working) and 15 `starved` (ran to
length with the GPU idle in 99% of samples). That audit only covers adversarial
cells; non-adversarial runs found broken by hand are listed in
[`recollection_extra.csv`](camera_ready_results_20261022/recollection_extra.csv)
and appear in the blocks below alongside them.

## One-time setup

```bash
# LFS skip matters: the corpus is ~12k parquet payloads you do not need in
# order to collect, and pulling them is a multi-GB download.
GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \
    https://github.com/robirahman/GPU-monitoring.git
cd GPU-monitoring

python3 -m venv venv && . venv/bin/activate
pip install -r requirements_v2.txt          # pinned torch 2.5.1+cu121, Linux only
# AMD instead:  pip install -r requirements_amd.txt

# CIFAR-10, so the first run does not spend its 300 s downloading
python -c "import torchvision; torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True)"

export STAMP=$(date -u +%Y%m%d)a
export HOST=$(hostname | cut -d. -f1)       # not `hostname -s`: Git Bash's hostname has no -s
mkdir -p data logs
```

**Windows with an NVIDIA GPU** (the RTX 4050 laptop). Use Git Bash, not
PowerShell -- the blocks below are bash. `requirements_v2.txt` will not
install: it pins `triton` and the `nvidia-*-cu12` wheels, which have no Windows
build, and it lacks the Hugging Face packages. Replace the venv lines with:

```bash
python -m venv venv && . venv/Scripts/activate
pip install -r requirements_windows_nvidia.txt
```

and fetch the models the inference and fine-tuning cells load, so no run spends
its timeout downloading (that is how `deepseek_r1_inference` and the first reps
of `bert_sst2` and `gpt2_wikitext2` were lost on the laptop):

```bash
python -c "import whisper; whisper.load_model('large-v3-turbo', device='cpu')"
python -c "from transformers import AutoModelForCausalLM as M, AutoTokenizer as T; [f.from_pretrained('deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B') for f in (M, T)]"
python -c "from transformers import AutoModel as M, AutoTokenizer as T; [f.from_pretrained(n) for n in ('gpt2', 'bert-base-uncased') for f in (M, T)]"
python -c "import datasets; datasets.load_dataset('wikitext', 'wikitext-2-raw-v1'); datasets.load_dataset('glue', 'sst2')"
```

Collect with `scripts/run_workload.py` on Windows too: it reads NVML and takes
`--require-gpu-activity`, which `scripts/run_workload_windows.py` (the
DirectML/AMD path) does not. On AMD Linux use `scripts/run_workload_amd.py`.

Check the GPU is visible before starting: `python -c "import torch;
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`.

## What to collect, by device

Each block is idempotent -- a cell that already has a clean log is skipped, so
re-running the block after an interruption resumes rather than redoing.

`--require-gpu-activity` is the point of the exercise. It exits 5 if the run
aborts, truncates, or leaves the GPU idle, which is exactly how all 46 of these
jobs were produced in the first place. Let it fail loudly rather than filing
another run that looks fine until someone counts the gradient steps six months
later.

### A100-SXM4-80GB

The five starved cells were collected on GCP as `_mw8` on 2026-10-26 (stamp `20261026a`); the four aborted cells still need a run.

4 aborted, 5 starved. **9 cells x 3 reps = 27 runs**, about 2.5 h.

```bash
CELLS="adversarial_F_interleave_30 adversarial_H_mimicry_cufft adversarial_H_mimicry_mining adversarial_J_pid_75 adversarial_K_online_learning_mw8 adversarial_L_diluted_10_mw8 adversarial_L_diluted_20_mw8 adversarial_L_diluted_2_mw8 adversarial_L_diluted_5_mw8"
for cell in $CELLS; do
  for rep in 1 2 3; do
    log="logs/${STAMP}__${HOST}__${cell#adversarial_}_rep${rep}.log"
    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
    python scripts/run_workload.py --workload "$cell" \
      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "SPAR_CELL_DONE cell=${cell} rep=${rep} exit=${rc} utc=$(date -u +%FT%TZ)" | tee -a "$log"
  done
done
```

### B200

**Blocked on quota.** B200 has none in this project and increase requests for us-central1 and us-east4 were auto-denied on 2026-10-26. Needs a support case, or a non-GCP provider.

6 aborted, 5 starved, 2 truncated. **12 cells x 3 reps = 36 runs**, about 3.3 h.

```bash
CELLS="adversarial_E_memory_minimal adversarial_F_interleave_50 adversarial_G_clock_throttled adversarial_H_mimicry_cufft adversarial_I_stochastic_30 adversarial_J_pid_75 adversarial_K_online_learning_mw8 adversarial_L_diluted_10_mw8 adversarial_L_diluted_20_mw8 adversarial_L_diluted_2_mw8 adversarial_L_diluted_5_mw8 adversarial_training"
for cell in $CELLS; do
  for rep in 1 2 3; do
    log="logs/${STAMP}__${HOST}__${cell#adversarial_}_rep${rep}.log"
    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
    python scripts/run_workload.py --workload "$cell" \
      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "SPAR_CELL_DONE cell=${cell} rep=${rep} exit=${rc} utc=$(date -u +%FT%TZ)" | tee -a "$log"
  done
done
```

### GeForce RTX 4050 Laptop GPU

Local laptop, Windows; run the block from Git Bash with the venv active. Every adversarial run on this device aborted within 6-7 s, so it has no usable adversarial coverage at all, and so did every CIFAR training run and most of the inference cells. [`recollection_extra.csv`](camera_ready_results_20261022/recollection_extra.csv) lists the non-adversarial ones and their likely causes. `bert_sst2` and `gpt2_wikitext2` each lost one rep to a first-run download; all three reps are re-collected so the set comes from one session.

Plug in the charger and set Windows to never sleep before starting. Blender has to be on `PATH`:

```bash
export PATH="/c/Program Files/Blender Foundation/Blender 5.1:$PATH"
blender --version | head -1
```

15 aborted, 2 idle_prefix, 1 starved, 1 truncated. **19 cells x 3 reps = 57 runs**, about 4.1 h.

```bash
CELLS="adversarial_A_util_modulation adversarial_B_low_util adversarial_D_temporal_disruption adversarial_composite_00 adversarial_composite_30 adversarial_composite_50 adversarial_composite_70 adversarial_composite_90 adversarial_composite_95 bert_sst2 blender_bmw deepseek_r1_inference ethash_cuda gpt2_wikitext2 pytorch_mlp_cifar10 pytorch_resnet_cifar10 pytorch_resnet_cifar10_amp resnet50_inference whisper_inference"
for cell in $CELLS; do
  case "$cell" in
    adversarial_composite_*) extra="--timeout 240 -- python workloads/adversarial_composite.py --idle-fraction 0.${cell##*_} --duration 120 --prealloc-mb 2048" ;;
    gpt2_wikitext2) extra="--timeout 1200 -- python workloads/gpt2_finetune.py --steps 400 --batch-size 2" ;;
    bert_sst2) extra="--timeout 900 -- python workloads/bert_finetune.py --steps 300 --batch-size 8" ;;
    ethash_cuda) extra="--timeout 360 -- python workloads/ethash_cuda.py --duration 300 --dag-size 256" ;;
    *) extra="" ;;
  esac
  for rep in 1 2 3; do
    log="logs/${STAMP}__${HOST}__${cell#adversarial_}_rep${rep}.log"
    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
    python scripts/run_workload.py --workload "$cell" \
      --output-dir data/ --no-dcgm --require-gpu-activity $extra 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "SPAR_CELL_DONE cell=${cell} rep=${rep} exit=${rc} utc=$(date -u +%FT%TZ)" | tee -a "$log"
  done
done
```

### H100 NVL

Not a GCP SKU. Whichever provider ran the originals.

10 truncated. **5 cells x 3 reps = 15 runs**, about 1.4 h.

```bash
CELLS="adversarial_composite_10 adversarial_composite_20 adversarial_composite_40 adversarial_composite_60 adversarial_composite_80"
for cell in $CELLS; do
  for rep in 1 2 3; do
    log="logs/${STAMP}__${HOST}__${cell#adversarial_}_rep${rep}.log"
    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
    python scripts/run_workload.py --workload "$cell" \
      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "SPAR_CELL_DONE cell=${cell} rep=${rep} exit=${rc} utc=$(date -u +%FT%TZ)" | tee -a "$log"
  done
done
```

### H200

**Blocked on quota.** Same as B200: requests for us-central1 and us-east4 auto-denied on 2026-10-26.

5 starved. **5 cells x 3 reps = 15 runs**, about 1.4 h.

```bash
CELLS="adversarial_K_online_learning_mw8 adversarial_L_diluted_10_mw8 adversarial_L_diluted_20_mw8 adversarial_L_diluted_2_mw8 adversarial_L_diluted_5_mw8"
for cell in $CELLS; do
  for rep in 1 2 3; do
    log="logs/${STAMP}__${HOST}__${cell#adversarial_}_rep${rep}.log"
    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue
    python scripts/run_workload.py --workload "$cell" \
      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"
    rc=${PIPESTATUS[0]}
    echo "SPAR_CELL_DONE cell=${cell} rep=${rep} exit=${rc} utc=$(date -u +%FT%TZ)" | tee -a "$log"
  done
done
```

## Why the starved cells are collected under a different name

`adversarial_L_diluted_2` and its siblings are in the list as `_mw8` variants,
not as themselves. Re-running the original command reproduces the starvation:
`num_workers=0` puts `Resize(224)` on one CPU thread and the GPU waits, and
the archived runs did about four batches in 350 s. The `_mw8` cells pass
`--num-workers 8` and are GPU-bound.

That makes them a **different workload**, which is why they carry a different
label -- `adversarial_round3.py` says so in its own docstring. Do not pool them
with the `num_workers=0` runs, and do not delete the originals: they are what
the camera-ready measured.

## Before you ship it back

```bash
# every run should report health=ok and exit=0
for f in logs/${STAMP}__*.log; do
  printf "%-62s health=%-10s steps=%-7s %s\n" "$(basename "$f")" \
    "$(grep -o 'Run health: [a-z_]*' "$f" | tail -1 | cut -d' ' -f3)" \
    "$(grep -c '^STEP ' "$f")" "$(grep -o 'SPAR_CELL_DONE.*' "$f" | tail -1)"
done
```

`health` is the collector's verdict on the telemetry (it also writes it to a
`.health.json` beside each parquet). A run that is not `ok`, or that exited
non-zero, is the failure this whole exercise is about. Do not ship it; find
out why first. Read `exit` as well as `health`: a workload with no
`--duration` in its command (Blender, the fine-tuning cells) is only checked
for starvation, so one that crashes after a few busy seconds still reads
`health=ok` -- but exits non-zero.

`steps` counts `STEP` lines, which only some workloads print:
`adversarial_round2`/`round3`/`advanced`/`symgemm`/`whitebox` do;
`adversarial_training.py` (strategies A/B/D), `adversarial_composite.py` and
the non-adversarial workloads do not, so `steps=0` is expected for every laptop
cell. Where a workload does print them, zero is a failure.

## Shipping it back

Either push the branch:

```bash
# only the parquets of this session's runs that exited 0 (with
# --require-gpu-activity that implies health=ok), plus their health files
OK=$(for f in logs/${STAMP}__*.log; do
       grep -q "SPAR_CELL_DONE.*exit=0" "$f" &&
         grep -o 'Output file: .*' "$f" | tail -1 | sed 's#.*[\\/]#data/#; s/\r$//'
     done)
git add $OK ${OK//.parquet/.health.json} logs/${STAMP}__*.log
git commit -m "Collect <device> re-runs, stamp ${STAMP}"
git push origin neurips-camera-ready
```

or, if you would rather not put parquets through LFS from a laptop, drop them
in the bucket and let the analysis box ingest:

```bash
gcloud storage cp data/*.parquet \
  gs://focus-surfer-494820-g0-telemetry/runs/${STAMP}/${HOST}/data/
gcloud storage cp logs/${STAMP}__*.log \
  gs://focus-surfer-494820-g0-telemetry/runs/${STAMP}/${HOST}/logs/
```

Then on the analysis box:

```bash
PYTHONPATH=. python3 scripts/ingest_rerun_references.py ${STAMP}
PYTHONPATH=. python3 scripts/audit_run_health.py          # re-verify, idempotent
```

`audit_run_health.py` leaves already-recorded exclusions alone, so re-running
it after a collection only adds what is new.
