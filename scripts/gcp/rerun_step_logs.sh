#!/usr/bin/env bash
# Re-run the camera-ready adversarial cells WITH per-step logging.
#
# WHY. STEP logging landed on 2026-07-24 (8ee55e53), after every camera-ready
# run had been collected, so those cells have no per-window ground truth. This
# re-runs them unchanged -- same registry command, same pinned torch -- on the
# same GPU model, so each one yields a logged sibling whose batch rate lets
# classifier/schedule_reconstruction.py place the gradient steps of the
# archived run.
#
# It is NOT a replacement for the archived runs and must not be pooled with
# them: it is a reference measurement taken on today's host.
#
# Idempotent by design. Every cell/rep writes one log and skips if that log is
# already non-empty, so a re-invocation after an interruption resumes rather
# than redoing -- which matters because the node cannot be killed remotely.
#
#   sudo STAMP=20261023a bash /opt/spar/gcp/rerun_step_logs.sh
set -uo pipefail

REPO="${REPO_DIR:-/opt/spar/repo}"
STAMP="${STAMP:?export STAMP, e.g. 20261023a}"
BUCKET="${BUCKET:-gs://focus-surfer-494820-g0-telemetry}"
REPS="${REPS:-3}"
STOP_WHEN_DONE="${STOP_WHEN_DONE:-1}"
# single | ddp | all. 8x A100-80GB capacity is scarce, so the eleven 1-GPU
# cells run on an a2-ultragpu-1g that lands easily while the six torchrun
# cells wait for an 8g node, instead of all seventeen waiting together.
CELL_SET="${CELL_SET:-all}"
# Optional extended-regex filter over cell names. Only some strategies emit
# STEP lines today, so the instrumented ones run first and the rest follow once
# their generator has been taught to log.
ONLY_CELLS="${ONLY_CELLS:-}"
HOST="$(hostname -s)"
PY="${REPO}/venv-main/bin/python"
GCS="${BUCKET}/runs/${STAMP}/${HOST}"
say() { echo "[$(date -u '+%H:%M:%SZ')] [rerun] $*"; }

cd "$REPO" || { echo "no repo at $REPO" >&2; exit 1; }

# ONE campaign per node, enforced. Two of these running at once put two
# workloads on the same GPU, and the telemetry of both is then a measurement
# of neither -- which is exactly what happened on 2026-10-23T01:43Z when a
# driver re-launched this while the first pass was still going. The lock is
# held for the whole campaign and released when the process exits, so a second
# invocation exits immediately rather than queueing behind the first.
exec 9>/var/lock/spar_rerun.lock
if ! flock -n 9; then
  echo "[rerun] another campaign already holds /var/lock/spar_rerun.lock -- refusing to share the GPU" >&2
  exit 75
fi
mkdir -p data logs meta

# Cells needing a logged reference on A100-SXM4-80GB, from
# docs/camera_ready_results_20261022/step_log_rerun_plan.csv.
SINGLE=(
  adversarial_E_memory_minimal adversarial_F_interleave_30 adversarial_F_interleave_50
  adversarial_G_clock_throttled adversarial_H_mimicry_cufft adversarial_H_mimicry_mining
  adversarial_I_stochastic_30 adversarial_J_pid_75 adversarial_K_online_learning
  adversarial_L_diluted_2 adversarial_L_diluted_5
)
# torchrun --nproc_per_node=8; the archived runs used 8 GPUs, and the rate only
# transfers at the same width.
DDP=(
  adversarial_K_ddp adversarial_K_ddp_accum_4 adversarial_K_ddp_accum_8
  adversarial_L_ddp_10 adversarial_L_ddp_5 adversarial_L_ddp_stagger_5
)

sync_out() {
  gcloud storage rsync -r data "${GCS}/data" >/dev/null 2>&1 || true
  gcloud storage rsync -r logs "${GCS}/logs" >/dev/null 2>&1 || true
}

run_cell() {  # run_cell <cell> <rep> <mode>
  local cell="$1" rep="$2" mode="$3"
  if [ -n "$ONLY_CELLS" ] && ! printf '%s' "$cell" | grep -Eq "$ONLY_CELLS"; then return 0; fi
  local log="logs/${STAMP}__${HOST}__${cell}_rep${rep}.log"
  if [ -s "$log" ] && grep -q "SPAR_CELL_DONE" "$log"; then
    say "skip ${cell} rep${rep} (already complete)"; return 0
  fi
  say "run ${cell} rep${rep} (${mode})"
  {
    echo "SPAR_CELL_BEGIN cell=${cell} rep=${rep} stamp=${STAMP} host=${HOST} utc=$(date -u +%FT%TZ)"
    if [ "$mode" = ddp ]; then
      "$PY" scripts/run_workload_multi_gpu.py --workload "$cell" --num-gpus 8 \
            --output-dir data/ --no-dcgm 2>&1
    else
      "$PY" scripts/run_workload.py --workload "$cell" --gpu-index 0 \
            --output-dir data/ --no-dcgm 2>&1
    fi
    echo "SPAR_CELL_DONE cell=${cell} rep=${rep} exit=$? utc=$(date -u +%FT%TZ)"
  } 2>&1 | tee "$log"
  sync_out
}

say "campaign ${STAMP} on ${HOST}: cell set '${CELL_SET}', ${REPS} reps"
"$PY" -c 'import torch;print("torch",torch.__version__,"gpus",torch.cuda.device_count())' \
  | sed 's/^/[rerun] /'

for rep in $(seq 1 "$REPS"); do
  if [ "$CELL_SET" != ddp ]; then
    for cell in "${SINGLE[@]}"; do run_cell "$cell" "$rep" single; done
  fi
  if [ "$CELL_SET" != single ]; then
    for cell in "${DDP[@]}"; do run_cell "$cell" "$rep" ddp; done
  fi
done

# A manifest the analysis side can trust without re-deriving what ran.
"$PY" - "$STAMP" "$HOST" <<'PYEOF' > "meta/rerun_${STAMP}_${HOST}.json"
import json, os, re, subprocess, sys, glob
stamp, host = sys.argv[1], sys.argv[2]
logs = sorted(glob.glob(f"logs/{stamp}__{host}__*.log"))
cells = []
for p in logs:
    text = open(p, errors="replace").read()
    steps = len(re.findall(r"^STEP ts=", text, re.M))
    done = "SPAR_CELL_DONE" in text
    m = re.search(r"SPAR_CELL_BEGIN cell=(\S+) rep=(\d+)", text)
    cells.append(dict(log=os.path.basename(p), cell=m[1] if m else None,
                      rep=int(m[2]) if m else None, step_lines=steps, complete=done))
def sh(c):
    try: return subprocess.run(c, shell=True, capture_output=True, text=True).stdout.strip()
    except Exception: return ""
json.dump(dict(stamp=stamp, host=host, cells=cells,
               purpose="reference batch-rate measurement for schedule reconstruction; "
                       "NOT part of the camera-ready corpus",
               gpu=sh("nvidia-smi --query-gpu=name --format=csv,noheader | head -1"),
               driver=sh("nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1"),
               torch=sh(f"{os.environ.get('PY','')} -c 'import torch;print(torch.__version__)'"),
               cpu=sh("lscpu | sed -n 's/^Model name: *//p' | head -1"),
               machine_type=sh("curl -sf -H Metadata-Flavor:Google http://metadata.google.internal/computeMetadata/v1/instance/machine-type"),
               staged_commit=open(".spar_staged_commit").read().strip() if os.path.exists(".spar_staged_commit") else None,
               ), sys.stdout, indent=2)
PYEOF

gcloud storage rsync -r meta "${GCS}/meta" >/dev/null 2>&1 || true
sync_out
say "campaign complete: $(ls logs/${STAMP}__*.log 2>/dev/null | wc -l) logs, $(ls data/*.parquet 2>/dev/null | wc -l) parquets"
gcloud storage cp /dev/null "${GCS}/control/campaign.done" >/dev/null 2>&1 || true

if [ "$STOP_WHEN_DONE" = 1 ]; then
  say "powering off (the driver deletes the instance once GCS is verified)"
  shutdown -h +1
fi
