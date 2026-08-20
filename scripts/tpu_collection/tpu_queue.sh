#!/bin/bash
# Driver-side TPU collection queue -- fans each cell out to ALL workers of the
# slice over `gcloud compute tpus tpu-vm ssh --worker=all` and books it as done
# only when every worker's marker lands in GCS.
#
# Conventions inherited from the Round 6 GPU queue (round6_symgemm_queue.sh):
#   * cell name = registry label = parquet prefix = log = .done marker
#   * resumable: a completed cell drops logs/.done/<cell>.done and is skipped
#   * a failed cell is recorded (.failed) and the queue CONTINUES
#   * rep-major ordering so a truncated run still leaves a balanced design
#   * GAP idle settle between cells
#   * SMOKE=1 -> 60 s per cell, smoke_ prefix, output to logs/smoke_data/
#   * G1 gate: the CPU gradient self-test must pass before any TPU time is spent
#
# DECISIVE-CELL-FIRST within each rep (banks the key contrast inside one short
# Spot window before preemption can strike):
#   tpuT16_train_fsdp7b -> _infershard_7b -> _inferrep_7b -> _other_fft -> _idle
#
# The SCOPE is fixed by the provisioned slice (v6e-16 => tpuT16, 4 workers).
# Provision a smaller slice and set SCOPE/NUM_WORKERS to run a lower rung.
#
#   SMOKE=1 bash scripts/tpu_collection/tpu_queue.sh      # shakedown first
#   nohup   bash scripts/tpu_collection/tpu_queue.sh &    # real run
set -uo pipefail

# ── Config (override via env) ────────────────────────────────────────────────
PROJECT=${PROJECT:-focus-surfer-494820-g0}
ZONE=${ZONE:-us-east1-d}
TPU_NAME=${TPU_NAME:-spar-tpu-16}
REPO_DIR=${REPO_DIR:-\$HOME/SPAR-GPU-monitoring}   # path ON the worker
SCOPE=${SCOPE:-tpuT16}
NUM_WORKERS=${NUM_WORKERS:-4}                       # v6e-16 = 4 hosts
BUCKET=${BUCKET:-gs://spar-tpu-telemetry}
SESSION=${SESSION:-$(date -u +%Y%m%d_%H%M%S)}
GCS=${BUCKET}/${SESSION}
STAMP=${STAMP:-$SESSION}
DUR=${DUR:-600}
GAP=${GAP:-60}
REPS=${REPS:-3}
SMOKE=${SMOKE:-0}
STRETCH=${STRETCH:-0}          # 1 => also run the 1b + 30b cells

SMOKE_FLAG=""
PREFIX=""
if [ "$SMOKE" = "1" ]; then
  SMOKE_FLAG="--smoke"; PREFIX="smoke_"; REPS=1; GAP=10
  echo "SMOKE MODE: 60 s per cell, 1 rep, smoke_ prefix, output to logs/smoke_data/"
fi

mkdir -p logs logs/.done
MASTER=logs/tpu_queue_master.log
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" | tee -a "$MASTER"; }

ssh_all() {  # ssh_all "<remote command>"
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" --worker=all --command="$1"
}

# ── G1 gate: math must be right before any TPU time is spent ─────────────────
say "G1: CPU gradient self-test on worker 0"
# JAX_PLATFORMS=cpu is REQUIRED, not incidental: self_test() is documented as a
# CPU-only gate, and on the TPU backend jnp matmuls default to bfloat16, so the
# finite-difference check (lp-lm ~ 4e-6 against a loss of ~4.6) is pure noise --
# it reported rel_err=6.1e-2 vs its 5e-2 tolerance and aborted the whole TPU run
# even though the gradient was correct (9.4e-3 on CPU). The XLA flag gives the
# mesh path 8 host devices to exercise, per the same docstring.
# Capture first, THEN test. The old form piped ssh_all into `grep -q`, but grep
# exits on its first match and SIGPIPEs the ssh upstream -- and under `pipefail`
# that makes the whole pipeline non-zero, so the gate aborted even when the
# self-test had printed PASS. Deciding on the captured text keeps the verdict
# independent of ssh/tee exit status.
G1_OUT="$(ssh_all "cd $REPO_DIR && JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 PYTHONPATH=$REPO_DIR python3 workloads/tpu_fsdp_training.py --self-test" 2>&1)"
printf '%s\n' "$G1_OUT" >> "$MASTER"
printf '%s\n' "$G1_OUT" | grep -E 'loss0=|self-test' || true
if printf '%s\n' "$G1_OUT" | grep -q "self-test FAIL" \
   || ! printf '%s\n' "$G1_OUT" | grep -q "self-test PASS"; then
  say "ABORT: self-test did not report PASS -- the transformer/optimizer is wrong,"
  say "       every collected trace would be meaningless."
  exit 1
fi
say "G1 PASS: analytic gradient matches finite difference on CPU"

# ── One cell across all workers, with GCS .done accounting ───────────────────
run_cell() {
  local cell="$1"
  local label="${PREFIX}${cell}"
  local marker="logs/.done/${label}.done"
  if [ -f "$marker" ]; then
    say "SKIP  $cell (already: $(cat "$marker"))"
    return 0
  fi

  # Deterministic shared run_id: every worker of THIS cell agrees with no
  # coordination channel (plan section 2.1).
  local rid
  rid=$(printf '%s' "${STAMP}_${cell}_${REP:-0}" | sha1sum | cut -c1-12)

  local remote="source \$HOME/.spar_tpu_env 2>/dev/null; cd $REPO_DIR && \
PYTHONPATH=$REPO_DIR python3 scripts/run_workload_tpu.py \
--workload $cell --run-id $rid --worker-id \$TPU_WORKER_ID \
--gcs-prefix $GCS $SMOKE_FLAG --multihost-required"

  say "START $cell (run_id=$rid) -> $GCS/runs/$rid/"
  local t0=$SECONDS
  ssh_all "$remote" >>"$MASTER" 2>&1
  local rc=$?
  local dur=$((SECONDS - t0))

  # GCS done accounting: one marker per worker must have landed.
  local got
  got=$(gsutil ls "$GCS/runs/$rid/w*/${label}.w*.done" 2>/dev/null | wc -l | tr -d ' ')
  if [ "$rc" -eq 0 ] && [ "$got" -ge "$NUM_WORKERS" ]; then
    echo "$(stamp) rc=0 dur=${dur}s run_id=$rid workers=$got" > "$marker"
    say "OK    $cell (${dur}s, $got/$NUM_WORKERS workers)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s run_id=$rid workers=$got" \
      > "logs/.done/${label}.failed"
    say "FAIL  $cell rc=$rc ($got/$NUM_WORKERS workers) -- continuing"
  fi
  sleep "$GAP"
}

# ── Cell list (decisive-cell-first) ──────────────────────────────────────────
CORE=(
  "${SCOPE}_train_fsdp7b"
  "${SCOPE}_infershard_7b"
  "${SCOPE}_inferrep_7b"
  "${SCOPE}_other_fft"
  "${SCOPE}_idle"
)

say "===== TPU queue start (scope=$SCOPE, workers=$NUM_WORKERS, session=$SESSION) ====="

for REP in $(seq 1 "$REPS"); do
  export REP
  say "----- rep $REP / $REPS -----"
  for cell in "${CORE[@]}"; do
    run_cell "${cell}_rep${REP}"
  done
  if [ "$STRETCH" = "1" ]; then
    # 1b comparison point + 30b stretch (tpuT16 only), collected late so a
    # truncated run keeps the decisive 7b contrast intact.
    run_cell "${SCOPE}_train_fsdp1b_rep${REP}"
    run_cell "${SCOPE}_inferrep_1b_rep${REP}"
    if [ "$SCOPE" = "tpuT16" ]; then
      run_cell "tpuT16_train_fsdp30b_rep${REP}"
      run_cell "tpuT16_infershard_30b_rep${REP}"
    fi
  fi
done

say "===== TPU queue complete ====="
say "done:   $(ls logs/.done/${PREFIX}${SCOPE}_*.done 2>/dev/null | wc -l) cells"
say "failed: $(ls logs/.done/${PREFIX}${SCOPE}_*.failed 2>/dev/null | wc -l) cells"
if [ "$SMOKE" = "1" ]; then
  say "SMOKE COMPLETE. If clean, clear markers and run the real queue:"
  say "  rm -f logs/.done/smoke_* && nohup bash $0 &"
fi
