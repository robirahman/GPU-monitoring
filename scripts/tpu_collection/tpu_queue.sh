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
# The SCOPE is fixed by the provisioned slice (v6e-16 => tpuT16). NUM_WORKERS is
# now DERIVED from the live slice by default -- do not hardcode it. (v6e is 8
# chips/host, so v6e-16 = 2 hosts = 2 workers, NOT the 4 the old default assumed;
# a wrong NUM_WORKERS makes the .done accounting mark every cell FAILED.)
#
# MULTISLICE (SCOPE=tpuTMS): set NODE_COUNT=2 (and NODE_PREFIX). The queue then
# fans each cell to ALL workers of BOTH slice nodes concurrently, so
# jax.distributed.initialize() forms one cross-slice mesh and megascale.* (the DCN
# channel) accumulates. VALIDATE THIS ON A 2x v6e-1 SMOKE FIRST (assert
# dcn_collective_lat_us_p50 != -1); if DCN stays dark, the runtime did not auto-set
# the megascale coordinator env -- export MEGASCALE_COORDINATOR_ADDRESS on the workers.
#
#   ACCEL=v6e-1 bash scripts/tpu_collection/tpu_queue.sh --probe-only   # metric-name gate
#   SMOKE=1 bash scripts/tpu_collection/tpu_queue.sh                    # shakedown first
#   nohup   bash scripts/tpu_collection/tpu_queue.sh &                  # real run
set -uo pipefail

PROBE_ONLY=0
[ "${1:-}" = "--probe-only" ] && { PROBE_ONLY=1; shift; }

# ── Config (override via env) ────────────────────────────────────────────────
PROJECT=${PROJECT:-focus-surfer-494820-g0}
ZONE=${ZONE:-us-east1-d}
TPU_NAME=${TPU_NAME:-spar-tpu-16}
REPO_DIR=${REPO_DIR:-\$HOME/SPAR-GPU-monitoring}   # path ON the worker
SCOPE=${SCOPE:-tpuT16}
NODE_COUNT=${NODE_COUNT:-1}                         # >1 => multislice (tpuTMS)
NODE_PREFIX=${NODE_PREFIX:-$TPU_NAME}
NUM_WORKERS=${NUM_WORKERS:-}                        # empty => derive from live slice
BUCKET=${BUCKET:-gs://spar-tpu-telemetry}
SESSION=${SESSION:-$(date -u +%Y%m%d_%H%M%S)}
GCS=${BUCKET}/${SESSION}
STAMP=${STAMP:-$SESSION}
DUR=${DUR:-600}
GAP=${GAP:-60}
REPS=${REPS:-3}
SMOKE=${SMOKE:-0}
# ADVERSARIAL=1 runs the MN-I ladder + honest controls instead of the honest
# CORE ladder. Kept behind a flag so the two stages stay separately schedulable
# and a truncated adversarial run cannot eat the honest baseline's budget.
ADVERSARIAL=${ADVERSARIAL:-0}
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

# Slice node name(s): $TPU_NAME for single-slice, ${NODE_PREFIX}-0..N-1 multislice.
slice_nodes() {
  if [ "$NODE_COUNT" -gt 1 ]; then
    for i in $(seq 0 $((NODE_COUNT - 1))); do echo "${NODE_PREFIX}-${i}"; done
  else
    echo "$TPU_NAME"
  fi
}

ssh_all() {  # ssh_all "<cmd>" -- fan to ALL workers of ALL slice nodes
  local cmd="$1" rc=0 n; local pids=()
  if [ "$NODE_COUNT" -gt 1 ]; then
    # Multislice: launch on every slice CONCURRENTLY so the workers form ONE
    # cross-slice jax mesh (the --node-count runtime supplies the megascale
    # coordinator env). Serial per-slice ssh would deadlock -- slice 0 blocks on a
    # collective barrier that slice 1 has not reached. VALIDATE on a 2x v6e-1 smoke:
    # if dcn_collective_lat_us_p50 stays -1, export MEGASCALE_COORDINATOR_ADDRESS.
    # Export a PER-SLICE identity. MEASURED on a live 2x v6e-1: every slice's
    # worker reports agent-worker-number=0 and there is no MEGASCALE_SLICE_ID in
    # the env, so without this both slices resolve to worker_id 0 -- slice 1's
    # parquet and .done marker OVERWRITE slice 0's under runs/<rid>/w0/, `got`
    # counts 1, and the >= NUM_WORKERS check marks a healthy cell FAILED. That is
    # exactly the "1/2 workers" failure seen on every tpuTMS cell.
    local _si=0
    for n in $(slice_nodes); do
      gcloud compute tpus tpu-vm ssh "$n" --project="$PROJECT" --zone="$ZONE" \
        --worker=all --command="export SPAR_TPU_SLICE_ID=${_si}; $cmd" & pids+=($!)
      _si=$((_si + 1))
    done
    for p in "${pids[@]}"; do wait "$p" || rc=1; done
    return $rc
  fi
  gcloud compute tpus tpu-vm ssh "$TPU_NAME" \
    --project="$PROJECT" --zone="$ZONE" --worker=all --command="$cmd"
}

ssh_worker0() {  # worker 0 of the first slice only
  gcloud compute tpus tpu-vm ssh "$(slice_nodes | head -1)" \
    --project="$PROJECT" --zone="$ZONE" --worker=0 --command="$1"
}

# ── NUM_WORKERS: derive from the live slice(s) unless the caller pinned it ─────
derive_workers() {
  local total=0 n c
  for n in $(slice_nodes); do
    c=$(gcloud compute tpus tpu-vm describe "$n" --project="$PROJECT" --zone="$ZONE" \
        --format='value(networkEndpoints[].ipAddress)' 2>/dev/null \
        | tr ';,' '\n' | grep -c '[0-9]')
    total=$((total + c))
  done
  echo "$total"
}
if [ -z "$NUM_WORKERS" ]; then
  NUM_WORKERS=$(derive_workers)
  if ! [ "$NUM_WORKERS" -ge 1 ] 2>/dev/null; then
    say "WARN: could not derive worker count (slice not up yet?); defaulting to 1"
    NUM_WORKERS=1
  fi
  say "derived NUM_WORKERS=$NUM_WORKERS from $NODE_COUNT slice(s)"
fi

# ── --probe-only: metric-name drift gate for a new TPU generation, then exit ──
if [ "$PROBE_ONLY" = "1" ]; then
  say "PROBE: scripts/tpu_metric_probe.py on worker 0 (does this gen serve the same"
  say "       libtpu metric names the collector classifies by?)"
  ssh_worker0 "cd $REPO_DIR && PYTHONPATH=$REPO_DIR python3 scripts/tpu_metric_probe.py --hold 20" \
    2>&1 | tee -a "$MASTER"
  say "PROBE done -- compare served names vs collect_tpu_telemetry.py _GRPC_METRICS"
  say "       (libtpu 0.0.17 baseline). A drift means re-pin requirements_tpu.txt."
  exit 0
fi

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
--workload $cell --run-id $rid \
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

# Adversarial ladder + matched honest controls, DECISIVE-CELL-FIRST so a run cut
# short still carries the comparison the rung exists for:
#   1-3  the claim under test (F=0.5), the evading cell (F=0), and the
#        like-for-like honest partner that isolates the apparatus
#   4-5  the rest of the F ladder
#   6-8  the false-positive probes -- duty and kernel axes, which on GPU produced
#        an honest workload that false-alarmed HARDER than the F=0.05 disguise --
#        plus the DCN comm probe that has no NVIDIA counterpart
ADV=(
  "${SCOPE}_advtrain_satint_f50"
  "${SCOPE}_advtrain_satfiller_only"
  "${SCOPE}_nonml_sat"
  "${SCOPE}_advtrain_satint_f20"
  "${SCOPE}_advtrain_satint_f05"
  "${SCOPE}_nonml_duty30"
  "${SCOPE}_nonml_duty50_hash"
  "${SCOPE}_nonml_comm"
)

# Multislice only: the honest control matched to the TRAINING footprint (~16.5 GB)
# rather than to F=0's 5.2 GB. Registered for tpuTMS* rungs only -- see
# registry_tpu._ALLOC_MATCHED_TRAIN -- so emitting it for a single-slice scope
# would ask for a label that does not exist.
# The 16 GB matched control is multislice-only: its --match-mem-mb target was
# measured per chip on 2x v6e-1, and the same 1b model sharded over 8 chips holds
# a different per-chip footprint, so a single global target would be wrong at
# other scopes. Each scope needs its own measured number before this cell is
# meaningful there.
case "$SCOPE" in
  tpuTMS*) ADV+=("${SCOPE}_nonml_sat_16g") ;;
esac

# CLASS REFERENCES for EVERY scope, appended last so a truncated run still keeps
# the decisive adversarial comparison. The campaign runs with ADVERSARIAL=1 and
# STRETCH unset, which collected ONLY disguised training and honest non-ML -- no
# ml_inference at all, and no honest (undisguised) training, on single-slice and
# multislice alike. Two consequences, both of which would have quietly weakened
# the TPU answer rather than breaking anything visibly:
#   * Stage 2 discriminates train vs infer, so with no inference class it cannot
#     be trained on TPU data, and the headline operating point is defined on the
#     full pipeline's p_train. Without these cells the TPU rung could only report
#     a Stage-1 score, which answers a weaker question than "does the claim
#     survive on this hardware".
#   * every ml_training row would be a DISGUISED run, so a detector fitted on it
#     learns "disguised training", not "training".
# Both classes are wanted at both slice topologies: single-slice keeps the
# collectives on ICI (or absent entirely at tpuT1) while multislice puts them on
# DCN, and that contrast is the point of the TPU rung.
# tpu_sharded_inference had never run on hardware before today -- it carried the
# same multi-host deadlock -- so these may need a pass of their own; being last
# means a failure costs nothing else.
ADV+=("${SCOPE}_train_fsdp1b" "${SCOPE}_inferrep_1b" "${SCOPE}_infershard_1b")

# ONLY_CELLS: comma-separated substrings; empty runs everything. Exists so a
# re-run can collect just the cells that are missing instead of re-billing a
# slice for the ones already in hand -- the campaign's completion markers live in
# ${BUCKET}/collected and are campaign-wide, so a relaunch that wants one new
# cell would otherwise either skip the scope entirely or repeat all of it.
ONLY_CELLS=${ONLY_CELLS:-}
_wanted(){
  [ -z "$ONLY_CELLS" ] && return 0
  local c=$1 pat
  local IFS=,
  for pat in $ONLY_CELLS; do
    [ -n "$pat" ] || continue
    case "$c" in *"$pat"*) return 0 ;; esac
  done
  return 1
}

say "===== TPU queue start (scope=$SCOPE, workers=$NUM_WORKERS, session=$SESSION) ====="

for REP in $(seq 1 "$REPS"); do
  export REP
  say "----- rep $REP / $REPS -----"
  if [ "$ADVERSARIAL" = "1" ]; then
    for cell in "${ADV[@]}"; do
      _wanted "$cell" || continue
      run_cell "${cell}_rep${REP}"
    done
    # tpuT16 only: the scope that structurally matches the 2x8 H100 setup, so the
    # kernel-axis and high-HBM controls are collected where the comparison is
    # tightest rather than at every scope.
    if [ "$SCOPE" = "tpuT16" ]; then
      _wanted tpuT16_nonml_duty50_fft && run_cell "tpuT16_nonml_duty50_fft_rep${REP}"
      _wanted tpuT16_nonml_himem && run_cell "tpuT16_nonml_himem_rep${REP}"
    fi
  else
    for cell in "${CORE[@]}"; do
      _wanted "$cell" || continue
      run_cell "${cell}_rep${REP}"
    done
  fi
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
