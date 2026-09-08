#!/bin/bash
# Multi-node collection queue: the communication-scope ladder S1 < S8 < S16.
#
# Runs IDENTICALLY on both a3-highgpu-8g nodes. The only per-node inputs are
# three env vars exported at launch (see below). Cells are ordered
# decisive-first so a Spot preemption (~35 min was observed) still banks the
# headline S16 train-vs-sharded-inference contrast, then downshifts to
# fully-single-node S8 and S1 phases.
#
# ── LAUNCH (both nodes; node0 first) ────────────────────────────────────────
#   # on gpu-a:
#   export MASTER_ADDR=$(hostname -I | awk '{print $1}')   # node0 internal IP
#   export NODE_RANK=0 RUN_STAMP=20260819a
#   # on gpu-b:
#   export MASTER_ADDR=<node0 internal IP>  NODE_RANK=1 RUN_STAMP=20260819a
#   # then on BOTH:
#   nohup bash scripts/rebuttal_collection/multinode_queue.sh &
#
# Both queues iterate the SAME cell list in the SAME order, and every S16 cell
# rendezvouses through c10d at $MASTER_ADDR:29500 -- so the rendezvous IS the
# per-cell barrier; no separate coordination channel is needed. The shared
# run_id per cell is derived deterministically (sha1 of RUN_STAMP_cell_rep) so
# both nodes agree without talking. A collective returns rc=0 only if all 16
# ranks finished, so a local .done marker is a valid proxy for global success.
#
# ── SMOKE FIRST ─────────────────────────────────────────────────────────────
#   SMOKE=1 bash scripts/rebuttal_collection/multinode_queue.sh
# 60 s cells, smoke_ prefixed, output to logs/smoke_data/ (never data/).

set -uo pipefail
REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }
export PYTHONPATH=$REPO
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

PY=${PY:-python3}                       # venv-main python (telemetry + training)
PY_VLLM=${PY_VLLM:-$PY}                 # venv-vllm python (serving cells only)
# `torchrun` and `ray` are venv console-scripts, and root's secure_path does NOT
# include the venv bin dirs -- while run_workload_multi_gpu.py execs the command
# vector directly (no shell), so a bare `torchrun` raises FileNotFoundError and
# the cell dies in ~4 s having produced only baseline telemetry. A bare `ray`
# fails the same way, but its failure was swallowed by `|| true`, leaving the
# S16 shard cell to spin up a LOCAL 8-GPU Ray and block forever on a 16-GPU
# placement group. Resolve both from the interpreter that owns them.
TORCHRUN=${TORCHRUN:-$(dirname "$PY")/torchrun}
RAY=${RAY:-$(dirname "$PY_VLLM")/ray}
MASTER_ADDR=${MASTER_ADDR:?export MASTER_ADDR to the node0 internal IP}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:?export NODE_RANK (0 on node0, 1 on node1)}
RUN_STAMP=${RUN_STAMP:?export RUN_STAMP (identical string on both nodes)}
NNODES=${NNODES:-2}

# Remember whether the CALLER set these, so SMOKE mode does not clobber an
# explicit override. Passing TIMEOUT=600 with SMOKE=1 previously had no effect --
# the smoke block reassigned TIMEOUT=240 unconditionally, so a cell that needed
# longer kept timing out and looked broken when it was only slow.
_DUR_SET=${DUR+x}; _TIMEOUT_SET=${TIMEOUT+x}; _TIMEOUT_BIG_SET=${TIMEOUT_BIG+x}
_GAP_SET=${GAP+x}; _REPS_SET=${REPS+x}
DUR=${DUR:-600}                         # 600 s -> 39 windows at 30 s / 15 s
TIMEOUT=${TIMEOUT:-900}
TIMEOUT_BIG=${TIMEOUT_BIG:-1500}        # 30B / model loads
GAP=${GAP:-60}
REPS=${REPS:-3}
RENDEZVOUS_TIMEOUT=${RENDEZVOUS_TIMEOUT:-120}
RAY_JOIN_TIMEOUT=${RAY_JOIN_TIMEOUT:-300}   # both nodes must register their 8 GPUs
INFER_MODEL=${INFER_MODEL:-llm_infer_qwen25_1_5b_fp16} # matched to the fsdp1b train arm
# Second inference negative, same model at batch 1 (latency-bound) rather than
# batch 8 (throughput-bound). Adding a DISTINCT cell doubles the inferrep job
# count per rep, which is what clears the eval's low_n threshold; extra reps of
# the same cell would only add correlated jobs.
INFER_MODEL_BS1=${INFER_MODEL_BS1:-llm_infer_qwen25_1_5b_bs1}
SHARD_MODEL=${SHARD_MODEL:-qwen25_72b_tp2}             # realistic sharded-inference model
# Matched-scale sharded negative. Qwen2.5-1.5B has 12 attention heads, so TP8 is
# impossible (vLLM needs num_heads % tp == 0); 16 ranks are reached as TP4xPP4,
# which still keeps each TP group inside one node and crosses nodes on the
# pipeline boundary, like the 72B cell's TP8xPP2. Documented because the
# factorization differs between the two shard variants by necessity.
SHARD_MODEL_SMALL=${SHARD_MODEL_SMALL:-qwen25_1_5b_fp16}
# Disguised-training cells (workloads/adversarial_multinode.py). ORACLE_MODEL is
# the persisted honest-only detector the adversary queries; without it MN-O
# degrades to plain dilution and says so in its own log.
ORACLE_MODEL=${ORACLE_MODEL:-/opt/spar/repo/classifier/models/honest_detector.joblib}
ADV_DILUTION=${ADV_DILUTION:-10}
SMOKE=${SMOKE:-0}
# ONLY_CELLS: extended-regex whitelist of cell names. Everything else is skipped
# cheaply. Used to smoke just the decisive cells, and to re-run one cell after a
# preemption without walking the whole queue.
ONLY_CELLS=${ONLY_CELLS:-}

PREFIX=""
OUTDIR="data"
if [ "$SMOKE" = "1" ]; then
  [ -z "$_DUR_SET" ]         && DUR=60
  [ -z "$_TIMEOUT_SET" ]     && TIMEOUT=420   # > the inferrep cells' fixed 300 s
  [ -z "$_TIMEOUT_BIG_SET" ] && TIMEOUT_BIG=600
  [ -z "$_GAP_SET" ]         && GAP=10
  [ -z "$_REPS_SET" ]        && REPS=1
  PREFIX="smoke_"
  OUTDIR="logs/smoke_data"   # keep smoke telemetry out of data/ (loader rglobs data/**)
  mkdir -p logs/smoke_data
  echo "SMOKE MODE: 60 s cells, output to logs/smoke_data/ (NOT usable data)."
fi

# Markers live under the run stamp and carry the smoke prefix: a smoke cell that
# passed used to leave logs/.done/<cell>_node<N>.done, which the REAL run then
# read as "already collected" and skipped -- banking zero data for that cell.
MARKER_DIR="logs/.done/${RUN_STAMP}"
mkdir -p logs "$MARKER_DIR" "$OUTDIR"

# Fail fast: a missing launcher is a config error, not a per-cell failure, and
# every cell in the queue would fail identically.
for _b in "$TORCHRUN" "$RAY"; do
  [ -x "$_b" ] || { echo "FATAL: launcher not executable: $_b (check PY / PY_VLLM)" >&2; exit 1; }
done
MASTER=logs/multinode_master.log
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] node$NODE_RANK $*" | tee -a "$MASTER"; }

# Deterministic shared run_id: identical on both nodes for the same cell+rep.
rid_for() { printf '%s' "${RUN_STAMP}_$1_$2" | sha1sum | cut -c1-12; }

# Cell whitelist gate. Checked by run_stage AND by mn_ray_shard -- the latter
# brings a Ray cluster up before it ever reaches run_stage, so it has to test
# the filter itself or a skipped cell would still pay the cluster setup.
cell_enabled() {  # cell_enabled <stage>
  [ -z "$ONLY_CELLS" ] && return 0
  echo "$1" | grep -qE "$ONLY_CELLS"
}

# --- host-NIC sidecar: captures the inter-server channel NVML cannot see. ----
SIDE_PID=""
start_sidecar() {  # start_sidecar <label> <run_id>
  $PY scripts/collect_host_telemetry.py --duration $((DUR + 120)) \
      --output "$OUTDIR/nic_${1}_node${NODE_RANK}_${2}.parquet" \
      --run-id "$2" --label "$1" --node-rank "$NODE_RANK" \
      --checkpoint-sec 60 >> "logs/${1}_node${NODE_RANK}_${2}.nic.log" 2>&1 &
  SIDE_PID=$!
}
stop_sidecar() { [ -n "$SIDE_PID" ] && kill "$SIDE_PID" 2>/dev/null; wait "$SIDE_PID" 2>/dev/null; SIDE_PID=""; }

# --- generic stage wrapper: per-node marker, .failed on error, always continue.
run_stage() {  # run_stage <stage> <run_id> -- <command...>
  local stage="$1" rid="$2"; shift 2; [ "$1" = "--" ] && shift
  cell_enabled "$stage" || { say "SKIP  $stage (ONLY_CELLS filter)"; return 0; }
  # rid = sha1(RUN_STAMP_cell_rep), so keying the marker on it makes it unique
  # per REP. Keyed on cell alone, rep 2 and rep 3 saw rep 1's marker and skipped
  # every cell -- REPS=3 silently collected a single rep.
  local marker="${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.done"
  local logf="logs/${PREFIX}${stage}_node${NODE_RANK}_${rid}.log"
  if [ -f "$marker" ]; then say "SKIP  $stage ($(cat "$marker"))"; return 0; fi

  # Reap strays before starting. Nothing else should be holding a GPU between
  # cells, and a leftover process is not benign: a timed-out vLLM cell used to
  # orphan 8 processes pinning 73 GB each, after which every later cell on the
  # node died of CUDA OOM while asking for ~7 GB. Warning about it was not
  # enough -- the queue has to clear it, since the run is unattended.
  local nproc; nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  if [ "$nproc" -ne 0 ]; then
    say "WARN  $stage: $nproc stray compute process(es) resident -- reaping"
    for spid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -TERM "$spid" 2>/dev/null || true
    done
    sleep 10
    for spid in $(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null); do
      kill -KILL "$spid" 2>/dev/null || true
    done
    sleep 5
    nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$nproc" -ne 0 ]; then
      say "WARN  $stage: $nproc process(es) STILL resident after reap -- cell may OOM"
    else
      say "      reaped clean; GPUs free"
    fi
  fi

  say "START $stage (rid=$rid) -> $logf"
  start_sidecar "$stage" "$rid"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?
  local dur=$((SECONDS - t0))
  stop_sidecar
  if [ $rc -eq 0 ]; then
    echo "$(stamp) rc=0 dur=${dur}s rid=$rid" > "$marker"
    say "OK    $stage (${dur}s)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s rid=$rid" > "${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.failed"
    say "FAIL  $stage rc=$rc (${dur}s) -- continuing"
    tail -5 "$logf" | sed 's/^/        /' | tee -a "$MASTER"
  fi
  sleep "$GAP"
  return 0
}

# ── S16 cells (both nodes participate) ───────────────────────────────────────

# 2-node torchrun cell: the c10d rendezvous is the cross-node barrier.
mn_torchrun() {  # mn_torchrun <stage> <run_id> <timeout> -- <workload.py args...>
  local stage="$1" rid="$2" to="$3"; shift 3; [ "$1" = "--" ] && shift
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
      --num-gpus 8 --run-id "$rid" --node-rank "$NODE_RANK" \
      --output-dir "$OUTDIR" --timeout "$to" -- \
      "$TORCHRUN" --nnodes="$NNODES" --node-rank="$NODE_RANK" \
        --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv-conf="join_timeout=${RENDEZVOUS_TIMEOUT}" --nproc-per-node=8 \
        "$@"
}

# Sharded vLLM inference across both nodes via Ray. Asymmetric: node0 is the
# Ray head and drives the job + its own telemetry; node1 joins the Ray cluster
# and only records telemetry (its GPUs are driven by the remote Ray worker).
# Barrier on the Ray cluster really holding BOTH nodes' GPUs. There is no c10d
# rendezvous in this cell, so without the barrier a half-formed cluster is
# invisible: vLLM asks for a 16-GPU placement group, Ray can never schedule it,
# and the cell burns its entire TIMEOUT_BIG with idle GPUs (seen 2026-08-20).
wait_ray_gpus() {  # wait_ray_gpus <want> <timeout_s>
  local want="$1" to="$2" t0=$SECONDS got=""
  while [ $((SECONDS - t0)) -lt "$to" ]; do
    got=$("$PY_VLLM" -c 'import ray;ray.init(address="auto",logging_level="ERROR");print(int(ray.cluster_resources().get("GPU",0)));ray.shutdown()' \
          2>/dev/null | tail -1 | tr -dc '0-9')
    if [ "${got:-0}" -ge "$want" ]; then say "ray cluster ready: ${got} GPUs"; return 0; fi
    sleep 5
  done
  say "ray cluster NOT ready: ${got:-0}/${want} GPUs after ${to}s"
  return 1
}

# Wait until no compute process is left on THIS node's GPUs. Used by the non-head
# node of a sharded cell: its telemetry window is only DUR seconds, but the head's
# cell also covers model load and vLLM startup, so this node finishes first. If it
# then leaves -- `ray stop`, or the next cell's stray-reap -- it kills the head's
# workers that are running HERE, which surfaced as "RayWorkerProc rank=[11] died
# unexpectedly" mid-generation and failed every cross-node shard cell we ever ran.
# Polling local compute processes needs no coordination channel: the head's
# workers ARE local processes, so their exit is the completion signal.
wait_gpus_idle() {  # wait_gpus_idle <timeout_s>
  local to="$1" t0=$SECONDS n=0
  while [ $((SECONDS - t0)) -lt "$to" ]; do
    n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
    if [ "$n" -eq 0 ]; then
      say "      head's workers exited after $((SECONDS - t0))s; GPUs idle"
      return 0
    fi
    sleep 15
  done
  say "WARN  $n compute process(es) still resident after ${to}s -- proceeding"
  return 1
}

ensure_ray_cluster() {  # ensure_ray_cluster <logfile>
  local rl="$1" want=$((NNODES * 8)) got
  got=$("$PY_VLLM" -c 'import ray;ray.init(address="auto",logging_level="ERROR");print(int(ray.cluster_resources().get("GPU",0)));ray.shutdown()' \
        2>/dev/null | tail -1 | tr -dc '0-9')
  if [ "${got:-0}" -ge "$want" ]; then say "      reusing ray cluster (${got} GPUs)"; return 0; fi

  "$RAY" stop --force >>"$rl" 2>&1 || true
  sleep 5
  if [ "$NODE_RANK" -eq 0 ]; then
    "$RAY" start --head --port=6379 --num-gpus=8 >>"$rl" 2>&1 || return 1
  else
    # The head may still be binding; retry rather than failing the cell outright.
    local waited=0
    until "$RAY" start --address="${MASTER_ADDR}:6379" --num-gpus=8 >>"$rl" 2>&1; do
      waited=$((waited + 10))
      [ "$waited" -ge 120 ] && { say "      could not join ray head after ${waited}s"; return 1; }
      sleep 10
    done
  fi
  wait_ray_gpus "$want" "$RAY_JOIN_TIMEOUT"
}

mn_ray_shard() {  # mn_ray_shard <stage> <run_id> <model> <tp> <pp> [extra vllm args...]
  local stage="$1" rid="$2" model="${3:-$SHARD_MODEL}" tp="${4:-8}" pp="${5:-2}"
  local extra=("${@:6}")   # e.g. --max-model-len/--prompt-tokens for the KV cell
  cell_enabled "$stage" || { say "SKIP  $stage (ONLY_CELLS filter)"; return 0; }
  local raylog="logs/${PREFIX}${stage}_node${NODE_RANK}_${rid}.ray.log"
  local marker="${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.done"
  [ -f "$marker" ] && { say "SKIP  $stage ($(cat "$marker"))"; return 0; }

  ray_bail() {  # ray_bail <why>
    say "FAIL  $stage: $1 -- skipping cell (see $raylog)"
    echo "$(stamp) $1" > "${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.failed"
    "$RAY" stop >>"$raylog" 2>&1 || true
    return 0
  }

  # Build the cluster ONCE and reuse it across shard cells. Starting and stopping
  # it per cell raced: with two shard cells in a run, the second saw 8/16 GPUs on
  # the head and 0/16 on the worker, because one node was tearing the cluster down
  # while the other was bringing it up. ensure_ray_cluster is idempotent -- it
  # probes first and only (re)builds when the cluster is not already complete.
  ensure_ray_cluster "$raylog" || { ray_bail ray-cluster-incomplete; return 0; }

  if [ "$NODE_RANK" -eq 0 ]; then
    # RAY_ADDRESS pins vLLM to the cluster we just built. Without it ray.init()
    # can quietly spin up a LOCAL single-node instance and then deadlock.
    run_stage "$stage" "$rid" -- \
      env RAY_ADDRESS="${MASTER_ADDR}:6379" \
      $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
        --num-gpus 8 --run-id "$rid" --node-rank 0 --output-dir "$OUTDIR" \
        --timeout "$TIMEOUT_BIG" -- \
        $PY_VLLM workloads/llm_inference_vllm.py --model "$model" \
          --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" \
          --distributed-executor-backend ray --duration "$DUR" "${extra[@]}"
  else
    # node1 records telemetry for the shared run_id while node0 drives vLLM.
    run_stage "$stage" "$rid" -- \
      $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
        --num-gpus 8 --run-id "$rid" --node-rank 1 --telemetry-only \
        --duration "$DUR" --output-dir "$OUTDIR"
    # Do NOT tear down until the head's workers on this node have exited.
    wait_gpus_idle "$TIMEOUT_BIG" || true
  fi
  # Deliberately NOT stopping Ray here: the next shard cell reuses this cluster.
  # It is torn down once at the end of the queue.
}

# Replicated inference: 8 independent single-GPU replicas per node, no
# rendezvous, per-replica run_ids. Both nodes run this independently.
mn_replicated() {  # mn_replicated <stage> <run_id> <infer-label>
  local stage="$1" rid="$2" label="$3"
  # --label: scope_ladder_eval derives the ladder rung from the s1_/s8_/s16_
  # prefix of the RECORDED label, so the telemetry must carry the cell name --
  # recording the registry label instead dropped this whole arm as scope "na".
  # --workload-python: the registry command bakes in venv-main's interpreter,
  # which has no vllm.
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --parallel --workload "$label" \
      --label "${PREFIX}${stage}" --workload-python "$PY_VLLM" \
      --num-gpus 8 --node-rank "$NODE_RANK" --output-dir "$OUTDIR" --timeout "$TIMEOUT"
}

# ── S8 cells (each node runs its own single-node job concurrently) ───────────
s8_torchrun() {  # s8_torchrun <stage> <run_id> <timeout> -- <workload.py args...>
  local stage="$1" rid="$2" to="$3"; shift 3; [ "$1" = "--" ] && shift
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
      --num-gpus 8 --run-id "$rid" --node-rank "$NODE_RANK" \
      --output-dir "$OUTDIR" --timeout "$to" -- \
      "$TORCHRUN" --standalone --nproc-per-node=8 "$@"
}

# Single-node sharded inference: collectives without training, inside one node.
s8_shard() {  # s8_shard <stage> <run_id> <model> <tp> <pp>
  local stage="$1" rid="$2" model="$3" tp="$4" pp="${5:-1}"
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" --num-gpus 8 \
      --run-id "$rid" --node-rank "$NODE_RANK" --output-dir "$OUTDIR" \
      --timeout "$TIMEOUT_BIG" -- \
      $PY_VLLM workloads/llm_inference_vllm.py --model "$model" \
        --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" --duration "$DUR"
}

# ── S1 cells (single-GPU: world=1 training, no collectives) ──────────────────
s1_train() {  # s1_train <stage> <run_id> -- <workload.py args...>
  local stage="$1" rid="$2"; shift 2; [ "$1" = "--" ] && shift
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
      --num-gpus 1 --run-id "$rid" --node-rank "$NODE_RANK" \
      --output-dir "$OUTDIR" --timeout "$TIMEOUT" -- \
      "$TORCHRUN" --standalone --nproc-per-node=1 "$@"
}
s1_replicated() {  # s1_replicated <stage> <run_id> <label>
  mn_replicated "$1" "$2" "$3"
}

# ── The queue: Phase A (S16) -> B (S8) -> C (S1), rep-major, priority-first ──

say "queue start: NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT REPS=$REPS DUR=$DUR SMOKE=$SMOKE ONLY_CELLS=${ONLY_CELLS:-<all>}"

for rep in $(seq 1 "$REPS"); do
  say "===== rep $rep / $REPS ====="

  # -- Phase A: S16 (needs both nodes). Priority 1 cells first. --
  c=s16_train_fsdp1b;              mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/fsdp_training.py --size 1b --duration "$DUR"
  c=s16_infershard_qwen72b_tp8pp2;   mn_ray_shard "$c" "$(rid_for "$c" "$rep")" "$SHARD_MODEL" 8 2
  c=s16_infershard_qwen1_5b_tp4pp4;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" "$SHARD_MODEL_SMALL" 4 4
  c=s16_train_ddp_resnet50;        mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/pytorch_training_ddp.py --model resnet50 --duration "$DUR"
  c=s16_inferrep_qwen1_5b;           mn_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL"

  # -- Disguised training: real FSDP training that tries to read as inference or
  # non-ML. Same --size 1b model as the honest s*_train_fsdp1b rungs, so the
  # disguise is the ONLY difference. Run these with
  #   ONLY_CELLS='advtrain'
  # when the honest ladder is already banked under an earlier stamp.
  c=s16_advtrain_interleave_f50;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.5 --duration "$DUR"
  c=s16_advtrain_interleave_f20;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.2 --duration "$DUR"
  c=s16_advtrain_interleave_f10;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.1 --duration "$DUR"
  c=s16_advtrain_oracle;           mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-model "$ORACLE_MODEL" --duration "$DUR"

  # filler_only is the CONTROL that decides whether MN-I can work at all: the
  # disguise blends training with this filler, so if the filler ALONE does not
  # read as `other` there is nothing for the blend to hide behind, and any
  # MN-I result would be uninterpretable. --train-frac 0 now genuinely trains
  # nothing (the old 1/period floor made that impossible).
  c=s16_advtrain_filler_only;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.0 --duration "$DUR"
  # Fine end of the sweep: where does detection actually break, and what does
  # that cost? Only meaningful now that fractions below 0.05 are not clamped.
  c=s16_advtrain_interleave_f05;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.05 --duration "$DUR"
  c=s16_advtrain_interleave_f02;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.02 --duration "$DUR"

  # Oracle-threshold sweep. At the default 0.3 the S16 oracle NEVER opens the
  # gate: measured p_train stayed 0.66-0.88 for 600 s, so the only 3 training
  # steps were cold-start allows and the attack degenerates to not training.
  # One degenerate point is not a result; these trace the frontier, the MN-O
  # analogue of MN-I's --train-frac sweep. A looser threshold buys real training
  # steps at the price of scoring nearer the detector's alert region.
  c=s16_advtrain_oracle_tau50;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.5 --oracle-model "$ORACLE_MODEL" --duration "$DUR"
  c=s16_advtrain_oracle_tau70;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.7 --oracle-model "$ORACLE_MODEL" --duration "$DUR"
  c=s16_advtrain_oracle_tau90;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.9 --oracle-model "$ORACLE_MODEL" --duration "$DUR"

  # ---- HONEST controls: what actually makes filler_only score p_train ~0.49? --
  # filler_only never trains yet alerts on 100% of windows, while honest H100
  # non-ML scores 0.001 -- including the very cuFFT/ethash kernels it runs. It
  # confounds three things: a resident sharded model, the filler's compute, and
  # the deadline's collectives. These three cells separate them. Resident
  # footprint measured from the pilot: 1.4 GB at s16.
  #   stalled_train  model  + idle    + deadline comm   (labelled ml_training)
  #   nonml_hpc      dummy  + filler  + deadline comm   (labelled other)
  #   nonml_comm     none   + heavy all_reduce          (labelled other)
  # nonml_hpc is the decisive one: identical to filler_only except the model is a
  # dummy allocation of the same size. If it also scores ~0.49 the tell is just
  # "big allocation + busy GPU" and we have a specificity problem, not a result.
  c=s16_stalled_train_fsdp1b;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/stalled_training.py --size 1b --duration "$DUR"
  c=s16_nonml_hpc;              mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 1.4 --duration "$DUR"
  c=s16_nonml_comm;             mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode comm --comm-mb 256 --duration "$DUR"

  # ---- MN-I RE-RUN with a saturating filler + the missing honest profiles ----
  # The first MN-I ladder measured nothing about evasion: its filler ran at 55%
  # utilisation / 160 W, while the honest non-ML class it was imitating sits at
  # 91.7% / 509 W (cufft_benchmark) or 49.7% / 122 W (mining_ethash_proxy). The
  # disguise never reached the region it aimed for, so being detected was
  # uninformative. NonMLFiller now does ~2000x the work per launch.
  # DELIBERATELY NEW LABELS (satint/satfiller/nonml_sat): the old cells are
  # already in the corpus under advtrain_interleave_*/nonml_hpc, and reusing
  # those names would silently pool two different filler implementations.
  c=s16_advtrain_satfiller_only;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.0 --duration "$DUR"
  c=s16_advtrain_satint_f05;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.05 --duration "$DUR"
  c=s16_advtrain_satint_f20;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.2 --duration "$DUR"
  c=s16_advtrain_satint_f50;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.5 --duration "$DUR"
  # Honest non-ML, labelled `other`. nonml_sat is the like-for-like partner of
  # satfiller_only (same kernels, same alloc, no model). himem probes the region
  # above ~2 GB where the benign class has almost no support (its only
  # high-memory member is nbody_short_dur120, 132 windows). lowpower probes the
  # partial-utilisation region our earlier cells fell into by accident.
  c=s16_nonml_sat;               mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 1.4 --duty 1.0 --duration "$DUR"
  c=s16_nonml_himem;             mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 16 --duty 1.0 --duration "$DUR"
  c=s16_nonml_lowpower;          mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.3 --duration "$DUR"

  # ---- honest partial-utilisation variety, for a credible FALSE-POSITIVE rate ----
  # nonml_lowpower (duty 0.3) scored p_train 0.323 -- ABOVE the F=0.05 disguise at
  # 0.295 -- so an honest half-idle HPC job is flagged harder than disguised
  # training. That kills tau=0.1 outright (100% FP). One workload at 6 runs is far
  # too thin to calibrate a threshold against, so vary all three axes that move a
  # non-ML job around this region: duty (utilisation/power), resident allocation,
  # and kernel mix. All labelled `other` -- these are genuinely honest workloads.
  c=s16_nonml_lp_d15;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.15 --filler-mode mixed --duration "$DUR"
  c=s16_nonml_lp_d50;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.5 --filler-mode mixed --duration "$DUR"
  c=s16_nonml_lp_d70;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.7 --filler-mode mixed --duration "$DUR"
  c=s16_nonml_lp_d30m16;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 16 --duty 0.3 --filler-mode mixed --duration "$DUR"
  c=s16_nonml_lp_d50fft;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 8 --duty 0.5 --filler-mode fft --duration "$DUR"
  c=s16_nonml_lp_d50hash;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 8 --duty 0.5 --filler-mode hash --duration "$DUR"
  c=s16_inferrep_qwen1_5b_bs1;       mn_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL_BS1"
  # ---- Genuinely-multi-node cells (findings section 22) --------------------
  # s16_train_fsdp30b does NOT need two nodes: 30b is 32.5B params, so FSDP +
  # AdamW is ~390 GB against 640 GB on one 8x H100-80GB node. It spans two
  # nodes only because we launched it that way, which makes it the wrong
  # baseline for "a training job too large for one machine".
  #
  #   70b = 78.4B params -> 157 GB bf16 params + 157 GB grads + 627 GB fp32
  #   AdamW moments = ~940 GB, which does NOT fit on one node and does fit
  #   across two. H100 ONLY -- it does not fit on the A100-40GB pair's 640 GB.
  #
  # MN-P is the matching disguise: FSDP within each node (parameter traffic on
  # NVLink), pipeline across nodes (NIC carries only activations + their
  # grads). The microbatch ladder is the evasion-cost curve -- mb1 is the
  # quietest NIC and the worst bubble, mb16 the reverse.
  #
  # Run at 1b, matching s16_train_fsdp1b's --batch-size/--seq-len exactly, so
  # the ONLY difference between the honest cell and the disguise is which
  # collectives cross the node boundary. A 30b MN-P cell would also be
  # forced-multi-node on the A100 pair, but its stage is 16.25B parameters =
  # 32.5 GB of host bf16 init per rank (260 GB across the 8 ranks of a node),
  # which is slow enough to distort a 600 s cell; the forced-multi-node
  # property is established by s16_train_fsdp70b on H100 instead.
  # Measured at 1b on the A100 pair: 373 Mbps inter-node against
  # s16_train_fsdp1b's 16,491 Mbps, a 44x reduction.
  c=s16_train_fsdp70b;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/fsdp_training.py --size 70b --batch-size 1 --seq-len 512 --duration "$DUR"

  # ---- MULTI-NODE-OBLIGATORY cells ----------------------------------------
  # A cell belongs here only if the JOB CANNOT RUN on one node. On H100 a node
  # is 8 x 80 GB = 640 GB; the pair is 1280 GB.
  #
  #   train_fsdp70b (above)     78.4B params, FSDP+AdamW ~940 GB      > 640
  #   infershard_qwen72b_kv32k  weights 145 GB + 64 x ~32k tokens of KV at
  #                             0.328 MB/token = 675 GB -> ~820 GB    > 640
  #   nonml_himem_obl           44 GB x 16 GPUs = 704 GB, needs >8 GPUs
  #   advtrain_pipe70b_mb1      the SAME 940 GB job as train_fsdp70b, run as
  #                             pipeline-across-nodes instead of FSDP
  #
  # Everything else in this queue spans two nodes because it was LAUNCHED that
  # way. These four cannot be run any other way, which is the regime a frontier
  # training run actually lives in.
  #
  # Qwen2.5-72B's weights alone are only 145 GB and fit on one node, so the
  # obligation is carried by KV cache and therefore by --prompt-tokens: raising
  # --max-model-len alone caps sequence length without lengthening sequences,
  # and the stock ~40-token prompts leave the cache nearly empty.
  c=s16_infershard_qwen72b_kv32k; mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --max-model-len 32768 --prompt-tokens 32000 --max-tokens 128 \
      --batch-size 64 --gpu-memory-utilization 0.92
  c=s16_nonml_himem_obl;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 44 --duty 1.0 --duration "$DUR"
  # 1b first, as a cheap proof of the rank0-init/broadcast path before the 70b
  # cell spends ~78 GB of host init and 45 min on it.
  c=s16_advtrain_pipe1b_rank0; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/adversarial_multinode_pipeline.py --size 1b --rank0-init 1 --microbatches 1 --batch-size 8 --seq-len 1024 --duration "$DUR"
  c=s16_advtrain_pipe70b_mb1;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" 2700 -- workloads/adversarial_multinode_pipeline.py --size 70b --rank0-init 1 --microbatches 1 --batch-size 1 --seq-len 512 --duration "$DUR"
  c=s16_advtrain_pipe1b_mb1;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/adversarial_multinode_pipeline.py --size 1b --microbatches 1 --batch-size 8 --seq-len 1024 --duration "$DUR"
  c=s16_advtrain_pipe1b_mb4;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/adversarial_multinode_pipeline.py --size 1b --microbatches 4 --batch-size 8 --seq-len 1024 --duration "$DUR"
  c=s16_advtrain_pipe1b_mb16;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/adversarial_multinode_pipeline.py --size 1b --microbatches 16 --batch-size 8 --seq-len 1024 --duration "$DUR"

  # ---- Training-vs-HPC disguise (per-GPU channel, not the NIC) -------------
  # Section 21 found AdamW's startup memory delta the sharpest single tell
  # (first_30s_mem_delta_mb 2601 SGD -> 7820 AdamW). MN-F reserves the measured
  # peak into the caching allocator BEFORE model init so the job begins at its
  # plateau; MN-B fills the collective wait with the same hash/fft kernels the
  # honest `other` class uses. Both train EVERY step, so unlike MN-I the cost is
  # contention only. Arena targets measured from this corpus: H100 9216 MB,
  # A100-40GB 7680 MB (mem_used_mb max over honest s16_train_fsdp1b streams).
  ARENA_MB=${ARENA_MB:-9216}
  c=s16_advtrain_flatarena;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-F  --size 1b --arena-mb "$ARENA_MB" --duration "$DUR"
  c=s16_advtrain_bubble;     mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-B  --size 1b --bubble-frac 1.0 --duration "$DUR"
  c=s16_advtrain_flatbubble; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-FB --size 1b --arena-mb "$ARENA_MB" --bubble-frac 1.0 --duration "$DUR"

  if [ "$rep" -le 2 ]; then
    c=s16_train_fsdp30b;           mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/fsdp_training.py --size 30b --batch-size 1 --seq-len 512 --duration "$DUR"
    c=s16_train_ddpstagger_k;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/adversarial_ddp.py --strategy L-stagger --dilution 5 --duration "$DUR"
    c=s16_train_ddpkaccum8;        mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/adversarial_ddp.py --strategy K-accum --accum 8 --duration "$DUR"
  fi

  # -- Phase B: S8 (each node runs its own single-node job) --
  c=s8_train_fsdp1b;               s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/fsdp_training.py --size 1b --duration "$DUR"
  c=s8_train_ddp_resnet50;         s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/pytorch_training_ddp.py --model resnet50 --duration "$DUR"
  c=s8_infershard_qwen72b_tp8;     s8_shard "$c" "$(rid_for "$c" "$rep")" "$SHARD_MODEL" 8 1
  c=s8_infershard_qwen1_5b_tp4pp2; s8_shard "$c" "$(rid_for "$c" "$rep")" "$SHARD_MODEL_SMALL" 4 2
  c=s8_inferrep_qwen1_5b;            mn_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL"
  c=s8_advtrain_interleave_f50;    s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.5 --duration "$DUR"
  c=s8_advtrain_interleave_f20;    s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.2 --duration "$DUR"
  c=s8_advtrain_interleave_f10;    s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.1 --duration "$DUR"
  c=s8_advtrain_oracle;            s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-model "$ORACLE_MODEL" --duration "$DUR"

  # filler_only is the CONTROL that decides whether MN-I can work at all: the
  # disguise blends training with this filler, so if the filler ALONE does not
  # read as `other` there is nothing for the blend to hide behind, and any
  # MN-I result would be uninterpretable. --train-frac 0 now genuinely trains
  # nothing (the old 1/period floor made that impossible).
  c=s8_advtrain_filler_only;    s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.0 --duration "$DUR"
  # Fine end of the sweep: where does detection actually break, and what does
  # that cost? Only meaningful now that fractions below 0.05 are not clamped.
  c=s8_advtrain_interleave_f05;    s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.05 --duration "$DUR"
  c=s8_advtrain_interleave_f02;    s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.02 --duration "$DUR"

  # Oracle-threshold sweep. At the default 0.3 the S16 oracle NEVER opens the
  # gate: measured p_train stayed 0.66-0.88 for 600 s, so the only 3 training
  # steps were cold-start allows and the attack degenerates to not training.
  # One degenerate point is not a result; these trace the frontier, the MN-O
  # analogue of MN-I's --train-frac sweep. A looser threshold buys real training
  # steps at the price of scoring nearer the detector's alert region.
  c=s8_advtrain_oracle_tau50;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.5 --oracle-model "$ORACLE_MODEL" --duration "$DUR"
  c=s8_advtrain_oracle_tau70;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.7 --oracle-model "$ORACLE_MODEL" --duration "$DUR"
  c=s8_advtrain_oracle_tau90;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.9 --oracle-model "$ORACLE_MODEL" --duration "$DUR"

  # ---- HONEST controls: what actually makes filler_only score p_train ~0.49? --
  # filler_only never trains yet alerts on 100% of windows, while honest H100
  # non-ML scores 0.001 -- including the very cuFFT/ethash kernels it runs. It
  # confounds three things: a resident sharded model, the filler's compute, and
  # the deadline's collectives. These three cells separate them. Resident
  # footprint measured from the pilot: 1.8 GB at s8.
  #   stalled_train  model  + idle    + deadline comm   (labelled ml_training)
  #   nonml_hpc      dummy  + filler  + deadline comm   (labelled other)
  #   nonml_comm     none   + heavy all_reduce          (labelled other)
  # nonml_hpc is the decisive one: identical to filler_only except the model is a
  # dummy allocation of the same size. If it also scores ~0.49 the tell is just
  # "big allocation + busy GPU" and we have a specificity problem, not a result.
  c=s8_stalled_train_fsdp1b;   s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/stalled_training.py --size 1b --duration "$DUR"
  c=s8_nonml_hpc;              s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 1.8 --duration "$DUR"
  c=s8_nonml_comm;             s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode comm --comm-mb 256 --duration "$DUR"

  # ---- MN-I RE-RUN with a saturating filler + the missing honest profiles ----
  # The first MN-I ladder measured nothing about evasion: its filler ran at 55%
  # utilisation / 160 W, while the honest non-ML class it was imitating sits at
  # 91.7% / 509 W (cufft_benchmark) or 49.7% / 122 W (mining_ethash_proxy). The
  # disguise never reached the region it aimed for, so being detected was
  # uninformative. NonMLFiller now does ~2000x the work per launch.
  # DELIBERATELY NEW LABELS (satint/satfiller/nonml_sat): the old cells are
  # already in the corpus under advtrain_interleave_*/nonml_hpc, and reusing
  # those names would silently pool two different filler implementations.
  c=s8_advtrain_satfiller_only;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.0 --duration "$DUR"
  c=s8_advtrain_satint_f05;      s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.05 --duration "$DUR"
  c=s8_advtrain_satint_f20;      s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.2 --duration "$DUR"
  c=s8_advtrain_satint_f50;      s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.5 --duration "$DUR"
  # Honest non-ML, labelled `other`. nonml_sat is the like-for-like partner of
  # satfiller_only (same kernels, same alloc, no model). himem probes the region
  # above ~2 GB where the benign class has almost no support (its only
  # high-memory member is nbody_short_dur120, 132 windows). lowpower probes the
  # partial-utilisation region our earlier cells fell into by accident.
  c=s8_nonml_sat;               s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 1.8 --duty 1.0 --duration "$DUR"
  c=s8_nonml_himem;             s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 16 --duty 1.0 --duration "$DUR"
  c=s8_nonml_lowpower;          s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.3 --duration "$DUR"

  # ---- honest partial-utilisation variety, for a credible FALSE-POSITIVE rate ----
  # nonml_lowpower (duty 0.3) scored p_train 0.323 -- ABOVE the F=0.05 disguise at
  # 0.295 -- so an honest half-idle HPC job is flagged harder than disguised
  # training. That kills tau=0.1 outright (100% FP). One workload at 6 runs is far
  # too thin to calibrate a threshold against, so vary all three axes that move a
  # non-ML job around this region: duty (utilisation/power), resident allocation,
  # and kernel mix. All labelled `other` -- these are genuinely honest workloads.
  c=s8_nonml_lp_d15;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.15 --filler-mode mixed --duration "$DUR"
  c=s8_nonml_lp_d50;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.5 --filler-mode mixed --duration "$DUR"
  c=s8_nonml_lp_d70;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.7 --filler-mode mixed --duration "$DUR"
  c=s8_nonml_lp_d30m16;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 16 --duty 0.3 --filler-mode mixed --duration "$DUR"
  c=s8_nonml_lp_d50fft;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 8 --duty 0.5 --filler-mode fft --duration "$DUR"
  c=s8_nonml_lp_d50hash;  s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 8 --duty 0.5 --filler-mode hash --duration "$DUR"
  c=s8_inferrep_qwen1_5b_bs1;        mn_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL_BS1"

  # -- Phase C: S1 (single GPU, no collectives -- the degenerate baseline) --
  c=s1_train_fsdp1b;               s1_train "$c" "$(rid_for "$c" "$rep")"                   -- workloads/fsdp_training.py --size 1b --duration "$DUR"
  c=s1_train_ddp_resnet50;         s1_train "$c" "$(rid_for "$c" "$rep")"                   -- workloads/pytorch_training_ddp.py --model resnet50 --duration "$DUR"
  c=s1_inferrep_qwen1_5b;            s1_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL"
  # S1 note: with one GPU there is no collective, so MN-O here reduces to the
  # single-GPU oracle attack. Expect it to behave differently from S8/S16 by
  # construction -- that difference IS the scope result.
  c=s1_advtrain_interleave_f50;    s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.5 --duration "$DUR"
  c=s1_advtrain_interleave_f20;    s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.2 --duration "$DUR"
  c=s1_advtrain_interleave_f10;    s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.1 --duration "$DUR"
  c=s1_advtrain_oracle;            s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-model "$ORACLE_MODEL" --duration "$DUR"

  # filler_only is the CONTROL that decides whether MN-I can work at all: the
  # disguise blends training with this filler, so if the filler ALONE does not
  # read as `other` there is nothing for the blend to hide behind, and any
  # MN-I result would be uninterpretable. --train-frac 0 now genuinely trains
  # nothing (the old 1/period floor made that impossible).
  c=s1_advtrain_filler_only;    s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.0 --duration "$DUR"
  # Fine end of the sweep: where does detection actually break, and what does
  # that cost? Only meaningful now that fractions below 0.05 are not clamped.
  c=s1_advtrain_interleave_f05;    s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.05 --duration "$DUR"
  c=s1_advtrain_interleave_f02;    s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.02 --duration "$DUR"

  # Oracle-threshold sweep. At the default 0.3 the S16 oracle NEVER opens the
  # gate: measured p_train stayed 0.66-0.88 for 600 s, so the only 3 training
  # steps were cold-start allows and the attack degenerates to not training.
  # One degenerate point is not a result; these trace the frontier, the MN-O
  # analogue of MN-I's --train-frac sweep. A looser threshold buys real training
  # steps at the price of scoring nearer the detector's alert region.
  c=s1_advtrain_oracle_tau50;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.5 --oracle-model "$ORACLE_MODEL" --duration "$DUR"
  c=s1_advtrain_oracle_tau70;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.7 --oracle-model "$ORACLE_MODEL" --duration "$DUR"
  c=s1_advtrain_oracle_tau90;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-O --size 1b --dilution "$ADV_DILUTION" --oracle-threshold 0.9 --oracle-model "$ORACLE_MODEL" --duration "$DUR"

  # ---- HONEST controls: what actually makes filler_only score p_train ~0.49? --
  # filler_only never trains yet alerts on 100% of windows, while honest H100
  # non-ML scores 0.001 -- including the very cuFFT/ethash kernels it runs. It
  # confounds three things: a resident sharded model, the filler's compute, and
  # the deadline's collectives. These three cells separate them. Resident
  # footprint measured from the pilot: 10.7 GB at s1.
  #   stalled_train  model  + idle    + deadline comm   (labelled ml_training)
  #   nonml_hpc      dummy  + filler  + deadline comm   (labelled other)
  #   nonml_comm     none   + heavy all_reduce          (labelled other)
  # nonml_hpc is the decisive one: identical to filler_only except the model is a
  # dummy allocation of the same size. If it also scores ~0.49 the tell is just
  # "big allocation + busy GPU" and we have a specificity problem, not a result.
  c=s1_stalled_train_fsdp1b;   s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/stalled_training.py --size 1b --duration "$DUR"
  c=s1_nonml_hpc;              s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 10.7 --duration "$DUR"
  c=s1_nonml_comm;             s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode comm --comm-mb 256 --duration "$DUR"

  # ---- MN-I RE-RUN with a saturating filler + the missing honest profiles ----
  # The first MN-I ladder measured nothing about evasion: its filler ran at 55%
  # utilisation / 160 W, while the honest non-ML class it was imitating sits at
  # 91.7% / 509 W (cufft_benchmark) or 49.7% / 122 W (mining_ethash_proxy). The
  # disguise never reached the region it aimed for, so being detected was
  # uninformative. NonMLFiller now does ~2000x the work per launch.
  # DELIBERATELY NEW LABELS (satint/satfiller/nonml_sat): the old cells are
  # already in the corpus under advtrain_interleave_*/nonml_hpc, and reusing
  # those names would silently pool two different filler implementations.
  c=s1_advtrain_satfiller_only;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.0 --duration "$DUR"
  c=s1_advtrain_satint_f05;      s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.05 --duration "$DUR"
  c=s1_advtrain_satint_f20;      s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.2 --duration "$DUR"
  c=s1_advtrain_satint_f50;      s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/adversarial_multinode.py --strategy MN-I --size 1b --train-frac 0.5 --duration "$DUR"
  # Honest non-ML, labelled `other`. nonml_sat is the like-for-like partner of
  # satfiller_only (same kernels, same alloc, no model). himem probes the region
  # above ~2 GB where the benign class has almost no support (its only
  # high-memory member is nbody_short_dur120, 132 windows). lowpower probes the
  # partial-utilisation region our earlier cells fell into by accident.
  c=s1_nonml_sat;               s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 10.7 --duty 1.0 --duration "$DUR"
  c=s1_nonml_himem;             s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 16 --duty 1.0 --duration "$DUR"
  c=s1_nonml_lowpower;          s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.3 --duration "$DUR"

  # ---- honest partial-utilisation variety, for a credible FALSE-POSITIVE rate ----
  # nonml_lowpower (duty 0.3) scored p_train 0.323 -- ABOVE the F=0.05 disguise at
  # 0.295 -- so an honest half-idle HPC job is flagged harder than disguised
  # training. That kills tau=0.1 outright (100% FP). One workload at 6 runs is far
  # too thin to calibrate a threshold against, so vary all three axes that move a
  # non-ML job around this region: duty (utilisation/power), resident allocation,
  # and kernel mix. All labelled `other` -- these are genuinely honest workloads.
  c=s1_nonml_lp_d15;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.15 --filler-mode mixed --duration "$DUR"
  c=s1_nonml_lp_d50;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.5 --filler-mode mixed --duration "$DUR"
  c=s1_nonml_lp_d70;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 2 --duty 0.7 --filler-mode mixed --duration "$DUR"
  c=s1_nonml_lp_d30m16;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 16 --duty 0.3 --filler-mode mixed --duration "$DUR"
  c=s1_nonml_lp_d50fft;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 8 --duty 0.5 --filler-mode fft --duration "$DUR"
  c=s1_nonml_lp_d50hash;  s1_train "$c" "$(rid_for "$c" "$rep")" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 8 --duty 0.5 --filler-mode hash --duration "$DUR"
  c=s1_inferrep_qwen1_5b_bs1;        s1_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL_BS1"
done

"$RAY" stop --force >/dev/null 2>&1 || true
say "queue complete (node$NODE_RANK)."
