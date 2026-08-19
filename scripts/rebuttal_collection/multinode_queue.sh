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
MASTER_ADDR=${MASTER_ADDR:?export MASTER_ADDR to the node0 internal IP}
MASTER_PORT=${MASTER_PORT:-29500}
NODE_RANK=${NODE_RANK:?export NODE_RANK (0 on node0, 1 on node1)}
RUN_STAMP=${RUN_STAMP:?export RUN_STAMP (identical string on both nodes)}
NNODES=${NNODES:-2}

DUR=${DUR:-600}                         # 600 s -> 39 windows at 30 s / 15 s
TIMEOUT=${TIMEOUT:-900}
TIMEOUT_BIG=${TIMEOUT_BIG:-1500}        # 30B / model loads
GAP=${GAP:-60}
REPS=${REPS:-3}
RENDEZVOUS_TIMEOUT=${RENDEZVOUS_TIMEOUT:-120}
INFER_MODEL=${INFER_MODEL:-llm_infer_qwen25_7b_fp16}   # replicated-inference registry label
SHARD_MODEL=${SHARD_MODEL:-qwen25_72b_tp2}             # vLLM sharded model key
SMOKE=${SMOKE:-0}

PREFIX=""
OUTDIR="data"
if [ "$SMOKE" = "1" ]; then
  DUR=60; TIMEOUT=240; TIMEOUT_BIG=600; GAP=10; REPS=1; PREFIX="smoke_"
  OUTDIR="logs/smoke_data"   # keep smoke telemetry out of data/ (loader rglobs data/**)
  mkdir -p logs/smoke_data
  echo "SMOKE MODE: 60 s cells, output to logs/smoke_data/ (NOT usable data)."
fi

mkdir -p logs logs/.done "$OUTDIR"
MASTER=logs/multinode_master.log
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] node$NODE_RANK $*" | tee -a "$MASTER"; }

# Deterministic shared run_id: identical on both nodes for the same cell+rep.
rid_for() { printf '%s' "${RUN_STAMP}_$1_$2" | sha1sum | cut -c1-12; }

# --- host-NIC sidecar: captures the inter-server channel NVML cannot see. ----
SIDE_PID=""
start_sidecar() {  # start_sidecar <label> <run_id>
  $PY scripts/collect_host_telemetry.py --duration $((DUR + 120)) \
      --output "$OUTDIR/nic_${1}_node${NODE_RANK}_${2}.parquet" \
      --run-id "$2" --label "$1" --node-rank "$NODE_RANK" \
      --checkpoint-sec 60 >> "logs/${1}_node${NODE_RANK}.nic.log" 2>&1 &
  SIDE_PID=$!
}
stop_sidecar() { [ -n "$SIDE_PID" ] && kill "$SIDE_PID" 2>/dev/null; wait "$SIDE_PID" 2>/dev/null; SIDE_PID=""; }

# --- generic stage wrapper: per-node marker, .failed on error, always continue.
run_stage() {  # run_stage <stage> <run_id> -- <command...>
  local stage="$1" rid="$2"; shift 2; [ "$1" = "--" ] && shift
  local marker="logs/.done/${stage}_node${NODE_RANK}.done"
  local logf="logs/${stage}_node${NODE_RANK}.log"
  if [ -f "$marker" ]; then say "SKIP  $stage ($(cat "$marker"))"; return 0; fi

  local nproc; nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | wc -l)
  [ "$nproc" -ne 0 ] && say "WARN  $stage: $nproc compute process(es) already resident"

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
    echo "$(stamp) rc=$rc dur=${dur}s rid=$rid" > "logs/.done/${stage}_node${NODE_RANK}.failed"
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
      torchrun --nnodes="$NNODES" --node-rank="$NODE_RANK" \
        --rdzv-backend=c10d --rdzv-endpoint="${MASTER_ADDR}:${MASTER_PORT}" \
        --rdzv-conf="join_timeout=${RENDEZVOUS_TIMEOUT}" --nproc-per-node=8 \
        "$@"
}

# Sharded vLLM inference across both nodes via Ray. Asymmetric: node0 is the
# Ray head and drives the job + its own telemetry; node1 joins the Ray cluster
# and only records telemetry (its GPUs are driven by the remote Ray worker).
mn_ray_shard() {  # mn_ray_shard <stage> <run_id>
  local stage="$1" rid="$2"
  if [ "$NODE_RANK" -eq 0 ]; then
    ray start --head --port=6379 --num-gpus=8 >/dev/null 2>&1 || true
    sleep 10
    run_stage "$stage" "$rid" -- \
      $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
        --num-gpus 8 --run-id "$rid" --node-rank 0 --output-dir "$OUTDIR" \
        --timeout "$TIMEOUT_BIG" -- \
        $PY_VLLM workloads/llm_inference_vllm.py --model "$SHARD_MODEL" \
          --tensor-parallel-size 8 --pipeline-parallel-size 2 \
          --distributed-executor-backend ray --duration "$DUR"
    ray stop >/dev/null 2>&1 || true
  else
    ray start --address="${MASTER_ADDR}:6379" --num-gpus=8 >/dev/null 2>&1 || true
    # node1 records telemetry for the shared run_id while node0 drives vLLM.
    run_stage "$stage" "$rid" -- \
      $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
        --num-gpus 8 --run-id "$rid" --node-rank 1 --telemetry-only \
        --duration "$DUR" --output-dir "$OUTDIR"
    ray stop >/dev/null 2>&1 || true
  fi
}

# Replicated inference: 8 independent single-GPU replicas per node, no
# rendezvous, per-replica run_ids. Both nodes run this independently.
mn_replicated() {  # mn_replicated <stage> <run_id> <infer-label>
  local stage="$1" rid="$2" label="$3"
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --parallel --workload "$label" \
      --num-gpus 8 --node-rank "$NODE_RANK" --output-dir "$OUTDIR" --timeout "$TIMEOUT"
}

# ── S8 cells (each node runs its own single-node job concurrently) ───────────
s8_torchrun() {  # s8_torchrun <stage> <run_id> <timeout> -- <workload.py args...>
  local stage="$1" rid="$2" to="$3"; shift 3; [ "$1" = "--" ] && shift
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
      --num-gpus 8 --run-id "$rid" --node-rank "$NODE_RANK" \
      --output-dir "$OUTDIR" --timeout "$to" -- \
      torchrun --standalone --nproc-per-node=8 "$@"
}

# ── S1 cells (single-GPU: world=1 training, no collectives) ──────────────────
s1_train() {  # s1_train <stage> <run_id> -- <workload.py args...>
  local stage="$1" rid="$2"; shift 2; [ "$1" = "--" ] && shift
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
      --num-gpus 1 --run-id "$rid" --node-rank "$NODE_RANK" \
      --output-dir "$OUTDIR" --timeout "$TIMEOUT" -- \
      torchrun --standalone --nproc-per-node=1 "$@"
}
s1_replicated() {  # s1_replicated <stage> <run_id> <label>
  mn_replicated "$1" "$2" "$3"
}

# ── The queue: Phase A (S16) -> B (S8) -> C (S1), rep-major, priority-first ──

say "queue start: NNODES=$NNODES NODE_RANK=$NODE_RANK MASTER=$MASTER_ADDR:$MASTER_PORT REPS=$REPS DUR=$DUR SMOKE=$SMOKE"

for rep in $(seq 1 "$REPS"); do
  say "===== rep $rep / $REPS ====="

  # -- Phase A: S16 (needs both nodes). Priority 1 cells first. --
  c=s16_train_fsdp7b;              mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/fsdp_training.py --size 7b --duration "$DUR"
  c=s16_infershard_vllm_tp8pp2;    mn_ray_shard "$c" "$(rid_for "$c" "$rep")"
  c=s16_train_ddp_resnet50;        mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/pytorch_training_ddp.py --model resnet50 --duration "$DUR"
  c=s16_inferrep_qwen7b;           mn_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL"
  if [ "$rep" -le 2 ]; then
    c=s16_train_fsdp30b;           mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/fsdp_training.py --size 30b --batch-size 1 --seq-len 512 --duration "$DUR"
    c=s16_train_ddpstagger_k;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/adversarial_ddp.py --strategy L-stagger --dilution 5 --duration "$DUR"
    c=s16_train_ddpkaccum8;        mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/adversarial_ddp.py --strategy K-accum --accum 8 --duration "$DUR"
  fi

  # -- Phase B: S8 (each node runs its own single-node job) --
  c=s8_train_fsdp7b;               s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/fsdp_training.py --size 7b --duration "$DUR"
  c=s8_train_ddp_resnet50;         s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/pytorch_training_ddp.py --model resnet50 --duration "$DUR"
  c=s8_infershard_vllm_tp8;        run_stage "$c" "$(rid_for "$c" "$rep")" -- \
                                     $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${c}" --num-gpus 8 \
                                       --run-id "$(rid_for "$c" "$rep")" --node-rank "$NODE_RANK" --output-dir "$OUTDIR" --timeout "$TIMEOUT_BIG" -- \
                                       $PY_VLLM workloads/llm_inference_vllm.py --model "$SHARD_MODEL" --tensor-parallel-size 8 --duration "$DUR"
  c=s8_inferrep_qwen7b;            mn_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL"

  # -- Phase C: S1 (single GPU, no collectives -- the degenerate baseline) --
  c=s1_train_fsdp7b;               s1_train "$c" "$(rid_for "$c" "$rep")"                   -- workloads/fsdp_training.py --size 7b --duration "$DUR"
  c=s1_inferrep_qwen7b;            s1_replicated "$c" "$(rid_for "$c" "$rep")" "$INFER_MODEL"
done

say "queue complete (node$NODE_RANK)."
