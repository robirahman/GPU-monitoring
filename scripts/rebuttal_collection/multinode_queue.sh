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
export HF_HOME=${HF_HOME:-${HOME:-/root}/.cache/huggingface}

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
# Serving cells start a vLLM SERVER and then drive it, so the outer timeout has
# to cover weight load + engine init + --duration, not just a generate loop.
# Measured 2026-09-21: every inferserve cell was killed at rc=124 after 1564 s
# against TIMEOUT_BIG=1500 while its own --startup-timeout was 2400.
TIMEOUT_SERVE=${TIMEOUT_SERVE:-3600}
GAP=${GAP:-60}
REPS=${REPS:-3}
# 120 s was too short. The sharded-inference cells are asymmetric -- node0
# drives vLLM/Ray for the whole duration while node1 only records telemetry --
# so node1 reaches the NEXT torchrun cell first and its join_timeout expires
# before node0 arrives. That is what failed s16_train_fsdp7b_seq2k twice on
# 2026-09-08 (throwTimeoutError at 147 s and 71 s), with nothing wrong in the
# workload itself.
RENDEZVOUS_TIMEOUT=${RENDEZVOUS_TIMEOUT:-600}
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

# Per-GPU memory, measured. Some cells simply do not fit on a 40 GB card and
# there is no point discovering that by OOM on a rented pair: s16_train_lora7b
# allocates 36.9 GB of fp32 weights per rank (DDP replicates the whole model, and
# LoRA freezes parameters -- it does not shrink them) and died on all 8 ranks of
# an A100-40GB, 2026-09-23. A skipped cell costs nothing; an OOMed cell costs its
# whole timeout and writes a .failed marker that looks like a real result.
GPU_MEM_GB="${GPU_MEM_GB:-$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null \
  | head -1 | awk '{printf "%d", $1/1024}')}"
need_gb() {  # need_gb <gb-per-gpu> <stage> -- true if this card is big enough
  local need="$1" stage="$2"
  if [ -n "${GPU_MEM_GB:-}" ] && [ "${GPU_MEM_GB:-0}" -ge "$need" ] 2>/dev/null; then
    return 0
  fi
  say "SKIP  $stage (needs ${need} GB/GPU, this card reports ${GPU_MEM_GB:-unknown} GB)"
  return 1
}

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
  # Graceful stop. preempt_watch.sh touches logs/.preempted on the ~30 s ACPI
  # notice, with the comment "mark so the collection queue stops launching new
  # cells" -- but NOTHING EVER READ THAT MARK. The queue kept starting cells into
  # a node about to be destroyed, and each died mid-collective and wrote a
  # .failed marker indistinguishable from a real workload fault. The same file
  # also gives an operator a way to retarget a running queue without a remote
  # kill (which this project cannot do): touch logs/.stop, let the current cell
  # finish, then re-arm with a different ONLY_CELLS.
  if [ -f "logs/.preempted" ] || [ -f "logs/.stop" ]; then
    say "STOP  $stage (logs/.preempted or logs/.stop present -- launching no new cells)"
    return 0
  fi
  # rid = sha1(RUN_STAMP_cell_rep), so keying the marker on it makes it unique
  # per REP. Keyed on cell alone, rep 2 and rep 3 saw rep 1's marker and skipped
  # every cell -- REPS=3 silently collected a single rep.
  #
  # Clear any bandwidth shaping before EVERY cell, shaped or not. A cell that
  # died mid-run with a qdisc installed would silently cap every later cell in
  # the queue, and an unshaped cell recorded under a shaped name is worse than a
  # missing one. Reset-on-entry is the guarantee; the trap in mn_torchrun_bw
  # only covers a clean exit.
  #
  # EXCEPT when the caller installed the shaping on purpose. mn_torchrun_bw
  # shapes and THEN calls mn_torchrun, which lands here -- so for the first four
  # _bw100 cells ever run, 2026-10-03, this line tore down the qdisc seconds
  # after it went up and every one of them collected at line rate under a
  # shaped name: exactly the outcome the comment above calls worse than a
  # missing cell. netshape.sh had already written {"applied": true} and was
  # telling the truth about the moment it was asked.
  if [ "${NETSHAPE_OWNED:-0}" = "1" ]; then
    say "      (shaping owned by this cell -- not clearing it)"
  else
    bash "${GCPDIR:-/opt/spar/gcp}/netshape.sh" down >/dev/null 2>&1 || true
  fi

  local marker="${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.done"
  local logf="logs/${PREFIX}${stage}_node${NODE_RANK}_${rid}.log"
  if [ -f "$marker" ]; then say "SKIP  $stage ($(cat "$marker"))"; return 0; fi

  # Realign with the peer before a cell that has to rendezvous. Placed after
  # the filter and marker checks so a skipped cell costs no GCS round-trip and
  # a node that already holds this cell's marker never blocks on its peer.
  cell_barrier "$stage" "$rid"

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
# Cross-node barrier before a rendezvous cell. Raising join_timeout does not
# fix this and 600 s did not: the RAY cells are asymmetric, node0 drives vLLM for
# 678-962 s while node1 only records telemetry and returns in 600-601 s, so each
# one puts node1 minutes ahead. After a few cells the two nodes are on DIFFERENT
# cells at the same time and a torchrun cell can never rendezvous however long
# it waits -- s16_train_fsdp7b_seq2k failed that way at both 120 s and 600 s.
# A marker each side publishes and waits for realigns them per cell.
cell_barrier() {  # cell_barrier <stage> <rid>
  [ "${NNODES:-1}" -gt 1 ] || return 0
  # NODE_NAME is NOT in the queue's environment on either launch path -- verified
  # by reading /proc/<queue pid>/environ on the live A100-80 node, which has
  # NNODES and PEER but no NODE_NAME. So this guard was returning 0 SILENTLY on
  # every cell and the barrier has never once run; that is why no barrier line
  # has ever appeared in any queue log. Fall back to the hostname, which is what
  # node_env's NODE_NAME is derived from anyway.
  local stage="$1" rid="$2" me="${NODE_NAME:-$(hostname -s)}" peer="" ctrl
  [ -n "$me" ] || { say "  barrier $stage: WARNING cannot determine this node's name -- NOT waiting"; return 0; }
  # PEER FIRST. Deriving the peer from NODE0/NODE1 in node_env is wrong whenever
  # the pair is not named spar-gpu-0/1: env.sh exports NODE0/NODE1 with those
  # defaults and bootstrap writes them into node_env verbatim, so on the
  # A100-80GB pair (spar-gpua-0/1) node_env said NODE0=spar-gpu-0 and every node
  # resolved its peer to a host that does not exist in that region. The barrier
  # then never aligned, the two nodes drifted a whole cell apart, and once the
  # faster node finished its pass the slower one lost its rendezvous partner --
  # which is what failed every training cell on 2026-09-21 with
  # RendezvousTimeoutError and then ray-cluster-incomplete.
  # run_stage_and_stop.sh already knows the real peer; take it from there.
  if [ -n "${PEER:-}" ] && [ "${PEER}" != "$me" ]; then
    peer="$PEER"
  elif [ -n "${NODE0:-}" ] && [ -n "${NODE1:-}" ] \
       && { [ "$me" = "$NODE0" ] || [ "$me" = "$NODE1" ]; }; then
    if [ "$me" = "$NODE0" ]; then peer="$NODE1"; else peer="$NODE0"; fi
  fi
  # Loud, because a silently-unresolved peer is exactly how the nodes drifted.
  [ -n "$peer" ] || { say "  barrier $stage: WARNING no peer resolved (PEER unset and "\
"NODE_NAME=$me matches neither NODE0=${NODE0:-} nor NODE1=${NODE1:-}) -- NOT waiting"; return 0; }
  # BUCKET and SESSION live in /etc/spar/node_env, which launch_queue.sh SOURCES
  # but does not export -- so this child queue inherits neither. Under `set -u`
  # that made the FIRST barrier call that ever got as far as this line abort the
  # entire queue:
  #   multinode_queue.sh: line 250: BUCKET: unbound variable   2026-09-21T14:41:40Z
  # and because the queue died, launch_queue's STOP_WHEN_DONE never ran, so the
  # A100-80GB pair idled from 14:42Z until its MIG max-run-duration deleted it at
  # 20:46Z -- six hours of 16 idle A100-80GBs. Re-read node_env here rather than
  # relying on the caller's environment, and degrade to "do not wait" instead of
  # killing the run if it is still unresolved.
  if [ -z "${BUCKET:-}" ] || [ -z "${SESSION:-}" ]; then
    # shellcheck disable=SC1091
    [ -r /etc/spar/node_env ] && . /etc/spar/node_env
  fi
  if [ -z "${BUCKET:-}" ] || [ -z "${SESSION:-}" ]; then
    say "  barrier $stage: WARNING BUCKET/SESSION unresolved -- NOT waiting"; return 0
  fi
  ctrl="${BUCKET}/runs/${SESSION}/control/barrier"
  echo "$(stamp)" > "/tmp/barrier.${rid}"
  gcloud storage cp "/tmp/barrier.${rid}" \
    "${ctrl}/${stage}_${rid}_${me}.ready" >/dev/null 2>&1 || true
  local t0=$SECONDS
  while [ $((SECONDS - t0)) -lt "${BARRIER_WAIT:-900}" ]; do
    if gcloud storage ls "${ctrl}/${stage}_${rid}_${peer}.ready" >/dev/null 2>&1; then
      say "  barrier $stage: $peer aligned after $((SECONDS - t0))s"; return 0
    fi
    sleep 10
  done
  say "  barrier $stage: TIMED OUT after ${BARRIER_WAIT:-900}s waiting for $peer -- proceeding"
}

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

# mn_torchrun_bw <stage> <run_id> <timeout> -- <workload.py args...>
# Identical to mn_torchrun except the PEER LINK is capped for the cell's
# duration. Shaping goes on AFTER cell_barrier (which talks to GCS) and comes
# off unconditionally, including on timeout and on signal.
#
# A _bw cell that runs UNSHAPED is a mislabelled cell, and the name is
# load-bearing: rid_for() derives the shared run_id and the GCS barrier key from
# it, so a node cannot rename at runtime without deadlocking against its peer.
# The honest outcome when shaping is unavailable is to skip -- the unshaped half
# of the round still runs and returns its value.
mn_torchrun_bw() {
  local stage="$1" rid="$2" to="$3"; shift 3; [ "$1" = "--" ] && shift
  if ! NETSHAPE_CELL="$stage" NETSHAPE_RID="$rid" \
       bash "${GCPDIR:-/opt/spar/gcp}/netshape.sh" up \
         --ttl "$((to + 300))" --json "$OUTDIR" 2>&1 | grep -q "shaped "; then
    say "SKIP  $stage (shaping unavailable; a _bw cell that is not shaped would"
    say "      be a mislabelled cell, so it is skipped rather than faked)"
    bash "${GCPDIR:-/opt/spar/gcp}/netshape.sh" down >/dev/null 2>&1 || true
    return 0
  fi
  trap 'bash "${GCPDIR:-/opt/spar/gcp}/netshape.sh" down >/dev/null 2>&1 || true' EXIT INT TERM
  export NETSHAPE_OWNED=1          # read by run_stage's reset-on-entry
  mn_torchrun "$stage" "$rid" "$to" -- "$@"
  local rc=$?
  unset NETSHAPE_OWNED
  NETSHAPE_CELL="$stage" NETSHAPE_RID="$rid" \
    bash "${GCPDIR:-/opt/spar/gcp}/netshape.sh" down --json "$OUTDIR" >/dev/null 2>&1 || true
  trap - EXIT INT TERM

  # MEASURE, do not assume. HTB's `overlimits` counter is the only evidence
  # that the cap ever bit; "applied" merely says tc accepted the commands, and
  # that is precisely the claim that was true while the cell ran at line rate.
  # A _bw cell that was not actually throttled is a mislabelled cell, so its
  # .done marker is withdrawn and the next pass re-runs it.
  local js="${OUTDIR}/netshape_${stage}_node${NODE_RANK}_${rid}.json"
  local over
  over="$(grep -oE '"class_overlimits": *[0-9]+' "$js" 2>/dev/null | grep -oE '[0-9]+$')"
  if [ -z "$over" ] || [ "$over" -eq 0 ] 2>/dev/null; then
    say "FAIL  $stage ran UNSHAPED (overlimits=${over:-absent}) -- withdrawing"
    say "      its .done marker so it is collected again rather than trusted"
    rm -f "${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.done"
    return 1
  fi
  say "      shaping confirmed: HTB overlimits=$over"
  return $rc
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

# Sharded SERVING under a real arrival process. Same Ray cluster and the same
# TP8/PP2 shape as mn_ray_shard, but the driver is workloads/llm_serving_async.py:
# a vLLM OpenAI server plus an HTTP load generator, instead of a blocking
# llm.generate() loop over a fixed, uniformly-padded batch.
#
# WHY A SEPARATE LAUNCHER. Every LLM inference cell in the corpus is that
# blocking loop, so every one of them is PERIODIC: prefill burst, decode ramp,
# drain to idle, repeat, at a fixed period with one prompt length and one output
# length. That is a training loop with the backward pass deleted, which is why
# held-out long-context inference is not recognised as inference (section 32b:
# the infershard_qwen family scores 0.0185 when its family is held out). These
# cells have no period to lock onto: Poisson or saturating arrivals, lognormal
# prompt and output lengths, continuous batching with requests always in flight.
mn_ray_serve() {  # mn_ray_serve <stage> <run_id> <model> <mode> <tp> <pp> [extra...]
  local stage="$1" rid="$2" model="$3" mode="$4" tp="${5:-8}" pp="${6:-2}"
  local extra=("${@:7}")
  cell_enabled "$stage" || { say "SKIP  $stage (ONLY_CELLS filter)"; return 0; }
  local raylog="logs/${PREFIX}${stage}_node${NODE_RANK}_${rid}.ray.log"
  local marker="${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.done"
  [ -f "$marker" ] && { say "SKIP  $stage ($(cat "$marker"))"; return 0; }

  ensure_ray_cluster "$raylog" || {
    say "FAIL  $stage: ray-cluster-incomplete -- skipping cell (see $raylog)"
    echo "$(stamp) ray-cluster-incomplete" > "${MARKER_DIR}/${PREFIX}${stage}_node${NODE_RANK}_${rid}.failed"
    return 0
  }

  if [ "$NODE_RANK" -eq 0 ]; then
    run_stage "$stage" "$rid" -- \
      env RAY_ADDRESS="${MASTER_ADDR}:6379" \
      $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
        --num-gpus 8 --run-id "$rid" --node-rank 0 --output-dir "$OUTDIR" \
        --timeout "$TIMEOUT_SERVE" -- \
        $PY_VLLM workloads/llm_serving_async.py --model "$model" --mode "$mode" \
          --tensor-parallel-size "$tp" --pipeline-parallel-size "$pp" \
          --distributed-executor-backend ray --duration "$DUR" "${extra[@]}"
  else
    run_stage "$stage" "$rid" -- \
      $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
        --num-gpus 8 --run-id "$rid" --node-rank 1 --telemetry-only \
        --duration "$DUR" --output-dir "$OUTDIR"
    wait_gpus_idle "$TIMEOUT_BIG" || true
  fi
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

mn_replicated_main() {  # mn_replicated_main <stage> <run_id> <registry-label>
  local stage="$1" rid="$2" label="$3"
  # As mn_replicated, but WITHOUT --workload-python "$PY_VLLM". These cells are
  # torch/transformers/diffusers workloads that must run under venv-main; handing
  # them venv-vllm's interpreter would fail on the first `import torchvision` or
  # `import diffusers`, since that venv is `pip install vllm ray` and nothing
  # else. run_workload_multi_gpu.py is itself launched with $PY, so the registry
  # command's baked-in sys.executable is already venv-main -- which is exactly
  # what these need and exactly what the vLLM cells had to override.
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --parallel --workload "$label" \
      --label "${PREFIX}${stage}" \
      --num-gpus 8 --node-rank "$NODE_RANK" --output-dir "$OUTDIR" --timeout "$TIMEOUT"
}

# ── S8 cells (each node runs its own single-node job concurrently) ───────────
s8_single() {  # s8_single <stage> <run_id> <timeout> -- <workload.py args...>
  # ONE process per node that sees all 8 GPUs, for workloads that shard
  # themselves. llm_inference_hf_sharded.py hands placement to accelerate's
  # device_map and never calls init_process_group; under s8_torchrun's
  # --nproc-per-node=8 all eight replicas each tried to map a 72B model across
  # all eight cards, so GPU 0 held two 15.8 GiB copies plus six stubs and OOMed
  # at 39.49 GiB. Same telemetry: run_workload_multi_gpu still samples 8 GPUs.
  local stage="$1" rid="$2" to="$3"; shift 3; [ "$1" = "--" ] && shift
  run_stage "$stage" "$rid" -- \
    $PY scripts/run_workload_multi_gpu.py --workload "${PREFIX}${stage}" \
      --num-gpus 8 --run-id "$rid" --node-rank "$NODE_RANK" \
      --output-dir "$OUTDIR" --timeout "$to" -- \
      $PY "$@"
}

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
  # ---- PREFILL LADDER: the negatives the detector was missing ------------
  # Section 24: honest long-context inference (72B, 64 x ~31k-token requests)
  # alerted on 48 of 48 runs at p_train 0.864, indistinguishable from honest
  # training's 0.853. At 31k tokens per request the cell is PREFILL-dominated,
  # and prefill is a large dense compute-bound forward pass -- which is what
  # training looks like. Every inference negative the detector was fitted on is
  # decode-dominated and memory-bound, so it had never seen this regime.
  #
  # This ladder varies ONLY prompt length at fixed model and topology, so the
  # prefill fraction is the single moving part and the detector can learn the
  # boundary instead of memorising two disjoint regimes:
  #
  #   kv1k    1,024 tok  ->  decode-dominated (the regime already in the corpus)
  #   kv8k    8,000 tok  ->  mixed
  #   kv32k  32,000 tok  ->  prefill-dominated (already collected)
  #
  # kv32k_out2k is the control that separates "long context" from
  # "prefill-dominated": same 32k prompt, but 2048 output tokens instead of 128,
  # so most of the wall clock is decode again. If the detector keys on context
  # LENGTH it will treat this like kv32k; if it keys on the prefill/decode
  # BALANCE it will not.
  #
  # train_fsdp7b_seq2k is the matching move from the training side -- a longer
  # forward pass per step, so training and inference overlap on activation
  # footprint and utilisation and the discriminator has to find the backward
  # pass and the optimizer state instead. 2048 is the ceiling: pos_emb is
  # nn.Embedding(2048, hidden), and raising it would change the parameter count
  # of every existing training cell, so that is a deliberate change, not a
  # side effect of this ladder.
  c=s16_infershard_qwen72b_kv1k;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --max-model-len 32768 --prompt-tokens 1024 --max-tokens 128 \
      --batch-size 64 --gpu-memory-utilization 0.92
  c=s16_infershard_qwen72b_kv8k;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --max-model-len 32768 --prompt-tokens 8000 --max-tokens 128 \
      --batch-size 64 --gpu-memory-utilization 0.92
  # prompt + output must fit INSIDE max_model_len, and max_model_len cannot
  # exceed the model's own max_position_embeddings. The first version asked for
  # 34816 (=32000+2048+headroom) and vLLM refused at engine init on every rep:
  #   User-specified max_model_len (34816) is greater than the derived
  #   max_model_len (max_position_embeddings=32768.0 ...)
  # The cell then recorded 600 s of IDLE GPUs -- and because node1's role in a
  # ray cell is telemetry-only, the queue logged it as OK. 30000+2048=32048
  # fits. batch stays 64 to match kv32k, and 64*32048 = 2.05M tokens of KV is
  # inside the 3.21M the pair has and above the ~1.34M one node has, so this
  # cell is still multi-node-obligatory.
  c=s16_infershard_qwen72b_kv32k_out2k; mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --max-model-len 32768 --prompt-tokens 30000 --max-tokens 2048 \
      --batch-size 64 --gpu-memory-utilization 0.92

  # TENSOR-PARALLEL SIZE IS MODEL-SPECIFIC. vLLM requires
  # num_attention_heads % tensor_parallel_size == 0. Qwen2.5-7B has 28 heads and
  # 1.5B has 12, so TP8 is INVALID for both -- measured on the A100 pair
  # 2026-09-21, where every 7B/1.5B cell died in ~20 s with
  #   Total number of attention heads (28) must be divisible by
  #   tensor parallel size (8)
  # The 72B has 64 heads and is fine at TP8/PP2. Everything smaller uses TP4/PP4,
  # which divides both 28 and 12 and still spans all 16 GPUs -- the same shape
  # s16_infershard_qwen1_5b_tp4pp4 has always used. NOTE: this means the two 7B
  # and 1.5B cells queued on 09-17 could never have run, on any hardware.
  # ---- Section 25 corpus gaps ---------------------------------------------
  # (1) DIVERSE long-context inference. Section 25 showed that adding kv32k to
  # the corpus fixed kv32k (13/40) while its own siblings, differing only in
  # batch size, still alerted 44/48 and 45/48 -- reps of one cell do not
  # generalise even to that cell's family. So vary MODEL and CONTEXT, not just
  # concurrency. All three models are already cached on the H100 pair and all
  # have max_position_embeddings=32768, so prompt+output stays inside it.
  # NOTE: no --max-model-len. vLLM defaults it to the model's own
  # max_position_embeddings, and specifying it by hand is what killed
  # kv32k_out2k on every rep (34816 > 32768, refused at engine init while the
  # collector recorded 600 s of idle GPUs that the queue logged as OK). Letting
  # vLLM derive it removes that failure mode; prompt+output here is far inside
  # any Qwen2.5 limit.
  c=s16_infershard_qwen7b_kv16k_b32;   mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_7b_fp16 4 4 \
      --prompt-tokens 16000 --max-tokens 128 \
      --batch-size 32 --gpu-memory-utilization 0.92
  c=s16_infershard_qwen1_5b_kv16k_b48; mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_1_5b_fp16 4 4 \
      --prompt-tokens 16000 --max-tokens 128 \
      --batch-size 48 --gpu-memory-utilization 0.92
  c=s16_infershard_qwen72b_kv16k_b24;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 16000 --max-tokens 128 \
      --batch-size 24 --gpu-memory-utilization 0.92
  c=s16_infershard_qwen72b_kv4k_b48;   mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 4000 --max-tokens 128 \
      --batch-size 48 --gpu-memory-utilization 0.92

  # ---- Section 33: APERIODIC serving, and training without a periodic step ----
  # Section 32b measured the gap these fill. Held out by FAMILY, three of four
  # inference families score below 0.32 (infershard_qwen 0.0185) while ten of
  # eleven training families score above 0.83 -- the detector has learned a
  # training-shaped prior and every inference cell we own is shaped like it,
  # because every one is a blocking generate() over a uniformly padded batch.
  #
  # Each --mode below becomes its OWN family under the section-32 family rule
  # (inferserve_poisson, inferserve_saturate, ...), which is the point: the four
  # cells queued on 2026-09-18 all collapse into the single existing
  # infershard_qwen family and so cannot move a family-held-out number.
  #
  # poisson  : exponential arrivals, lognormal prompt AND output lengths.
  # saturate : queue never drains -- busy like training, but with no period.
  # bursty   : heavy/light phases far longer than any training step.
  # prefill  : long prompts, max_tokens=1. The most training-like thing
  #            inference does, isolated so the detector must learn it explicitly.
  # decode   : short prompts, long generations. Bandwidth-bound, the other pole.
  c=s16_inferserve_poisson_7b;   mn_ray_serve "$c" "$(rid_for "$c" "$rep")" qwen25_7b_fp16  poisson  4 4 \
      --qps 6 --concurrency 64
  c=s16_inferserve_saturate_7b;  mn_ray_serve "$c" "$(rid_for "$c" "$rep")" qwen25_7b_fp16  saturate 4 4 \
      --concurrency 96
  c=s16_inferserve_bursty_7b;    mn_ray_serve "$c" "$(rid_for "$c" "$rep")" qwen25_7b_fp16  bursty   4 4 \
      --qps 4 --burst-sec 25 --concurrency 64
  c=s16_inferserve_prefill_72b;  mn_ray_serve "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 prefill  8 2 \
      --qps 1.5 --concurrency 24 --prompt-lo 4000 --prompt-hi 24000
  c=s16_inferserve_decode_72b;   mn_ray_serve "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 decode   8 2 \
      --qps 2 --concurrency 32 --out-lo 512 --out-hi 2048
  c=s16_inferserve_poisson_72b;  mn_ray_serve "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 poisson  8 2 \
      --qps 2 --concurrency 48
  c=s16_inferserve_chunked_72b;  mn_ray_serve "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 poisson  8 2 \
      --qps 2 --concurrency 48 --enable-chunked-prefill

  # ---- MATCHED MEMORY FOOTPRINT: the confound the corpus actually has --------
  # MEASURED 2026-09-19 on the NVIDIA multi-node subset. Memory is the feature the
  # model leans on hardest (6 of its top 8 are mem_used_mb_*), and in this corpus
  # it separates the classes almost perfectly -- in the WRONG direction:
  #
  #   training   median 14,164 MB   (487 .. 53,878)
  #   inference  median 77,026 MB   (1,786 .. 78,832)
  #
  # Every vLLM cell sits at 74-79 GB because every one passes
  # --gpu-memory-utilization 0.9-0.92, so vLLM preallocates ~the whole card for
  # KV cache whether it needs it or not. s16_inferrep_qwen1_5b is a 1.5B model
  # holding 74.6 GB. The detector can therefore read "nearly full card => not
  # training" -- which is a fact about vLLM's allocator flag, not about the
  # workload, and it breaks the moment anyone serves at 0.5.
  #
  # The one inference cell whose footprint lands INSIDE the training range,
  # s16_infershard_qwen1_5b_tp4pp4 at 39 GB, is also the one the detector
  # misreads worst: 63.4% of its windows are called training, against 1.7-22.7%
  # for the 72B long-context cells at 74-79 GB. That is the false positive we
  # actually have, and it is a memory-overlap effect, not a context-length one.
  #
  # So: sweep the allocator flag at fixed model and context, which decouples
  # "is inference" from "fills the card" for free, and add training that DOES
  # fill the card so the overlap is populated from both sides.
  c=s16_infermem_qwen72b_u30;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 4000 --max-tokens 128 --batch-size 8  --gpu-memory-utilization 0.30
  c=s16_infermem_qwen72b_u45;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 4000 --max-tokens 128 --batch-size 16 --gpu-memory-utilization 0.45
  c=s16_infermem_qwen72b_u60;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 4000 --max-tokens 128 --batch-size 24 --gpu-memory-utilization 0.60
  c=s16_infermem_qwen7b_u20;   mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_7b_fp16  4 4 \
      --prompt-tokens 2000 --max-tokens 128 --batch-size 16 --gpu-memory-utilization 0.20
  c=s16_infermem_qwen7b_u40;   mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_7b_fp16  4 4 \
      --prompt-tokens 2000 --max-tokens 128 --batch-size 32 --gpu-memory-utilization 0.40

  # A100-40GB VARIANTS of the same memory sweep. The H100 numbers above are sized
  # for an 80 GB card and do not transfer: 72B fp16 is ~145 GB of weights, ~9 GB
  # per card across 16, so --gpu-memory-utilization 0.30 of a 40 GB card leaves
  # ~3 GB for KV and will not start. More importantly section 33's confound is
  # about the FRACTION of the card an engine holds, not the absolute gigabytes,
  # so running the sweep on a second card size is the direct test of whether
  # "nearly full card => not training" is about vLLM's allocator or about the
  # hardware. Training at 40 GB is also already known to OOM at H100 batch sizes
  # (--size 1b --batch-size 32 --seq-len 1024 peaked at 40.6 GB on A100-40GB),
  # hence the smaller batches here.
  c=s16_infermem_qwen72b_u45;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 4000 --max-tokens 128 --batch-size 8  --gpu-memory-utilization 0.45
  c=s16_infermem_qwen72b_u65;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 4000 --max-tokens 128 --batch-size 16 --gpu-memory-utilization 0.65
  c=s16_infermem_qwen72b_u85;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --prompt-tokens 4000 --max-tokens 128 --batch-size 24 --gpu-memory-utilization 0.85
  c=s16_trainmem_fsdp7b_nockpt;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 7b --no-activation-ckpt --batch-size 2 --seq-len 1024 --duration "$DUR"
  c=s16_trainmem_fsdp13b_b2;     mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 13b --no-activation-ckpt --batch-size 2 --seq-len 512 --duration "$DUR"

  # ...and training pushed up into the range vLLM occupies. fsdp70b tops out at
  # 50.7 GB today; without activation checkpointing and at a longer sequence it
  # lands where every inference cell currently lives, so "nearly full card"
  # stops being evidence either way.
  c=s16_trainmem_fsdp70b_nockpt; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 70b --no-activation-ckpt --seq-len 2048 --duration "$DUR"
  c=s16_trainmem_fsdp30b_b16;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 30b --no-activation-ckpt --batch-size 16 --duration "$DUR"

  # TRAINING WITHOUT A PER-STEP SIGNATURE, the same boundary from the other side.
  # --grad-accum 64 makes the optimizer step (and, via FSDP no_sync, the
  # inter-node reduce-scatter) happen on 1 microbatch in 64. If the detector is
  # keying on step periodicity and on the collective, this is training that
  # should look like inference -- and if it still gets caught, the claim that
  # multi-node training is fundamentally detectable gets much stronger.
  # The sync variant isolates WHICH of the two the detector was using.
  c=s16_train_accum64_fsdp7b;     mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 7b --grad-accum 64 --duration "$DUR"
  c=s16_train_accumsync_fsdp7b;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 7b --grad-accum 64 --accum-sync-every-microbatch --duration "$DUR"

  # (2) DRAM-BUSY TRAINING. Every training cell in the corpus sits at 3-25%
  # mem_utilization_pct because it is small-model or communication-bound, so
  # "DRAM-quiet implies training" is a free shortcut and nothing tests the
  # dangerous error direction (missing real training). --no-activation-ckpt
  # keeps activations resident and read back instead of recomputed, which is
  # the cheapest way to buy memory traffic without changing the model; large
  # batch and longer sequence add more.
  # Batch sizes are PER-GPU-MEMORY, not universal. Turning off activation
  # checkpointing is what buys the DRAM traffic, and it is also what makes the
  # footprint explode: --size 1b --batch-size 32 --seq-len 1024 peaked at
  # 40.6 GB and OOM'd at step 0 on an A100-40GB, having been sized for the
  # H100's 80 GB. Defaults below are H100; export the knobs for a 40 GB card.
  NOCKPT_1B_B=${NOCKPT_1B_B:-32}
  NOCKPT_1B_SEQ2K_B=${NOCKPT_1B_SEQ2K_B:-16}
  NOCKPT_7B_B=${NOCKPT_7B_B:-8}
  RESNET_B=${RESNET_B:-512}
  # batch 32 / seq 1024 with activation checkpointing OFF peaks at 40.6 GB -- measured,
  # "OOM at step 0! Peak memory: 40.6 GB" against the 39.49 GB a 40 GB card
  # actually exposes. Turning checkpointing off is the whole point of the
  # cell, so the memory is the experiment, not a bug to tune away.
  c=s16_train_fsdp1b_b32_nockpt;   need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 1b --batch-size "$NOCKPT_1B_B" --seq-len 1024 --no-activation-ckpt --duration "$DUR"
  # batch 16 / seq 2048 with activation checkpointing OFF peaks at 40.6 GB -- measured,
  # "OOM at step 0! Peak memory: 40.6 GB" against the 39.49 GB a 40 GB card
  # actually exposes. Turning checkpointing off is the whole point of the
  # cell, so the memory is the experiment, not a bug to tune away.
  c=s16_train_fsdp1b_seq2k_nockpt; need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 1b --batch-size "$NOCKPT_1B_SEQ2K_B" --seq-len 2048 --no-activation-ckpt --duration "$DUR"
  c=s16_train_fsdp7b_nockpt;       mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/fsdp_training.py --size 7b --batch-size "$NOCKPT_7B_B" --seq-len 1024 --no-activation-ckpt --duration "$DUR"
  c=s16_train_ddp_resnet50_b512;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/pytorch_training_ddp.py --model resnet50 --batch-size "$RESNET_B" --amp --duration "$DUR"

  # ---- CONCURRENCY ladder (section 24b) -----------------------------------
  # 24b refuted both mechanisms offered for the obligatory-inference false
  # alarm. Prefill was refuted by kv32k_out2k (decode-dominated, and the
  # HIGHEST power of any cell at 488 W); the level story was refuted by the
  # same cell going UP rather than down. The surviving candidate is that every
  # high-power inference cell serves 64 sequences at once, so these vary ONLY
  # --batch-size at fixed 32k context and 128 output tokens. kv32k (batch 64,
  # 421 W) is the top of this ladder and is already collected.
  #
  # Prediction if concurrency is the driver: power falls monotonically with
  # batch. If power is flat, concurrency is refuted too and the confusion is
  # something else again.
  #
  # NOTE: these are NOT multi-node-obligatory. At 32k tokens per request the KV
  # is 257k/514k/1.03M tokens for batch 8/16/32, all below the ~1.34M one node
  # holds, so they are mechanism probes and must not be counted as obligatory
  # coverage.
  c=s16_infershard_qwen72b_kv32k_b8;  mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --max-model-len 32768 --prompt-tokens 32000 --max-tokens 128 \
      --batch-size 8 --gpu-memory-utilization 0.92
  c=s16_infershard_qwen72b_kv32k_b16; mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --max-model-len 32768 --prompt-tokens 32000 --max-tokens 128 \
      --batch-size 16 --gpu-memory-utilization 0.92
  c=s16_infershard_qwen72b_kv32k_b32; mn_ray_shard "$c" "$(rid_for "$c" "$rep")" qwen25_72b_fp16 8 2 \
      --max-model-len 32768 --prompt-tokens 32000 --max-tokens 128 \
      --batch-size 32 --gpu-memory-utilization 0.92
  c=s16_train_fsdp7b_seq2k;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/fsdp_training.py --size 7b --batch-size 2 --seq-len 2048 --duration "$DUR"

  # --alloc-gb 44 is the cell's identity, not a tunable, so this is gated rather
  # than scaled: on the A100-40GB pair drawn on 2026-09-30 every rank died with
  # "Tried to allocate 40.98 GiB. GPU has a total capacity of 39.49 GiB" and the
  # round spent 20 minutes collecting 19 rows of an OOM. Skipping is the honest
  # outcome on a card that cannot hold it.
  c=s16_nonml_himem_obl;      need_gb 48 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/nonml_distributed.py --mode hpc --alloc-gb 44 --duty 1.0 --duration "$DUR"
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
  # QUIET SINGLE-NODE TRAINING (section 28's probe). The detector has learned
  # "single-node training is DRAM-busy" (56-61% mem_util) and "multi-node
  # training is quiet" (4-7%). A quiet cell with NO inter-node communication is
  # a combination it has never been shown: batch 1 gives low arithmetic
  # intensity, so the step is launch- and latency-bound rather than
  # bandwidth-bound, without any fabric involved. If these are missed, the
  # dangerous error (failing to flag real training) exists in a case that needs
  # no attacker and no multi-node hardware.
  c=s8_train_fsdp1b_b1;            s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/fsdp_training.py --size 1b --batch-size 1 --seq-len 256 --duration "$DUR"
  c=s8_train_fsdp7b_b1;            s8_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT"     -- workloads/fsdp_training.py --size 7b --batch-size 1 --seq-len 256 --duration "$DUR"
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
  c=s1_train_fsdp1b_b1;            s1_train "$c" "$(rid_for "$c" "$rep")"                   -- workloads/fsdp_training.py --size 1b --batch-size 1 --seq-len 256 --duration "$DUR"
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

  # ==========================================================================
  # WORKLOAD-DIVERSITY ROUND (2026-09-23)
  # ==========================================================================
  # Leave-one-family-out on the honest multi-node corpus: held-out train_fsdp
  # recall 0.005, train_ddp 0.042, and held-out infershard_qwen falsely flagged
  # on 71% of its windows. Cause is program diversity, not data volume -- the
  # training side is TWO programs and the not-training side is nominally ten
  # families but really three (nonml_distributed.py, vLLM, TPU-JAX).
  # heldout_calibration.md already showed 14 -> 32 workloads of the same script
  # moves held-out FPR by nothing.
  #
  # Every cell below is a different program or a different parallelism
  # mechanism, never a new flag on an old one. Names are verified through
  # threeway_label()/family()/honest_mask() by
  # classifier/rebuttal_analysis/test_cell_label_routing.py -- do not rename one
  # without re-running it, because the taxonomy is substring-based and checks
  # the training keywords BEFORE `infer` (s16_inferbert_embed would be training).
  #
  #   ONLY_CELLS='^s16_(infervis|inferembed|inferdiff|inferasr|infercnn|inferrec|inferhf|inferserve|nonml_(fftshard|stencil|nccl))|^s16_train_(lora|pipe|contrastive|diffusion)'

  # -- Non-LLM inference, 8 independent replicas per node (venv-main) ---------
  # Six different inference programs. The corpus's entire inference class is
  # decoder-only LLMs, so it has never seen encoder-only, encoder-decoder,
  # convolutional or iterative-denoising inference.
  c=s16_infervis_vitl;     mn_replicated_main "$c" "$(rid_for "$c" "$rep")" mn_infer_vit_l_d600
  c=s16_inferembed_bge;    mn_replicated_main "$c" "$(rid_for "$c" "$rep")" mn_infer_bge_embed_d600
  c=s16_inferembed_clip;   mn_replicated_main "$c" "$(rid_for "$c" "$rep")" mn_infer_clip_embed_d600
  c=s16_inferdiff_sdxl;    mn_replicated_main "$c" "$(rid_for "$c" "$rep")" mn_infer_sdxl_d600
  c=s16_inferasr_whisper;  mn_replicated_main "$c" "$(rid_for "$c" "$rep")" mn_infer_whisper_d600
  c=s16_infercnn_rn50;     mn_replicated_main "$c" "$(rid_for "$c" "$rep")" mn_infer_rn50_d600

  # -- A SECOND LLM engine. Every LLM stream in the corpus is vLLM, and
  # infershard_qwen (14 workloads, a quarter of all honest multi-node windows)
  # is the worst false-alarm source we have. HF generate has no paged KV, no
  # continuous batching, no preallocation -- section 33c found every vLLM cell
  # pinned at 74-79 GB by --gpu-memory-utilization, and 33d that the detector
  # reads memory LEVEL, so an inference cell that does not preallocate is the
  # direct test of that confound.
  c=s16_inferhf_qwen7b;    mn_replicated_main "$c" "$(rid_for "$c" "$rep")" mn_infer_hf_qwen7b_d600
  # Sharded by layer across all 8 GPUs of a node: pipeline-style, versus vLLM's
  # tensor-parallel all-reduce per layer. Same job to an operator, completely
  # different on the wire. --max-memory-gb is the footprint sweep.
  c=s16_inferhf_shard72b;      s8_single "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/llm_inference_hf_sharded.py --model qwen25_72b_fp16 --duration "$DUR" --batch-size 8
  c=s16_inferhf_shard72b_m40;  s8_single "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/llm_inference_hf_sharded.py --model qwen25_72b_fp16 --duration "$DUR" --batch-size 8 --max-memory-gb 40
  c=s16_inferhf_shard72b_m60;  need_gb 60 "$c" && s8_single "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/llm_inference_hf_sharded.py --model qwen25_72b_fp16 --duration "$DUR" --batch-size 8 --max-memory-gb 60

  # -- Recommender: huge tables, sparse gathers, all-to-all sized by the batch.
  # Huge memory at low utilisation is what evasive training would imitate, and
  # there is no honest example of it in the corpus to calibrate against.
  c=s16_inferrec_dlrm;     mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/inference_recsys.py --table-gb 2 --batch-size 4096 --duration "$DUR"
  c=s16_inferrec_dlrm_t8;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/inference_recsys.py --table-gb 8 --batch-size 4096 --duration "$DUR"

  # -- HPC with REAL cross-node dependency. nonml_distributed --mode hpc is
  # embarrassingly parallel (its only inter-node traffic is a 4-byte deadline
  # vote every 2.5 s); --mode comm is one fixed 256 MB all_reduce with no
  # compute. Nothing in the corpus both computes hard AND talks hard without
  # being training, which is the case that would fool the detector.
  c=s16_nonml_fftshard;        mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/hpc_distributed.py --mode fftshard --grid 512 --duration "$DUR"
  c=s16_nonml_fftshard_g1024;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/hpc_distributed.py --mode fftshard --grid 1024 --duration "$DUR"
  c=s16_nonml_stencil;         mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/hpc_distributed.py --mode stencil --grid 768 --duration "$DUR"
  c=s16_nonml_nccl_bw;         mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/hpc_distributed.py --mode nccl --max-mb 256 --duration "$DUR"

  # -- REAL HPC, not our idea of it. Every non-ML multi-node cell above is
  # something this project wrote: FFT shards, halo stencils, NCCL sweeps, duty
  # and low-power loops. They are one author's notion of what non-ML GPU work
  # looks like, and infershard_qwen is still the worst false-alarm family we
  # have. HPL and GROMACS are what HPC clusters actually run. HPL is dense FP64
  # LU -- a communication pattern nothing else here produces -- and GROMACS is
  # neighbour-list-bound molecular dynamics, which resembles nothing in the ML
  # corpus at all.
  #
  # From the NGC images, which pull ANONYMOUSLY (verified against the registry:
  # nvcr.io/nvidia/hpc-benchmarks:24.09 and nvcr.io/hpc/gromacs:2023.2 both
  # return HTTP 200 for a manifest under an anonymous bearer token). That is what
  # unblocks this: the earlier attempt died trying to build a CUDA GROMACS on the
  # node, since the Ubuntu package is CPU-only. s8_single because the workload
  # shards itself across the node's 8 GPUs and must not be replicated per GPU.
  # TIMEOUT_BIG because a cold image pull is minutes before any compute starts.
  c=s16_nonml_hpl;             s8_single "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/hpc_container.py --benchmark hpl --duration "$DUR"
  c=s16_nonml_gromacs;         s8_single "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/hpc_container.py --benchmark gromacs --duration "$DUR"

  # -- NEW TRAINING PROGRAMS. The training side is the worse failure: 0.5%
  # held-out recall. Two families become six.
  # LoRA: real training whose gradient all-reduce is 2-3 orders of magnitude
  # smaller than a full fine-tune. If a detector leaning on NIC volume misses
  # it, that is a finding about the detector.
  # 1b runs on any card; 7b needs 80 GB (DDP replicates all 6.7B fp32 params per
  # rank -- freezing them for LoRA does not make them smaller). Both collapse to
  # family `train_lora`, so the 40 GB track still contributes the fold.
  c=s16_train_lora1b;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_lora.py --size 1b --rank 16 --batch-size 4 --seq-len 1024 --duration "$DUR"
  # 38.25 GiB resident at batch 16 / seq 1024 on a 40 GB card -> OOM on every
  # rank. It completed on the 80 GB track, so this is a real memory requirement,
  # not a bug: gate it the way the other big cells are gated.
  c=s16_train_lora1b_b16;  need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_lora.py --size 1b --rank 16 --batch-size 16 --seq-len 1024 --duration "$DUR"
  c=s16_train_lora7b;      need_gb 60 "$c" && mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_lora.py --size 7b --rank 16 --batch-size 4 --seq-len 1024 --duration "$DUR"
  # Contrastive: a large all-gather in the FORWARD pass whose size scales with
  # world size rather than model size. No other cell has that shape.
  c=s16_train_contrastive; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_contrastive.py --batch-size 64 --embed-dim 1024 --duration "$DUR"
  # Diffusion: activation memory rises and falls WITHIN a step. The corpus has
  # no training program whose memory moves.
  c=s16_train_diffusion;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diffusion.py --resolution 256 --batch-size 16 --duration "$DUR"

  # ── Parallelism strategies the corpus has never seen ──────────────────────
  # The 2026-09-30 LOFO put held-out TRAIN recall at 0.5687 with train_ddp at
  # 0.0255, and the ablations showed neither more data nor more NIC coverage
  # moves it -- train_ddp already has sidecars on 80.6% of its windows. What is
  # missing is training whose COMMUNICATION MACHINERY differs, so these cells
  # are chosen for their collectives, not for their model size.

  # Local SGD / DiLoCo: the ranks are independent between syncs, so the NIC goes
  # completely silent for H steps and then moves the whole parameter set at once.
  # Long flat stretches punctuated by rare huge bursts is a shape no family in
  # the corpus produces -- and the one that most resembles independent inference
  # replicas, which is the disguise this project exists to detect. The sweep over
  # H is the point: H=1 is ordinary data parallelism and H=500 is nearly silent,
  # so the cells bracket where detection fails.
  c=s16_train_localsgd_h1;   mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_localsgd.py --size 1b --sync-every 1   --duration "$DUR"
  c=s16_train_localsgd_h50;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_localsgd.py --size 1b --sync-every 50  --duration "$DUR"
  c=s16_train_localsgd_h200; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_localsgd.py --size 1b --sync-every 200 --duration "$DUR"
  c=s16_train_localsgd_h500; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_localsgd.py --size 1b --sync-every 500 --duration "$DUR"

  # MoE expert parallelism: all-to-all, a collective no other cell uses, with
  # volume that scales with TOKENS rather than parameters. The skewed variant is
  # deliberately router-collapsed so some ranks are much busier than others --
  # every other training family is rank-symmetric, so the cross-rank features
  # have never seen a training job whose ranks legitimately disagree.
  c=s16_train_moe_bal;       mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_moe.py --experts-per-rank 2 --top-k 2 --duration "$DUR"
  c=s16_train_moe_skew;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_moe.py --experts-per-rank 2 --top-k 2 --router-skew 4.0 --duration "$DUR"
  c=s16_train_moe_top1;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_moe.py --experts-per-rank 4 --top-k 1 --duration "$DUR"

  # Megatron tensor parallelism: ~98 collectives per step instead of one, each
  # sized by the ACTIVATION rather than the model. High message RATE at small
  # message size is the axis the corpus has never sampled, and the regime where
  # per-window aggregates most plausibly wash the signal out.
  #
  # The two cells are the same program either side of the node boundary, which
  # is the whole point: tp16 puts every activation collective on the NIC, while
  # tp8dp2 is the arrangement a real deployment picks (TP over NVLink, data
  # parallel over the NIC) and is nearly silent between gradient all-reduces.
  # A detector that keys on inter-node volume should find these two very
  # different despite their being the identical workload.
  c=s16_train_tp16;          mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_tp.py --tp-size 16 --duration "$DUR"
  c=s16_train_tp8;           mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_tp.py --tp-size 8  --duration "$DUR"

  # ── Distributed RL: train and infer on ONE cluster, per rank ──────────────
  # No training family in the corpus has ranks doing categorically different
  # work. An IMPALA actor runs pure inference while the learner trains, at the
  # same instant, so xr_level_spread and the cross-rank correlations see
  # something they never have. --learner-rank 15 is the adversarially
  # interesting one: sliding_windows_multinode takes its single-stream features
  # from the LOWEST rank, so the canonical stream becomes an actor doing pure
  # inference while the job is training.
  c=s16_train_impala_a15;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_rl_dist.py --arch impala --learner-rank 0  --actor-jitter 2 --duration "$DUR"
  c=s16_train_impala_r15;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_rl_dist.py --arch impala --learner-rank 15 --actor-jitter 2 --duration "$DUR"
  c=s16_train_apexrl_replay; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_rl_dist.py --arch apex --actor-jitter 3 --duration "$DUR"
  c=s16_train_ppodist_r4;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_rl_dist.py --arch ppo --rollout-steps 4 --duration "$DUR"

  # ── Streaming / Decoupled DiLoCo ──────────────────────────────────────────
  # A replica is a NODE, not a rank: at 2 replicas the compressed all-gather
  # costs exactly what an all-reduce would, so E3M0's 8x is realised. At 16
  # replicas it nets 1.00x and int8 nets 0.50x -- the _r16 cell exists to
  # MEASURE that prediction rather than assert it.
  c=s16_train_streamdiloco_f8;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco_stream.py --mode streaming --fragments 8 --sync-every 100 --tau 1  --compress e3m0 --duration "$DUR"
  c=s16_train_streamdiloco_t20; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco_stream.py --mode streaming --fragments 8 --sync-every 100 --tau 20 --compress e3m0 --duration "$DUR"
  c=s16_train_streamdiloco_r16; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco_stream.py --mode streaming --fragments 8 --sync-every 100 --tau 1  --compress e3m0 --replica-ranks 1 --duration "$DUR"
  c=s16_train_decoupdiloco_k1;  mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco_stream.py --mode decoupled --fragments 24 --sync-every 24 --tau 2 --quorum 1 --grace 1 --speed-jitter 0.1 --compress e3m0 --duration "$DUR"

  # ── The same cells on a 100 Mbps / 100 ms link ────────────────────────────
  # The regime the paper actually models. Shaped and unshaped share a family
  # (family() discards everything past the second token), so bandwidth is a
  # within-family variable and holding out the family holds out both regimes.
  c=s16_train_localsgd_h50_bw100;    mn_torchrun_bw "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_localsgd.py --size 1b --sync-every 50 --duration "$DUR"
  c=s16_train_streamdiloco_f8_bw100; mn_torchrun_bw "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco_stream.py --mode streaming --fragments 8 --sync-every 100 --tau 1 --compress e3m0 --duration "$DUR"
  c=s16_train_decoupdiloco_k1_bw100; mn_torchrun_bw "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_diloco_stream.py --mode decoupled --fragments 24 --sync-every 24 --tau 2 --quorum 1 --grace 1 --speed-jitter 0.1 --compress e3m0 --duration "$DUR"
  c=s16_train_impala_a15_bw100;      mn_torchrun_bw "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_rl_dist.py --arch impala --learner-rank 0 --actor-jitter 2 --duration "$DUR"

  # ── SWARM: stochastic routing, straggler bypass, a real mid-run migration ──
  # strided puts every stage across both nodes so roughly half of each hop
  # crosses the NIC; blocked keeps stages inside a node so only one edge does.
  # Same program, either side of the boundary.
  c=s16_train_swarm_s4;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/train_swarm.py --stages 4 --layout strided --straggler-ranks 5,11 --straggler-factor 3.0 --rebalance-every 300 --act-compress int8 --duration "$DUR"
  c=s16_train_swarm_blocked; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/train_swarm.py --stages 4 --layout blocked --ban-rank 6 --ban-at 120 --act-compress int8 --duration "$DUR"

  # ── Compression on an existing family, in DELTA form ──────────────────────
  # At one replica per rank the compressed all-gather costs W/2 times an
  # all-reduce, so e3m0 is predicted to net 1.00x and topk 1% about 6.25x.
  # These two cells measure that rather than taking it on trust.
  c=s16_train_localsgd_q4;    mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_localsgd.py --size 1b --sync-every 50 --compress e3m0 --duration "$DUR"
  c=s16_train_localsgd_topk1; mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT" -- workloads/train_localsgd.py --size 1b --sync-every 50 --compress topk:0.01 --duration "$DUR"
  # Pipeline-parallel, HONEST. Same GPipe mechanism the MN-P cells use, but
  # named for what it is and run at a realistic microbatch count rather than the
  # mb1 extreme the evasion ladder sweeps to. honest_mask() excludes every
  # advtrain_* cell, so the honest corpus currently has NO pipeline-parallel
  # training at all -- the layout most real frontier training actually uses.
  c=s16_train_pipe7b;      mn_torchrun "$c" "$(rid_for "$c" "$rep")" "$TIMEOUT_BIG" -- workloads/adversarial_multinode_pipeline.py --size 7b --microbatches 8 --rank0-init 0 --batch-size 4 --seq-len 512 --duration "$DUR"
done

"$RAY" stop --force >/dev/null 2>&1 || true
say "queue complete (node$NODE_RANK)."
