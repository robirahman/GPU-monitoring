#!/usr/bin/env bash
# Launch the multi-node collection queue DETACHED ON THE NODE, so it survives the
# driving Cloud Shell being closed or idle-killed. Node identity comes from
# /etc/spar/node_env (authoritative, written by bootstrap) rather than from the
# caller, so the same command line is correct on both nodes.
#
#   sudo RUN_STAMP=20260820b bash scripts/gcp/launch_queue.sh                    # real
#   sudo RUN_STAMP=... STOP_WHEN_DONE=1 bash .../launch_queue.sh   # power off when done
#
# STOP_WHEN_DONE exists because an unattended run is billed until someone
# notices it finished. The 2x A100 pair collected its 33 cells in 5.9 h on
# 2026-09-04 and then sat IDLE for 19.4 h until Spot preempted it -- about $658
# of the run's ~$910. shutdown triggers spar-final-sync.service, and because the
# instances were created with --instance-termination-action=STOP the boot disk
# (repo, venvs, phase markers) survives for a restart.
#   sudo RUN_STAMP=20260820b SMOKE=1 ONLY_CELLS='^s16_' bash .../launch_queue.sh # smoke
set -uo pipefail
. /etc/spar/node_env
REPO="${REPO_DIR:-/opt/spar/repo}"
cd "$REPO" || { echo "cannot cd $REPO" >&2; exit 1; }
: "${RUN_STAMP:?export RUN_STAMP (identical on both nodes)}"
mkdir -p logs
LOG="${LOG:-logs/queue_${RUN_STAMP}$([ "${SMOKE:-0}" = 1 ] && echo _smoke).log}"

# vllm probes FlashInfer for top-k/top-p sampling via an UNGUARDED import chain
# (topk_topp_sampler.flashinfer_sampler_supported -> v1/attention/backends/
# flashinfer -> `from flashinfer import ...`). flashinfer is deliberately NOT
# installed (see bootstrap_node.sh: it needs nvcc and hangs the engine), so the
# probe has to be switched off BEFORE it imports. The env check returns early,
# so this is sufficient; without it every vllm cell dies with
# ModuleNotFoundError after loading weights.
export VLLM_USE_FLASHINFER_SAMPLER=0

MASTER_ADDR="${MASTER_ADDR:-spar-gpu-0}" \
MASTER_PORT="${MASTER_PORT:-29500}" \
NODE_RANK="$NODE_RANK" \
NNODES="${NNODES:-2}" \
RUN_STAMP="$RUN_STAMP" \
PY="$REPO/venv-main/bin/python" \
PY_VLLM="$REPO/venv-vllm/bin/python" \
SMOKE="${SMOKE:-0}" \
STOP_WHEN_DONE="${STOP_WHEN_DONE:-0}" \
ONLY_CELLS="${ONLY_CELLS:-}" \
  nohup bash -c '
    bash scripts/rebuttal_collection/multinode_queue.sh
    rc=$?
    echo "[launch_queue] queue exited rc=$rc"
    bash /opt/spar/gcp/sync_data.sh --once || true
    if [ "${STOP_WHEN_DONE:-0}" = "1" ]; then
      echo "[launch_queue] STOP_WHEN_DONE=1 -- syncing and powering off in 2 min"
      bash /opt/spar/gcp/sync_data.sh --once || true
      shutdown -h +2 "spar: collection finished, stopping to stop the bill"
    fi' >"$LOG" 2>&1 </dev/null &
pid=$!
disown
echo "launched node${NODE_RANK} pid=$pid stamp=$RUN_STAMP smoke=${SMOKE:-0} only='${ONLY_CELLS:-<all>}' stop_when_done=${STOP_WHEN_DONE:-0} log=$REPO/$LOG"
