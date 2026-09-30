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
# EXPORT them. Sourcing alone leaves these as shell variables of THIS script, so
# every child -- multinode_queue.sh above all -- runs without them. That is what
# killed the 2026-09-21 A100-80GB queue on its first barrier (BUCKET: unbound
# variable) and what the STOP_WHEN_DONE rendezvous below already had to work
# around by re-sourcing node_env in its own subshell.
export BUCKET SESSION PROJECT ZONE NODE0 NODE1 MASTER_ADDR MASTER_PORT 2>/dev/null || true
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

# --- periodic sync ----------------------------------------------------------
# Without this, telemetry reaches GCS only when the queue exits or the shutdown
# hook fires. run_when_free.sh starts the loop; the MIG/bootstrap path did not,
# so the 2026-09-29 round ran 2 h with 77 parquets on local disk and nothing in
# the bucket. Those nodes were created with --instance-termination-action=DELETE
# -- the boot disk goes with the VM, so the ~30 s ExecStop flush was the only
# thing standing between that round and total loss. Starting the loop HERE
# covers every launch path, because every path comes through this script.
#
# The guard is a PIDFILE, not pgrep: `pgrep -f "sync_data.sh --loop"` matches
# the command line of the shell issuing it, so it always reports the loop as
# already running and the loop never starts.
SYNC_PID_FILE=/tmp/spar_sync_loop.pid
if [ ! -s "$SYNC_PID_FILE" ] || ! kill -0 "$(cat "$SYNC_PID_FILE" 2>/dev/null)" 2>/dev/null; then
  setsid bash "${GCPDIR:-/opt/spar/gcp}/sync_data.sh" --loop "${SYNC_SECS:-120}" \
    </dev/null >/tmp/sync_loop.log 2>&1 &
  echo $! > "$SYNC_PID_FILE"
  echo "[launch_queue] periodic sync loop started (pid $(cat "$SYNC_PID_FILE"), every ${SYNC_SECS:-120}s)"
else
  echo "[launch_queue] periodic sync loop already running (pid $(cat "$SYNC_PID_FILE"))"
fi

MASTER_ADDR="${MASTER_ADDR:-spar-gpu-0}" \
MASTER_PORT="${MASTER_PORT:-29500}" \
NODE_RANK="$NODE_RANK" \
NNODES="${NNODES:-2}" \
RUN_STAMP="$RUN_STAMP" \
PY="$REPO/venv-main/bin/python" \
PY_VLLM="$REPO/venv-vllm/bin/python" \
SMOKE="${SMOKE:-0}" \
STOP_WHEN_DONE="${STOP_WHEN_DONE:-0}" \
PEER_WAIT="${PEER_WAIT:-2400}" \
PEER="${PEER:-}" \
NODE_NAME="${NODE_NAME:-$(hostname -s)}" \
ONLY_CELLS="${ONLY_CELLS:-}" \
  nohup bash -c '
    bash scripts/rebuttal_collection/multinode_queue.sh
    rc=$?
    echo "[launch_queue] queue exited rc=$rc"
    bash /opt/spar/gcp/sync_data.sh --once || true
    if [ "${STOP_WHEN_DONE:-0}" = "1" ]; then
      echo "[launch_queue] STOP_WHEN_DONE=1 -- syncing"
      bash /opt/spar/gcp/sync_data.sh --once || true
      # WAIT FOR THE PEER before powering off. Whichever node drains first used
      # to shut down immediately, which strands the other: on 2026-09-08
      # spar-gpu-0 finished and powered off at 23:32Z, and spar-gpu-1 then spent
      # 1.9 h failing every remaining 2-node cell against a host that no longer
      # resolved ("Temporary failure in name resolution") while still billing.
      # A queue-done marker each side can see turns that into a rendezvous.
      # node_env is sourced by THIS script but its variables are not exported,
      # so the bash -c child above cannot see them. Source it here. The previous
      # version asserted them with ${VAR:?}, which killed the wrapper outright:
      #   bash: line 15: BUCKET: parameter null or not set
      # STOP_WHEN_DONE then never ran and the pair idled ~35 min on 2026-09-09.
      # A missing variable must cost us the peer rendezvous, never the shutdown.
      . /etc/spar/node_env 2>/dev/null || true
      CTRL=""
      if [ -n "${BUCKET:-}" ] && [ -n "${SESSION:-}" ] && [ -n "${NODE_NAME:-}" ]; then
        CTRL="${BUCKET}/runs/${SESSION}/control"
      else
        echo "[launch_queue] BUCKET/SESSION/NODE_NAME missing -- skipping the peer rendezvous, still powering off"
      fi
      if [ -n "$CTRL" ]; then
        echo "$(date -u +%FT%TZ) rc=$rc" > "/tmp/${NODE_NAME}.queue_done"
        gcloud storage cp "/tmp/${NODE_NAME}.queue_done" \
          "${CTRL}/${NODE_NAME}.queue_done" >/dev/null 2>&1 || true
      fi
      # Same peer-derivation trap as cell_barrier had: if NODE_NAME matches
      # NEITHER NODE0 nor NODE1 (a pair not named spar-gpu-0/1), the && || chain
      # silently yields PEER=$NODE0 -- a host that does not exist -- and both
      # nodes then idle until PEER_WAIT expires instead of powering off. Prefer
      # the PEER the caller passed; only derive when this node is actually one of
      # the two, and leave it empty rather than guessing.
      if [ -n "${PEER:-}" ] && [ "${PEER}" != "${NODE_NAME:-}" ]; then
        : # caller-supplied, trust it
      elif [ -n "${NODE0:-}" ] && [ -n "${NODE1:-}" ] \
           && { [ "$NODE_NAME" = "$NODE0" ] || [ "$NODE_NAME" = "$NODE1" ]; }; then
        [ "$NODE_NAME" = "$NODE0" ] && PEER="$NODE1" || PEER="$NODE0"
      else
        PEER=""
        echo "[launch_queue] WARNING: cannot resolve peer (NODE_NAME=${NODE_NAME:-} "\
"NODE0=${NODE0:-} NODE1=${NODE1:-}) -- skipping the rendezvous, still powering off"
      fi
      if [ -n "$PEER" ] && [ -n "$CTRL" ]; then
        echo "[launch_queue] waiting up to ${PEER_WAIT:-2400}s for $PEER to finish too"
        _t0=$SECONDS
        while [ $((SECONDS - _t0)) -lt "${PEER_WAIT:-2400}" ]; do
          gcloud storage ls "${CTRL}/${PEER}.queue_done" >/dev/null 2>&1 && \
            { echo "[launch_queue] $PEER is done too"; break; }
          sleep 30
        done
      else
        echo "[launch_queue] NODE0/NODE1 not in node_env -- cannot identify peer, not waiting"
      fi
      echo "[launch_queue] powering off in 2 min"
      shutdown -h +2 "spar: collection finished, stopping to stop the bill"
    fi' >"$LOG" 2>&1 </dev/null &
pid=$!
disown
echo "launched node${NODE_RANK} pid=$pid stamp=$RUN_STAMP smoke=${SMOKE:-0} only='${ONLY_CELLS:-<all>}' stop_when_done=${STOP_WHEN_DONE:-0} log=$REPO/$LOG"
