#!/usr/bin/env bash
# Launch one queue stage, wait for it, flush telemetry, then power the node off.
#
#   sudo STAMP=20260822c PEER=spar-gpu-1 ONLY='^s(16|8|1)_(stalled_train|nonml)_' \
#        bash /opt/spar/repo/scripts/gcp/run_stage_and_stop.sh
#
# Self-detaches on first call (same nohup idiom as launch_queue.sh) so the
# invoking ssh returns immediately instead of blocking for the whole stage.
#
# Powering off is peer-gated: a node that shuts down while its PEER is mid-cell
# kills every remaining multi-node cell, so each node publishes a done-marker to
# GCS and waits for the other's before calling shutdown. Every wait is bounded --
# a hang must not leave 16 H100s idling, which is the whole point of this script.
set -uo pipefail
. /etc/spar/node_env
REPO="${REPO_DIR:-/opt/spar/repo}"
: "${STAMP:?set STAMP}"
: "${PEER:?set PEER}"
ONLY="${ONLY:-}"
REPS="${REPS:-1}"
NODE="$(hostname -s)"
CTRL="${BUCKET}/runs/${SESSION}/control"
RELLOG="logs/queue_${STAMP}.log"
LOGP="$REPO/$RELLOG"
MARK="stage_${STAMP}_complete.txt"

if [ "${_DETACHED:-0}" != 1 ]; then
  export _DETACHED=1
  nohup bash "$0" "$@" >"$REPO/logs/runstop_${STAMP}.log" 2>&1 </dev/null &
  disown
  echo "armed node=${NODE} pid=$! stamp=${STAMP} only='${ONLY}'"
  exit 0
fi

echo "$(date -u +%FT%TZ) launching stage stamp=$STAMP only='$ONLY'"
# PEER is what the per-cell barrier needs; node_env's NODE0/NODE1 cannot be
# trusted for it on a pair that is not named spar-gpu-0/1.
RUN_STAMP="$STAMP" REPS="$REPS" ONLY_CELLS="$ONLY" LOG="$RELLOG" PEER="$PEER" \
  bash "$REPO/scripts/gcp/launch_queue.sh"

sleep 90
for _ in $(seq 1 480); do                       # <= 8 h
  grep -q 'queue complete' "$LOGP" 2>/dev/null && break
  sleep 60
done
echo "$(date -u +%FT%TZ) stage finished (or timed out)"

bash "$REPO/scripts/gcp/sync_data.sh" --once || true
printf 'done %s\n' "$(date -u +%FT%TZ)" | gcloud storage cp - "$CTRL/$NODE/$MARK" || true

echo "$(date -u +%FT%TZ) waiting for peer $PEER"
for _ in $(seq 1 60); do                        # <= 30 min
  gcloud storage ls "$CTRL/$PEER/$MARK" >/dev/null 2>&1 && break
  sleep 30
done

# Final flush BEFORE powering off. On 2026-08-21 spar-gpu-1 shut down with a full
# run of telemetry still on local disk because a racing rsync had died; passes are
# flock-serialised now, and this runs twice for good measure.
bash "$REPO/scripts/gcp/sync_data.sh" --once || true
bash "$REPO/scripts/gcp/sync_data.sh" --final || true
echo "$(date -u +%FT%TZ) powering off"
shutdown -h +2
