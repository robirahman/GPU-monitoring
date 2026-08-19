#!/usr/bin/env bash
# preempt_watch.sh -- poll the GCE metadata "preempted" flag; on TRUE, gracefully
# stop the workload and do a final telemetry flush before the STOP lands.
# Runs as the spar-preempt-watch.service (installed by bootstrap_node.sh).
#
# A Spot preemption of a STOP-action instance gives ~30 s of ACPI-shutdown notice.
# Detecting it here (within POLL seconds) buys a head start over the
# spar-final-sync.service ExecStop backstop; both are idempotent, so running both
# is fine.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

POLL="${SPAR_PREEMPT_POLL:-5}"
NODE="${NODE_NAME:-$(hostname -s)}"
PREEMPT_URL="http://metadata.google.internal/computeMetadata/v1/instance/preempted"

spar_say "preempt_watch started on ${NODE} (poll ${POLL}s)"

while true; do
  STATE="$(curl -sf -H "Metadata-Flavor: Google" "$PREEMPT_URL" 2>/dev/null || echo "")"
  if [[ "$STATE" == "TRUE" ]]; then
    spar_say "PREEMPTION signalled on ${NODE} -- graceful stop + final sync"
    # 1) mark so the collection queue stops launching new cells and the surviving
    #    node can record the partner's loss / downshift (S16 -> S8 -> S1).
    mkdir -p "${REPO_DIR:-.}/logs" 2>/dev/null || true
    touch "${REPO_DIR:-.}/logs/.preempted" 2>/dev/null || true

    # 2) SIGTERM the workload drivers so collectors get a clean stop and flush their
    #    final parquet (they also checkpoint *.partial periodically). We deliberately
    #    do NOT kill collect_telemetry itself first -- let the workload exit, then the
    #    collectors finalise on their own.
    pkill -TERM -f 'torchrun'                 2>/dev/null || true
    pkill -TERM -f 'run_workload_multi_gpu.py' 2>/dev/null || true
    pkill -TERM -f 'ray'                      2>/dev/null || true
    sleep 3

    # 3) final flush (data-first, time-boxed). This is the same path the shutdown
    #    ExecStop uses; whichever fires first wins, the other is a cheap no-op delta.
    bash "${SCRIPT_DIR}/sync_data.sh" --final || true

    spar_say "preempt_watch: final sync issued, exiting"
    exit 0
  fi
  sleep "$POLL"
done
