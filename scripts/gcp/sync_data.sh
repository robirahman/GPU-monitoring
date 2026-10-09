#!/usr/bin/env bash
# sync_data.sh -- push this node's telemetry to its per-node GCS prefix.
# Runs ON a provisioned node (identity from /etc/spar/node_env via env.sh).
#
#   sync_data.sh --once        one delta pass (call between queue cells)
#   sync_data.sh --loop [N]    every N seconds (default 120) until killed
#   sync_data.sh --final       preemption/shutdown path: deltas only, data-first,
#                              best-effort inside the ~30 s STOP window
#
# Layout (contract 2.7):  gs://.../runs/<SESSION>/<NODE_NAME>/{data,logs,meta}
# and .done markers mirrored to  gs://.../runs/<SESSION>/control/<NODE_NAME>/
#
# We NEVER pass --delete-unmatched-destination-objects: GCS is append-only truth,
# nothing collected is ever removed. rsync is inherently delta, so re-runs are cheap
# and the in-progress *.partial checkpoints ride along for crash-safety.
set -uo pipefail   # NOT -e: a slow/failed sub-sync must not abort a --final flush
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

MODE="${1:---once}"
LOOP_SECS="${2:-120}"

cd "${REPO_DIR:-.}" 2>/dev/null || { echo "sync_data: REPO_DIR '${REPO_DIR:-}' not found"; exit 1; }
NODE="${NODE_NAME:-$(hostname -s)}"
DEST="${BUCKET}/runs/${SESSION}/${NODE}"
CTRL="${BUCKET}/runs/${SESSION}/control/${NODE}"
mkdir -p data logs logs/.done

# Serialise every rsync on this node. `gcloud storage rsync` keeps a per-invocation
# listing cache under ~/.config/gcloud/surface_data/storage/rsync_files/, and two
# concurrent invocations clobber each other's state:
#
#   ERROR: (gcloud.storage.rsync) Unable to read file
#   [/root/.config/gcloud/.../rsync_files/FULL_900575....file:__data]: [Errno 2]
#
# On 2026-08-21 that raced the queue's between-cell sync against the shutdown
# chain's sync on spar-gpu-1: the LOGS pass survived, the DATA pass died, and the
# node powered off with a full run of telemetry still on local disk. Nothing
# reported an error because every call ends in `|| true`. The data was recoverable
# only because the boot disk is retained. flock makes the passes queue instead.
SYNC_LOCK=${SYNC_LOCK:-/var/lock/spar_sync.lock}
# _locked_rsync <timeout_s|0> <lock_wait_s> [gcloud storage rsync args...]
# timeout is applied INSIDE the function: `timeout` cannot invoke a shell
# function, so wrapping the call site would silently fail with
# "failed to run command '_locked_rsync'".
_locked_rsync() {
  local t="$1" w="$2"; shift 2
  if command -v flock >/dev/null 2>&1; then
    flock -w "$w" 9 || echo "[sync] WARNING: no $SYNC_LOCK within ${w}s; running unlocked" >&2
  fi
  if [ "$t" != 0 ]; then
    timeout "$t" gcloud storage rsync "$@"
  else
    gcloud storage rsync "$@"
  fi
} 9>>"$SYNC_LOCK"

# one rsync pass of a local dir -> its GCS subprefix (skip if the dir is empty/missing)
push() {  # $1 local dir   $2 gcs subpath   [extra rsync args...]
  local src="$1" sub="$2"; shift 2
  [[ -d "$src" ]] || return 0
  # -r recursive; -x excludes are regex on the full path
  _locked_rsync 0 "${SYNC_LOCK_WAIT:-900}" -r "$@" "$src" "${DEST}/${sub}" 2>&1 | sed "s/^/[sync ${sub}] /" || true
}

sync_once() {
  spar_say "sync --once  ${NODE} -> ${DEST}"
  # data/ holds the parquets (incl. *.partial) -- the payload. The ONE exclusion is
  # data/cifar10: that is a 341 MB staged INPUT dataset, not telemetry. Without
  # this it is uploaded from every node and pulled back again at ingest, which
  # dwarfs the actual telemetry (6 MB of ladder parquets vs 341 MB of CIFAR).
  push data data -x '.*/cifar10/.*'
  # .done/.failed markers -> shared control/ so the other node / driver can account completion.
  # NB: use $CTRL directly, NOT a "../control" relative path -- gcloud does NOT normalise
  # ".." in gs:// URLs (it becomes a literal object-name component).
  [[ -d logs/.done ]] && _locked_rsync 0 "${SYNC_LOCK_WAIT:-900}" -r logs/.done "$CTRL" 2>&1 | sed 's/^/[sync ctrl] /' || true
  # STEP logs (needed for the epoch join). Exclude the smoke sandbox + pycache.
  push logs logs -x '.*/smoke_data/.*' -x '.*/__pycache__/.*'
  # meta: bootstrap report etc.
  [[ -f "logs/bootstrap_report_${NODE}.json" ]] && \
    gcloud storage cp "logs/bootstrap_report_${NODE}.json" "${DEST}/meta/" >/dev/null 2>&1 || true
}

case "$MODE" in
  --once)
    sync_once
    ;;
  --loop)
    spar_say "sync --loop every ${LOOP_SECS}s (ctrl-c / kill to stop)"
    while true; do sync_once; sleep "$LOOP_SECS"; done
    ;;
  --final)
    # Preemption/shutdown: prioritise the irreplaceable parquets, then markers, then
    # logs with whatever of the ~30 s window remains. Each step is time-boxed so one
    # slow transfer cannot eat the whole window.
    spar_say "sync --final (preemption/shutdown) ${NODE} -> ${DEST}"
    _locked_rsync 18 8 -r data "${DEST}/data" 2>&1 | sed 's/^/[final data] /' || true
    _locked_rsync 4 2 -r logs/.done "${CTRL}" 2>&1 | sed 's/^/[final ctrl] /' || true
    _locked_rsync 6 2 -r -x '.*/smoke_data/.*' -x '.*/__pycache__/.*' logs "${DEST}/logs" 2>&1 | sed 's/^/[final logs] /' || true
    # breadcrumb so ingest knows this node flushed on the way down
    printf 'final-sync %s\n' "$(spar_ts)" | gcloud storage cp - "${DEST}/meta/final_sync.txt" >/dev/null 2>&1 || true
    spar_say "sync --final done"
    ;;
  *)
    echo "usage: $0 [--once | --loop [SECS] | --final]"; exit 2 ;;
esac
