#!/bin/bash
# Multi-zone TPU teardown watchdog.
#
# Fires on whichever comes first:
#   1. the DONE marker in GCS (the driver finished, success or failure)
#   2. a global deadline (cleans up requests that never provision)
#   3. ACTIVE_MAX_SECS since a node was FIRST seen ACTIVE -- the load-bearing one,
#      because a pending request is free but a provisioned slice is not.
#
# `--valid-until-duration` bounds the queued REQUEST, not the life of a node that
# actually provisions, so this is the only real teardown guarantee.
#
# ── SESSION LOCKING (added after an orphan-billing incident) ─────────────────
# This watchdog used to unconditionally `gcloud compute instances stop` its host
# VM when done. That treats spar-analysis-0 as private to one job, and it is
# shared. Measured consequence on 2026-08-31: an older watchdog hit its active cap,
# correctly tore down ITS slice, then stopped the VM -- killing a newer driver
# mid-flight AND the newer watchdog with it. Two multislice groups (4 chips)
# subsequently provisioned with nothing supervising them and billed unsupervised
# until a human noticed.
#
# So: every watchdog and driver takes a lock under $LOCKDIR for its SESSION, and
# STOP_SELF only stops the VM when no OTHER lock is held. A watchdog still always
# tears down its own TPUs -- only the VM stop is deferred.
set -uo pipefail
source /tmp/mz_common.sh
MARKER=${MARKER:?set MARKER to the GCS DONE path}
MAX_SECS=${MAX_SECS:-32400}
ACTIVE_MAX_SECS=${ACTIVE_MAX_SECS:-3600}
SELF=${SELF:-spar-analysis-0}
STOP_SELF=${STOP_SELF:-1}
SESSION=${SESSION:-wd$$}
LOCKDIR=${LOCKDIR:-/tmp/spar_locks}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }

mkdir -p "$LOCKDIR"
MYLOCK="$LOCKDIR/wd.$SESSION.$$"
echo "$$ $(date -u)" > "$MYLOCK"
release(){ rm -f "$MYLOCK" 2>/dev/null || true; }
trap release EXIT

others(){ ls -1 "$LOCKDIR" 2>/dev/null | grep -v "^$(basename "$MYLOCK")$" | wc -l | tr -d ' '; }

n_other=$(others)
if [ "$n_other" -gt 0 ]; then
  say "NOTE: $n_other other spar session lock(s) present -- will NOT stop $SELF on exit"
  say "      locks: $(ls -1 "$LOCKDIR" | grep -v "^$(basename "$MYLOCK")$" | tr '\n' ' ')"
fi

DEADLINE=$(( $(date +%s) + MAX_SECS )); first_active=0
say "watchdog armed: ${#RESOURCES[@]} requests, deadline $(date -u -d @$DEADLINE +%H:%MZ), active cap ${ACTIVE_MAX_SECS}s, session=$SESSION"
reason=""
while :; do
  if gcloud storage ls "$MARKER" --project=$PROJECT >/dev/null 2>&1; then reason=marker; break; fi
  now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { reason=deadline; break; }
  any=0; states=""
  for r in "${RESOURCES[@]}"; do IFS=: read -r z q n a <<<"$r"
    st=$(qr_state "$z" "$q"); [ -n "$st" ] && any=1
    states+="$(echo "$z"|cut -c1-12)=$st "
    if [ "$st" = "ACTIVE" ] && [ "$first_active" = "0" ]; then
      first_active=$now; say "FIRST ACTIVE: $q ($z, $a)"
    fi
  done
  [ "$any" = "0" ] && { reason=all-gone; break; }
  if [ "$first_active" != "0" ] && [ $(( now - first_active )) -ge "$ACTIVE_MAX_SECS" ]; then
    reason=active-cap; break
  fi
  say "  $states $(( (DEADLINE-now)/60 ))m left$([ "$first_active" != "0" ] && echo " | billed $(( (now-first_active)/60 ))m")"
  sleep 60
done

say "TEARDOWN: $reason"
kill_all
sleep 20
say "== verify-zero =="
for r in "${RESOURCES[@]}"; do IFS=: read -r z q n a <<<"$r"
  echo "  $z qr=$(qr_state "$z" "$q" || echo none)"
done
gcloud storage cp /tmp/mz_watchdog.log \
  "gs://focus-surfer-494820-g0-telemetry/tpu/mz_watchdog_$(date -u +%Y%m%d_%H%M%S).log" \
  --project=$PROJECT 2>/dev/null || true

release
n_other=$(others)
if [ "$STOP_SELF" != "1" ]; then
  say "watchdog done ($reason); STOP_SELF=0, leaving $SELF running"
elif [ "$n_other" -gt 0 ]; then
  say "watchdog done ($reason); $n_other other session lock(s) still held -- NOT stopping $SELF"
  say "      held by: $(ls -1 "$LOCKDIR" | tr '\n' ' ')"
else
  say "watchdog done ($reason); no other sessions -- stopping $SELF"
  gcloud compute instances stop "$SELF" --project=$PROJECT --zone=us-central1-a --quiet >/dev/null 2>&1 || true
fi
