#!/bin/bash
set -uo pipefail
source /tmp/mz_common.sh
MARKER=${MARKER:?}
MAX_SECS=${MAX_SECS:-32400}
ACTIVE_MAX_SECS=${ACTIVE_MAX_SECS:-3600}
SELF=${SELF:-spar-analysis-0}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
DEADLINE=$(( $(date +%s) + MAX_SECS )); first_active=0
say "watchdog armed: ${#RESOURCES[@]} requests, deadline $(date -u -d @$DEADLINE +%H:%MZ), active cap ${ACTIVE_MAX_SECS}s"
reason=""
while :; do
  if gcloud storage ls "$MARKER" --project=$PROJECT >/dev/null 2>&1; then reason=marker; break; fi
  now=$(date +%s); [ "$now" -ge "$DEADLINE" ] && { reason=deadline; break; }
  any=0; states=""
  for r in "${RESOURCES[@]}"; do IFS=: read -r z q n a <<<"$r"
    st=$(qr_state "$z" "$q"); [ -n "$st" ] && any=1
    states+="$(echo $z|cut -c1-12)=$st "
    [ "$st" = "ACTIVE" ] && [ "$first_active" = "0" ] && { first_active=$now; say "FIRST ACTIVE: $q ($z, $a)"; }
  done
  [ "$any" = "0" ] && { reason=all-gone; break; }
  [ "$first_active" != "0" ] && [ $(( now - first_active )) -ge "$ACTIVE_MAX_SECS" ] && { reason=active-cap; break; }
  say "  $states $(( (DEADLINE-now)/60 ))m left$([ "$first_active" != "0" ] && echo " | billed $(( (now-first_active)/60 ))m")"
  sleep 60
done
say "TEARDOWN: $reason"; kill_all; sleep 20
say "== verify-zero =="
for r in "${RESOURCES[@]}"; do IFS=: read -r z q n a <<<"$r"; echo "  $z qr=$(qr_state "$z" "$q" || echo none)"; done
gcloud storage cp /tmp/mz_watchdog.log "gs://focus-surfer-494820-g0-telemetry/tpu/mz_watchdog_$(date -u +%Y%m%d_%H%M%S).log" --project=$PROJECT 2>/dev/null || true
gcloud compute instances stop "$SELF" --project=$PROJECT --zone=us-central1-a --quiet >/dev/null 2>&1 || true
