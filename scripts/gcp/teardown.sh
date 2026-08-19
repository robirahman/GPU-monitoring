#!/usr/bin/env bash
# teardown.sh -- the kill switch + end-of-sitting checklist. Run from YOUR machine.
#
#   bash scripts/gcp/teardown.sh            # best-effort final sync, then delete + verify-zero
#   bash scripts/gcp/teardown.sh --no-sync  # skip the final sync (e.g. nodes already gone)
#
# An idle pair burns ~$78/h and the most you lose to a hard kill is <=5 min of data
# (60 s .partial checkpoints + per-cell sync). WHEN IN DOUBT, TEAR DOWN.
#
# The bucket is NEVER deleted here -- that's where the telemetry lives. Pull it with
# pull_results.sh first if you haven't.
set -uo pipefail   # NOT -e: every resource class must be checked even if one delete errors
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
require_cmd gcloud

DO_SYNC=1; [[ "${1:-}" == "--no-sync" ]] && DO_SYNC=0
gcloud config set project "$PROJECT" >/dev/null 2>&1 || true

# --- best-effort final sync on each node before we pull the plug -------------
if [[ "$DO_SYNC" == "1" ]]; then
  for n in "$NODE0" "$NODE1"; do
    if gcloud compute instances describe "$n" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
      spar_say "final sync on $n (best-effort, 40s cap)"
      timeout 40 gcloud compute ssh "$n" --zone="$ZONE" --project="$PROJECT" \
        --command="sudo /usr/bin/env bash /opt/spar/gcp/sync_data.sh --final" >/dev/null 2>&1 || \
        spar_say "  (final sync on $n failed/skipped -- continuing to delete)"
    fi
  done
fi

# --- delete instances --------------------------------------------------------
for n in "$NODE0" "$NODE1"; do
  if gcloud compute instances describe "$n" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
    spar_say "deleting instance $n"
    gcloud compute instances delete "$n" --zone="$ZONE" --project="$PROJECT" --quiet || true
  fi
done

# --- delete TPU vm + any queued resource ------------------------------------
if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$TPU_ZONE" --project="$PROJECT" >/dev/null 2>&1; then
  spar_say "deleting TPU $TPU_NAME"
  gcloud compute tpus tpu-vm delete "$TPU_NAME" --zone="$TPU_ZONE" --project="$PROJECT" --quiet || true
fi
if gcloud compute tpus queued-resources describe "${TPU_NAME}-qr" --zone="$TPU_ZONE" --project="$PROJECT" >/dev/null 2>&1; then
  spar_say "deleting queued-resource ${TPU_NAME}-qr"
  gcloud compute tpus queued-resources delete "${TPU_NAME}-qr" --zone="$TPU_ZONE" --project="$PROJECT" --quiet --force || true
fi

# --- STOP-mode boot disks LINGER after an instance is only stopped (not deleted).
#     Sweep any disk matching our node names in the zone and delete leftovers. -----
LEFT_DISKS="$(gcloud compute disks list --project="$PROJECT" \
  --filter="zone:(${ZONE}) AND (name=${NODE0} OR name=${NODE1})" \
  --format="value(name)" 2>/dev/null || true)"
if [[ -n "$LEFT_DISKS" ]]; then
  while read -r d; do
    [[ -z "$d" ]] && continue
    spar_say "deleting lingering disk $d"
    gcloud compute disks delete "$d" --zone="$ZONE" --project="$PROJECT" --quiet || true
  done <<< "$LEFT_DISKS"
fi

# ============================================================================
# VERIFY-ZERO -- the checklist. Every row must read 0/none.
# ============================================================================
echo
echo "==================== teardown verify-zero ($PROJECT) ===================="
count() { local n; n="$(eval "$1" 2>/dev/null | grep -c . || true)"; echo "${n:-0}"; }

I_N="$(count "gcloud compute instances list --project=$PROJECT --filter='name~spar-gpu' --format='value(name)'")"
D_N="$(count "gcloud compute disks list --project=$PROJECT --filter='name~spar-gpu' --format='value(name)'")"
T_N="$(count "gcloud compute tpus tpu-vm list --project=$PROJECT --zone=$TPU_ZONE --format='value(name)'")"
Q_N="$(count "gcloud compute tpus queued-resources list --project=$PROJECT --zone=$TPU_ZONE --format='value(name)'")"
R_N="$(count "gcloud compute reservations list --project=$PROJECT --format='value(name)'")"

status() { [[ "$1" == "0" ]] && echo OK || echo "LEFTOVER"; }
printf "%-26s %-6s %s\n" "instances (spar-gpu)"    "$I_N" "$(status "$I_N")"
printf "%-26s %-6s %s\n" "disks (spar-gpu)"        "$D_N" "$(status "$D_N")"
printf "%-26s %-6s %s\n" "tpu-vms ($TPU_ZONE)"     "$T_N" "$(status "$T_N")"
printf "%-26s %-6s %s\n" "queued-resources"        "$Q_N" "$(status "$Q_N")"
printf "%-26s %-6s %s\n" "reservations"            "$R_N" "$(status "$R_N")"
echo "------------------------------------------------------------------------"
echo "bucket $BUCKET was NOT deleted (your telemetry lives there)."
echo "reminder: set a billing budget alert (one-time) at ~\$3k if you haven't."
TOTAL=$((I_N + D_N + T_N + Q_N + R_N))
if [[ "$TOTAL" -eq 0 ]]; then
  echo "VERIFY-ZERO CLEAN: nothing is billing."
  exit 0
else
  echo "WARNING: $TOTAL resource(s) still present -- re-run teardown or delete by hand."
  echo "  gcloud compute reservations list   # note: reservations auto-provision + bill; delete any you didn't intend"
  exit 1
fi
