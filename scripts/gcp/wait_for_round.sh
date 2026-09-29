#!/usr/bin/env bash
# wait_for_round.sh -- block until a collection round is genuinely over.
#
# This has now reported a false "ALL TRACKS COMPLETE" three times, each for a
# different reason, so the design notes matter more than the code:
#
#   1. A gcloud --filter expression that matched nothing, read as "no nodes".
#      Fixed by filtering locally with grep against a listing we can eyeball.
#   2. PAT defaulting to ^spar-gpu while the round ran as spar-nic-*. Fixed by
#      requiring the run to have been SEEN before it can be seen to finish.
#   3. (this one) The resize-request check called a subcommand that does not
#      exist -- `instance-group-manager-resize-requests list` -- against a
#      HARDCODED zone list that did not contain the round's zones. Both faults
#      independently return nothing, and nothing was read as "no requests". A
#      pair appeared for a moment and vanished, which satisfied (2), and the
#      waiter called a round complete while ten requests sat ACCEPTED.
#
# The shape of every one of those: something that produced no output was treated
# as evidence of absence. So this version takes its zones from the round's own
# manifest in GCS rather than a list I maintain by hand, uses the real
# subcommand (`resize-requests list MIG --zone Z`, MIG positional), and treats a
# query it cannot run as UNKNOWN -- never as absence.
#
#   RUN_STAMP=20260929a SESSION=20260929 PAT='^spar-hpc' \
#     bash scripts/gcp/wait_for_round.sh [poll_seconds]
set -uo pipefail
PROJECT="${PROJECT:-focus-surfer-494820-g0}"
BUCKET="${BUCKET:-gs://focus-surfer-494820-g0-telemetry}"
POLL="${1:-600}"
PAT="${PAT:?set PAT to the node-name prefix for this round, e.g. ^spar-hpc}"
RUN_STAMP="${RUN_STAMP:?set RUN_STAMP, e.g. 20260929a}"
SESSION="${SESSION:-${RUN_STAMP%[a-z]}}"
CTRL="${BUCKET}/runs/${SESSION}/control"
MANIFEST="${CTRL}/manifest_${RUN_STAMP}.txt"

say() { echo "[$(date -u +%H:%MZ)] $*"; }

# Zones come from the manifest the provisioner wrote, so they are by
# construction the zones this round actually queued. A hand-kept default is what
# made failure (3) possible.
MF=$(mktemp); trap 'rm -f "$MF"' EXIT
if ! timeout 120 gsutil cat "$MANIFEST" > "$MF" 2>/dev/null || [ ! -s "$MF" ]; then
  echo "FATAL: cannot read $MANIFEST -- without it the zone list would be a guess," >&2
  echo "       and guessing the zone list is exactly how this reported a false" >&2
  echo "       completion before. Check RUN_STAMP/SESSION." >&2
  exit 2
fi
say "watching $(wc -l < "$MF") zone(s) from $MANIFEST"

seen_any=0
clean=0
while :; do
  # --- instances -----------------------------------------------------------
  raw=$(timeout 120 gcloud compute instances list --project="$PROJECT" \
          --format="value(name,status,zone)" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ] || [ -z "$raw" ]; then
    # Empty overall cannot be right: this project always has spar-analysis-0.
    say "instance listing unavailable (rc=$rc) -- not counting"
    clean=0; sleep "$POLL"; continue
  fi
  inst=$(printf '%s\n' "$raw" | grep -E "$PAT" | tr '\n' ';')

  # --- resize requests, per the manifest -----------------------------------
  reqs=""; req_unknown=0
  while read -r z mig req; do
    [ -z "${z:-}" ] && continue
    out=$(timeout 60 gcloud compute instance-groups managed resize-requests list \
            "$mig" --project="$PROJECT" --zone="$z" \
            --format="value(name,state)" 2>/dev/null); qrc=$?
    if [ $qrc -ne 0 ]; then
      # A MIG deleted by teardown makes this fail too, which is legitimately
      # "gone" -- but so does a transient API error, and the two are not
      # distinguishable here. Confirm the MIG is really absent before believing
      # the request is.
      if timeout 45 gcloud compute instance-groups managed describe "$mig" \
           --project="$PROJECT" --zone="$z" >/dev/null 2>&1; then
        req_unknown=1                     # MIG exists, query failed => unknown
      fi
      continue
    fi
    # Terminal states are not "queued". ACCEPTED/PROVISIONING still are.
    live=$(printf '%s\n' "$out" | grep -viE "cancelled|failed|succeeded" | grep -c . )
    [ "${live:-0}" -gt 0 ] && reqs="$reqs $z:$(printf '%s' "$out" | tr '\n' ',')"
  done < "$MF"

  # A queued request means the round is alive even with no instances yet, so it
  # counts as having seen it. Failure (2) only tracked instances.
  { [ -n "$inst" ] || [ -n "${reqs// /}" ]; } && seen_any=1

  if [ "$seen_any" = "0" ]; then
    say "nothing matching '$PAT' and no live requests yet -- run has not started"
    clean=0; sleep "$POLL"; continue
  fi
  if [ "$req_unknown" = "1" ]; then
    say "a resize-request query failed on a MIG that still exists -- unknown, not counting"
    clean=0; sleep "$POLL"; continue
  fi

  if [ -z "$inst" ] && [ -z "${reqs// /}" ]; then
    clean=$((clean+1))
    say "no nodes, no live requests (${clean}/2)"
    [ $clean -ge 2 ] && { echo "ALL TRACKS COMPLETE at $(date -u +%FT%TZ)"; exit 0; }
  else
    clean=0
    say "inst=${inst:-none} reqs=${reqs:-none}"
  fi
  sleep "$POLL"
done
