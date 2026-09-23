#!/usr/bin/env bash
# wait_for_round.sh -- block until every collection pair of a round is gone.
#
# Flex-start nodes self-delete when their queue finishes, so "no spar-gpu*
# instances and no open resize requests" is completion. The trap is that a
# FAILED query also returns nothing, and an earlier version of this treated that
# as success: it declared "ALL TRACKS COMPLETE" eleven minutes into a
# thirteen-hour run, with 32 GPUs still busy, because its filter
# (name~^spar-gpu(f|u|b)) matched nothing in gcloud's filter dialect.
#
# So: completion requires POSITIVE evidence. The listing must succeed, must be
# non-empty overall (the project always has spar-analysis-0), and must contain no
# collection node. A failed or empty-overall listing is "unknown" and waits.
#
#   bash scripts/gcp/wait_for_round.sh [poll_seconds]
set -uo pipefail
PROJECT="${PROJECT:-focus-surfer-494820-g0}"
POLL="${1:-600}"
PAT="${PAT:-^spar-gpu}"
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-east4-a us-east4-c us-west1-b asia-southeast1-b asia-southeast1-c europe-west4-a}"

clean=0
while :; do
  raw=$(timeout 120 gcloud compute instances list --project="$PROJECT" \
          --format="value(name,status,zone)" 2>/dev/null); rc=$?
  if [ $rc -ne 0 ] || [ -z "$raw" ]; then
    # Query failed, or returned nothing at all -- which cannot be right, since
    # this project always has at least the analysis box. Do not count it.
    echo "[$(date -u +%H:%MZ)] listing unavailable (rc=$rc) -- not treating as complete"
    clean=0; sleep "$POLL"; continue
  fi
  inst=$(printf '%s\n' "$raw" | grep -E "$PAT" | tr '\n' ';')

  reqs=""
  for z in $ZONES; do
    r=$(timeout 45 gcloud beta compute instance-groups managed \
          instance-group-manager-resize-requests list --project "$PROJECT" --zone "$z" \
          --format="value(name)" 2>/dev/null | tr '\n' ' ')
    [ -n "${r// /}" ] && reqs="$reqs $z:$r"
  done

  if [ -z "$inst" ] && [ -z "${reqs// /}" ]; then
    clean=$((clean+1))
    echo "[$(date -u +%H:%MZ)] no collection nodes, no queued requests (${clean}/2)"
    [ $clean -ge 2 ] && { echo "ALL TRACKS COMPLETE at $(date -u +%FT%TZ)"; exit 0; }
  else
    clean=0
    echo "[$(date -u +%H:%MZ)] inst=${inst:-none} reqs=${reqs:-none}"
  fi
  sleep "$POLL"
done
