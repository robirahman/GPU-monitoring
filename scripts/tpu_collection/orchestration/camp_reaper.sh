#!/bin/bash
# Reap TPU nodes the campaign is NOT using.
#
# WHY. campaign.sh releases every non-claimed request ONCE, at claim time, then
# spends hours inside run_scope. Requests it leaves parked are still live
# (--valid-until-duration=8h) and can be granted at any point during that window,
# with nothing watching. Measured 2026-09-02: seconds after CLAIMED v5t8 @
# us-south1-a, a 2x v6e-1 group came up in us-east5-b and a v5litepod-16 in
# us-central1-a -- together ~$25/hr that would have billed for the whole 7-hour
# v5e-8 run. The one-shot release also races: 88 backgrounded kill_req calls fire
# at once with all output discarded, so failures are invisible.
#
# WHAT IT REAPS. Only TPU nodes whose name does not match the prefix named in the
# campaign log's most recent CLAIMED line, and the queued resource behind them.
# Deliberately conservative:
#   * does nothing at all until a CLAIMED line exists, so it cannot interfere
#     with the parking pass or claim a race against find_active;
#   * never touches a node whose name starts with the claimed prefix;
#   * never touches anything matching SKIP_RE (other jobs share this project);
#   * skips nodes already DELETING;
#   * exits when the campaign driver exits, so it cannot outlive its purpose.
#
# It does NOT delete parked requests in WAITING_FOR_RESOURCES: a parked request
# is free, and deleting it would defeat the whole point of parking broadly.
set -uo pipefail
PROJECT=${PROJECT:-focus-surfer-494820-g0}
# NEWEST campaign log, not a hardcoded one. Pinning it to a single file created a
# real hazard on 2026-09-02: after relaunching camp3 as camp4, the running reaper
# was still reading campaign3.log, whose last CLAIMED line named a slice that no
# longer existed -- so it would have treated camp4's legitimately claimed group as
# a stray and reaped the very instance the campaign was collecting on. It could
# not be killed (kills are refused in this environment), and the fix was to
# truncate the stale log so the reaper parsed no claim and stood by. Auto-detecting
# removes the trap.
LOG=${LOG:-}
_newest_log(){
  ls -t /tmp/campaign*.log 2>/dev/null | grep -v '\.done$' | head -1
}
DRIVER=${DRIVER:-"bash /tmp/campaign.sh"}
ZONES=${ZONES:-"us-east5-a us-east5-b us-east1-d us-south1-a europe-west4-a europe-west4-b us-central1-a asia-northeast1-b"}
SKIP_RE=${SKIP_RE:-'^sigil'}
INTERVAL=${INTERVAL:-120}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }

say "reaper armed: zones=[$ZONES] skip=/$SKIP_RE/ interval=${INTERVAL}s log=${LOG:-auto}"
while :; do
  pgrep -f "$DRIVER" >/dev/null 2>&1 || { say "driver gone; reaper exiting"; break; }

  lg=${LOG:-$(_newest_log)}
  if [ -z "$lg" ] || [ ! -s "$lg" ]; then
    say "no campaign log with content yet -- standing by"; sleep "$INTERVAL"; continue
  fi
  claim=$(grep -a "CLAIMED" "$lg" 2>/dev/null | tail -1)
  if [ -z "$claim" ]; then say "no claim yet -- standing by"; sleep "$INTERVAL"; continue; fi
  # "[hh:mm:ssZ] CLAIMED <key> (<prefix> @ <zone>)"
  pfx=$(echo "$claim" | sed 's/.*(\([^ ]*\) @ .*/\1/')
  czone=$(echo "$claim" | sed 's/.*@ //; s/).*//' | tr -d ' ')
  [ -z "$pfx" ] && { say "could not parse claim: $claim"; sleep "$INTERVAL"; continue; }

  for z in $ZONES; do
    while read -r nm st; do
      [ -z "$nm" ] && continue
      case "$nm" in ${pfx}*) continue ;; esac          # the claimed group
      echo "$nm" | grep -qE "$SKIP_RE" && continue      # someone else's job
      [ "$st" = "DELETING" ] && continue
      say "REAP $nm ($st) in $z -- not the claimed group ($pfx @ $czone)"
      gcloud compute tpus tpu-vm delete "$nm" --project="$PROJECT" --zone="$z" --quiet >/dev/null 2>&1 &
      # the request behind it, or it simply provisions again
      qr=$(echo "$nm" | sed 's/-[0-9]*$//')-qr
      gcloud compute tpus queued-resources delete "$qr" --project="$PROJECT" --zone="$z" --force --quiet >/dev/null 2>&1 &
    done < <(gcloud compute tpus tpu-vm list --project="$PROJECT" --zone="$z" \
             --format='value(name,state)' 2>/dev/null)
  done
  wait
  sleep "$INTERVAL"
done
