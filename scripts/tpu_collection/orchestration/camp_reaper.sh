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
# Exit only once a driver has been SEEN and then disappears. Exiting on a missing
# driver at startup was wrong: launched in the gap between one campaign's exit and
# the next one's start, the reaper died instantly ("driver gone; reaper exiting"
# 0 seconds after arming) and left the new campaign unsupervised -- which is
# precisely the window it exists to cover.
seen_driver=0
grace_left=${GRACE_CYCLES:-30}
while :; do
  if pgrep -f "$DRIVER" >/dev/null 2>&1; then
    seen_driver=1
  elif [ "$seen_driver" = "1" ]; then
    say "driver gone; reaper exiting"; break
  else
    grace_left=$((grace_left - 1))
    if [ "$grace_left" -le 0 ]; then
      say "no driver appeared within the grace window; reaper exiting"; break
    fi
    say "waiting for a driver to appear ($grace_left cycle(s) of grace left)"
    sleep "$INTERVAL"; continue
  fi

  lg=${LOG:-$(_newest_log)}
  if [ -z "$lg" ] || [ ! -s "$lg" ]; then
    say "no campaign log with content yet -- standing by"; sleep "$INTERVAL"; continue
  fi
  # A claim only counts while it is STILL HELD. campaign.sh logs "torn down <key>"
  # when it releases a scope, and between claims there is no claimed group at all
  # -- so treating the last CLAIMED line as current makes every arriving grant
  # look like a stray. Measured 2026-09-02 20:56Z: t1 had been collected and torn
  # down, and the reaper deleted a v5litepod-16 that was CREATING for the NEXT
  # claim. That shape has now been granted three times and reaped unused three
  # times, twice by this script. Between claims the reaper must do nothing: the
  # campaign's own release loop handles duplicates once it claims.
  claim_ln=$(grep -an "CLAIMED" "$lg" 2>/dev/null | tail -1)
  down_ln=$(grep -an "torn down" "$lg" 2>/dev/null | tail -1)
  claim_no=${claim_ln%%:*}; down_no=${down_ln%%:*}
  if [ -n "$down_no" ] && [ -n "$claim_no" ] && [ "$down_no" -gt "$claim_no" ]; then
    say "last claim was torn down (line $down_no > $claim_no); nothing is held -- standing by"
    sleep "$INTERVAL"; continue
  fi
  claim=${claim_ln#*:}
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
