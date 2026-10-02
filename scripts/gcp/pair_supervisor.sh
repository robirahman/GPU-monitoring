#!/usr/bin/env bash
# pair_supervisor.sh -- keep a collection pair alive across preemptions.
#
# Why this exists
# ---------------
# provision_spot_pair.sh exits 0 the moment a pair lands, which is correct for a
# provisioner and wrong for a round. On 2026-10-01 a pair landed at 21:28Z, was
# preempted at 21:35Z before bootstrap even finished, and nothing relaunched it:
# seven hours passed with no instances and no work. Four preemptions that day,
# surviving 23, 23 and 7 minutes.
#
# The round is restartable -- run_stage skips any cell whose .done marker exists
# -- so short windows DO accumulate, but only if something keeps reopening them.
# That is this script's whole job.
#
#   RUN_STAMP=... SESSION=... ONLY_CELLS=... bash scripts/gcp/pair_supervisor.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PROJECT="${PROJECT:-focus-surfer-494820-g0}"
RUN_STAMP="${RUN_STAMP:?set RUN_STAMP}"
NODE0="${SPOT_NODE0:-spar-spot-0}"; NODE1="${SPOT_NODE1:-spar-spot-1}"
MAX_CYCLES="${MAX_CYCLES:-40}"
CHECK_SECS="${CHECK_SECS:-120}"
say() { echo "[$(date -u '+%H:%M:%SZ')] [supervisor] $*"; }

alive() {
  local n
  n=$(gcloud compute instances list --project="$PROJECT" \
        --filter="name=($NODE0 OR $NODE1) AND status=RUNNING" \
        --format='value(name)' 2>/dev/null | wc -l)
  echo "${n:-0}"
}

for cycle in $(seq 1 "$MAX_CYCLES"); do
  n=$(alive)
  if [ "$n" -ge 2 ]; then
    sleep "$CHECK_SECS"; continue
  fi
  # A lone survivor collects nothing -- every s16 cell needs the rendezvous --
  # and bills while failing, which is how 1.5 h went on 2026-10-01. Clear it
  # before reprovisioning rather than leaving half a pair up.
  if [ "$n" -eq 1 ]; then
    say "half a pair left; clearing before relaunch"
    for nm in "$NODE0" "$NODE1"; do
      z=$(gcloud compute instances list --project="$PROJECT" \
            --filter="name=$nm" --format='value(zone)' 2>/dev/null | head -1)
      [ -n "$z" ] && gcloud compute instances delete "$nm" --project="$PROJECT" \
        --zone="$z" --quiet >/dev/null 2>&1
    done
  fi
  say "cycle $cycle: no pair -- provisioning"
  bash scripts/gcp/provision_spot_pair.sh >> "/tmp/supervisor_${RUN_STAMP}.log" 2>&1 \
    && say "cycle $cycle: pair up" \
    || say "cycle $cycle: no capacity this sweep"
  sleep "$CHECK_SECS"
done
say "max cycles reached; stopping"
