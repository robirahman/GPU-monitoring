#!/bin/bash
# Persistent TPU collection campaign: park requests for EVERY uncollected target,
# claim whatever provisions first, run that scope's cells, re-park, repeat.
#
# ── Why this shape ───────────────────────────────────────────────────────────
# Measured provisioning record for this project:
#
#     size                    attempts   landed   typical wait
#     v6e-1 (1 chip)               5        5      60s - 2min
#     2x v6e-1 (multislice)        2        2      2 - 10min
#     v6e-8 (1 host)               1        0      never
#     v6e-16 (2 hosts)             5        1      22min (once)
#
# Host-sized and larger capacity is scarce: 1 for 6. Serially requesting one size
# and waiting is therefore a losing strategy -- each failed attempt burns a
# multi-hour wall-clock window (a 3h v6e-8 wait produced nothing). But a QUEUED
# REQUEST IS FREE until it provisions, so parking many requests across sizes,
# generations and zones simultaneously costs nothing and converts an availability
# gamble into a lottery we keep entering.
#
# Every target is independently useful, so whichever lands is progress:
#   tpuTMS on 2x v6e-1   the DCN channel -- the plan's core question, and the
#                        configuration that has never failed to materialise
#   tpuT8  on v6e-8      single-host baseline
#   tpuT16 on v6e-16     cross-host ICI (HLO-exec engages)
#   tpuTMS on 2x v6e-8   full multislice, both boundaries at scale
#   v5e variants         cross-generation comparison (cheaper silicon)
#
# Completion is recorded in GCS, so the campaign is resumable across Cloud Shell
# disconnects: a scope already collected is skipped on the next pass.
#
# Usage (from spar-analysis-0, NOT Cloud Shell -- it must outlive the terminal):
#   SESSION=camp_$(date -u +%Y%m%d_%H%M%S) nohup bash campaign.sh &
set -uo pipefail
PROJECT=${PROJECT:-focus-surfer-494820-g0}
SESSION=${SESSION:?set SESSION}
BUCKET=${BUCKET:-gs://focus-surfer-494820-g0-telemetry/tpu}
GCS=${BUCKET}/${SESSION}
DONEDIR=${BUCKET}/collected          # per-scope completion markers, campaign-wide
REPS=${REPS:-3}
DUR=${DUR:-600}
MAX_HOURS=${MAX_HOURS:-24}
PER_RUN_CAP_SEC=${PER_RUN_CAP_SEC:-21600}   # 6h hard cap on any one billed slice
LOCKDIR=${LOCKDIR:-/tmp/spar_locks}; mkdir -p "$LOCKDIR"
MYLOCK="$LOCKDIR/camp.$SESSION.$$"; echo "$$" > "$MYLOCK"
ZONES=${ZONES:-"us-east5-a us-east5-b europe-west4-a us-central1-a"}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
cleanup(){ rm -f "$MYLOCK" 2>/dev/null||true; }
trap cleanup EXIT

# target: key|accel|node_count|runtime|scope
TARGETS=(
  "ms1|v6e-1|2|v2-alpha-tpuv6e|tpuTMS"
  "t8|v6e-8|1|v2-alpha-tpuv6e|tpuT8"
  "t16|v6e-16|1|v2-alpha-tpuv6e|tpuT16"
  "ms8|v6e-8|2|v2-alpha-tpuv6e|tpuTMS"
  "v5t8|v5litepod-8|1|v2-alpha-tpuv5-lite|tpuT8"
  "v5t16|v5litepod-16|1|v2-alpha-tpuv5-lite|tpuT16"
  "v5ms|v5litepod-1|2|v2-alpha-tpuv5-lite|tpuTMS"
)

collected(){ gcloud storage ls "${DONEDIR}/$1.done" --project=$PROJECT >/dev/null 2>&1; }
mark_collected(){ echo "$(date -u) $2" >/tmp/c.$1; gcloud storage cp /tmp/c.$1 "${DONEDIR}/$1.done" --project=$PROJECT >/dev/null 2>&1; }

qr_state(){ gcloud compute tpus queued-resources describe "$2" --project=$PROJECT --zone="$1" --format='value(state.state)' 2>/dev/null; }
kill_req(){ local z=$1 qr=$2 pfx=$3 nc=$4
  if [ "$nc" -gt 1 ]; then for i in $(seq 0 $((nc-1))); do
      gcloud compute tpus tpu-vm delete "${pfx}-${i}" --project=$PROJECT --zone="$z" --quiet >/dev/null 2>&1 || true; done
  fi
  gcloud compute tpus tpu-vm delete "$pfx" --project=$PROJECT --zone="$z" --quiet >/dev/null 2>&1 || true
  gcloud compute tpus queued-resources delete "$qr" --project=$PROJECT --zone="$z" --force --quiet >/dev/null 2>&1 || true
}

park_all(){   # park a request for every uncollected target in every zone
  local parked=0
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r key accel nc rt scope <<<"$t"
    collected "$key" && continue
    for z in $ZONES; do
      local nm="c${key}$(echo "$z" | tr -d '-' | cut -c3-7)"
      local shape=(--node-id="$nm")
      [ "$nc" -gt 1 ] && shape=(--node-count="$nc" --node-prefix="$nm")
      gcloud compute tpus queued-resources create "${nm}-qr" \
        --project=$PROJECT --zone="$z" "${shape[@]}" \
        --accelerator-type="$accel" --runtime-version="$rt" \
        --valid-until-duration=8h >/dev/null 2>&1 && parked=$((parked+1))
    done
  done
  say "parked $parked request(s)"
}

find_active(){  # echo "zone qr prefix nodecount scope key" for the first ACTIVE
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r key accel nc rt scope <<<"$t"
    collected "$key" && continue
    for z in $ZONES; do
      local nm="c${key}$(echo "$z" | tr -d '-' | cut -c3-7)"
      [ "$(qr_state "$z" "${nm}-qr")" = "ACTIVE" ] && { echo "$z ${nm}-qr $nm $nc $scope $key"; return 0; }
    done
  done
  return 1
}

run_scope(){  # zone prefix nodecount scope key
  local z=$1 pfx=$2 nc=$3 scope=$4 key=$5
  local first="$pfx"; [ "$nc" -gt 1 ] && first="${pfx}-0"
  say "RUN $key: scope=$scope nodes=$nc @ $z (first=$first)"
  # setup every worker of every slice, in parallel
  for i in $(seq 0 $((nc-1))); do
    local n="$pfx"; [ "$nc" -gt 1 ] && n="${pfx}-${i}"
    gcloud compute tpus tpu-vm scp /tmp/.pat "$n":/tmp/.pat --project=$PROJECT --zone="$z" --worker=all >/dev/null 2>&1 &
  done; wait
  for i in $(seq 0 $((nc-1))); do
    local n="$pfx"; [ "$nc" -gt 1 ] && n="${pfx}-${i}"
    ( gcloud compute tpus tpu-vm ssh "$n" --project=$PROJECT --zone="$z" --worker=all --command="set -e
sudo apt-get update -q >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip git >/dev/null 2>&1
cd \$HOME && rm -rf repo
git -c credential.helper='!f(){ echo username=x-access-token; echo \"password=\$(cat /tmp/.pat)\"; };f' clone -q --branch multinode-tpu-experiments https://github.com/robirahman/GPU-monitoring.git repo
cd repo && pip install -q -r requirements_tpu.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1|tail -1
rm -f /tmp/.pat; echo SETUP_OK_\$(hostname)" 2>&1 | grep -cE 'SETUP_OK' ) &
  done; wait
  say "  setup done; launching queue"
  PROJECT=$PROJECT ZONE="$z" TPU_NAME="$pfx" NODE_COUNT="$nc" NODE_PREFIX="$pfx" \
    SCOPE="$scope" REPO_DIR='$HOME/repo' BUCKET="$BUCKET" SESSION="${SESSION}_${key}" \
    ADVERSARIAL=1 REPS=$REPS DUR=$DUR GAP=30 \
    timeout "$PER_RUN_CAP_SEC" bash /tmp/tpu_queue.sh 2>&1 \
    | grep -iE 'derived NUM_WORKERS|START|OK  |FAIL|queue complete|done:|failed:' | tail -40
  local rc=$?
  say "  queue finished (rc=$rc)"
  mark_collected "$key" "scope=$scope zone=$z nodes=$nc rc=$rc"
}

END=$(( $(date +%s) + MAX_HOURS*3600 ))
say "campaign start: ${#TARGETS[@]} targets, zones=[$ZONES], reps=$REPS dur=${DUR}s, until $(date -u -d @$END +%H:%MZ)"
park_all
while [ "$(date +%s)" -lt "$END" ]; do
  remaining=0
  for t in "${TARGETS[@]}"; do IFS='|' read -r key _ _ _ _ <<<"$t"; collected "$key" || remaining=$((remaining+1)); done
  [ "$remaining" -eq 0 ] && { say "ALL TARGETS COLLECTED"; break; }
  if hit=$(find_active); then
    read -r z qr pfx nc scope key <<<"$hit"
    say "CLAIMED $key ($pfx @ $z)"
    # release every other parked request so nothing else provisions while we work
    for t in "${TARGETS[@]}"; do
      IFS='|' read -r k2 a2 n2 r2 s2 <<<"$t"
      for z2 in $ZONES; do
        nm2="c${k2}$(echo "$z2" | tr -d '-' | cut -c3-7)"
        [ "${nm2}-qr" = "$qr" ] && continue
        kill_req "$z2" "${nm2}-qr" "$nm2" "$n2" &
      done
    done
    run_scope "$z" "$pfx" "$nc" "$scope" "$key"
    kill_req "$z" "$qr" "$pfx" "$nc"
    say "torn down $key; re-parking remaining targets"
    park_all
  else
    say "  nothing ACTIVE yet ($remaining target(s) left, $(( (END-$(date +%s))/60 ))m budget)"
    sleep 120
  fi
done
say "campaign end"
for t in "${TARGETS[@]}"; do IFS='|' read -r key _ _ _ scope <<<"$t"
  collected "$key" && say "  COLLECTED $key ($scope)" || say "  missing   $key ($scope)"; done
# final sweep: nothing may be left billing
for t in "${TARGETS[@]}"; do IFS='|' read -r key a nc r s <<<"$t"
  for z in $ZONES; do nm="c${key}$(echo "$z" | tr -d '-' | cut -c3-7)"; kill_req "$z" "${nm}-qr" "$nm" "$nc" & done; done
wait
say "final teardown sweep complete"
