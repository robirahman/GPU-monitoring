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
# Overridable so a relaunch can start from a clean slate without deleting
# anything: point DONEDIR at a new prefix and every target is revisited.
DONEDIR=${DONEDIR:-${BUCKET}/collected}   # per-scope completion markers
REPS=${REPS:-3}
DUR=${DUR:-600}
MAX_HOURS=${MAX_HOURS:-24}
PER_RUN_CAP_SEC=${PER_RUN_CAP_SEC:-21600}   # 6h hard cap on any one billed slice
LOCKDIR=${LOCKDIR:-/tmp/spar_locks}; mkdir -p "$LOCKDIR"
MYLOCK="$LOCKDIR/camp.$SESSION.$$"; echo "$$" > "$MYLOCK"
# Widened after every host-sized request in four zones went ungranted for a
# week. Parking is free until a request provisions, so the only cost of more
# zones is the release loop having more to sweep. v5litepod is not offered in
# several of these; those creates just fail and are counted as unparked.
ZONES=${ZONES:-"us-east5-a us-east5-b us-east1-d us-south1-a europe-west4-a europe-west4-b us-central1-a asia-northeast1-b"}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
cleanup(){ rm -f "$MYLOCK" 2>/dev/null||true; }
trap cleanup EXIT

# target: key|accel|node_count|runtime|scope
#
# ORDER MATTERS: find_active() claims the FIRST ACTIVE target it finds, so the
# rare shapes must come first. Listed commonest-first (as it was), an easily
# granted v6e-1 group is claimed every cycle and a v6e-16 that happened to be
# ACTIVE at the same moment gets released unused.
#
# What capacity has actually granted, over a week of parking: v6e-1 freely (13x),
# 2x v6e-1 multislice 3x, v6e-16 once (2026-08-26). NEVER ACTIVE: v6e-8, 2x
# v6e-8, v5litepod-8, v5litepod-16, 2x v5litepod-1. Hence ms4/ms8x1 -- a scale
# ladder built from the 1-chip slices that DO provision, which also puts every
# collective on DCN. They are last because they are the fallback, not the prize.
#
# Note two targets can share a scope (ms1 and ms8 are both 2 slices, differing in
# chips per slice). Their GCS prefixes differ by key, and chip count must be read
# from the telemetry rather than the scope label -- see registry_tpu.py.
TARGETS=(
  "t16|v6e-16|1|v2-alpha-tpuv6e|tpuT16"
  "ms8|v6e-8|2|v2-alpha-tpuv6e|tpuTMS"
  "t8|v6e-8|1|v2-alpha-tpuv6e|tpuT8"
  "v5t16|v5litepod-16|1|v2-alpha-tpuv5-lite|tpuT16"
  "v5t8|v5litepod-8|1|v2-alpha-tpuv5-lite|tpuT8"
  "v5ms|v5litepod-1|2|v2-alpha-tpuv5-lite|tpuTMS"
  "t4|v6e-4|1|v2-alpha-tpuv6e|tpuT4"
  "ms8x1|v6e-1|8|v2-alpha-tpuv6e|tpuTMS8"
  "ms4|v6e-1|4|v2-alpha-tpuv6e|tpuTMS4"
  "ms1|v6e-1|2|v2-alpha-tpuv6e|tpuTMS"
  "t1|v6e-1|1|v2-alpha-tpuv6e|tpuT1"
)
# t1 and t4 exist so the corpus has SINGLE-SLICE coverage of every class, not
# just multislice. A single slice keeps collectives on ICI (or has none at all at
# tpuT1, one chip), while a multislice group puts the same collectives on DCN --
# and that contrast is what the TPU rung is for. t8/t16 would have served, but
# neither has ever provisioned; v6e-1 has been granted 13 times and v6e-4 is
# worth trying as the smallest untested single-host shape. t1 is last in the
# order because it is the most available, so it never starves a rarer shape.

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

# The per-zone request name MUST be unique per zone. It used to be
#   c${key}$(echo "$z" | tr -d '-' | cut -c3-7)
# and us-east5-a / us-east5-b both truncate to "east5", so both zones got the
# SAME name. Two consequences, both observed on 2026-09-01:
#   1. after claiming us-east5-a, the release loop's
#      `[ "${nm2}-qr" = "$qr" ] && continue` matched the OTHER zone's
#      identically-named request, so it was never released -- its 2 v6e chips
#      billed idle alongside the claimed pair;
#   2. nothing would have reclaimed them until the final sweep, MAX_HOURS later.
# Use the full zone slug: unique, and still a legal TPU node name.
park_all(){   # park a request for every uncollected target in every zone
  local parked=0
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r key accel nc rt scope <<<"$t"
    collected "$key" && continue
    for z in $ZONES; do
      local nm="c${key}$(echo "$z" | tr -d '-')"
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
      local nm="c${key}$(echo "$z" | tr -d '-')"
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
  local setup_out=/tmp/setup.$key.log; : > "$setup_out"
  for i in $(seq 0 $((nc-1))); do
    local n="$pfx"; [ "$nc" -gt 1 ] && n="${pfx}-${i}"
    ( gcloud compute tpus tpu-vm ssh "$n" --project=$PROJECT --zone="$z" --worker=all --command="set -e
sudo apt-get update -q >/dev/null 2>&1
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip git >/dev/null 2>&1
cd \$HOME && rm -rf repo
git -c credential.helper='!f(){ echo username=x-access-token; echo \"password=\$(cat /tmp/.pat)\"; };f' clone -q --branch multinode-tpu-experiments https://github.com/robirahman/GPU-monitoring.git repo
cd repo && pip install -q -r requirements_tpu.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1|tail -1
rm -f /tmp/.pat
python3 -c 'import jax; print(\"JAXOK\", jax.__version__, jax.device_count())'
echo SETUP_OK_\$(hostname)" 2>&1 >> "$setup_out" ) &
  done; wait
  # ASSERT setup. Previously the SETUP_OK count went to a discarded subshell, so a
  # failed clone or pip install passed silently -- and then every cell died in ~35s
  # with 0/N workers, which is almost certainly what happened overnight.
  local n_ok; n_ok=$(grep -c 'SETUP_OK' "$setup_out" 2>/dev/null || echo 0)
  local n_jax; n_jax=$(grep -c 'JAXOK' "$setup_out" 2>/dev/null || echo 0)
  say "  setup: SETUP_OK=$n_ok JAXOK=$n_jax (need >=1 of each)"
  if [ "$n_ok" -lt 1 ] || [ "$n_jax" -lt 1 ]; then
    say "  SETUP FAILED -- not running cells, not marking collected. Tail:"
    tail -12 "$setup_out" | sed 's/^/      /'
    return 1
  fi
  say "  setup verified; launching queue"
  PROJECT=$PROJECT ZONE="$z" TPU_NAME="$pfx" NODE_COUNT="$nc" NODE_PREFIX="$pfx" \
    SCOPE="$scope" REPO_DIR='$HOME/repo' BUCKET="$BUCKET" SESSION="${SESSION}_${key}" \
    ADVERSARIAL=1 REPS=$REPS DUR=$DUR GAP=30 \
    timeout "$PER_RUN_CAP_SEC" bash /tmp/tpu_queue.sh > /tmp/q.$key.log 2>&1
  local rc=$?
  grep -iE 'derived NUM_WORKERS|START|OK  |FAIL|queue complete|done:|failed:' /tmp/q.$key.log | tail -40
  # `rc=$?` after a PIPELINE captures the LAST command's status (tail), not the
  # queue's. That is why the campaign recorded rc=0 for three runs in which every
  # cell failed -- and then marked them collected, so they would never be retried.
  # Same defect class as the earlier driver bug; do not reintroduce a pipe here.
  local n_done n_failed
  n_done=$(grep -oE 'done: +[0-9]+' /tmp/q.$key.log | tail -1 | grep -oE '[0-9]+' || echo 0)
  n_failed=$(grep -oE 'failed: +[0-9]+' /tmp/q.$key.log | tail -1 | grep -oE '[0-9]+' || echo 0)
  say "  queue finished: rc=$rc done=${n_done:-0} failed=${n_failed:-0}"
  gcloud storage cp /tmp/q.$key.log "${GCS}/control/queue_$key.log" --project=$PROJECT >/dev/null 2>&1 || true
  # Collected means CELLS LANDED, not "the command returned".
  if [ "${n_done:-0}" -ge 1 ]; then
    mark_collected "$key" "scope=$scope zone=$z nodes=$nc rc=$rc done=$n_done failed=$n_failed"
    say "  MARKED COLLECTED ($n_done cells)"
  else
    say "  NOT marking collected: 0 cells landed (failed=$n_failed) -- target stays in the pool"
  fi
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
        nm2="c${k2}$(echo "$z2" | tr -d '-')"
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
  for z in $ZONES; do nm="c${key}$(echo "$z" | tr -d '-')"; kill_req "$z" "${nm}-qr" "$nm" "$nc" & done; done
wait
say "final teardown sweep complete"
