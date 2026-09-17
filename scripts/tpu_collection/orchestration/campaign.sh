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
# ORDER: MULTISLICE FIRST, then single-slice, each rarest-first.
#
# Reordered 2026-09-02 because the coverage gap moved. Single-slice now has the
# full class set -- the v5litepod-8 run collected 33 cells including honest
# training and both inference workloads -- while MULTISLICE has only adversarial
# and non-ML cells: the tpuTMS corpus predates the class references entirely, so
# Stage 2 cannot be trained on it and the MN-T type-confusion attack has nothing
# to attack. Multislice is also the only topology where the DCN channels carry
# any value at all (verified: dcn_collective_lat_us_* is NaN on a single slice).
# So the multislice shapes are claimed ahead of shapes we already have.
#
# What capacity has actually granted: v6e-1 freely (13x), 2x v6e-1 three times,
# v6e-16 once, v5litepod-8 twice (2026-09-02, first time ever), v5litepod-16 once
# (granted then reaped unused). Never ACTIVE: v6e-8, 2x v6e-8, 2x v5litepod-1.
# ORDER: GAP SHAPES FIRST. Reordered 2026-09-03 for the gap-filling campaign.
#
# What the corpus already has (571 streams): single slice on v5litepod-8 and
# v6e-1, multislice on 2x v6e-1 and 2x v5litepod-1, all three classes on each.
# What it does NOT have, in priority order:
#
#  1. MULTI-HOST SINGLE SLICE -- the only topology that isolates cross-HOST ICI
#     without DCN. v5litepod-16 (16 chips, 2 hosts) has been GRANTED FOUR TIMES
#     and used zero times, every time because a lower-priority target was already
#     claimed. It is now first, so a grant at any claim boundary wins. v6e-16 is
#     second, to get the same topology on the other generation.
#  2. Single-host 8-chip v6e (t8) -- the v6e single-slice ladder stops at 1 chip.
#  3. The multislice SCALE ladder (4x and 8x v6e-1), never granted.
#  4. Re-collect v5litepod-8: its 264 streams carry only 7 of the 11 telemetry
#     channels (they predate the 2026-09-02 07:40Z fix) and lack nonml_sat_hi and
#     the MN-T ladder entirely.
#
# ms1 and t1 sit last because they are already well covered (45+9 and 36 cells);
# they stay in the list only so a spare claim is not wasted.
TARGETS=(
  "v5t16|v5litepod-16|1|v2-alpha-tpuv5-lite|tpuT16"
  "t16|v6e-16|1|v2-alpha-tpuv6e|tpuT16"
  "ms8|v6e-8|2|v2-alpha-tpuv6e|tpuTMS"
  "t8|v6e-8|1|v2-alpha-tpuv6e|tpuT8"
  "ms8x1|v6e-1|8|v2-alpha-tpuv6e|tpuTMS8"
  "ms4|v6e-1|4|v2-alpha-tpuv6e|tpuTMS4"
  "v5t8|v5litepod-8|1|v2-alpha-tpuv5-lite|tpuT8"
  "t4|v6e-4|1|v2-alpha-tpuv6e|tpuT4"
  "ms1|v6e-1|2|v2-alpha-tpuv6e|tpuTMS"
  "t1|v6e-1|1|v2-alpha-tpuv6e|tpuT1"
)

# ONLY_TARGETS: bash-regex whitelist of target KEYS, for a gap-filling run that
# must not disturb what is already in flight. Without it a relaunch re-parks
# every uncollected key: it would request a SECOND ms4 group while the first is
# still collecting (a duplicate multislice group is what the us-east5-a/b zone
# collision billed us for), and it would spend the driver on t1, which lands in
# ~60 s every time, instead of waiting out the host-sized shapes that are the
# entire reason for the run.
if [ -n "${ONLY_TARGETS:-}" ]; then
  _kept=()
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r _k _rest <<<"$t"
    [[ "$_k" =~ $ONLY_TARGETS ]] && _kept+=("$t")
  done
  if [ "${#_kept[@]}" -eq 0 ]; then
    echo "ONLY_TARGETS='$ONLY_TARGETS' matched no target key -- refusing to run with an empty pool" >&2
    exit 2
  fi
  TARGETS=("${_kept[@]}")
fi
# t1 and t4 exist so the corpus has SINGLE-SLICE coverage of every class, not
# just multislice. A single slice keeps collectives on ICI (or has none at all at
# tpuT1, one chip), while a multislice group puts the same collectives on DCN --
# and that contrast is what the TPU rung is for. t8/t16 would have served, but
# neither has ever provisioned; v6e-1 has been granted 13 times and v6e-4 is
# worth trying as the smallest untested single-host shape. t1 is last in the
# order because it is the most available, so it never starves a rarer shape.

# A marker's EXISTENCE is not evidence that anything was collected. A relaunch on
# 2026-09-02 stamped every in-flight target with "stopped by relaunch", and t8, t16
# and ms8 were consequently skipped as already-collected for two days while having
# no data at all. So a marker now has to carry a stream count, and one without a
# streams= field is treated as stale rather than believed.
collected(){
  local body n
  body="$(gcloud storage cat "${DONEDIR}/$1.done" --project=$PROJECT 2>/dev/null || true)"
  [ -n "$body" ] || return 1
  n="$(printf '%s' "$body" | sed -nE 's/.*streams=([0-9]+).*/\1/p' | head -1)"
  if [ -z "$n" ]; then
    say "  IGNORING ${1}.done: no streams= evidence -- target stays in the pool"
    return 1
  fi
  [ "$n" -ge 1 ]
}
mark_collected(){
  local key="$1" detail="$2" n
  # Count what actually landed in GCS for this run, so the marker is checkable.
  n="$(gcloud storage ls "${BUCKET}/${SESSION}_${key}/**" --project=$PROJECT 2>/dev/null \
        | grep -c '\.parquet$' || true)"
  echo "$(date -u) streams=${n:-0} $detail" >/tmp/c.$key
  gcloud storage cp /tmp/c.$key "${DONEDIR}/$key.done" --project=$PROJECT >/dev/null 2>&1
}

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
  # PARALLEL, and that is not a micro-optimisation. Sequentially, each
  # `queued-resources create` took ~50 s against this project, so 11 targets x 8
  # zones was a ~70 MINUTE opening pass -- and the campaign cannot claim or
  # release anything until it finishes. Measured on 2026-09-02: two v5litepod-8
  # requests were granted 12 minutes into parking, in us-south1-a and
  # europe-west4-b, and both sat READY and unsupervised because the driver was
  # still on its 17th of 88 creates. One of them was a duplicate that the release
  # loop would have dropped immediately had the loop been running. Fan out with
  # bounded concurrency so the window is a minute, not an hour.
  local parked=0 pids=() rcfile
  rcfile=$(mktemp)
  for t in "${TARGETS[@]}"; do
    IFS='|' read -r key accel nc rt scope <<<"$t"
    collected "$key" && continue
    for z in $ZONES; do
      local nm="c${key}$(echo "$z" | tr -d '-')"
      local shape=(--node-id="$nm")
      [ "$nc" -gt 1 ] && shape=(--node-count="$nc" --node-prefix="$nm")
      (
        gcloud compute tpus queued-resources create "${nm}-qr" \
          --project=$PROJECT --zone="$z" "${shape[@]}" \
          --accelerator-type="$accel" --runtime-version="$rt" \
          --valid-until-duration=8h >/dev/null 2>&1 && echo x >> "$rcfile"
      ) &
      pids+=($!)
      # Bound concurrency: a few dozen concurrent gcloud invocations is enough to
      # collapse the window without thrashing the API or the VM.
      if [ "${#pids[@]}" -ge 16 ]; then
        for pp in "${pids[@]}"; do wait "$pp"; done
        pids=()
      fi
    done
  done
  for pp in "${pids[@]}"; do wait "$pp"; done
  parked=$(wc -l < "$rcfile" | tr -d ' ')
  rm -f "$rcfile"
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
  # `grep -c` prints 0 AND exits 1 when there is no match, so `|| echo 0`
  # appended a SECOND zero and n_ok became "0\n0". Every later comparison then
  # died with "integer expression expected", evaluated false, and the guard
  # fell through -- launching the whole queue precisely in the no-worker case
  # it was written to catch. Strip to digits and default, so the zero case is a
  # real zero.
  local n_ok n_jax
  n_ok=$(grep -c 'SETUP_OK' "$setup_out" 2>/dev/null); n_ok=${n_ok//[^0-9]/}; n_ok=${n_ok:-0}
  n_jax=$(grep -c 'JAXOK' "$setup_out" 2>/dev/null); n_jax=${n_jax//[^0-9]/}; n_jax=${n_jax:-0}
  say "  setup: SETUP_OK=$n_ok JAXOK=$n_jax of $nc worker(s) (need >=1 of each)"
  [ "$n_ok" -lt "$nc" ] && say "  WARNING: only $n_ok/$nc workers completed setup"
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
say "campaign start: ${#TARGETS[@]} targets, zones=[$ZONES], reps=$REPS dur=${DUR}s, until $(date -u -d @$END +%Y-%m-%dT%H:%MZ)"
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
