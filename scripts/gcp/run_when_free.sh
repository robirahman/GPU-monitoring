#!/usr/bin/env bash
# Wait for a GPU pair to stop, then restart it and launch a queue on it.
#
# WHY THIS EXISTS. Two things now stop a pair on their own: STOP_WHEN_DONE=1
# powers it off when its queue drains (added after an idle pair burned ~$658),
# and Spot preemption stops it whenever capacity is reclaimed. Both leave the
# boot disk intact, so the next run is a restart rather than a rebuild -- but
# something has to notice and do it. Polling here means the follow-up run
# starts minutes after the machine frees up instead of whenever a human looks.
#
#   ZONE=us-central1-a NODE0=spar-gpu-0 NODE1=spar-gpu-1 \
#   RUN_STAMP=h100obl1 ONLY_CELLS='^s16_(a|b)$' ARENA_MB=9216 \
#     nohup bash scripts/gcp/run_when_free.sh &
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
ZONE="${ZONE:?set ZONE}"
NODE0="${NODE0:?set NODE0}"; NODE1="${NODE1:?set NODE1}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
RUN_STAMP="${RUN_STAMP:?set RUN_STAMP}"
ONLY_CELLS="${ONLY_CELLS:-}"
POLL="${POLL:-300}"
MAX_WAIT="${MAX_WAIT:-86400}"
say() { echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] $*"; }
st()  { gcloud compute instances describe "$1" --zone="$ZONE" --project="$PROJECT" \
          --format='value(status)' 2>/dev/null; }

say "waiting for $NODE0 + $NODE1 to stop (poll ${POLL}s, giving up after ${MAX_WAIT}s)"
t0=$SECONDS
while :; do
  s0="$(st "$NODE0")"; s1="$(st "$NODE1")"
  [ "$s0" != "RUNNING" ] && [ "$s1" != "RUNNING" ] && { say "both stopped ($s0/$s1)"; break; }
  if [ $((SECONDS - t0)) -ge "$MAX_WAIT" ]; then
    say "GIVING UP after ${MAX_WAIT}s -- still $s0/$s1"; exit 1
  fi
  sleep "$POLL"
done

say "starting the pair"
gcloud compute instances start "$NODE0" "$NODE1" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1

for n in "$NODE0" "$NODE1"; do
  for _ in $(seq 1 30); do
    gcloud compute ssh "$n" --zone="$ZONE" --project="$PROJECT" \
      --command=true >/dev/null 2>&1 && { say "$n reachable"; break; }
    sleep 20
  done
done

say "staging repo + runbook scripts"
ZONE="$ZONE" PROJECT="$PROJECT" bash scripts/gcp/stage_repo.sh "$NODE0" "$NODE1" || exit 1
for n in "$NODE0" "$NODE1"; do
  gcloud compute scp --zone="$ZONE" --project="$PROJECT" scripts/gcp/*.sh "$n:/tmp/" >/dev/null 2>&1
  gcloud compute ssh "$n" --zone="$ZONE" --project="$PROJECT" \
    --command='sudo cp /tmp/*.sh /opt/spar/gcp/' >/dev/null 2>&1
done

# STOP_WHEN_DONE again, so this follow-up run also retires itself.
i=0
for n in "$NODE0" "$NODE1"; do
  say "launching queue on $n (node_rank $i)"
  gcloud compute ssh "$n" --zone="$ZONE" --project="$PROJECT" --command="
    pgrep -f 'sync_data.sh --loop' >/dev/null || sudo nohup bash /opt/spar/gcp/sync_data.sh --loop 120 >/tmp/sync_loop.log 2>&1 &
    sleep 1
    sudo RUN_STAMP='$RUN_STAMP' NNODES=2 ARENA_MB='${ARENA_MB:-9216}' STOP_WHEN_DONE=1 \
      ONLY_CELLS='$ONLY_CELLS' bash /opt/spar/gcp/launch_queue.sh" 2>&1 | grep -a launched
  i=$((i+1))
done
say "done -- $RUN_STAMP launched"
