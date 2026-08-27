#!/bin/bash
# Probe driver: single question -- can ANY runtime expose libtpu metrics?
# Runs the probe TWICE on the winner: once with the image's own python (does the
# runtime image ship working monitoring out of the box?) and once after installing
# our pinned deps (does pip libtpu change anything?). Uploads both JSONs.
set -uo pipefail
source /tmp/mz_common.sh
SESSION=${SESSION:?}
GCS=gs://focus-surfer-494820-g0-telemetry/tpu/${SESSION}
MARKER=${GCS}/control/DONE.txt
WAIT_MIN=${WAIT_MIN:-460}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
finish(){ say "writing DONE"; echo "$(date -u)" >/tmp/done.txt
  gcloud storage cp /tmp/probe_driver.log "${GCS}/control/driver.log" --project=$PROJECT 2>/dev/null||true
  gcloud storage cp /tmp/done.txt "$MARKER" --project=$PROJECT 2>/dev/null||true; }
trap finish EXIT
deadline=$(( $(date +%s) + WAIT_MIN*60 )); WZ=""; WQ=""; WN=""; WA=""
say "watching ${#RESOURCES[@]} requests"
while [ "$(date +%s)" -lt "$deadline" ]; do
  alive=0
  for r in "${RESOURCES[@]}"; do IFS=: read -r z q n a <<<"$r"
    st=$(qr_state "$z" "$q"); [ -n "$st" ] && alive=1
    [ "$st" = "ACTIVE" ] && { WZ=$z; WQ=$q; WN=$n; WA=$a; break 2; }
  done
  [ "$alive" = "0" ] && { say "all gone"; exit 5; }
  sleep 30
done
[ -z "$WZ" ] && { say "none provisioned"; exit 4; }
say "WINNER: $WN ($WA @ $WZ); releasing others"
for r in "${RESOURCES[@]}"; do IFS=: read -r z q n a <<<"$r"; [ "$q" = "$WQ" ] || kill_one "$z" "$q" "$n"; done
T(){ gcloud compute tpus tpu-vm ssh "$WN" --project=$PROJECT --zone="$WZ" --worker=0 --command="$1" 2>&1; }
say "--- upload probe ---"
gcloud compute tpus tpu-vm scp /home/robi/tpu_metric_probe.py "$WN":/tmp/probe.py --project=$PROJECT --zone="$WZ" --worker=0 >/dev/null 2>&1
T "ls -la /tmp/probe.py; python3 -c 'import sys;print(sys.version)'" | tail -3
say "--- PASS 1: image python, NO pip install (does the runtime ship monitoring?) ---"
T "cd /tmp && timeout 300 python3 probe.py --hold 40 --json /tmp/p1.json 2>&1 | tail -45" | tail -55
say "--- PASS 2: after installing pinned deps ---"
T "pip install -q -U pip >/dev/null 2>&1; pip install -q 'jax[tpu]==0.6.2' libtpu==0.0.17 tpu-info==0.14.2 psutil -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1 | tail -2; cd /tmp && timeout 300 python3 probe.py --hold 40 --json /tmp/p2.json 2>&1 | tail -45" | tail -60
say "--- collect JSONs ---"
for f in p1 p2; do
  gcloud compute tpus tpu-vm ssh "$WN" --project=$PROJECT --zone="$WZ" --worker=0 --command="cat /tmp/$f.json 2>/dev/null" > /tmp/$f.json 2>/dev/null
  gcloud storage cp /tmp/$f.json "${GCS}/probe/${WA//\//_}_$f.json" --project=$PROJECT 2>/dev/null || true
done
say "probe driver complete on $WA"
