#!/bin/bash
# One question: what does the libtpu metrics server actually serve?
# A single chip answers it -- the metric LIST does not depend on slice size, and
# only the megascale.* subset needs multislice to carry data.
set -uo pipefail
source /tmp/mz_common.sh
SESSION=${SESSION:?}
GCS=gs://focus-surfer-494820-g0-telemetry/tpu/${SESSION}
MARKER=${GCS}/control/DONE.txt
WAIT_MIN=${WAIT_MIN:-420}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }
finish(){ say "DONE"; echo "$(date -u)" >/tmp/done.txt
  gcloud storage cp /tmp/ls_driver.log "${GCS}/control/driver.log" --project=$PROJECT 2>/dev/null||true
  gcloud storage cp /tmp/done.txt "$MARKER" --project=$PROJECT 2>/dev/null||true; }
trap finish EXIT
deadline=$(( $(date +%s) + WAIT_MIN*60 )); WZ=""; WQ=""; WN=""; WA=""
say "watching ${#RESOURCES[@]} single-chip requests"
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
say "--- setup ---"
gcloud compute tpus tpu-vm scp /tmp/.pat "$WN":/tmp/.pat --project=$PROJECT --zone="$WZ" --worker=0 >/dev/null 2>&1
T "set -e
sudo apt-get update -q >/dev/null 2>&1; sudo DEBIAN_FRONTEND=noninteractive apt-get install -y -q python3-pip git >/dev/null 2>&1
cd \$HOME && rm -rf repo
git -c credential.helper='!f(){ echo username=x-access-token; echo \"password=\$(cat /tmp/.pat)\"; };f' clone -q --branch multinode-tpu-experiments https://github.com/robirahman/GPU-monitoring.git repo
cd repo && pip install -q -r requirements_tpu.txt -f https://storage.googleapis.com/jax-releases/libtpu_releases.html 2>&1|tail -1
rm -f /tmp/.pat; python3 -c 'import jax;print(\"jax\",jax.__version__,\"chips\",jax.device_count())'" | tail -5
say "--- THE ANSWER: ListSupportedMetrics + read each ---"
T "cd \$HOME/repo && TPU_RUNTIME_METRICS_PORTS=8431 timeout 420 python3 scripts/tpu_metric_probe.py --hold 100 --json /tmp/probe.json 2>&1 | tail -90" | tail -100
say "--- save probe json ---"
gcloud compute tpus tpu-vm ssh "$WN" --project=$PROJECT --zone="$WZ" --worker=0 --command="cat /tmp/probe.json" 2>/dev/null > /tmp/probe.json
[ -s /tmp/probe.json ] && gcloud storage cp /tmp/probe.json "${GCS}/probe/${WA}_listmetrics.json" --project=$PROJECT 2>/dev/null && say "json uploaded ($(stat -c%s /tmp/probe.json) bytes)" || say "json empty/not captured"
say "ls_driver complete on $WA"
