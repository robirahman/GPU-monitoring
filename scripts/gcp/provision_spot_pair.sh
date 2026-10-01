#!/usr/bin/env bash
# provision_spot_pair.sh -- run a collection round on SPOT, as a second horse.
#
# Why this exists
# ---------------
# On 2026-10-01 the DWS Flex Start round 20260930a had been queued 7.4 hours:
# nine zones ACCEPTED with no grant, one FAILED with VM_MIN_COUNT_NOT_REACHED.
# It is not quota -- NVIDIA_A100_GPUS is 16 in every queued region, exactly one
# pair, all unused. It is capacity.
#
# But every one of those regions also has PREEMPTIBLE_NVIDIA_A100_GPUS = 64, a
# separate pool four times larger, and a create-and-delete probe found Spot
# capacity available the same minute in us-west1-b and asia-southeast1-c -- two
# of the zones where DWS had been waiting all day.
#
# So this runs the same round on Spot while the DWS requests STAY QUEUED. It
# does NOT cancel them: they are the fallback if Spot is preempted. What it does
# do is take the GCS claim, which is what makes that fallback safe -- a DWS pair
# that later boots into an already-claimed round stands down and deletes itself
# instead of collecting the same cells twice and billing for the privilege.
#
# ALL-OR-NOTHING, by hand. A 2-node job needs both nodes; node0 running alone
# bills ~$10/h and collects nothing, so if node1 fails to land, node0 is deleted
# before moving to the next zone.
#
#   RUN_STAMP=20260930a SESSION=20260930 ONLY_CELLS='...' REPS=2 \
#     ZONES='us-west1-b asia-southeast1-c' bash scripts/gcp/provision_spot_pair.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
source scripts/gcp/env.sh 2>/dev/null || true

PROJECT="${PROJECT:-focus-surfer-494820-g0}"
BUCKET="${BUCKET:-gs://${PROJECT}-telemetry}"
RUN_STAMP="${RUN_STAMP:?set RUN_STAMP}"
SESSION="${SESSION:-${RUN_STAMP%[a-z]}}"
NODE0="${SPOT_NODE0:-spar-spot-0}"
NODE1="${SPOT_NODE1:-spar-spot-1}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-8g}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-500GB}"
MASTER_PORT="${MASTER_PORT:-29500}"
REPS="${REPS:-2}"
ONLY_CELLS="${ONLY_CELLS:-^s16_}"
ZONES="${ZONES:-us-west1-b asia-southeast1-c us-central1-a europe-west4-b}"
MAX_RUN="${MAX_RUN:-43200s}"
STAGING="${BUCKET}/staging/${RUN_STAMP}"
CTRL="${BUCKET}/runs/${SESSION}/control"
say() { echo "[$(date -u '+%H:%M:%SZ')] $*"; }

existing="$(gcloud compute instances list --project="$PROJECT" \
  --filter="name=($NODE0 OR $NODE1)" --format='value(name,zone,status)' 2>/dev/null)"
[ -n "$existing" ] && { say "REFUSING: $NODE0/$NODE1 already exist:"; printf '  %s\n' "$existing"; exit 1; }

# Same archive the MIG path stages: `git archive` streams from the object
# database, so the 9p working tree cannot corrupt it.
read -r -a STAGE_PATHS <<< "${SPAR_STAGE_PATHS:-workloads scripts classifier requirements_v2.txt requirements_gcp_extra.txt requirements_tpu.txt requirements.txt}"
COMMIT="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git archive --format=tar HEAD -- "${STAGE_PATHS[@]}" | gzip -9 > "$TMP/spar_repo.tar.gz"
say "repo archive $(du -h "$TMP/spar_repo.tar.gz" | cut -f1) at ${COMMIT:0:12} -> $STAGING/"
gcloud storage cp "$TMP/spar_repo.tar.gz" "$STAGING/spar_repo.tar.gz" >/dev/null 2>&1 || {
  say "FATAL: could not stage the repo archive"; exit 1; }

cat > "$TMP/startup.sh" <<STARTUP
#!/usr/bin/env bash
exec > >(tee -a /var/log/spar-startup.log) 2>&1
set -uo pipefail
PROJECT="$PROJECT"; BUCKET="$BUCKET"; SESSION="$SESSION"; STAMP="$RUN_STAMP"
CTRL="$CTRL"; STAGING="$STAGING"; NODE0="$NODE0"; NODE1="$NODE1"
ONLY="$ONLY_CELLS"; REPS="$REPS"; MASTER_PORT="$MASTER_PORT"
REPO=/opt/spar/repo
NODE="\$(hostname -s)"
if [ "\$NODE" = "\$NODE0" ]; then IDX=0; PEER="\$NODE1"; else IDX=1; PEER="\$NODE0"; fi
echo "=== spot startup \$NODE idx=\$IDX peer=\$PEER \$(date -u +%FT%TZ) ==="

tok() { curl -s -H 'Metadata-Flavor: Google' \\
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \\
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])'; }
enc() { python3 -c 'import sys,urllib.parse;print(urllib.parse.quote(sys.argv[1],safe=""))' "\$1"; }
bkt() { printf '%s' "\${CTRL#gs://}" | cut -d/ -f1; }
pre() { printf '%s' "\${CTRL#gs://}" | cut -d/ -f2-; }

gcs_put() {  # gcs_put <local> <object-path-under-prefix>
  curl -sf -X POST -H "Authorization: Bearer \$(tok)" \\
    -H 'Content-Type: text/plain' --data-binary @"\$1" \\
    "https://storage.googleapis.com/upload/storage/v1/b/\$(bkt)/o?uploadType=media&name=\$(enc "\$(pre)/\$2")" >/dev/null
}
gcs_get() {  # gcs_get <object-path-under-prefix> <local>
  curl -sf -H "Authorization: Bearer \$(tok)" \\
    "https://storage.googleapis.com/storage/v1/b/\$(bkt)/o/\$(enc "\$(pre)/\$1")?alt=media" -o "\$2"
}
# ifGenerationMatch=0 means "create only if absent", so exactly one pair can
# take the claim however many are racing.
gcs_claim() {
  curl -sf -X POST -H "Authorization: Bearer \$(tok)" -H 'Content-Type: text/plain' \\
    --data-binary @"\$1" \\
    "https://storage.googleapis.com/upload/storage/v1/b/\$(bkt)/o?uploadType=media&ifGenerationMatch=0&name=\$(enc "\$(pre)/claim_\$STAMP.txt")" >/dev/null
}

mkdir -p /opt/spar && cd /opt/spar
for _ in \$(seq 1 30); do
  gcloud storage cp "\$STAGING/spar_repo.tar.gz" /opt/spar/repo.tgz >/dev/null 2>&1 && break
  sleep 20
done
mkdir -p "\$REPO" && tar xzf /opt/spar/repo.tgz -C "\$REPO"

# bootstrap_node.sh reads identity from metadata; set the per-node bits it
# cannot infer from a shared template.
mkdir -p /etc/spar
SPAR_NODE_INDEX="\$IDX" NODE_INDEX="\$IDX" PEER_NAME="\$PEER" \\
  bash "\$REPO/scripts/gcp/bootstrap_node.sh" || echo "bootstrap returned \$?"

echo "\$NODE \$(date -u +%FT%TZ)" > /tmp/ready.txt
gcs_put /tmp/ready.txt "\$NODE/ready_\$STAMP.txt"

# Wait for the peer before claiming: a lone Spot node that claimed and then
# found no peer would lock the round out of its DWS fallback for nothing.
got=0
for _ in \$(seq 1 240); do
  gcs_get "\$PEER/ready_\$STAMP.txt" /tmp/peer_ready && { got=1; break; }
  sleep 30
done
if [ "\$got" -ne 1 ]; then
  echo "peer \$PEER never became ready -- standing down without claiming"
  gcloud compute instances delete "\$NODE" --zone="\$(curl -s -H 'Metadata-Flavor: Google' \\
    http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)" \\
    --quiet || shutdown -h +1
  exit 0
fi

# Take the claim. NOT cancelling the DWS resize requests on purpose: they stay
# queued as the fallback if Spot is preempted, and the claim is what makes a
# later DWS grant stand down instead of double-collecting.
if [ "\$IDX" = "0" ]; then
  echo "spot:\$NODE \$(date -u +%FT%TZ)" > /tmp/claim.txt
  if gcs_claim /tmp/claim.txt; then echo "claimed \$STAMP for the spot pair"
  else echo "claim already held -- another pair owns this round; standing down"
       gcloud compute instances delete "\$NODE0" "\$NODE1" --zone="\$(curl -s -H 'Metadata-Flavor: Google' \\
         http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)" \\
         --quiet || shutdown -h +1
       exit 0
  fi
fi

# preempt_watch.sh touches logs/.preempted on the ~30 s ACPI notice, which
# run_stage reads so the queue stops launching new cells into a dying node, and
# flushes telemetry with sync_data.sh --final. On Spot that is the difference
# between losing one cell and losing the round.
setsid bash "\$REPO/scripts/gcp/preempt_watch.sh" </dev/null >/var/log/spar-preempt.log 2>&1 & disown

HOME="\${HOME:-/root}" STAMP="\$STAMP" PEER="\$PEER" ONLY="\$ONLY" REPS="\$REPS" \\
  bash "\$REPO/scripts/gcp/run_stage_and_stop.sh"
echo "=== stage armed \$(date -u +%FT%TZ) ==="
STARTUP

mk() {  # mk <zone> <name> <idx>
  gcloud compute instances create "$2" --project="$PROJECT" --zone="$1" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type=pd-balanced \
    --provisioning-model=SPOT --instance-termination-action=DELETE \
    --max-run-duration="$MAX_RUN" --scopes=cloud-platform \
    --metadata-from-file=startup-script="$TMP/startup.sh" \
    --metadata="spar-session=${SESSION},spar-master-addr=${NODE0},spar-master-port=${MASTER_PORT},spar-bucket=${BUCKET},spar-node0-name=${NODE0},spar-node1-name=${NODE1},spar-node-index=$3,spar-run-stamp=${RUN_STAMP}" \
    --labels="spar-role=gpu,spar-stamp=$(printf '%s' "$RUN_STAMP" | tr -cd 'a-z0-9-')" \
    >/dev/null 2>&1
}
rm_() { gcloud compute instances delete "$2" --project="$PROJECT" --zone="$1" --quiet >/dev/null 2>&1; }

for z in $ZONES; do
  say "trying $z"
  if ! mk "$z" "$NODE0" 0; then say "  $NODE0 unavailable in $z"; continue; fi
  say "  $NODE0 landed -- racing for $NODE1"
  if mk "$z" "$NODE1" 1; then
    say "PAIR UP in $z (SPOT): $NODE0 + $NODE1"
    say "  cells:  $ONLY_CELLS  (REPS=$REPS)"
    say "  DWS requests for $RUN_STAMP are deliberately LEFT QUEUED as the fallback."
    say "  watch:  gcloud compute instances list --filter='name~^spar-spot'"
    exit 0
  fi
  say "  $NODE1 failed -- releasing $NODE0 rather than leaving half a pair billing"
  rm_ "$z" "$NODE0"
done
say "no zone had capacity for a Spot pair"
exit 1
