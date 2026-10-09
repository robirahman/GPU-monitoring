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
# Remember what the CALLER set before env.sh gets a say. env.sh defaults
# MACHINE_TYPE to a3-highgpu-8g, so a plain `MACHINE_TYPE="${MACHINE_TYPE:-a2-...}"`
# after sourcing it silently never applies -- which is how this script spent
# four zones trying to create H100s while reporting "no capacity" for A100s.
_MT_SET="${MACHINE_TYPE:-}"
source scripts/gcp/env.sh 2>/dev/null || true

PROJECT="${PROJECT:-focus-surfer-494820-g0}"
BUCKET="${BUCKET:-gs://${PROJECT}-telemetry}"
RUN_STAMP="${RUN_STAMP:?set RUN_STAMP}"
SESSION="${SESSION:-${RUN_STAMP%[a-z]}}"
NODE0="${SPOT_NODE0:-spar-spot-0}"
NODE1="${SPOT_NODE1:-spar-spot-1}"
MACHINE_TYPE="${_MT_SET:-a2-highgpu-8g}"
# Several machine types, tried in order within each zone. A100 and H100 draw on
# different capacity pools, and on 2026-10-01 A100 was exhausted in all 15 zones
# we hold quota in -- for both Spot and on-demand -- so trying only one type
# means sitting out a drought that the other type may not be in.
MACHINE_TYPES="${MACHINE_TYPES:-$MACHINE_TYPE}"
# The baked node image. Validated on a3-highgpu-8g (H100 80GB) 2026-10-03:
# bootstrap_node.sh ran COMPLETE in 44 s -- every phase marker, torch seeing 8
# devices, nvlink topology and DCGM nvlink OK, GCS writable, chrony at +4 ms --
# against the tens of minutes a cold node spends building two venvs and pulling
# 16 GB of NGC images. On Spot, where a pair can be reclaimed inside half an
# hour, that difference is most of the round.
#
# It carries apt, both venvs, docker + the NVIDIA container toolkit, both NGC
# images and CIFAR-10, with the apt/container/repo phase markers baked. The
# DRIVER is deliberately not baked: the base DL image installs it at boot and a
# baked one would pin a driver to whatever GPU the builder happened to have.
# The repo tree is overwritten from GCS by the startup script before bootstrap,
# so the baked copy is never the one that runs.
IMAGE_FAMILY="${IMAGE_FAMILY:-spar-node}"
IMAGE_PROJECT="${IMAGE_PROJECT:-$PROJECT}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-500GB}"
MASTER_PORT="${MASTER_PORT:-29500}"
REPS="${REPS:-2}"
ONLY_CELLS="${ONLY_CELLS:-^s16_}"
ZONES="${ZONES:-us-west1-b asia-southeast1-c us-central1-a europe-west4-b}"
MAX_RUN="${MAX_RUN:-43200s}"
# SPOT by default; STANDARD gives a non-preemptible pair. Three Spot pairs were
# reclaimed inside 25 minutes each on 2026-10-01, against a 5.1 h round, so the
# cheaper option can cost more wall clock than it saves in dollars.
PROVISIONING_MODEL="${PROVISIONING_MODEL:-SPOT}"
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

# Sweep this script's OWN leavings before adding more. Each run drops ~14 MB of
# tarball plus the create logs in a mktemp dir and deliberately never removes
# it, so a supervisor cycling for hours fills /tmp by itself -- which is exactly
# what happened on 2026-10-03: sixteen dirs, 1.8 GB, and the disk at 99%.
for _old in /tmp/tmp.*; do
  [ -d "$_old" ] && [ -e "$_old/startup.sh" ] && [ "$_old" != "${TMP:-}" ] \
    && rm -rf "$_old"
done
_free_kb="$(df -Pk /tmp | awk 'NR==2{print $4}')"
if [ "${_free_kb:-0}" -lt 102400 ]; then
  say "FATAL: only $((_free_kb/1024)) MB free on /tmp -- a short write here is"
  say "       silent, and an empty startup-script boots a pair that does nothing"
  exit 1
fi

TMP="$(mktemp -d)"   # holds the create logs; swept by the loop above next run
git archive --format=tar HEAD -- "${STAGE_PATHS[@]}" | gzip -9 > "$TMP/spar_repo.tar.gz"

# SIZE-CHECK BOTH ARTEFACTS, because neither failure announces itself. On
# 2026-10-03 /tmp filled, `git archive | gzip` wrote 0 bytes, the startup-script
# heredoc wrote 0 bytes, and gcloud attached the empty file as metadata without
# a murmur: `No startup scripts to run`, and an H100 pair booted the baked image
# and idled for 28 minutes at $0.65/min. `gcloud storage cp` of a 0-byte file
# SUCCEEDS, so the error check that was already here could never have caught it.
_arc_kb=$(( $(wc -c < "$TMP/spar_repo.tar.gz") / 1024 ))
if [ "$_arc_kb" -lt 1024 ]; then
  say "FATAL: repo archive is ${_arc_kb} KB -- expected megabytes. Refusing to"
  say "       stage it; a pair booted against a truncated archive fails late."
  exit 1
fi
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

# The startup script is the ONLY thing that makes a node do any work. An empty
# one costs a full pair's hourly rate for as long as nobody looks.
_su_b=$(wc -c < "$TMP/startup.sh")
if [ "$_su_b" -lt 512 ]; then
  say "FATAL: startup-script is ${_su_b} bytes -- the heredoc did not write."
  say "       Refusing to create a pair that would boot and sit idle."
  exit 1
fi

mk() {  # mk <zone> <name> <idx>
  gcloud compute instances create "$2" --project="$PROJECT" --zone="$1" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type=pd-balanced \
    --provisioning-model="$PROVISIONING_MODEL" --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --max-run-duration="$MAX_RUN" --scopes=cloud-platform \
    --metadata-from-file=startup-script="$TMP/startup.sh" \
    --metadata="spar-session=${SESSION},spar-master-addr=${NODE0},spar-master-port=${MASTER_PORT},spar-bucket=${BUCKET},spar-node0-name=${NODE0},spar-node1-name=${NODE1},spar-node-index=$3,spar-run-stamp=${RUN_STAMP}" \
    --labels="spar-role=gpu,spar-stamp=$(printf '%s' "$RUN_STAMP" | tr -cd 'a-z0-9-')" \
    2>&1 | tee "$TMP/create.$2.log" | grep -qE "^NAME|Created \["
}
why() {  # the reason the last create failed, not just that it did
  grep -viE "^$|global DNS|zonal-dns|^WARNING|Disk size|^NAME|resize_pd" \
    "$TMP/create.$1.log" 2>/dev/null | head -4 | tr '\n' ' '
}
rm_() { gcloud compute instances delete "$2" --project="$PROJECT" --zone="$1" --quiet >/dev/null 2>&1; }

# A nonzero gcloud exit does NOT mean the VM was not created. On 2026-10-01 the
# create of node0 in us-west1-b returned `gcloud crashed (ReadTimeout)` after
# 300 s -- and the instance existed and was RUNNING. The script read the nonzero
# status as "this zone has no capacity", moved on without ever trying node1, and
# left an 8xA100 billing while every later zone collided on the name it had
# supposedly failed to take.
#
# So: never infer absence from a failed call. Ask.
exists() { gcloud compute instances describe "$2" --project="$PROJECT" --zone="$1" \
             --format='value(name)' >/dev/null 2>&1; }

for pass in $(seq 1 "${PASSES:-1}"); do
for z in $ZONES; do
 for MACHINE_TYPE in $MACHINE_TYPES; do
  say "pass $pass: trying $z ($MACHINE_TYPE)"
  if ! mk "$z" "$NODE0" 0; then
    # Verify before believing it. A timeout is "unknown", not "absent".
    if exists "$z" "$NODE0"; then
      say "  $NODE0 create reported failure but the VM EXISTS in $z -- continuing with it"
    else
      say "  $NODE0 failed in $z: $(why "$NODE0")"; continue
    fi
  fi
  say "  $NODE0 landed in $z -- racing for $NODE1"
  if mk "$z" "$NODE1" 1 || exists "$z" "$NODE1"; then
    say "PAIR UP in $z ($PROVISIONING_MODEL $MACHINE_TYPE): $NODE0 + $NODE1"
    say "  cells:  $ONLY_CELLS  (REPS=$REPS)"
    say "  DWS requests for $RUN_STAMP are deliberately LEFT QUEUED as the fallback."
    say "  watch:  gcloud compute instances list --filter='name~^spar-spot'"
    exit 0
  fi
  say "  $NODE1 failed in $z: $(why "$NODE1")"
  say "  releasing $NODE0 rather than leaving half a pair billing"
  rm_ "$z" "$NODE0"
  # And CONFIRM the release, because the next zone's create will collide on the
  # name if this silently failed -- which is what turned one bad zone into five.
  for _ in $(seq 1 30); do exists "$z" "$NODE0" || break; sleep 10; done
  exists "$z" "$NODE0" && say "  WARNING: $NODE0 still present in $z after delete"
 done
done
[ "${PASSES:-1}" -gt 1 ] && [ "$pass" -lt "${PASSES:-1}" ] && \
  { say "pass $pass found nothing; sleeping ${SLEEP_BETWEEN:-180}s"; sleep "${SLEEP_BETWEEN:-180}"; }
done
say "no zone had capacity for a Spot pair after ${PASSES:-1} pass(es)"
exit 1
