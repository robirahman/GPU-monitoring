#!/usr/bin/env bash
# SUPERSEDED by provision_flexstart_mig.sh -- DO NOT USE. This script does not
# queue, despite what the comments below claim. MEASURED 2026-09-18: both
# `instances create --provisioning-model=FLEX_START` calls came back as
# PROVISIONING and then FAILED 2.5 minutes later with
#     code: ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS
#     reason: stockout   zonesAvailable: ''
# A plain FLEX_START instance create makes ONE immediate attempt. The durable DWS
# queue is a MIG resize request, which stays ACCEPTED with
# "Waiting for resources" and retries until capacity appears -- and is
# all-or-nothing for the count, so the half-a-pair bounding below is unnecessary
# there. Kept only for the startup-script pattern it established.
#
# Queue an H100 PAIR on Flex Start (DWS) and let it run the stage UNATTENDED.
#
# WHY FLEX START. Measured 2026-09-18 (docs/h100_capacity_2026-09-18.md): we now
# hold on-demand H100 quota in 9 regions, and a3-highgpu-8g STANDARD is stocked
# out in all 18 zones of them -- 18/18 ZONE_RESOURCE_POOL_EXHAUSTED, with
# `zonesAvailable: ''`. Spot has capacity in 15 zones, but these four cells are
# long-context 72B inference and Spot preemption is exactly what cost us the
# 2026-08-20 campaign three times in one night. Flex Start is the third door: it
# QUEUES for capacity instead of failing, then runs uninterrupted for a bounded
# window. `--max-run-duration` and `--instance-termination-action` are mandatory
# with it (verified: the API rejects the create without them), and it is
# accelerator-only (e2 is refused), so there is no cheap dry-run of the create.
#
# WHY THE NODE IS SELF-CONTAINED. Flex Start may grant capacity hours from now,
# long after this Cloud Shell is gone, so nothing may depend on the driving
# shell being alive. stage_repo.sh pushes the tree over scp from here, which
# cannot work for a VM that does not exist yet -- so the archive goes to GCS
# first and the node pulls it in its startup script. That archive is the SAME
# whitelist stage_repo.sh uses -- 14 MB gzipped at HEAD (stage_repo.sh's comment
# still says 0.6 MB; the whitelist has grown since), with LFS parquets as
# pointers, so the corpus itself never travels.
#
# WHY NOT `instances bulk create`. It would be genuinely all-or-nothing, but
# this SDK's bulk create offers only RESERVATION_BOUND/SPOT/STANDARD -- no
# FLEX_START (checked on 583.0.0, GA and beta). So the two VMs are queued
# SEPARATELY and the half-a-pair risk is real; it is bounded three ways: each
# node powers itself off if its peer never publishes a ready marker
# (PAIR_WAIT), --max-run-duration caps the blast radius absolutely, and
# --instance-termination-action=DELETE means nothing survives to bill quietly.
#
#   RUN_STAMP=20260918f bash scripts/gcp/provision_flexstart_pair.sh
#   ZONES="us-east4-a us-east5-a" RUN_STAMP=... bash .../provision_flexstart_pair.sh
#
# Cancel everything it queued:
#   gcloud compute operations list --project=$PROJECT --filter='status!=DONE AND targetLink~spar-gpu' \
#     --format='value(name,zone)'   # then `operations delete`, and delete any landed VM
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=env.sh
source scripts/gcp/env.sh

: "${RUN_STAMP:?export RUN_STAMP (identical on both nodes, e.g. 20260918f)}"
MACHINE_TYPE="${MACHINE_TYPE:-a3-highgpu-8g}"
# Zones we HOLD on-demand H100 quota in as of 2026-09-18. Flex Start draws on the
# same GPUS_PER_GPU_FAMILY quota as STANDARD, so a zone we have no quota in would
# be rejected outright rather than queued.
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-east4-a us-east4-b \
us-east4-c us-east5-a us-west1-a us-west1-b us-west4-a europe-west4-b europe-west4-c \
europe-west1-b europe-west1-c europe-west3-a europe-west3-c asia-southeast1-b asia-southeast1-c}"
# The four Section-25 long-context inference cells, and only those.
ONLY_CELLS="${ONLY_CELLS:-^s16_infershard_(qwen7b_kv16k_b32|qwen1_5b_kv16k_b48|qwen72b_kv16k_b24|qwen72b_kv4k_b48)}"
# Absolute cap on the run. Bootstrap (drivers, CUDA, venvs, vLLM) is ~50 min on a
# cold a3, the 72B weights are ~145 GB to pull, and four cells at 600 s plus model
# loads is ~2 h -- 6 h leaves headroom without letting a wedged node bill all day.
MAX_RUN="${MAX_RUN:-21600s}"
# How long a node that came up alone waits for its peer before giving up and
# powering off. One idle a3-highgpu-8g is ~$88/h, so this is a money knob.
PAIR_WAIT="${PAIR_WAIT:-5400}"
# Reps per cell. The corpus convention is 3, but the value of these four cells is
# DIVERSITY across cells, not depth within one -- section 25 showed reps of a
# single cell do not generalise even to its own siblings -- and each cell already
# yields 16 streams (8 GPUs x 2 nodes) per rep. 2 keeps 4 cells x 2 reps plus a
# ~1 h cold bootstrap inside MAX_RUN with margin; 3 would not.
REPS="${REPS:-2}"
STAGING="${BUCKET}/staging/${RUN_STAMP}"
# DRY_RUN=1 renders the startup script to $DRY_OUT and stops: no GCS upload, no
# create. Flex Start is accelerator-only, so the real create cannot be rehearsed
# on a cheap machine type -- rendering and shell-checking the guest script is the
# only rehearsal available, and it is where the bugs would be.
DRY_RUN="${DRY_RUN:-0}"
DRY_OUT="${DRY_OUT:-/tmp/spar_flexstart_startup.sh}"
CTRL="${BUCKET}/runs/${SESSION}/control"
say() { echo "[$(date -u '+%H:%M:%SZ')] $*"; }

# ── 1. Publish the tree the nodes will pull ──────────────────────────────────
read -r -a STAGE_PATHS <<< "${SPAR_STAGE_PATHS:-workloads scripts classifier requirements_v2.txt requirements_gcp_extra.txt requirements_tpu.txt requirements.txt}"
COMMIT="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git archive --format=tar HEAD -- "${STAGE_PATHS[@]}" | gzip -9 > "$TMP/spar_repo.tar.gz"
say "repo archive $(du -h "$TMP/spar_repo.tar.gz" | cut -f1) at ${COMMIT:0:12} -> $STAGING/"
[ "$DRY_RUN" = 1 ] || gcloud storage cp "$TMP/spar_repo.tar.gz" "$STAGING/spar_repo.tar.gz" --project="$PROJECT" >/dev/null
[ "$DRY_RUN" = 1 ] || printf '%s\n' "$COMMIT" | gcloud storage cp - "$STAGING/COMMIT" --project="$PROJECT" >/dev/null

# ── 2. The startup script the node runs on every boot ────────────────────────
# Idempotent by construction: bootstrap_node.sh is phase-markered and may reboot
# (exit 42), after which spar-bootstrap.service resumes it and THIS script runs
# again from the top. Everything below the bootstrap call therefore waits on
# /var/lib/spar/bootstrap.done rather than assuming it just produced it.
cat > "$TMP/startup.sh" <<STARTUP
#!/usr/bin/env bash
set -uo pipefail
exec >>/var/log/spar-startup.log 2>&1
echo "=== spar startup \$(date -u +%FT%TZ) boot ==="
REPO=/opt/spar/repo
STAGING="$STAGING"
CTRL="$CTRL"
STAMP="$RUN_STAMP"
ONLY='$ONLY_CELLS'
PAIR_WAIT=$PAIR_WAIT
REPS=$REPS
NODE="\$(hostname -s)"
PEER="\$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/spar-peer-name || true)"
IDX="\$(curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/attributes/spar-node-index || echo 0)"

# GCS without gcloud: the image may not have the CLI before bootstrap installs
# anything, but the metadata server always has a token and curl is always there.
gcs_get() {  # gcs_get <gs://bucket/object> <dest>
  local u="\${1#gs://}" bkt="\${1#gs://}" obj tok
  bkt="\${u%%/*}"; obj="\${u#*/}"
  tok="\$(curl -sf -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')" || return 1
  if ! curl -sf -H "Authorization: Bearer \$tok" -o "\$2" \
    "https://storage.googleapis.com/storage/v1/b/\$bkt/o/\$(python3 -c \
      'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "\$obj")?alt=media"; then
    # curl -f writes no body on an HTTP error but DOES create the -o file. Left
    # behind, that empty file makes the peer-rendezvous check below believe the
    # peer arrived, and a lone node would then run every 2-node cell against a
    # host that does not exist.
    rm -f "\$2"; return 1
  fi
}
gcs_put() {  # gcs_put <local> <gs://bucket/object>
  local u="\${2#gs://}" bkt obj tok
  bkt="\${u%%/*}"; obj="\${u#*/}"
  tok="\$(curl -sf -H 'Metadata-Flavor: Google' \
    http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
    | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])')" || return 1
  curl -sf -X POST -H "Authorization: Bearer \$tok" -H 'Content-Type: text/plain' \
    --data-binary "@\$1" \
    "https://storage.googleapis.com/upload/storage/v1/b/\$bkt/o?uploadType=media&name=\$(python3 -c \
      'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "\$obj")" >/dev/null
}

# cloud-init holds the apt lock on first boot; racing it is what broke the
# startup-script driver install in the provisioning session (see
# provision_gpu_pair.sh). Wait it out rather than fighting it.
cloud-init status --wait >/dev/null 2>&1 || true

if [ ! -f "\$REPO/.spar_staged_commit" ]; then
  install -d "\$REPO"
  for i in 1 2 3 4 5; do
    gcs_get "\$STAGING/spar_repo.tar.gz" /tmp/spar_repo.tar.gz && break
    echo "repo pull attempt \$i failed"; sleep 20
  done
  tar -xzf /tmp/spar_repo.tar.gz -C "\$REPO" && rm -f /tmp/spar_repo.tar.gz
  gcs_get "\$STAGING/COMMIT" "\$REPO/.spar_staged_commit" || true
  echo "staged \$(ls "\$REPO" | wc -l) entries at \$(cat "\$REPO/.spar_staged_commit" 2>/dev/null)"
fi

if [ ! -f /var/lib/spar/bootstrap.done ]; then
  echo "running bootstrap (node index \$IDX)"
  NODE_INDEX="\$IDX" SPAR_AUTO_REBOOT=1 bash "\$REPO/scripts/gcp/bootstrap_node.sh"
  rc=\$?
  echo "bootstrap rc=\$rc"
  # 42 = reboot required; spar-bootstrap.service resumes it and this script
  # re-runs on the next boot. Anything else non-zero is a hard failure, and a
  # hard failure must not leave 8 H100s idling until MAX_RUN.
  if [ "\$rc" -ne 0 ] && [ "\$rc" -ne 42 ]; then
    echo "bootstrap FAILED rc=\$rc -- powering off rather than idling"
    printf 'bootstrap-failed rc=%s %s\n' "\$rc" "\$(date -u +%FT%TZ)" > /tmp/m
    gcs_put /tmp/m "\$CTRL/\$NODE/startup_failed_\$STAMP.txt" || true
    shutdown -h +1; exit 1
  fi
  [ "\$rc" -eq 42 ] && exit 0
fi

for _ in \$(seq 1 120); do
  [ -f /var/lib/spar/bootstrap.done ] && break
  sleep 30
done
[ -f /var/lib/spar/bootstrap.done ] || { echo "bootstrap never completed"; shutdown -h +1; exit 1; }

# ── pair rendezvous ─────────────────────────────────────────────────────────
# Flex Start grants the two VMs INDEPENDENTLY, so one can be up for a long time
# before (or without) the other. Publish readiness, wait a bounded time, and
# power off alone rather than burn the whole MAX_RUN waiting for a peer that a
# capacity grant never produced.
printf 'ready %s\n' "\$(date -u +%FT%TZ)" > /tmp/ready
gcs_put /tmp/ready "\$CTRL/\$NODE/ready_\$STAMP.txt" || true
echo "published ready; waiting up to \$PAIR_WAIT s for \$PEER"
waited=0; got_peer=0
while [ "\$waited" -lt "\$PAIR_WAIT" ]; do
  if gcs_get "\$CTRL/\$PEER/ready_\$STAMP.txt" /tmp/peer_ready; then got_peer=1; break; fi
  sleep 30; waited=\$((waited + 30))
done
if [ "\$got_peer" -ne 1 ]; then
  echo "peer \$PEER never appeared in \$PAIR_WAIT s -- powering off"
  shutdown -h +1; exit 0
fi
echo "peer \$PEER is ready after \${waited}s"

# run_stage_and_stop.sh self-detaches, runs the stage, syncs telemetry to GCS,
# waits for the peer's done-marker and powers the node off.
STAMP="\$STAMP" PEER="\$PEER" ONLY="\$ONLY" REPS="\$REPS" \
  bash "\$REPO/scripts/gcp/run_stage_and_stop.sh"
echo "=== stage armed \$(date -u +%FT%TZ) ==="
STARTUP
chmod +x "$TMP/startup.sh"
say "startup script $(wc -c < "$TMP/startup.sh") bytes; cells: $ONLY_CELLS"
if [ "$DRY_RUN" = 1 ]; then
  cp "$TMP/startup.sh" "$DRY_OUT"
  bash -n "$DRY_OUT" && say "DRY RUN: rendered + parsed -> $DRY_OUT (nothing uploaded, nothing queued)"
  exit $?
fi

# ── 3. Queue the pair ────────────────────────────────────────────────────────
submit() {  # submit <zone> <name> <idx> <peer>
  gcloud compute instances create "$2" --project="$PROJECT" --zone="$1" \
    --machine-type="$MACHINE_TYPE" \
    --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
    --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type="$BOOT_DISK_TYPE" \
    --network="$NETWORK" \
    --provisioning-model=FLEX_START \
    --max-run-duration="$MAX_RUN" \
    --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --metadata-from-file=startup-script="$TMP/startup.sh" \
    --metadata="spar-node-index=$3,spar-node-name=$2,spar-peer-name=$4,spar-session=${SESSION},spar-master-addr=${NODE0},spar-master-port=${MASTER_PORT:-29500},spar-bucket=${BUCKET}" \
    --labels="spar-role=gpu,spar-session=${SESSION},spar-provisioning=flexstart" \
    --async 2>&1
}

existing="$(gcloud compute instances list --project="$PROJECT" \
  --filter="name=($NODE0 OR $NODE1)" --format='value(name,zone,status)' 2>/dev/null)"
if [ -n "$existing" ]; then
  say "REFUSING: $NODE0/$NODE1 already exist -- delete or rename first:"
  printf '  %s\n' "$existing"
  exit 1
fi

for z in $ZONES; do
  say "queueing the pair in $z"
  o0="$(submit "$z" "$NODE0" 0 "$NODE1")"; r0=$?
  if [ "$r0" -ne 0 ]; then say "  $NODE0 rejected: $(printf '%s' "$o0" | tail -1)"; continue; fi
  o1="$(submit "$z" "$NODE1" 1 "$NODE0")"; r1=$?
  if [ "$r1" -ne 0 ]; then
    say "  $NODE1 rejected: $(printf '%s' "$o1" | tail -1)"
    say "  releasing the queued $NODE0 rather than leaving half a pair"
    gcloud compute instances delete "$NODE0" --project="$PROJECT" --zone="$z" --quiet >/dev/null 2>&1
    continue
  fi
  say "QUEUED in $z: $NODE0 + $NODE1 (FLEX_START, max-run $MAX_RUN, delete on expiry)"
  say "  watch:  gcloud compute instances list --project=$PROJECT --filter='labels.spar-provisioning=flexstart'"
  say "  log:    gcloud compute ssh $NODE0 --zone=$z --command='sudo tail -f /var/log/spar-startup.log'"
  say "  cells:  $ONLY_CELLS"
  say "  data:   $BUCKET/runs/$SESSION/"
  exit 0
done
say "no zone accepted the pair -- nothing is queued and nothing is billing"
exit 1
