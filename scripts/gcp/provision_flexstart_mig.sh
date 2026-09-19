#!/usr/bin/env bash
# Queue an H100 pair on DWS Flex Start VIA A MIG RESIZE REQUEST, and run the
# stage unattended when capacity is granted.
#
# WHY THIS REPLACES provision_flexstart_pair.sh. That script used
# `instances create --provisioning-model=FLEX_START` and I described it as
# queued. It was not. MEASURED 2026-09-18: both inserts failed 2.5 minutes after
# submission with
#     code: ZONE_RESOURCE_POOL_EXHAUSTED_WITH_DETAILS
#     reason: stockout   zonesAvailable: ''
# A plain FLEX_START instance create makes ONE immediate attempt. The durable
# DWS queue is a managed-instance-group RESIZE REQUEST, which waits for capacity
# (up to --requested-run-duration) and is genuinely all-or-nothing for the count
# -- so it also removes the half-a-pair problem the old script had to bound with
# a peer-wait timeout.
#
# The TEMPLATE must carry --max-run-duration as well: the API refuses a
# FLEX_START template without one ("Max Run Duration or Termination Time must be
# specified"), even though the resize request also takes --requested-run-duration.
#
# SHAPE: instance template (FLEX_START) -> zonal MIG at size 0 -> resize request
# naming both instances. The MIG is created with --default-action-on-vm-failure=
# do-nothing so that a node which finishes its stage and powers itself off is NOT
# recreated and re-billed; the node also best-effort deletes itself from the MIG.
#
# PER-INSTANCE IDENTITY: a template gives both VMs identical metadata, so node
# index and peer name are derived from the HOSTNAME instead. The resize request
# pins the names (--instances), which is what makes that safe.
#
#   RUN_STAMP=20260919a bash scripts/gcp/provision_flexstart_mig.sh
#   DRY_RUN=1 RUN_STAMP=x bash scripts/gcp/provision_flexstart_mig.sh   # render only
#
# Watch / cancel:
#   gcloud compute instance-groups managed resize-requests list SPAR_MIG --zone=Z
#   gcloud compute instance-groups managed resize-requests cancel REQ --resize-request=REQ ...
#   bash scripts/gcp/provision_flexstart_mig.sh --teardown   # removes request, MIG, template
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
# shellcheck source=env.sh
source scripts/gcp/env.sh

RUN_STAMP="${RUN_STAMP:-$(date -u +%Y%m%d%H%M)}"
ZONE="${ZONE:-us-central1-a}"
MACHINE_TYPE="${MACHINE_TYPE:-a3-highgpu-8g}"
# NOT ${NODE0:-...}: env.sh exports NODE0/NODE1 as spar-gpu-0/1, which are the
# EXISTING stopped Spot pair. Inheriting them made the first resize request fail
# with RESOURCE_ALREADY_EXISTS -> VM_MIN_COUNT_NOT_REACHED, because a resize
# request creates instances by name and those names are taken. Use a
# FLEX-specific override so the pair cannot collide with anything already there.
NODE0="${FLEX_NODE0:-spar-gpuf-0}"
NODE1="${FLEX_NODE1:-spar-gpuf-1}"
TPL="${TPL:-spar-flex-tpl-${RUN_STAMP}}"
MIG="${MIG:-spar-flex-mig-${RUN_STAMP}}"
RREQ="${RREQ:-spar-flex-req-${RUN_STAMP}}"
# How long DWS may keep the VMs once it grants them. run_stage_and_stop.sh caps
# its own wait at 8 h and then syncs and powers off, so this is the outer bound,
# not the expected runtime.
RUN_DURATION="${RUN_DURATION:-12h}"
REPS="${REPS:-2}"
# Everything new from section 33, plus the four long-context cells that have been
# queued and unrun since 2026-09-17.
ONLY_CELLS="${ONLY_CELLS:-^s16_(infershard_(qwen7b_kv16k_b32|qwen1_5b_kv16k_b48|qwen72b_kv16k_b24|qwen72b_kv4k_b48)|inferserve_|infermem_|trainmem_|train_accum)}"
STAGING="${BUCKET}/staging/${RUN_STAMP}"
CTRL="${BUCKET}/runs/${SESSION}/control"
DRY_RUN="${DRY_RUN:-0}"
DRY_OUT="${DRY_OUT:-/tmp/spar_flexmig_startup.sh}"
say() { echo "[$(date -u '+%H:%M:%SZ')] $*"; }

if [ "${1:-}" = "--teardown" ]; then
  say "tearing down ${RREQ} / ${MIG} / ${TPL} in ${ZONE}"
  gcloud compute instance-groups managed resize-requests cancel "$MIG" \
    --resize-request="$RREQ" --project="$PROJECT" --zone="$ZONE" --quiet 2>&1 | tail -1
  gcloud compute instance-groups managed delete "$MIG" \
    --project="$PROJECT" --zone="$ZONE" --quiet 2>&1 | tail -1
  gcloud compute instance-templates delete "$TPL" --project="$PROJECT" --quiet 2>&1 | tail -1
  exit 0
fi

# A resize request creates instances BY NAME, so a name already in use fails the
# whole request -- and it fails at GRANT time, potentially hours later.
existing="$(gcloud compute instances list --project="$PROJECT" \
  --filter="name=($NODE0 OR $NODE1)" --format='value(name,zone,status)' 2>/dev/null)"
if [ -n "$existing" ]; then
  say "REFUSING: $NODE0/$NODE1 already exist -- delete, rename, or set FLEX_NODE0/FLEX_NODE1:"
  printf '  %s\n' "$existing"
  exit 1
fi

# ── 1. Publish the tree the nodes pull (a grant may land hours from now) ──────
read -r -a STAGE_PATHS <<< "${SPAR_STAGE_PATHS:-workloads scripts classifier requirements_v2.txt requirements_gcp_extra.txt requirements_tpu.txt requirements.txt}"
COMMIT="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"; trap 'rm -rf "$TMP"' EXIT
git archive --format=tar HEAD -- "${STAGE_PATHS[@]}" | gzip -9 > "$TMP/spar_repo.tar.gz"
say "repo archive $(du -h "$TMP/spar_repo.tar.gz" | cut -f1) at ${COMMIT:0:12} -> $STAGING/"
if [ "$DRY_RUN" != 1 ]; then
  gcloud storage cp "$TMP/spar_repo.tar.gz" "$STAGING/spar_repo.tar.gz" --project="$PROJECT" >/dev/null
  printf '%s\n' "$COMMIT" | gcloud storage cp - "$STAGING/COMMIT" --project="$PROJECT" >/dev/null
fi

# ── 2. Startup script (runs on every boot; bootstrap may reboot once) ─────────
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
REPS=$REPS
PROJECT="$PROJECT"
ZONE="$ZONE"
MIG="$MIG"
NODE0="$NODE0"
NODE1="$NODE1"
NODE="\$(hostname -s)"
# A template hands both VMs the SAME metadata, so identity comes from the name.
# The resize request pins the names, which is what makes this reliable.
if [ "\$NODE" = "\$NODE0" ]; then IDX=0; PEER="\$NODE1"; else IDX=1; PEER="\$NODE0"; fi
echo "identity: node=\$NODE idx=\$IDX peer=\$PEER"

tok() { curl -sf -H 'Metadata-Flavor: Google' \
  http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token \
  | python3 -c 'import sys,json;print(json.load(sys.stdin)["access_token"])'; }
enc() { python3 -c 'import urllib.parse,sys;print(urllib.parse.quote(sys.argv[1],safe=""))' "\$1"; }
gcs_get() {
  local u="\${1#gs://}" bkt obj t
  bkt="\${u%%/*}"; obj="\${u#*/}"; t="\$(tok)" || return 1
  if ! curl -sf -H "Authorization: Bearer \$t" -o "\$2" \
    "https://storage.googleapis.com/storage/v1/b/\$bkt/o/\$(enc "\$obj")?alt=media"; then
    # curl -f writes no body on an HTTP error but DOES create the -o file, and a
    # leftover empty file would make a wait-for-marker loop think it arrived.
    rm -f "\$2"; return 1
  fi
}
gcs_put() {
  local u="\${2#gs://}" bkt obj t
  bkt="\${u%%/*}"; obj="\${u#*/}"; t="\$(tok)" || return 1
  curl -sf -X POST -H "Authorization: Bearer \$t" -H 'Content-Type: text/plain' \
    --data-binary "@\$1" \
    "https://storage.googleapis.com/upload/storage/v1/b/\$bkt/o?uploadType=media&name=\$(enc "\$obj")" >/dev/null
}
mig_self_delete() {
  local t; t="\$(tok)" || return 1
  curl -sf -X POST -H "Authorization: Bearer \$t" -H 'Content-Type: application/json' \
    -d "{\"instances\":[\"zones/\$ZONE/instances/\$NODE\"]}" \
    "https://compute.googleapis.com/compute/v1/projects/\$PROJECT/zones/\$ZONE/instanceGroupManagers/\$MIG/deleteInstances" >/dev/null
}

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
  if [ "\$rc" -ne 0 ] && [ "\$rc" -ne 42 ]; then
    echo "bootstrap FAILED rc=\$rc -- removing this node rather than idling 8 H100s"
    printf 'bootstrap-failed rc=%s %s\n' "\$rc" "\$(date -u +%FT%TZ)" > /tmp/m
    gcs_put /tmp/m "\$CTRL/\$NODE/startup_failed_\$STAMP.txt" || true
    mig_self_delete || shutdown -h +1
    exit 1
  fi
  [ "\$rc" -eq 42 ] && exit 0            # rebooting; this script re-runs on boot
fi

for _ in \$(seq 1 120); do
  [ -f /var/lib/spar/bootstrap.done ] && break
  sleep 30
done
if [ ! -f /var/lib/spar/bootstrap.done ]; then
  echo "bootstrap never completed"; mig_self_delete || shutdown -h +1; exit 1
fi

printf 'ready %s\n' "\$(date -u +%FT%TZ)" > /tmp/ready
gcs_put /tmp/ready "\$CTRL/\$NODE/ready_\$STAMP.txt" || true

# The resize request is all-or-nothing, so the peer exists by construction --
# but it may still be mid-bootstrap. Wait for its marker, bounded.
echo "waiting for peer \$PEER"
got=0
for _ in \$(seq 1 240); do            # <= 2 h, a cold a3 bootstrap is ~50 min
  if gcs_get "\$CTRL/\$PEER/ready_\$STAMP.txt" /tmp/peer_ready; then got=1; break; fi
  sleep 30
done
if [ "\$got" -ne 1 ]; then
  echo "peer \$PEER never became ready -- removing this node"
  mig_self_delete || shutdown -h +1; exit 0
fi
echo "peer ready; arming the stage"

STAMP="\$STAMP" PEER="\$PEER" ONLY="\$ONLY" REPS="\$REPS" \
  bash "\$REPO/scripts/gcp/run_stage_and_stop.sh"

# run_stage_and_stop self-detaches, runs the queue, syncs to GCS, rendezvouses
# with the peer and calls shutdown. Deleting the VM from the MIG is strictly
# better than leaving it TERMINATED (no disk left billing, no chance of a repair
# recreating it), so watch for both done-markers and do that first.
(
  MARK="stage_\${STAMP}_complete.txt"
  for _ in \$(seq 1 600); do          # <= 10 h
    if gcs_get "\$CTRL/\$NODE/\$MARK" /tmp/self_done \
       && gcs_get "\$CTRL/\$PEER/\$MARK" /tmp/peer_done; then
      echo "both stages complete -- deleting \$NODE from \$MIG"
      sleep 120                        # let the final sync finish
      mig_self_delete || true
      exit 0
    fi
    sleep 60
  done
) &
echo "=== stage armed \$(date -u +%FT%TZ) ==="
STARTUP
chmod +x "$TMP/startup.sh"
say "startup script $(wc -c < "$TMP/startup.sh") bytes"
if [ "$DRY_RUN" = 1 ]; then
  cp "$TMP/startup.sh" "$DRY_OUT"
  bash -n "$DRY_OUT" && say "DRY RUN: rendered + parsed -> $DRY_OUT (nothing created)"
  exit $?
fi

# ── 3. Template -> MIG -> resize request ─────────────────────────────────────
say "creating template $TPL"
gcloud compute instance-templates create "$TPL" --project="$PROJECT" \
  --machine-type="$MACHINE_TYPE" \
  --image-family=ubuntu-2204-lts --image-project=ubuntu-os-cloud \
  --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type="$BOOT_DISK_TYPE" \
  --network="$NETWORK" \
  --provisioning-model=FLEX_START \
  --instance-termination-action=DELETE \
  --max-run-duration="$RUN_DURATION" \
  --maintenance-policy=TERMINATE \
  --scopes=cloud-platform \
  --metadata-from-file=startup-script="$TMP/startup.sh" \
  --metadata="spar-session=${SESSION},spar-master-addr=${NODE0},spar-master-port=${MASTER_PORT:-29500},spar-bucket=${BUCKET}" \
  --labels="spar-role=gpu,spar-session=${SESSION},spar-provisioning=flexmig" 2>&1 | tail -2 || exit 1

say "creating MIG $MIG (size 0, repair disabled)"
gcloud compute instance-groups managed create "$MIG" --project="$PROJECT" --zone="$ZONE" \
  --template="$TPL" --size=0 --default-action-on-vm-failure=do-nothing 2>&1 | tail -2 || exit 1

say "creating resize request $RREQ for $NODE0 + $NODE1 (run duration $RUN_DURATION)"
gcloud compute instance-groups managed resize-requests create "$MIG" --project="$PROJECT" \
  --zone="$ZONE" --resize-request="$RREQ" --instances="${NODE0},${NODE1}" \
  --requested-run-duration="$RUN_DURATION" 2>&1 | tail -3 || exit 1

say "QUEUED. DWS will provision BOTH nodes together when capacity appears."
say "  state:  gcloud compute instance-groups managed resize-requests describe $MIG \\"
say "            --resize-request=$RREQ --project=$PROJECT --zone=$ZONE"
say "  nodes:  gcloud compute instances list --project=$PROJECT --filter='name~^spar-gpuf'"
say "  log:    gcloud compute ssh $NODE0 --zone=$ZONE --command='sudo tail -f /var/log/spar-startup.log'"
say "  cells:  $ONLY_CELLS  (REPS=$REPS)"
say "  data:   $BUCKET/runs/$SESSION/"
say "  cancel: RUN_STAMP=$RUN_STAMP bash scripts/gcp/provision_flexstart_mig.sh --teardown"
