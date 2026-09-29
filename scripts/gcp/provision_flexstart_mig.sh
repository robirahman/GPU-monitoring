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
# ONE request per zone, all racing for capacity. Whichever pair boots first
# claims a GCS lock and cancels every other request; a pair that boots into a
# zone that already lost deletes itself immediately. The lock write uses
# ifGenerationMatch=0, so exactly one zone can win even if two grant at the same
# instant, and it is keyed on the ZONE (not the node) so both nodes of the
# winning pair survive.
ZONES="${ZONES:-us-central1-a us-east4-a us-east5-a us-west1-a europe-west4-b asia-southeast1-b}"
# One zone per REGION. GPU quota is regional, not zonal: NVIDIA_A100_80GB_GPUS is
# 16 in us-central1 as a whole, and a pair is 2 x 8 = 16. So queuing
# us-central1-a AND us-central1-c asks that one pool for 32 and both requests
# fail with QUOTA_EXCEEDED -- a wider fan-out becomes LESS likely to land, which
# is the opposite of the point. That is what killed 20260926{u,v,w}: every list
# I passed had two or three us-central1 zones in it, and I spent hours blaming
# zones, then billing, before `gcloud compute instance-groups managed
# list-errors` named it outright.
_seen_regions=""
_deduped=""
for _z in $ZONES; do
  _r="${_z%-*}"                                  # us-central1-c -> us-central1
  case " $_seen_regions " in
    *" $_r "*) echo "[zones] dropping $_z: already racing $_r (GPU quota is regional)" >&2 ;;
    *) _seen_regions="$_seen_regions $_r"; _deduped="$_deduped $_z" ;;
  esac
done
ZONES="${_deduped# }"
ZONE="${ZONE:-${ZONES%% *}}"
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
# IMAGE: the Deep Learning VM, NOT pristine Ubuntu. MEASURED 2026-09-19 from the
# startup log the 20260919c pair shipped before deleting itself: on
# ubuntu-2204-lts the bootstrap's driver phase pulls cuda-drivers ->
# nvidia-driver-open 615.71.09, whose DKMS module does not build against this
# image's kernel --
#     Building module(s)...(bad exit status: 2)
#     Error! Bad return status for module build on kernel: 6.8.0-1067-gcp
#     Errors were encountered while processing: nvidia-dkms-open
#       nvidia-driver-open nvidia-open cuda-drivers
# -- so apt exits 1 and bootstrap dies ~70 minutes in. The DLVM image ships a
# matched driver (580) already built, and bootstrap_node.sh:148 skips phase (b)
# outright when 8 GPUs are already live, so the DKMS build never happens.
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
# Everything new from section 33, plus the four long-context cells that have been
# queued and unrun since 2026-09-17.
ONLY_CELLS="${ONLY_CELLS:-^s16_(infershard_(qwen7b_kv16k_b32|qwen1_5b_kv16k_b48|qwen72b_kv16k_b24|qwen72b_kv4k_b48)|inferserve_|infermem_|trainmem_|train_accum)}"
STAGING="${BUCKET}/staging/${RUN_STAMP}"
CTRL="${BUCKET}/runs/${SESSION}/control"
DRY_RUN="${DRY_RUN:-0}"
DRY_OUT="${DRY_OUT:-/tmp/spar_flexmig_startup.sh}"
say() { echo "[$(date -u '+%H:%M:%SZ')] $*"; }

if [ "${1:-}" = "--teardown" ]; then
  for z in $ZONES; do
    say "[$z] tearing down spar-flex-{req,mig,tpl}-${RUN_STAMP}-${z}"
    # --resize-requests, PLURAL and comma-separated. The singular form is not a
    # flag gcloud accepts, so with stderr discarded this cancelled nothing, ever,
    # and quietly: teardown deleted the MIG and template while leaving the resize
    # request queued. Those orphans keep drawing on the REGIONAL GPU quota, which
    # is the same pool a relaunch needs -- so a teardown-then-retry could be
    # starved by the very requests the teardown claimed to have removed.
    # Failures are reported now rather than swallowed.
    if ! out=$(gcloud compute instance-groups managed resize-requests cancel \
         "spar-flex-mig-${RUN_STAMP}-${z}" \
         --resize-requests="spar-flex-req-${RUN_STAMP}-${z}" \
         --project="$PROJECT" --zone="$z" --quiet 2>&1); then
      case "$out" in
        *"not found"*|*"was not found"*) : ;;   # already gone: fine
        *) say "[$z] WARN cancel failed: $(printf '%s' "$out" | head -1)" ;;
      esac
    fi
    gcloud compute instance-groups managed delete "spar-flex-mig-${RUN_STAMP}-${z}" \
      --project="$PROJECT" --zone="$z" --quiet >/dev/null 2>&1
    gcloud compute instance-templates delete "spar-flex-tpl-${RUN_STAMP}-${z}" \
      --project="$PROJECT" --quiet >/dev/null 2>&1
  done
  say "done"
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
ZONE="__ZONE__"
MIG="__MIG__"
RREQ="__RREQ__"
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
# Upload the startup log BEFORE the node disappears. The 20260919b pair failed
# bootstrap rc=1 within 3.5 minutes and self-deleted correctly -- and took the
# only copy of /var/log/spar-startup.log with it, so the cause was unknowable.
# Every exit path now ships the log first.
ship_log() {
  local tag="\${1:-exit}"
  {
    echo "=== nvidia-smi ==="; nvidia-smi 2>&1 | head -15
    for m in /var/lib/dkms/nvidia/*/build/make.log; do
      [ -f "\$m" ] && { echo "=== \$m ==="; tail -80 "\$m"; }
    done
    echo "=== journal ==="; journalctl -u google-startup-scripts --no-pager | tail -200
  } >> /var/log/spar-startup.log 2>&1 || true
  gcs_put /var/log/spar-startup.log "\$CTRL/\$NODE/startup_\${tag}_\$STAMP.log" || true
}
# Release the claim if this zone is giving up. Without this the round deadlocks:
# on 2026-09-20 a partial grant in us-east4-a claimed at 21:47:49Z, cancelled the
# other five zones, was rolled back seconds later, and left the lock held by a
# zone that could not provision. peer_up() stops a partial grant claiming at all;
# this makes the lock self-healing when a claimed zone fails for any other reason.
release_claim() {
  local t cur; t="\$(tok)" || return 1
  gcs_get "\$CTRL/claim_\$STAMP.txt" /tmp/claim_now || return 0
  cur="\$(cat /tmp/claim_now 2>/dev/null)"
  [ "\$cur" = "\$ZONE" ] || return 0        # not mine to release
  curl -sf -X DELETE -H "Authorization: Bearer \$t" \
    "https://storage.googleapis.com/storage/v1/b/\$(printf '%s' "\${CTRL#gs://}" | cut -d/ -f1)/o/\$(enc "\$(printf '%s' "\${CTRL#gs://}" | cut -d/ -f2-)/claim_\$STAMP.txt")" >/dev/null \
    && echo "released the claim for \$ZONE"
}
mig_self_delete() {
  local t; t="\$(tok)" || return 1
  curl -sf -X POST -H "Authorization: Bearer \$t" -H 'Content-Type: application/json' \
    -d "{\"instances\":[\"zones/\$ZONE/instances/\$NODE\"]}" \
    "https://compute.googleapis.com/compute/v1/projects/\$PROJECT/zones/\$ZONE/instanceGroupManagers/\$MIG/deleteInstances" >/dev/null
}

# Atomic create-if-absent: ifGenerationMatch=0 fails with 412 if the object
# already exists, so exactly one zone can take the claim even if two are granted
# in the same instant.
gcs_put_ifabsent() {
  local u="\${2#gs://}" bkt obj t
  bkt="\${u%%/*}"; obj="\${u#*/}"; t="\$(tok)" || return 1
  curl -sf -X POST -H "Authorization: Bearer \$t" -H 'Content-Type: text/plain' \
    --data-binary "@\$1" \
    "https://storage.googleapis.com/upload/storage/v1/b/\$bkt/o?uploadType=media&ifGenerationMatch=0&name=\$(enc "\$obj")" >/dev/null
}
cancel_request() {   # cancel_request <zone> <mig> <req>
  local t; t="\$(tok)" || return 1
  curl -sf -X POST -H "Authorization: Bearer \$t" -H 'Content-Length: 0' \
    "https://compute.googleapis.com/compute/v1/projects/\$PROJECT/zones/\$1/instanceGroupManagers/\$2/resizeRequests/\$3/cancel" >/dev/null
}

# Is my peer actually here? A resize request is all-or-nothing, but DWS still
# creates VMs opportunistically and ROLLS BACK a partial grant: observed
# 2026-09-20 in us-east4-a, where spar-gpuf-1 existed alone for ~30 s before
# being torn down while the request stayed ACCEPTED and retried. A solo VM that
# lived long enough to claim the lock would cancel the other five zones and then
# vanish -- collecting nothing and costing us the round. So the claim is gated on
# the peer existing first.
peer_up() {
  local t st; t="\$(tok)" || return 1
  st="\$(curl -sf -H "Authorization: Bearer \$t" \
    "https://compute.googleapis.com/compute/v1/projects/\$PROJECT/zones/\$ZONE/instances/\$PEER" \
    | python3 -c 'import sys,json
try: print(json.load(sys.stdin).get("status",""))
except Exception: print("")' 2>/dev/null)" || return 1
  [ "\$st" = "RUNNING" ] || [ "\$st" = "PROVISIONING" ] || [ "\$st" = "STAGING" ]
}

# Claim the race for THIS ZONE (not this node -- both nodes of the winning pair
# must survive, and the peer may claim first).
claim() {
  local i
  for i in \$(seq 1 20); do            # <= 10 min
    peer_up && break
    echo "peer \$PEER not up yet (attempt \$i)"; sleep 30
  done
  if ! peer_up; then
    echo "peer \$PEER never appeared -- NOT claiming (a partial grant must not"
    echo "  cancel the other zones); standing down"
    return 1
  fi
  printf '%s' "\$ZONE" > /tmp/claim
  if gcs_put_ifabsent /tmp/claim "\$CTRL/claim_\$STAMP.txt"; then
    echo "claimed the run for zone \$ZONE"; return 0
  fi
  gcs_get "\$CTRL/claim_\$STAMP.txt" /tmp/claim_read || return 1
  local won; won="\$(cat /tmp/claim_read)"
  if [ "\$won" = "\$ZONE" ]; then
    echo "zone \$ZONE already claimed by my peer -- continuing"; return 0
  fi
  echo "zone \$won won the race; \$ZONE stands down"; return 1
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

# Decide the race BEFORE the ~50 min bootstrap, so a losing zone costs minutes.
if ! claim; then
  gcs_put /tmp/claim_read "\$CTRL/\$NODE/stood_down_\$STAMP.txt" || true
  cancel_request "\$ZONE" "\$MIG" "\$RREQ" || true
  ship_log stooddown
  mig_self_delete || shutdown -h +1
  exit 0
fi

# Winner: cancel every other zone's request so nothing else can be granted.
if gcs_get "\$CTRL/manifest_\$STAMP.txt" /tmp/manifest; then
  while read -r z m r; do
    [ -z "\$z" ] && continue
    [ "\$z" = "\$ZONE" ] && continue
    echo "cancelling \$r in \$z"
    cancel_request "\$z" "\$m" "\$r" || echo "  cancel failed (may already be done)"
  done < /tmp/manifest
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
    ship_log bootstrapfail; release_claim
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
  echo "bootstrap never completed"; ship_log nobootstrap; release_claim
  mig_self_delete || shutdown -h +1; exit 1
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
  ship_log nopeer; release_claim; mig_self_delete || shutdown -h +1; exit 0
fi
echo "peer ready; arming the stage"

# HOME is NOT set in the GCE startup-script environment. multinode_queue.sh runs
# under `set -u` and referenced \$HOME, so on the unattended path it died with
# "HOME: unbound variable" before running a single cell -- which is exactly how
# the A100-80 pair sat idle for 3.6 h on 2026-09-21. The ssh path had HOME and
# never showed it.
HOME="\${HOME:-/root}" STAMP="\$STAMP" PEER="\$PEER" ONLY="\$ONLY" REPS="\$REPS" \
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

# ── 3. One template + MIG + resize request PER ZONE ──────────────────────────
# The manifest is published BEFORE any request, so the first pair to boot can
# read it and cancel the others even if this shell is long gone.
MANIFEST="$TMP/manifest.txt"
: > "$MANIFEST"
for z in $ZONES; do
  printf '%s %s %s\n' "$z" "spar-flex-mig-${RUN_STAMP}-${z}" "spar-flex-req-${RUN_STAMP}-${z}" >> "$MANIFEST"
done
gcloud storage cp "$MANIFEST" "${CTRL}/manifest_${RUN_STAMP}.txt" --project="$PROJECT" >/dev/null
say "manifest ($(wc -l < "$MANIFEST") zones) -> ${CTRL}/manifest_${RUN_STAMP}.txt"
# A stale claim from an earlier attempt would make every zone stand down.
gcloud storage rm "${CTRL}/claim_${RUN_STAMP}.txt" --project="$PROJECT" 2>/dev/null || true

queued=0
for z in $ZONES; do
  ztpl="spar-flex-tpl-${RUN_STAMP}-${z}"
  zmig="spar-flex-mig-${RUN_STAMP}-${z}"
  zreq="spar-flex-req-${RUN_STAMP}-${z}"
  sed -e "s|__ZONE__|${z}|g" -e "s|__MIG__|${zmig}|g" -e "s|__RREQ__|${zreq}|g" \
      "$TMP/startup.sh" > "$TMP/startup.${z}.sh"

  say "[$z] template $ztpl"
  gcloud compute instance-templates create "$ztpl" --project="$PROJECT" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type="$BOOT_DISK_TYPE" \
    --network="$NETWORK" \
    --provisioning-model=FLEX_START \
    --instance-termination-action=DELETE \
    --max-run-duration="$RUN_DURATION" \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --metadata-from-file=startup-script="$TMP/startup.${z}.sh" \
    --metadata="spar-session=${SESSION},spar-master-addr=${NODE0},spar-master-port=${MASTER_PORT:-29500},spar-bucket=${BUCKET},spar-node0-name=${NODE0},spar-node1-name=${NODE1}" \
    --labels="spar-role=gpu,spar-session=${SESSION},spar-provisioning=flexmig" >/dev/null 2>&1 \
    || { say "[$z] template create FAILED -- skipping zone"; continue; }

  gcloud compute instance-groups managed create "$zmig" --project="$PROJECT" --zone="$z" \
    --template="$ztpl" --size=0 --default-action-on-vm-failure=do-nothing >/dev/null 2>&1 \
    || { say "[$z] MIG create FAILED -- skipping zone"; continue; }

  if gcloud compute instance-groups managed resize-requests create "$zmig" --project="$PROJECT" \
       --zone="$z" --resize-request="$zreq" --instances="${NODE0},${NODE1}" \
       --requested-run-duration="$RUN_DURATION" >/dev/null 2>&1; then
    say "[$z] QUEUED $zreq"
    queued=$((queued + 1))
  else
    say "[$z] resize request FAILED -- skipping zone"
  fi
done

say "queued in $queued zone(s). The first pair to boot claims ${CTRL}/claim_${RUN_STAMP}.txt"
say "  and cancels the rest; a pair in a losing zone deletes itself before bootstrap."
say "  state:  for z in $ZONES; do gcloud compute instance-groups managed resize-requests describe \\"
say "            spar-flex-mig-${RUN_STAMP}-\$z --resize-request=spar-flex-req-${RUN_STAMP}-\$z \\"
say "            --project=$PROJECT --zone=\$z --format='value(state)'; done"
say "  nodes:  gcloud compute instances list --project=$PROJECT --filter='name~^spar-gpuf'"
say "  logs:   gcloud storage ls ${CTRL}/"
say "  cells:  $ONLY_CELLS  (REPS=$REPS)"
say "  cancel: RUN_STAMP=$RUN_STAMP ZONES='$ZONES' bash scripts/gcp/provision_flexstart_mig.sh --teardown"
