#!/usr/bin/env bash
# provision_gpu_pair.sh -- create the 2x a3-highgpu-8g (16x H100) Spot pair, the
# telemetry bucket, and the internal firewall rule. Also documents the DWS Flex
# Start variant (for >2 h uninterrupted blocks) and the TPU v6e-16 create.
#
# Run from YOUR machine (needs gcloud + an authenticated account that is Owner on
# the project). This ONLY provisions; drivers/DCGM/venvs are bootstrap_node.sh's
# job, run afterward over SSH.
#
#   bash scripts/gcp/provision_gpu_pair.sh            # create the GPU pair (+bucket +firewall)
#   bash scripts/gcp/provision_gpu_pair.sh --tpu      # ALSO create the v6e-16 TPU
#   bash scripts/gcp/provision_gpu_pair.sh --tpu-only # just the TPU
#
# WHY no --metadata startup-script for driver install: in the provisioning session
# a startup-script raced cloud-init's apt lock and the driver install failed. So we
# create BARE instances and let bootstrap_node.sh (which first does
# `cloud-init status --wait`) install drivers over SSH. See that script.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

require_cmd gcloud

DO_GPU=1; DO_TPU=0
case "${1:-}" in
  --tpu)      DO_TPU=1 ;;
  --tpu-only) DO_TPU=1; DO_GPU=0 ;;
  ""|--gpu)   : ;;
  *) echo "usage: $0 [--tpu|--tpu-only|--gpu]"; exit 2 ;;
esac

gcloud config set project "$PROJECT" >/dev/null 2>&1 || true

# ----------------------------------------------------------------------------
# One-time: telemetry bucket. Idempotent -- describe first, create only if absent.
# ----------------------------------------------------------------------------
ensure_bucket() {
  if gcloud storage buckets describe "$BUCKET" >/dev/null 2>&1; then
    spar_say "bucket $BUCKET already exists"
  else
    spar_say "creating bucket $BUCKET in $REGION"
    # Uniform bucket-level access; single-region next to the compute for cheap egress.
    gcloud storage buckets create "$BUCKET" \
      --project="$PROJECT" \
      --location="$REGION" \
      --uniform-bucket-level-access
  fi
}

# ----------------------------------------------------------------------------
# One-time: internal firewall. We REUSE default-allow-internal if present (it
# already permits all intra-network TCP/UDP/ICMP, covering :29500 + NCCL's dynamic
# ports). Create-if-missing only. Nothing is exposed publicly.
# ----------------------------------------------------------------------------
ensure_firewall() {
  if gcloud compute firewall-rules describe "$FW_INTERNAL" --project="$PROJECT" >/dev/null 2>&1; then
    spar_say "firewall $FW_INTERNAL already exists (covers :$MASTER_PORT + NCCL)"
  else
    spar_say "creating firewall $FW_INTERNAL (internal-only)"
    # 10.128.0.0/9 is the default auto-mode network's aggregate internal range.
    gcloud compute firewall-rules create "$FW_INTERNAL" \
      --project="$PROJECT" \
      --network="$NETWORK" \
      --direction=INGRESS \
      --action=ALLOW \
      --rules=tcp:0-65535,udp:0-65535,icmp \
      --source-ranges=10.128.0.0/9
  fi
}

# ----------------------------------------------------------------------------
# One GPU node.  $1 = node index (0/1)  $2 = instance name
# ----------------------------------------------------------------------------
create_node() {
  local idx="$1" name="$2"
  if gcloud compute instances describe "$name" --zone="$ZONE" --project="$PROJECT" >/dev/null 2>&1; then
    spar_say "instance $name already exists -- skipping create (use teardown.sh to remove)"
    return 0
  fi
  spar_say "creating $name (index $idx, $MACHINE_TYPE, $PROVISIONING_MODEL) in $ZONE"
  # Metadata carries only NON-SECRET identity so a self-heal reboot (bootstrap's
  # spar-bootstrap.service) can re-derive who this node is. The git PAT is NEVER here.
  gcloud compute instances create "$name" \
    --project="$PROJECT" \
    --zone="$ZONE" \
    --machine-type="$MACHINE_TYPE" \
    "${IMAGE_FLAGS[@]}" \
    --boot-disk-size="$BOOT_DISK_SIZE" \
    --boot-disk-type="$BOOT_DISK_TYPE" \
    --network="$NETWORK" \
    --provisioning-model="$PROVISIONING_MODEL" \
    ${PROVISIONING_MODEL:+$([ "$PROVISIONING_MODEL" = SPOT ] && echo --instance-termination-action=STOP)} \
    --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --metadata="spar-node-index=${idx},spar-node-name=${name},spar-session=${SESSION},spar-master-addr=${MASTER_ADDR},spar-master-port=${MASTER_PORT},spar-bucket=${BUCKET}" \
    --labels="spar-role=gpu,spar-session=${SESSION}"
  # WHY these flags:
  #  --provisioning-model=SPOT            : the only H100 quota that exists here is Spot (=64).
  #  --instance-termination-action=STOP   : on preemption STOP (don't DELETE) so the boot disk --
  #                                         repo, venvs, /var/lib/spar phase markers, cached telemetry --
  #                                         survives; re-provision = `gcloud compute instances start`.
  #  --maintenance-policy=TERMINATE       : GPU VMs cannot live-migrate; required for accelerator types.
  #  --scopes=cloud-platform              : lets the node's default SA rsync telemetry to $BUCKET and
  #                                         read the metadata server. (Default compute SA needs write on
  #                                         the bucket -- it does as project Editor; verify if you locked IAM down.)
  #  a3-highgpu-8g bundles its 8 H100s    : so there is deliberately NO --accelerator flag.
}

if [[ "$DO_GPU" == "1" ]]; then
  ensure_bucket
  ensure_firewall
  create_node 0 "$NODE0"
  create_node 1 "$NODE1"
  spar_say "GPU pair requested. Poll readiness with:  gcloud compute instances list --filter=\"name~spar-gpu\""
  cat <<EOF

Next:
  1) Copy the runbook scripts to each node and bootstrap (drivers+DCGM+venvs), e.g.:
       for n in $NODE0 $NODE1; do
         gcloud compute scp --zone=$ZONE "${SCRIPT_DIR}"/*.sh "\$n:/tmp/" ;
       done
       gcloud compute ssh $NODE0 --zone=$ZONE --command="sudo NODE_INDEX=0 SPAR_AUTO_REBOOT=1 bash /tmp/bootstrap_node.sh"
       gcloud compute ssh $NODE1 --zone=$ZONE --command="sudo NODE_INDEX=1 SPAR_AUTO_REBOOT=1 bash /tmp/bootstrap_node.sh"
     (SPAR_AUTO_REBOOT=1 lets the node reboot+resume itself on a kernel-module mismatch.)
  2) From node0:  bash /opt/spar/gcp/verify_pair.sh
  3) TEAR DOWN when done -- an idle pair burns ~\$78/h:  bash scripts/gcp/teardown.sh
EOF
fi

# ============================================================================
# TPU v6e-16 (us-east1-d, Spot). Enable with --tpu / --tpu-only.
# ============================================================================
create_tpu() {
  if gcloud compute tpus tpu-vm describe "$TPU_NAME" --zone="$TPU_ZONE" --project="$PROJECT" >/dev/null 2>&1; then
    spar_say "TPU $TPU_NAME already exists -- skipping"
    return 0
  fi
  spar_say "creating TPU $TPU_NAME ($TPU_ACCEL, Spot) in $TPU_ZONE"
  gcloud compute tpus tpu-vm create "$TPU_NAME" \
    --project="$PROJECT" \
    --zone="$TPU_ZONE" \
    --accelerator-type="$TPU_ACCEL" \
    --version="$TPU_RUNTIME" \
    --spot \
    --service-account="$TPU_SA"
  # NOTE: the tpu-service-account must exist with tpu.admin/storage.admin/logging/monitoring
  # (docs/cloud_tpu.md has the one-time create+role-binding loop). --spot uses the ample
  # preemptible v6e quota. Spot preempted a v6e-16 at ~35 min in the session -- for a long
  # uninterrupted block prefer the queued-resource variant below.
}
if [[ "$DO_TPU" == "1" ]]; then
  ensure_bucket
  create_tpu
fi

# ============================================================================
# ---- REFERENCE variants (commented; do not run inline) ---------------------
# ============================================================================
#
# DWS FLEX START (use for a >2 h block you don't want preempted mid-run). Flex Start
# queues the request and, once scheduled, runs preemption-protected up to
# --max-run-duration. CAVEAT: this flag surface has drifted across gcloud releases
# (FLEX_START used to be spelled via a resize-request / DWS queue); verify against
# `gcloud compute instances create --help | grep -A2 provisioning-model` on YOUR version.
#
#   gcloud compute instances create "$NODE0" \
#     --project="$PROJECT" --zone="$ZONE" --machine-type="$MACHINE_TYPE" \
#     "${IMAGE_FLAGS[@]}" --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type="$BOOT_DISK_TYPE" \
#     --network="$NETWORK" --maintenance-policy=TERMINATE --scopes=cloud-platform \
#     --provisioning-model=FLEX_START \
#     --instance-termination-action=DELETE \
#     --max-run-duration=7200s          # <= your block length; Flex Start bills for the reservation window
#
# TPU as a QUEUED RESOURCE (survives transient capacity gaps; --valid-until-duration
# bounds how long the request keeps trying so it can't provision-and-bill unwatched):
#
#   gcloud compute tpus queued-resources create "${TPU_NAME}-qr" \
#     --project="$PROJECT" --zone="$TPU_ZONE" \
#     --node-id="$TPU_NAME" --accelerator-type="$TPU_ACCEL" --runtime-version="$TPU_RUNTIME" \
#     --spot --valid-until-duration=2h
#
# TPU multislice stretch rung (2x v6e-8 => guaranteed DCN metrics) is created the same
# way with two node-ids under one queued resource; see the TPU track scripts.
