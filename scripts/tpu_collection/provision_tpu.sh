#!/bin/bash
# Provision / tear down the TPU slice for the comm-scope ladder.
#
#   create   -- request a Spot v6e-16 via a queued resource (preemptible,
#               auto-expires with --valid-until-duration so a forgotten slice
#               cannot bill forever)
#   status   -- show the queued-resource + node state
#   ssh      -- convenience ssh to worker 0 (or --worker=all via CMD=)
#   delete   -- delete the queued resource + node (idempotent)
#   retry    -- delete, wait, re-create (preemption recovery)
#   verify-zero -- assert nothing is left running (kill-switch hygiene)
#
# Spot preemption is REAL (a v6e-16 died at ~35 min in testing), so every knob
# here assumes the slice can vanish: short valid-until, queued-resource model,
# and a delete that never errors on an already-gone resource.
#
# Usage:
#   bash scripts/tpu_collection/provision_tpu.sh create
#   ACCEL=v6e-8 TPU_NAME=spar-tpu-8 bash scripts/tpu_collection/provision_tpu.sh create
#   bash scripts/tpu_collection/provision_tpu.sh delete
set -uo pipefail

# ── Configuration (override via env) ─────────────────────────────────────────
PROJECT=${PROJECT:-focus-surfer-494820-g0}
ZONE=${ZONE:-us-east1-d}                 # v6e-16 verified live here (plan section on GCP)
ACCEL=${ACCEL:-v6e-16}                    # v6e-16 = 4 hosts x 4 chips
RUNTIME=${RUNTIME:-v2-alpha-tpuv6e}
TPU_NAME=${TPU_NAME:-spar-tpu-16}
QR_NAME=${QR_NAME:-${TPU_NAME}-qr}        # queued-resource id
SA=${SA:-tpu-service-account@focus-surfer-494820-g0.iam.gserviceaccount.com}
VALID_UNTIL=${VALID_UNTIL:-4h}            # queued resource self-expires after this
SPOT=${SPOT:-1}                           # 1 = Spot (preemptible); 0 = on-demand

ACTION=${1:-status}

gc() { gcloud "$@" --project="$PROJECT" --zone="$ZONE"; }

create() {
  echo "Creating queued resource $QR_NAME -> $TPU_NAME ($ACCEL, $ZONE, spot=$SPOT)"
  local spot_flag=()
  [ "$SPOT" = "1" ] && spot_flag=(--spot)
  gcloud compute tpus queued-resources create "$QR_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    --node-id="$TPU_NAME" \
    --accelerator-type="$ACCEL" \
    --runtime-version="$RUNTIME" \
    --valid-until-duration="$VALID_UNTIL" \
    --service-account="$SA" \
    "${spot_flag[@]}"
  echo "Submitted. Poll with: bash $0 status"
}

status() {
  echo "== queued resource =="
  gc compute tpus queued-resources describe "$QR_NAME" \
    --format="value(state.state)" 2>/dev/null || echo "(none)"
  echo "== node =="
  gc compute tpus tpu-vm describe "$TPU_NAME" \
    --format="value(state)" 2>/dev/null || echo "(none)"
}

ssh_workers() {
  # CMD='...' bash provision_tpu.sh ssh   -> run on all workers
  local worker=${WORKER:-0}
  if [ -n "${CMD:-}" ]; then
    gc compute tpus tpu-vm ssh "$TPU_NAME" --worker="${WORKER:-all}" --command="$CMD"
  else
    gc compute tpus tpu-vm ssh "$TPU_NAME" --worker="$worker"
  fi
}

delete() {
  echo "Deleting node + queued resource (idempotent)..."
  gc compute tpus tpu-vm delete "$TPU_NAME" --quiet 2>/dev/null || true
  # --force also reaps the underlying node if the QR still owns it.
  gc compute tpus queued-resources delete "$QR_NAME" --quiet --force 2>/dev/null || true
  echo "Delete requested."
}

verify_zero() {
  echo "== remaining TPU nodes in $ZONE =="
  gc compute tpus tpu-vm list --format="value(name,state)" 2>/dev/null || true
  echo "== remaining queued resources in $ZONE =="
  gc compute tpus queued-resources list --format="value(name,state.state)" 2>/dev/null || true
  echo "If anything is listed above, it is still billing -- delete it."
}

case "$ACTION" in
  create)      create ;;
  status)      status ;;
  ssh)         ssh_workers ;;
  delete)      delete ;;
  retry)       delete; echo "waiting 30s before re-create..."; sleep 30; create ;;
  verify-zero) verify_zero ;;
  *) echo "usage: $0 {create|status|ssh|delete|retry|verify-zero}"; exit 2 ;;
esac
