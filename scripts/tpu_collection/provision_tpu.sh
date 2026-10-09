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
ACCEL=${ACCEL:-v6e-16}                    # v6e = 8 chips/host => v6e-16 = 2 hosts (NOT 4)
RUNTIME=${RUNTIME:-v2-alpha-tpuv6e}
TPU_NAME=${TPU_NAME:-spar-tpu-16}
QR_NAME=${QR_NAME:-${TPU_NAME}-qr}        # queued-resource id
SA=${SA:-tpu-service-account@focus-surfer-494820-g0.iam.gserviceaccount.com}
VALID_UNTIL=${VALID_UNTIL:-4h}            # queued resource self-expires after this
SPOT=${SPOT:-1}                           # 1 = Spot (preemptible); 0 = on-demand
# NODE_COUNT>1 requests a MULTISLICE group: N slices scheduled as one unit, named
# ${NODE_PREFIX}-0 .. ${NODE_PREFIX}-(N-1). Needed because megascale.* metrics --
# including megascale.collective_end_to_end_latencies, the DCN channel Phase 0
# identified as the only plausible substitute for the power signal TPU does not
# expose -- answer UNAVAILABLE "has no accumulated data yet" on a single slice.
# They exist but only accumulate across slices. Cost warning: N slices bill
# simultaneously, and all N must land together or none are useful.
NODE_COUNT=${NODE_COUNT:-1}
NODE_PREFIX=${NODE_PREFIX:-$TPU_NAME}

ACTION=${1:-status}

gc() { gcloud "$@" --project="$PROJECT" --zone="$ZONE"; }

create() {
  echo "Creating queued resource $QR_NAME -> $TPU_NAME ($ACCEL, $ZONE, spot=$SPOT, nodes=$NODE_COUNT)"
  local spot_flag=()
  [ "$SPOT" = "1" ] && spot_flag=(--spot)
  # Single slice uses --node-id; multislice uses --node-count + --node-prefix,
  # and the two forms are mutually exclusive.
  local shape_flags=(--node-id="$TPU_NAME")
  if [ "$NODE_COUNT" -gt 1 ]; then
    shape_flags=(--node-count="$NODE_COUNT" --node-prefix="$NODE_PREFIX")
    echo "  MULTISLICE: $NODE_COUNT x $ACCEL as ${NODE_PREFIX}-0..$((NODE_COUNT-1))"
    echo "  NOTE: all $NODE_COUNT slices bill at once once ACTIVE."
  fi
  gcloud compute tpus queued-resources create "$QR_NAME" \
    --project="$PROJECT" --zone="$ZONE" \
    "${shape_flags[@]}" \
    --accelerator-type="$ACCEL" \
    --runtime-version="$RUNTIME" \
    --valid-until-duration="$VALID_UNTIL" \
    --service-account="$SA" \
    "${spot_flag[@]}"
  echo "Submitted. Poll with: bash $0 status"
}

status() {
  [ "$NODE_COUNT" -gt 1 ] && echo "(multislice: expecting ${NODE_PREFIX}-0..$((NODE_COUNT-1)))"
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
  echo "Deleting node(s) + queued resource (idempotent)..."
  if [ "$NODE_COUNT" -gt 1 ]; then
    # Multislice nodes are ${NODE_PREFIX}-0..N-1; deleting only $TPU_NAME would
    # leave the rest billing.
    for i in $(seq 0 $((NODE_COUNT-1))); do
      gc compute tpus tpu-vm delete "${NODE_PREFIX}-${i}" --quiet 2>/dev/null || true
    done
  fi
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
