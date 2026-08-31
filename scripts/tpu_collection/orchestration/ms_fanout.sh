#!/bin/bash
# Fan a command to every worker of every slice in a multislice group, IN PARALLEL,
# with per-slice identity.
#
# Both properties are mandatory, and both were learned the expensive way on a live
# 2x v6e-1:
#
# 1. PARALLEL. jax.distributed.initialize() blocks until EVERY process joins. A
#    sequential loop over slices deadlocks: slice 0 waits for slice 1, which has
#    not been started yet. The first multislice driver hung 8 minutes this way and
#    burned most of a rental.
#
# 2. PER-SLICE IDENTITY. MEGASCALE_SLICE_ID / MEGASCALE_NUM_SLICES /
#    MEGASCALE_COORDINATOR_ADDRESS are ALL UNSET on a --node-count group, even
#    though jax still forms the mesh correctly (verified device_count=2,
#    process 0/1 of 2). So nothing on the worker can tell which slice it is, and
#    run_workload_tpu would resolve both to worker_id 0 -- one slice's parquet and
#    .done marker silently overwriting the other's. The orchestration knows, so it
#    exports SPAR_TPU_SLICE_ID.
#
# Also: no gcloud ssh retry loop. gcloud retries on failure, and each retry starts
# a NEW jax process that collides with the still-registered old one
# ("ABORTED: unexpectedly tried to connect with a different incarnation"),
# producing SIGABRT storms. One attempt per slice; let the caller decide.
#
# Usage:  ms_fanout.sh <zone> <node-prefix> <n_slices> '<remote command>'
set -uo pipefail
ZONE="${1:?zone}"; PREFIX="${2:?node prefix}"; N="${3:?n slices}"; CMD="${4:?command}"
PROJECT=${PROJECT:-focus-surfer-494820-g0}
pids=()
for i in $(seq 0 $((N-1))); do
  (
    gcloud compute tpus tpu-vm ssh "${PREFIX}-${i}" \
      --project="$PROJECT" --zone="$ZONE" --worker=all \
      --command="export SPAR_TPU_SLICE_ID=${i}; ${CMD}" 2>&1 \
      | grep -vE 'SSH:|batch size|SSH key found|Warning: Permanently' \
      | sed "s/^/[slice${i}] /"
  ) &
  pids+=($!)
done
rc=0
for pid in "${pids[@]}"; do wait "$pid" || rc=1; done
exit $rc
