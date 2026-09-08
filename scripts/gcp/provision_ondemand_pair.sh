#!/usr/bin/env bash
# Win the on-demand capacity lottery for a PAIR, all-or-nothing.
#
# WHY. Spot preempted the A100 pair three times, twice before the cells we
# cared about ever ran, so some runs need a non-preemptible pair. But on-demand
# A100 capacity is transient in a way that defeats a single create: on
# 2026-09-08 a probe found us-east1-b available and the very next create in
# that zone returned ZONE_RESOURCE_POOL_EXHAUSTED. Quota is not the constraint
# (16 A100s in each of ~9 regions, all unused) -- capacity is.
#
# The pair is the unit, because a 2-node job needs both nodes at once. Creating
# them independently risks node0 running (and billing ~$29/h) while node1 never
# lands, so if the second create fails this releases the first and moves on
# rather than leaving half a pair burning money.
#
# Flex Start would also give an uninterruptible block, but plain
# `instances create --provisioning-model=FLEX_START` queues each VM
# SEPARATELY, which has the same half-a-pair problem with a longer fuse.
#
#   ZONES="us-east1-b us-east4-a" NODE0=spar-a100od-0 NODE1=spar-a100od-1 \
#     bash scripts/gcp/provision_ondemand_pair.sh
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
NODE0="${NODE0:?set NODE0}"; NODE1="${NODE1:?set NODE1}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-highgpu-8g}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-500GB}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
PASSES="${PASSES:-6}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-120}"
ZONES="${ZONES:-us-east1-b us-east1-c us-east1-d us-east4-a us-east4-b us-east4-c \
us-central1-a us-central1-b us-central1-c us-central1-f us-west1-b us-west4-a \
europe-west4-a europe-west4-b asia-east1-a asia-east1-b asia-northeast1-a}"
say() { echo "[$(date -u '+%H:%M:%SZ')] $*"; }

mk() {  # mk <zone> <name> <idx>
  gcloud compute instances create "$2" --project="$PROJECT" --zone="$1" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type=pd-balanced \
    --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
    --scopes=cloud-platform \
    --metadata="spar-node-index=$3,spar-node-name=$2" \
    --labels="spar-role=gpu" >/dev/null 2>&1
}
rm_() { gcloud compute instances delete "$2" --project="$PROJECT" --zone="$1" --quiet >/dev/null 2>&1; }

for pass in $(seq 1 "$PASSES"); do
  for z in $ZONES; do
    say "pass $pass: trying $z"
    if ! mk "$z" "$NODE0" 0; then continue; fi
    say "  $NODE0 landed in $z -- racing for $NODE1"
    if mk "$z" "$NODE1" 1; then
      say "PAIR UP in $z: $NODE0 + $NODE1 (STANDARD, non-preemptible)"
      echo "SPAR_PAIR_ZONE=$z"
      exit 0
    fi
    say "  $NODE1 failed -- releasing $NODE0 rather than leaving half a pair billing"
    rm_ "$z" "$NODE0"
  done
  say "pass $pass found nothing; sleeping ${SLEEP_BETWEEN}s"
  sleep "$SLEEP_BETWEEN"
done
say "GIVING UP after $PASSES passes -- no zone had capacity for both nodes"
exit 1
