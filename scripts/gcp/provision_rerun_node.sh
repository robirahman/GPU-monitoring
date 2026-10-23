#!/usr/bin/env bash
# Win the on-demand capacity lottery for ONE a2-ultragpu-8g.
#
# Quota is not the constraint here (16 A100-80GB in each of four regions, all
# unused); capacity is, and it moves between zones minute to minute. Hence a
# lottery rather than a single create.
#
# a2-ultragpu-8g rather than a 1g variant because six of the seventeen cells
# are `torchrun --nproc_per_node=8` DDP runs, and a batch rate only transfers
# to the archived run at the same width.
set -uo pipefail
PROJECT="${PROJECT:-focus-surfer-494820-g0}"
NODE="${NODE:-spar-rerun-0}"
MACHINE_TYPE="${MACHINE_TYPE:-a2-ultragpu-8g}"
IMAGE_FAMILY="${IMAGE_FAMILY:-common-cu129-ubuntu-2204-nvidia-580}"
IMAGE_PROJECT="${IMAGE_PROJECT:-deeplearning-platform-release}"
# Regions with NVIDIA_A100_80GB_GPUS=16; us-central1 first because the analysis
# box is there and staging the repo stays in-zone.
ZONES="${ZONES:-us-central1-a us-central1-c us-east4-c us-east5-a us-east5-b europe-west4-a europe-west4-b}"
PASSES="${PASSES:-4}"
SLEEP_BETWEEN="${SLEEP_BETWEEN:-90}"
say() { echo "[$(date -u '+%H:%M:%SZ')] [provision] $*"; }

existing="$(gcloud compute instances list --project="$PROJECT" \
  --filter="name=${NODE}" --format='value(zone)' 2>/dev/null | head -1)"
if [ -n "$existing" ]; then
  say "$NODE already exists in $existing"; echo "SPAR_ZONE=$existing"; exit 0
fi

for pass in $(seq 1 "$PASSES"); do
  for z in $ZONES; do
    say "pass ${pass}: trying ${z}"
    if gcloud compute instances create "$NODE" --project="$PROJECT" --zone="$z" \
        --machine-type="$MACHINE_TYPE" \
        --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
        --boot-disk-size=500GB --boot-disk-type=pd-balanced \
        --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
        --scopes=cloud-platform \
        --labels=spar-role=rerun,spar-purpose=step-log-reference \
        --metadata=spar-purpose=step-log-reference >/dev/null 2>&1; then
      say "UP in ${z} (STANDARD, 8x A100-80GB)"
      echo "SPAR_ZONE=$z"
      exit 0
    fi
  done
  say "pass ${pass} exhausted; sleeping ${SLEEP_BETWEEN}s"
  sleep "$SLEEP_BETWEEN"
done
say "no capacity after ${PASSES} passes"
exit 1
