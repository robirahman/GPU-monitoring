#!/usr/bin/env bash
# Get a 1-GPU A100 SXM4 80GB for the Round 7 organic-control collection.
#
# Differences from scripts/gcp/arm_a100_1g.sh, which this is modelled on:
#   * stages the WORKING TREE, not git HEAD -- the new workloads are not
#     committed and staging from HEAD would silently ship a node without them.
#   * does not launch a campaign. The gate is the end of this script; the
#     collection is launched by hand once the smoke test has passed.
#   * STOP_WHEN_DONE is not set anywhere: per the lesson in arm_a100_1g.sh,
#     a stopped 1-GPU A100 could not be restarted (STOCKOUT), so the node is
#     left running and teardown is a human decision.
set -uo pipefail
P=focus-surfer-494820-g0
REPO=/srv/gpu-monitoring/cloudshell-workspace
STATE=/srv/gpu-monitoring/r7-state
NEW=spar-a100-r7
ZONES="us-central1-a us-central1-c us-east4-c us-east5-a us-east5-b europe-west4-a europe-west4-b asia-southeast1-c"
mkdir -p "$STATE"
say() { echo "[arm-r7] $(date -u '+%H:%M:%SZ') $*"; }
nssh() { yes "" 2>/dev/null | timeout 900 gcloud compute ssh "$1" --zone="$2" --project=$P --quiet --command="$3" 2>/dev/null; }

NODE=""; ZONE=""
for attempt in $(seq 1 60); do
  for z in $ZONES; do
    if gcloud compute instances create "$NEW" --project=$P --zone="$z" \
        --machine-type=a2-ultragpu-1g \
        --image-family=common-cu129-ubuntu-2204-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=500GB --boot-disk-type=pd-balanced \
        --maintenance-policy=TERMINATE --metadata=install-nvidia-driver=True \
        --scopes=cloud-platform >/dev/null 2>&1; then
      NODE=$NEW; ZONE=$z; say "CREATED $NEW in $z"; break 2
    fi
  done
  say "attempt $attempt/60: no 1-GPU A100 capacity in any zone; retry in 120s"
  sleep 120
done
if [ -z "$NODE" ]; then say "STOCKOUT: gave up after 60 attempts (2h)"; exit 1; fi
echo "$NODE $ZONE" > "$STATE/node"

say "waiting for sshd"
for _ in $(seq 1 40); do
  [ "$(nssh "$NODE" "$ZONE" 'echo SSH_OK' | tr -d '[:space:]')" = "SSH_OK" ] && break
  sleep 30
done
[ "$(nssh "$NODE" "$ZONE" 'echo SSH_OK' | tr -d '[:space:]')" = "SSH_OK" ] || { say "unreachable"; exit 1; }

say "staging working tree"
cd "$REPO"
tar -czf /tmp/r7.tar.gz workloads scripts classifier requirements_v2.txt requirements.txt 2>/dev/null
git rev-parse HEAD > /tmp/r7_commit.txt
gcloud compute scp --zone="$ZONE" --project=$P --quiet /tmp/r7.tar.gz /tmp/r7_commit.txt \
  "$REPO/scripts/gcp/setup_node.sh" "$NODE":/tmp/ >/dev/null 2>&1
nssh "$NODE" "$ZONE" '
  sudo install -d /opt/spar/repo /opt/spar/gcp
  sudo tar -xzf /tmp/r7.tar.gz -C /opt/spar/repo
  sudo cp /tmp/r7_commit.txt /opt/spar/repo/.spar_staged_commit
  sudo cp /tmp/setup_node.sh /opt/spar/gcp/
  sudo chmod +x /opt/spar/gcp/*.sh
  echo STAGED_OK'
nssh "$NODE" "$ZONE" 'sudo bash /opt/spar/gcp/setup_node.sh' | tail -5

say "gating on torch + GPU + transformers"
gate=$(nssh "$NODE" "$ZONE" '
  # transformers MUST be <5. requirements_v2.txt:53: 5.20.2 with torch 2.5.1
  # prints "Disabling PyTorch because PyTorch >= 2.6 is required" and makes
  # AutoModelForCausalLM unavailable, so every cell aborts in seconds with the
  # GPU untouched. The deeplearning-platform image installs 5.x by default, so
  # an unpinned install here silently loses the whole campaign.
  sudo /opt/spar/repo/venv-main/bin/pip install -q "transformers<5" accelerate sentencepiece 2>&1 | tail -2
  sudo /opt/spar/repo/venv-main/bin/python -c "
import sys, torch, transformers
from transformers import AutoModelForCausalLM
ok = torch.cuda.device_count() >= 1 and transformers.__version__.split('.')[0] == '4'
print(\"GATE torch\", torch.__version__, \"transformers\", transformers.__version__,
      \"gpus\", torch.cuda.device_count(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"none\")
sys.exit(0 if ok else 1)
" && echo GATE_OK')
echo "$gate" | tail -3
echo "$gate" | grep -q GATE_OK && say "READY: $NODE ($ZONE)" || { say "GATE FAILED; node left up"; exit 1; }
