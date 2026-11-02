#!/usr/bin/env bash
# Get a 1-GPU A100 back and run llm_infer_mistral_7b_q4 on it.
#
# Two lessons are baked in.
#
# STOP_WHEN_DONE=1 cost us the node: spar-a100-1g powered itself off after the
# failed campaign and then could not be restarted -- `instances start` returned
# STOCKOUT, because 1-GPU A100 capacity had gone in the 45 minutes since it
# landed. On a device this scarce the node is worth more than the idle billing,
# so this runs with STOP_WHEN_DONE=0 and leaves teardown to a human who has
# seen the data.
#
# And the first attempt launched with transformers missing, because the
# installer's output went through `| tail -3` and the verification line was
# non-fatal. Here the gate exits non-zero before anything launches.
set -uo pipefail
P=focus-surfer-494820-g0
REPO=/srv/gpu-monitoring/cloudshell-workspace
STATE=/srv/gpu-monitoring/a100-state
OLD=spar-a100-1g; OLD_ZONE=us-central1-a
NEW=spar-a100-1gb
ZONES="us-central1-a us-central1-c us-east4-c us-east5-a us-east5-b europe-west4-a europe-west4-b asia-southeast1-c"
mkdir -p "$STATE"
say() { echo "[arm-a100] $(date -u '+%H:%M:%SZ') $*"; }
nssh() { yes "" 2>/dev/null | timeout 900 gcloud compute ssh "$1" --zone="$2" --project=$P --quiet --command="$3" 2>/dev/null; }

[ -f "$STATE/launched" ] && { say "already launched"; exit 0; }

NODE=""; ZONE=""
for attempt in $(seq 1 120); do
  # the stopped node still has its venv and CIFAR, so prefer it
  if gcloud compute instances start "$OLD" --project=$P --zone="$OLD_ZONE" --quiet >/dev/null 2>&1; then
    NODE=$OLD; ZONE=$OLD_ZONE; say "restarted $OLD in $ZONE (disk intact)"; break
  fi
  for z in $ZONES; do
    if gcloud compute instances create "$NEW" --project=$P --zone="$z" \
        --machine-type=a2-ultragpu-1g \
        --image-family=common-cu129-ubuntu-2204-nvidia-580 \
        --image-project=deeplearning-platform-release \
        --boot-disk-size=500GB --boot-disk-type=pd-balanced \
        --provisioning-model=STANDARD --maintenance-policy=TERMINATE \
        --scopes=cloud-platform >/dev/null 2>&1; then
      NODE=$NEW; ZONE=$z; say "created $NEW in $z (needs staging)"; break 2
    fi
  done
  [ -n "$NODE" ] && break
  say "attempt $attempt: no 1-GPU A100 capacity anywhere; retrying in 120s"
  sleep 120
done
[ -z "$NODE" ] && { say "gave up after 120 attempts"; exit 1; }
echo "$NODE $ZONE" > "$STATE/node"

say "waiting for sshd"
for _ in $(seq 1 40); do
  [ "$(nssh "$NODE" "$ZONE" 'echo SSH_OK' | tr -d '[:space:]')" = "SSH_OK" ] && break
  sleep 30
done
[ "$(nssh "$NODE" "$ZONE" 'echo SSH_OK' | tr -d '[:space:]')" = "SSH_OK" ] || { say "unreachable"; exit 1; }

# a freshly created node has no repo; a restarted one does
if [ "$NODE" = "$NEW" ]; then
  say "staging repo + setup"
  cd "$REPO"
  git archive --format=tar HEAD -- workloads scripts classifier requirements_v2.txt requirements.txt \
    | gzip -9 > /tmp/a100b.tar.gz
  git rev-parse HEAD > /tmp/a100b_commit.txt
  gcloud compute scp --zone="$ZONE" --project=$P --quiet /tmp/a100b.tar.gz /tmp/a100b_commit.txt \
    "$REPO/scripts/gcp/setup_node.sh" "$REPO/scripts/gcp/rerun_step_logs.sh" "$NODE":/tmp/ >/dev/null 2>&1
  nssh "$NODE" "$ZONE" '
    sudo install -d /opt/spar/repo /opt/spar/gcp
    sudo tar -xzf /tmp/a100b.tar.gz -C /opt/spar/repo
    sudo cp /tmp/a100b_commit.txt /opt/spar/repo/.spar_staged_commit
    sudo cp /tmp/setup_node.sh /tmp/rerun_step_logs.sh /opt/spar/gcp/
    sudo chmod +x /opt/spar/gcp/*.sh
    echo STAGED_OK'
  nssh "$NODE" "$ZONE" 'sudo bash /opt/spar/gcp/setup_node.sh' | tail -5
fi

say "installing transformers and gating on it"
gate=$(nssh "$NODE" "$ZONE" '
  sudo /opt/spar/repo/venv-main/bin/pip install transformers accelerate sentencepiece 2>&1 | tail -4
  sudo /opt/spar/repo/venv-main/bin/python -c "
import sys, torch, transformers
print(\"GATE torch\", torch.__version__, \"transformers\", transformers.__version__, \"gpus\", torch.cuda.device_count())
sys.exit(0 if torch.__version__.startswith(\"2.5.1\") and torch.cuda.device_count() >= 1 else 1)
" && echo GATE_OK')
echo "$gate" | tail -4
if ! echo "$gate" | grep -q GATE_OK; then
  say "gate FAILED -- not launching; node left up for inspection"
  exit 1
fi

say "launching 3 reps (STOP_WHEN_DONE=0: the node is scarcer than the billing)"
nssh "$NODE" "$ZONE" "cd /opt/spar/repo && sudo rm -f logs/20261102b__*.log && sudo bash -c 'setsid env STAMP=20261102b REPS=3 STOP_WHEN_DONE=0 REQUIRE_GPU_ACTIVITY=1 CELLS=llm_infer_mistral_7b_q4 nohup bash /opt/spar/gcp/rerun_step_logs.sh > /var/log/spar_mistral2.log 2>&1 < /dev/null &'"
sleep 25
if nssh "$NODE" "$ZONE" 'pgrep -f "[r]erun_step_logs.sh" >/dev/null && echo LAUNCH_OK' | grep -q LAUNCH_OK; then
  date -u +%FT%TZ > "$STATE/launched"
  say "campaign running on $NODE ($ZONE)"
else
  say "campaign did not start"; exit 1
fi
