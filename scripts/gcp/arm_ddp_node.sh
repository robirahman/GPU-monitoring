#!/usr/bin/env bash
# Stage and launch the DDP wave the moment an 8-GPU node appears.
#
# a2-ultragpu-8g capacity is transient -- every zone has been STOCKOUT for an
# hour -- so the window may open at 4am. This watches for the node the two
# lotteries are hunting and brings it up unattended.
#
# WHAT THE LAST DRIVER GOT WRONG, and this one does not: it treated a failed
# SSH as evidence of a state change. `nssh ... | grep -q RUNNING` is false both
# when the campaign has finished and when the connection broke, and on a broken
# connection it concluded "done" and launched a second campaign onto a busy
# GPU. Here every transition needs POSITIVE evidence -- a marker file, a
# string the node actually printed -- and anything else means "unknown, look
# again later".
set -uo pipefail
P=focus-surfer-494820-g0
REPO=/srv/gpu-monitoring/cloudshell-workspace
STATE=/srv/gpu-monitoring/ddp-state
NODE="${NODE:-spar-rerun-0}"
STAMP="${STAMP:-20261023c}"
mkdir -p "$STATE"
say() { echo "[$(date -u '+%H:%M:%SZ')] [arm-ddp] $*"; }
nssh() { yes "" 2>/dev/null | timeout 900 gcloud compute ssh "$NODE" --zone="$1" --project=$P --quiet --command="$2" 2>/dev/null; }

[ -f "$STATE/launched" ] && { say "already launched; nothing to do"; exit 0; }

say "watching for ${NODE}"
ZONE=""
while [ -z "$ZONE" ]; do
  ZONE=$(timeout 120 gcloud compute instances list --project=$P \
          --filter="name=${NODE} AND status=RUNNING" --format="value(zone)" 2>/dev/null | head -1)
  [ -z "$ZONE" ] && sleep 120
done
say "node is RUNNING in ${ZONE}"

# Positive readiness gate: the node must say "SSH_OK" itself.
say "waiting for sshd"
for _ in $(seq 1 40); do
  [ "$(nssh "$ZONE" 'echo SSH_OK' | tr -d "[:space:]")" = "SSH_OK" ] && break
  sleep 30
done
[ "$(nssh "$ZONE" 'echo SSH_OK' | tr -d "[:space:]")" = "SSH_OK" ] || {
  say "node never became reachable; leaving it alone for a human"; exit 1; }

if [ ! -f "$STATE/staged" ]; then
  say "staging repo at $(cd "$REPO" && git rev-parse --short HEAD)"
  cd "$REPO"
  git archive --format=tar HEAD -- workloads scripts classifier requirements_v2.txt requirements.txt \
    | gzip -9 > /tmp/ddp_repo.tar.gz
  git rev-parse HEAD > /tmp/ddp_commit.txt
  gcloud compute scp --zone="$ZONE" --project=$P --quiet \
    /tmp/ddp_repo.tar.gz /tmp/ddp_commit.txt \
    "$REPO/scripts/gcp/setup_node.sh" "$REPO/scripts/gcp/rerun_step_logs.sh" \
    "$NODE":/tmp/ >/dev/null 2>&1
  out=$(nssh "$ZONE" '
    sudo install -d /opt/spar/repo /opt/spar/gcp
    sudo tar -xzf /tmp/ddp_repo.tar.gz -C /opt/spar/repo
    sudo cp /tmp/ddp_commit.txt /opt/spar/repo/.spar_staged_commit
    sudo cp /tmp/setup_node.sh /tmp/rerun_step_logs.sh /opt/spar/gcp/
    sudo chmod +x /opt/spar/gcp/*.sh
    echo STAGED_OK')
  [[ "$out" == *STAGED_OK* ]] || { say "staging did not confirm; stopping"; exit 1; }
  touch "$STATE/staged"; say "staged"
fi

if [ ! -f "$STATE/setup" ]; then
  say "running node setup (venv + CIFAR); this takes a few minutes"
  nssh "$ZONE" 'sudo bash /opt/spar/gcp/setup_node.sh' | tail -8
  touch "$STATE/setup"
fi

# The DDP cells are `torchrun --nproc_per_node=8`. Launching them on a node
# that cannot see 8 GPUs produces eight ranks fighting over fewer devices, or
# a hang. Gate on the count the node reports, not on the machine type we asked
# for.
ngpu=$(nssh "$ZONE" '/opt/spar/repo/venv-main/bin/python -c "import torch;print(torch.cuda.device_count())"' | tr -d "[:space:]")
say "node reports ${ngpu:-?} visible GPUs"
if [ "$ngpu" != "8" ]; then
  say "refusing to launch the 8-way DDP cells on ${ngpu:-unknown} GPUs"; exit 1
fi

say "launching DDP wave ${STAMP}"
nssh "$ZONE" "cd /opt/spar/repo && sudo bash -c 'setsid env STAMP=${STAMP} REPS=3 CELL_SET=ddp STOP_WHEN_DONE=1 \
  nohup bash /opt/spar/gcp/rerun_step_logs.sh > /var/log/spar_rerun_ddp.log 2>&1 < /dev/null &'"
sleep 25
alive=$(nssh "$ZONE" 'pgrep -f "[r]erun_step_logs.sh" >/dev/null && echo LAUNCH_OK || echo LAUNCH_NO')
if [[ "$alive" == *LAUNCH_OK* ]]; then
  date -u +%FT%TZ > "$STATE/launched"
  echo "$ZONE" > "$STATE/zone"
  say "DDP wave running in ${ZONE}; node powers itself off when done"
else
  say "campaign did not come up; left for inspection"; exit 1
fi
