#!/usr/bin/env bash
# provision_single.sh -- one GPU node, one cell list, resumable after preemption.
#
# Why this exists
# ---------------
# Every other provisioner in scripts/gcp/ builds a PAIR: it creates node0, races
# for node1, and deletes node0 if node1 does not land. That is correct for the
# s16 cells, which need a 16-rank rendezvous. It is wrong for an s8 cell, which
# runs `torchrun --standalone --nproc-per-node=8` inside one node and never
# talks to a peer -- a pair would double the bill to leave one node idle.
#
# It also skips launch_queue.sh / run_stage_and_stop.sh and calls the queue
# directly. Those two wrappers are peer-gated in four places (cell_barrier, the
# stage-complete wait, the queue_done rendezvous, STOP_WHEN_DONE) and each gate
# is satisfied in a different way by a node with no peer. Threading NNODES=1
# through all of them is more fragile than not using them: a gate that fails
# open costs 30 idle minutes on an 8xH100.
#
# RESUME AFTER PREEMPTION
# -----------------------
# multinode_queue.sh already writes a per-cell-per-rep marker:
#     logs/.done/<RUN_STAMP>/<cell>_node0_<rid>.done,  rid = sha1(STAMP_cell_rep)
# and run_stage skips any cell whose marker exists. sync_data.sh pushes that
# tree to <CTRL>/.done, including on the ~30 s preemption notice. What was
# missing is the other half: nothing ever pulled it back, so a replacement node
# re-collected everything. This startup script restores <CTRL>/.done first,
# which makes re-running with the SAME RUN_STAMP resume at the rep that was
# interrupted and skip the ones already banked.
#
#   RUN_STAMP=20261105a ONLY_CELLS='^s8_train_ppodist' REPS=2 \
#     bash scripts/gcp/provision_single.sh
#
# On-demand is tried first and Spot only if ALLOW_SPOT=1, because an 8xH100
# reclaimed mid-cell costs a restart; with resume that is survivable, without it
# it is the whole round.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"
_MT_SET="${MACHINE_TYPE:-}"
source scripts/gcp/env.sh 2>/dev/null || true

PROJECT="${PROJECT:-focus-surfer-494820-g0}"
BUCKET="${BUCKET:-gs://${PROJECT}-telemetry}"
RUN_STAMP="${RUN_STAMP:?set RUN_STAMP}"
SESSION="${SESSION:-${RUN_STAMP%[a-z]}}"
NODE="${SINGLE_NODE:-spar-rl8-0}"
MACHINE_TYPE="${_MT_SET:-a3-highgpu-8g}"
MACHINE_TYPES="${MACHINE_TYPES:-$MACHINE_TYPE}"
IMAGE_FAMILY="${IMAGE_FAMILY:-spar-node}"
IMAGE_PROJECT="${IMAGE_PROJECT:-$PROJECT}"
BOOT_DISK_SIZE="${BOOT_DISK_SIZE:-500GB}"
REPS="${REPS:-2}"
DUR="${DUR:-600}"
ONLY_CELLS="${ONLY_CELLS:?set ONLY_CELLS}"
ZONES="${ZONES:-us-central1-a us-central1-b us-central1-c us-west1-a us-west1-b europe-west1-b asia-northeast1-b}"
MAX_RUN="${MAX_RUN:-14400s}"
ALLOW_SPOT="${ALLOW_SPOT:-1}"
STAGING="${BUCKET}/staging/${RUN_STAMP}"
CTRL="${BUCKET}/runs/${SESSION}/control"
say() { echo "[$(date -u '+%H:%M:%SZ')] $*"; }

existing="$(gcloud compute instances list --project="$PROJECT" \
  --filter="name=($NODE)" --format='value(name,zone,status)' 2>/dev/null)"
[ -n "$existing" ] && { say "REFUSING: $NODE already exists:"; printf '  %s\n' "$existing"; exit 1; }

# Sweep this script's own leavings before adding more (see provision_spot_pair.sh:
# sixteen stale dirs filled /tmp at 99% on 2026-10-03 and the archive wrote 0 bytes).
for _old in /tmp/tmp.*; do
  [ -d "$_old" ] && [ -e "$_old/startup.sh" ] && [ "$_old" != "${TMP:-}" ] && rm -rf "$_old"
done
_free_kb="$(df -Pk /tmp | awk 'NR==2{print $4}')"
[ "${_free_kb:-0}" -lt 102400 ] && { say "FATAL: only $((_free_kb/1024)) MB free on /tmp"; exit 1; }

read -r -a STAGE_PATHS <<< "${SPAR_STAGE_PATHS:-workloads scripts classifier requirements_v2.txt requirements_gcp_extra.txt requirements_tpu.txt requirements.txt}"
COMMIT="$(git rev-parse HEAD)"
TMP="$(mktemp -d)"
git archive --format=tar HEAD -- "${STAGE_PATHS[@]}" | gzip -9 > "$TMP/spar_repo.tar.gz"
_arc_kb=$(( $(wc -c < "$TMP/spar_repo.tar.gz") / 1024 ))
[ "$_arc_kb" -lt 1024 ] && { say "FATAL: repo archive is ${_arc_kb} KB -- refusing to stage"; exit 1; }
# `git archive` streams from HEAD, so an uncommitted queue edit is NOT staged and
# the node would run the old cell list. Say so loudly rather than discovering it
# in the node's log an hour later.
if ! git diff --quiet HEAD -- scripts/rebuttal_collection/multinode_queue.sh; then
  say "FATAL: multinode_queue.sh differs from HEAD. git archive stages HEAD, so"
  say "       your edit would NOT reach the node. Commit it first."
  exit 1
fi
say "repo archive $(du -h "$TMP/spar_repo.tar.gz" | cut -f1) at ${COMMIT:0:12} -> $STAGING/"
gcloud storage cp "$TMP/spar_repo.tar.gz" "$STAGING/spar_repo.tar.gz" >/dev/null 2>&1 \
  || { say "FATAL: could not stage the repo archive"; exit 1; }

cat > "$TMP/startup.sh" <<STARTUP
#!/usr/bin/env bash
exec > >(tee -a /var/log/spar-startup.log) 2>&1
set -uo pipefail
PROJECT="$PROJECT"; BUCKET="$BUCKET"; SESSION="$SESSION"; STAMP="$RUN_STAMP"
CTRL="$CTRL"; STAGING="$STAGING"; ONLY="$ONLY_CELLS"; REPS="$REPS"; DUR="$DUR"
REPO=/opt/spar/repo
NODE="\$(hostname -s)"
echo "=== single-node startup \$NODE \$(date -u +%FT%TZ) stamp=\$STAMP ==="

mkdir -p /opt/spar && cd /opt/spar
for _ in \$(seq 1 30); do
  gcloud storage cp "\$STAGING/spar_repo.tar.gz" /opt/spar/repo.tgz >/dev/null 2>&1 && break
  sleep 20
done
mkdir -p "\$REPO" && tar xzf /opt/spar/repo.tgz -C "\$REPO"

# bootstrap derives identity from metadata; node0 and node1 are both this node,
# so every peer derivation in the tree resolves to self and no gate can block.
mkdir -p /etc/spar
SPAR_NODE_INDEX=0 NODE_INDEX=0 PEER_NAME="\$NODE" \\
  bash "\$REPO/scripts/gcp/bootstrap_node.sh" || echo "bootstrap returned \$?"
[ -d /opt/spar/gcp ] || ln -s "\$REPO/scripts/gcp" /opt/spar/gcp

# ---- RESUME: restore the .done markers banked by any earlier attempt --------
mkdir -p "\$REPO/logs/.done"
if gcloud storage rsync -r "\$CTRL/.done" "\$REPO/logs/.done" >/dev/null 2>&1; then
  n=\$(find "\$REPO/logs/.done/\$STAMP" -name '*.done' 2>/dev/null | wc -l)
  echo "resume: restored \$n completed-cell marker(s) for \$STAMP from \$CTRL/.done"
  find "\$REPO/logs/.done/\$STAMP" -name '*.done' 2>/dev/null | sed 's/^/  /'
else
  echo "resume: no marker tree at \$CTRL/.done (first attempt)"
fi
# A .failed marker is NOT a skip -- run_stage only reads .done -- so an
# interrupted rep is retried, which is the behaviour we want.

setsid bash "\$REPO/scripts/gcp/preempt_watch.sh" </dev/null >/var/log/spar-preempt.log 2>&1 & disown
setsid bash "\$REPO/scripts/gcp/sync_data.sh" --loop 120 </dev/null >/var/log/spar-sync.log 2>&1 & disown

cd "\$REPO"
. /etc/spar/node_env
# Call the queue directly: NNODES=1 disables cell_barrier, and skipping
# launch_queue.sh / run_stage_and_stop.sh avoids their four peer gates.
NNODES=1 NODE_RANK=0 MASTER_ADDR="\$NODE" MASTER_PORT=29500 \\
  RUN_STAMP="\$STAMP" REPS="\$REPS" DUR="\$DUR" ONLY_CELLS="\$ONLY" \\
  GCPDIR=/opt/spar/gcp \\
  bash scripts/rebuttal_collection/multinode_queue.sh \\
  > "\$REPO/logs/single_\$STAMP.log" 2>&1
echo "queue exited rc=\$? \$(date -u +%FT%TZ)"

bash "\$REPO/scripts/gcp/sync_data.sh" --once || true
bash "\$REPO/scripts/gcp/sync_data.sh" --final || true
printf 'done %s\n' "\$(date -u +%FT%TZ)" | gcloud storage cp - "\$CTRL/\$NODE/single_\$STAMP.complete" || true
echo "=== deleting \$NODE to stop the bill \$(date -u +%FT%TZ) ==="
gcloud compute instances delete "\$NODE" --quiet --zone="\$(curl -s -H 'Metadata-Flavor: Google' \\
  http://metadata.google.internal/computeMetadata/v1/instance/zone | cut -d/ -f4)" || shutdown -h +2
STARTUP

_su_b=$(wc -c < "$TMP/startup.sh")
[ "$_su_b" -lt 512 ] && { say "FATAL: startup-script is ${_su_b} bytes -- heredoc did not write"; exit 1; }

mk() {  # mk <zone> <provisioning-model>
  gcloud compute instances create "$NODE" --project="$PROJECT" --zone="$1" \
    --machine-type="$MACHINE_TYPE" \
    --image-family="$IMAGE_FAMILY" --image-project="$IMAGE_PROJECT" \
    --boot-disk-size="$BOOT_DISK_SIZE" --boot-disk-type=pd-balanced \
    --provisioning-model="$2" --instance-termination-action=DELETE \
    --maintenance-policy=TERMINATE \
    --max-run-duration="$MAX_RUN" --scopes=cloud-platform \
    --metadata-from-file=startup-script="$TMP/startup.sh" \
    --metadata="spar-session=${SESSION},spar-master-addr=${NODE},spar-master-port=29500,spar-bucket=${BUCKET},spar-node0-name=${NODE},spar-node1-name=${NODE},spar-peer-name=${NODE},spar-node-index=0,spar-run-stamp=${RUN_STAMP}" \
    --labels="spar-role=gpu,spar-stamp=$(printf '%s' "$RUN_STAMP" | tr -cd 'a-z0-9-')" \
    2>&1 | tee "$TMP/create.$1.$2.log" | grep -qE "^NAME|Created \["
}
why() { grep -viE "^$|global DNS|zonal-dns|^WARNING|Disk size|^NAME|resize_pd" \
          "$TMP/create.$1.$2.log" 2>/dev/null | head -3 | tr '\n' ' '; }
# A nonzero gcloud exit does NOT mean the VM was not created (ReadTimeout on
# 2026-10-01 left an 8xA100 running after reporting failure). Never infer
# absence from a failed call -- ask.
exists() { gcloud compute instances describe "$NODE" --project="$PROJECT" --zone="$1" \
             --format='value(name)' >/dev/null 2>&1; }

MODELS="STANDARD"
[ "$ALLOW_SPOT" = "1" ] && MODELS="STANDARD SPOT"
for model in $MODELS; do
 for z in $ZONES; do
  for MACHINE_TYPE in $MACHINE_TYPES; do
    say "trying $z ($MACHINE_TYPE, $model)"
    if mk "$z" "$model" || exists "$z"; then
      say "UP in $z ($model $MACHINE_TYPE): $NODE"
      say "  cells: $ONLY_CELLS  REPS=$REPS DUR=$DUR  stamp=$RUN_STAMP"
      [ "$model" = "SPOT" ] && say "  SPOT: if reclaimed, re-run this script with the SAME RUN_STAMP to resume."
      say "  watch: gcloud compute instances list --filter='name=$NODE'"
      exit 0
    fi
    say "  failed in $z: $(why "$z" "$model")"
  done
 done
 say "no zone had $model capacity for $MACHINE_TYPES"
done
exit 1
