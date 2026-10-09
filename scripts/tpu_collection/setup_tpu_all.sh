#!/usr/bin/env bash
# Run setup_tpu_host.sh on every TPU worker, reading the GitHub PAT from a local
# 0600 file.
#
# Why a file and not an argument: a PAT typed into a command line ends up in
# shell history, in `ps` output, and -- if an assistant is driving the session --
# in the conversation transcript. Writing it once with `read -rsp` keeps it out
# of all three:
#
#   read -rsp "PAT: " P && printf '%s' "$P" > ~/.spar_pat && chmod 600 ~/.spar_pat && unset P
#   bash scripts/tpu_collection/setup_tpu_all.sh
#
# setup_tpu_host.sh rewrites the cloned remote to its tokenless form, so the
# credential does not persist in .git/config on the workers. Revoke the PAT
# after teardown regardless.
set -uo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=../gcp/env.sh
source "${SCRIPT_DIR}/../gcp/env.sh"

PAT_FILE="${PAT_FILE:-$HOME/.spar_pat}"
[ -r "$PAT_FILE" ] || { echo "FATAL: no readable PAT file at $PAT_FILE" >&2; exit 1; }
PAT="$(tr -d ' \r\n' < "$PAT_FILE")"
[ -n "$PAT" ] || { echo "FATAL: $PAT_FILE is empty" >&2; exit 1; }
echo "using PAT from $PAT_FILE (${#PAT} chars)"

ZONE_ARG="${TPU_ZONE}"   # env.sh's ZONE is the GPU zone; the TPU lives elsewhere
echo "== pushing setup_tpu_host.sh to all workers of ${TPU_NAME} (${ZONE_ARG}) =="
gcloud compute tpus tpu-vm scp "${SCRIPT_DIR}/setup_tpu_host.sh" \
  "${TPU_NAME}:~" --zone="$ZONE_ARG" --worker=all || exit 1

echo "== running setup on all workers (pip + clone, ~5-10 min) =="
REPO_HOST_PATH="${REPO_HOST_PATH:-github.com/robirahman/GPU-monitoring.git}"
gcloud compute tpus tpu-vm ssh "$TPU_NAME" --zone="$ZONE_ARG" --worker=all \
  --command="REPO_URL='https://x-access-token:${PAT}@${REPO_HOST_PATH}' BRANCH='${REPO_BRANCH}' BUCKET='${BUCKET}' SESSION='${SESSION}' bash ~/setup_tpu_host.sh"
rc=$?
unset PAT
echo "== setup exit rc=$rc =="
exit $rc
