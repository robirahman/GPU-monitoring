#!/usr/bin/env bash
# Ship the current working tree to the nodes WITHOUT a git credential.
#
# WHY: bootstrap phase (d) cloned a private repo, which needs a PAT on every node
# (and the last one leaked into a chat + a cached /etc/spar/git_pat). `git archive`
# of HEAD is a few MB because LFS-tracked parquets are stored as POINTER files, and
# the nodes never read the corpus -- they generate telemetry. So there is no reason
# for a compute node to hold a credential at all.
#
# Usage: ZONE=us-central1-c bash scripts/gcp/stage_repo.sh spar-a100-0 spar-a100-1
set -euo pipefail
cd "$(git rev-parse --show-toplevel)"
ZONE="${ZONE:?set ZONE}"
PROJECT="${PROJECT:-$(gcloud config get-value project 2>/dev/null)}"
REPO_DIR="${REPO_DIR:-/opt/spar/repo}"
REF="${REF:-HEAD}"
COMMIT="$(git rev-parse "$REF")"

# Whitelist, not the whole tree: HEAD is 682 MB (tracked .ncu-rep profiles and
# collection logs under data/, plus report/ and literature/ PDFs), none of which a
# compute node reads. These paths are 0.6 MB gzipped.
read -r -a STAGE_PATHS <<< "${SPAR_STAGE_PATHS:-workloads scripts classifier requirements_v2.txt requirements_gcp_extra.txt requirements_tpu.txt requirements.txt}"
TAR="$(mktemp -d)/spar_repo.tar.gz"
git archive --format=tar "$REF" -- "${STAGE_PATHS[@]}" | gzip -9 > "$TAR"
echo "[stage_repo] $(du -h "$TAR" | cut -f1) at commit ${COMMIT:0:12} -> $* "

for n in "$@"; do
  gcloud compute scp --zone="$ZONE" --project="$PROJECT" "$TAR" "$n:/tmp/spar_repo.tar.gz" >/dev/null
  gcloud compute ssh "$n" --zone="$ZONE" --project="$PROJECT" --command="
    set -e
    sudo install -d '$REPO_DIR'
    sudo tar -xzf /tmp/spar_repo.tar.gz -C '$REPO_DIR'
    printf '%s\n' '$COMMIT' | sudo tee '$REPO_DIR/.spar_staged_commit' >/dev/null
    rm -f /tmp/spar_repo.tar.gz
    echo \"[stage_repo] \$(hostname): \$(ls '$REPO_DIR' | wc -l) entries, commit \$(cat '$REPO_DIR/.spar_staged_commit')\"" 2>&1 | grep -a stage_repo
done
rm -rf "$(dirname "$TAR")"
