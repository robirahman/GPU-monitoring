#!/usr/bin/env bash
# pull_results.sh -- mirror this session's GCS prefix down to the local machine.
# Run from YOUR machine (in the repo root, or anywhere -- it uses absolute-ish paths).
#
#   bash scripts/gcp/pull_results.sh                    # pull $SESSION
#   SESSION=20260820_run1 bash scripts/gcp/pull_results.sh
#
# Lands under ./gcs_runs/<SESSION>/<node>/{data,logs,meta} + control/. Then run
# ingest_session.sh to fold it into data/gcp/<SESSION>/ + logs/ for the analysis.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"
require_cmd gcloud

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
DEST="${REPO_ROOT}/gcs_runs/${SESSION}"
mkdir -p "$DEST"

spar_say "pulling ${GCS_SESSION}/  ->  ${DEST}/"
if ! gcloud storage ls "${GCS_SESSION}/" >/dev/null 2>&1; then
  echo "Nothing at ${GCS_SESSION}/ -- is SESSION=${SESSION} correct? (list with: gcloud storage ls ${BUCKET}/runs/)"
  exit 1
fi
gcloud storage rsync -r "${GCS_SESSION}/" "${DEST}/"

echo
spar_say "pulled. Summary:"
du -sh "$DEST" 2>/dev/null || true
NP="$(find "$DEST" -name '*.parquet' 2>/dev/null | wc -l | tr -d ' ')"
NPART="$(find "$DEST" -name '*.parquet.partial' 2>/dev/null | wc -l | tr -d ' ')"
NLOG="$(find "$DEST" -name '*.log' 2>/dev/null | wc -l | tr -d ' ')"
echo "  parquet files : $NP"
echo "  .partial files: $NPART   (ingest keeps >=600-row ones as *_truncated.parquet)"
echo "  STEP logs     : $NLOG"
echo "next:  bash scripts/gcp/ingest_session.sh"
