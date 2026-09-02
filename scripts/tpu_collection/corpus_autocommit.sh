#!/bin/bash
# Ingest every TPU campaign session from GCS into data/tpu and commit, on a loop.
#
# WHY. GCS is the durability layer -- run_workload_tpu uploads each cell's
# parquets as the cell finishes, so a slice lost mid-run costs at most the cell in
# flight (verified: 256 objects present for the v5e run while it was still going).
# What GCS does NOT give is a versioned, reviewable corpus alongside the code that
# produced it, and a long rental can otherwise accumulate many hours of data whose
# only copy lives in a bucket nobody has looked at.
#
# So: discover session prefixes, ingest anything new, commit. Idempotent -- the
# ingest rewrites the same filenames, so a no-change cycle produces no commit.
set -uo pipefail
cd "$(dirname "$0")/../.." || exit 1
export PATH="$HOME/.local/bin:$PATH"     # git-lfs, so the pre-push hook works
export CLOUDSDK_CORE_PROJECT=${CLOUDSDK_CORE_PROJECT:-focus-surfer-494820-g0}
BUCKET=${BUCKET:-gs://focus-surfer-494820-g0-telemetry/tpu}
BRANCH=${BRANCH:-multinode-tpu-experiments}
INTERVAL=${INTERVAL:-1800}
OUT=${OUT:-data/tpu}
say(){ echo "[$(date -u +%H:%M:%SZ)] $*"; }

while :; do
  mapfile -t prefixes < <(gcloud storage ls "$BUCKET/" 2>/dev/null \
      | grep -aE "/camp[0-9]*_[0-9]{8}_[0-9]{6}[^/]*/$" || true)
  say "${#prefixes[@]} session prefix(es) found"
  for p in "${prefixes[@]}"; do
    # Only sessions that actually have runs/, and skip the probe/smoke ones.
    gcloud storage ls "${p}runs/" >/dev/null 2>&1 || continue
    timeout 900 python3 scripts/tpu_collection/ingest_tpu_corpus.py \
      --prefix "${p%/}" --out "$OUT" \
      --manifest "$OUT/../tpu_manifests/$(basename "${p%/}").csv" \
      >/tmp/autocommit_ingest.log 2>&1 \
      && say "  ingested $(basename "${p%/}"): $(grep -ac KEEP /tmp/autocommit_ingest.log) kept" \
      || say "  ingest FAILED for $(basename "${p%/}") (see /tmp/autocommit_ingest.log)"
  done

  n=$(git status --porcelain "$OUT" data/tpu_manifests 2>/dev/null | wc -l | tr -d ' ')
  if [ "${n:-0}" -gt 0 ]; then
    git add "$OUT" data/tpu_manifests 2>/dev/null
    streams=$(ls "$OUT"/*.parquet 2>/dev/null | wc -l | tr -d ' ')
    if git commit -q -m "TPU corpus snapshot: $streams stream(s)

Automated snapshot from scripts/tpu_collection/corpus_autocommit.sh. GCS holds the
authoritative per-cell uploads; this keeps a versioned copy of the ingested corpus
(hang tails trimmed, idle-span cells dropped) next to the code that produced it."; then
      say "committed $streams stream(s) ($n path(s) changed)"
      git push -q origin "$BRANCH" 2>/dev/null && say "pushed" || say "push failed (will retry next cycle)"
    else
      say "nothing to commit after add"
    fi
  else
    say "no corpus change"
  fi
  sleep "$INTERVAL"
done
