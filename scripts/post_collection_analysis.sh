#!/usr/bin/env bash
# post_collection_analysis.sh -- pull a finished collection run and re-measure.
#
# Turns the whole "did this round help?" loop into one command, because doing it
# by hand across a session boundary is how the 20260904 ingest got killed halfway
# and left five successful infermem cells outside the corpus for two days.
#
#   bash scripts/post_collection_analysis.sh 20260923 [20260923a]
#
# Steps, each skippable if already done:
#   1. disk guard      -- /home is a 4.8 GB disk that has hit 100% twice
#   2. pull            -- GCS session -> /var/tmp (the big volume), not /home
#   3. ingest          -- completed parquets -> data/gcp/<session>/, idle screened
#   4. window cache    -- REBUILT (not restored): the corpus changed
#   5. inventory       -- do the new families exist, with host counts > 1?
#   6. leave-one-family-out, with the tau sweep
#
# The headline is step 6: held-out training recall was 0.005 (train_fsdp) and
# 0.042 (train_ddp), and held-out infershard_qwen was falsely flagged on 71% of
# its windows. Success is those moving AT ONE SHARED THRESHOLD -- a recall gain
# bought by lowering tau until the false-alarm rate follows it is not a result.
set -uo pipefail
cd "$(git rev-parse --show-toplevel)"

SESSION="${1:?usage: post_collection_analysis.sh <session> [run_stamp]}"
STAMP="${2:-}"
BUCKET="${BUCKET:-gs://focus-surfer-494820-g0-telemetry}"
WORK="${SPAR_WORK:-/var/tmp/spar_analysis}"
PULL="${PULL_DIR:-/var/tmp/spar_pull/${SESSION}}"
MIN_HOME_MB="${MIN_HOME_MB:-800}"

say() { echo "[$(date -u +%FT%TZ)] $*"; }

# 1. Disk guard. The ingest copies into data/gcp/ on /home; running it with no
#    headroom fills the disk and then git cannot even write its index lock.
avail=$(df -Pm /home 2>/dev/null | awk 'NR==2{print $4}')
say "/home has ${avail:-?} MB free (need >= ${MIN_HOME_MB})"
if [ -n "$avail" ] && [ "$avail" -lt "$MIN_HOME_MB" ]; then
  echo "REFUSING: free space first. Biggest offloadable directories:" >&2
  du -sh data/gcp/* 2>/dev/null | sort -rh | head -5 >&2
  echo "  (archive an old session to ${BUCKET}/archive/ and delete it locally)" >&2
  exit 1
fi

# 2. Pull to the big volume.
mkdir -p "$PULL"
say "pulling ${BUCKET}/runs/${SESSION} -> $PULL"
gsutil -m rsync -r "${BUCKET}/runs/${SESSION}" "$PULL" 2>&1 | tail -2
say "pulled $(find "$PULL" -name '*.parquet' 2>/dev/null | wc -l) parquet(s)"

# 3. Ingest. Quarantines streams whose median utilisation is ~0, which is how a
#    cell whose peer died still writes full-length files full of zeros.
say "ingesting -> data/gcp/${SESSION}"
SESSION="$SESSION" SRC="$PULL" bash scripts/gcp/ingest_session.sh 2>&1 \
  | grep -E "copied|quarantined|partial summary|ingest complete" || true

# 3b. Which cells actually succeeded? A .done marker is the only thing that
#     vouches for a cell; parquets are written whether or not the workload worked.
if [ -n "$STAMP" ]; then
  say "cell outcomes for stamp ${STAMP}:"
  for node in spar-gpuf-0 spar-gpuf-1 spar-a100-0 spar-a100-1; do
    m=$(gsutil ls "${BUCKET}/runs/${SESSION}/${node}/logs/.done/${STAMP}/**" 2>/dev/null || true)
    [ -z "$m" ] && continue
    echo "  $node: $(echo "$m" | grep -c '\.done$') done, $(echo "$m" | grep -c '\.failed$') failed"
    echo "$m" | grep '\.failed$' | xargs -r -n1 basename 2>/dev/null | sed 's/^/      FAILED /' | head -20
  done
fi

# 4. Rebuild the window cache. SPAR_NO_CACHE_PULL=1 is essential: the builder
#    restores from GCS by default, and restoring a cache built before this round
#    would silently analyse the OLD corpus and report no change.
say "rebuilding the window cache (the corpus changed, so no restore)"
mkdir -p "$WORK"
rm -f "$WORK/win_full.parquet" "$WORK/raw_downcast.parquet"
SPAR_NO_CACHE_PULL=1 SPAR_WORK="$WORK" PYTHONPATH=. \
  python3 scripts/build_window_cache.py "$WORK/win_full.parquet" 30 15 3 2>&1 | tail -6

# 5. Inventory: did the new families land, and are they really multi-node?
say "inventory"
PYTHONPATH=. python3 scripts/multinode_inventory.py --out "$WORK/inv_mn.parquet" 2>&1 \
  | sed -n '1,6p;/--- TRAIN/,/^$/p'

# 6. The measurement this round exists for.
say "leave-one-family-out + tau sweep"
PYTHONPATH=. python3 classifier/rebuttal_analysis/mn_binary_lofo.py \
  --windows "$WORK/win_full.parquet" --inventory "$WORK/inv_mn.parquet" \
  --out "$WORK/mn_binary_lofo_${SESSION}.csv" 2>&1 | tail -45

cat <<'NOTE'

Compare against the 2026-09-22 baseline (classifier/rebuttal_analysis/results/):
  train_fsdp       0.0051 window / 0.0417 job
  train_ddp        0.0416 window / 0.1500 job
  infershard_qwen  0.2856 window (71% false alarms) / 0.1667 job
  families 12 (2 training)

A real improvement is held-out training recall UP and infershard_qwen's
false-alarm rate DOWN at the SAME tau. Read the tau table down a column.
NOTE
