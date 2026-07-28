#!/bin/bash
# Hourly snapshot pusher for the rebuttal collection queue.
#
# Commits whatever telemetry exists so far to a dedicated snapshot branch and
# pushes it, so a lost instance costs at most one hour of collection instead of
# an entire 8 h run. Pairs with the --checkpoint-sec snapshots written by
# scripts/collect_telemetry.py (default every 300 s), which are what make an
# in-progress run recoverable at all.
#
# What gets committed, and why it is cheap:
#   data/*.parquet.partial  in-progress telemetry, ~23 KB per 300 samples
#                           (~2.2 MB at 8 h). NOT matched by the *.parquet LFS
#                           rule, so these are ordinary blobs.
#   data/*.parquet          completed runs -> Git LFS (upload path verified).
#   logs/*.log             STEP logs, ~56 MB per 8 h stage. Committed RAW, not
#                           gzipped: they are append-only text, so successive
#                           versions delta-compress well in the packfile, while
#                           gzip snapshots would each be a fresh incompressible
#                           blob. Keeping them raw also means the re-scorer can
#                           read them straight out of the repo.
#   logs/.done/*            per-stage completion markers (queue resume state).
#
# Exits after a final push once the queue reports completion.

REPO=/root/GPU-monitoring
BRANCH=rebuttal-experiments-data
INTERVAL=3600
PUSHLOG=/root/rebuttal_push.log

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" >> "$PUSHLOG"; }

# GitHub HARD-rejects any single file over 100 MB, and an 8 h WB-I STEP log
# projects to ~93 MB — too thin a margin. Once a log crosses the threshold,
# commit it gzipped (~9x on this content) and stop tracking the raw version, so
# a rejected push can never take the remaining hourly snapshots down with it.
# The raw log stays on local disk for the re-scorer; only transport is gzipped.
LOG_GZ_THRESHOLD=$((40 * 1024 * 1024))

prepare_big_logs() {
  local f sz
  for f in logs/*.log; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f" 2>/dev/null) || continue
    [ "$sz" -gt "$LOG_GZ_THRESHOLD" ] || continue

    if gzip -c "$f" > "${f}.gz.tmp" 2>>"$PUSHLOG"; then
      mv -f "${f}.gz.tmp" "${f}.gz"
    else
      rm -f "${f}.gz.tmp"
      say "gzip failed for $f — leaving as-is"
      continue
    fi

    # Untrack the raw log (once) and keep `git add -A` from re-adding it.
    if git ls-files --error-unmatch "$f" >/dev/null 2>&1; then
      git rm -q --cached "$f" 2>>"$PUSHLOG"
      say "switched $f to gzipped transport ($(( sz / 1048576 )) MB raw)"
    fi
    grep -qxF "$f" .git/info/exclude 2>/dev/null || echo "$f" >> .git/info/exclude
  done
}

snapshot() {
  prepare_big_logs
  git add -A data logs 2>>"$PUSHLOG"

  if git diff --cached --quiet; then
    say "no changes to snapshot"
    return 0
  fi

  # Describe the snapshot: which stage is running, how much telemetry exists.
  local stage rows done_n fail_n
  stage=$(grep -E 'START|OK|FAIL' logs/queue_master.log 2>/dev/null | tail -1 | sed 's/^\[[^]]*\] //')
  done_n=$(ls logs/.done/*.done 2>/dev/null | wc -l)
  fail_n=$(ls logs/.done/*.failed 2>/dev/null | wc -l)
  rows=$(python3 - <<'PY' 2>/dev/null
import glob
import pyarrow.parquet as pq
tot = 0
for f in glob.glob('data/*.parquet.partial'):
    try:
        tot += pq.read_metadata(f).num_rows
    except Exception:
        pass
print(tot)
PY
)
  [ -z "$rows" ] && rows="?"

  git commit -q -m "snapshot $(stamp): ${rows} in-progress telemetry samples" \
    -m "queue: ${stage:-unknown}" \
    -m "stages complete: ${done_n}, failed: ${fail_n}" \
    -m "Automated hourly snapshot so partial results survive instance loss." \
    -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
    2>>"$PUSHLOG"

  if git push -q origin "$BRANCH" 2>>"$PUSHLOG"; then
    say "pushed $(git rev-parse --short HEAD) — ${rows} samples, ${done_n} stages done"
  else
    # Committed locally, so nothing is lost; the next cycle retries the push.
    say "PUSH FAILED for $(git rev-parse --short HEAD) — will retry next cycle"
  fi
}

say "===== pusher starting (pid $$), branch $BRANCH, interval ${INTERVAL}s ====="

# Establish the remote branch on the first cycle.
git push -q -u origin "$BRANCH" 2>>"$PUSHLOG" \
  && say "remote branch $BRANCH created" \
  || say "remote branch push deferred to first snapshot"

while true; do
  snapshot

  if grep -q '===== queue complete =====' logs/queue_master.log 2>/dev/null; then
    say "queue complete — final snapshot pushed, pusher exiting"
    exit 0
  fi

  sleep "$INTERVAL"
done
