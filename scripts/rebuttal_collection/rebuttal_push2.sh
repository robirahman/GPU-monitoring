#!/bin/bash
# Snapshot pusher v2 for the window-correlation experiment (queue v4).
#
# Why v2: v1 exited when it saw "===== queue complete =====" in
# queue_master.log. That string is now in the log twice (queues v2 and v3), so
# restarting v1 would match on its first pass and exit instantly. v2 instead
# tracks whether a queue PROCESS is alive, which is correct across queue
# generations, and requires two consecutive absences before exiting so it does
# not quit in the gap between phases.
#
# Interval is 30 min rather than 60: the remaining run is 4 h, so this bounds a
# crash to ~30 min of loss instead of an hour, and a partial parquet is ~1 MB.
#
# Commits (see v1 for the full rationale):
#   data/*.parquet.partial  in-progress telemetry, ordinary blobs (~23 KB/300 s)
#   data/*.parquet          completed runs -> Git LFS
#   logs/*.log              STEP logs, raw so successive versions delta well;
#                           gzipped only above 40 MB, since GitHub hard-rejects
#                           any single file over 100 MB
#   logs/.done/*            queue resume markers

REPO=/root/GPU-monitoring
BRANCH=rebuttal-experiments-data
INTERVAL=1800
PUSHLOG=/root/rebuttal_push.log
LOG_GZ_THRESHOLD=$((40 * 1024 * 1024))

cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" >> "$PUSHLOG"; }

queue_alive() {
  # This script's own cmdline is rebuttal_push2.sh, so it cannot self-match.
  pgrep -f 'rebuttal_queue[0-9]*\.sh' >/dev/null 2>&1
}

prepare_big_logs() {
  local f sz
  for f in logs/*.log; do
    [ -f "$f" ] || continue
    sz=$(stat -c%s "$f" 2>/dev/null) || continue
    [ "$sz" -gt "$LOG_GZ_THRESHOLD" ] || continue
    if gzip -c "$f" > "${f}.gz.tmp" 2>>"$PUSHLOG"; then
      mv -f "${f}.gz.tmp" "${f}.gz"
    else
      rm -f "${f}.gz.tmp"; say "gzip failed for $f — leaving as-is"; continue
    fi
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
  if git diff --cached --quiet; then say "no changes to snapshot"; return 0; fi

  local stage rows done_n
  stage=$(grep -E 'START|OK|FAIL|chosen' logs/queue_master.log 2>/dev/null | tail -1 | sed 's/^\[[^]]*\] //')
  done_n=$(ls logs/.done/*.done 2>/dev/null | wc -l)
  rows=$(python3 - <<'PY' 2>/dev/null
import glob
import pyarrow.parquet as pq
t = 0
for f in glob.glob('data/*.parquet.partial'):
    try: t += pq.read_metadata(f).num_rows
    except Exception: pass
print(t)
PY
)
  [ -z "$rows" ] && rows="?"

  git commit -q -m "snapshot $(stamp): ${rows} in-progress telemetry samples" \
    -m "queue: ${stage:-unknown}" \
    -m "stages complete: ${done_n}" \
    -m "Automated snapshot so partial results survive instance loss." \
    -m "Co-Authored-By: Claude Opus 5 (1M context) <noreply@anthropic.com>" \
    2>>"$PUSHLOG"

  if git push -q origin "$BRANCH" 2>>"$PUSHLOG"; then
    say "pushed $(git rev-parse --short HEAD) — ${rows} samples, ${done_n} stages done"
  else
    say "PUSH FAILED for $(git rev-parse --short HEAD) — retrying next cycle"
  fi
}

say "===== pusher v2 starting (pid $$), branch $BRANCH, interval ${INTERVAL}s ====="
snapshot                      # immediate first snapshot, don't wait a cycle

gone=0
while true; do
  sleep "$INTERVAL"
  snapshot
  if queue_alive; then
    gone=0
  else
    gone=$((gone + 1))
    say "no queue process detected (${gone}/2)"
    if [ $gone -ge 2 ]; then
      snapshot
      say "queue gone — final snapshot pushed, pusher exiting"
      exit 0
    fi
  fi
done
