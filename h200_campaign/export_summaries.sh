#!/bin/bash
# Bundle per-run summaries (not raw waveforms) of every campaign session for analysis off the box.
#   export_summaries.sh [OUT.tgz]
OUT=${1:-$HOME/campaign/exports/summaries_$(date -u +%Y%m%dT%H%MZ).tgz}
mkdir -p "$(dirname "$OUT")"
cd ~/campaign/sessions || exit 1
for d in */*/; do [ -f "$d/events.jsonl" ] && [ ! -f "$d/events.jsonl.gz" ] && gzip -kf "$d/events.jsonl"; done
find . -maxdepth 3 -type f \( -name plan.json -o -name ledger.jsonl -o -name SESSION_COMPLETE -o -name manifest.json \
  -o -name validation.json -o -name "*metrics.json" -o -name health_intervals.csv -o -name alignment_edges.csv \
  -o -name events.jsonl.gz -o -name "c_*_meta.json" -o -name timeline.json \) > /tmp/h200_export_list.txt
tar czf "$OUT" -T /tmp/h200_export_list.txt && echo "$OUT"
