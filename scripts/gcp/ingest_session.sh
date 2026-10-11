#!/usr/bin/env bash
# ingest_session.sh -- fold a pulled session into the repo's analysis layout.
# Run from YOUR machine AFTER pull_results.sh.
#
#   bash scripts/gcp/ingest_session.sh
#
#  * completed parquets  ->  data/gcp/<SESSION>/   (auto-loaded by load_all_data's rglob)
#  * STEP logs           ->  logs/<SESSION>__<node>__<stage>.log  (joined by in-file "Run ID:")
#  * orphaned *.partial  ->  >=600 rows: kept as <base>_truncated.parquet ; else discarded
#
# The parquet filenames already carry _node{n}_gpu{i}_<run12> so they stay unique when
# flattened into one dir -- we copy, we do NOT concatenate (the loader concatenates).
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=env.sh
source "${SCRIPT_DIR}/env.sh"

REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
# SRC is overridable so a large pull can be staged somewhere with room. On this
# workstation /home is a 4.8 GB disk that has hit 100% once already, while the
# scratchpad lives on a 119 GB volume -- staging there and ingesting from it
# halves the peak footprint, since only the ingested copy lands on /home.
SRC="${SRC:-${REPO_ROOT}/gcs_runs/${SESSION}}"
DEST_DATA="${REPO_ROOT}/data/gcp/${SESSION}"
DEST_LOGS="${REPO_ROOT}/logs"
[[ -d "$SRC" ]] || { echo "no pulled session at $SRC -- run pull_results.sh first"; exit 1; }
mkdir -p "$DEST_DATA" "$DEST_LOGS"

PY="${VENV_MAIN}/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3)"

# --- 1. completed parquets ---------------------------------------------------
# Completion/progress evidence takes precedence over duty cycle. A quiet rank
# may be an actor or a sparse learner, not a failed launch. Preserve uncertain
# captures in their own quarantine for review; never relabel them as idle.
spar_say "copying completed parquets with progress-aware quality screening"
PYTHONPATH="$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}" "$PY" - "$SRC" "$DEST_DATA" "${MIN_UTIL_MED:-1}" <<'PYEOF'
import csv, glob, os, re, shutil, sys
from pathlib import Path
import pandas as pd
from classifier.collection_quality import assess_collection
from classifier.data_sources import data_role
src, dest, threshold = sys.argv[1], sys.argv[2], float(sys.argv[3])
records = []
for f in sorted(Path(src).rglob("*.parquet")):
    base = f.name
    status, reason = "accepted", "sidecar"
    if data_role(f) == "accelerator":
        d = pd.read_parquet(f)
        median = float(d.gpu_utilization_pct.median()) if "gpu_utilization_pct" in d else None
        ids = d.run_id.dropna().astype(str).unique() if "run_id" in d else []
        # Match within the same staged host, never borrow a peer's completion.
        host_root = f.parent.parent
        logs = []
        for log in (host_root / "logs").glob("*.log"):
            if ".nic." in log.name or not any(rid in log.name for rid in ids):
                continue
            logs.append(log.read_text(errors="replace"))
        status, reason = assess_collection(base, median, "\n".join(logs), threshold)
    target = Path(dest) if status == "accepted" else Path(dest) / ("_quarantine_" + status)
    target.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(f, target / base)
    records.append(dict(file=base, disposition=status, evidence=reason))
with open(Path(dest) / "collection_quality.csv", "w", newline="") as out:
    writer = csv.DictWriter(out, ["file", "disposition", "evidence"])
    writer.writeheader(); writer.writerows(records)
print(pd.Series([r["disposition"] for r in records]).value_counts().to_string())
PYEOF

# --- 2. STEP logs ------------------------------------------------------------
spar_say "copying STEP logs -> ${DEST_LOGS}/"
NL=0
while IFS= read -r f; do
  [[ -z "$f" ]] && continue
  # .../gcs_runs/<SESSION>/<node>/logs/<stage>.log  -> derive node + stage
  node="$(basename "$(dirname "$(dirname "$f")")")"
  stage="$(basename "$f" .log)"
  cp -f "$f" "${DEST_LOGS}/${SESSION}__${node}__${stage}.log"
  NL=$((NL+1))
done < <(find "$SRC" -type f -path '*/logs/*.log' ! -path '*/smoke_data/*' 2>/dev/null)
echo "  copied $NL STEP log(s)"

# --- 3. orphaned *.partial ---------------------------------------------------
spar_say "resolving orphaned *.partial checkpoints (keep >=600 rows as *_truncated.parquet)"
"$PY" - "$SRC" "$DEST_DATA" <<'PY'
import sys, os, glob, shutil
src, dest = sys.argv[1], sys.argv[2]
try:
    import pyarrow.parquet as pq
except Exception as e:
    print(f"  pyarrow unavailable ({e}); leaving .partial files untouched")
    sys.exit(0)
kept = discarded = bad = 0
for p in glob.glob(os.path.join(src, "**", "*.parquet.partial"), recursive=True):
    try:
        nrows = pq.ParquetFile(p).metadata.num_rows
    except Exception as e:
        # a .partial can be a half-written footer; treat as unreadable/too-short
        print(f"  UNREADABLE {os.path.basename(p)} ({e}) -> discard")
        bad += 1
        continue
    if nrows >= 600:
        base = os.path.basename(p)[:-len(".parquet.partial")]  # strip suffix
        partial_dir = os.path.join(dest, "_quarantine_partial")
        os.makedirs(partial_dir, exist_ok=True)
        out = os.path.join(partial_dir, base + "_truncated.parquet")
        shutil.copyfile(p, out)
        print(f"  KEPT {os.path.basename(p)} ({nrows} rows) -> {os.path.basename(out)}")
        kept += 1
    else:
        print(f"  DISCARD {os.path.basename(p)} ({nrows} rows < 600)")
        discarded += 1
print(f"  partial summary: kept={kept} discarded={discarded} unreadable={bad}")
PY

echo
spar_say "ingest complete for session ${SESSION}"
echo "  data  -> ${DEST_DATA}   ($(find "$DEST_DATA" -name '*.parquet' | wc -l | tr -d ' ') parquet files)"
echo "  logs  -> ${DEST_LOGS}/${SESSION}__*.log"
echo "next:  python classifier/rebuttal_analysis/scope_ladder_eval.py"
