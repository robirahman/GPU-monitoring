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
SRC="${REPO_ROOT}/gcs_runs/${SESSION}"
DEST_DATA="${REPO_ROOT}/data/gcp/${SESSION}"
DEST_LOGS="${REPO_ROOT}/logs"
[[ -d "$SRC" ]] || { echo "no pulled session at $SRC -- run pull_results.sh first"; exit 1; }
mkdir -p "$DEST_DATA" "$DEST_LOGS"

PY="${VENV_MAIN}/bin/python"; [[ -x "$PY" ]] || PY="$(command -v python3)"

# --- 1. completed parquets ---------------------------------------------------
# QUARANTINE IDLE STREAMS. The telemetry collector starts before the workload
# and records for the full duration whether or not the workload ever runs, so a
# multi-node cell whose PEER died still produces full-length, correctly-named
# parquets full of zeros. Measured on 2026-09-08: spar-a100-1 was preempted 13
# minutes into a run and spar-a100-0 went on to write 64 parquets labelled
# s16_advtrain_pipe1b_mb{1,4,16} with util_med=0.0 and mem_med=519 MB -- a bare
# CUDA context. Copying those in would have put idle windows into the corpus
# under an ADVERSARIAL training label, which is precisely the contamination the
# TPU ingest already screens for with --min-util-med.
#
# MIN_UTIL_MED=0 disables the screen; `idle`-labelled cells are exempt because
# being idle is the point of them.
spar_say "copying completed parquets -> ${DEST_DATA}/ (idle screen: median util > ${MIN_UTIL_MED:-1}%)"
"$PY" - "$SRC" "$DEST_DATA" "${MIN_UTIL_MED:-1}" <<'PYEOF'
import os, shutil, sys, glob
import pandas as pd

src, dest, thresh = sys.argv[1], sys.argv[2], float(sys.argv[3])
quar = os.path.join(dest, "_quarantine_idle")
kept = dropped = 0
for f in sorted(glob.glob(os.path.join(src, "**", "*.parquet"), recursive=True)):
    base = os.path.basename(f)
    if base.endswith(".partial"):
        continue
    if thresh > 0 and not base.startswith(("idle", "nic_")):
        try:
            u = pd.read_parquet(f, columns=["gpu_utilization_pct"])
            med = float(u["gpu_utilization_pct"].median())
        except Exception:
            med = None
        if med is not None and med <= thresh:
            os.makedirs(quar, exist_ok=True)
            shutil.copyfile(f, os.path.join(quar, base))
            print(f"  QUARANTINE {base[:70]} (median util {med:.1f}% <= {thresh}%)")
            dropped += 1
            continue
    shutil.copyfile(f, os.path.join(dest, base))
    kept += 1
print(f"  copied {kept} completed parquet(s); "
      f"quarantined {dropped} idle stream(s) -> {quar if dropped else '(none)'}")
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
        out = os.path.join(dest, base + "_truncated.parquet")
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
