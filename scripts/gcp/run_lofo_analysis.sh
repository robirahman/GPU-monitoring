#!/usr/bin/env bash
# Runs ON spar-analysis-0, detached. Rebuilds the multi-node window cache and
# re-runs leave-one-family-out, with the drop-one-round ablation.
#
# Detached on purpose: the operator's link is intermittent, and a dropped SSH
# must not kill a multi-hour build. Every phase announces itself and the whole
# log is tailed remotely; results are pushed to GCS so retrieving them needs no
# SSH at all.
set -uo pipefail
exec > >(tee -a /var/log/spar-lofo.log) 2>&1
REPO=/opt/spar/repo
WORK=/var/tmp/spar_analysis
BUCKET=gs://focus-surfer-494820-g0-telemetry
OUTDIR="$BUCKET/analysis/20261005"
BRANCH=multinode-tpu-experiments
say() { echo "[$(date -u +%H:%M:%SZ)] [lofo] $*"; }
die() { say "FATAL: $*"; echo "FAILED" > "$WORK/STATUS"; exit 1; }
mkdir -p "$WORK"
echo "RUNNING" > "$WORK/STATUS"

say "=== phase 1: repo ==="
cd "$REPO" || die "no repo at $REPO"
git config lfs.concurrenttransfers 8
ok=0
for i in 1 2 3; do
  git fetch --all --prune && git checkout "$BRANCH" && git pull --ff-only && { ok=1; break; }
  say "  git pull attempt $i failed; retrying"; sleep 20
done
[ "$ok" = 1 ] || die "git pull failed -- check credentials on this box"
say "  HEAD $(git log --oneline -1)"
ok=0
for i in 1 2 3; do git lfs pull && { ok=1; break; }; say "  lfs pull attempt $i failed"; sleep 30; done
[ "$ok" = 1 ] || die "git lfs pull failed"
say "  data/gcp sessions: $(ls data/gcp 2>/dev/null | tr '\n' ' ')"
say "  parquets on disk : $(find data -name '*.parquet' | wc -l)"

say "=== phase 2: round 20261003d from GCS (it never reached GitHub) ==="
# The operator's uplink cannot carry LFS uploads, so round d's 18 parquets are
# in GCS but not in the git history. Pull them straight from the bucket; the
# ingest screen is the same one every other session went through.
RID=4bc3b5ea8487
if ! ls data/gcp/20261003/*"$RID"* >/dev/null 2>&1; then
  ST="$REPO/gcs_runs/20261003"; rm -rf "$ST"
  for n in spar-spot-0 spar-spot-1; do
    mkdir -p "$ST/$n/data" "$ST/$n/logs"
    gcloud storage cp "$BUCKET/runs/20261003/$n/data/*${RID}*.parquet"         "$ST/$n/data/" >/dev/null 2>&1
    gcloud storage cp "$BUCKET/runs/20261003/$n/data/*${RID}*.parquet.partial" "$ST/$n/data/" >/dev/null 2>&1
    gcloud storage cp "$BUCKET/runs/20261003/$n/logs/*${RID}*.log"             "$ST/$n/logs/" >/dev/null 2>&1
  done
  say "  staged $(find "$ST" -name '*.parquet*' | wc -l) file(s)"
  SESSION=20261003 bash scripts/gcp/ingest_session.sh 2>&1 | tail -4
  rm -rf "$ST"
fi
say "  round d streams present: $(ls data/gcp/20261003/*"$RID"* 2>/dev/null | wc -l)"

say "=== phase 3: python env ==="
PY=""
for cand in "$REPO/venv-analysis/bin/python" /usr/bin/python3; do
  [ -x "$cand" ] || continue
  if "$cand" -c "import pandas,sklearn,pyarrow,numpy" 2>/dev/null; then PY="$cand"; break; fi
done
if [ -z "$PY" ]; then
  say "  building venv-analysis (no interpreter has the stack)"
  apt-get -o DPkg::Lock::Timeout=600 -y install python3-venv python3-pip >/dev/null 2>&1
  python3 -m venv "$REPO/venv-analysis" || die "venv creation failed"
  "$REPO/venv-analysis/bin/pip" install --upgrade pip -q
  "$REPO/venv-analysis/bin/pip" install -q pandas numpy scikit-learn pyarrow scipy \
    || die "pip install failed"
  PY="$REPO/venv-analysis/bin/python"
fi
say "  using $PY ($("$PY" -c 'import pandas,sklearn;print("pandas",pandas.__version__,"sklearn",sklearn.__version__)'))"

say "=== phase 4: corpus inventory ==="
PYTHONPATH="$REPO" "$PY" scripts/corpus_inventory.py --out "$WORK/inv4.parquet" 2>&1 | tail -6
[ -f "$WORK/inv4_dedup.parquet" ] || die "inventory did not produce inv4_dedup.parquet"

say "=== phase 5: multi-node window cache (30 s window, 15 s stride) ==="
say "  RAM: $(free -g | awk 'NR==2{print $2" GB total, "$7" avail"}')"
# build_mn_window_cache.py skips the expensive raw load when raw_mn.parquet
# already exists. /var/tmp survives reboots, so on 2026-10-06 it silently reused
# a RAW dated 2026-09-25 and built the whole cache from September's corpus: 34
# families, not one of the six new ones, and an ablation that dropped 0 windows
# because there was nothing of theirs to drop. The run looked clean and the
# numbers were a month old.
if [ "${SPAR_REUSE_RAW:-0}" != "1" ]; then
  for f in "$WORK"/raw_mn.parquet "$WORK"/mnpart_*.parquet; do
    [ -e "$f" ] && { say "  discarding stale $(basename "$f") ($(stat -c %y "$f" | cut -d. -f1))"; rm -f "$f"; }
  done
fi
PYTHONPATH="$REPO" SPAR_WORK="$WORK" "$PY" scripts/build_mn_window_cache.py \
  "$WORK/win_stream_30_15.parquet" 30 15 4 2>&1 | tail -20
[ -f "$WORK/win_stream_30_15.parquet" ] || die "window cache not produced"
say "  cache $(du -h "$WORK/win_stream_30_15.parquet" | cut -f1)"

say "=== phase 6: ablation cache (same windows minus the new rounds) ==="
# Separates "the rounds changed the corpus" from "the rounds changed the
# answer" -- the control used on 2026-09-30, after I once reported a recall
# drop as caused by a round when the baseline was a different cache.
PYTHONPATH="$REPO" "$PY" - "$WORK" "$REPO" <<'PYEOF'
import sys, os, re, glob
import pandas as pd
work, repo = sys.argv[1], sys.argv[2]
w = pd.read_parquet(os.path.join(work, "win_stream_30_15.parquet"))
print("  cache columns:", [c for c in w.columns if not c.startswith(("x_", "xr_"))][:14])

new_cells = set()
for s in ("20261001", "20261003"):
    for p in glob.glob(os.path.join(repo, "data/gcp", s, "*.parquet")):
        b = os.path.basename(p)
        b = re.sub(r"^nic_", "", b)
        b = re.sub(r"_node[01]_gpu[0-9]+.*", "", b)
        b = re.sub(r"_node[01]_[0-9a-f]{12}.*", "", b)
        if b.startswith("s16_"):
            new_cells.add(b)
print(f"  new-round cells: {len(new_cells)}")

# Match on the workload label, not the run_id: the 12-hex rid in the FILENAME
# is not what lands in the cache's run_id column, which is how the first
# attempt dropped nothing at all and still reported a clean ablation.
lab = "grp" if "grp" in w.columns else "workload_label"
hit = w[lab].astype(str).isin(new_cells)
print(f"  windows {len(w):,} -> {int((~hit).sum()):,} ({int(hit.sum()):,} dropped, on '{lab}')")
if int(hit.sum()) == 0:
    sys.exit("  FATAL: ablation dropped 0 windows -- it is not an ablation. "
             "Refusing to emit a 'control' identical to the full run.")
w[~hit].to_parquet(os.path.join(work, "win_noNEW.parquet"), index=False)
PYEOF
[ -f "$WORK/win_noNEW.parquet" ] || die "ablation cache not produced"

say "=== phase 7: LOFO, full corpus ==="
PYTHONPATH="$REPO" "$PY" classifier/rebuttal_analysis/multinode_family_holdout.py \
  --windows "$WORK/win_stream_30_15.parquet" --inventory "$WORK/inv4_dedup.parquet" \
  --out "$WORK/lofo_full_20261005.json" 2>&1 | tail -45

PYTHONPATH="$REPO" "$PY" - "$WORK" <<'PYEOF'
import json, sys, os
d = json.load(open(os.path.join(sys.argv[1], "lofo_full_20261005.json")))
fams = set(d["families"])
want = {"train_streamdiloco","train_decoupdiloco","train_swarm",
        "train_impala","train_apexrl","train_ppodist"}
missing = sorted(want - fams)
print(f"  families in run: {len(fams)}; six new present: {sorted(want & fams)}")
if missing:
    sys.exit(f"  FATAL: new families absent from the LOFO: {missing}. "
             f"The cache did not include the new sessions.")
PYEOF
[ $? -eq 0 ] || die "the new families never reached the LOFO"

say "=== phase 8: LOFO, ablation (minus 20261001+20261003) ==="
PYTHONPATH="$REPO" "$PY" classifier/rebuttal_analysis/multinode_family_holdout.py \
  --windows "$WORK/win_noNEW.parquet" --inventory "$WORK/inv4_dedup.parquet" \
  --out "$WORK/lofo_ablation_20261005.json" 2>&1 | tail -30

say "=== phase 9: publish ==="
gcloud storage cp "$WORK"/lofo_*_20261005.json "$OUTDIR/" >/dev/null 2>&1 \
  && say "  results -> $OUTDIR/" || say "  WARN: could not upload results"
gcloud storage cp /var/log/spar-lofo.log "$OUTDIR/spar-lofo.log" >/dev/null 2>&1 || true
echo "DONE" > "$WORK/STATUS"
say "=== COMPLETE ==="
