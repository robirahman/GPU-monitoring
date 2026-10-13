# Reproducing this snapshot

Use a volume outside the Cloud Shell home filesystem, with several GB free. The
[artifact manifest](artifacts/manifest.json) identifies the exact Cloud Storage
source/input archive and cache generations and their SHA-256 hashes. Download the
archive at that generation, verify its hash, and extract it into that volume. It
contains `repo/` with hydrated selected inputs and `results/` with the frozen cache
and its provenance sidecar. The archive is sufficient for the selected evaluation;
it is not a backup of every file in the expanded corpus.

The fits ran under Python 3.10.12 and the pinned core library versions in
[requirements.txt](artifacts/requirements.txt). Create an isolated environment on
the external volume. Matplotlib is required only for rendering the figure. The
58 regression tests ran under Python 3.12.3 with the same core library versions;
`validation.json` records the test environment separately.

From the extracted `repo/`, use its code and inputs to repeat the fits. Here
`EVAL_DIR` denotes an absolute directory on the external volume:

```bash
export EVAL_DIR=/mnt/persistent/followup
export PYTHONPATH=.
python classifier/rebuttal_analysis/paired_family_lofo.py \
  --windows "$EVAL_DIR/results/windows_30_15.parquet" \
  --exclude-campaigns 20261001,20261003 \
  --out "$EVAL_DIR/repeated/mechanism_paired.json"
python classifier/rebuttal_analysis/paired_family_lofo.py \
  --windows "$EVAL_DIR/results/windows_30_15.parquet" \
  --family-level backbone --out "$EVAL_DIR/repeated/backbone.json"
python classifier/rebuttal_analysis/paired_family_lofo.py \
  --windows "$EVAL_DIR/results/windows_30_15.parquet" \
  --steps-only --out "$EVAL_DIR/repeated/steps_only.json"
```

Use the review/report scripts from this result branch to check or render those
outputs; keep `PYTHONPATH` pointed at the extracted frozen repository. The saved
bundles can be audited without fitting again, using the command in the result
[report](README.md). Git LFS must hydrate the saved prediction parquets and
compressed split manifests first. The audit accepts compressed or uncompressed
split JSON.

All three evaluations use 300 trees, seed 0, 30-second windows, 15-second stride,
and threshold 0.5. The evaluator validates the content-bound cache before reuse.
Changing inputs, classifier source, library versions, or the label schema requires
a separately versioned cache/evaluation; do not bypass that validation to append
the recovered preempted captures to this snapshot.
