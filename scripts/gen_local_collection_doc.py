"""Generate docs/LOCAL_COLLECTION.md from the re-collection list."""
import pathlib
import sys

import pandas as pd

root = pathlib.Path(sys.argv[1])
d = pd.read_csv(root / "docs/camera_ready_results_20261022/recollection_list.csv")

# A starved cell must not be re-run as itself: the same command reproduces the
# starvation. Its _mw8 sibling is the one to collect.
def cells_for(group):
    out = []
    for _, r in group.sort_values(["verdict", "cell"]).iterrows():
        out.append((r.cell + "_mw8") if r.verdict == "starved" else r.cell)
    return sorted(set(out))

blocks = []
for hw, g in d.groupby("hw"):
    cells = cells_for(g)
    counts = g.verdict.value_counts().to_dict()
    why = ", ".join(f"{v} {k}" for k, v in sorted(counts.items()))
    blocks.append((hw, cells, why, len(cells) * 3))

STATUS = {
    "A100-SXM4-80GB": "The five starved cells were collected on GCP as `_mw8` on "
                      "2026-10-26 (stamp `20261026a`); the four aborted cells still "
                      "need a run.",
    "B200": "**Blocked on quota.** B200 has none in this project and increase "
            "requests for us-central1 and us-east4 were auto-denied on "
            "2026-10-26. Needs a support case, or a non-GCP provider.",
    "H200": "**Blocked on quota.** Same as B200: requests for us-central1 and "
            "us-east4 auto-denied on 2026-10-26.",
    "H100 NVL": "Not a GCP SKU. Whichever provider ran the originals.",
    "GeForce RTX 4050 Laptop GPU": "Local laptop. Every adversarial run on this "
                                   "device aborted within 6-7 s, so it currently "
                                   "has no usable adversarial coverage at all.",
}

lines = []
for hw, cells, why, runs in blocks:
    lines.append(f"### {hw}\n")
    if hw in STATUS:
        lines.append(STATUS[hw] + "\n")
    lines.append(f"{why}. **{len(cells)} cells x 3 reps = {runs} runs**, about "
                 f"{runs * 5.5 / 60:.1f} h.\n")
    lines.append("```bash")
    lines.append(f'CELLS="{" ".join(cells)}"')
    lines.append("for cell in $CELLS; do")
    lines.append("  for rep in 1 2 3; do")
    lines.append('    log="logs/${STAMP}__$(hostname -s)__${cell#adversarial_}_rep${rep}.log"')
    lines.append('    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue')
    lines.append('    python scripts/run_workload.py --workload "$cell" \\')
    lines.append('      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"')
    lines.append("  done")
    lines.append("done")
    lines.append("```\n")
body = "\n".join(lines)

doc = f'''# Collecting on a machine that is not in Google Cloud

For the GPUs this project cannot reach: the RTX 4050 laptop, the RTX 5080
desktop, H100 NVL and anything rented elsewhere. Everything here runs from a
clone of `neurips-camera-ready`.

The list it works from is
[`recollection_list.csv`](camera_ready_results_20261022/recollection_list.csv),
which `scripts/audit_run_health.py` regenerates. 46 tier-A adversarial jobs are
unusable: 19 `aborted` (stopped seconds in with the GPU never active),
12 `truncated` (stopped early with the GPU working) and 15 `starved` (ran to
length with the GPU idle in 99% of samples).

## One-time setup

```bash
# LFS skip matters: the corpus is ~12k parquet payloads you do not need in
# order to collect, and pulling them is a multi-GB download.
GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \\
    https://github.com/robirahman/GPU-monitoring.git
cd GPU-monitoring

python3 -m venv venv && . venv/bin/activate
pip install -r requirements_v2.txt          # pinned torch 2.5.1+cu121
# AMD instead:  pip install -r requirements_amd.txt
# Windows:      pip install -r requirements_amd_windows.txt

# CIFAR-10, so the first run does not spend its 300 s downloading
python -c "import torchvision; torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True)"

export STAMP=$(date -u +%Y%m%d)a
mkdir -p data logs
```

Check the GPU is visible before starting: `python -c "import torch;
print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"`. On AMD use
`scripts/run_workload_amd.py`, on Windows `scripts/run_workload_windows.py`;
the flags are the same.

## What to collect, by device

Each block is idempotent -- a cell that already has a clean log is skipped, so
re-running the block after an interruption resumes rather than redoing.

`--require-gpu-activity` is the point of the exercise. It exits 5 if the run
aborts, truncates, or leaves the GPU idle, which is exactly how all 46 of these
jobs were produced in the first place. Let it fail loudly rather than filing
another run that looks fine until someone counts the gradient steps six months
later.

{body}
## Why the starved cells are collected under a different name

`adversarial_L_diluted_2` and its siblings are in the list as `_mw8` variants,
not as themselves. Re-running the original command reproduces the starvation:
`num_workers=0` puts `Resize(224)` on one CPU thread and the GPU waits, and
the archived runs did about four batches in 350 s. The `_mw8` cells pass
`--num-workers 8` and are GPU-bound.

That makes them a **different workload**, which is why they carry a different
label -- `adversarial_round3.py` says so in its own docstring. Do not pool them
with the `num_workers=0` runs, and do not delete the originals: they are what
the camera-ready measured.

## Before you ship it back

```bash
# every run should report a non-zero step count and exit=0
for f in logs/${{STAMP}}__*.log; do
  printf "%-62s steps=%-7s %s\\n" "$(basename "$f")" \\
    "$(grep -c '^STEP ' "$f")" "$(grep -o 'SPAR_CELL_DONE.*' "$f")"
done
# and the collector's own verdict, written beside each parquet
cat data/*.health.json 2>/dev/null | grep -c '"verdict": "ok"'
```

A log with `steps=0` is the failure this whole exercise is about. Do not ship
it; find out why first.

## Shipping it back

Either push the branch:

```bash
git add data/*.parquet data/*.health.json logs/${{STAMP}}__*.log
git commit -m "Collect <device> re-runs, stamp ${{STAMP}}"
git push origin neurips-camera-ready
```

or, if you would rather not put parquets through LFS from a laptop, drop them
in the bucket and let the analysis box ingest:

```bash
gcloud storage cp data/*.parquet \\
  gs://focus-surfer-494820-g0-telemetry/runs/${{STAMP}}/$(hostname -s)/data/
gcloud storage cp logs/${{STAMP}}__*.log \\
  gs://focus-surfer-494820-g0-telemetry/runs/${{STAMP}}/$(hostname -s)/logs/
```

Then on the analysis box:

```bash
PYTHONPATH=. python3 scripts/ingest_rerun_references.py ${{STAMP}}
PYTHONPATH=. python3 scripts/audit_run_health.py          # re-verify, idempotent
```

`audit_run_health.py` leaves already-recorded exclusions alone, so re-running
it after a collection only adds what is new.
'''
(root / "docs/LOCAL_COLLECTION.md").write_text(doc)
print("wrote docs/LOCAL_COLLECTION.md")
for hw, cells, why, runs in blocks:
    print(f"  {hw}: {len(cells)} cells, {runs} runs ({why})")
