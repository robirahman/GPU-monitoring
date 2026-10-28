#!/usr/bin/env python3
"""Write docs/quickstart_b200.md and docs/quickstart_h200.md.

One self-contained file per device: clone, install, verify, collect, check,
ship back. Generated from recollection_list.csv so the cell lists are the real
ones and cannot drift from the audit.
"""
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEVICES = {
    "B200": dict(
        file="quickstart_b200.md",
        smi="NVIDIA B200",
        reqs="requirements_blackwell.txt",
        stack_note=(
            "**This device needs a different PyTorch from the rest of the "
            "corpus.** `requirements_v2.txt` pins torch 2.5.1+cu121, whose "
            "wheels carry kernels for sm_50 through sm_90; B200 is sm_100, so "
            "that build cannot launch a kernel on it at all. "
            "`requirements_blackwell.txt` pins a CUDA 12.8 build instead. The "
            "preflight below proves the GPU can actually run a kernel before "
            "you spend an hour finding out it cannot."),
    ),
    "H200": dict(
        file="quickstart_h200.md",
        smi="NVIDIA H200",
        reqs="requirements_v2.txt",
        stack_note=(
            "H200 is Hopper (sm_90), so the corpus pin in "
            "`requirements_v2.txt` -- torch 2.5.1+cu121 -- runs on it "
            "unchanged. Use it: matching the stack the rest of the corpus was "
            "collected under is worth more than a newer torch."),
    ),
}

PREFLIGHT = '''```bash
# Refuses to go further unless the device is the right one AND can run a
# kernel. A Blackwell card with a cu121 wheel imports fine and fails only when
# something touches the GPU, which would otherwise be 40 minutes from now.
python - <<'PY'
import sys, torch
want = "%(smi)s"
assert torch.cuda.is_available(), "CUDA not available"
name = torch.cuda.get_device_name(0)
cap = torch.cuda.get_device_capability(0)
print(f"device      : {name}")
print(f"capability  : sm_{cap[0]}{cap[1]}")
print(f"torch       : {torch.__version__}")
print(f"torch cuda  : {torch.version.cuda}")
print(f"arch list   : {torch.cuda.get_arch_list()}")
if want not in name:
    sys.exit(f"FAIL: expected {want!r}, found {name!r}. The corpus keys on "
             f"gpu_name, so this would land as a different device.")
x = torch.randn(2048, 2048, device="cuda")
(x @ x).sum().item()          # the actual test: can it launch a kernel?
torch.cuda.synchronize()
print("preflight OK")
PY
```
'''


def main():
    d = pd.read_csv(ROOT / "docs/camera_ready_results_20261022/recollection_list.csv")
    d["collect_as"] = d.apply(
        lambda r: r.cell + "_mw8" if r.verdict == "starved" else r.cell, axis=1)

    from workloads.registry import WORKLOAD_REGISTRY as R

    for hw, cfg in DEVICES.items():
        g = d[d.hw.eq(hw)]
        cells = sorted(c for c in set(g.collect_as) if R.get(c))
        skipped = sorted(c for c in set(g.collect_as) if not R.get(c))
        secs = sum(float(g[g.collect_as.eq(c)].duration_s.iloc[0]) for c in cells) * 3

        t = [f"# Collecting the missing {hw} runs\n",
             f"Everything needed to fill the {hw} gap in the camera-ready "
             f"corpus, on a machine with one {hw}. "
             f"**{len(cells)} cells x 3 reps = {len(cells) * 3} runs, "
             f"about {secs / 3600:.1f} h of workload.**\n",
             f"One GPU is enough. None of these cells is a `torchrun` job -- "
             f"the only reason this cannot run on Google Cloud is that GCP "
             f"sells {hw} exclusively in 8-GPU nodes, and our quota request "
             f"for one was denied (support case 76886544).\n",
             "## 0. The stack\n", cfg["stack_note"] + "\n",
             "## 1. Clone\n",
             "```bash",
             "# LFS skip matters: the corpus is ~12k parquet payloads this box",
             "# never reads, and smudging them is a multi-GB download.",
             "GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \\",
             "    https://github.com/robirahman/GPU-monitoring.git",
             "cd GPU-monitoring",
             "```\n",
             "## 2. Install\n",
             "```bash",
             "python3 -m venv venv && . venv/bin/activate",
             "pip install --upgrade pip",
             f"pip install -r {cfg['reqs']}",
             "",
             "# CIFAR-10, so the first run does not spend its 300 s downloading",
             "python -c \"import torchvision; torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True)\"",
             "```\n",
             "## 3. Preflight -- do not skip\n",
             PREFLIGHT % {"smi": cfg["smi"]},
             "## 4. Collect\n",
             "```bash",
             "export STAMP=$(date -u +%Y%m%d)a",
             "mkdir -p data logs",
             "",
             f'CELLS="{" ".join(cells)}"',
             "for cell in $CELLS; do",
             "  for rep in 1 2 3; do",
             '    log="logs/${STAMP}__$(hostname -s)__${cell#adversarial_}_rep${rep}.log"',
             '    # idempotent: a cell with a clean log is skipped, so an',
             '    # interrupted session resumes instead of redoing.',
             '    grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue',
             '    python scripts/run_workload.py --workload "$cell" \\',
             '      --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"',
             "  done",
             "done",
             "```\n",
             "`--require-gpu-activity` exits 5 if a run aborts, stops short, or "
             "leaves the GPU idle. Every job this file exists to replace was "
             "produced by a run that exited cleanly and collected nothing, so "
             "let it fail loudly rather than filing another one.\n"]

        if skipped:
            t.append(f"`{'`, `'.join(skipped)}` is in the corpus but has no "
                     f"registry entry, so there is no command to reproduce it. "
                     f"It is deliberately not in the list above; it carries "
                     f"{int(g[g.collect_as.isin(skipped)].windows.sum())} "
                     f"windows.\n")

        t += ["## 5. Check before shipping\n",
              "```bash",
              "# Every cell here emits STEP lines. A run with steps=0 is the",
              "# exact failure this collection exists to undo -- do not ship it.",
              'for f in logs/${STAMP}__*.log; do',
              '  printf "%-52s steps=%-7s %s\\n" "$(basename "$f")" \\',
              '    "$(grep -c \'^STEP \' "$f")" "$(grep -o \'SPAR_CELL_DONE.*\' "$f")"',
              "done",
              "",
              "# and the collector's own verdict, written beside each parquet",
              'grep -h \'"verdict"\' data/*.health.json | sort | uniq -c',
              "```\n",
              f"Expected: {len(cells) * 3} logs, every one with a non-zero step "
              f"count and `exit=0`, and {len(cells) * 3} health verdicts all "
              "`ok`.\n",
              "## 6. Ship it back\n",
              "```bash",
              "git add data/*.parquet data/*.health.json logs/${STAMP}__*.log",
              f'git commit -m "Collect {hw} re-runs, stamp ${{STAMP}}"',
              "git push origin neurips-camera-ready",
              "```\n",
              "If pushing parquets through LFS from a rented box is awkward, "
              "put them in the bucket instead and the analysis box will ingest:\n",
              "```bash",
              "gcloud storage cp data/*.parquet \\",
              "  gs://focus-surfer-494820-g0-telemetry/runs/${STAMP}/$(hostname -s)/data/",
              "gcloud storage cp logs/${STAMP}__*.log \\",
              "  gs://focus-surfer-494820-g0-telemetry/runs/${STAMP}/$(hostname -s)/logs/",
              "```\n",
              "## Why five cells are named `_mw8`\n",
              "`adversarial_L_diluted_*` and `adversarial_K_online_learning` "
              "appear as `_mw8` variants. Re-running the originals just "
              "reproduces the fault: `num_workers=0` puts `Resize(224)` on one "
              "CPU thread and the GPU waits, which is how the archived runs "
              f"managed about four batches in 324 s with the {hw} busy in under "
              "2% of samples. The `_mw8` cells pass `--num-workers 8` and are "
              "GPU-bound.\n",
              "That makes them a **different workload**, which is why they "
              "carry a different label. Do not pool them with the "
              "`num_workers=0` runs, and do not delete the originals -- those "
              "are what the camera-ready measured.\n"]

        (ROOT / "docs" / cfg["file"]).write_text("\n".join(t) + "\n")
        print(f"wrote docs/{cfg['file']}  ({len(cells)} cells, "
              f"{len(cells) * 3} runs, {secs / 3600:.1f} h)")


if __name__ == "__main__":
    raise SystemExit(main())
