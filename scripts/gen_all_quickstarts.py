#!/usr/bin/env python3
"""Write one quickstart per instance type for the tier-A runs still owed.

Since 2026-10-29 a re-collection can be promoted into tier A when it is the
same cell on the same gpu_name (data/tier_a_recollections.json), so a broken
job is now worth replacing even where healthy siblings exist: the replacement
adds its windows back to the frozen corpus rather than only to coverage
claims.

Grouped so one rental covers one device. Cells needing more than one GPU are
split into their own block, because they decide the instance size.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from workloads.registry import WORKLOAD_REGISTRY as R  # noqa: E402

# device -> (slug, what to rent, extra notes)
# What `torch.cuda.get_device_name(0)` must contain. Checked in full rather
# than by last token: "NVL" alone matches H100 NVL and H200 NVL both, and the
# corpus keys on gpu_name, so landing on the wrong one silently creates a
# different device.
EXPECT = {
    "GeForce RTX 4050 Laptop GPU": "RTX 4050",
    "GeForce RTX 5080": "RTX 5080",
    "H100 80GB HBM3": "H100 80GB",
    "H100 NVL": "H100 NVL",
    "H200": "H200",
    "H200 NVL": "H200 NVL",
    "B200": "B200",
    "A100-SXM4-80GB": "A100-SXM4-80GB",
    "AMD RX 9070 XT": "9070",
    "AMD Instinct MI300X VF": "MI300X",
}

RENT = {
    "GeForce RTX 4050 Laptop GPU": ("rtx4050", "the laptop you already have", ""),
    "GeForce RTX 5080": ("rtx5080", "1x RTX 5080 (Vast.ai)",
        "Prefer driver **581.95**, the dominant archived 5080 driver. "
        "`phi3_mini` is the only LLM that failed on this device -- 48 of 51 "
        "other inference jobs ran fine on the same box and driver -- so "
        "confirm the model loads before committing to the full run."),
    "H100 80GB HBM3": ("h100_80gb", "1x H100 SXM 80GB", ""),
    "H100 NVL": ("h100_nvl", "1x H100 **NVL** (94GB PCIe)",
        "NVL specifically, not SXM 80GB: they report different `gpu_name` and "
        "would land as different devices."),
    "H200": ("h200", "1x H200 141GB, plus 8x H200 for the FSDP cells", ""),
    "H200 NVL": ("h200_nvl", "1x H200 **NVL**",
        "Distinct from plain `H200` in this corpus -- both exist as separate "
        "devices."),
    "B200": ("b200", "1x B200",
        "Needs `requirements_blackwell.txt` (cu128). The repo's default pin "
        "is torch 2.5.1+cu121, which carries no sm_100 kernels.\n\n"
        "`adversarial_G_clock_throttled` calls `nvidia-smi -pl` and "
        "`--lock-gpu-clocks`, which need privileged access. It was skipped on "
        "the last B200 rental for exactly that reason. Take a bare-metal or "
        "privileged instance if you want it; otherwise expect it to fail and "
        "leave the other three cells unaffected."),
    "A100-SXM4-80GB": ("a100_80gb", "1x A100 80GB, plus 8x A100 80GB for the FSDP cells",
        "Reachable on GCP: this project holds NVIDIA_A100_80GB_GPUS=16 in "
        "four regions."),
    "AMD RX 9070 XT": ("rx9070xt", "1x AMD RX 9070 XT",
        "Use `requirements_amd.txt` and `scripts/run_workload_amd.py`."),
    "AMD Instinct MI300X VF": ("mi300x", "1x MI300X",
        "Use `requirements_amd.txt` and `scripts/run_workload_amd.py`."),
}


# AMD is out of scope: not available on Vast.ai, and the RX 9070 XT and
# MI300X runs are not part of the NeurIPS corpus the camera-ready reports.
# `unknown` has no device to rent. Dropped rather than deleted, so re-adding
# them later is one edit.
SKIP_DEVICES = {"AMD RX 9070 XT", "AMD Instinct MI300X VF", "unknown"}


def gpus(cell):
    e = R.get(cell)
    if not e:
        return None
    for tok in (str(x) for x in e["command"]):
        m = re.match(r"--nproc_per_node=(\d+)$", tok)
        if m:
            return int(m[1])
    return 1


# Anything collected on or after this date is a re-collection rather than part
# of the frozen corpus. Tier A closed 2026-06-18; the campaign began 2026-10-23.
RECOLLECT_FROM = 1792000000.0  # 2026-10-20 UTC

_RECOLLECTED = None


def _manifest():
    return pd.read_parquet(ROOT / "docs/camera_ready_manifest_20261022/streams.parquet")


def already_done(cell, hw):
    """Has this cell already been re-collected on this device?

    Asked of the manifest, not of filenames. Multi-GPU parquets are written as
    `<cell>_node0_gpu<N>_<DEVICE>_<run_id>.parquet` with no date in the name,
    so the date-glob this replaces reported every one of them as outstanding --
    including H200 fsdp_70b_nvlink the day after it was collected, which would
    have sent someone to re-rent an 8-GPU node for finished work.
    """
    global _RECOLLECTED
    if _RECOLLECTED is None:
        m = _manifest()
        fresh = m[m.t0 >= RECOLLECT_FROM]
        _RECOLLECTED = {(str(b), str(h)) for b, h in zip(fresh.base, fresh.hw)}
    return (cell, hw) in _RECOLLECTED or (cell + "_mw8", hw) in _RECOLLECTED


def done_jobs(cell, hw):
    """(name_as_collected, n_jobs) for a cell already re-collected on hw."""
    m = _manifest()
    for base in (cell, cell + "_mw8"):
        g = m[(m.base == base) & (m.hw == hw) & (m.t0 >= RECOLLECT_FROM)]
        if len(g):
            return base, int(g.job_id.nunique())
    return None


def main():
    d = pd.read_csv(ROOT / "docs/camera_ready_results_20261022/recollection_list.csv")
    d["gpus"] = d.cell.map(gpus)
    d["runnable"] = d.cell.map(lambda c: bool(R.get(c)))
    d["done"] = [already_done(c, h) for c, h in zip(d.cell, d.hw)]
    # starved cells are collected under their worker-fed name
    d["collect_as"] = [c + "_mw8" if v == "starved" and R.get(c + "_mw8") else c
                       for c, v in zip(d.cell, d.verdict)]
    todo = d[~d.done & d.runnable & ~d.hw.isin(SKIP_DEVICES)].copy()

    summary = []
    for hw, g in todo.groupby("hw"):
        if hw not in RENT:
            continue
        slug, rent, note = RENT[hw]
        single = sorted({c for c, n in zip(g.collect_as, g.gpus) if n == 1})
        multi = sorted({(c, int(n)) for c, n in zip(g.collect_as, g.gpus) if n and n > 1})
        runs = (len(single) + len(multi)) * 3
        summary.append((hw, rent, len(single), len(multi), runs))

        t = [f"# Collecting the missing {hw} runs\n",
             f"**Rent: {rent}.** {len(single) + len(multi)} cells x 3 reps = "
             f"{runs} runs.\n"]
        if note:
            t.append(note + "\n")
        if single and multi:
            n = max(x[1] for x in multi)
            t.append(f"**One rental covers everything here.** Rent the "
                     f"{n}x node and run the single-GPU cells on it too -- "
                     f"they take device 0 and leave the rest idle, which "
                     f"costs nothing extra and saves a second rental.\n")
        t += ["These replace tier-A jobs that aborted, stopped short, or ran "
              "with the GPU idle. Under the policy recorded in "
              "`data/tier_a_recollections.json`, a re-collection of the same "
              "cell on the same `gpu_name` can be promoted back into tier A, "
              "so this is repairing the frozen corpus and not only its "
              "coverage claims.\n",
              "## 1. Clone and install\n", "```bash",
              "GIT_LFS_SKIP_SMUDGE=1 git clone -b neurips-camera-ready \\",
              "    https://github.com/robirahman/GPU-monitoring.git",
              "cd GPU-monitoring",
              "python3 -m venv venv && . venv/bin/activate",
              "pip install --upgrade pip"]
        if "AMD" in hw:
            t.append("pip install -r requirements_amd.txt")
        elif hw == "B200":
            t.append("pip install -r requirements_blackwell.txt   # cu128, sm_100")
        else:
            t.append("pip install -r requirements_v2.txt")
        # Any quantized inference cell needs a GPTQ backend that no base
        # requirements file declares; see requirements_gptq.txt.
        if any("mistral_7b_q4" in c for c in list(single) + [c for c, _ in multi]):
            t.append("pip install transformers accelerate sentencepiece  # llm_infer_* cells")
        t += ["pip install psutil   # so the run records its CPU, cores and RAM",
              "python -c \"import torchvision; torchvision.datasets.CIFAR10('data/cifar10', train=False, download=True)\"",
              "",
              "# Do NOT add requirements_gptq.txt to this venv. On 2026-11-02 it",
              "# pulled torch 2.14.1 over the pinned 2.5.1 and gptqmodel still did",
              "# not import. The quantized cell runs without it -- see the note",
              "# at the end of this file if it ever does not.",
              "export STAMP=$(date -u +%Y%m%d)a",
              "mkdir -p data logs", "```\n",
              "## 2. Preflight\n", "```bash",
              "python - <<'PY'",
              "import sys, torch",
              f"want = {EXPECT[hw]!r}",
              f"forbid = {'H200' if hw == 'H100 NVL' else ('H100' if hw == 'H200 NVL' else '')!r}",
              "assert torch.cuda.is_available(), 'CUDA not available'",
              "name = torch.cuda.get_device_name(0)",
              "print('device :', name)",
              "print('torch  :', torch.__version__, '| cuda', torch.version.cuda)",
              "print('arch   :', torch.cuda.get_arch_list())",
              "if want not in name or (forbid and forbid in name):",
              "    sys.exit(f'FAIL: need a device named like {want!r}, got {name!r}')",
              "x = torch.randn(2048, 2048, device='cuda'); (x @ x).sum().item()",
              "torch.cuda.synchronize(); print('preflight OK')",
              "PY", "```\n"]

        runner = ("scripts/run_workload_amd.py" if "AMD" in hw
                  else "scripts/run_workload.py")
        if single:
            t += [f"## 3. The {len(single)} single-GPU cells\n", "```bash",
                  f'CELLS="{" ".join(single)}"',
                  "for cell in $CELLS; do for rep in 1 2 3; do",
                  '  log="logs/${STAMP}__$(hostname -s)__${cell#adversarial_}_rep${rep}.log"',
                  '  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue',
                  f'  python {runner} --workload "$cell" \\',
                  '    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"',
                  '  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=${PIPESTATUS[0]}" >> "$log"',
                  "done; done", "```\n"]
        if multi:
            n = max(x[1] for x in multi)
            t += [f"## 4. The multi-GPU cells -- these need a {n}x {hw} node\n",
                  "```bash",
                  f'CELLS="{" ".join(c for c, _ in multi)}"',
                  "for cell in $CELLS; do for rep in 1 2 3; do",
                  '  log="logs/${STAMP}__$(hostname -s)__${cell}_rep${rep}.log"',
                  '  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue',
                  f'  python scripts/run_workload_multi_gpu.py --workload "$cell" --num-gpus {n} \\',
                  '    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"',
                  '  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=${PIPESTATUS[0]}" >> "$log"',
                  "done; done", "```\n"]

        finished = []
        for cell in sorted(set(d[d.hw.eq(hw)].collect_as)):
            base = cell[:-4] if cell.endswith("_mw8") else cell
            info = done_jobs(base, hw)
            if info:
                finished.append(info)
        if finished:
            t += [f"## Already collected on {hw} -- do not re-run these\n",
                  "| cell | runs already collected |", "| :-- | --: |"]
            for base, jobs in sorted(set(finished)):
                t.append(f"| `{base}` | {jobs} |")
            t.append("")
        cannot = sorted(set(d[d.hw.eq(hw) & ~d.runnable].cell))
        if cannot:
            t += [f"## Cannot be reproduced on {hw}\n",
                  f"`{'`, `'.join(cannot)}` have no registry entry, so there "
                  f"is no command to run them. Gaps to state in the paper, "
                  f"not work for this rental.\n"]

        t += ["## 5. Check, then ship\n", "```bash",
              'for f in logs/${STAMP}__*.log; do',
              '  printf "%-52s steps=%-7s %s\\n" "$(basename "$f")" \\',
              '    "$(grep -c \'^STEP \' "$f")" "$(grep -o \'SPAR_CELL_DONE.*\' "$f")"',
              "done",
              'grep -h \'"verdict"\' data/*.health.json | sort | uniq -c',
              "",
              "git add data/*.parquet data/*.health.json logs/${STAMP}__*.log",
              f'git commit -m "Collect {hw} re-runs, stamp ${{STAMP}}"',
              "git push origin neurips-camera-ready", "```\n",
              "Every health verdict should read `ok`. Not every cell emits "
              "STEP lines -- only the instrumented adversarial strategies do "
              "-- so `steps=0` on an inference or baseline cell is expected, "
              "but `verdict` other than `ok` is not.\n"]

        (ROOT / "docs" / f"quickstart_{slug}.md").write_text("\n".join(t) + "\n")

    # Expected windows restored, from the healthy siblings of each cell.
    man = pd.read_parquet(ROOT / "docs/camera_ready_manifest_20261022/streams.parquet")
    tier_a = man[man.tier.eq("A_camera_ready")]
    ex = {x["job_id"] for x in json.loads(
        (ROOT / "data/quality_exclusions.json").read_text())["excluded_jobs"]}
    yields = {}
    for (cell, hw), g in tier_a.groupby(["base", "hw"]):
        h = g[~g.job_id.isin(ex)]
        if len(h):
            yields[(cell, hw)] = float(h.groupby("job_id").nominal_windows.sum().mean())

    index = []
    for hw, g in todo.groupby("hw"):
        if hw not in RENT:
            continue
        cells = sorted(set(g.collect_as))
        win = 0.0
        for c in cells:
            base = c[:-4] if c.endswith("_mw8") else c
            y = yields.get((base, hw))
            if y is None:
                ys = [v for (cc, _), v in yields.items() if cc == base]
                y = sum(ys) / len(ys) if ys else 19.0
            win += y * len(g[g.collect_as.eq(c)])
        mg = sorted({int(n) for n in g.gpus.dropna() if n > 1})
        index.append(dict(device=hw, rent=RENT[hw][1], slug=RENT[hw][0],
                          cells=len(cells), runs=len(cells) * 3,
                          windows=int(round(win)), multi=mg[0] if mg else 1))
    idx = pd.DataFrame(index).sort_values("windows", ascending=False)

    lines = ["# Which rentals to take, in order\n",
             "Ranked by the tier-A windows each one restores, estimated from "
             "the healthy siblings of the same cell on the same device. "
             "Re-collections can be promoted back into tier A "
             "(`data/tier_a_recollections.json`), so these windows return to "
             "the frozen corpus rather than only to coverage claims.\n",
             "| # | rent | cells | runs | windows restored | quickstart |",
             "| --: | :-- | --: | --: | --: | :-- |"]
    for i, (_, r) in enumerate(idx.iterrows(), 1):
        lines.append(f"| {i} | {r.rent} | {r.cells} | {r.runs} | "
                     f"~{r.windows} | `quickstart_{r.slug}.md` |")
    lines += ["",
              f"**{int(idx.runs.sum())} runs, ~{int(idx.windows.sum())} windows, "
              f"{len(idx)} rentals.**\n",
              "Where a device has both single- and multi-GPU cells, one "
              "multi-GPU rental covers both: a 1-GPU cell runs on device 0 and "
              "leaves the rest idle.\n",
              "The bottom of this list earns its place poorly. H100 NVL is one "
              "cell worth about four windows -- a whole rental for a rounding "
              "error. Worth stating as a gap instead unless the rental is "
              "nearly free.\n",
              "Not reproducible at all: `baseline_infer`, `baseline_train`, "
              "`ffmpeg_vaapi`, `nvlink_latency`, `nvlink_p2p` have no registry "
              "entry, so there is no command to re-run them.\n"]
    (ROOT / "docs/RENTAL_PRIORITY.md").write_text("\n".join(lines) + "\n")
    print("wrote docs/RENTAL_PRIORITY.md")
    print(idx[["device", "cells", "runs", "windows"]].to_string(index=False))

    s = pd.DataFrame(summary, columns=["device", "rent", "1-GPU cells",
                                       "multi-GPU cells", "runs"])
    s = s.sort_values("runs", ascending=False)
    print(s.to_string(index=False))
    print("\ntotal runs:", int(s.runs.sum()), " rentals:", len(s))
    skipped = d[~d.runnable]
    if len(skipped):
        print("\nnot reproducible (no registry entry):",
              ", ".join(sorted(set(skipped.cell))))
    done = d[d.done]
    print("already re-collected:", done.job_id.nunique(), "jobs")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
