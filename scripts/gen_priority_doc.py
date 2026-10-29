#!/usr/bin/env python3
"""Rank the outstanding re-collections by tier-A windows recovered per run.

Counting jobs is the wrong metric. A cell whose healthy runs are too short to
produce a single 30 s window returns nothing however many times it is
re-collected, and several of the outstanding cells are exactly that. This
estimates the gain from how many windows a HEALTHY run of the same cell on the
SAME device actually yields, falling back to other devices only where the
combination is empty.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from workloads.registry import WORKLOAD_REGISTRY as R  # noqa: E402

SKIP = ("AMD",)  # not on Vast.ai and not in the NeurIPS corpus
SLUG = {"B200": "b200", "H200": "h200", "H200 NVL": "h200_nvl",
        "H100 80GB HBM3": "h100_80gb", "H100 NVL": "h100_nvl",
        "GeForce RTX 5080": "rtx5080", "A100-SXM4-80GB": "a100_80gb"}
NOTE = {
    "B200": "`adversarial_G_clock_throttled` calls `nvidia-smi -pl` and "
            "`--lock-gpu-clocks`; it needs a **privileged** container or it "
            "will skip the throttle and collect an undisguised run.",
    "A100-SXM4-80GB": "Running on GCP (`a2-ultragpu-8g`). "
                      "`fsdp_70b_nvlink` has no healthy A100 run anywhere in "
                      "the corpus and is expected to OOM -- AdamW fp32 state "
                      "for 70b is ~560 GB against 8x80 GB -- so treat a "
                      "failure as a hardware limit to state, not a collection "
                      "error.",
    "H200": "Two of the three cells are single-GPU and yield ~2 windows a run. "
            "The third, `fsdp_70b_nvlink`, needs an **8x H200** node for ~8 "
            "windows a run; renting eight GPUs for that is poor value unless "
            "you want the 70b point on H200 specifically.",
}


def gpus(cell):
    e = R.get(cell)
    if not e:
        return None
    for tok in (str(x) for x in e["command"]):
        m = re.match(r"--nproc_per_node=(\d+)$", tok)
        if m:
            return int(m[1])
    return 1


def main():
    man = pd.read_parquet(ROOT / "docs/camera_ready_manifest_20261022/streams.parquet")
    a = man[man.tier.eq("A_camera_ready")]
    ex = {x["job_id"] for x in
          json.loads((ROOT / "data/quality_exclusions.json").read_text())["excluded_jobs"]}
    healthy = a[~a.job_id.isin(ex)]
    GLOBAL = float(healthy.nominal_windows.median())

    d = pd.read_csv(ROOT / "docs/camera_ready_results_20261022/recollection_list.csv")
    d["runnable"] = d.cell.map(lambda c: bool(R.get(c)))

    def done(cell, hw):
        tok = re.sub(r"[^A-Za-z0-9]+", "_", hw)
        return any(list((ROOT / "data").rglob(f"{b}_*{tok}*_2026102*.parquet"))
                   for b in (cell, cell + "_mw8"))

    d["done"] = [done(c, h) for c, h in zip(d.cell, d.hw)]
    d["collect_as"] = [c + "_mw8" if v == "starved" and R.get(c + "_mw8") else c
                       for c, v in zip(d.cell, d.verdict)]
    todo = d[~d.done & d.runnable & ~d.hw.str.contains("|".join(SKIP))].copy()
    todo["gpus"] = todo.collect_as.map(gpus)

    per_device, per_cell = [], []
    for hw, g in todo.groupby("hw"):
        cells = sorted(set(g.collect_as))
        empty = gain = 0
        for c in cells:
            base = c[:-4] if c.endswith("_mw8") else c
            same = healthy[healthy.base.eq(base) & healthy.hw.eq(hw)]
            n = same.job_id.nunique()
            if n:
                per = float(same.groupby("job_id").nominal_windows.sum().median())
                src = "healthy runs on this device"
            else:
                empty += 1
                other = healthy[healthy.base.eq(base)]
                per = (float(other.groupby("job_id").nominal_windows.sum().median())
                       if len(other) else GLOBAL)
                src = "healthy runs on other devices"
            gain += 3 * per
            per_cell.append((hw, c, int(gpus(c) or 1), int(round(per)), src))
        per_device.append(dict(device=hw, cells=len(cells), runs=len(cells) * 3,
                               empty=empty, windows=int(round(gain)),
                               max_gpus=int(g.gpus.max())))

    p = pd.DataFrame(per_device).sort_values("windows", ascending=False)
    p["per_run"] = (p.windows / p.runs).round(1)

    t = ["# Re-collection priority\n",
         "Ranked by tier-A windows recovered, not by job count. A cell whose "
         "healthy runs are too short to fill a single 30 s window returns "
         "nothing however often it is re-collected, and several outstanding "
         "cells are exactly that -- `nbody_sim` and `resnet50_inference` on "
         "B200 both yield a median of **0 windows** per healthy run.\n",
         "AMD is excluded: not available on Vast.ai and not part of the "
         "NeurIPS corpus.\n",
         "| # | rent | cells | runs | est. windows | per run | empty combos |",
         "| --: | :-- | --: | --: | --: | --: | --: |"]
    for i, (_, r) in enumerate(p.iterrows(), 1):
        gpu = f" (+{r.max_gpus}x node)" if r.max_gpus > 1 else ""
        t.append(f"| {i} | 1x {r.device}{gpu} | {r.cells} | {r.runs} | "
                 f"**{r.windows}** | {r.per_run} | {r.empty} |")
    t += ["", f"**{int(p.runs.sum())} runs, ~{int(p.windows.sum())} tier-A "
              f"windows** across {len(p)} rentals.\n"]

    for _, r in p.iterrows():
        t.append(f"## {r.device}\n")
        if r.device in NOTE:
            t.append(NOTE[r.device] + "\n")
        slug = SLUG.get(r.device)
        if slug:
            t.append(f"Quickstart: [`docs/quickstart_{slug}.md`](quickstart_{slug}.md)\n")
        t += ["| cell | GPUs | est. windows per run | estimated from |",
              "| :-- | --: | --: | :-- |"]
        for hw, c, n, per, src in per_cell:
            if hw == r.device:
                t.append(f"| `{c}` | {n} | {per} | {src} |")
        t.append("")

    (ROOT / "docs/RECOLLECTION_PRIORITY.md").write_text("\n".join(t) + "\n")
    print("wrote docs/RECOLLECTION_PRIORITY.md")
    print(p.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
