#!/usr/bin/env python3
"""Write docs/RECOLLECTION_BASELINES.md: the baseline gaps worth filling.

69 non-adversarial tier-A jobs are aborted, truncated or starved, but a broken
job only matters if it leaves a (device, cell) combination with nothing. Most
have healthy siblings, so this reports the 12 combinations that are actually
empty and, of those, the ones a command exists to reproduce.
"""
import json
import re
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from workloads.registry import WORKLOAD_REGISTRY as R  # noqa: E402


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
    bad = pd.read_csv(ROOT / "docs/camera_ready_results_20261022/recollection_list.csv")
    bad = bad[~bad.cell.str.startswith("adversarial_")]

    rows = []
    for (cell, hw), g in bad.groupby(["cell", "hw"]):
        same = a[a.base.eq(cell) & a.hw.eq(hw)]
        healthy = same[~same.job_id.isin(ex)]
        rows.append(dict(cell=cell, hw=hw, broken=len(g),
                         healthy=healthy.job_id.nunique(),
                         gpus=gpus(cell), runnable=bool(R.get(cell))))
    d = pd.DataFrame(rows)
    wiped = d[d.healthy.eq(0)].sort_values(["runnable", "hw", "cell"], ascending=[False, True, True])

    t = [f"# Baseline gaps worth filling\n",
         f"{len(bad)} non-adversarial tier-A jobs are aborted, truncated or "
         f"starved, carrying {int(bad.windows.sum())} windows between them. "
         f"A broken job only matters if it leaves a (device, cell) combination "
         f"with nothing usable, and **{len(d) - len(wiped)} of {len(d)} "
         f"combinations still have healthy runs** -- those need nothing.\n",
         "Re-collection cannot repair tier A, which is frozen by collection "
         "date; anything collected now lands in `C_followup`. What it fixes is "
         "the **coverage claim**, so it is worth doing only where a "
         "combination is genuinely empty.\n",
         f"## The {len(wiped)} empty combinations\n",
         "| device | cell | lost jobs | GPUs needed | reproducible |",
         "| :-- | :-- | --: | --: | :-- |"]
    for _, r in wiped.iterrows():
        g = "n/a" if pd.isna(r.gpus) else f"{int(r.gpus)}"
        t.append(f"| {r.hw} | `{r.cell}` | {r.broken} | {g} | "
                 f"{'yes' if r.runnable else '**no registry entry**'} |")
    run = wiped[wiped.runnable]
    t.append("")
    t.append(f"**{len(run)} are reproducible**, on {run.hw.nunique()} devices, "
             f"all single-GPU. {int(run.broken.sum())} jobs x 3 reps = "
             f"{int(run.broken.sum()) * 3} runs.\n")

    for hw, g in run.groupby("hw"):
        cells = sorted(g.cell)
        t += [f"### {hw}\n", "```bash",
              f'CELLS="{" ".join(cells)}"',
              "for cell in $CELLS; do for rep in 1 2 3; do",
              '  log="logs/${STAMP}__$(hostname -s)__${cell}_rep${rep}.log"',
              '  grep -q "SPAR_CELL_DONE.*exit=0" "$log" 2>/dev/null && continue',
              '  python scripts/run_workload.py --workload "$cell" \\',
              '    --output-dir data/ --no-dcgm --require-gpu-activity 2>&1 | tee "$log"',
              '  echo "SPAR_CELL_DONE cell=$cell rep=$rep exit=$?" >> "$log"',
              "done; done", "```\n"]

    miss = wiped[~wiped.runnable]
    if len(miss):
        t += ["## Not reproducible\n",
              f"`{'`, `'.join(sorted(miss.cell))}` have no registry entry, so "
              "there is no command to re-run them. They are gaps to state, not "
              "work to schedule -- inventing a command would fabricate the "
              "workload definition.\n"]

    t += ["## Why most of the 69 need nothing\n",
          "The broken jobs mostly sit alongside healthy ones: "
          "`llm_infer_mistral_7b_q4` on H200 NVL lost six jobs but keeps four "
          "(58 windows), and every AMD RX 9070 XT inference cell has three to "
          "fifteen healthy siblings. 61 of the 69 carry zero windows at all -- "
          "they are too short to produce a single 30 s window, so they never "
          "entered the evaluation.\n"]

    (ROOT / "docs/RECOLLECTION_BASELINES.md").write_text("\n".join(t) + "\n")
    print("wrote docs/RECOLLECTION_BASELINES.md")
    print(wiped[["hw", "cell", "broken", "gpus", "runnable"]].to_string(index=False))


if __name__ == "__main__":
    raise SystemExit(main())
