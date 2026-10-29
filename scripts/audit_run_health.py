#!/usr/bin/env python3
"""Find tier-A runs that never ran, or ran without using the GPU.

Two failure modes hide in this corpus, and both look like completed runs from
the outside -- a telemetry file, a clean exit, a row in the manifest:

  aborted: the run stops a long way short of its `--duration` with the GPU
           never once registering utilisation. A 6-32 s run yields 0-1 windows,
           so it is absent from the evaluation rather than wrong in it, but the
           cell still appears in coverage tables as though it had been measured.

  truncated: stops just as short, but the GPU was working the whole time. Two
           `adversarial_training` jobs on B200 ran at 98% busy and stopped at
           20 s of 300. Same consequence -- no usable windows -- but a
           different cause, so it is worth re-collecting separately rather
           than filed under "never ran".

  starved: the run lasts its full duration while the GPU is idle in almost
           every sample. These DO produce windows -- 20 apiece -- and those
           windows are labelled as adversarial training while the device did
           essentially nothing. That is the one that moves published numbers.

`busy` is the fraction of samples with any utilisation at all, not the mean.
Mean separates these badly: `E_memory_minimal` averages 2.6% utilisation by
design while being busy in 91% of its samples, and `L_diluted_2` averages 0.9%
while being busy in 2%. The first is a quiet workload, the second is an idle
GPU, and only `busy` tells them apart.

    PYTHONPATH=. python3 scripts/audit_run_health.py            # report
    PYTHONPATH=. python3 scripts/audit_run_health.py --write    # + exclusions, lists
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.data_sources import ROOT, read_parquet

MANIFEST = "docs/camera_ready_manifest_20261022/streams.parquet"
OUT_DIR = "docs/camera_ready_results_20261022"
# A run is aborted if it stopped before this fraction of its requested duration
# AND the GPU never activated. Both halves matter: a short run that used the GPU
# is a short run, not a broken one.
ABORT_SPAN_FRAC = 0.5
ABORT_BUSY = 0.05
# Busy in under a tenth of its samples, over a full-length run.
STARVED_BUSY = 0.10
DEFAULT_DURATION = 300.0

# Cells that are SUPPOSED to leave the GPU idle. `idle` is the idle-GPU
# baseline; the EC* cells are the edge-case probes built to be near-silent.
# Judging them by busy fraction asks them to fail at their own purpose.
IDLE_BY_DESIGN = {
    "idle", "idle_h100",
    "EC1_phantom", "EC2_silent", "EC3_sparse", "EC5_frozen",
    "EC6_low_int", "EC7_amp_mask", "EC8_mem_idle",
}


def requested_duration(cell: str):
    """`--duration N` from the cell's registry command, or None if unknown.

    None matters. A cell with no registry entry declares no duration, and
    assuming the 300 s default is what turned ten complete 128 s
    adversarial_composite runs into "truncated".
    """
    try:
        from workloads.registry import WORKLOAD_REGISTRY
    except Exception:
        return None
    entry = WORKLOAD_REGISTRY.get(cell)
    if not entry:
        return None
    cmd = [str(x) for x in entry.get("command", [])]
    for i, tok in enumerate(cmd):
        if tok == "--duration" and i + 1 < len(cmd):
            try:
                return float(cmd[i + 1])
            except ValueError:
                pass
        m = re.match(r"--duration=(\d+(?:\.\d+)?)$", tok)
        if m:
            return float(m[1])
    return None


def audit(root: Path, adversarial_only: bool = False) -> pd.DataFrame:
    m = pd.read_parquet(root / MANIFEST)
    a = m[m.tier.eq("A_camera_ready")].copy()
    if adversarial_only:
        a = a[a.base.str.startswith("adversarial_")]
    rows = []
    for _, r in a.iterrows():
        p = root / r.file
        busy = mean_util = float("nan")
        if p.exists():
            try:
                f = read_parquet(p)
                if len(f) and "gpu_utilization_pct" in f:
                    u = f["gpu_utilization_pct"].astype(float)
                    busy, mean_util = float((u > 0).mean()), float(u.mean())
            except Exception:
                pass
        rows.append(dict(job_id=r.job_id, cell=r.base, hw=r.hw, file=r.file,
                         span_s=float(r.span_s), windows=int(r.nominal_windows),
                         busy=busy, mean_util=mean_util))
    s = pd.DataFrame(rows)
    # One row per job; a DDP job's eight streams share a span and a verdict.
    j = s.groupby(["job_id", "cell", "hw"], as_index=False).agg(
        span_s=("span_s", "max"), windows=("windows", "sum"),
        busy=("busy", "mean"), mean_util=("mean_util", "mean"),
        streams=("file", "count"))
    j["duration_s"] = j.cell.map(requested_duration)
    j["verdict"] = "ok"
    declared = j.duration_s.notna()
    short = declared & (j.span_s < ABORT_SPAN_FRAC * j.duration_s.fillna(0))
    j.loc[short & (j.busy.fillna(0) < ABORT_BUSY), "verdict"] = "aborted"
    j.loc[short & (j.busy.fillna(0) >= ABORT_BUSY), "verdict"] = "truncated"
    j.loc[j.verdict.eq("ok") & (j.busy < STARVED_BUSY)
          & ~j.cell.isin(IDLE_BY_DESIGN), "verdict"] = "starved"
    j.loc[j.cell.isin(IDLE_BY_DESIGN) & j.verdict.eq("ok"), "verdict"] = "ok"
    return j.sort_values(["verdict", "hw", "cell"])


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adversarial-only", action="store_true",
                    help="only the adversarial families (the pre-2026-10-29 scope)")
    ap.add_argument("--write", action="store_true",
                    help="update data/quality_exclusions.json and write the lists")
    args = ap.parse_args()
    root = Path(ROOT)
    j = audit(root, adversarial_only=args.adversarial_only)

    bad = j[j.verdict.ne("ok")]
    scope = "adversarial" if args.adversarial_only else "all"
    print(f"tier-A jobs ({scope}): {len(j)}  "
          f"({int(j.duration_s.isna().sum())} with no declared duration, "
          f"span tests skipped for those)")
    for v in ("aborted", "truncated", "starved"):
        g = j[j.verdict.eq(v)]
        print(f"  {v}: {len(g)} jobs, {int(g.windows.sum())} windows "
              f"({g.windows.sum() / max(j.windows.sum(), 1) * 100:.1f}% of the table)")
    print()
    print(bad[["verdict", "cell", "hw", "span_s", "duration_s", "busy", "windows"]]
          .to_string(index=False))

    if not args.write:
        print("\n(dry run; pass --write to record)")
        return 0

    out = root / OUT_DIR
    out.mkdir(parents=True, exist_ok=True)
    cols = ["verdict", "cell", "hw", "job_id", "streams", "span_s", "duration_s",
            "busy", "mean_util", "windows"]
    bad[cols].to_csv(out / "recollection_list.csv", index=False)
    print(f"\nwrote {OUT_DIR}/recollection_list.csv ({len(bad)} jobs)")

    # Per-device coverage, after excluding the unusable.
    cov = j.assign(usable=j.verdict.eq("ok")).groupby("hw").agg(
        cells=("cell", "nunique"), jobs=("job_id", "count"),
        usable_jobs=("usable", "sum"), windows=("windows", "sum"))
    cov["usable_cells"] = j[j.verdict.eq("ok")].groupby("hw").cell.nunique().reindex(cov.index).fillna(0).astype(int)
    cov["usable_windows"] = j[j.verdict.eq("ok")].groupby("hw").windows.sum().reindex(cov.index).fillna(0).astype(int)
    cov.reset_index().to_csv(out / "adversarial_coverage_by_device.csv", index=False)
    print(f"wrote {OUT_DIR}/adversarial_coverage_by_device.csv")

    # Only `aborted` and `starved` are excluded. A truncated run is short, not
    # wrong: the H100 NVL composite jobs stopped at 128 s of 300 with the GPU
    # 90% busy and contributed seven sound windows apiece, and dropping those
    # would delete real measurements to tidy a list. They stay in the
    # re-collection list so the short runs can be redone at full length.
    EXCLUDE = ("aborted", "starved")
    man = root / "data/quality_exclusions.json"
    doc = json.loads(man.read_text())
    have = {e["job_id"] for e in doc["excluded_jobs"]}
    added = 0
    for _, r in bad[bad.verdict.isin(EXCLUDE)].iterrows():
        if r.job_id in have:
            continue
        doc["excluded_jobs"].append({
            "job_id": r.job_id,
            "reason": f"gpu_{r.verdict}",
            "evidence": (f"span {r.span_s:.0f}s of {r.duration_s:.0f}s requested, "
                         f"GPU busy in {r.busy:.1%} of samples"),
        })
        have.add(r.job_id)
        added += 1
    doc["excluded_jobs"].sort(key=lambda e: e["job_id"])
    man.write_text(json.dumps(doc, indent=2) + "\n")
    print(f"added {added} jobs ({'/'.join(EXCLUDE)} only) to data/quality_exclusions.json "
          f"({len(doc['excluded_jobs'])} total)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
