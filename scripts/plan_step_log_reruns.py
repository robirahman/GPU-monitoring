#!/usr/bin/env python3
"""What re-runs would buy, per GPU model, for the camera-ready window labels.

Tier A predates STEP logging, so its mixed-work windows have no per-window
truth. classifier/schedule_reconstruction.py can recover a cell's windows once
a reference run of the SAME cell on the SAME device has been logged -- the
batch rate is a property of the run, not of the schedule.

This lists the (cell, device) pairs that would each need one short logged
re-run, what it would unlock, and what it would cost.

  PYTHONPATH=. python3 scripts/plan_step_log_reruns.py
"""
import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.schedule_reconstruction import reference_rates  # noqa: E402
from classifier.workload_labels import program_work  # noqa: E402

MANIFEST = PROJECT_ROOT / "docs/camera_ready_manifest_20261022/streams.parquet"
OUT = PROJECT_ROOT / "docs/camera_ready_results_20261022"


def registry_command(cell):
    try:
        from workloads.registry import WORKLOADS
    except Exception:                                  # noqa: BLE001
        return None
    entry = WORKLOADS.get(cell)
    if not entry:
        return None
    return " ".join(str(c) for c in entry.get("command", []))


def main():
    d = pd.read_parquet(MANIFEST)
    a = d[d.tier.eq("A_camera_ready")].copy()
    a["mixed"] = a.base.map(program_work).isna()
    mixed = a[a.mixed]

    rows = []
    for (cell, hw), g in mixed.groupby(["base", "hw"]):
        lo, _, n_ref = reference_rates(cell, hw)
        rows.append(dict(cell=cell, device=hw, jobs=int(g.job_id.nunique()),
                         windows=int(g.nominal_windows.sum()),
                         median_run_s=round(float(g.span_s.median()), 1),
                         has_reference=bool(n_ref), reference_runs=int(n_ref)))
    plan = pd.DataFrame(rows).sort_values(["has_reference", "windows"],
                                          ascending=[True, False])
    need = plan[~plan.has_reference].copy()
    # One logged re-run per pair, three reps for a rate interval rather than a point.
    need["rerun_gpu_hours"] = need.median_run_s * 3 / 3600.0

    OUT.mkdir(parents=True, exist_ok=True)
    plan.to_csv(OUT / "step_log_rerun_plan.csv", index=False)

    pd.set_option("display.width", 200)
    covered = plan[plan.has_reference]
    print("== already covered by a logged reference run ==")
    print(covered.to_string(index=False) if len(covered) else "  none")
    print(f"\n== need one logged re-run each ({len(need)} cell x device pairs) ==")
    by_device = (need.groupby("device")
                 .agg(pairs=("cell", "nunique"), jobs=("jobs", "sum"),
                      windows=("windows", "sum"), gpu_hours=("rerun_gpu_hours", "sum"))
                 .sort_values("windows", ascending=False))
    print(by_device.to_string(float_format=lambda v: f"{v:7.2f}"))
    print(f"\nTOTAL: {len(need)} re-runs x 3 reps, "
          f"{need.rerun_gpu_hours.sum():.1f} GPU-hours, "
          f"unlocking {int(need.windows.sum()):,} tier A windows "
          f"({int(need.jobs.sum()):,} jobs)")
    print(f"\nwrote {OUT / 'step_log_rerun_plan.csv'}")

    sample = need.head(12)[["cell", "device", "windows", "median_run_s"]].copy()
    sample["command"] = sample.cell.map(registry_command)
    print("\n== highest-value pairs, with the command the registry would run ==")
    print(sample.to_string(index=False))


if __name__ == "__main__":
    main()
