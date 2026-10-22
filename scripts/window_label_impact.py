#!/usr/bin/env python3
"""How many camera-ready windows change label once work is observed, not assumed.

Reads only the tier A mixed-work streams the manifest identified, windows them
through the same path every evaluator now uses, and compares the window label
against the job objective it used to inherit.

  PYTHONPATH=. python3 scripts/window_label_impact.py
"""
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.data_sources import read_parquet  # noqa: E402
from classifier.threeway_improved import sliding_windows  # noqa: E402
from classifier.workload_labels import threeway_label  # noqa: E402

MANIFEST = PROJECT_ROOT / "docs/camera_ready_manifest_20261022/streams.parquet"
OUT = PROJECT_ROOT / "docs/camera_ready_results_20261022"
WINDOW, STRIDE = 30.0, 15.0


def main():
    streams = pd.read_parquet(MANIFEST)
    tier_a = streams[streams.tier.eq("A_camera_ready") & streams.mixed_work]
    print(f"{len(tier_a):,} tier A mixed-work streams, {tier_a.job_id.nunique():,} jobs", flush=True)

    frames = []
    for i, row in enumerate(tier_a.itertuples(index=False)):
        try:
            d = read_parquet(PROJECT_ROOT / row.file)
        except Exception as exc:                       # noqa: BLE001
            print(f"  skip {row.file}: {type(exc).__name__}", flush=True)
            continue
        # The loader's column aliases, applied the same way load_all_data does.
        d = d.rename(columns={"timestamp_epoch": "ts", "timestamp": "ts", "t": "ts",
                              "gpu_util": "gpu_utilization_pct", "power_w": "power_draw_w"})
        if "ts" not in d.columns:
            print(f"  skip {row.file}: no timestamp column", flush=True)
            continue
        for col, value in (("run_id", row.run_id), ("workload_label", row.workload_label),
                           ("gpu_uuid", row.gpu_uuid), ("hostname", row.hostname),
                           ("gpu_name", row.gpu_name)):
            if col not in d.columns:
                d[col] = value
        d["threeway_label"] = d["workload_label"].map(threeway_label)
        frames.append(d)
        if (i + 1) % 100 == 0:
            print(f"  loaded {i + 1:,}/{len(tier_a):,}", flush=True)
    raw = pd.concat(frames, ignore_index=True)
    print(f"{len(raw):,} rows", flush=True)

    win = sliding_windows(raw, WINDOW, STRIDE)
    win["job_objective"] = win["workload_label"].map(threeway_label)
    win["changed"] = win.threeway_label.ne(win.job_objective)

    per_cell = (win.groupby("workload_label")
                .agg(windows=("threeway_label", "size"),
                     source=("label_source", lambda s: s.mode().iat[0] if len(s) else ""),
                     now_training=("threeway_label", lambda s: int((s == "ml_training").sum())),
                     now_inference=("threeway_label", lambda s: int((s == "ml_inference").sum())),
                     unresolved=("label_source", lambda s: int((s == "unresolved").sum())),
                     changed=("changed", "sum"))
                .sort_values("changed", ascending=False))
    per_cell["changed_share"] = per_cell.changed / per_cell.windows

    OUT.mkdir(parents=True, exist_ok=True)
    per_cell.to_csv(OUT / "window_label_impact.csv")
    moved = per_cell[per_cell.changed > 0]
    pd.set_option("display.width", 200)
    print("\n== cells whose windows no longer all inherit the job label ==")
    print(moved.to_string(float_format=lambda v: f"{v:6.3f}") if len(moved) else "  none")
    print(f"\ntotal tier A mixed-work windows: {int(per_cell.windows.sum()):,}")
    print(f"windows whose label changed:     {int(per_cell.changed.sum()):,}")
    print(f"windows still unresolved:        {int(per_cell.unresolved.sum()):,}")
    print(f"\nwrote {OUT / 'window_label_impact.csv'}")


if __name__ == "__main__":
    main()
