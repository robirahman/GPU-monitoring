#!/usr/bin/env python3
"""What tier A's numbers would be if the RTX 4050 gap had not existed.

All nine RTX 4050 adversarial jobs in tier A aborted within 6-7 s with the GPU
never activating, so the device contributes no adversarial evidence at all and
its jobs are excluded. The cells were re-collected on 2026-10-27 -- same
laptop, same driver 560.94 -- and those runs are healthy.

This scores tier A with those re-collected runs folded in. The result is NOT
tier A's number: the runs are six months later and came through a Windows code
path that did not exist in April. It answers one question only -- whether the
gap is load-bearing for any conclusion -- and the answer is only useful if
reported as a sensitivity check beside the real figure.

    PYTHONPATH=. python3 scripts/sensitivity_recollected.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.threeway_improved import load_all_data, sliding_windows  # noqa: E402
from scripts.regen_headline import FOLDS, STRIDE, WINDOW, evaluate  # noqa: E402

# The nine cells whose tier-A runs aborted on this device.
RECOLLECTED = sorted((PROJECT_ROOT / "data").glob(
    "adversarial_*_NVIDIA_GeForce_RTX_4050_Laptop_GPU_*_20261027_*.parquet"))


def run(tag, extra):
    raw = load_all_data(tier="A_camera_ready", extra_files=extra)
    print(f"{tag}: {len(raw):,} rows, {raw.run_id.nunique():,} runs", flush=True)
    out = []
    for name, mode, events in (("published (leaky, job labels)", "orig", None),
                               ("causal, job labels", "causal", None),
                               ("causal, window labels", "causal", "auto")):
        win = sliding_windows(raw, WINDOW, STRIDE, run_level_mode=mode, work_events=events)
        res = evaluate(win, name)
        res["config"] = tag
        out.append(res)
        print(f"  {name}: {res.get('windows')} windows, "
              f"acc={res.get('accuracy')}", flush=True)
    return out


def main() -> int:
    print(f"re-collected RTX 4050 adversarial parquets: {len(RECOLLECTED)}")
    if not RECOLLECTED:
        print("none found; nothing to compare")
        return 1
    rows = run("tier A as collected", None)
    rows += run("tier A + re-collected 4050", [str(p) for p in RECOLLECTED])
    df = pd.DataFrame(rows)
    print()
    print(df[["config", "variant", "windows", "workloads", "accuracy", "std",
              "training_share"]].to_string(index=False))

    out = PROJECT_ROOT / "docs/camera_ready_results_20261022/sensitivity_recollected_4050.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
