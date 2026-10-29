#!/usr/bin/env python3
"""Tier A's numbers if the unusable jobs had been collected properly.

Three configurations, each scored the same way:

  tier A as collected          the real figure
  + B200/H200                  the two devices re-collected 2026-10-28/29
  + every re-collection        those plus A100 (2026-10-23/26) and the
                               RTX 4050 laptop (2026-10-27)

None of these is tier A's number. The re-collected runs are months later and
some came through software paths that did not exist when the corpus was
frozen; the starved cells are collected under `_mw8` labels because re-running
the originals reproduces the starvation, so they ADD a workload rather than
substituting for one. The archived jobs they answer for are already excluded
by data/quality_exclusions.json, so the net effect is those cells going from
absent to present.

    PYTHONPATH=. python3 scripts/sensitivity_recollected.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.threeway_improved import load_all_data, sliding_windows  # noqa: E402
from scripts.regen_headline import STRIDE, WINDOW, evaluate  # noqa: E402

DATA = PROJECT_ROOT / "data"


def files(*globs):
    out = []
    for g in globs:
        out += [str(p) for p in sorted(DATA.glob(g))]
    return sorted(set(out))


# Re-collections, by the stamp in the collector's filename.
B200_H200 = files("adversarial_*_NVIDIA_B200_*_2026102[89]_*.parquet",
                  "adversarial_*_NVIDIA_H200_*_2026102[89]_*.parquet")
A100 = files("adversarial_*_NVIDIA_A100_SXM4_80GB_*_2026102[36]_*.parquet")
RTX4050 = files("adversarial_*_NVIDIA_GeForce_RTX_4050_Laptop_GPU_*_20261027_*.parquet")

# The B200/H200-only configuration was dropped: the question it answered --
# does one device group matter on its own -- is less useful than the full
# counterfactual, and it cost an hour of corpus loading to produce.
CONFIGS = [
    ("tier A as collected", None),
    ("+ every re-collection", B200_H200 + A100 + RTX4050),
]


def run(tag, extra):
    raw = load_all_data(tier="A_camera_ready", extra_files=extra)
    print(f"\n{tag}: {len(raw):,} rows, {raw.run_id.nunique():,} runs", flush=True)
    out = []
    for name, mode, events in (("published (leaky, job labels)", "orig", None),
                               ("causal, job labels", "causal", None),
                               ("causal, window labels", "causal", "auto")):
        win = sliding_windows(raw, WINDOW, STRIDE, run_level_mode=mode, work_events=events)
        res = evaluate(win, name)
        res["config"] = tag
        out.append(res)
        print(f"  {name}: {res.get('windows')} windows, acc={res.get('accuracy')}",
              flush=True)
        if events == "auto":
            res2 = evaluate(win, "causal, window labels, steps-only", steps_only=True)
            res2["config"] = tag
            out.append(res2)
            print(f"  steps-only: {res2.get('windows')} windows, "
                  f"acc={res2.get('accuracy')}", flush=True)
    return out


def main() -> int:
    print(f"re-collected parquets -- B200/H200 {len(B200_H200)}, "
          f"A100 {len(A100)}, RTX 4050 {len(RTX4050)}")
    rows = []
    for tag, extra in CONFIGS:
        rows += run(tag, extra)
    df = pd.DataFrame(rows)
    cols = ["config", "variant", "windows", "workloads", "accuracy", "std", "training_share"]
    print("\n" + df[[c for c in cols if c in df]].to_string(index=False))
    out = PROJECT_ROOT / "docs/camera_ready_results_20261022/sensitivity_recollected.csv"
    df.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(PROJECT_ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
