#!/usr/bin/env python3
"""Pull a re-run campaign out of GCS and report what it unlocks.

The campaign writes `<cell>_rep<N>` logs under the cell's registry name, which
starts with `adversarial_`. schedule_reconstruction.reference_rates strips that
prefix before matching (`adversarial_L_diluted_2` -> `L_diluted_2_rep\\d+`), so
the logs are renamed on the way in; left alone they sit in logs/ and are never
found, which is a silent miss rather than an error.

A re-run measures the rate of *a* run of that cell on that device, never of the
archived run, so this also prints the telemetry agreement between the two: if
the archived and re-run utilisation and power distributions do not overlap, the
rate should not be transferred and the cell is reported as such rather than
reconstructed.

    PYTHONPATH=. python3 scripts/ingest_rerun_references.py 20261023b
"""
from __future__ import annotations

import argparse
import re
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from classifier.data_sources import ROOT, read_parquet
from classifier.schedule_reconstruction import (
    dilution_factor, gpu_family, reference_rates)

BUCKET = "gs://focus-surfer-494820-g0-telemetry"
# <stamp>__<host>__<registry cell>_rep<N>.log
LOG_NAME = re.compile(r"^(?P<stamp>[^_]+)__(?P<host>.+?)__(?P<cell>.+)_rep(?P<rep>\d+)\.log$")


def fetch(stamp: str, dest: Path) -> Path:
    """Mirror the campaign's GCS prefix locally (idempotent)."""
    dest.mkdir(parents=True, exist_ok=True)
    src = f"{BUCKET}/runs/{stamp}"
    subprocess.run(["gcloud", "storage", "rsync", "-r", src, str(dest)],
                   check=True, capture_output=True)
    return dest


def install(staged: Path, stamp: str, root: Path) -> tuple[int, int]:
    """Copy parquets into data/gcp/<stamp>/ and logs into logs/, renamed."""
    data_dir = root / "data" / "gcp" / stamp
    logs_dir = root / "logs"
    data_dir.mkdir(parents=True, exist_ok=True)
    logs_dir.mkdir(parents=True, exist_ok=True)
    n_pq = n_log = 0
    for pq in staged.rglob("*.parquet"):
        # the campaign syncs its CIFAR copy too; that is not telemetry
        if "cifar" in pq.parts:
            continue
        shutil.copy2(pq, data_dir / pq.name)
        n_pq += 1
    for log in staged.rglob("*.log"):
        m = LOG_NAME.match(log.name)
        if not m:
            continue
        cell = re.sub(r"^adversarial_", "", m["cell"])
        out = logs_dir / f"{m['stamp']}__{m['host']}__{cell}_rep{m['rep']}.log"
        shutil.copy2(log, out)
        n_log += 1
    return n_pq, n_log


def telemetry_agreement(cell: str, device: str, stamp: str, root: Path) -> dict:
    """How close the re-run's telemetry sits to the archived run's.

    Compares MEAN utilisation, the fraction of samples with any utilisation at
    all, and the wall-clock span. Not the median, which was the first version
    and was useless here: a GPU idle in more than half its samples has a median
    of 0 whatever the rest of the run did, so it scored a dead 27-second run
    and a healthy diluted one identically at 0.

    Span matters on its own. Four of these cells' archived runs stop around
    30 s into a 300 s workload with the GPU never once registering activity --
    those are aborted runs, not slow ones, and no rate can be recovered from
    them.
    """
    out = {"a_span": np.nan, "r_span": np.nan, "a_util": np.nan, "r_util": np.nan,
           "a_busy": np.nan, "r_busy": np.nan, "verdict": "no archived run"}
    want = re.sub(r"[^A-Za-z0-9]+", "_", device)
    arch = sorted(p for p in (root / "data").rglob(f"{cell}_*.parquet")
                  if want in p.name and "/gcp/2026102" not in str(p))
    rerun = sorted((root / "data" / "gcp" / stamp).glob(f"{cell}_*.parquet"))
    if not arch or not rerun:
        out["verdict"] = "no archived run" if not arch else "no re-run"
        return out

    def summarise(paths):
        seen, spans, means, busy = set(), [], [], []
        for q in paths:
            try:
                d = read_parquet(q)
            except Exception:
                continue
            if not len(d) or "gpu_utilization_pct" not in d:
                continue
            rid = d["run_id"].iloc[0] if "run_id" in d else q.name
            if rid in seen:          # the tree holds some runs twice
                continue
            seen.add(rid)
            u = d["gpu_utilization_pct"].astype(float)
            spans.append(float(d.timestamp_epoch.max() - d.timestamp_epoch.min()))
            means.append(float(u.mean()))
            busy.append(float((u > 0).mean()))
        if not spans:
            return None
        return (float(np.mean(spans)), float(np.mean(means)), float(np.mean(busy)))

    a, r = summarise(arch), summarise(rerun)
    if a is None or r is None:
        out["verdict"] = "unreadable (LFS pointer?)"
        return out
    out["a_span"], out["a_util"], out["a_busy"] = a
    out["r_span"], out["r_util"], out["r_busy"] = r

    # An archived run that stopped at a fraction of the re-run's duration with
    # a GPU that never activated did not run; it died.
    if a[0] < 0.5 * r[0] and a[2] < 0.05:
        out["verdict"] = "ARCHIVED RUN ABORTED"
    # Ran to length, but did ~nothing while the re-run was busy throughout.
    elif a[2] < 0.10 and r[2] > 0.50:
        out["verdict"] = "ARCHIVED RUN STARVED"
    elif abs(a[1] - r[1]) <= 8 and abs(a[2] - r[2]) <= 0.15:
        out["verdict"] = "comparable"
    else:
        out["verdict"] = "diverged"
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stamp", help="campaign stamp, e.g. 20261023b")
    ap.add_argument("--device", default="A100-SXM4-80GB")
    ap.add_argument("--staging", default=None)
    ap.add_argument("--no-fetch", action="store_true")
    args = ap.parse_args()

    root = Path(ROOT)
    staged = Path(args.staging or f"/srv/gpu-monitoring/rerun-staging/{args.stamp}")
    if not args.no_fetch:
        print(f"[ingest] fetching {BUCKET}/runs/{args.stamp} -> {staged}")
        fetch(args.stamp, staged)
    n_pq, n_log = install(staged, args.stamp, root)
    print(f"[ingest] installed {n_pq} parquets and {n_log} logs")

    cells = sorted({re.sub(r"^adversarial_", "", LOG_NAME.match(p.name)["cell"])
                    for p in staged.rglob("*.log") if LOG_NAME.match(p.name)})
    rows = []
    for cell in cells:
        full = f"adversarial_{cell}"
        lo, hi, n = reference_rates(full, args.device, root=root)
        agree = telemetry_agreement(full, args.device, args.stamp, root)
        rows.append({
            "cell": full,
            "N": dilution_factor(full),
            "reps": n,
            "rate_lo": None if lo is None else round(lo, 4),
            "rate_hi": None if hi is None else round(hi, 4),
            "spread": None if lo is None or not lo else round(hi / lo, 2),
            "a_span": None if agree["a_span"] != agree["a_span"] else round(agree["a_span"]),
            "r_span": None if agree["r_span"] != agree["r_span"] else round(agree["r_span"]),
            "a_util": None if agree["a_util"] != agree["a_util"] else round(agree["a_util"], 1),
            "r_util": None if agree["r_util"] != agree["r_util"] else round(agree["r_util"], 1),
            "a_busy": None if agree["a_busy"] != agree["a_busy"] else round(agree["a_busy"], 2),
            "r_busy": None if agree["r_busy"] != agree["r_busy"] else round(agree["r_busy"], 2),
            "verdict": agree["verdict"],
        })
    df = pd.DataFrame(rows)
    print()
    print(df.to_string(index=False))
    print()
    usable = df[(df.N.notna()) & (df.reps > 0) & (df.verdict == "comparable")]
    print(f"[ingest] {len(usable)} of {len(df)} cells are deterministic, have a "
          f"measured rate, and match their archived run's telemetry")
    if len(usable) < len(df[df.N.notna()]):
        print("[ingest] cells with a schedule but no transferable rate:")
        for _, r in df[(df.N.notna()) & (df.verdict != "comparable")].iterrows():
            print(f"           {r.cell}: {r.verdict}")
    out = root / "docs" / "camera_ready_results_20261022" / f"rerun_references_{args.stamp}.csv"
    df.to_csv(out, index=False)
    print(f"[ingest] wrote {out.relative_to(root)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
