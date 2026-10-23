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


def telemetry_agreement(cell: str, device: str, stamp: str, root: Path) -> dict:  # noqa: C901
    """How close the re-run's telemetry sits to the archived run's.

    Returns the overlap of the two utilisation distributions and the ratio of
    median power. A rate measured on a run that does not look like the archived
    one is not that run's rate.
    """
    out = {"archived_util": np.nan, "rerun_util": np.nan,
           "power_ratio": np.nan, "verdict": "no archived run"}
    # The device has to match. The same cell ran on nine GPU models, and the
    # first archived file for `adversarial_E_memory_minimal` is an RX 9070 XT
    # run -- comparing an A100 re-run against that says nothing about either.
    # The collector writes the device into the filename with non-alphanumerics
    # flattened to underscores.
    want = re.sub(r"[^A-Za-z0-9]+", "_", device)
    arch = sorted(p for p in (root / "data").rglob(f"{cell}_*.parquet")
                  if want in p.name)
    arch = [p for p in arch if f"/gcp/{stamp}/" not in str(p)]
    rerun = sorted((root / "data" / "gcp" / stamp).glob(f"{cell}_*.parquet"))
    if not arch or not rerun:
        out["verdict"] = "no archived run" if not arch else "no re-run"
        return out

    def load(paths):
        # data_sources.read_parquet resolves an LFS pointer through
        # SPAR_LFS_OBJECTS; the working tree holds pointers, not payloads, so
        # pandas on its own sees a 130-byte text file and raises.
        frames = []
        for p in paths:
            try:
                frames.append(read_parquet(p))
            except Exception:
                continue
        return pd.concat(frames) if frames else None

    a, r = load(arch), load(rerun)
    if a is None or r is None:
        out["verdict"] = "unreadable (LFS pointer?)"
        return out
    ucol = next((c for c in ("gpu_utilization_pct", "utilization_gpu") if c in a and c in r), None)
    pcol = next((c for c in ("power_draw_w", "power_w") if c in a and c in r), None)
    if ucol:
        out["archived_util"] = float(a[ucol].median())
        out["rerun_util"] = float(r[ucol].median())
    if pcol and float(a[pcol].median()) > 0:
        out["power_ratio"] = float(r[pcol].median()) / float(a[pcol].median())
    # Deliberately blunt: this is a go/no-go on transferring a rate, not a test.
    du = abs(out["archived_util"] - out["rerun_util"])
    if np.isnan(du):
        out["verdict"] = "no comparable column"
    elif du <= 10 and (np.isnan(out["power_ratio"]) or 0.8 <= out["power_ratio"] <= 1.25):
        out["verdict"] = "comparable"
    else:
        out["verdict"] = "DIVERGED - do not transfer rate"
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
            "archived_util": round(agree["archived_util"], 1) if agree["archived_util"] == agree["archived_util"] else None,
            "rerun_util": round(agree["rerun_util"], 1) if agree["rerun_util"] == agree["rerun_util"] else None,
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
