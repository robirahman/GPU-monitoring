#!/usr/bin/env python3
"""Freeze the corpus and partition it into the three paper eras.

Tier A (camera-ready) -- collected on or before 2026-06-17, the arXiv v1 cut
  (https://arxiv.org/html/2606.19262v1). These streams back the submitted and
  accepted results.
Tier B (rebuttal) -- 2026-07-24 .. 2026-08-04: the NeurIPS rebuttal window
  (27 July - 3 August) plus the two days of preparation that introduced STEP
  logging and the Round-6 symGEMM collection that ran out its tail.
Tier C (follow-up) -- 2026-08-19 onward: the multi-node scope ladder, the TPU
  campaigns, the H200-NVL scope study and the October A100 pilots. All of it
  is Future Work and none of it belongs in a camera-ready number.

Collection time comes from the telemetry's own first timestamp, never from the
commit date: `data/gcp/20260820` through `20260921` were all committed on
2026-09-23, weeks after they ran. Streams that fall between the tier windows
are reported as `unassigned` rather than silently dropped, because an era
boundary is a judgement call and hiding the residue would make it invisible.

Like scripts/corpus_inventory.py this reads only identity and timestamp columns
out of the parquet footers, so it runs in about a gigabyte rather than the
~12.4 GB load_all_data needs.

  PYTHONPATH=. python3 scripts/camera_ready_manifest.py --out docs/camera_ready_manifest_20261022
"""
import argparse
import datetime as dt
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.data_sources import accelerator_files, resolve_payload  # noqa: E402
from classifier.multinode_features import assign_stream_and_job_ids  # noqa: E402
from classifier.workload_labels import (SCOPE, canonical_workload, program_work,  # noqa: E402
                                        threeway_label, workload_family, workload_kind,
                                        LABEL_SCHEMA_VERSION, FAMILY_SCHEMA_VERSION)
from classifier.window_labels import build_work_index, default_log_paths  # noqa: E402
from scripts.audit_data_inventory import filename_identity, timestamp_series, time_summary  # noqa: E402
from scripts.corpus_inventory import hw_from_stem, family as hardware_family, RANK  # noqa: E402

# The window geometry every cache builder uses (build_window_cache.py:96-99).
WINDOW_SEC, STRIDE_SEC = 30.0, 15.0

MANIFEST_SCHEMA_VERSION = "camera-ready-manifest-2026-10-22.1"


def _utc(date):
    return dt.datetime.strptime(date, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()


# (name, inclusive start, exclusive end, description). Ordered; first match wins.
TIERS = [
    ("A_camera_ready", None, _utc("2026-06-18"),
     "On or before 2026-06-17 (arXiv v1). Backs the submitted and accepted results."),
    ("unassigned_pre_rebuttal", _utc("2026-06-18"), _utc("2026-07-24"),
     "Between arXiv v1 and the start of rebuttal work. Needs adjudication."),
    ("B_rebuttal", _utc("2026-07-24"), _utc("2026-08-05"),
     "Rebuttal window 27 Jul - 3 Aug, plus preparation and the Round-6 tail."),
    ("unassigned_post_rebuttal", _utc("2026-08-05"), _utc("2026-08-19"),
     "After the rebuttal closed, before the follow-up campaigns began."),
    ("C_followup", _utc("2026-08-19"), None,
     "Multi-node, TPU, H200-NVL scope and October A100 pilots. Future Work."),
]


def tier_of(t0):
    for name, start, end, _ in TIERS:
        if (start is None or t0 >= start) and (end is None or t0 < end):
            return name
    return "unassigned"


def path_stamp(path):
    """Collection date implied by the path, for cross-checking the telemetry clock.

    Two conventions carry it: a `data/gcp/<YYYYmmdd>` session directory, and the
    `_YYYYmmdd_HHMMSS` suffix the legacy single-GPU collector wrote into the
    filename. Either may be absent, which is not an error -- it only means the
    timestamp column is the sole evidence.
    """
    for part in Path(path).parts:
        m = re.fullmatch(r"(\d{8})[a-z]*", part)
        if m:
            return m[1]
        m = re.fullmatch(r"followup_\w+?_(\d{8})T\d{6}Z", part)
        if m:
            return m[1]
    # Anchor to the end: a run id is also eight digits, so an unanchored match
    # happily returns `..._06806384_` and calls the run id a collection date.
    m = re.search(r"_(\d{8})_\d{6}$", Path(path).stem)
    return m[1] if m else None


def scan(files):
    """One row per (file, stream): identity, sample count and observed span."""
    rows, unreadable = [], []
    for i, p in enumerate(files):
        try:
            table = pq.ParquetFile(resolve_payload(p))
            names = set(table.schema.names)
            want = [c for c in ("ts", "timestamp_epoch", "timestamp", "timestamp_utc",
                                "gpu_name", "workload_label", "run_id", "gpu_uuid",
                                "hostname") if c in names]
            d = table.read(columns=want).to_pandas()
            d["t"] = timestamp_series(d)
        except Exception as exc:                       # noqa: BLE001 - recorded, not raised
            unreadable.append({"file": str(p), "error": f"{type(exc).__name__}: {exc}"})
            continue
        d = d[np.isfinite(d.t) & d.t.between(1e8, 4e9)]
        if len(d) < 2:
            unreadable.append({"file": str(p), "error": "fewer than two usable timestamps"})
            continue
        label, hw, rid, host, device = filename_identity(p)
        for col, fallback in (("workload_label", label), ("gpu_name", hw), ("run_id", rid),
                              ("gpu_uuid", device or Path(p).stem), ("hostname", host)):
            if col not in d:
                d[col] = fallback
            d[col] = d[col].fillna(fallback)
        stamp = path_stamp(p)
        for key, g in d.groupby(["run_id", "gpu_uuid", "hostname", "workload_label", "gpu_name"],
                                dropna=False):
            run_id, gpu_uuid, hostname, wl, gpu = (str(k) for k in key)
            t = time_summary(g.t.to_numpy())
            base = RANK.sub("", wl)
            rows.append(dict(
                file=str(Path(p).relative_to(PROJECT_ROOT)), stem=Path(p).stem,
                run_id=run_id, gpu_uuid=gpu_uuid, hostname=hostname,
                workload_label=wl, base=base, gpu_name=gpu,
                hw=hardware_family(gpu if gpu and gpu != "unknown" else hw_from_stem(Path(p).stem)),
                stream=f"{run_id}|{gpu_uuid}|{hostname}|{base}",
                path_stamp=stamp, **t))
        if (i + 1) % 2000 == 0:
            print(f"  scanned {i + 1:,}/{len(files):,}", flush=True)
    return pd.DataFrame(rows), unreadable


def deduplicate(d):
    """load_all_data collapses on (ts, run_id, gpu_uuid); keep the longest copy.

    The same stream is written into more than one directory, so the file-level
    count overstates the corpus by about a third (10,840 files were 6,602
    streams on 2026-09-18).
    """
    return (d.sort_values(["span_s", "samples"], ascending=False)
             .drop_duplicates("stream", keep="first")
             .reset_index(drop=True))


def enrich(dd, work_index):
    dd = dd.copy()
    dd["tier"] = dd.t0.map(tier_of)
    dd["cls"] = dd.base.map(threeway_label)
    dd["cell"] = dd.base.map(canonical_workload)
    dd["evasion_family"] = dd.base.map(lambda b: workload_family(b, level="mechanism"))
    dd["kind"] = dd.base.map(workload_kind)
    dd["scope"] = dd.base.map(lambda b: (SCOPE.match(str(b)).group(1)
                                         if SCOPE.match(str(b)) else "single-GPU (legacy)"))
    # Nominal windows: what a 30 s / 15 s sweep yields over the covered span.
    # Multi-node pooling and rank-coverage gating change the realised count, so
    # this is the corpus's capacity, not the cache's output.
    covered = dd.gap_capped_s.to_numpy()
    dd["nominal_windows"] = np.where(covered >= WINDOW_SEC,
                                     np.floor((covered - WINDOW_SEC) / STRIDE_SEC) + 1, 0).astype(int)
    dd["stream_hours"] = dd.span_s / 3600.0
    dd["covered_stream_hours"] = dd.gap_capped_s / 3600.0
    dd["has_step_log"] = dd.run_id.isin(work_index)
    # A program whose work is not knowable from its name: the window label has
    # to come from evidence, because the job label is a blend by construction.
    dd["mixed_work"] = dd.base.map(program_work).isna()

    ids = assign_stream_and_job_ids(dd.assign(ts=dd.t0))
    dd["job_id"] = ids["job_id"].to_numpy()
    dd["stream_id"] = ids["stream_id"].to_numpy()

    # The path says one collection date and the telemetry clock says another:
    # worth knowing about, never worth silently preferring one.
    stamp = pd.to_datetime(dd.path_stamp, format="%Y%m%d", errors="coerce", utc=True)
    clock = pd.to_datetime(dd.t0, unit="s", utc=True).dt.normalize()
    dd["stamp_clock_delta_days"] = (clock - stamp).dt.total_seconds() / 86400.0
    return dd


def aggregate(frame):
    hosts = frame.groupby("job_id").hostname.nunique()
    job_span = frame.groupby("job_id").agg(t0=("t0", "min"), t1=("t1", "max"))
    adv = frame[frame.kind.eq("adversarial")]
    return {
        "files": int(frame.file.nunique()),
        "physical_jobs": int(frame.job_id.nunique()),
        "gpu_streams": int(frame.stream_id.nunique()),
        "workload_configurations": int(frame.cell.nunique()),
        "evasion_families": int(frame.evasion_family.nunique()),
        "adversarial_evasion_families": int(adv.evasion_family.nunique()),
        "adversarial_jobs": int(adv.job_id.nunique()),
        "windows_30s_15s_nominal": int(frame.nominal_windows.sum()),
        "accelerator_hours": round(float(frame.stream_hours.sum()), 3),
        "covered_accelerator_hours": round(float(frame.covered_stream_hours.sum()), 3),
        "wall_clock_hours": round(float(((job_span.t1 - job_span.t0) / 3600.0).sum()), 3),
        "samples": int(frame.samples.sum()),
        "hardware_models": int(frame.hw.nunique()),
        "multi_host_jobs": int((hosts > 1).sum()),
        "jobs_with_step_log": int(frame[frame.has_step_log].job_id.nunique()),
        "mixed_work_jobs": int(frame[frame.mixed_work].job_id.nunique()),
        "mixed_work_windows": int(frame[frame.mixed_work].nominal_windows.sum()),
        "mixed_work_jobs_with_step_log": int(
            frame[frame.mixed_work & frame.has_step_log].job_id.nunique()),
        "by_class": {k: int(v) for k, v in frame.groupby("cls").job_id.nunique().items()},
        "first_collected": pd.to_datetime(frame.t0.min(), unit="s", utc=True).strftime("%Y-%m-%d"),
        "last_collected": pd.to_datetime(frame.t1.max(), unit="s", utc=True).strftime("%Y-%m-%d"),
    }


# What the paper claims for the camera-ready corpus (Methodology, Table 2).
PAPER_TIER_A = {"runs": 1404, "workloads": 162, "workloads_table2_sum": 149,
                "gpu_hours": 117, "wall_clock_hours": 65, "readings": 445310,
                "by_class_runs": {"ml_training": 867, "ml_inference": 352, "other": 185}}


def render(manifest, dd, out):
    tiers = manifest["tiers"]
    cols = [("physical_jobs", "jobs"), ("gpu_streams", "streams"),
            ("workload_configurations", "configs"), ("evasion_families", "families"),
            ("windows_30s_15s_nominal", "windows"), ("accelerator_hours", "accel-h"),
            ("samples", "samples")]
    lines = [f"# Camera-ready corpus manifest, {manifest['generated_utc'][:10]}", "",
             "Generated by `PYTHONPATH=. python3 scripts/camera_ready_manifest.py`.",
             "Re-run it rather than trusting these numbers after the next collection.", "",
             "Tiers are keyed on the telemetry's own first timestamp. Commit dates are",
             "useless here: `data/gcp/20260820`-`20260921` were all committed on 2026-09-23,",
             "weeks after they ran.", "",
             "`accel-h` is summed per-device stream span, so it is already duration x device",
             "count. `windows` is the nominal 30 s / 15 s capacity of each stream's covered",
             "span, not the realised cache output -- multi-node pooling and rank-coverage",
             "gating change the latter.", "",
             "| tier | " + " | ".join(c[1] for c in cols) + " | first | last |",
             "| :-- | " + " | ".join("--:" for _ in cols) + " | :-- | :-- |"]
    for name, _, _, _ in TIERS:
        a = tiers.get(name)
        if not a:
            lines.append(f"| `{name}` | " + " | ".join("0" for _ in cols) + " | - | - |")
            continue
        cells = [f"{a[k]:,}" if isinstance(a[k], int) else f"{a[k]:,.1f}" for k, _ in cols]
        lines.append(f"| `{name}` | " + " | ".join(cells) +
                     f" | {a['first_collected']} | {a['last_collected']} |")
    lines += ["", "## Tier definitions", ""]
    lines += [f"- **`{n}`** -- {desc}" for n, _, _, desc in TIERS]

    rec = manifest["tier_a_vs_paper"]
    lines += ["", "## Tier A against the paper", "",
              "Reported as measured. No filter was tuned to hit a published figure.", "",
              "| quantity | paper | manifest | delta |", "| :-- | --: | --: | --: |"]
    for k, (paper, got) in rec["comparison"].items():
        delta = "-" if paper is None else f"{got - paper:+,.1f}"
        lines.append(f"| {k} | {paper if paper is not None else '-'} | {got:,} | {delta} |")
    lines += ["", *rec["notes"]]

    lines += ["", "## Per-window evidence", "",
              "Per-window ground truth exists only where a launcher log recorded `STEP`",
              "lines. STEP logging was added on 2026-07-24 (`8ee55e53`), after every Tier A",
              "run had already been collected.", "",
              "`mixed` counts jobs whose program interleaves kinds of work, so the job label",
              "cannot stand for a window: `workload_labels.program_work` returns None. Those",
              "are the jobs where window-level and run-level scoring actually disagree.", "",
              "| tier | jobs | with STEP | mixed-work jobs | mixed with STEP | mixed windows |",
              "| :-- | --: | --: | --: | --: | --: |"]
    for name, _, _, _ in TIERS:
        a = tiers.get(name)
        if not a:
            lines.append(f"| `{name}` | 0 | 0 | 0 | 0 | 0 |")
            continue
        lines.append(f"| `{name}` | {a['physical_jobs']:,} | {a['jobs_with_step_log']:,} | "
                     f"{a['mixed_work_jobs']:,} | {a['mixed_work_jobs_with_step_log']:,} | "
                     f"{a['mixed_work_windows']:,} |")

    if manifest["unreadable"]:
        lines += ["", f"## Unreadable inputs ({len(manifest['unreadable'])})", "",
                  "Grouped by directory; `manifest.json` carries the full list with reasons.", ""]
        by_dir = {}
        for r in manifest["unreadable"]:
            rel = str(Path(r["file"]))
            if rel.startswith(str(PROJECT_ROOT)):
                rel = str(Path(rel).relative_to(PROJECT_ROOT))
            by_dir.setdefault(str(Path(rel).parent), []).append(r["error"].split(":")[0])
        for directory, errors in sorted(by_dir.items()):
            kinds = ", ".join(sorted(set(errors)))
            lines.append(f"- `{directory}` -- {len(errors)} file(s), {kinds}")
    if manifest["not_scanned"]:
        lines += ["", "## Present on disk but outside the scan", ""]
        for note, paths in manifest["not_scanned"].items():
            lines.append(f"- {note}: {len(paths)} file(s)")
    (out / "README.md").write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="docs/camera_ready_manifest_20261022")
    ap.add_argument("--streams", default=None,
                    help="reuse a previously written file-level scan instead of rescanning")
    a = ap.parse_args()
    out = PROJECT_ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)

    files = accelerator_files(PROJECT_ROOT)
    print(f"{len(files):,} accelerator parquet files", flush=True)
    # The unreadable list is part of the result, not scan noise: reusing a scan
    # must not quietly drop it and make the corpus look complete.
    unreadable_path = out / "unreadable.json"
    if a.streams:
        d = pd.read_parquet(a.streams)
        if not unreadable_path.is_file():
            raise SystemExit(f"--streams needs the matching {unreadable_path}; rescan instead")
        unreadable = json.loads(unreadable_path.read_text())
    else:
        d, unreadable = scan(files)
        unreadable_path.write_text(json.dumps(unreadable, indent=2) + "\n")
        print(f"usable {len(d):,} file-streams, unreadable {len(unreadable)}", flush=True)

    dd = deduplicate(d)
    print(f"{len(dd):,} distinct streams after deduplication", flush=True)
    work_index = build_work_index(default_log_paths(PROJECT_ROOT))
    print(f"{len(work_index):,} run_ids carry STEP evidence", flush=True)
    dd = enrich(dd, work_index)

    tiers = {name: aggregate(g) for name, g in dd.groupby("tier") if len(g)}
    overall = aggregate(dd)
    tier_a = tiers.get("A_camera_ready", {})
    comparison = {
        "physical jobs (paper: runs)": (PAPER_TIER_A["runs"], tier_a.get("physical_jobs", 0)),
        "workload configurations": (PAPER_TIER_A["workloads"], tier_a.get("workload_configurations", 0)),
        "accelerator-hours": (PAPER_TIER_A["gpu_hours"], tier_a.get("accelerator_hours", 0)),
        "wall-clock hours": (PAPER_TIER_A["wall_clock_hours"], tier_a.get("wall_clock_hours", 0)),
        "telemetry readings": (PAPER_TIER_A["readings"], tier_a.get("samples", 0)),
    }
    notes = [
        "- The paper's Table 2 lists 93 + 40 + 16 = 149 workloads while the prose says 162;"
        " the discrepancy predates this manifest.",
        "- Paper 'runs' are physical jobs. Here a job is `assign_stream_and_job_ids`'s"
        " `job_id`, which collapses the sibling files of one multi-GPU execution.",
        "- 'telemetry readings' counts per-device samples, matching the paper's"
        " 'readings x 9 signals' phrasing.",
    ]

    manifest = {
        "schema": MANIFEST_SCHEMA_VERSION,
        "generated_utc": dt.datetime.now(dt.timezone.utc).isoformat(),
        "label_schema": LABEL_SCHEMA_VERSION,
        "family_schema": FAMILY_SCHEMA_VERSION,
        "window": {"window_sec": WINDOW_SEC, "stride_sec": STRIDE_SEC},
        "tier_windows": [{"tier": n, "start": s, "end": e, "description": d} for n, s, e, d in TIERS],
        "overall": overall,
        "tiers": tiers,
        "tier_a_vs_paper": {"comparison": comparison, "notes": notes},
        "unreadable": unreadable,
        "not_scanned": {
            "data/gcp/20260820c partial writes (accelerator_files globs *.parquet only)":
                [str(p.relative_to(PROJECT_ROOT))
                 for p in (PROJECT_ROOT / "data").rglob("*.parquet.partial")],
        },
    }
    # The partition must be exact: every stream in one tier, none in two.
    assert sum(t["gpu_streams"] for t in tiers.values()) == overall["gpu_streams"], \
        "tiers do not partition the corpus"

    # Parquet for re-running the aggregation, gzipped CSV for the repo: the
    # catch-all `*.parquet filter=lfs` rule would otherwise push a regenerable
    # artifact through LFS. A local .gitignore keeps the parquets out of git.
    d.to_parquet(out / "streams_file_level.parquet", index=False)
    dd.to_parquet(out / "streams.parquet", index=False)
    dd.to_csv(out / "streams.csv.gz", index=False, compression="gzip")
    (out / ".gitignore").write_text("*.parquet\n")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    render(manifest, dd, out)
    print(f"\nwrote {out}/manifest.json, streams.parquet, README.md")
    for name in (t[0] for t in TIERS):
        if name in tiers:
            t = tiers[name]
            print(f"  {name:26s} jobs {t['physical_jobs']:6,d}  streams {t['gpu_streams']:6,d}  "
                  f"accel-h {t['accelerator_hours']:10,.1f}  windows {t['windows_30s_15s_nominal']:9,d}")


if __name__ == "__main__":
    main()
