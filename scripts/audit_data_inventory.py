#!/usr/bin/env python3
"""Inventory tracked telemetry directly from external LFS objects, without checkout.

Completion requires every tracked LFS payload and verifies its SHA256. Duplicate
objects are read once and stream timestamps are unioned across overlapping files.
Counts separate quarantine, sidecars, schema errors and unverified collections.
"""
import argparse
import collections
import datetime
import json
import os
from pathlib import Path
import re
import subprocess
import sys

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.data_sources import pointer, resolve_payload, sha256, quarantined, data_role, filename_workload_label
from classifier.workload_labels import (canonical_workload, threeway_label, label_evidence,
                                        workload_kind, workload_family, _legacy_label,
                                        LABEL_SCHEMA_VERSION, FAMILY_SCHEMA_VERSION)
from classifier.multinode_features import assign_stream_and_job_ids, expected_rank_count
from classifier.workload_descriptions import describe_workload


def filename_identity(path):
    stem = Path(path).stem
    stem = re.sub(r"^nic_", "", stem)
    label = re.split(r"_(?:NVIDIA|AMD|Google_TPU)_", stem)[0]
    if "week4/results/" in str(path) or "week7" in Path(path).parts:
        label = filename_workload_label(path)
    hardware = "unknown"
    m = re.search(r"_(NVIDIA|AMD|Google_TPU)_(.*)", stem)
    if m:
        tail = re.split(r"_[0-9a-f]{8,}(?:_|$)|_\d{8}_\d{6}", m[2])[0]
        hardware = (m[1] + " " + tail).replace("_", " ")
    ids = re.findall(r"_([0-9a-f]{8,})(?=_|$)", stem)
    rid = ids[-1] if ids else stem
    if not m:
        label = re.sub(r"_[0-9a-f]{8,}.*$", "", label)
    label = re.sub(r"_(?:scope|dcgm|bmc|wall_power|sensors)$", "", label)
    rank = re.search(r"_node(\d+)(?:_gpu(\d+))?|_w(\d+)c(\d+)", stem)
    host = ("node" + rank[1] if rank[1] else "w" + rank[3]) if rank else ""
    device = rank[0].strip("_") if rank else ""
    return label, hardware, rid, host, device


def timestamp_series(frame):
    for name in ("ts", "timestamp_epoch", "timestamp", "timestamp_utc", "time", "epoch", "epoch_s", "t"):
        if name not in frame:
            continue
        v = frame[name]
        if pd.api.types.is_datetime64_any_dtype(v):
            return v.dt.as_unit("ns").astype("int64") / 1e9
        numeric = pd.to_numeric(v, errors="coerce")
        if numeric.notna().any():
            return numeric
        parsed = pd.to_datetime(v, utc=True, errors="coerce")
        if parsed.notna().any():
            return (parsed.dt.as_unit("ns").astype("int64") / 1e9).where(parsed.notna())
    raise ValueError("No supported timestamp column")


def time_summary(times):
    times = np.unique(np.asarray(times, dtype=float))
    delta = np.diff(times)
    med = float(np.median(delta)) if len(delta) else 0.0
    covered = float(delta[delta <= max(5 * med, 5)].sum())
    return dict(samples=len(times), t0=float(times[0]), t1=float(times[-1]),
                span_s=float(times[-1] - times[0]), gap_capped_s=covered, median_dt=med)


def render_inventory(out):
    """Render measured tables without re-reading raw payloads; audit routing too."""
    out = Path(out)
    summary = json.loads((out / "summary.json").read_text())
    streams = pd.read_csv(out / "streams.csv", keep_default_na=False)
    # A report refresh cannot silently change the class population underlying
    # workload tables. A semantic change requires a full inventory rebuild.
    current = streams.workload_label.map(threeway_label)
    if not current.eq(streams.threeway_label).all():
        raise ValueError("Inventory labels changed; rebuild before rendering")
    streams["previous_routing"] = streams.workload_label.map(_legacy_label)
    streams.to_csv(out / "streams.csv", index=False)
    changed = streams.threeway_label.ne(streams.previous_routing)
    streams[changed].to_csv(out / "label_corrections.csv", index=False)
    accepted = streams.role.eq("accelerator") & ~streams.quarantine.astype(str).str.lower().eq("true")
    summary["label_corrected_streams"] = int((accepted & changed).sum())
    summary["report_code_sha256"] = sha256(Path(__file__))
    summary["label_comparison_code_sha256"] = sha256(Path(__file__).resolve().parents[1] / "classifier/workload_labels.py")
    summary["description_code_sha256"] = sha256(Path(__file__).resolve().parents[1] / "classifier/workload_descriptions.py")
    observation_note = "Device and host counts describe recorded telemetry; they do not prove every device participated in computation."
    if observation_note not in summary["notes"]:
        summary["notes"].append(observation_note)
    f = pd.read_csv(out / "files.csv")
    coverage = []
    for role, g in f.groupby("role"):
        measured = streams[streams.role == role]
        coverage.append(dict(role=role, paths=len(g), unique_objects=g.oid.nunique(),
                             unique_bytes=int(g.drop_duplicates("oid").bytes.sum()),
                             measured_streams=len(measured), samples=int(measured.samples.sum()),
                             stream_hours=float(measured.stream_hours.sum())))
    coverage = pd.DataFrame(coverage)
    coverage.to_csv(out / "role_coverage.csv", index=False)
    jobs = pd.read_csv(out / "jobs.csv")
    quality = jobs.groupby("collection_status").agg(jobs=("job_id", "size"), samples=("samples", "sum"),
                                                     stream_hours=("stream_hours", "sum"))
    quality.to_csv(out / "collection_quality.csv")
    exclusions = json.loads((out / "quality_exclusions.json").read_text())["excluded_jobs"]
    failed = jobs[jobs.job_id.isin([x["job_id"] for x in exclusions])]
    summary["failed_jobs_excluded_from_training"] = len(failed)
    summary["failed_job_stream_hours"] = float(failed.stream_hours.sum())
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    workloads = pd.read_csv(out / "workloads.csv")
    workloads["description"] = workloads.cell.map(describe_workload)
    workloads.to_csv(out / "workloads.csv", index=False)
    hardware = pd.read_csv(out / "hardware.csv")
    errors = pd.read_csv(out / "read_errors.csv") if (out / "read_errors.csv").stat().st_size > 1 else pd.DataFrame()
    html = "<!doctype html><html lang='en'><meta charset='utf-8'><title>GPU workload inventory</title>"
    html += "<style>body{font:15px system-ui;margin:2em}table{border-collapse:collapse}td,th{padding:5px;border:1px solid #ddd}th{position:sticky;top:0;background:white}td{max-width:50em;overflow-wrap:anywhere}pre{white-space:pre-wrap}</style>"
    html += "<h1>GPU workload inventory</h1><p>" + " ".join(summary["notes"]) + "</p>"
    links = ("summary.json", "workloads.csv", "hardware.csv", "jobs.csv", "streams.csv", "files.csv",
             "label_corrections.csv", "log_evidence.csv", "quality_exclusions.json", "read_errors.csv", "waveforms.csv")
    html += "<p>Download tables: " + " · ".join(f"<a href='{name}'>{name}</a>" for name in links) + "</p>"
    html += "<h2>Summary</h2><pre>" + json.dumps(summary, indent=2) + "</pre>"
    html += "<h2>Artifact and sensor coverage</h2><p>This table includes quarantine; sensor hours are not accelerator-hours.</p>" + coverage.to_html(index=False)
    html += "<h2>Collection evidence outside quarantine</h2>" + quality.to_html()
    html += "<h2>Unreadable artifacts</h2>" + errors.to_html(index=False)
    html += "<h2>Accelerator hardware outside quarantine</h2>" + hardware.to_html(index=False)
    html += "<h2>Workloads</h2><input id='q' placeholder='Filter workload or hardware'><span id='count'></span>"
    html += workloads.to_html(index=False, table_id="workloads")
    html += "<script>const q=document.querySelector('#q');q.oninput=()=>{let n=0;for(const r of document.querySelectorAll('#workloads tbody tr')){r.hidden=!r.textContent.toLowerCase().includes(q.value.toLowerCase());n+=!r.hidden;}document.querySelector('#count').textContent=n+' rows';};</script></html>"
    (out / "inventory.html").write_text(html)
    return summary


def scan_logs(root, paths, tracked=None):
    tracked = set(paths) if tracked is None else set(tracked)
    rows = []
    for path_index, rel in enumerate(paths):
        if not rel.startswith("logs/") or not rel.endswith(".log") or ".nic." in rel:
            continue
        record = None
        with resolve_payload(root / rel).open(errors="replace") as f:
            for line in f:
                start = re.search(r"Multi-GPU run: (.*?), (\d+) GPUs, node=(\d+), run_id=(\S+)", line)
                if start:
                    if record:
                        rows.append(record)
                    record = dict(path=rel, tracked=rel in tracked, cell=start[1], devices=int(start[2]), node=int(start[3]),
                                  run_id=start[4], rc=None, train_steps=0, infer_steps=0, sync_steps=0)
                if record is None:
                    continue
                step = re.search(r"STEP\s+ts=[\d.]+\s+kind=(train|infer)\b", line)
                if step:
                    record[step[1] + "_steps"] += 1
                    record["sync_steps"] += int(bool(re.search(r"\bsync=1\b", line)))
                end = re.search(r"Multi-GPU run complete:.*?\(rc=(-?\d+)\)", line)
                if end:
                    record["rc"] = int(end[1])
        if record:
            rows.append(record)
    return pd.DataFrame(rows)


def inventory(root, objects, out, allow_incomplete=False):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    os.environ["SPAR_LFS_OBJECTS"] = str(objects)
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "-z"]).decode().strip("\0").split("\0")
    head = subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"]).decode().strip()
    implementation_root = Path(__file__).resolve().parents[1]
    code_hashes = {name: sha256(implementation_root / name) for name in (
        "scripts/audit_data_inventory.py", "classifier/workload_labels.py", "classifier/data_sources.py",
        "classifier/workload_descriptions.py", "classifier/multinode_features.py")}
    files, payloads, lfs, missing = [], {}, {}, []
    for path_index, rel in enumerate(paths):
        if path_index % 2000 == 0:
            print(f"Checking paths {path_index}/{len(paths)}", flush=True)
        p = root / rel
        if not p.is_file():
            missing.append(dict(path=rel, reason="tracked file absent"))
            continue
        ptr = pointer(p)
        payload = None
        try:
            payload = resolve_payload(p, objects)
        except FileNotFoundError as exc:
            missing.append(dict(path=rel, reason=str(exc)))
        if ptr:
            lfs[ptr[0]] = dict(oid=ptr[0], bytes=ptr[1], payload=payload)
        telemetry_json = rel.startswith("week4/results/edge_cases/") and p.name.endswith("_telemetry.json")
        if p.suffix not in (".parquet", ".npy") and not telemetry_json:
            continue
        oid = ptr[0] if ptr else sha256(p)
        label, hw, rid, host, device = filename_identity(p)
        role = "raw_waveform" if p.suffix == ".npy" else data_role(p)
        r = dict(path=rel, oid=oid, bytes=ptr[1] if ptr else p.stat().st_size,
                 available=payload is not None, quarantine=quarantined(p), role=role,
                 label_from_filename=label, hardware_from_filename=hw,
                 run_from_filename=rid, host_from_filename=host, device_from_filename=device)
        files.append(r)
        if payload:
            payloads.setdefault(oid, (payload, []))[1].append(r)
    pd.DataFrame(missing).to_csv(out / "missing_objects.csv", index=False)
    completeness = dict(unique_lfs_objects=len(lfs), lfs_bytes=sum(x["bytes"] for x in lfs.values()),
                        missing_paths=len(missing), complete=not missing, verified_sha256=False)
    (out / "completion.json").write_text(json.dumps(completeness, indent=2))
    if missing and not allow_incomplete:
        raise RuntimeError(f"LFS download incomplete ({len(missing)} paths); inventory not finalized")
    verified = 0
    for oid, record in lfs.items():
        if record["payload"] is None:
            continue
        if sha256(record["payload"]) != oid:
            raise ValueError(f"Corrupt LFS object {oid}")
        verified += 1
        if verified % 2000 == 0:
            print(f"Verified {verified}/{len(lfs)} LFS objects", flush=True)
    completeness["verified_objects"] = verified
    completeness["verified_sha256"] = verified == len(lfs)
    (out / "completion.json").write_text(json.dumps(completeness, indent=2))
    missing_paths = {m["path"] for m in missing}
    log_paths = sorted((set(paths) | {str(p.relative_to(root)) for p in (root / "logs").glob("*.log")}) - missing_paths)
    logs = scan_logs(root, log_paths, tracked=paths)
    logs.to_csv(out / "log_evidence.csv", index=False)
    streams, errors, waveforms = {}, [], []
    for i, (oid, (payload, refs)) in enumerate(payloads.items()):
        r = refs[0]
        try:
            if r["role"] == "raw_waveform":
                arr = np.load(payload, mmap_mode="r", allow_pickle=False)
                waveforms.append(dict(oid=oid, path=r["path"], shape=str(arr.shape), dtype=str(arr.dtype),
                                      bytes=r["bytes"]))
                del arr
                continue
            is_json = r["path"].endswith(".json")
            if is_json:
                d = pd.DataFrame(json.loads(payload.read_text()))
                names, nrows = set(d.columns), len(d)
            else:
                pf = pq.ParquetFile(payload)
                names, nrows = set(pf.schema.names), pf.metadata.num_rows
            if "window_start" in names and not names & {"ts", "timestamp_epoch", "timestamp", "timestamp_utc"}:
                for ref in refs:
                    ref["role"] = "derived_features"
                    ref["file_rows"] = nrows
                continue
            cols = [c for c in ("ts", "timestamp_epoch", "timestamp", "timestamp_utc", "time", "epoch", "epoch_s", "t",
                                "gpu_name", "gpu_uuid", "hostname", "run_id", "workload_label",
                                "threeway_label", "gpu_utilization_pct", "world_size", "expected_ranks", "gpu_index", "gpu") if c in names]
            for ref in refs:
                ref["file_rows"] = nrows
                ref["schema_columns"] = "|".join(sorted(names))
            d = d[cols].copy() if is_json else pf.read(columns=cols).to_pandas()
            d["t"] = timestamp_series(d)
            # Relative-time recordings remain explicitly uncounted, not guessed
            # into calendar time or multiplied into accelerator-hours.
            d = d[np.isfinite(d.t) & d.t.between(1e8, 4e9)].copy()
            if d.empty:
                raise ValueError("No finite epoch timestamps (possibly relative-time data)")
            if "gpu_uuid" not in d:
                for device_col in ("gpu", "gpu_index"):
                    if device_col in d:
                        d["gpu_uuid"] = "gpu" + d[device_col].astype(str)
                        break
            for col, fallback in (("workload_label", r["label_from_filename"]), ("gpu_name", r["hardware_from_filename"]),
                                  ("gpu_uuid", r["device_from_filename"] or oid[:12]),
                                  ("hostname", r["host_from_filename"]), ("run_id", "source:" + oid)):
                if col not in d:
                    d[col] = fallback
                d[col] = d[col].fillna(fallback).astype(str).replace({"": fallback, "None": fallback, "nan": fallback})
            # Accepted copies supersede quarantined duplicate paths only in this
            # evidence table; both dispositions remain visible in files.csv.
            quarantine = all(x["quarantine"] for x in refs)
            for ident, g in d.groupby(["run_id", "gpu_uuid", "hostname", "workload_label", "gpu_name"], dropna=False):
                rid, uid, host, lab, hw = ident
                key = (r["role"], quarantine, *ident)
                z = streams.setdefault(key, dict(role=r["role"], quarantine=quarantine, run_id=rid,
                                      gpu_uuid=uid, hostname=host, workload_label=lab, gpu_name=hw,
                                      times=[], paths=set(), stored_labels=set()))
                z["times"].append(g.t.to_numpy())
                z["paths"].update(ref["path"] for ref in refs)
                if "threeway_label" in g:
                    z["stored_labels"].update(g.threeway_label.dropna().astype(str))
        except Exception as exc:
            errors.append(dict(path=r["path"], oid=oid, reason=f"{type(exc).__name__}: {exc}"))
        if (i + 1) % 1000 == 0:
            print(f"Read {i + 1}/{len(payloads)} unique payloads", flush=True)
    rows = []
    for z in streams.values():
        z.update(time_summary(np.concatenate(z.pop("times"))))
        z["paths"] = "|".join(sorted(z["paths"]))
        z["stored_labels"] = "|".join(sorted(z["stored_labels"]))
        lab = z["workload_label"]
        z.update(cell=canonical_workload(lab), family=workload_family(lab), kind=workload_kind(lab),
                 threeway_label=threeway_label(lab), routing_evidence=label_evidence(lab)[1],
                 previous_routing=_legacy_label(lab))
        rows.append(z)
    d = pd.DataFrame(rows)
    d["ts"] = d.t0
    ids = pd.concat([assign_stream_and_job_ids(g) for _, g in d.groupby(["role", "quarantine"])], ignore_index=True)
    ids["stream_hours"] = ids.span_s / 3600
    ids["covered_stream_hours"] = ids.gap_capped_s / 3600
    ids.to_csv(out / "streams.csv", index=False)
    pd.DataFrame(files).to_csv(out / "files.csv", index=False)
    pd.DataFrame(errors).to_csv(out / "read_errors.csv", index=False)
    pd.DataFrame(waveforms).to_csv(out / "waveforms.csv", index=False)
    ids[ids.threeway_label != ids.previous_routing].to_csv(out / "label_corrections.csv", index=False)
    ids[ids.routing_evidence == "unreviewed_other"].to_csv(out / "labels_for_review.csv", index=False)
    jobs, exclusions = [], []
    acc = ids[(ids.role == "accelerator") & ~ids.quarantine].copy()
    for job, g in acc.groupby("job_id"):
        expected = expected_rank_count(g)
        log = logs[logs.run_id.isin(g.run_id)] if len(logs) else logs
        if len(log) and g.cell.iloc[0].startswith(("s1_", "s8_")):
            nodes = set(g.workload_label.str.extract(r"_node(\d+)", expand=False).dropna().astype(int))
            nodes.update(map(int, re.findall(r"_node(\d+)(?:_|\b)", "|".join(g.paths))))
            log = log[log.node.isin(nodes)]
        failed = len(log) and (log.rc.dropna() != 0).any()
        complete = len(log) and log.rc.notna().all() and log.rc.eq(0).all()
        conflicting = bool(len(log) and log.groupby("node").rc.agg(
            lambda codes: codes.eq(0).any() and (codes.dropna() != 0).any()).any())
        hosts = g.hostname.replace("", np.nan).nunique()
        status = "failed_log" if failed else ("completed_logged_hosts" if complete else "unverified")
        if conflicting:
            status = "conflicting_completion_evidence"
        if failed and not conflicting:
            exclusions.append(dict(job_id=job, reason="nonzero_workload_exit",
                                   logs=sorted(log.loc[log.rc.notna() & log.rc.ne(0), "path"].unique())))
        if expected and g.stream_id.nunique() < expected:
            status = "incomplete_ranks"
        jobs.append(dict(job_id=job, cell=g.cell.iloc[0], family=g.family.iloc[0], hardware=g.gpu_name.iloc[0],
                         threeway_label=g.threeway_label.iloc[0], kind=g.kind.iloc[0], hosts=hosts,
                         observed_devices=g.stream_id.nunique(), expected_devices=expected,
                         collection_status=status, log_hosts=log.node.nunique() if len(log) else 0,
                         train_step_lines=int(log.train_steps.sum()) if len(log) else 0,
                         samples=int(g.samples.sum()), stream_hours=float(g.stream_hours.sum()),
                         covered_stream_hours=float(g.covered_stream_hours.sum()),
                         elapsed_span_s=float(g.t1.max() - g.t0.min())))
    jobs = pd.DataFrame(jobs)
    jobs.to_csv(out / "jobs.csv", index=False)
    (out / "quality_exclusions.json").write_text(json.dumps(dict(
        schema="2026-10-11.1", source_commit=head, excluded_jobs=exclusions), indent=2))
    groups = acc.groupby(["cell", "family", "gpu_name", "threeway_label", "kind"], dropna=False)
    workloads = groups.agg(streams=("stream_id", "nunique"), jobs=("job_id", "nunique"), samples=("samples", "sum"),
                            stream_hours=("stream_hours", "sum"), covered_stream_hours=("covered_stream_hours", "sum"),
                            median_stream_seconds=("span_s", "median"), min_stream_seconds=("span_s", "min"),
                            max_stream_seconds=("span_s", "max")).reset_index()
    workloads.insert(1, "description", workloads.cell.map(describe_workload))
    topo = jobs.groupby(["cell", "hardware"]).agg(
        min_devices=("observed_devices", "min"), max_devices=("observed_devices", "max"),
        min_hosts=("hosts", "min"), max_hosts=("hosts", "max"),
        elapsed_job_hours=("elapsed_span_s", lambda s: s.sum() / 3600),
        log_completed_jobs=("collection_status", lambda s: int(s.eq("completed_logged_hosts").sum())),
        failed_or_incomplete_jobs=("collection_status", lambda s: int(s.isin(["failed_log", "incomplete_ranks"]).sum())))
    workloads = workloads.merge(topo, left_on=["cell", "gpu_name"], right_index=True, how="left")
    workloads.to_csv(out / "workloads.csv", index=False)
    hardware = acc.groupby("gpu_name").agg(streams=("stream_id", "nunique"), jobs=("job_id", "nunique"),
                                           cells=("cell", "nunique"), samples=("samples", "sum"),
                                           stream_hours=("stream_hours", "sum"), covered_stream_hours=("covered_stream_hours", "sum"))
    hardware.to_csv(out / "hardware.csv")
    summary = dict(generated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(), commit=head,
                   code_sha256=code_hashes,
                   labels=LABEL_SCHEMA_VERSION, families=FAMILY_SCHEMA_VERSION, completion=completeness,
                   parquet_paths=sum(x["path"].endswith(".parquet") for x in files),
                   telemetry_json_paths=sum(x["path"].endswith("_telemetry.json") for x in files),
                   unique_parquet_objects=len({x["oid"] for x in files if x["path"].endswith(".parquet")}),
                   quarantine_paths=sum(x["quarantine"] for x in files), read_errors=len(errors),
                   accepted_accelerator_streams=len(acc), accelerator_samples=int(acc.samples.sum()),
                   accelerator_stream_hours=float(acc.stream_hours.sum()),
                   accelerator_covered_stream_hours=float(acc.covered_stream_hours.sum()),
                   jobs=len(jobs), cells=acc.cell.nunique(), label_corrected_streams=int((acc.threeway_label != acc.previous_routing).sum()),
                   notes=["Accepted means outside quarantine, not verified useful compute.",
                          "Durations are timestamp spans and gap-capped coverage, not active optimizer time.",
                          "Legacy job identities use 20-second buckets and are provisional.",
                          "Completion is evidenced only for available logged hosts; inspect log_hosts against topology.",
                          "Raw waveforms and sidecars are counted separately from accelerator-hours.",
                          "Telemetry covers tracked files plus current code fixes; log evidence includes marked untracked local logs."])
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    summary = render_inventory(out)
    print(json.dumps(summary, indent=2), flush=True)
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--repo", default=str(Path(__file__).resolve().parents[1]))
    ap.add_argument("--lfs", required=True, help="External objects directory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--allow-incomplete", action="store_true", help="Diagnostic only; final inventory requires all objects")
    a = ap.parse_args()
    inventory(a.repo, a.lfs, a.out, a.allow_incomplete)


if __name__ == "__main__":
    main()
