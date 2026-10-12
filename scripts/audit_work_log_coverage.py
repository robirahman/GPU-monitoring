#!/usr/bin/env python3
"""Audit TPU disguise STEP joins using the saved raw-stream inventory.

Reports 30 s / 15 s windows over recorded spans, not feature-cache eligibility.
No telemetry payloads or GPU are required. This checks log identities, temporal
overlap and work labels before a full cache rebuild.
"""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from classifier.data_sources import sha256
from classifier.multinode_features import assign_stream_and_job_ids
from classifier.window_labels import build_work_index, default_log_paths, job_work_intervals, label_windows
from classifier.workload_labels import program_work
from scripts.gcp.fetch_work_logs import tracked_tpu_ids


def audit(streams_path, root, out, window=30., stride=15.):
    root, out = Path(root), Path(out)
    streams = pd.read_csv(streams_path, keep_default_na=False)
    streams = streams[streams.role.eq("accelerator") & ~streams.quarantine.astype(str).str.lower().eq("true")].copy()
    streams["source_path"] = streams.paths
    streams["ts"] = streams.t0
    streams = assign_stream_and_job_ids(streams)
    targets = streams[streams.run_id.isin(tracked_tpu_ids(root)) & streams.workload_label.map(program_work).isna()]
    # Two measured endpoints per stream preserve each job's telemetry extent.
    endpoints = pd.concat([targets.assign(ts=targets.t0), targets.assign(ts=targets.t1)], ignore_index=True)
    logs = [p for p in default_log_paths(root) if "/tpu/" in p.as_posix()]
    events = job_work_intervals(endpoints, build_work_index(logs))
    rows, windows = [], []
    for job, g in targets.groupby("job_id"):
        lo, hi = float(g.t0.min()), float(g.t1.max())
        starts = np.arange(lo, hi - window + 1e-6, stride)
        iv = events.get(job)
        if iv is None:
            labels = ["unresolved"] * len(starts)
            train, infer = np.full(len(starts), np.nan), np.full(len(starts), np.nan)
        else:
            lab = label_windows(starts, starts + window, iv)
            labels, train, infer = lab["label"], lab["train_s"], lab["infer_s"]
        counts = pd.Series(labels, dtype=str).value_counts().to_dict()
        rows.append(dict(job_id=job, run_id=g.run_id.iloc[0], workload_label=g.workload_label.iloc[0],
                         observed_hosts=g.hostname.nunique(), observed_streams=g.stream_id.nunique(),
                         first_telemetry=lo, last_telemetry=hi, matched_temporal_evidence=iv is not None,
                         probe_windows=len(starts), **{k+"_windows": int(counts.get(k, 0))
                                                     for k in ("ml_training", "ml_inference", "other", "unresolved")}))
        windows.extend(dict(job_id=job, window_start=float(s), window_end=float(s+window),
                            threeway_label=str(label), train_s=float(tr), infer_s=float(inf))
                       for s, label, tr, inf in zip(starts, labels, train, infer))
    jobs = pd.DataFrame(rows)
    out.mkdir(parents=True, exist_ok=True)
    jobs.to_csv(out / "tpu_jobs.csv", index=False)
    pd.DataFrame(windows).to_csv(out / "tpu_span_windows.csv", index=False)
    summary = dict(commit=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
                   inventory_sha256=sha256(streams_path), logs={str(p.relative_to(root)): sha256(p) for p in logs},
                   target_jobs=len(jobs), jobs_with_temporally_matched_steps=int(jobs.matched_temporal_evidence.sum()),
                   jobs_without_temporally_matched_steps=int((~jobs.matched_temporal_evidence).sum()),
                   window_seconds=window, stride_seconds=stride,
                   note="Span probes check time joins; actual cache windows additionally require sample/rank coverage.")
    (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps({k:v for k,v in summary.items() if k != "logs"}, indent=2))
    return summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--streams", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    audit(args.streams, args.repo, args.out)


if __name__ == "__main__":
    main()
