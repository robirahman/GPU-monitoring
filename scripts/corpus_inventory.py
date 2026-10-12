#!/usr/bin/env python3
"""Corpus inventory: how much telemetry we have, on what hardware, of what kind.

Answers that without paying for a full load. It reads only the identity and
timestamp columns straight out of the parquet footers -- about 1 GB resident,
against the ~12.4 GB peak `load_all_data` needs on the full corpus, which is
what makes this runnable on the 15 GB analysis box while something else is busy.

File selection mirrors load_all_data sections 1-5: data/** minus the nic_
sidecars and anything under a _quarantine directory, plus week4/data,
week4/results/ddp and week4/results/dataset_scale.

It reports BOTH a file-level and a deduplicated view, and they differ by a third.
The same stream is written into more than one directory, and load_all_data
collapses those on (ts, run_id, gpu_uuid) -- 10,840 files were 6,602 distinct
streams on 2026-09-18. The deduplicated figures are the corpus; the file-level
ones are only what is on disk.

  PYTHONPATH=. python3 scripts/corpus_inventory.py [--out inventory.parquet]
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from classifier.threeway_improved import threeway_label  # noqa: E402

RANK = re.compile(r"_node\d+_gpu\d+|_gpu\d+|_w\d+c\d+")
ACCEL = ("_Google_TPU_", "_NVIDIA_", "_AMD_")
# tpuTMS4 (a 4-slice multislice group) must precede tpuTMS, and the optional
# smoke_ prefix must be tolerated -- without both, 32 genuinely multi-host runs
# fell into the "legacy" bucket. Verified against the worker tokens in the
# filenames: a tpuT16 run is w0..w3 x c0..c3 (4 hosts), while a tpuT8 run is
# w0c0..w0c7 (ONE host, 8 chips) -- the two ran on differently-shaped v5e slices,
# which is exactly why host count is measured here rather than inferred from the
# chip count in the label.
from classifier.workload_labels import SCOPE, ADV, canonical_workload, workload_kind
from classifier.data_sources import accelerator_files, resolve_payload
from scripts.audit_data_inventory import timestamp_series, filename_identity


def corpus_files():
    return accelerator_files(PROJECT_ROOT)


def hw_from_stem(stem):
    """Hardware from the filename, for the frames that carry no gpu_name column.

    Taking only the first token after the accelerator tag truncates
    "GeForce RTX 5080" to "GeForce"; strip the trailing run id / timestamp
    instead and keep the whole name.
    """
    for tok in ACCEL:
        if tok in stem:
            tail = stem.split(tok)[1]
            tail = re.sub(r"_[0-9a-f]{8,}.*$|_\d{8}_\d{6}.*$", "", tail)
            pre = tok.strip("_").replace("_", " ")
            return f"{pre} {tail.replace('_', ' ')}".strip()
    return "unknown"


def family(g):
    g = str(g)
    if "TPU" in g:
        return "Google TPU " + g.split("TPU")[-1].strip().split("-")[0]
    if "AMD" in g or "Radeon" in g:
        return g.replace("AMD Radeon ", "AMD ").strip()
    return g.replace("NVIDIA ", "").strip()


def scan(files):
    rows, bad = [], 0
    for i, p in enumerate(files):
        payload = resolve_payload(p)
        pf = pq.ParquetFile(payload)
        names = set(pf.schema.names)
        want = [c for c in ("ts", "timestamp_epoch", "timestamp", "timestamp_utc", "gpu_name",
                            "workload_label", "run_id", "gpu_uuid", "hostname") if c in names]
        d = pf.read(columns=want).to_pandas()
        d["t"] = timestamp_series(d)
        d = d[np.isfinite(d.t) & d.t.between(1e8, 4e9)].copy()
        if len(d) < 2:
            bad += 1
            continue
        label, hw, rid, host, device = filename_identity(p)
        for col, fallback in (("workload_label", label), ("gpu_name", hw), ("run_id", rid),
                              ("gpu_uuid", device or p.stem), ("hostname", host)):
            if col not in d:
                d[col] = fallback
            d[col] = d[col].fillna(fallback)
        for (rid, uid, host, lab, gpu), g in d.groupby(
                ["run_id", "gpu_uuid", "hostname", "workload_label", "gpu_name"], dropna=False):
            span = float(g.t.max() - g.t.min())
            base = RANK.sub("", lab)
            rows.append(dict(file=p.stem, gpu=gpu, hw=family(gpu), label=lab,
                             base=canonical_workload(lab), run_id=rid, host=host,
                             stream=f"{rid}|{uid}|{host}|{base}", rows=len(g), span_s=span,
                             hz=(len(g)-1)/span if span else np.nan,
                             t0=float(g.t.min()), t1=float(g.t.max())))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{len(files):,}", flush=True)
    return pd.DataFrame(rows), bad


def topology(dd):
    """Which runs genuinely spanned more than one MACHINE -- measured, not assumed.

    Host count alone is not enough. multinode_queue.sh derives the run_id from
    (stamp, cell, rep) with no node component, so on a SINGLE-NODE cell (s1_/s8_)
    both machines run their own independent execution under the same run_id and
    a naive host count calls them multi-node. classifier/multinode_features.py
    resolves that the same way: scope tells you the truth, s1/s8 are per-node by
    construction and s16 / the multi-host TPU scopes genuinely span hosts. For
    TPU the host count IS measured, because chips-per-host differs by generation
    (v5e is 4, v6e is 8), so the same tpuT8 label is one machine on v6e and two
    on v5e.
    """
    hosts = dd.groupby("run_id")["host"].nunique()
    dd = dd.copy()
    dd["hosts"] = dd.run_id.map(hosts).fillna(1).astype(int)
    per_node = dd.scope.isin(["s1", "s8"])
    dd["multinode"] = np.where(per_node, False,
                               dd.hosts > 1)
    return dd


def report(d):
    d = d.copy()
    d["cls"] = d.base.map(threeway_label)
    d["min"] = d.span_s / 60
    d["scope"] = d.base.map(lambda l: (SCOPE.match(str(l)).group(1)
                                       if SCOPE.match(str(l)) else "single-GPU (legacy)"))
    d["kind"] = d.base.map(workload_kind)

    def block(title, frame):
        g = frame.groupby("hw").agg(
            wkl=("base", "nunique"), runs=("run_id", "nunique"),
            streams=("file", "size"), minutes=("min", "sum"),
            samples=("rows", "sum")).sort_values("minutes", ascending=False)
        print(f"\n== {title} ==")
        print(f"{'hardware':28s} {'wkls':>5s} {'runs':>6s} {'streams':>8s} "
              f"{'stream-min':>11s} {'stream-h':>9s} {'samples':>11s}")
        for k, r in g.iterrows():
            print(f"{k:28s} {int(r.wkl):5d} {int(r.runs):6d} {int(r.streams):8d} "
                  f"{r.minutes:11,.0f} {r.minutes / 60:9,.0f} {int(r.samples):11,d}")
        print(f"{'TOTAL':28s} {frame.base.nunique():5d} {frame.run_id.nunique():6d} "
              f"{len(frame):8d} {frame['min'].sum():11,.0f} "
              f"{frame['min'].sum() / 60:9,.0f} {int(frame['rows'].sum()):11,d}")

    block("file level (what is on disk)", d)
    dd = d.sort_values("span_s", ascending=False).drop_duplicates("stream", keep="first")
    block("longest copy per stream (use audit_data_inventory.py for timestamp unions)", dd)

    for name, frame in (("class", "cls"), ("scope", "scope"), ("kind", "kind")):
        g = dd.groupby(frame).agg(wkl=("base", "nunique"), runs=("run_id", "nunique"),
                                  streams=("file", "size"), minutes=("min", "sum"))
        print(f"\n== deduplicated, by {name} ==")
        for k, r in g.sort_values("minutes", ascending=False).iterrows():
            print(f"  {str(k):22s} {int(r.wkl):4d} workloads {int(r.runs):5d} runs "
                  f"{int(r.streams):6d} streams {r.minutes:9,.0f} stream-min")

    p = dd.pivot_table(index="hw", columns="cls", values="min", aggfunc="sum").fillna(0)
    p["total"] = p.sum(axis=1)
    print("\n== deduplicated stream-minutes, hardware x class ==")
    print(p.sort_values("total", ascending=False).round(0).to_string())

    dd = topology(dd)
    print("\n== measured machines per run, by scope ==")
    h = dd.groupby("scope").agg(runs=("run_id", "nunique"),
                                median_hosts=("hosts", "median"),
                                max_hosts=("hosts", "max"),
                                multinode=("multinode", "any"))
    for k, r in h.iterrows():
        print(f"  {str(k):22s} runs {int(r.runs):5d}  hosts/run median "
              f"{r.median_hosts:4.0f} max {int(r.max_hosts):3d}  "
              f"counted as multi-machine: {bool(r.multinode)}")

    mn = dd[dd.multinode]
    block("MULTI-MACHINE ONLY (multi-node GPU / multi-host TPU)", mn)
    p2 = mn.pivot_table(index="hw", columns="cls", values="min", aggfunc="sum").fillna(0)
    p2["total"] = p2.sum(axis=1)
    print("\n== multi-machine stream-minutes, hardware x class ==")
    print(p2.sort_values("total", ascending=False).round(0).to_string())
    print("\n== multi-machine, by scope ==")
    g4 = mn.groupby("scope").agg(wkl=("base", "nunique"), runs=("run_id", "nunique"),
                                 streams=("file", "size"), minutes=("min", "sum"))
    for k, r in g4.sort_values("minutes", ascending=False).iterrows():
        print(f"  {str(k):22s} {int(r.wkl):4d} workloads {int(r.runs):5d} runs "
              f"{int(r.streams):6d} streams {r.minutes:9,.0f} stream-min")
    print("\n== multi-machine, honest vs adversarial ==")
    for k, r in mn.groupby("kind").agg(wkl=("base", "nunique"), runs=("run_id", "nunique"),
                                       streams=("file", "size"),
                                       minutes=("min", "sum")).iterrows():
        print(f"  {str(k):22s} {int(r.wkl):4d} workloads {int(r.runs):5d} runs "
              f"{int(r.streams):6d} streams {r.minutes:9,.0f} stream-min")

    w = dd.groupby("run_id").agg(t0=("t0", "min"), t1=("t1", "max"))
    print(f"\nwall-clock across distinct runs: {((w.t1 - w.t0) / 3600).sum():,.0f} h "
          f"against {dd['min'].sum() / 60:,.0f} stream-h "
          f"({pd.to_datetime(dd.t0.min(), unit='s'):%Y-%m-%d} .. "
          f"{pd.to_datetime(dd.t1.max(), unit='s'):%Y-%m-%d})")
    return dd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None, help="write the per-stream table here")
    a = ap.parse_args()
    files = corpus_files()
    print(f"{len(files):,} parquet files", flush=True)
    d, bad = scan(files)
    print(f"usable {len(d):,} files, unreadable/empty {bad}")
    dd = report(d)
    if a.out:
        d.to_parquet(a.out, index=False)
        dd.to_parquet(a.out.replace(".parquet", "_dedup.parquet"), index=False)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
