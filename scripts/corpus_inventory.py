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
SCOPE = re.compile(r"^(s1|s8|s16|tpuT4|tpuT8|tpuT16|tpuTMS)_")
ADV = re.compile(r"(advtrain|adversarial|whitebox|symgemm|typeconf|stalled|diluted)", re.I)


def corpus_files():
    files = []
    for p in sorted((PROJECT_ROOT / "data").rglob("*.parquet")):
        if p.name.startswith("nic_"):
            continue                       # host-NIC sidecars, not GPU streams
        if any("_quarantine" in part for part in p.parts):
            continue                       # idle-screened streams are not corpus
        files.append(p)
    for sub in ("week4/data", "week4/results/ddp", "week4/results/dataset_scale"):
        d = PROJECT_ROOT / sub
        if d.exists():
            files += sorted(d.rglob("*.parquet"))
    return files


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
        try:
            names = set(pq.ParquetFile(p).schema.names)
            tcol = "ts" if "ts" in names else (
                "timestamp_epoch" if "timestamp_epoch" in names else None)
            if tcol is None:
                bad += 1
                continue
            want = [c for c in (tcol, "gpu_name", "workload_label", "run_id",
                                "gpu_uuid", "hostname") if c in names]
            d = pq.read_table(p, columns=want).to_pandas()
        except Exception:
            bad += 1
            continue
        t = pd.to_numeric(d[tcol], errors="coerce").dropna()
        t = t[(t > 1e8) & (t < 4e9)]           # plausible epoch seconds only
        if len(t) < 2:
            bad += 1
            continue
        gpu = str(d["gpu_name"].iloc[0]) if "gpu_name" in d.columns else ""
        if not gpu or gpu in ("nan", "None"):
            gpu = hw_from_stem(p.stem)
        lab = str(d["workload_label"].iloc[0]) if "workload_label" in d.columns else p.stem
        rid = str(d["run_id"].iloc[0]) if "run_id" in d.columns else p.stem
        uid = str(d["gpu_uuid"].iloc[0]) if "gpu_uuid" in d.columns else p.stem
        span = float(t.max() - t.min())
        rows.append(dict(file=p.stem, gpu=gpu, hw=family(gpu), label=lab,
                         base=RANK.sub("", lab), run_id=rid,
                         stream=f"{rid}|{uid}", rows=len(t), span_s=span,
                         hz=(len(t) - 1) / span if span > 0 else np.nan,
                         t0=float(t.min()), t1=float(t.max())))
        if (i + 1) % 2000 == 0:
            print(f"  {i + 1:,}/{len(files):,}", flush=True)
    return pd.DataFrame(rows), bad


def report(d):
    d = d.copy()
    d["cls"] = d.base.map(threeway_label)
    d["min"] = d.span_s / 60
    d["scope"] = d.base.map(lambda l: (SCOPE.match(str(l)).group(1)
                                       if SCOPE.match(str(l)) else "single-GPU (legacy)"))
    d["kind"] = np.where(d.base.str.contains(ADV, na=False), "adversarial", "honest")

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
    block("deduplicated (the corpus, as load_all_data sees it)", dd)

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
