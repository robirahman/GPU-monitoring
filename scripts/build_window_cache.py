#!/usr/bin/env python3
"""Build the per-stream sliding-window cache once, in parallel, without OOMing.

Every analysis downstream wants the same thing: the whole corpus windowed PER
STREAM with `job_id` attached. Doing that inside each analysis script costs ~25
minutes and a 12.4 GB peak, and doing it naively does two things wrong:

  1. `sliding_windows` groups by `run_id`, and both the TPU queue and the
     multi-node GPU queue deliberately mandate ONE shared run_id across every
     worker of a cell -- all 16 chips of tpuT16_advtrain_satfiller_only_rep1
     carry run_id=c106c001eec0 with 16 distinct gpu_uuids. Grouping by run_id
     interleaves 16 chips into one series, so a 30 s window holds ~480 samples
     from 16 devices instead of 30 from one and every acf / fft / transition
     feature becomes a chip-ordering artifact. 559 of 2,693 run_ids span more
     than one stream.
  2. Holding the raw frame and the growing list of ~230k window dicts at the
     same time climbs past this box's memory. Measured: the one-shot version was
     at 12.2 GB RSS with 1 GB free after 80 minutes and had not finished.

So: load once, spill a downcast copy, then window stream-subsets in worker
processes that each read back only their own rows.

  PYTHONPATH=. python3 scripts/build_window_cache.py OUT.parquet [win] [stride] [nproc]
"""
import gc
import multiprocessing as mp
import os
import resource
import subprocess
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

WORK = os.environ.get("SPAR_WORK", "/var/tmp/spar_analysis")
RAW = os.path.join(WORK, "raw_downcast.parquet")

# /var/tmp is wiped on every VM restart -- three times so far, twice destroying a
# 25-minute build and once taking the only copy of an analysis script with it.
# /home is a 4.8 GB disk with ~700 MB free, so it cannot hold the 111 MB cache
# plus its spill. GCS is the only durable option, and a 111 MB download beats a
# 25-minute rebuild by two orders of magnitude.
CACHE_GCS = os.environ.get(
    "SPAR_CACHE_GCS",
    "")


def gcs_pull(out):
    """Restore a previously built cache. Returns True if `out` now exists."""
    if os.path.exists(out) or not CACHE_GCS:
        return os.path.exists(out)
    url = f"{CACHE_GCS}/{os.path.basename(out)}"
    rc = subprocess.call(["gsutil", "-q", "cp", url, out],
                         stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if rc == 0 and os.path.exists(out):
        print(f"restored {out} from {url}", flush=True)
        return True
    return False


def gcs_push(*paths):
    """Best-effort upload. A failed upload must never fail the build."""
    if not CACHE_GCS:
        return
    for p in paths:
        if not os.path.exists(p):
            continue
        rc = subprocess.call(["gsutil", "-q", "cp", p, f"{CACHE_GCS}/"],
                             stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        print(f"{'uploaded' if rc == 0 else 'UPLOAD FAILED for'} {p}", flush=True)


def rss():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def window_chunk(args):
    idx, streams, win, stride = args
    from classifier.threeway_improved import sliding_windows
    t = pq.read_table(RAW, filters=[("run_id", "in", list(streams))]).to_pandas()
    w = sliding_windows(t, win, stride)
    out = os.path.join(WORK, f"part_{idx:03d}.parquet")
    w.to_parquet(out, index=False)
    return idx, len(streams), len(w), out


def main():
    out = sys.argv[1]
    win = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    stride = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 3
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()
    from classifier.cache_provenance import provenance, cache_matches, stamp_cache
    manifest = provenance(window_sec=win, stride_sec=stride, granularity="per_stream")
    if cache_matches(out, manifest):
        print(f"validated cache {out}", flush=True)
        return

    if not cache_matches(RAW, manifest):
        from classifier.threeway_improved import load_all_data
        from classifier.multinode_features import assign_stream_and_job_ids
        df = load_all_data()
        print(f"[{time.time()-t0:.0f}s] raw rows {len(df):,}  peakRSS {rss():.2f} GB",
              flush=True)
        df = assign_stream_and_job_ids(df)
        mapping = df[["stream_id", "job_id", "node_id"]].drop_duplicates()
        hosts = mapping.groupby("job_id").node_id.nunique()
        mapping["meta_n_hosts"] = mapping.job_id.map(hosts)
        mapping.drop(columns="node_id").to_parquet(
            os.path.join(WORK, "stream_to_job.parquet"), index=False)
        df["run_id"] = df["stream_id"]
        before = df.memory_usage(deep=True).sum() / 1e9
        for c in df.columns:
            if c == "ts":
                continue                       # epoch seconds need float64
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
            elif df[c].dtype == object or str(df[c].dtype) in ("str", "string"):
                df[c] = df[c].astype("category")
        print(f"  frame {before:.2f} -> {df.memory_usage(deep=True).sum()/1e9:.2f} GB; "
              f"streams {df.run_id.nunique():,}", flush=True)
        df["run_id"] = df["run_id"].astype(str)    # categorical breaks pq filters
        df.to_parquet(RAW, index=False)
        stamp_cache(RAW, manifest)
        print(f"[{time.time()-t0:.0f}s] spilled {os.path.getsize(RAW)/1e9:.2f} GB", flush=True)
        del df
        gc.collect()

    streams = pq.read_table(RAW, columns=["run_id"]).to_pandas().run_id.astype(str).unique()
    np.random.default_rng(0).shuffle(streams)      # even chunk cost
    chunks = [(i, list(c), win, stride)
              for i, c in enumerate(np.array_split(streams, nproc * 3), 1)]
    print(f"[{time.time()-t0:.0f}s] {len(streams):,} streams in {len(chunks)} chunks "
          f"across {nproc} workers", flush=True)

    parts = []
    with mp.Pool(nproc, maxtasksperchild=1) as pool:
        for idx, ns, nw, p in pool.imap_unordered(window_chunk, chunks):
            parts.append(p)
            print(f"[{time.time()-t0:.0f}s] chunk {idx}: {ns:,} streams -> {nw:,} windows",
                  flush=True)

    from classifier.threeway_improved import workload_group
    s2j = pd.read_parquet(os.path.join(WORK, "stream_to_job.parquet"))
    if s2j.stream_id.duplicated().any():
        raise ValueError("Ambiguous stream/job map; rebuild identities")
    m = s2j.set_index("stream_id")["job_id"]
    w = pd.concat([pd.read_parquet(p) for p in sorted(parts)], ignore_index=True)
    w["job_id"] = w.run_id.map(m).fillna(w.run_id)
    # Keep measured topology with the cache, avoiding a second inventory whose
    # fallback stream identities can differ on legacy missing-host/device rows.
    w["meta_n_hosts"] = w.run_id.map(s2j.set_index("stream_id").meta_n_hosts)
    w["grp"] = workload_group(w.workload_label.values)
    w.to_parquet(out, index=False)
    stamp_cache(out, manifest)
    print(f"[{time.time()-t0:.0f}s] windows {len(w):,} -> {out}", flush=True)
    print(f"  streams {w.run_id.nunique():,}  jobs {w.job_id.nunique():,}  "
          f"groups {w.grp.nunique():,}  peakRSS {rss():.2f} GB", flush=True)
    print(w.threeway_label.value_counts().to_string(), flush=True)
    gcs_push(out, out + ".provenance.json", os.path.join(WORK, "stream_to_job.parquet"))


if __name__ == "__main__":
    main()
