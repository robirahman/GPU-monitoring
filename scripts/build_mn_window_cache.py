#!/usr/bin/env python3
"""Build the MULTI-NODE window cache: NIC sidecar + cross-rank features.

Why this exists
---------------
`build_window_cache.py` calls `sliding_windows`, the per-stream featuriser. It
has no inter-node traffic and no cross-rank structure, because `load_all_data`
deliberately skips `nic_*.parquet` and nothing in that path ever calls
`attach_nic_sidecar` or `cross_rank_features`.

So every multi-node conclusion this project has drawn -- including the
2026-09-23 leave-one-family-out -- was computed WITHOUT the signals that
separate distributed training from distributed inference. 313 nic_*.parquet
files sit in the corpus unread, `sliding_windows_multinode` and
`cross_rank_features` exist and are fully implemented, and `stage2_mask_multinode`
already knows how to classify `xr_*` names as shape rather than level. Only the
wiring was missing; `scope_ladder_eval.py` was the sole caller.

That matters because the measured spread is enormous. Median inter-node
throughput over multi-node cells:

    train_fsdp  16077 Mbps        inferrec_dlrm  5907 Mbps
    contrastive  1947             nonml (hpc)    2835
    diffusion     580             inferserve       93
    pipe          513             infershard        37
    ddp           184
    lora            0.003

FSDP against sharded LLM inference is ~440x, which is the separation this
project has been relying on. But the 2026-09-23 round broke the heuristic: the
new HPC and recommender cells are louder than four of the six training families,
and LoRA is silent. So these features are necessary but plainly not sufficient,
and wiring them in is the only way to measure which.

Granularity
-----------
`sliding_windows_multinode` emits ONE row per (job, window) with rank-0's
single-stream features plus the cross-rank block, where `sliding_windows` emits
one row per (stream, window). Fewer, wider rows. This writes a SEPARATE cache
rather than replacing win_full.parquet, because the per-stream cache is what
every other analysis consumes and the two are not interchangeable.

Chunking
--------
Workers are chunked by job_id, never by stream. `assign_stream_and_job_ids`
reconstructs legacy jobs from (base_label, gpu, time bucket), so splitting a job
across workers would reassign its ranks to different job_ids and silently
destroy exactly the cross-rank structure this cache exists to capture.

    PYTHONPATH=. python3 scripts/build_mn_window_cache.py OUT [win] [stride] [nproc]
"""
import gc
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

WORK = os.environ.get("SPAR_WORK", "/var/tmp/spar_analysis")
RAW = os.path.join(WORK, "raw_mn.parquet")


def rss():
    import resource
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1e6


def job_chunk(args):
    idx, jobs, win, stride = args
    from classifier.multinode_features import sliding_windows_multinode
    t = pq.read_table(RAW, filters=[("job_id", "in", list(jobs))]).to_pandas()
    w = sliding_windows_multinode(t, win, stride)
    out = os.path.join(WORK, f"mnpart_{idx:03d}.parquet")
    if w.empty:
        return idx, len(jobs), 0, None
    w.to_parquet(out, index=False)
    return idx, len(jobs), len(w), out


def main():
    out = sys.argv[1]
    win = float(sys.argv[2]) if len(sys.argv) > 2 else 30.0
    stride = float(sys.argv[3]) if len(sys.argv) > 3 else 15.0
    nproc = int(sys.argv[4]) if len(sys.argv) > 4 else 4
    os.makedirs(WORK, exist_ok=True)
    t0 = time.time()

    if not os.path.exists(RAW):
        from classifier.threeway_improved import load_all_data
        from classifier.multinode_features import (
            attach_nic_sidecar, normalize_comm_columns, assign_stream_and_job_ids,
        )
        df = load_all_data()
        print(f"[{time.time()-t0:.0f}s] raw rows {len(df):,}  peakRSS {rss():.2f} GB", flush=True)

        before = [c for c in df.columns if c.startswith("nic_")]
        df = attach_nic_sidecar(df)
        after = [c for c in df.columns if c.startswith("nic_")]
        print(f"  nic sidecar: {len(before)} -> {len(after)} nic columns {after}", flush=True)
        if not after:
            # Loud, because a silent no-op here reproduces exactly the bug this
            # script exists to fix -- a cache that looks fine and has no NIC in it.
            print("  WARNING: no nic_* columns after attach_nic_sidecar. Check that "
                  "nic_*.parquet exist under data/ and carry run_id + timestamp_epoch.",
                  flush=True)

        df = normalize_comm_columns(df)
        comm = [c for c in ("nic_total_mbps", "nvlink_total_mbps", "pcie_total_mbps")
                if c in df.columns]
        print(f"  comm channels present: {comm}", flush=True)

        df = assign_stream_and_job_ids(df)
        nz = float((df["nic_total_mbps"] > 0).mean()) if "nic_total_mbps" in df else 0.0
        print(f"  rows with nic_total_mbps > 0: {nz:.1%}", flush=True)

        for c in df.columns:
            if c == "ts":
                continue
            if df[c].dtype == np.float64:
                df[c] = df[c].astype(np.float32)
        df["job_id"] = df["job_id"].astype(str)
        df.to_parquet(RAW, index=False)
        print(f"[{time.time()-t0:.0f}s] spilled {os.path.getsize(RAW)/1e9:.2f} GB", flush=True)
        del df
        gc.collect()

    jobs = pq.read_table(RAW, columns=["job_id"]).to_pandas().job_id.astype(str).unique()
    np.random.default_rng(0).shuffle(jobs)
    chunks = [(i, list(c), win, stride)
              for i, c in enumerate(np.array_split(jobs, nproc * 3), 1)]
    print(f"[{time.time()-t0:.0f}s] {len(jobs):,} jobs in {len(chunks)} chunks "
          f"across {nproc} workers", flush=True)

    parts = []
    with mp.Pool(nproc, maxtasksperchild=1) as pool:
        for idx, nj, nw, p in pool.imap_unordered(job_chunk, chunks):
            if p:
                parts.append(p)
            print(f"[{time.time()-t0:.0f}s] chunk {idx}: {nj:,} jobs -> {nw:,} windows",
                  flush=True)

    if not parts:
        raise SystemExit("no windows produced")

    from classifier.threeway_improved import workload_group
    w = pd.concat([pd.read_parquet(p) for p in sorted(parts)], ignore_index=True)
    # sliding_windows_multinode sets run_id = job_id by design, so detection
    # groups by physical job. mn_binary_lofo wants both names present.
    w["job_id"] = w["run_id"]
    w["grp"] = workload_group(w.workload_label.values)
    w.to_parquet(out, index=False)

    xr = [c for c in w.columns if c.startswith("xr_")]
    xrg = [c for c in w.columns if c.startswith("xragg_")]
    print(f"[{time.time()-t0:.0f}s] windows {len(w):,} -> {out}", flush=True)
    print(f"  jobs {w.job_id.nunique():,}  groups {w.grp.nunique():,}  "
          f"xr_ features {len(xr)}  xragg_ features {len(xrg)}  peakRSS {rss():.2f} GB",
          flush=True)
    print(w.threeway_label.value_counts().to_string(), flush=True)
    if not xr:
        raise SystemExit("REFUSING: no xr_ features in the output -- the cross-rank "
                         "join produced nothing, which is the bug this script fixes")


if __name__ == "__main__":
    main()
