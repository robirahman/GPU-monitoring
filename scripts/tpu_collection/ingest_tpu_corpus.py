#!/usr/bin/env python3
"""Pull TPU cell telemetry out of GCS into data/tpu/, trimming hang tails.

WHY THE TRIM EXISTS. A cell's exit status and its telemetry are independent. On
the 2026-09-01 multislice run, `nonml_comm` ran all 1078 of its slots and then
hung at `jax.distributed` teardown until the cell timeout fired; the MN-I cells
at F>0 deadlocked after ~600 s of correct interleaved training. In every such
case the collector kept sampling an IDLE chip for the remainder of the cell, so
the parquet ends with minutes of zeros that belong to no workload:

    tpuTMS_advtrain_satint_f20_rep1 w0: 1284 rows / 1283 s, of which the real
    interleave runs t=124..~700 s at util 100%, and the tail drags the
    gpu_utilization_pct median from 100 down to 0.

Averaged in, that tail would make a disguised-training cell look idle. So the
usable span is bounded by the workload's own STEP lines, which the worker log
carries, not by the end of the parquet.

Filenames are preserved exactly as collected, because
threeway_improved._normalize_columns recovers the workload label by splitting the
stem on `_Google_TPU_` -- rename them and the label silently becomes garbage.

Usage:
    python3 scripts/tpu_collection/ingest_tpu_corpus.py \
        --prefix gs://focus-surfer-494820-g0-telemetry/tpu/camp2_20260901_145215_ms1 \
        --out data/tpu --manifest /tmp/tpu_manifest.csv
    # add --dry-run to see what would be kept without writing anything
"""

import argparse
import csv
import os
import re
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

STEP_TS = re.compile(r"^STEP ts=([0-9.]+)\b")


def sh(args, timeout=600):
    return subprocess.run(args, capture_output=True, text=True, timeout=timeout)


def gcs_ls(prefix):
    out = sh(["gcloud", "storage", "ls", "-r", prefix.rstrip("/") + "/"])
    if out.returncode != 0:
        print(f"ERROR listing {prefix}: {out.stderr.strip()[:200]}", file=sys.stderr)
        return []
    return [l.strip() for l in out.stdout.splitlines() if l.strip().endswith((".parquet", ".log"))]


def step_span(log_text):
    """(first, last) STEP epoch seconds, or (None, None) if the cell emitted none."""
    ts = [float(m.group(1)) for m in
          (STEP_TS.match(l) for l in log_text.splitlines()) if m]
    return (min(ts), max(ts)) if ts else (None, None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--prefix", required=True, help="GCS prefix holding runs/")
    ap.add_argument("--out", default="data/tpu")
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--pad-sec", type=float, default=2.0,
                    help="keep this much telemetry past the last STEP")
    ap.add_argument("--min-rows", type=int, default=60,
                    help="drop a stream with fewer usable rows than this")
    ap.add_argument("--min-util-med", type=float, default=5.0,
                    help="drop a non-idle cell whose kept span is essentially "
                         "idle (see the compile-then-hang note below); 0 disables")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    objs = gcs_ls(a.prefix)
    parquets = [o for o in objs if o.endswith(".parquet")]
    logs = {o.rsplit("/", 1)[0] + "/" + o.rsplit("/", 1)[1]: o
            for o in objs if o.endswith(".log")}
    print(f"{len(parquets)} parquet(s), {len(logs)} log(s) under {a.prefix}")
    # ONE bulk download instead of per-object calls. The first version issued a
    # `storage cat` for the log of every parquet plus a `cp` for the parquet
    # itself -- and on a v5litepod-8 each cell writes EIGHT per-chip parquets that
    # all share one worker log, so an 11-cell rep meant ~176 sequential gcloud
    # invocations and the run hit a 10-minute wall before finishing. gcloud
    # parallelises a recursive copy internally, and the whole payload is a few MB.
    if not a.dry_run:
        os.makedirs(a.out, exist_ok=True)

    rows_out = []
    with tempfile.TemporaryDirectory() as td:
        bulk = sh(["gcloud", "storage", "cp", "-r",
                   a.prefix.rstrip("/") + "/runs", td], timeout=3000)
        if bulk.returncode != 0:
            print(f"ERROR bulk download: {bulk.stderr.strip()[:300]}", file=sys.stderr)
            return
        local_of = {}
        for root, _dirs, files in os.walk(td):
            for fn in files:
                local_of.setdefault(fn, os.path.join(root, fn))
        print(f"downloaded {len(local_of)} object(s)")

        span_cache = {}
        for pq in sorted(parquets):
            wdir = pq.rsplit("/", 1)[0]
            stem = pq.rsplit("/", 1)[1][:-len(".parquet")]
            # One worker log serves every chip stream in its directory, so parse
            # it once per directory rather than once per parquet.
            if wdir not in span_cache:
                span_cache[wdir] = (None, None)
                cand = [o for o in logs.values()
                        if o.startswith(wdir + "/") and o.endswith(".log")]
                if cand:
                    lp = local_of.get(cand[0].rsplit("/", 1)[1])
                    if lp and os.path.exists(lp):
                        with open(lp, "r", errors="replace") as fh:
                            span_cache[wdir] = step_span(fh.read())
            span = span_cache[wdir]
            local = local_of.get(stem + ".parquet")
            if not local or not os.path.exists(local):
                print(f"  SKIP {stem}: not in the bulk download")
                continue
            d = pd.read_parquet(local)
            n0 = len(d)
            if "timestamp_epoch" not in d.columns:
                print(f"  SKIP {stem}: no timestamp_epoch")
                continue
            first, last = span
            if last is None:
                # No STEP lines at all: the workload never produced ground truth,
                # so there is no defensible span. Keep nothing rather than guess.
                print(f"  DROP {stem}: no STEP lines in the worker log ({n0} rows)")
                rows_out.append(dict(stem=stem, kept=0, dropped=n0, reason="no STEP lines"))
                continue
            keep = d.timestamp_epoch <= (last + a.pad_sec)
            d2 = d[keep].copy()
            u = d2.get("gpu_utilization_pct", pd.Series(dtype=float)).replace(-1, np.nan)
            if len(d2) < a.min_rows:
                print(f"  DROP {stem}: only {len(d2)} usable rows")
                rows_out.append(dict(stem=stem, kept=0, dropped=n0, reason="too few rows"))
                continue
            # Compile-then-hang guard. A cell can emit STEP lines for its first
            # couple of minutes and then wedge, and the span bounded by those
            # STEPs is then almost entirely XLA compilation at ~0% utilisation.
            # Measured on 2026-09-01: tpuTMS_advtrain_satint_f05_rep3 kept 149
            # rows over 148 s at util median 1.6, and f20_rep2 172 rows at util
            # median 0.2 -- while f05_rep1 and f20_rep1 kept ~677 rows at util
            # median 100.0 and are genuinely usable. Admitting the former would
            # teach the detector that MN-I at F=0.05 looks idle, which is the
            # opposite of true, so they are dropped unless asked for. Cells whose
            # label says they ARE idle are exempt.
            util_med = float(u.median()) if len(u.dropna()) else float("nan")
            is_idle_cell = "_idle" in stem
            if (a.min_util_med > 0 and not is_idle_cell
                    and util_med == util_med and util_med < a.min_util_med):
                print(f"  DROP {stem}: kept span is idle (util median "
                      f"{util_med:.1f} < {a.min_util_med}) -- compile-then-hang")
                rows_out.append(dict(stem=stem, kept=0, dropped=n0,
                                     util_med=round(util_med, 2),
                                     reason="idle span (compile-then-hang)"))
                continue
            span_s = float(d2.timestamp_epoch.max() - d2.timestamp_epoch.min())
            live = float(d2.get("tpu_metrics_live", pd.Series([0])).mean())
            print(f"  KEEP {stem}: {len(d2)}/{n0} rows, span {span_s:.0f}s, "
                  f"trimmed {n0-len(d2)} tail rows, util med "
                  f"{(u.median() if len(u) else float('nan')):.1f}, live {live:.2f}")
            rows_out.append(dict(stem=stem, kept=len(d2), dropped=n0 - len(d2),
                                 span_s=round(span_s, 1),
                                 util_med=round(float(u.median()), 2) if len(u) else "",
                                 live=round(live, 3), reason=""))
            if not a.dry_run:
                d2.to_parquet(os.path.join(a.out, stem + ".parquet"), index=False)

    kept = sum(1 for r in rows_out if r["kept"])
    print(f"\n{kept} stream(s) ingested, {len(rows_out)-kept} dropped, "
          f"{sum(r['dropped'] for r in rows_out)} tail rows trimmed"
          + ("  (dry run: nothing written)" if a.dry_run else f" -> {a.out}"))
    if a.manifest and rows_out:
        keys = sorted({k for r in rows_out for k in r})
        with open(a.manifest, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=keys)
            w.writeheader()
            [w.writerow(r) for r in rows_out]
        print(f"manifest -> {a.manifest}")


if __name__ == "__main__":
    main()
