"""DCGM Tier-2 profiling logger — the counterpart to pico_scope_logger.py.

Streams `dcgmi dmon` for the profiling field set (SM/tensor/DRAM/pipe active,
occupancy) plus basic power/util, stamping each sample with wall-clock epoch on
arrival (dmon emits no timestamps), and writes a parquet joinable to the NVML
telemetry and scope envelope by `epoch` / `run_id`.

Requires `nv-hostengine` running (as root, for profiling counters). Fields that
read N/A (warmup, or unsupported) become NaN.

Usage:
  python scripts/dcgm_logger.py --seconds 320 --out run_dcgm.parquet [--gpu 0] [--interval-ms 200]
"""
import argparse
import signal
import subprocess
import sys
import time

import numpy as np
import pandas as pd

# field id -> column name (matches DcgmCollector.PROF_FIELDS + a couple basics)
FIELDS = [
    (1001, "dcgm_gr_engine_active"),
    (1002, "dcgm_sm_active"),
    (1003, "dcgm_sm_occupancy"),
    (1004, "dcgm_tensor_active"),
    (1005, "dcgm_dram_active"),
    (1006, "dcgm_fp64_pipe_active"),
    (1007, "dcgm_fp32_pipe_active"),
    (1008, "dcgm_fp16_pipe_active"),
    (203, "dcgm_gpu_util"),
    (155, "dcgm_power_usage"),
]

STOP = False
def _sig(_n, _f):
    global STOP
    STOP = True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=320.0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval-ms", type=int, default=200)
    ap.add_argument("--run-id", type=str, default="")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    ids = ",".join(str(f[0]) for f in FIELDS)
    names = [f[1] for f in FIELDS]
    cmd = ["dcgmi", "dmon", "-e", ids, "-d", str(args.interval_ms), "-i", str(args.gpu)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            text=True, bufsize=1)

    rows = []
    t0 = time.time()
    try:
        for line in proc.stdout:
            if STOP or (time.time() - t0) >= args.seconds:
                break
            s = line.strip()
            # data rows look like: "GPU 0   0.114  0.091  ..."
            if not s.startswith("GPU"):
                continue
            parts = s.split()
            # parts[0]=="GPU", parts[1]==index, then the field values
            vals = parts[2:]
            if len(vals) < len(FIELDS):
                continue
            rec = {"epoch": time.time()}
            for name, v in zip(names, vals[: len(FIELDS)]):
                rec[name] = np.nan if v in ("N/A", "NA", "") else float(v)
            rows.append(rec)
    finally:
        proc.send_signal(signal.SIGTERM)
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    df = pd.DataFrame(rows)
    if args.run_id:
        df["run_id"] = args.run_id
    df.to_parquet(args.out)
    n = len(df)
    dram = df["dcgm_dram_active"].notna().mean() if n else 0
    print(f"wrote {args.out}: {n} samples, dram_active non-NaN frac {dram:.2f}", flush=True)


if __name__ == "__main__":
    main()
