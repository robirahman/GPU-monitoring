#!/usr/bin/env python3
"""Quick temporal-signal readout for TPU telemetry parquets.

Updated for the current TPU collector schema (scripts/collect_tpu_telemetry.py),
which writes GPU-schema column names so the classifier ingests TPU parquets
unchanged:
    gpu_utilization_pct  <- TPU duty cycle   (0.0 when dark)
    mem_used_mb          <- HBM usage
plus TPU-native communication columns NVIDIA lacks:
    hlo_exec_us_p50/p90/p99            (always-on comm channel)
    dcn_collective_lat_us_p50/p90/p99  (cross-host collective latency; -1 dark)

Usage:
    python evaluate_tpu.py data/tpuT16_train_fsdp7b_w0c0_*.parquet \
                           data/tpuT16_inferrep_7b_w0c0_*.parquet
    python evaluate_tpu.py            # falls back to any data/tpuT*.parquet
"""

import glob
import sys

import numpy as np
import pandas as pd

from classifier.threeway_improved import fft_dominant_period, fft_peak_power, acf_lag


def _series(df, new, *legacy, scale=1.0):
    """Fetch a column by its current name, tolerating the pre-rewrite names."""
    for name in (new, *legacy):
        if name in df.columns:
            return df[name].values.astype(float) * scale
    return None


def analyze_trace(file_path):
    df = pd.read_parquet(file_path)

    # Current schema first; fall back to the old tpu_* names if an archived
    # parquet is passed in.
    util = _series(df, "gpu_utilization_pct", "tpu_utilization_pct")
    mem = _series(df, "mem_used_mb", "tpu_mem_used_gb")  # old col was GB
    if mem is not None and "mem_used_mb" not in df.columns:
        mem = mem * 1024.0  # legacy GB -> MB

    print(f"--- Analysis for {file_path} ---")
    if util is None:
        print("  no utilization column found; skipping")
        print()
        return

    print(f"Rows: {len(df)}  Mean Util: {util.mean():.2f}%  Std Util: {util.std():.2f}%")
    if mem is not None:
        print(f"Mem used (MB): mean={mem.mean():.0f} max={mem.max():.0f}")

    print(f"FFT Dominant Period: {fft_dominant_period(util):.2f} samples")
    print(f"FFT Peak Power:      {fft_peak_power(util):.4f}")
    print(f"ACF(1): {acf_lag(util, 1):.4f}   ACF(5): {acf_lag(util, 5):.4f}")

    # TPU-native communication channel -- the whole point of the TPU rung.
    hlo = _series(df, "hlo_exec_us_p90")
    if hlo is not None and np.any(hlo >= 0):
        live = hlo[hlo >= 0]
        print(f"HLO exec p90 (us): mean={live.mean():.1f}  "
              f"period={fft_dominant_period(live):.2f} samples")
    dcn = _series(df, "dcn_collective_lat_us_p90")
    if dcn is not None and np.any(dcn >= 0):
        live = dcn[dcn >= 0]
        print(f"DCN collective p90 (us): mean={live.mean():.1f} "
              f"(fires on multislice / cross-host collectives)")
    elif dcn is not None:
        print("DCN collective latency: dark (-1) -- single-slice; "
              "HLO exec is the always-on comm channel")
    print()


def main(argv):
    paths = []
    for a in argv:
        paths.extend(sorted(glob.glob(a)))
    if not paths:
        paths = sorted(glob.glob("data/tpuT*.parquet"))
    if not paths:
        print("No TPU parquets found. Pass file paths or populate data/tpuT*.parquet.")
        return
    for p in paths:
        analyze_trace(p)


if __name__ == "__main__":
    main(sys.argv[1:])
