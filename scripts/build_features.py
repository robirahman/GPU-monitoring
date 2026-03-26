#!/usr/bin/env python3
import argparse
import numpy as np
import pandas as pd

FEATURE_COLS = [
    "gpu_util_pct",
    "mem_util_pct",
    "mem_used_mb",
    "power_w",
    "temp_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
]

KEEP_LABELS = ["idle", "compute_bound", "mem_stream", "memcpy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--infile", required=True, help="Input labeled file (*_labeled.parquet or .csv)")
    ap.add_argument("--out", default="", help="Output features file (.parquet or .csv). Default: *_features.parquet")
    args = ap.parse_args()

    # Load
    if args.infile.endswith(".csv"):
        df = pd.read_csv(args.infile)
    else:
        df = pd.read_parquet(args.infile)

    # Timestamp
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True, errors="coerce")

    # Keep only the labeled phases we care about
    if "label" not in df.columns:
        raise SystemExit("Input must contain a 'label' column. Did you run the labeler first?")
    df = df[df["label"].isin(KEEP_LABELS)].copy()

    # Ensure numeric types for base features
    for c in FEATURE_COLS + ["mem_total_mb"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    # Derived features
    df["mem_used_frac"] = df["mem_used_mb"] / df["mem_total_mb"]

    # power_per_util: power_w / gpu_util_pct, but avoid div-by-zero and NAType issues robustly
    util = df["gpu_util_pct"].astype(float).to_numpy()
    power = df["power_w"].astype(float).to_numpy()

    den = util.copy()
    den[den == 0] = np.nan
    p = power / den
    df["power_per_util"] = pd.Series(p).fillna(0.0).astype(float)

    # is_busy: 1 for “active workload” phases, 0 for idle
    df["is_busy"] = df["label"].isin(["compute_bound", "mem_stream", "memcpy"]).astype(int)

    # Output columns
    out_df = df[
        ["ts_utc", "gpu_index"]
        + FEATURE_COLS
        + ["mem_used_frac", "power_per_util", "label", "is_busy"]
    ].copy()

    # Default output name
    import os

    # Default output name
    if not args.out:
        base = args.infile
        if base.endswith(".csv"):
            base = base[:-4]
        elif base.endswith(".parquet"):
            base = base[:-8]
        # convert *_labeled -> *_features
        if base.endswith("_labeled"):
            base = base[:-8]
        args.out = base + "_features.parquet"
   # if not args.out:
   #     if args.infile.endswith(".csv"):
  #          args.out = args.infile.replace("_labeled.csv", "_features.parquet").replace(".csv", "_features.parquet")
 #       else:
#            args.out = args.infile.replace("_labeled.parquet", "_features.parquet").replace(".parquet", "_features.parquet")

    # Save
    if args.out.endswith(".csv"):
        out_df.to_csv(args.out, index=False)
    else:
        out_df.to_parquet(args.out, index=False)

    print("Wrote:", args.out)
    print("\nLabel counts:")
    print(out_df["label"].value_counts().to_string())


if __name__ == "__main__":
    main()
