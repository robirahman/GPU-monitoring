#!/usr/bin/env python3
"""Explore and verify collected GPU telemetry parquet files.

Produces a plain-text summary report covering:
  1. File / run inventory
  2. Per-workload aggregate statistics
  3. Workload signature comparison
  4. Data quality checks
"""

import argparse
import os
import sys
from pathlib import Path

import pandas as pd
import numpy as np

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DCGM_PROFILING_COLS = [
    "dcgm_tensor_active",
    "dcgm_fp16_pipe_active",
    "dcgm_fp32_pipe_active",
    "dcgm_fp64_pipe_active",
    "dcgm_sm_active",
    "dcgm_sm_occupancy",
    "dcgm_dram_active",
]

KEY_METRICS = [
    "gpu_utilization_pct",
    "mem_used_mb",
    "power_draw_w",
    "temperature_c",
]

SEPARATOR = "=" * 88


def _fmt(val, width=10):
    """Format a numeric value for table display."""
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A".rjust(width)
    if isinstance(val, float):
        return f"{val:.2f}".rjust(width)
    return str(val).rjust(width)


def _print_table(headers, rows, col_widths=None):
    """Print a plain-text table with column alignment."""
    if col_widths is None:
        col_widths = []
        for i, h in enumerate(headers):
            max_w = len(str(h))
            for row in rows:
                max_w = max(max_w, len(str(row[i])))
            col_widths.append(max_w + 2)

    header_line = "".join(str(h).ljust(w) for h, w in zip(headers, col_widths))
    print(header_line)
    print("-" * len(header_line))
    for row in rows:
        print("".join(str(v).ljust(w) for v, w in zip(row, col_widths)))


def _load_registry_labels():
    """Try to import workload labels from workloads/registry.py."""
    try:
        repo_root = Path(__file__).resolve().parent.parent
        sys.path.insert(0, str(repo_root))
        from workloads.registry import list_workloads
        return set(list_workloads())
    except Exception as exc:
        print(f"  [warning] Could not import workload registry: {exc}")
        return None


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_all_parquets(data_dir: str):
    """Load all .parquet files from *data_dir* (non-recursive by default,
    but also checks one level of subdirectories)."""
    data_path = Path(data_dir)
    files = list(data_path.glob("*.parquet"))
    # Also check immediate subdirectories
    files.extend(data_path.glob("*/*.parquet"))
    if not files:
        return pd.DataFrame(), []

    frames = []
    loaded_files = []
    for f in sorted(files):
        try:
            df = pd.read_parquet(f)
            df["_source_file"] = f.name
            frames.append(df)
            loaded_files.append(f)
        except Exception as exc:
            print(f"  [warning] Skipping {f.name}: {exc}")

    if not frames:
        return pd.DataFrame(), []
    return pd.concat(frames, ignore_index=True), loaded_files


# ---------------------------------------------------------------------------
# Section 1 — File / run inventory
# ---------------------------------------------------------------------------

def section_inventory(df, loaded_files):
    print(SEPARATOR)
    print("SECTION 1: FILE & RUN INVENTORY")
    print(SEPARATOR)
    print(f"\nTotal parquet files loaded: {len(loaded_files)}")
    print(f"Total samples (rows):      {len(df)}")
    print(f"Columns:                   {len(df.columns)}")
    print()

    # Determine which DCGM profiling columns are present
    dcgm_present = [c for c in DCGM_PROFILING_COLS if c in df.columns]
    has_dcgm_profiling = len(dcgm_present) > 0
    print(f"DCGM profiling columns present: {', '.join(dcgm_present) if dcgm_present else 'None'}")
    print()

    # Build per-workload / per-run summary
    if "workload_label" not in df.columns:
        print("  [error] Column 'workload_label' not found — cannot group by workload.\n")
        return

    # Try to compute duration from timestamp columns
    time_col = None
    for tc in ("timestamp_epoch", "timestamp_utc"):
        if tc in df.columns:
            time_col = tc
            break

    rows = []
    group_cols = ["workload_label"]
    if "run_id" in df.columns:
        group_cols.append("run_id")

    for wl, wl_df in sorted(df.groupby("workload_label")):
        if "run_id" in wl_df.columns:
            n_runs = wl_df["run_id"].nunique()
            samples_list = [g.shape[0] for _, g in wl_df.groupby("run_id")]
            samples_str = f"{min(samples_list)}-{max(samples_list)}" if min(samples_list) != max(samples_list) else str(samples_list[0])
        else:
            n_runs = wl_df["_source_file"].nunique()
            samples_str = str(len(wl_df) // max(n_runs, 1))

        # Duration
        dur_str = "N/A"
        if time_col == "timestamp_epoch" and "run_id" in wl_df.columns:
            durations = []
            for _, rdf in wl_df.groupby("run_id"):
                ts = pd.to_numeric(rdf["timestamp_epoch"], errors="coerce")
                dur = ts.max() - ts.min()
                if not np.isnan(dur):
                    durations.append(dur)
            if durations:
                avg_dur = np.mean(durations)
                dur_str = f"{avg_dur:.0f}s"
        elif time_col == "timestamp_utc" and "run_id" in wl_df.columns:
            durations = []
            for _, rdf in wl_df.groupby("run_id"):
                ts = pd.to_datetime(rdf["timestamp_utc"], errors="coerce")
                dur = (ts.max() - ts.min()).total_seconds()
                if not np.isnan(dur):
                    durations.append(dur)
            if durations:
                avg_dur = np.mean(durations)
                dur_str = f"{avg_dur:.0f}s"

        gpu_name = wl_df["gpu_name"].iloc[0] if "gpu_name" in wl_df.columns else "N/A"
        # If multiple GPU names, list them
        if "gpu_name" in wl_df.columns:
            gpu_names = wl_df["gpu_name"].unique()
            if len(gpu_names) > 1:
                gpu_name = ", ".join(sorted(str(g) for g in gpu_names))

        wl_dcgm = any(c in wl_df.columns and wl_df[c].notna().any() for c in DCGM_PROFILING_COLS)

        rows.append([wl, n_runs, samples_str, dur_str, gpu_name, "Yes" if wl_dcgm else "No"])

    headers = ["Workload", "Runs", "Samples/Run", "Avg Dur", "GPU(s)", "DCGM Prof"]
    _print_table(headers, rows)
    print()


# ---------------------------------------------------------------------------
# Section 2 — Per-workload aggregate statistics
# ---------------------------------------------------------------------------

def section_aggregate_stats(df):
    print(SEPARATOR)
    print("SECTION 2: PER-WORKLOAD AGGREGATE STATISTICS")
    print(SEPARATOR)

    if "workload_label" not in df.columns:
        print("  [error] Column 'workload_label' not found.\n")
        return

    dcgm_present = [c for c in DCGM_PROFILING_COLS if c in df.columns]

    for wl, wl_df in sorted(df.groupby("workload_label")):
        print(f"\n--- {wl} ({len(wl_df)} samples) ---")

        # Key metrics
        stat_headers = ["Metric", "Mean", "Std", "Min", "Max"]
        stat_rows = []
        for m in KEY_METRICS:
            if m not in wl_df.columns:
                stat_rows.append([m, "N/A", "N/A", "N/A", "N/A"])
                continue
            s = pd.to_numeric(wl_df[m], errors="coerce")
            stat_rows.append([
                m,
                _fmt(s.mean()),
                _fmt(s.std()),
                _fmt(s.min()),
                _fmt(s.max()),
            ])

        _print_table(stat_headers, stat_rows, col_widths=[25, 12, 12, 12, 12])

        # DCGM profiling means
        dcgm_with_data = [c for c in dcgm_present if wl_df[c].notna().any()]
        if dcgm_with_data:
            print(f"\n  DCGM profiling means:")
            for c in dcgm_with_data:
                s = pd.to_numeric(wl_df[c], errors="coerce")
                print(f"    {c:30s}  mean={s.mean():.4f}  std={s.std():.4f}")
    print()


# ---------------------------------------------------------------------------
# Section 3 — Workload signature comparison
# ---------------------------------------------------------------------------

def section_comparison(df):
    print(SEPARATOR)
    print("SECTION 3: WORKLOAD SIGNATURE COMPARISON")
    print(SEPARATOR)

    if "workload_label" not in df.columns:
        print("  [error] Column 'workload_label' not found.\n")
        return

    compare_metrics = [
        "gpu_utilization_pct",
        "mem_used_mb",
        "power_draw_w",
        "temperature_c",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "encoder_util_pct",
        "decoder_util_pct",
    ]
    # Only include metrics that actually exist
    compare_metrics = [m for m in compare_metrics if m in df.columns]

    headers = ["Workload", "Runs"] + [m.replace("_pct", "%").replace("_w", "W")
                                        .replace("_mb", "MB").replace("_c", "C")
                                        .replace("_mhz", "MHz").replace("_mbps", "Mbps")
                                        for m in compare_metrics]
    rows = []
    incomplete = []

    for wl, wl_df in sorted(df.groupby("workload_label")):
        n_runs = wl_df["run_id"].nunique() if "run_id" in wl_df.columns else wl_df["_source_file"].nunique()
        row = [wl, n_runs]
        for m in compare_metrics:
            s = pd.to_numeric(wl_df[m], errors="coerce")
            row.append(f"{s.mean():.1f}")
        rows.append(row)
        if n_runs < 3:
            incomplete.append((wl, n_runs))

    _print_table(headers, rows)
    print()

    if incomplete:
        print("  ** WARNING: The following workloads have fewer than 3 runs (collection may be incomplete):")
        for wl, n in incomplete:
            print(f"     - {wl}: {n} run(s)")
        print()


# ---------------------------------------------------------------------------
# Section 4 — Data quality checks
# ---------------------------------------------------------------------------

def section_quality(df, data_dir):
    print(SEPARATOR)
    print("SECTION 4: DATA QUALITY CHECKS")
    print(SEPARATOR)

    issues_found = 0

    # 4a — NaN-heavy columns
    print("\n4a. Columns with >50% NaN values:")
    nan_counts = df.isna().sum()
    total = len(df)
    nan_heavy = nan_counts[nan_counts > total * 0.5].sort_values(ascending=False)
    if len(nan_heavy) > 0:
        for col, cnt in nan_heavy.items():
            if col == "_source_file":
                continue
            pct = cnt / total * 100
            print(f"    {col:40s}  {cnt:>7d} / {total}  ({pct:.1f}%)")
            issues_found += 1
    else:
        print("    None — all columns have <=50% NaN values.")

    # 4b — Zero-variance columns (excluding constant metadata)
    print("\n4b. Zero-variance numeric columns:")
    metadata_cols = {"workload_label", "run_id", "gpu_name", "gpu_uuid",
                     "driver_version", "_source_file", "timestamp_utc"}
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    zero_var = []
    for c in numeric_cols:
        if c in metadata_cols:
            continue
        s = df[c].dropna()
        if len(s) > 0 and s.std() == 0:
            zero_var.append((c, s.iloc[0] if len(s) > 0 else "N/A"))
    if zero_var:
        for col, val in zero_var:
            print(f"    {col:40s}  constant value = {val}")
            issues_found += 1
    else:
        print("    None — all numeric columns have nonzero variance.")

    # 4c — Suspiciously short runs
    print("\n4c. Suspiciously short runs (<10 samples):")
    short_runs = []
    if "workload_label" in df.columns and "run_id" in df.columns:
        for (wl, rid), g in df.groupby(["workload_label", "run_id"]):
            if len(g) < 10:
                short_runs.append((wl, rid, len(g)))
    if short_runs:
        for wl, rid, n in short_runs:
            print(f"    {wl:40s}  run_id={rid}  samples={n}")
            issues_found += 1
    else:
        print("    None — all runs have >= 10 samples.")

    # 4d — Expected workload types from registry
    print("\n4d. Workload registry coverage:")
    registry_labels = _load_registry_labels()
    if registry_labels is not None and "workload_label" in df.columns:
        collected = set(df["workload_label"].unique())
        missing = sorted(registry_labels - collected)
        extra = sorted(collected - registry_labels)
        if missing:
            print(f"    Missing from data ({len(missing)}):")
            for m in missing:
                print(f"      - {m}")
            issues_found += len(missing)
        else:
            print("    All registered workloads have data collected.")
        if extra:
            print(f"    In data but not in registry ({len(extra)}):")
            for e in extra:
                print(f"      - {e}")
    elif registry_labels is None:
        print("    Could not load registry — skipping check.")
    else:
        print("    No workload_label column — skipping check.")

    print(f"\nTotal quality issues flagged: {issues_found}")
    print()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Explore and verify collected GPU telemetry parquet files."
    )
    parser.add_argument(
        "--data-dir",
        default="data/",
        help="Directory containing .parquet files (default: data/)",
    )
    args = parser.parse_args()

    data_dir = args.data_dir
    if not os.path.isabs(data_dir):
        # Resolve relative to repo root (parent of scripts/)
        repo_root = Path(__file__).resolve().parent.parent
        data_dir = str(repo_root / data_dir)

    print(SEPARATOR)
    print("GPU TELEMETRY DATA EXPLORATION REPORT")
    print(SEPARATOR)
    print(f"Data directory: {data_dir}")
    print()

    if not os.path.isdir(data_dir):
        print(f"[error] Data directory does not exist: {data_dir}")
        sys.exit(1)

    df, loaded_files = load_all_parquets(data_dir)
    if df.empty:
        print("[error] No parquet files found or all files failed to load.")
        sys.exit(1)

    section_inventory(df, loaded_files)
    section_aggregate_stats(df)
    section_comparison(df)
    section_quality(df, data_dir)

    print(SEPARATOR)
    print("END OF REPORT")
    print(SEPARATOR)


if __name__ == "__main__":
    main()
