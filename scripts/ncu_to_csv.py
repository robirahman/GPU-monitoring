#!/usr/bin/env python3
"""Convert Nsight Compute .ncu-rep report files to CSV.

Adapted from WAVE's preprocessing pipeline (src/gpu_pmc_analyzer/preprocessor.py).

Runs `ncu --csv --page raw -i <report>` and parses the output into a clean CSV
with per-kernel metrics and derived FLOP counts.

Usage:
    python ncu_to_csv.py report.ncu-rep -o output.csv
"""

import argparse
import logging
import subprocess
import sys
from io import StringIO

import pandas as pd

log = logging.getLogger("spar.ncu_csv")


def ncu_report_to_dataframe(report_path: str) -> pd.DataFrame:
    """Convert an .ncu-rep file to a pandas DataFrame using ncu --csv."""
    # Run ncu to export CSV
    result = subprocess.run(
        ["ncu", "--csv", "--page", "raw", "-i", report_path],
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        log.error("ncu --csv failed (code %d): %s", result.returncode, result.stderr)
        raise RuntimeError(f"ncu --csv failed: {result.stderr}")

    if not result.stdout.strip():
        raise RuntimeError("ncu --csv produced no output")

    # Parse CSV output
    # NCU CSV output may have header comment lines starting with "=="
    lines = result.stdout.strip().split("\n")
    csv_lines = [line for line in lines if not line.startswith("==")]
    csv_text = "\n".join(csv_lines)

    df = pd.read_csv(StringIO(csv_text))
    return df


def compute_derived_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived FLOP and memory operation counts following WAVE's methodology."""

    def safe_col(name: str) -> pd.Series:
        """Get column if it exists, else return zeros."""
        if name in df.columns:
            return pd.to_numeric(df[name], errors="coerce").fillna(0)
        return pd.Series(0, index=df.index)

    # FP32 FLOPs: 2*FMA + MUL + ADD
    df["derived_fp32_flops"] = (
        2 * safe_col("smsp__sass_thread_inst_executed_op_ffma_pred_on.sum")
        + safe_col("smsp__sass_thread_inst_executed_op_fmul_pred_on.sum")
        + safe_col("smsp__sass_thread_inst_executed_op_fadd_pred_on.sum")
    )

    # FP16 FLOPs: 2*FMA + MUL + ADD
    df["derived_fp16_flops"] = (
        2 * safe_col("smsp__sass_thread_inst_executed_op_hfma_pred_on.sum")
        + safe_col("smsp__sass_thread_inst_executed_op_hmul_pred_on.sum")
        + safe_col("smsp__sass_thread_inst_executed_op_hadd_pred_on.sum")
    )

    # FP64 FLOPs: 2*FMA + MUL + ADD
    df["derived_fp64_flops"] = (
        2 * safe_col("smsp__sass_thread_inst_executed_op_dfma_pred_on.sum")
        + safe_col("smsp__sass_thread_inst_executed_op_dmul_pred_on.sum")
        + safe_col("smsp__sass_thread_inst_executed_op_dadd_pred_on.sum")
    )

    # Tensor core FLOPs (sum of all tensor paths)
    tensor_cols = [c for c in df.columns if "tensor_src" in c]
    if tensor_cols:
        df["derived_tensor_ops"] = sum(safe_col(c) for c in tensor_cols)
    else:
        df["derived_tensor_ops"] = 0

    # Global memory operations
    df["derived_global_mem_ops"] = (
        safe_col("smsp__sass_inst_executed_op_global_ld.sum")
        + safe_col("smsp__sass_inst_executed_op_global_st.sum")
    )

    # Shared memory operations
    df["derived_shared_mem_ops"] = (
        safe_col("smsp__sass_inst_executed_op_shared_ld.sum")
        + safe_col("smsp__sass_inst_executed_op_shared_st.sum")
    )

    # Local memory operations
    df["derived_local_mem_ops"] = (
        safe_col("smsp__sass_inst_executed_op_local_ld.sum")
        + safe_col("smsp__sass_inst_executed_op_local_st.sum")
    )

    # L1 cache misses
    df["derived_l1_load_misses"] = safe_col(
        "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum"
    )
    df["derived_l1_store_misses"] = safe_col(
        "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum"
    )

    return df


def main():
    parser = argparse.ArgumentParser(
        description="Convert .ncu-rep report to CSV with derived metrics."
    )
    parser.add_argument("input", type=str, help="Path to .ncu-rep file")
    parser.add_argument("-o", "--output", type=str, default=None, help="Output CSV path")
    parser.add_argument(
        "--no-derived",
        action="store_true",
        help="Skip computing derived metrics (raw NCU output only)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Convert
    print(f"Converting {args.input} to CSV...")
    df = ncu_report_to_dataframe(args.input)
    print(f"  Raw data: {len(df)} kernel records, {len(df.columns)} columns")

    if not args.no_derived:
        df = compute_derived_metrics(df)
        derived_cols = [c for c in df.columns if c.startswith("derived_")]
        print(f"  Added {len(derived_cols)} derived metric columns")

    # Save
    output_path = args.output or args.input.replace(".ncu-rep", ".csv")
    df.to_csv(output_path, index=False)
    print(f"  Saved to {output_path}")


if __name__ == "__main__":
    main()
