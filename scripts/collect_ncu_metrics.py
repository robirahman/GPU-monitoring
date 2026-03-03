#!/usr/bin/env python3
"""SPAR Tier 3 Nsight Compute PMC Collection — adapted from WAVE (Xu et al., ASPLOS '26).

Wraps `ncu` with WAVE's 24 PMC metrics to profile an arbitrary workload command.
Produces a .ncu-rep report file and optionally converts it to CSV.

Usage:
    python collect_ncu_metrics.py --label pytorch_resnet_cifar10 \
        -- python workloads/pytorch_training.py --epochs 1 --batch-size 32

Note: NCU re-executes the workload under profiling, so expect 1200-5300% overhead
(documented by WAVE). This is a separate pass from Tier 1/2 collection.

Requires: Nsight Compute (`ncu`) installed and accessible, bare-metal GPU access.
"""

import argparse
import logging
import os
import subprocess
import sys
from datetime import datetime

log = logging.getLogger("spar.ncu")

# WAVE's 24 PMC metrics (from https://github.com/sept-usc/Wave)
# These cover floating-point ops, tensor core ops, memory ops, and L1 cache misses.
NCU_METRICS = [
    # L1 cache misses (global load/store)
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum",
    # Double precision (FP64) SASS instructions
    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    # Single precision (FP32) SASS instructions
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    # Half precision (FP16) SASS instructions
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    # Tensor core operations (availability varies by GPU architecture)
    "sm__ops_path_tensor_src_fp8.sum",
    "sm__ops_path_tensor_src_fp8_dst_fp16.sum",
    "sm__ops_path_tensor_src_fp8_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp64.sum",
    # Global memory load/store instructions
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    # Local memory load/store instructions
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    # Shared memory load/store instructions
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
]


def run_ncu_profiling(
    workload_command: list[str],
    output_dir: str,
    label: str,
    gpu_index: int = 0,
    replay_mode: str = "application",
    convert_csv: bool = True,
) -> str | None:
    """Run Nsight Compute profiling on a workload command.

    Returns the path to the .ncu-rep file, or None on failure.
    """
    # Check ncu is available
    if subprocess.run(["which", "ncu"], capture_output=True).returncode != 0:
        log.error(
            "ncu (Nsight Compute) not found. Install it or add to PATH.\n"
            "Tier 3 profiling requires bare-metal GPU access."
        )
        return None

    os.makedirs(output_dir, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_name = f"{label}_ncu_{date_str}"
    report_path = os.path.join(output_dir, report_name)  # ncu appends .ncu-rep

    # Build ncu command
    ncu_cmd = [
        "ncu",
        "--config-file", "off",
        "--export", report_path,
        "--force-overwrite",
        "--replay-mode", replay_mode,
        "--app-replay-mode", "relaxed",
        "--target-processes", "all",
        "--metrics", ",".join(NCU_METRICS),
    ]

    # Set target GPU
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)

    # Strip any leading '--' from the workload command (argparse artifact)
    clean_cmd = [c for c in workload_command if c != "--"] if workload_command[0] == "--" else workload_command
    full_cmd = ncu_cmd + clean_cmd

    print(f"Running Nsight Compute profiling...")
    print(f"  Report: {report_path}.ncu-rep")
    print(f"  Metrics: {len(NCU_METRICS)} PMC counters")
    print(f"  Replay mode: {replay_mode}")
    print(f"  WARNING: Expect 1200-5300% overhead (workload will be re-executed multiple times)")
    print()

    result = subprocess.run(full_cmd, env=env)
    report_file = f"{report_path}.ncu-rep"

    if result.returncode != 0:
        log.error("ncu exited with code %d", result.returncode)
        # Check if the report was still generated (ncu sometimes returns non-zero
        # but still produces partial output)
        if not os.path.exists(report_file):
            return None
        log.warning("Report file exists despite non-zero exit; proceeding with conversion.")

    print(f"\nNCU report saved: {report_file}")

    # Convert to CSV
    if convert_csv and os.path.exists(report_file):
        csv_path = f"{report_path}.csv"
        ncu_to_csv_script = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "ncu_to_csv.py"
        )
        if os.path.exists(ncu_to_csv_script):
            csv_result = subprocess.run(
                [sys.executable, ncu_to_csv_script, report_file, "-o", csv_path]
            )
            if csv_result.returncode == 0:
                print(f"CSV export: {csv_path}")
            else:
                log.warning("CSV conversion failed (exit code %d)", csv_result.returncode)
        else:
            log.warning("ncu_to_csv.py not found; skipping CSV conversion.")

    return report_file


def main():
    parser = argparse.ArgumentParser(
        description="Tier 3 Nsight Compute PMC profiling (adapted from WAVE).",
        epilog=(
            "Example:\n"
            "  python collect_ncu_metrics.py --label resnet -- "
            "python workloads/pytorch_training.py --epochs 1\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/",
        help="Output directory for .ncu-rep and .csv files",
    )
    parser.add_argument("--label", type=str, required=True, help="Workload label for filename")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU device index")
    parser.add_argument(
        "--replay-mode",
        type=str,
        default="application",
        choices=["application", "kernel"],
        help="NCU replay mode (default: application)",
    )
    parser.add_argument("--no-csv", action="store_true", help="Skip CSV conversion")

    args, workload_cmd = parser.parse_known_args()

    # Strip leading '--' from workload command
    if workload_cmd and workload_cmd[0] == "--":
        workload_cmd = workload_cmd[1:]

    if not workload_cmd:
        parser.error("No workload command provided. Use -- followed by the command.")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    report = run_ncu_profiling(
        workload_command=workload_cmd,
        output_dir=args.output_dir,
        label=args.label,
        gpu_index=args.gpu_index,
        replay_mode=args.replay_mode,
        convert_csv=not args.no_csv,
    )

    if report is None:
        print("ERROR: NCU profiling failed.", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
