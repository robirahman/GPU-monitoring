#!/usr/bin/env python3
"""Run Nsight Compute Tier 3 profiling on RTX 5080 for each workload category.

Requires admin privileges (NCU needs elevated access on Windows with driver 581+).
Collects WAVE's 24 PMC metrics for one short run per workload.

Usage (run from admin PowerShell):
    python scripts/run_ncu_profiling.py
"""

import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = str(REPO_ROOT / ".venv" / "Scripts" / "python.exe")
NCU = r"C:\Program Files\NVIDIA Corporation\Nsight Compute 2025.1.0\target\windows-desktop-win7-x64\ncu.exe"
OUTPUT_DIR = REPO_ROOT / "data" / "ncu_profiles"

# WAVE's 24 PMC metrics
NCU_METRICS = [
    "l1tex__t_sectors_pipe_lsu_mem_global_op_ld_lookup_miss.sum",
    "l1tex__t_sectors_pipe_lsu_mem_global_op_st_lookup_miss.sum",
    "smsp__sass_thread_inst_executed_op_dadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_dfma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_fmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_ffma_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hadd_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hmul_pred_on.sum",
    "smsp__sass_thread_inst_executed_op_hfma_pred_on.sum",
    "sm__ops_path_tensor_src_fp8.sum",
    "sm__ops_path_tensor_src_fp8_dst_fp16.sum",
    "sm__ops_path_tensor_src_fp8_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp16.sum",
    "sm__ops_path_tensor_src_fp16_dst_fp32.sum",
    "sm__ops_path_tensor_src_tf32_dst_fp32.sum",
    "sm__ops_path_tensor_src_fp64.sum",
    "smsp__sass_inst_executed_op_global_ld.sum",
    "smsp__sass_inst_executed_op_global_st.sum",
    "smsp__sass_inst_executed_op_local_ld.sum",
    "smsp__sass_inst_executed_op_local_st.sum",
    "smsp__sass_inst_executed_op_shared_ld.sum",
    "smsp__sass_inst_executed_op_shared_st.sum",
]

# Short workloads for profiling (short duration to limit NCU overhead)
WORKLOADS = [
    {
        "label": "resnet18_training",
        "cmd": [PY, str(REPO_ROOT / "workloads" / "pytorch_training.py"),
                "--model", "resnet18", "--epochs", "1", "--batch-size", "128"],
    },
    {
        "label": "resnet50_inference",
        "cmd": [PY, str(REPO_ROOT / "workloads" / "inference.py"),
                "--model", "resnet50", "--batch-size", "64", "--duration", "30"],
    },
    {
        "label": "cufft_hpc",
        "cmd": [PY, str(REPO_ROOT / "workloads" / "scientific_hpc.py"),
                "--workload", "cufft", "--duration", "30"],
    },
    {
        "label": "nbody_hpc",
        "cmd": [PY, str(REPO_ROOT / "workloads" / "scientific_hpc.py"),
                "--workload", "nbody", "--duration", "30"],
    },
    {
        "label": "mining_proxy",
        "cmd": [PY, str(REPO_ROOT / "workloads" / "mining_proxy.py"),
                "--duration", "30", "--dag-size-mb", "512"],
    },
    {
        "label": "rendering_proxy",
        "cmd": [PY, str(REPO_ROOT / "workloads" / "rendering_proxy.py"),
                "--duration", "30", "--rays", "16384"],
    },
]


def run_ncu(label, cmd):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = str(OUTPUT_DIR / f"{label}_ncu_{date_str}")

    ncu_cmd = [
        NCU,
        "--config-file", "off",
        "--export", report_path,
        "--force-overwrite",
        "--replay-mode", "application",
        "--app-replay-mode", "relaxed",
        "--target-processes", "all",
        "--launch-count", "10",  # Profile first 10 kernel launches only
        "--metrics", ",".join(NCU_METRICS),
    ] + cmd

    print(f"\n{'='*60}")
    print(f"  NCU profiling: {label}")
    print(f"  Report: {report_path}.ncu-rep")
    print(f"  WARNING: NCU re-executes kernels — expect 10-50x slowdown")
    print(f"{'='*60}")

    result = subprocess.run(ncu_cmd, cwd=str(REPO_ROOT))
    report_file = f"{report_path}.ncu-rep"

    if os.path.exists(report_file):
        size = os.path.getsize(report_file) / 1024
        print(f"  OK: {report_file} ({size:.0f} KB)")
        return report_file
    else:
        print(f"  FAILED: no report generated (exit code {result.returncode})")
        return None


def main():
    if not os.path.exists(NCU):
        print(f"ERROR: NCU not found at {NCU}")
        sys.exit(1)

    print(f"SPAR Tier 3 NCU Profiling — RTX 5080")
    print(f"  NCU: {NCU}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Metrics: {len(NCU_METRICS)} WAVE PMC counters")
    print(f"  Workloads: {len(WORKLOADS)}")

    results = []
    for wl in WORKLOADS:
        report = run_ncu(wl["label"], wl["cmd"])
        results.append((wl["label"], report))

    print(f"\n{'='*60}")
    print("PROFILING COMPLETE")
    print(f"{'='*60}")
    for label, report in results:
        status = f"OK ({os.path.getsize(report)/1024:.0f} KB)" if report and os.path.exists(report) else "FAILED"
        print(f"  {label:25s} {status}")


if __name__ == "__main__":
    main()
