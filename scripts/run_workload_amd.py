#!/usr/bin/env python3
"""SPAR Workload Launcher (AMD/ROCm) — analogue of scripts/run_workload.py.

Starts AMD GPU telemetry collection (amdsmi + optional RDC), launches a
workload subprocess, waits for completion, stops telemetry, and saves
labeled data.

Usage (registry mode):
    python run_workload_amd.py --workload idle
    python run_workload_amd.py --workload pytorch_resnet_cifar10

Usage (command mode):
    python run_workload_amd.py --workload my_label -- python my_script.py --arg val

Differences from the NVIDIA launcher:
  - Uses AmdTelemetryCollector instead of TelemetryCollector
  - Loads the AMD workload registry (workloads/registry_amd.py)
  - Sets HIP_VISIBLE_DEVICES + ROCR_VISIBLE_DEVICES instead of CUDA_VISIBLE_DEVICES.
    HIP also honours CUDA_VISIBLE_DEVICES for the HIP runtime, so we set both
    to be safe (PyTorch ROCm + raw HIP toolchain).
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from scripts.collect_amdsmi_telemetry import AmdTelemetryCollector
from workloads.registry_amd import (
    get_workload,
    list_workloads,
    WORKLOAD_REGISTRY_AMD,
)

log = logging.getLogger("spar.launcher_amd")


def generate_filename(workload_label: str, gpu_name: str, run_id: str, ext: str) -> str:
    gpu_short = gpu_name.replace(" ", "_").replace("-", "_").replace("/", "_")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{workload_label}_{gpu_short}_{run_id[:8]}_{date_str}.{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="SPAR Workload Launcher (AMD) — collect amdsmi telemetry during workload execution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workload", type=str, default=None,
                        help="Workload label (registry name or custom label with -- command)")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/")
    parser.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--no-rdc", action="store_true",
                        help="Disable RDC Tier 2 (skip even attempting to connect to rdcd)")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--list", action="store_true",
                        help="List registered AMD workloads and exit")

    args, extra_cmd = parser.parse_known_args()

    if args.list:
        print("Registered AMD workloads:")
        for label in list_workloads():
            info = WORKLOAD_REGISTRY_AMD[label]
            print(f"  {label:35s} {info['description']}")
        return

    if args.workload is None:
        parser.error("--workload is required (or use --list to see available workloads)")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if extra_cmd and extra_cmd[0] == "--":
        extra_cmd = extra_cmd[1:]

    if extra_cmd:
        workload_command = extra_cmd
        workload_label = args.workload
        timeout = args.timeout
    else:
        try:
            wl = get_workload(args.workload)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print("Use --list to see available workloads.", file=sys.stderr)
            sys.exit(1)
        workload_command = wl["command"]
        workload_label = args.workload
        timeout = args.timeout or wl.get("default_timeout")

    run_id = args.run_id or uuid.uuid4().hex[:12]
    output_dir = os.path.join(_REPO_ROOT, args.output_dir)

    try:
        collector = AmdTelemetryCollector(
            gpu_index=args.gpu_index,
            interval_sec=args.interval,
            workload_label=workload_label,
            run_id=run_id,
            enable_rdc=not args.no_rdc,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize amdsmi telemetry: {e}", file=sys.stderr)
        sys.exit(1)

    ext = "csv" if args.format == "csv" else "parquet"
    filename = generate_filename(
        workload_label, collector.metadata.gpu_name, run_id, ext
    )
    output_path = os.path.join(output_dir, filename)

    print("=" * 60)
    print("SPAR Workload Run (AMD)")
    print("=" * 60)
    print(f"  Workload:    {workload_label}")
    print(f"  Command:     {' '.join(workload_command)}")
    print(f"  GPU:         {collector.metadata.gpu_name} (index {args.gpu_index})")
    print(f"  ROCm:        {collector.metadata.rocm_version}")
    print(f"  Run ID:      {run_id}")
    print(f"  Output:      {output_path}")
    print(f"  Interval:    {args.interval}s")
    print(f"  RDC:         {'disabled' if args.no_rdc else 'enabled (if rdcd available)'}")
    print(f"  Timeout:     {timeout or 'none'}s")
    print("=" * 60)

    process = None
    exit_code = 0
    workload_duration = 0.0
    try:
        collector.start()
        print("\nCollecting pre-workload baseline (2s)...")
        time.sleep(2)

        print(f"\nLaunching workload: {' '.join(workload_command)}")
        print("-" * 40)
        env = os.environ.copy()
        # ROCm honours all three; set them all so the child process sees a
        # single visible device regardless of which API path the workload uses.
        env["HIP_VISIBLE_DEVICES"] = str(args.gpu_index)
        env["ROCR_VISIBLE_DEVICES"] = str(args.gpu_index)
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

        workload_start = time.time()
        process = subprocess.Popen(
            workload_command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
            cwd=_REPO_ROOT,
        )

        try:
            exit_code = process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"\nWARNING: Workload timed out after {timeout}s, killing...")
            process.kill()
            process.wait()
            exit_code = -9

        workload_duration = time.time() - workload_start
        print("-" * 40)
        print(f"Workload finished (exit_code={exit_code}, duration={workload_duration:.1f}s)")

        print("Collecting post-workload cooldown (2s)...")
        time.sleep(2)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Saving collected data...")
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        exit_code = -2

    collector.stop()
    collector.save(output_path, fmt=args.format)

    print("\n" + "=" * 60)
    print("Run Summary")
    print("=" * 60)
    print(f"  Samples collected: {collector.sample_count}")
    print(f"  Workload exit code: {exit_code}")
    print(f"  Output file: {output_path}")
    print("=" * 60)

    collector.cleanup()
    sys.exit(exit_code if exit_code is not None else 0)


if __name__ == "__main__":
    main()
