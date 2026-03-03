#!/usr/bin/env python3
"""SPAR Workload Launcher — orchestrates telemetry collection around workloads.

Starts GPU telemetry collection, launches a workload subprocess, waits for
completion, stops telemetry, and saves labeled data.

Usage (registry mode):
    python run_workload.py --workload idle
    python run_workload.py --workload pytorch_resnet_cifar10

Usage (command mode):
    python run_workload.py --workload my_custom_label -- python my_script.py --arg1 val1

Options:
    --gpu-index     GPU device index (default: 0)
    --output-dir    Directory for output files (default: data/)
    --format        Output format: parquet or csv (default: parquet)
    --interval      Telemetry polling interval in seconds (default: 1.0)
    --no-dcgm       Disable DCGM Tier 2 collection
    --timeout       Kill workload after N seconds (default: no limit)
    --ncu           Also run Nsight Compute Tier 3 profiling (second pass)
"""

import argparse
import logging
import os
import signal
import subprocess
import sys
import uuid
from datetime import datetime
from pathlib import Path

# Add repo root to path so we can import from workloads/
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from scripts.collect_telemetry import TelemetryCollector
from workloads.registry import get_workload, list_workloads, WORKLOAD_REGISTRY

log = logging.getLogger("spar.launcher")


def generate_filename(
    workload_label: str, gpu_name: str, run_id: str, ext: str
) -> str:
    """Generate output filename following the project convention:
    {workload}_{gpu}_{run_id_short}_{datetime}.{ext}"""
    gpu_short = gpu_name.replace(" ", "_").replace("-", "_")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{workload_label}_{gpu_short}_{run_id[:8]}_{date_str}.{ext}"


def run_ncu_profiling(
    workload_command: list[str],
    workload_label: str,
    output_dir: str,
    gpu_index: int,
):
    """Run Nsight Compute Tier 3 profiling as a second pass."""
    ncu_script = os.path.join(_REPO_ROOT, "scripts", "collect_ncu_metrics.py")
    if not os.path.exists(ncu_script):
        log.warning("collect_ncu_metrics.py not found; skipping Tier 3 profiling.")
        return

    cmd = [
        sys.executable,
        ncu_script,
        "--output-dir", output_dir,
        "--label", workload_label,
        "--gpu-index", str(gpu_index),
        "--",
    ] + workload_command

    print(f"\n--- Tier 3 (Nsight Compute) profiling pass ---")
    print(f"Command: {' '.join(cmd)}")
    result = subprocess.run(cmd)
    if result.returncode != 0:
        log.warning("NCU profiling exited with code %d", result.returncode)


def main():
    parser = argparse.ArgumentParser(
        description="SPAR Workload Launcher — collect GPU telemetry during workload execution.",
        epilog=(
            "Examples:\n"
            "  python run_workload.py --workload idle\n"
            "  python run_workload.py --workload pytorch_resnet_cifar10\n"
            "  python run_workload.py --workload custom -- python script.py\n"
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--workload",
        type=str,
        default=None,
        help="Workload label (registry name or custom label with -- command)",
    )
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU device index")
    parser.add_argument("--run-id", type=str, default=None, help="Run ID (auto-generated if omitted)")
    parser.add_argument("--output-dir", type=str, default="data/", help="Output directory")
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format",
    )
    parser.add_argument("--interval", type=float, default=1.0, help="Telemetry polling interval (seconds)")
    parser.add_argument("--no-dcgm", action="store_true", help="Disable DCGM Tier 2")
    parser.add_argument("--timeout", type=int, default=None, help="Kill workload after N seconds")
    parser.add_argument("--ncu", action="store_true", help="Also run Nsight Compute Tier 3")
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available registered workloads and exit",
    )

    # Parse known args first, then capture everything after '--' as the workload command
    args, extra_cmd = parser.parse_known_args()

    if args.list:
        print("Registered workloads:")
        for label in list_workloads():
            info = WORKLOAD_REGISTRY[label]
            print(f"  {label:30s} {info['description']}")
        return

    if args.workload is None:
        parser.error("--workload is required (or use --list to see available workloads)")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Resolve the workload command
    # Strip leading '--' that parse_known_args may leave in extra_cmd
    if extra_cmd and extra_cmd[0] == "--":
        extra_cmd = extra_cmd[1:]

    if extra_cmd:
        # Command mode: user passed -- followed by a command
        workload_command = extra_cmd
        workload_label = args.workload
        timeout = args.timeout
    else:
        # Registry mode: look up the workload
        try:
            wl = get_workload(args.workload)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            print(f"Use --list to see available workloads.", file=sys.stderr)
            sys.exit(1)
        workload_command = wl["command"]
        workload_label = args.workload
        timeout = args.timeout or wl.get("default_timeout")

    run_id = args.run_id or uuid.uuid4().hex[:12]
    output_dir = os.path.join(_REPO_ROOT, args.output_dir)

    # Create the telemetry collector
    try:
        collector = TelemetryCollector(
            gpu_index=args.gpu_index,
            interval_sec=args.interval,
            workload_label=workload_label,
            run_id=run_id,
            enable_dcgm=not args.no_dcgm,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize telemetry: {e}", file=sys.stderr)
        sys.exit(1)

    # Generate output filename
    ext = "csv" if args.format == "csv" else "parquet"
    filename = generate_filename(
        workload_label, collector.metadata.gpu_name, run_id, ext
    )
    output_path = os.path.join(output_dir, filename)

    # Print run summary
    print("=" * 60)
    print("SPAR Workload Run")
    print("=" * 60)
    print(f"  Workload:    {workload_label}")
    print(f"  Command:     {' '.join(workload_command)}")
    print(f"  GPU:         {collector.metadata.gpu_name} (index {args.gpu_index})")
    print(f"  Run ID:      {run_id}")
    print(f"  Output:      {output_path}")
    print(f"  Interval:    {args.interval}s")
    print(f"  DCGM:        {'disabled' if args.no_dcgm else 'enabled (if available)'}")
    print(f"  Timeout:     {timeout or 'none'}s")
    print(f"  NCU (Tier3): {'yes (second pass)' if args.ncu else 'no'}")
    print("=" * 60)

    process = None
    try:
        # Start telemetry
        collector.start()

        # Wait for baseline telemetry (2 seconds of pre-workload data)
        import time
        print("\nCollecting pre-workload baseline (2s)...")
        time.sleep(2)

        # Launch workload
        print(f"\nLaunching workload: {' '.join(workload_command)}")
        print("-" * 40)
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)

        workload_start = time.time()
        process = subprocess.Popen(
            workload_command,
            stdout=sys.stdout,
            stderr=sys.stderr,
            env=env,
            cwd=_REPO_ROOT,
        )

        # Wait for workload to complete
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

        # Post-workload cooldown telemetry
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
        workload_duration = 0

    # Stop telemetry and save
    collector.stop()
    collector.save(output_path, fmt=args.format)

    # Summary
    print("\n" + "=" * 60)
    print("Run Summary")
    print("=" * 60)
    print(f"  Samples collected: {collector.sample_count}")
    print(f"  Workload exit code: {exit_code}")
    print(f"  Output file: {output_path}")
    print("=" * 60)

    collector.cleanup()

    # Optional NCU profiling (second pass)
    if args.ncu:
        run_ncu_profiling(workload_command, workload_label, output_dir, args.gpu_index)


if __name__ == "__main__":
    main()
