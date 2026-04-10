#!/usr/bin/env python3
"""SPAR Workload Launcher with Wall-Power Logging.

Wraps run_workload.py to simultaneously collect:
  - Tier 1 GPU telemetry (pynvml via TelemetryCollector)
  - Tier 5 wall power (Shelly Plus Plug US via ShellyPowerLogger)

Both outputs share the same run_id so they can be joined by timestamp.

Usage:
    python run_workload_with_power.py --shelly-ip 192.168.1.100 --workload idle --no-dcgm
    python run_workload_with_power.py --shelly-ip 192.168.1.100 --workload pytorch_resnet_cifar10 --no-dcgm
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime

# Add repo root to path
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from scripts.collect_telemetry import TelemetryCollector
from scripts.shelly_power_logger import ShellyPowerLogger
from workloads.registry import get_workload, list_workloads, WORKLOAD_REGISTRY

log = logging.getLogger("spar.launcher_power")


def generate_filename(workload_label: str, gpu_name: str, run_id: str, suffix: str, ext: str) -> str:
    gpu_short = gpu_name.replace(" ", "_").replace("-", "_")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{workload_label}_{gpu_short}_{run_id[:8]}_{date_str}{suffix}.{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="Run workload with simultaneous GPU telemetry + wall-power logging.",
    )
    parser.add_argument("--workload", type=str, required=True, help="Workload label from registry")
    parser.add_argument("--shelly-ip", type=str, required=True, help="Shelly Plus Plug US IP address")
    parser.add_argument("--gpu-index", type=int, default=0, help="GPU device index")
    parser.add_argument("--output-dir", type=str, default="data/", help="Output directory")
    parser.add_argument("--interval", type=float, default=1.0, help="Telemetry polling interval (seconds)")
    parser.add_argument("--no-dcgm", action="store_true", help="Disable DCGM Tier 2")
    parser.add_argument("--timeout", type=int, default=None, help="Kill workload after N seconds")
    parser.add_argument("--ncu", action="store_true", help="Also run Nsight Compute Tier 3")
    parser.add_argument("--list", action="store_true", help="List available workloads and exit")

    args, extra_cmd = parser.parse_known_args()

    if args.list:
        print("Registered workloads:")
        for label in list_workloads():
            info = WORKLOAD_REGISTRY[label]
            print(f"  {label:30s} {info['description']}")
        return

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    # Resolve workload command
    if extra_cmd and extra_cmd[0] == "--":
        extra_cmd = extra_cmd[1:]

    if extra_cmd:
        workload_command = extra_cmd
        workload_label = args.workload
        timeout = args.timeout
    else:
        wl = get_workload(args.workload)
        workload_command = wl["command"]
        workload_label = args.workload
        timeout = args.timeout or wl.get("default_timeout")

    run_id = uuid.uuid4().hex[:12]
    output_dir = os.path.join(_REPO_ROOT, args.output_dir)

    # Initialize GPU telemetry collector
    collector = TelemetryCollector(
        gpu_index=args.gpu_index,
        interval_sec=args.interval,
        workload_label=workload_label,
        run_id=run_id,
        enable_dcgm=not args.no_dcgm,
    )

    # Initialize Shelly power logger
    power_logger = ShellyPowerLogger(
        shelly_ip=args.shelly_ip,
        interval_sec=args.interval,
        run_id=run_id,
        workload_label=workload_label,
    )

    # Test Shelly connection
    print(f"Testing Shelly connection at {args.shelly_ip}...")
    if not power_logger.test_connection():
        print("WARNING: Shelly plug not reachable. Proceeding with GPU telemetry only.")
        power_logger = None

    # Generate output filenames
    ext = "parquet"
    gpu_filename = generate_filename(workload_label, collector.metadata.gpu_name, run_id, "", ext)
    gpu_output = os.path.join(output_dir, gpu_filename)

    power_output = None
    if power_logger:
        power_dir = os.path.join(output_dir, "wall_power")
        os.makedirs(power_dir, exist_ok=True)
        power_filename = generate_filename(workload_label, collector.metadata.gpu_name, run_id, "_wall_power", ext)
        power_output = os.path.join(power_dir, power_filename)

    # Print run summary
    print("=" * 60)
    print("SPAR Workload Run (with Wall Power)")
    print("=" * 60)
    print(f"  Workload:     {workload_label}")
    print(f"  Command:      {' '.join(workload_command)}")
    print(f"  GPU:          {collector.metadata.gpu_name} (index {args.gpu_index})")
    print(f"  Run ID:       {run_id}")
    print(f"  GPU output:   {gpu_output}")
    print(f"  Power output: {power_output or 'N/A (Shelly unreachable)'}")
    print(f"  DCGM:         {'disabled' if args.no_dcgm else 'enabled (if available)'}")
    print(f"  Timeout:      {timeout or 'none'}s")
    print("=" * 60)

    process = None
    try:
        # Start both loggers
        collector.start()
        if power_logger:
            power_logger.start()

        # Pre-workload baseline
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

        # Post-workload cooldown
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

    # Stop and save
    collector.stop()
    collector.save(gpu_output)

    if power_logger:
        power_logger.stop()
        power_logger.save(power_output)

    # Summary
    print("\n" + "=" * 60)
    print("Run Summary")
    print("=" * 60)
    print(f"  GPU samples:   {collector.sample_count}")
    if power_logger:
        print(f"  Power samples: {power_logger.sample_count}")
    print(f"  Exit code:     {exit_code}")
    print(f"  GPU output:    {gpu_output}")
    if power_output:
        print(f"  Power output:  {power_output}")
    print("=" * 60)

    collector.cleanup()


if __name__ == "__main__":
    main()
