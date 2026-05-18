#!/usr/bin/env python3
"""SPAR Workload Launcher (AMD / Windows) — analogue of run_workload_amd.py.

Starts PDH + LibreHardwareMonitor telemetry collection, launches a workload
subprocess, waits for completion, stops telemetry, and saves a labeled
parquet row-set under data/ matching the existing project file naming.

Usage (registry mode):
    python run_workload_windows.py --workload idle
    python run_workload_windows.py --workload pytorch_resnet_cifar10
    python run_workload_windows.py --workload ffmpeg_amf_h264 --interval 0.5

Usage (custom command mode):
    python run_workload_windows.py --workload my_label -- python my_script.py --arg val

Differences from run_workload_amd.py:
  - PdhCollector instead of AmdTelemetryCollector (no ROCm / RDC on Windows)
  - WORKLOAD_REGISTRY_WINDOWS_AMD instead of registry_amd
  - DML_DEVICE_INDEX env var instead of HIP/ROCR/CUDA_VISIBLE_DEVICES
    (DirectML is the compute path on Windows-AMD, see workloads/_dml_device.py)
  - --lhm-url and --gpu-text passed through to the LHM HTTP reader
  - The collector by default auto-detects the discrete dGPU by max VRAM via
    PDH; --adapter-luid override is exposed for multi-dGPU systems
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

from scripts.collect_pdh_windows import PdhCollector
from workloads.registry_windows_amd import (
    WORKLOAD_REGISTRY_WINDOWS_AMD,
    get_workload,
    list_workloads,
)

log = logging.getLogger("spar.launcher_windows")


def generate_filename(workload_label: str, gpu_name: str, run_id: str, ext: str) -> str:
    gpu_short = (gpu_name or "Unknown_GPU").replace(" ", "_").replace("-", "_").replace("/", "_")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{workload_label}_{gpu_short}_{run_id[:8]}_{date_str}.{ext}"


def main():
    parser = argparse.ArgumentParser(
        description="SPAR Workload Launcher (AMD/Windows) — collect PDH+LHM telemetry "
                    "during workload execution.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--workload", type=str, default=None,
                        help="Workload label (registry name or custom label with -- command)")
    parser.add_argument("--dml-device-index", type=int, default=0,
                        help="DirectML adapter index for the workload (default 0 = dGPU on most boxes). "
                             "Sets DML_DEVICE_INDEX env var; consumed by workloads/_dml_device.py.")
    parser.add_argument("--adapter-luid", type=str, default=None,
                        help="Override PDH adapter LUID for telemetry "
                             "(default: auto-detect largest dedicated-VRAM adapter).")
    parser.add_argument("--lhm-url", type=str, default=None,
                        help="LibreHardwareMonitor HTTP endpoint "
                             "(default: http://localhost:8085/data.json)")
    parser.add_argument("--gpu-text", type=str, default=None,
                        help="GPU Text substring for LHM to prefer (default: 'RX 9070'/'RX 7900').")
    parser.add_argument("--run-id", type=str, default=None)
    parser.add_argument("--output-dir", type=str, default="data/")
    parser.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Telemetry sampling interval in seconds (default 1.0)")
    parser.add_argument("--timeout", type=int, default=None,
                        help="Override the workload's default timeout (seconds)")
    parser.add_argument("--baseline-pre-sec", type=float, default=2.0,
                        help="Idle-baseline seconds before launching the workload (default 2)")
    parser.add_argument("--baseline-post-sec", type=float, default=2.0,
                        help="Cooldown seconds after the workload exits (default 2)")
    parser.add_argument("--list", action="store_true",
                        help="List registered Windows-AMD workloads and exit")

    args, extra_cmd = parser.parse_known_args()

    if args.list:
        print("Registered Windows-AMD workloads:")
        for label in list_workloads():
            info = WORKLOAD_REGISTRY_WINDOWS_AMD[label]
            req = f"  [needs: {info['requires_model']}]" if info.get("requires_model") else ""
            print(f"  {label:32s} {info['description']}{req}")
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
        workload_command = list(wl["command"])
        workload_label = args.workload
        timeout = args.timeout or wl.get("default_timeout")
        if wl.get("requires_model"):
            model_path = next((workload_command[i + 1] for i, a in enumerate(workload_command)
                               if a == "--model"), None)
            if model_path and not os.path.isfile(model_path):
                print(f"ERROR: workload {workload_label!r} needs model file: {model_path}\n"
                      "Download a GGUF model and place it there before running this entry.",
                      file=sys.stderr)
                sys.exit(2)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    output_dir = args.output_dir if os.path.isabs(args.output_dir) \
        else os.path.join(_REPO_ROOT, args.output_dir)
    os.makedirs(output_dir, exist_ok=True)

    try:
        collector = PdhCollector(
            adapter_luid=args.adapter_luid,
            interval_sec=args.interval,
            workload_label=workload_label,
            run_id=run_id,
            lhm_url=args.lhm_url,
            gpu_text_hint=args.gpu_text,
        )
    except Exception as e:
        print(f"ERROR: Failed to initialize PDH telemetry: {e}", file=sys.stderr)
        sys.exit(1)

    gpu_name = collector._gpu_metadata.get("gpu_name", "Unknown_GPU")
    ext = "csv" if args.format == "csv" else "parquet"
    filename = generate_filename(workload_label, gpu_name, run_id, ext)
    output_path = os.path.join(output_dir, filename)

    print("=" * 60)
    print("SPAR Workload Run (AMD / Windows)")
    print("=" * 60)
    print(f"  Workload:    {workload_label}")
    print(f"  Command:     {' '.join(workload_command)}")
    print(f"  GPU:         {gpu_name}")
    print(f"  PDH adapter: {collector._adapter.luid}")
    print(f"  LHM:         {'on (' + collector._gpu_metadata.get('lhm_url', '?') + ')' if collector._gpu_metadata.get('lhm_available') else 'off — sensors will be NaN'}")
    print(f"  DML index:   {args.dml_device_index}")
    print(f"  Run ID:      {run_id}")
    print(f"  Output:      {output_path}")
    print(f"  Interval:    {args.interval}s")
    print(f"  Timeout:     {timeout or 'none'}s")
    print("=" * 60)

    process = None
    exit_code = 0
    workload_duration = 0.0
    try:
        collector.start()
        if args.baseline_pre_sec > 0:
            print(f"\nCollecting pre-workload baseline ({args.baseline_pre_sec}s)...")
            time.sleep(args.baseline_pre_sec)

        print(f"\nLaunching workload: {' '.join(workload_command)}")
        print("-" * 40)
        env = os.environ.copy()
        # DirectML adapter selection (consumed by workloads/_dml_device.py)
        env["DML_DEVICE_INDEX"] = str(args.dml_device_index)
        # ffmpeg AMF still needs PATH / WinGet probing; nothing extra here.

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

        if args.baseline_post_sec > 0:
            print(f"Collecting post-workload cooldown ({args.baseline_post_sec}s)...")
            time.sleep(args.baseline_post_sec)

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
    print(f"  Samples collected:  {collector.sample_count}")
    print(f"  Workload exit code: {exit_code}")
    print(f"  Workload duration:  {workload_duration:.1f}s")
    print(f"  Output file:        {output_path}")
    print("=" * 60)

    sys.exit(exit_code if exit_code is not None else 0)


if __name__ == "__main__":
    main()
