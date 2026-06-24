#!/usr/bin/env python3
"""SPAR Workload Launcher with Physical-Sensor Logging.

Runs a workload while collecting, under a shared run_id:
  - Tier 1 GPU telemetry  (NVML via TelemetryCollector)  -> data/<name>.parquet
  - Physical sensors      (SAIGE testbed snapshot API)    -> data/sensors/<name>_sensors.parquet

Physical sensors = UltraMic 200K acoustic mic + 2x PicoScope 2204A current
clamps on the GPU0/GPU1/GPU2/MOBO power rails + NVML power (node1 testbed).

Because the mic and MOBO rail are SHARED across the whole node, run ONE workload
at a time (others idle). Target GPU 0 so its dedicated current clamp is clean.

Usage:
    python scripts/run_workload_with_sensors.py --workload idle --gpu-index 0
    python scripts/run_workload_with_sensors.py --workload bert_sst2 --gpu-index 0 --timeout 220
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

from scripts.collect_telemetry import TelemetryCollector
from scripts.sensor_logger import SensorLogger
from workloads.registry import get_workload, list_workloads, WORKLOAD_REGISTRY


def gen_name(label, gpu_name, run_id, suffix, ext="parquet"):
    gpu_short = gpu_name.replace(" ", "_").replace("-", "_")
    return f"{label}_{gpu_short}_{run_id[:8]}_{datetime.now():%Y%m%d_%H%M%S}{suffix}.{ext}"


def main():
    p = argparse.ArgumentParser(description="Run workload with NVML + physical-sensor logging.")
    p.add_argument("--workload", required=True)
    p.add_argument("--gpu-index", type=int, default=0)
    p.add_argument("--output-dir", default="data/")
    p.add_argument("--interval", type=float, default=1.0)
    p.add_argument("--timeout", type=int, default=None)
    p.add_argument("--no-dcgm", action="store_true")
    p.add_argument("--store-waves", action="store_true", help="archive sensor wave arrays to .npz")
    p.add_argument("--list", action="store_true")
    args, extra = p.parse_known_args()

    if args.list:
        for lbl in list_workloads():
            print(f"  {lbl:32s} {WORKLOAD_REGISTRY[lbl]['description']}")
        return

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    if extra and extra[0] == "--":
        extra = extra[1:]
    if extra:
        cmd, label, timeout = extra, args.workload, args.timeout
    else:
        wl = get_workload(args.workload)
        cmd, label, timeout = wl["command"], args.workload, args.timeout or wl.get("default_timeout")

    run_id = uuid.uuid4().hex[:12]
    out_dir = os.path.join(_REPO_ROOT, args.output_dir)

    collector = TelemetryCollector(gpu_index=args.gpu_index, interval_sec=args.interval,
                                   workload_label=label, run_id=run_id, enable_dcgm=not args.no_dcgm)

    # Physical-sensor sidecar (fail soft if the testbed feed is unreachable)
    sensor_logger = None
    try:
        sensor_logger = SensorLogger(interval_sec=args.interval, run_id=run_id,
                                     workload_label=label, store_waves=args.store_waves)
        h = sensor_logger.client.health()
        print(f"Sensor feed OK: {h.get('sensors')}  channels={list(h.get('channel_map', {}).values())}")
    except Exception as e:
        print(f"WARNING: sensor feed unavailable ({e}). Proceeding with NVML only.")
        sensor_logger = None

    gpu_name = collector.metadata.gpu_name
    gpu_out = os.path.join(out_dir, gen_name(label, gpu_name, run_id, ""))
    sensor_out = None
    if sensor_logger:
        # Keep sensor parquets OUT of data/ — the NVML classifier's load_all_data()
        # recursively globs data/ and would ingest these as all-zero telemetry.
        sdir = os.path.join(_REPO_ROOT, "sensor_logs")
        os.makedirs(sdir, exist_ok=True)
        sensor_out = os.path.join(sdir, gen_name(label, gpu_name, run_id, "_sensors"))

    print("=" * 60)
    print(f"  Workload:    {label}\n  GPU:         {gpu_name} (index {args.gpu_index})")
    print(f"  Run ID:      {run_id}\n  NVML out:    {gpu_out}\n  Sensor out:  {sensor_out or 'N/A'}")
    print(f"  Timeout:     {timeout or 'none'}s")
    print("=" * 60)

    proc, exit_code = None, None
    try:
        collector.start()
        if sensor_logger:
            sensor_logger.start()
        print("\nPre-workload baseline (3s)...")
        time.sleep(3)

        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(args.gpu_index)
        print(f"Launching: {' '.join(cmd)}\n" + "-" * 40)
        t0 = time.time()
        proc = subprocess.Popen(cmd, stdout=sys.stdout, stderr=sys.stderr, env=env, cwd=_REPO_ROOT)
        try:
            exit_code = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"\nTimed out after {timeout}s, killing...")
            proc.kill(); proc.wait(); exit_code = -9
        print("-" * 40 + f"\nWorkload finished (exit={exit_code}, dur={time.time()-t0:.1f}s)")
        print("Post-workload cooldown (3s)...")
        time.sleep(3)
    except KeyboardInterrupt:
        if proc and proc.poll() is None:
            proc.terminate()
            try: proc.wait(timeout=5)
            except subprocess.TimeoutExpired: proc.kill()
        exit_code = -2

    collector.stop(); collector.save(gpu_out)
    n_sensor = 0
    if sensor_logger:
        sensor_logger.stop()
        if len(sensor_logger.to_dataframe()):
            sensor_logger.save(sensor_out)
            n_sensor = len(sensor_logger.to_dataframe())

    print("\n" + "=" * 60)
    print(f"  NVML samples:   {collector.sample_count}")
    print(f"  Sensor samples: {n_sensor}")
    print(f"  Exit code:      {exit_code}")
    print(f"  NVML output:    {gpu_out}")
    print(f"  Sensor output:  {sensor_out if n_sensor else 'N/A'}")
    print("=" * 60)
    collector.cleanup()


if __name__ == "__main__":
    main()
