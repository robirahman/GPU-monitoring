#!/usr/bin/env python3
"""Multi-GPU workload launcher — collects per-GPU telemetry for 1-8 GPUs.

Starts a TelemetryCollector on each GPU, launches the workload (typically via
torchrun for DDP), waits for completion, and saves per-GPU parquet files with
synchronized run_ids.

Usage:
    python scripts/run_workload_multi_gpu.py --workload ddp_resnet_cifar10 --num-gpus 8
    python scripts/run_workload_multi_gpu.py --num-gpus 8 --timeout 300 -- torchrun --nproc_per_node=8 workloads/adversarial_ddp.py --strategy K --duration 300

Single-GPU parallel mode (run 8 independent single-GPU workloads simultaneously):
    python scripts/run_workload_multi_gpu.py --parallel --workload adversarial_K_online_learning --num-gpus 8
"""

import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from scripts.collect_telemetry import TelemetryCollector
from workloads.registry import get_workload, WORKLOAD_REGISTRY

log = logging.getLogger("spar.multi_gpu")
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")


def generate_filename(label, gpu_name, gpu_index, run_id, ext="parquet"):
    gpu_short = gpu_name.replace(" ", "_").replace("-", "_")
    date_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{label}_gpu{gpu_index}_{gpu_short}_{run_id[:12]}_{date_str}.{ext}"


def run_multi_gpu(args, workload_cmd):
    """Launch one multi-GPU workload with per-GPU telemetry."""
    num_gpus = args.num_gpus
    run_id = uuid.uuid4().hex
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Multi-GPU run: {args.workload}, {num_gpus} GPUs, run_id={run_id[:12]}")

    # Start telemetry on all GPUs
    collectors = []
    for i in range(num_gpus):
        c = TelemetryCollector(
            gpu_index=i,
            interval_sec=args.interval,
            workload_label=args.workload,
            enable_dcgm=not args.no_dcgm,
        )
        c.start()
        collectors.append(c)
        log.info(f"  GPU {i}: telemetry started ({c.metadata.gpu_name})")

    # Brief baseline
    time.sleep(2)

    # Launch workload
    log.info(f"  Launching: {' '.join(workload_cmd)}")
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = ",".join(str(i) for i in range(num_gpus))

    workload_rc = 0
    try:
        timeout = args.timeout + 60 if args.timeout else None
        result = subprocess.run(
            workload_cmd,
            cwd=_REPO_ROOT,
            env=env,
            timeout=timeout,
        )
        workload_rc = result.returncode
        log.info(f"  Workload exited with code {result.returncode}")
    except subprocess.TimeoutExpired:
        log.warning(f"  Workload timed out after {args.timeout}s")
        workload_rc = 124
    except Exception as e:
        log.error(f"  Workload error: {e}")
        workload_rc = 1

    # Brief cooldown
    time.sleep(2)

    # Stop telemetry and save
    for i, c in enumerate(collectors):
        c.stop()
        fname = generate_filename(args.workload, c.metadata.gpu_name, i, run_id)
        fpath = str(output_dir / fname)
        c.save(fpath)
        rows = len(c.to_dataframe()) if hasattr(c, 'to_dataframe') else '?'
        log.info(f"  GPU {i}: saved {fpath} ({rows} rows)")

    return run_id, workload_rc


def run_parallel_single_gpu(args, workload_labels):
    """Run independent single-GPU workloads in parallel across GPUs."""
    import threading

    num_gpus = args.num_gpus
    run_id = uuid.uuid4().hex
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # If single label, replicate across all GPUs
    if len(workload_labels) == 1:
        workload_labels = workload_labels * num_gpus
    elif len(workload_labels) < num_gpus:
        # Pad with remaining labels cycling
        while len(workload_labels) < num_gpus:
            workload_labels.append(workload_labels[len(workload_labels) % len(workload_labels)])

    results = [None] * num_gpus

    def run_on_gpu(gpu_idx, label):
        gpu_run_id = uuid.uuid4().hex
        log.info(f"  GPU {gpu_idx}: starting {label}")

        c = TelemetryCollector(
            gpu_index=gpu_idx,
            interval_sec=args.interval,
            workload_label=label,
            enable_dcgm=not args.no_dcgm,
        )
        c.start()
        time.sleep(1)

        # Build command for this GPU
        try:
            wl = get_workload(label)
        except ValueError:
            log.error(f"  GPU {gpu_idx}: unknown workload '{label}'")
            c.stop()
            return

        cmd = list(wl["command"])
        env = os.environ.copy()
        env["CUDA_VISIBLE_DEVICES"] = str(gpu_idx)

        try:
            timeout = args.timeout + 30 if args.timeout else None
            result = subprocess.run(cmd, cwd=_REPO_ROOT, env=env, timeout=timeout)
            status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
        except subprocess.TimeoutExpired:
            status = "TIMEOUT"
        except Exception as e:
            status = f"ERROR: {e}"

        time.sleep(1)
        c.stop()

        fname = generate_filename(label, c.metadata.gpu_name, gpu_idx, gpu_run_id)
        fpath = str(output_dir / fname)
        c.save(fpath)
        log.info(f"  GPU {gpu_idx}: {label} {status} -> {fname}")
        results[gpu_idx] = (label, status)

    # Launch all GPUs in parallel
    threads = []
    for i in range(min(num_gpus, len(workload_labels))):
        t = threading.Thread(target=run_on_gpu, args=(i, workload_labels[i]))
        threads.append(t)
        t.start()

    for t in threads:
        t.join()

    return results


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU workload launcher")
    parser.add_argument("--workload", type=str, required=True,
                        help="Workload label (from registry or custom)")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--output-dir", type=str, default="data/")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--no-dcgm", action="store_true")
    parser.add_argument("--timeout", type=int, default=None)
    parser.add_argument("--parallel", action="store_true",
                        help="Run independent single-GPU workloads in parallel")
    parser.add_argument("cmd", nargs="*",
                        help="Custom command (after --)")
    args = parser.parse_args()

    if args.parallel:
        # Parallel single-GPU mode
        results = run_parallel_single_gpu(args, [args.workload])
        log.info(f"Parallel run complete: {results}")
    else:
        # Multi-GPU (DDP) mode
        if args.cmd:
            workload_cmd = args.cmd
        else:
            # Look up in registry
            try:
                wl = get_workload(args.workload)
            except ValueError as e:
                log.error(str(e))
                sys.exit(1)
            workload_cmd = list(wl["command"])
            if args.timeout is None:
                args.timeout = wl.get("default_timeout")

        run_id, rc = run_multi_gpu(args, workload_cmd)
        log.info(f"Multi-GPU run complete: {run_id[:12]} (rc={rc})")
        sys.exit(rc)


if __name__ == "__main__":
    main()
