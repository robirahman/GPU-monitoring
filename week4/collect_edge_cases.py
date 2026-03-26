#!/usr/bin/env python3
"""
Week 4 Step 1: Edge-Case Data Collection
Runs workloads on 2 H100 GPUs in parallel to collect edge-case telemetry:
  - Varied batch sizes (small/large)
  - CPU-bottlenecked training (tiny batches)
  - Short runs (<60s)
  - Mixed concurrent workloads (training on GPU0 + inference on GPU1)

Uses week4/workloads_w4.py (no torchvision dependency).
Outputs parquet files to week4/data/
"""

import os
import sys
import subprocess
import time
import threading
import uuid
import logging
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))

from scripts.collect_telemetry import TelemetryCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("week4.collect")

OUTPUT_DIR = REPO_ROOT / "week4" / "data"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Path to self-contained week4 workloads (no torchvision dependency)
W4_SCRIPT = str(REPO_ROOT / "week4" / "workloads_w4.py")


def cmd(workload, extra=None):
    c = [sys.executable, W4_SCRIPT, "--workload", workload]
    if extra:
        c.extend(extra)
    return c


# GPU 0 workloads: varied batch sizes + short runs
GPU0_WORKLOADS = [
    ("resnet18_small_batch", cmd("resnet_small_batch"),           300, "batch8"),
    ("resnet18_large_batch", cmd("resnet_large_batch"),           400, "batch2048"),
    ("resnet18_short_run",   cmd("resnet_short"),                 120, "short"),
    ("mlp_small_batch",      cmd("mlp_small"),                    300, "batch16"),
    ("mlp_large_batch",      cmd("mlp_large"),                    300, "batch4096"),
    ("cufft_short",          cmd("cufft", ["--duration", "120"]), 180, "dur120"),
]

# GPU 1 workloads: inference + HPC (run concurrently with GPU 0)
GPU1_WORKLOADS = [
    ("resnet50_inference_small", cmd("inference_small", ["--duration", "120"]),  200, "bs32"),
    ("resnet50_inference_large", cmd("inference_large", ["--duration", "120"]),  200, "bs1024"),
    ("nbody_short",              cmd("nbody",  ["--duration", "120"]),           180, "dur120"),
    ("idle",                     cmd("idle",   ["--duration", "120"]),           150, "h100"),
    ("cufft_concurrent",         cmd("cufft",  ["--duration", "120"]),           180, "concurrent"),
    ("resnet50_concurrent",      cmd("inference", ["--duration", "120"]),        200, "concurrent"),
]


def run_workload_with_telemetry(label, command, gpu_index, timeout=600, tag=""):
    """Collect telemetry while running a subprocess workload on a given GPU."""
    run_id = uuid.uuid4().hex[:8]
    full_label = f"{label}{'_' + tag if tag else ''}"
    filename = (f"{full_label}_H100_gpu{gpu_index}_{run_id}_"
                f"{time.strftime('%Y%m%d_%H%M%S')}.parquet")
    output_path = OUTPUT_DIR / filename

    log.info("[GPU%d] START  label=%s", gpu_index, full_label)

    collector = TelemetryCollector(
        gpu_index=gpu_index,
        interval_sec=1.0,
        workload_label=full_label,
        run_id=run_id,
        enable_dcgm=False,
    )
    collector.start()
    time.sleep(2)

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(gpu_index)
    t0 = time.time()
    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            cwd=str(REPO_ROOT),
        )
        try:
            stdout, _ = proc.communicate(timeout=timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            stdout, _ = proc.communicate()
        rc = proc.returncode
        duration = time.time() - t0
        log.info("[GPU%d] DONE   label=%s  exit=%d  dur=%.1fs",
                 gpu_index, full_label, rc, duration)
        if rc != 0 and stdout:
            out = stdout.decode("utf-8", errors="replace")
            log.warning("[GPU%d] STDERR: %s", gpu_index, out[-500:])
    except Exception as e:
        log.error("[GPU%d] ERROR  label=%s  %s", gpu_index, full_label, e)

    time.sleep(2)
    collector.stop()
    collector.save(str(output_path))
    collector.cleanup()
    log.info("[GPU%d] SAVED  %s  (%d samples)", gpu_index,
             output_path.name, collector.sample_count)
    return str(output_path)


def run_gpu_queue(workloads, gpu_index):
    saved = []
    for label, command, timeout, tag in workloads:
        path = run_workload_with_telemetry(label, command, gpu_index, timeout, tag)
        saved.append(path)
    return saved


def main():
    log.info("=" * 60)
    log.info("Week 4 Edge-Case Data Collection (2x H100 80GB)")
    log.info("GPU0: varied batch sizes + short runs")
    log.info("GPU1: inference + HPC (concurrent with GPU0)")
    log.info("Output: %s", OUTPUT_DIR)
    log.info("=" * 60)

    results = {"gpu0": [], "gpu1": []}
    errors = []

    def run_gpu0():
        try:
            results["gpu0"] = run_gpu_queue(GPU0_WORKLOADS, 0)
        except Exception as e:
            errors.append(str(e))
            log.error("GPU0 thread: %s", e)

    def run_gpu1():
        try:
            results["gpu1"] = run_gpu_queue(GPU1_WORKLOADS, 1)
        except Exception as e:
            errors.append(str(e))
            log.error("GPU1 thread: %s", e)

    t0 = time.time()
    t0_thread = threading.Thread(target=run_gpu0, name="GPU0")
    t1_thread = threading.Thread(target=run_gpu1, name="GPU1")
    t0_thread.start(); t1_thread.start()
    t0_thread.join(); t1_thread.join()

    total = time.time() - t0
    all_files = results["gpu0"] + results["gpu1"]
    log.info("=" * 60)
    log.info("Collection complete in %.1fs — %d files saved", total, len(all_files))
    if errors:
        log.warning("Errors: %s", errors)
    for f in all_files:
        log.info("  %s", Path(f).name)

    manifest = OUTPUT_DIR / "manifest.txt"
    with open(manifest, "w") as fh:
        fh.write(f"Week 4 edge-case collection — {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        fh.write(f"GPU: 2x NVIDIA H100 80GB HBM3\n")
        fh.write(f"Duration: {total:.1f}s\n\n")
        for f in all_files:
            fh.write(f"{Path(f).name}\n")
    log.info("Manifest: %s", manifest)


if __name__ == "__main__":
    main()
