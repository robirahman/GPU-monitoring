#!/usr/bin/env python3
"""SPAR Workload Launcher with PicoScope current-probe logging.

Wraps scripts/run_workload.py (as a subprocess, no code duplication) and runs
scripts/pico_scope_logger.py alongside it, sharing one run_id so the Tier 1
NVML telemetry and the oscilloscope envelope can be joined on timestamps.

Outputs into --output-dir:
  {label}_{gpu}_{run_id8}_{ts}.parquet             (Tier 1, written by run_workload.py)
  {label}_{gpu}_{run_id8}_{ts}_scope.parquet       (1 kHz envelope: mean/rms/min/max)
  {label}_{gpu}_{run_id8}_{ts}_scope_raw.npy       (first N seconds raw int16 @ ~1 MS/s)

Usage:
    python scripts/run_workload_with_scope.py --workload idle --no-dcgm
    ~/vllm-venv/bin/python scripts/run_workload_with_scope.py --workload llm_infer_qwen25_7b_fp16 \
        --scope-python ~/pico-venv/bin/python --no-dcgm
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import uuid
from datetime import datetime

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from workloads.registry import get_workload


def main():
    ap = argparse.ArgumentParser(description="Run workload with GPU telemetry + PicoScope logging.")
    ap.add_argument("--workload", type=str, required=True)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--output-dir", type=str, default="data/h200_nvl_scope")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--no-dcgm", action="store_true")
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--scope-python", type=str,
                    default=os.path.expanduser("~/pico-venv/bin/python"),
                    help="Interpreter with picosdk installed, used for the scope logger")
    ap.add_argument("--range-mv", type=int, default=5000)
    ap.add_argument("--raw-seconds", type=float, default=5.0)
    ap.add_argument("--dcgm", action="store_true",
                    help="Also run scripts/dcgm_logger.py (needs nv-hostengine as root)")
    ap.add_argument("--dcgm-python", type=str,
                    default=os.path.expanduser("~/pico-venv/bin/python"))
    args, extra = ap.parse_known_args()

    timeout = args.timeout or get_workload(args.workload).get("default_timeout") or 600
    run_id = args.run_id or uuid.uuid4().hex[:12]
    out_dir = os.path.join(_REPO_ROOT, args.output_dir)
    os.makedirs(out_dir, exist_ok=True)

    # scope filename mirrors run_workload.py's convention with a _scope suffix;
    # the gpu-name segment is filled in by matching the telemetry file afterwards,
    # so here we key on run_id + timestamp only.
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    scope_base = os.path.join(out_dir, f"{args.workload}_scope_{run_id}_{stamp}")
    scope_parquet = scope_base + ".parquet"
    scope_raw = scope_base + "_raw.npy"

    logger_cmd = [
        args.scope_python, os.path.join(_REPO_ROOT, "scripts", "pico_scope_logger.py"),
        "--seconds", str(timeout + 60),  # generous; we SIGTERM it when the workload ends
        "--range-mv", str(args.range_mv),
        "--out", scope_parquet,
    ]
    if args.raw_seconds > 0:
        logger_cmd += ["--save-raw", scope_raw, "--raw-seconds", str(args.raw_seconds)]

    print(f"[scope-wrapper] run_id={run_id}  scope -> {scope_parquet}", flush=True)
    scope_proc = subprocess.Popen(logger_cmd, stdout=sys.stdout, stderr=sys.stderr)

    dcgm_proc = None
    dcgm_parquet = None
    if args.dcgm:
        dcgm_parquet = os.path.join(out_dir, f"{args.workload}_dcgm_{run_id}_{stamp}.parquet")
        dcgm_cmd = [
            args.dcgm_python, os.path.join(_REPO_ROOT, "scripts", "dcgm_logger.py"),
            "--seconds", str(timeout + 60), "--gpu", str(args.gpu_index),
            "--run-id", run_id, "--out", dcgm_parquet,
        ]
        dcgm_proc = subprocess.Popen(dcgm_cmd, stdout=sys.stdout, stderr=sys.stderr)

    time.sleep(3)  # let the loggers reach steady state before the workload starts
    if scope_proc.poll() is not None:
        print("[scope-wrapper] ERROR: scope logger died at startup", flush=True)
        sys.exit(97)

    rw_cmd = [
        sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_workload.py"),
        "--workload", args.workload,
        "--gpu-index", str(args.gpu_index),
        "--output-dir", args.output_dir,
        "--interval", str(args.interval),
        "--run-id", run_id,
    ]
    if args.no_dcgm:
        rw_cmd.append("--no-dcgm")
    if args.timeout:
        rw_cmd += ["--timeout", str(args.timeout)]
    if extra:
        rw_cmd += extra

    exit_code = 96
    try:
        exit_code = subprocess.call(rw_cmd, cwd=_REPO_ROOT)
    finally:
        time.sleep(2)  # capture a couple seconds of post-workload cooldown
        for proc in (scope_proc, dcgm_proc):
            if proc is not None and proc.poll() is None:
                proc.send_signal(signal.SIGTERM)
        for proc in (scope_proc, dcgm_proc):
            if proc is None:
                continue
            try:
                proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                proc.kill()

    ok = os.path.exists(scope_parquet) and os.path.getsize(scope_parquet) > 10000
    dcgm_ok = (not args.dcgm) or (dcgm_parquet and os.path.exists(dcgm_parquet)
                                  and os.path.getsize(dcgm_parquet) > 1000)
    print(f"[scope-wrapper] workload exit={exit_code}  scope ok={ok}  dcgm ok={dcgm_ok}", flush=True)
    if exit_code == 0 and not (ok and dcgm_ok):
        exit_code = 98  # workload fine but a sensor capture missing -> rep must be retried
    sys.exit(exit_code)


if __name__ == "__main__":
    main()
