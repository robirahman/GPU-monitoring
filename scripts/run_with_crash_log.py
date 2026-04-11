#!/usr/bin/env python3
"""Crash-resilient workload runner with detailed logging.

Logs the start and completion time of every workload to a JSON file.
If the process is killed (e.g., by a system crash), the log shows
exactly which workload was running when the crash occurred.

Skips workloads that have already completed (based on existing parquet files).

Usage:
    python scripts/run_with_crash_log.py
    python scripts/run_with_crash_log.py --log-file data/crash_log.json
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
LAUNCHER = str(REPO_ROOT / "scripts" / "run_workload.py")
LOG_FILE = REPO_ROOT / "data" / "crash_log.json"


def load_log(log_file):
    if log_file.exists():
        with open(log_file) as f:
            return json.load(f)
    return {"runs": []}


def save_log(log_file, log_data):
    with open(log_file, "w") as f:
        json.dump(log_data, f, indent=2)


def count_existing(label):
    """Count how many completed parquet files exist for this label on RTX 5080."""
    pattern = str(REPO_ROOT / "data" / f"{label}*5080*.parquet")
    return len(glob.glob(pattern))


def run_one(label, timeout, log_data, log_file, cooldown=15):
    """Run a single workload, logging start/end times."""
    entry = {
        "label": label,
        "start_time": datetime.now().isoformat(),
        "end_time": None,
        "status": "STARTED",
        "exit_code": None,
    }
    log_data["runs"].append(entry)
    save_log(log_file, log_data)  # Flush to disk BEFORE running

    print(f"  [{datetime.now().strftime('%H:%M:%S')}] STARTING: {label}", flush=True)

    try:
        result = subprocess.run(
            [PY, LAUNCHER, "--workload", label, "--no-dcgm", "--timeout", str(timeout)],
            timeout=timeout + 60,
            cwd=str(REPO_ROOT),
        )
        entry["end_time"] = datetime.now().isoformat()
        entry["exit_code"] = result.returncode
        entry["status"] = "OK" if result.returncode == 0 else f"FAILED({result.returncode})"
    except subprocess.TimeoutExpired:
        entry["end_time"] = datetime.now().isoformat()
        entry["status"] = "TIMEOUT"
    except Exception as e:
        entry["end_time"] = datetime.now().isoformat()
        entry["status"] = f"ERROR: {e}"

    save_log(log_file, log_data)  # Flush after completion
    print(f"  [{datetime.now().strftime('%H:%M:%S')}] {entry['status']}: {label}", flush=True)

    time.sleep(cooldown)
    return entry["status"] == "OK"


def main():
    parser = argparse.ArgumentParser(description="Crash-resilient workload runner")
    parser.add_argument("--log-file", type=str, default=str(LOG_FILE))
    parser.add_argument("--reps", type=int, default=3)
    parser.add_argument("--cooldown", type=int, default=15)
    args = parser.parse_args()

    log_file = Path(args.log_file)
    log_data = load_log(log_file)

    # Workloads to run, ordered safest-first (least likely to crash)
    workloads = [
        # Diluted training — lower dilution = less mode switching = safer
        ("adversarial_L_diluted_20", 360),  # 5% training, 95% inference
        ("adversarial_L_diluted_10", 360),  # 10% training
        ("adversarial_L_diluted_5", 360),   # 20% training
        # These might crash — run last
        ("adversarial_L_diluted_2", 360),   # 50% training (crashed twice before)
        ("adversarial_K_online_learning", 360),  # 100% training-as-inference
    ]

    print(f"{'='*60}")
    print(f"Crash-Resilient Round 3 Collection")
    print(f"{'='*60}")
    print(f"  Log file: {log_file}")
    print(f"  Started: {datetime.now().isoformat()}")
    print(f"{'='*60}\n")

    for label, timeout in workloads:
        existing = count_existing(label)
        needed = args.reps - existing
        if needed <= 0:
            print(f"  SKIP: {label} — already have {existing}/{args.reps} reps")
            continue

        print(f"\n  {label}: need {needed} more reps ({existing}/{args.reps} done)")
        for rep in range(needed):
            success = run_one(label, timeout, log_data, log_file, args.cooldown)
            if not success:
                print(f"  WARNING: {label} failed, continuing to next workload")
                break

    # Summary
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    for label, _ in workloads:
        count = count_existing(label)
        status = "OK" if count >= args.reps else f"NEED {args.reps - count} MORE"
        print(f"  {label:40s} {count}/{args.reps} reps  [{status}]")

    # Check for crash evidence
    for entry in log_data["runs"]:
        if entry["status"] == "STARTED":
            print(f"\n  CRASH DETECTED: {entry['label']} started at {entry['start_time']} "
                  f"but never completed!")


if __name__ == "__main__":
    main()
