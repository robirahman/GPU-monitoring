#!/usr/bin/env python3
"""Controlled batch-size dilution sweep for evasion tax curve (Figure 2).

Runs Strategy L diluted training at N=1,2,3,5,10,20 plus a pure inference
baseline, all with identical parameters except the dilution factor:
  - Model: ResNet-50
  - Batch size: 128
  - Duration: 300s
  - Data: CIFAR-10 test set (Resize 224)

This produces a clean evasion tax curve where the ONLY variable is the
fraction of batches that include a backward pass.

Includes Tapo P115 wall-power logging if --tapo is passed.
Progress is tracked in a JSON file for crash resilience.

Usage:
    python scripts/run_dilution_sweep.py --tapo --no-dcgm
    python scripts/run_dilution_sweep.py --no-dcgm          # without wall power
    python scripts/run_dilution_sweep.py --dry-run           # preview workload list
    python scripts/run_dilution_sweep.py --reset             # clear progress
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = REPO_ROOT / "data" / "dilution_sweep_progress.json"
RUN_WORKLOAD = str(REPO_ROOT / "scripts" / "run_workload.py")
RUN_WORKLOAD_POWER = str(REPO_ROOT / "scripts" / "run_workload_with_power.py")
PY = sys.executable

# Ordered from least to most crash-prone
SWEEP_WORKLOADS = [
    "dilution_sweep_inference",  # pure inference (N=inf)
    "dilution_sweep_N20",        # 5% training
    "dilution_sweep_N10",        # 10% training
    "dilution_sweep_N5",         # 20% training
    "dilution_sweep_N3",         # 33% training
    "dilution_sweep_N2",         # 50% training
    "dilution_sweep_N1",         # 100% training (every batch)
]


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": [], "started": None}


def save_progress(progress):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def run_workload(label, use_tapo=False, no_dcgm=True):
    """Run a single workload via run_workload[_with_power].py."""
    if use_tapo:
        cmd = [PY, RUN_WORKLOAD_POWER, "--workload", label, "--tapo"]
    else:
        cmd = [PY, RUN_WORKLOAD, "--workload", label]

    if no_dcgm:
        cmd.append("--no-dcgm")

    print(f"\n{'='*60}")
    print(f"  Running: {label}")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Time: {datetime.now().strftime('%H:%M:%S')}")
    print(f"{'='*60}\n")

    result = subprocess.run(cmd, cwd=str(REPO_ROOT))
    return result.returncode


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Controlled dilution sweep")
    parser.add_argument("--tapo", action="store_true", help="Enable Tapo P115 wall power")
    parser.add_argument("--no-dcgm", action="store_true", default=True,
                        help="Disable DCGM (default: True)")
    parser.add_argument("--dry-run", action="store_true", help="Preview workloads")
    parser.add_argument("--reset", action="store_true", help="Clear progress")
    args = parser.parse_args()

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print("Progress cleared.")
        return

    if args.dry_run:
        progress = load_progress()
        done = {w for w in progress["completed"]}
        print("Dilution sweep workloads:")
        for label in SWEEP_WORKLOADS:
            status = "DONE" if label in done else "pending"
            print(f"  [{status:7s}] {label}")
        n_remaining = sum(1 for w in SWEEP_WORKLOADS if w not in done)
        est_min = n_remaining * 5.5  # ~300s + overhead per workload
        print(f"\n{n_remaining} remaining, estimated {est_min:.0f} minutes")
        return

    progress = load_progress()
    done = set(progress["completed"])
    remaining = [w for w in SWEEP_WORKLOADS if w not in done]

    if not remaining:
        print("All dilution sweep workloads already completed!")
        print(f"Progress file: {PROGRESS_FILE}")
        return

    print(f"Dilution Sweep: {len(remaining)} workloads remaining")
    print(f"  Completed: {len(done)}")
    est = len(remaining) * 5.5
    print(f"  Estimated time: {est:.0f} minutes")
    print(f"  Wall power: {'Tapo P115' if args.tapo else 'disabled'}")
    print()

    sweep_start = time.time()

    for i, label in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] Starting {label}...")
        progress["started"] = label
        save_progress(progress)

        rc = run_workload(label, use_tapo=args.tapo, no_dcgm=args.no_dcgm)

        if rc == 0:
            progress["completed"].append(label)
            progress["started"] = None
            save_progress(progress)
            print(f"  {label} completed successfully.")
        else:
            progress["failed"].append({"label": label, "exit_code": rc,
                                        "time": datetime.now().isoformat()})
            progress["started"] = None
            save_progress(progress)
            print(f"  WARNING: {label} failed with exit code {rc}. Continuing...")

    elapsed = time.time() - sweep_start
    print(f"\n{'='*60}")
    print(f"Dilution sweep complete!")
    print(f"  Total time: {timedelta(seconds=int(elapsed))}")
    print(f"  Completed: {len(progress['completed'])}")
    print(f"  Failed: {len(progress['failed'])}")
    print(f"  Data in: data/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
