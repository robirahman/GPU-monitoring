#!/usr/bin/env python3
"""Round 4 adversarial data collection on RTX 5080.

Runs 7 new adversarial strategies (M x4 idle fractions + N x3 accum steps)
with Tapo P115 wall-power logging and crash-resilient progress tracking.

Usage:
    python scripts/run_round4_collection.py --tapo --no-dcgm
    python scripts/run_round4_collection.py --dry-run
    python scripts/run_round4_collection.py --reset
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = REPO_ROOT / "data" / "round4_collection_progress.json"
RUN_WORKLOAD_POWER = str(REPO_ROOT / "scripts" / "run_workload_with_power.py")
RUN_WORKLOAD = str(REPO_ROOT / "scripts" / "run_workload.py")
PY = sys.executable

# Ordered: N strategies first (faster — ResNet-50 batch 128 runs quick),
# then M strategies (slower — gradient checkpointing + tiny batch)
WORKLOADS = [
    "adversarial_N_grad_accum_4",
    "adversarial_N_grad_accum_8",
    "adversarial_N_grad_accum_16",
    "adversarial_M_composite_memmin_10",
    "adversarial_M_composite_memmin_20",
    "adversarial_M_composite_memmin_30",
    "adversarial_M_composite_memmin_40",
]


def load_progress():
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": [], "started": None}


def save_progress(progress):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


def run_workload(label, use_tapo=False, no_dcgm=True):
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
    parser = argparse.ArgumentParser(description="Round 4 adversarial collection")
    parser.add_argument("--tapo", action="store_true", help="Enable Tapo P115 wall power")
    parser.add_argument("--no-dcgm", action="store_true", default=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print("Progress cleared.")
        return

    if args.dry_run:
        progress = load_progress()
        done = set(progress["completed"])
        print("Round 4 workloads:")
        for label in WORKLOADS:
            status = "DONE" if label in done else "pending"
            print(f"  [{status:7s}] {label}")
        n_remaining = sum(1 for w in WORKLOADS if w not in done)
        print(f"\n{n_remaining} remaining, estimated {n_remaining * 5.5:.0f} minutes")
        return

    progress = load_progress()
    done = set(progress["completed"])
    remaining = [w for w in WORKLOADS if w not in done]

    if not remaining:
        print("All Round 4 workloads already completed!")
        return

    print(f"Round 4 Collection: {len(remaining)} workloads remaining")
    print(f"  Completed: {len(done)}")
    print(f"  Estimated time: {len(remaining) * 5.5:.0f} minutes")
    print(f"  Wall power: {'Tapo P115' if args.tapo else 'disabled'}")

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
    print(f"Round 4 collection complete!")
    print(f"  Total time: {timedelta(seconds=int(elapsed))}")
    print(f"  Completed: {len(progress['completed'])}")
    print(f"  Failed: {len(progress['failed'])}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
