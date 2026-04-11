#!/usr/bin/env python3
"""SPAR Week 8: RTX 4050 Laptop Full Workload Collection

Runs all compatible workloads with parameters adjusted for 6 GB VRAM.
Each completed run saves a parquet file immediately, so no data is lost
if the process is interrupted. Progress is tracked in a JSON file for
seamless resume.

27 workloads x 3 reps = 81 total runs.  Estimated runtime: 8-12 hours.

Usage:
    python scripts/run_laptop_collection.py              # full run
    python scripts/run_laptop_collection.py --reps 1     # single-rep test
    python scripts/run_laptop_collection.py --dry-run    # preview workload list
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────

REPO_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = REPO_ROOT / "data" / "laptop_collection_progress.json"
RUN_WORKLOAD = str(REPO_ROOT / "scripts" / "run_workload.py")
WORKLOADS_DIR = str(REPO_ROOT / "workloads")
PY = sys.executable

# Blender install location (winget default)
BLENDER_DIR = r"C:\Program Files\Blender Foundation\Blender 5.1"

# ── Workload Definitions ──────────────────────────────────────────────────
#
# Each entry is a dict with:
#   label:    workload name (used in telemetry metadata and filenames)
#   timeout:  max seconds before killing the workload
#   cmd:      (optional) override command list — if omitted, uses registry
#
# Workloads requiring batch-size or memory adjustments for 6 GB VRAM
# specify an explicit cmd.  All others use the registry defaults.

def _wl(label, timeout, cmd=None):
    """Helper to build a workload entry."""
    entry = {"label": label, "timeout": timeout}
    if cmd is not None:
        entry["cmd"] = cmd
    return entry


WORKLOADS = [
    # ── Baseline ──────────────────────────────────────────────────────
    _wl("idle", 130),

    # ── ML Training (default batch sizes fit 6 GB) ────────────────────
    _wl("pytorch_resnet_cifar10", 900),
    _wl("pytorch_resnet_cifar10_amp", 900),
    _wl("pytorch_mlp_cifar10", 900),

    # ── ML Training (batch sizes reduced for 6 GB VRAM) ──────────────
    _wl("gpt2_wikitext2", 1200, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "gpt2_finetune.py"),
        "--steps", "400", "--batch-size", "2",
    ]),
    _wl("gpt2_wikitext2_amp", 1200, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "gpt2_finetune.py"),
        "--steps", "400", "--batch-size", "4", "--amp",
    ]),
    _wl("bert_sst2", 900, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "bert_finetune.py"),
        "--steps", "300", "--batch-size", "8",
    ]),
    _wl("bert_sst2_amp", 900, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "bert_finetune.py"),
        "--steps", "300", "--batch-size", "16", "--amp",
    ]),

    # ── ML Inference ──────────────────────────────────────────────────
    _wl("resnet50_inference", 660),
    _wl("deepseek_r1_inference", 420),
    _wl("qwen25_inference", 420),
    _wl("whisper_inference", 420),

    # ── Scientific HPC ────────────────────────────────────────────────
    _wl("cufft_benchmark", 660),
    _wl("nbody_sim", 660),

    # ── Crypto Mining ─────────────────────────────────────────────────
    _wl("mining_ethash_proxy", 660),
    _wl("ethash_cuda", 360),

    # ── Rendering ─────────────────────────────────────────────────────
    _wl("rendering_proxy", 660),
    _wl("blender_bmw", 900),

    # ── Adversarial Training ──────────────────────────────────────────
    _wl("adversarial_A_util_modulation", 360),
    _wl("adversarial_B_low_util", 360),
    _wl("adversarial_D_temporal_disruption", 360),

    # ── Adversarial Composite (prealloc reduced: 4096 -> 2048 MB) ────
    _wl("adversarial_composite_00", 180, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "adversarial_composite.py"),
        "--idle-fraction", "0.0", "--duration", "120", "--prealloc-mb", "2048",
    ]),
    _wl("adversarial_composite_30", 180, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "adversarial_composite.py"),
        "--idle-fraction", "0.3", "--duration", "120", "--prealloc-mb", "2048",
    ]),
    _wl("adversarial_composite_50", 180, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "adversarial_composite.py"),
        "--idle-fraction", "0.5", "--duration", "120", "--prealloc-mb", "2048",
    ]),
    _wl("adversarial_composite_70", 180, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "adversarial_composite.py"),
        "--idle-fraction", "0.7", "--duration", "120", "--prealloc-mb", "2048",
    ]),
    _wl("adversarial_composite_90", 180, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "adversarial_composite.py"),
        "--idle-fraction", "0.9", "--duration", "120", "--prealloc-mb", "2048",
    ]),
    _wl("adversarial_composite_95", 240, cmd=[
        PY, os.path.join(WORKLOADS_DIR, "adversarial_composite.py"),
        "--idle-fraction", "0.95", "--duration", "120", "--prealloc-mb", "2048",
    ]),
]

# ── Progress Tracking ─────────────────────────────────────────────────────

def load_progress():
    """Load the set of completed (label, rep) pairs from the progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return set(tuple(x) for x in data.get("completed", []))
    return set()


def save_progress(completed: set, failed: list):
    """Save progress to JSON.  Completed runs are never re-run on resume."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "completed": sorted(completed),
            "failed": [list(x) for x in failed],
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)


# ── Run a Single Workload ─────────────────────────────────────────────────

def run_one(wl: dict, rep: int, env: dict, cooldown: int) -> tuple[bool, str]:
    """Run a single workload via run_workload.py.  Returns (success, message)."""

    label = wl["label"]
    timeout = wl["timeout"]

    # Build the command for run_workload.py
    base = [PY, RUN_WORKLOAD, "--workload", label, "--no-dcgm", "--timeout", str(timeout)]

    if "cmd" in wl:
        # Command-mode: pass the explicit command after --
        cmd = base + ["--"] + wl["cmd"]
    else:
        # Registry-mode: run_workload.py looks up the command
        cmd = base

    print(f"\n{'='*60}")
    print(f"  {label}  (rep {rep})")
    print(f"  timeout={timeout}s")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=timeout + 60,  # grace period beyond workload timeout
        )
        elapsed = time.time() - start
        if result.returncode == 0:
            print(f"--- {label} rep {rep} OK ({elapsed:.0f}s) ---")
            return True, f"{elapsed:.0f}s"
        else:
            msg = f"exit code {result.returncode} ({elapsed:.0f}s)"
            print(f"--- {label} rep {rep} FAILED: {msg} ---")
            return False, msg
    except subprocess.TimeoutExpired:
        elapsed = time.time() - start
        msg = f"timed out after {elapsed:.0f}s"
        print(f"--- {label} rep {rep} TIMEOUT: {msg} ---")
        return False, msg
    except Exception as e:
        msg = str(e)
        print(f"--- {label} rep {rep} ERROR: {msg} ---")
        return False, msg
    finally:
        # Cooldown between runs
        if cooldown > 0:
            print(f"  Cooldown {cooldown}s...")
            time.sleep(cooldown)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTX 4050 laptop data collection")
    parser.add_argument("--reps", type=int, default=3, help="Repetitions per workload (default: 3)")
    parser.add_argument("--cooldown", type=int, default=10, help="Seconds between runs (default: 10)")
    parser.add_argument("--dry-run", action="store_true", help="Print workload list and exit")
    args = parser.parse_args()

    total_runs = len(WORKLOADS) * args.reps

    if args.dry_run:
        print(f"Laptop collection: {len(WORKLOADS)} workloads x {args.reps} reps = {total_runs} runs\n")
        for i, wl in enumerate(WORKLOADS, 1):
            mode = "custom-cmd" if "cmd" in wl else "registry"
            print(f"  {i:2d}. {wl['label']:40s} timeout={wl['timeout']:4d}s  [{mode}]")
        return

    # Set up environment with Blender on PATH
    env = os.environ.copy()
    if os.path.isdir(BLENDER_DIR):
        env["PATH"] = BLENDER_DIR + os.pathsep + env.get("PATH", "")
        print(f"Added Blender to PATH: {BLENDER_DIR}")
    else:
        print(f"WARNING: Blender dir not found at {BLENDER_DIR}")
        print("         blender_bmw workload will likely fail.")

    # Load progress for resume
    completed = load_progress()
    if completed:
        print(f"\nResuming: {len(completed)} runs already completed, skipping them.")

    # Build the run plan
    plan = []
    for wl in WORKLOADS:
        for rep in range(1, args.reps + 1):
            key = (wl["label"], rep)
            if key not in completed:
                plan.append((wl, rep))

    if not plan:
        print("\nAll runs already completed!  Delete progress file to re-run:")
        print(f"  {PROGRESS_FILE}")
        return

    remaining = len(plan)
    print(f"\n{'='*60}")
    print(f"SPAR RTX 4050 Laptop Collection")
    print(f"{'='*60}")
    print(f"  Workloads:     {len(WORKLOADS)}")
    print(f"  Reps:          {args.reps}")
    print(f"  Total planned: {total_runs}")
    print(f"  Already done:  {len(completed)}")
    print(f"  Remaining:     {remaining}")
    print(f"  Cooldown:      {args.cooldown}s")
    print(f"  Started:       {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Progress file: {PROGRESS_FILE}")
    print(f"{'='*60}")

    failed = []
    run_num = 0
    start_time = time.time()

    try:
        for wl, rep in plan:
            run_num += 1
            elapsed_total = time.time() - start_time
            avg_per_run = elapsed_total / run_num if run_num > 1 else 0
            eta = avg_per_run * (remaining - run_num) if run_num > 1 else 0

            print(f"\n[{run_num}/{remaining}]  "
                  f"elapsed {timedelta(seconds=int(elapsed_total))}  "
                  f"ETA {timedelta(seconds=int(eta))}")

            success, msg = run_one(wl, rep, env, args.cooldown)

            if success:
                completed.add((wl["label"], rep))
            else:
                failed.append((wl["label"], rep, msg))

            # Save progress after every run
            save_progress(completed, failed)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user.  Progress saved — rerun to resume.")
        save_progress(completed, failed)
        sys.exit(1)

    # Final summary
    total_elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Total time:  {timedelta(seconds=int(total_elapsed))}")
    print(f"  Completed:   {len(completed)} / {total_runs}")
    print(f"  Failed:      {len(failed)}")

    if failed:
        print(f"\nFailed runs:")
        for label, rep, msg in failed:
            print(f"  - {label} rep {rep}: {msg}")

    # Count output files
    data_dir = REPO_ROOT / "data"
    rtx4050_files = list(data_dir.glob("*RTX_4050*.parquet")) + list(data_dir.glob("*GeForce_RTX_4050*.parquet"))
    print(f"\n  RTX 4050 parquet files: {len(rtx4050_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
