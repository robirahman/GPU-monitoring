#!/usr/bin/env python3
"""SPAR Week 8: RTX 5080 Desktop Full Workload Collection

Runs all workloads at desktop-scale parameters with Tier 1 NVML telemetry.
Progress is tracked in a JSON file — if the process is interrupted (e.g. by
Windows Update), rerun the script and it will resume where it left off.

Optional Tier 5 wall-power logging via Shelly Plus Plug US or Tapo P115.

Usage:
    python scripts/run_desktop_collection.py                          # full run, no power
    python scripts/run_desktop_collection.py --shelly-ip 192.168.1.X  # with Shelly wall power
    python scripts/run_desktop_collection.py --tapo                   # with Tapo P115 wall power
    python scripts/run_desktop_collection.py --reps 1                 # single-rep test
    python scripts/run_desktop_collection.py --dry-run                # preview workload list
    python scripts/run_desktop_collection.py --reset                  # clear progress, start fresh
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
PROGRESS_FILE = REPO_ROOT / "data" / "desktop_collection_progress.json"
RUN_WORKLOAD = str(REPO_ROOT / "scripts" / "run_workload.py")
RUN_WORKLOAD_POWER = str(REPO_ROOT / "scripts" / "run_workload_with_power.py")
WORKLOADS_DIR = str(REPO_ROOT / "workloads")
PY = sys.executable

# ── Workload Definitions ──────────────────────────────────────────────────

def _wl(label, timeout, cmd=None):
    """Helper to build a workload entry."""
    entry = {"label": label, "timeout": timeout}
    if cmd is not None:
        entry["cmd"] = cmd
    return entry


WORKLOADS = [
    # ── Baseline ──────────────────────────────────────────────────────
    _wl("idle", 130),

    # ── ML Training (same params as A100 baseline) ────────────────────
    _wl("pytorch_resnet_cifar10", 900),
    _wl("pytorch_resnet_cifar10_amp", 900),
    _wl("pytorch_mlp_cifar10", 900),
    _wl("gpt2_wikitext2", 1200),
    _wl("gpt2_wikitext2_amp", 1200),
    _wl("bert_sst2", 900),
    _wl("bert_sst2_amp", 900),

    # ── ML Inference ──────────────────────────────────────────────────
    _wl("resnet50_inference", 660),

    # ── Scientific HPC ────────────────────────────────────────────────
    _wl("cufft_benchmark", 660),
    _wl("nbody_sim", 660),

    # ── Crypto Mining ─────────────────────────────────────────────────
    _wl("mining_ethash_proxy", 660),

    # ── Rendering ─────────────────────────────────────────────────────
    _wl("rendering_proxy", 660),

    # ── Other ─────────────────────────────────────────────────────────
    # ffmpeg_nvenc excluded: requires ffmpeg installation

    # ── Adversarial Round 1 ──────────────────────────────────────────
    _wl("adversarial_A_util_modulation", 360),
    _wl("adversarial_B_low_util", 360),
    _wl("adversarial_D_temporal_disruption", 360),
    _wl("adversarial_E_memory_minimal", 360),
    _wl("adversarial_F_interleave_30", 360),
    _wl("adversarial_F_interleave_50", 360),
    _wl("adversarial_G_clock_throttled", 360),

    # ── Adversarial Round 2 ──────────────────────────────────────────
    _wl("adversarial_H_mimicry_mining", 360),
    _wl("adversarial_H_mimicry_cufft", 360),
    _wl("adversarial_I_stochastic_30", 360),
    _wl("adversarial_J_pid_75", 360),

    # ── Adversarial Round 3 ──────────────────────────────────────────
    _wl("adversarial_K_online_learning", 360),
    _wl("adversarial_L_diluted_20", 360),
    _wl("adversarial_L_diluted_10", 360),
    _wl("adversarial_L_diluted_5", 360),
    _wl("adversarial_L_diluted_2", 360),  # crash-prone, run last

    # ── Adversarial Composite Sweep ───────────────────────────────────
    _wl("adversarial_composite_00", 180),
    _wl("adversarial_composite_30", 180),
    _wl("adversarial_composite_50", 180),
    _wl("adversarial_composite_70", 180),
    _wl("adversarial_composite_90", 180),
    _wl("adversarial_composite_95", 240),
]

# ── Progress Tracking ─────────────────────────────────────────────────────

def load_progress() -> set:
    """Load the set of completed (label, rep) pairs from the progress file."""
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            data = json.load(f)
        return set(tuple(x) for x in data.get("completed", []))
    return set()


def save_progress(completed: set, failed: list):
    """Save progress to JSON. Completed runs are never re-run on resume."""
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(PROGRESS_FILE, "w") as f:
        json.dump({
            "completed": sorted(completed),
            "failed": [list(x) for x in failed],
            "last_updated": datetime.now().isoformat(),
        }, f, indent=2)


# ── Run a Single Workload ─────────────────────────────────────────────────

def run_one(wl: dict, rep: int, env: dict, cooldown: int,
            shelly_ip: str | None = None,
            tapo: bool = False) -> tuple[bool, str]:
    """Run a single workload via run_workload.py. Returns (success, message)."""

    label = wl["label"]
    timeout = wl["timeout"]

    # Choose launcher: with or without wall-power logging
    if shelly_ip or tapo:
        launcher = RUN_WORKLOAD_POWER
        base = [PY, launcher, "--workload", label, "--no-dcgm",
                "--timeout", str(timeout)]
        if shelly_ip:
            base += ["--shelly-ip", shelly_ip]
        elif tapo:
            base += ["--tapo"]
    else:
        launcher = RUN_WORKLOAD
        base = [PY, launcher, "--workload", label, "--no-dcgm",
                "--timeout", str(timeout)]

    if "cmd" in wl:
        cmd = base + ["--"] + wl["cmd"]
    else:
        cmd = base

    power = "tapo" if tapo else ("shelly" if shelly_ip else "no")
    print(f"\n{'='*60}")
    print(f"  {label}  (rep {rep})")
    print(f"  timeout={timeout}s  wall_power={power}")
    print(f"{'='*60}")

    start = time.time()
    try:
        result = subprocess.run(
            cmd,
            cwd=str(REPO_ROOT),
            env=env,
            timeout=timeout + 60,  # grace period
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
        if cooldown > 0:
            print(f"  Cooldown {cooldown}s...")
            time.sleep(cooldown)


# ── Main ──────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="RTX 5080 desktop data collection")
    parser.add_argument("--reps", type=int, default=3,
                        help="Repetitions per workload (default: 3)")
    parser.add_argument("--cooldown", type=int, default=15,
                        help="Seconds between runs (default: 15)")
    parser.add_argument("--shelly-ip", type=str, default=None,
                        help="Shelly Plus Plug IP for Tier 5 wall-power logging")
    parser.add_argument("--tapo", action="store_true",
                        help="Use Tapo P115 for Tier 5 wall-power logging")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print workload list and exit")
    parser.add_argument("--reset", action="store_true",
                        help="Clear progress file and start from scratch")
    args = parser.parse_args()

    total_runs = len(WORKLOADS) * args.reps

    if args.dry_run:
        print(f"Desktop collection: {len(WORKLOADS)} workloads x {args.reps} reps"
              f" = {total_runs} runs\n")
        for i, wl in enumerate(WORKLOADS, 1):
            mode = "custom-cmd" if "cmd" in wl else "registry"
            print(f"  {i:2d}. {wl['label']:40s} timeout={wl['timeout']:4d}s  [{mode}]")
        return

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            print(f"Cleared progress file: {PROGRESS_FILE}")
        else:
            print("No progress file to clear.")
        return

    env = os.environ.copy()

    # Load progress for resume
    completed = load_progress()
    if completed:
        print(f"\nResuming: {len(completed)} runs already completed, skipping them.")

    # Seed progress with existing parquet files that match completed runs
    # (handles the case where runs completed but progress file was lost)
    data_dir = REPO_ROOT / "data"
    existing_files = list(data_dir.glob("*RTX_5080*.parquet")) + \
                     list(data_dir.glob("*GeForce_RTX_5080*.parquet"))
    existing_by_label = {}
    for f in existing_files:
        # Extract label from filename: {label}_{gpu}_{hash}_{date}.parquet
        parts = f.stem.split("_NVIDIA_")
        if parts:
            label = parts[0]
            existing_by_label.setdefault(label, []).append(f)

    # Build the run plan
    plan = []
    for wl in WORKLOADS:
        for rep in range(1, args.reps + 1):
            key = (wl["label"], rep)
            if key not in completed:
                plan.append((wl, rep))

    if not plan:
        print("\nAll runs already completed! Use --reset to start over.")
        return

    remaining = len(plan)
    print(f"\n{'='*60}")
    print(f"SPAR RTX 5080 Desktop Collection")
    print(f"{'='*60}")
    print(f"  Workloads:     {len(WORKLOADS)}")
    print(f"  Reps:          {args.reps}")
    print(f"  Total planned: {total_runs}")
    print(f"  Already done:  {len(completed)}")
    print(f"  Remaining:     {remaining}")
    print(f"  Cooldown:      {args.cooldown}s")
    power_mode = "Tapo P115" if args.tapo else (args.shelly_ip if args.shelly_ip else "disabled")
    print(f"  Wall power:    {power_mode}")
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

            success, msg = run_one(wl, rep, env, args.cooldown,
                                   shelly_ip=args.shelly_ip,
                                   tapo=args.tapo)

            if success:
                completed.add((wl["label"], rep))
            else:
                failed.append((wl["label"], rep, msg))

            # Save progress after every run
            save_progress(completed, failed)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Progress saved — rerun to resume.")
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

    rtx5080_files = list(data_dir.glob("*RTX_5080*.parquet")) + \
                    list(data_dir.glob("*GeForce_RTX_5080*.parquet"))
    print(f"\n  RTX 5080 parquet files: {len(rtx5080_files)}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
