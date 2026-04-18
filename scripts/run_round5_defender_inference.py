#!/usr/bin/env python3
"""Round 5b defender-response inference data collection orchestrator.

Drives the 16 new Round 5b inference workloads (see plan
`great-fix-the-class-drifting-biscuit.md` Part 3) with GPU-tier awareness:

  - tier=consumer  : 5080, 5090, 4050, 4090 — runs workloads 1-11 only
  - tier=datacenter: A100-80, H100, H200, B200 — runs full 1-14 + stretch TP
  - explicit --workloads flag overrides the tier filter

Modelled on `scripts/run_dilution_sweep.py`: JSON progress tracking for
crash resilience, `--dry-run` preview, `--reset` wipe, optional Tapo
wall-power logging, 3 reps per workload (unique run_ids in filenames).

Usage:
    python scripts/run_round5_defender_inference.py --tier consumer --no-dcgm --reps 3
    python scripts/run_round5_defender_inference.py --tier datacenter --reps 3
    python scripts/run_round5_defender_inference.py --workloads llm_infer_phi3_mini llm_infer_llama32_3b
    python scripts/run_round5_defender_inference.py --dry-run --tier datacenter
    python scripts/run_round5_defender_inference.py --reset
"""

import argparse
import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PROGRESS_FILE = REPO_ROOT / "data" / "round5_defender_inference_progress.json"
RUN_WORKLOAD = str(REPO_ROOT / "scripts" / "run_workload.py")
RUN_WORKLOAD_POWER = str(REPO_ROOT / "scripts" / "run_workload_with_power.py")
PY = sys.executable

# --- Workload tiers --------------------------------------------------------

CONSUMER_WORKLOADS = [
    # 1-9: single-GPU LLMs (all fit in ≤16 GB FP16 or via 4-bit), all open-license
    "llm_infer_phi3_mini",
    "llm_infer_qwen25_1p5b",
    "llm_infer_qwen25_3b",
    "llm_infer_smollm2_1p7b",
    "llm_infer_mistral_7b_q4",
    "llm_infer_tinyllama_bs8",
    "llm_infer_tinyllama_seq1024",
    "llm_infer_qwen25_3b_bs4",
    "llm_infer_qwen25_1p5b_streaming",
    # 10-11: non-LLM inference
    "llm_infer_sdxl",
    "llm_infer_clip_embed_batched",
]

# Subset of consumer workloads to replicate on datacenter for transfer checks
DATACENTER_SUBSET_CONSUMER = [
    "llm_infer_phi3_mini",
    "llm_infer_qwen25_3b",
    "llm_infer_mistral_7b_q4",
    "llm_infer_tinyllama_bs8",
    "llm_infer_qwen25_3b_bs4",
    "llm_infer_sdxl",
]

DATACENTER_LARGE_WORKLOADS = [
    # 12-14: single-GPU big models (>80 GB memory on H200/B200), all open-license
    "llm_infer_qwen25_72b_fp16",
    "llm_infer_mixtral_8x7b",
    "llm_infer_qwen25_72b_fp8",
    # 15: 2-GPU TP — standard datacenter workload
    "llm_infer_qwen25_72b_tp2",
]

DATACENTER_STRETCH_WORKLOADS = [
    # 16: 8-GPU TP — run only if 8×H200 is already allocated
    "llm_infer_deepseek_v3_tp8",
]


def workloads_for_tier(tier: str, include_stretch: bool) -> list[str]:
    if tier == "consumer":
        return CONSUMER_WORKLOADS
    if tier == "datacenter":
        full = DATACENTER_SUBSET_CONSUMER + DATACENTER_LARGE_WORKLOADS
        if include_stretch:
            full += DATACENTER_STRETCH_WORKLOADS
        return full
    raise ValueError(f"unknown tier: {tier}")


# --- Progress tracking ------------------------------------------------------

def load_progress() -> dict:
    if PROGRESS_FILE.exists():
        return json.loads(PROGRESS_FILE.read_text())
    return {"completed": [], "failed": [], "started": None}


def save_progress(progress: dict) -> None:
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE.write_text(json.dumps(progress, indent=2))


# --- Run one ---------------------------------------------------------------

def run_one(label: str, rep: int, use_tapo: bool, no_dcgm: bool) -> int:
    """Spawn run_workload[_with_power].py for a single (label, rep) pair."""
    script = RUN_WORKLOAD_POWER if use_tapo else RUN_WORKLOAD
    cmd = [PY, script, "--workload", label]
    if use_tapo:
        cmd.append("--tapo")
    if no_dcgm:
        cmd.append("--no-dcgm")

    print(f"\n{'='*60}")
    print(f"  Running: {label} (rep {rep})")
    print(f"  Command: {' '.join(cmd)}")
    print(f"  Time:    {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}\n")

    return subprocess.run(cmd, cwd=str(REPO_ROOT)).returncode


# --- Main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="Round 5b inference collection orchestrator")
    ap.add_argument("--tier", choices=["consumer", "datacenter"], default="consumer",
                    help="GPU tier (selects default workload list)")
    ap.add_argument("--workloads", nargs="+",
                    help="Override tier selection with explicit workload labels")
    ap.add_argument("--reps", type=int, default=3,
                    help="Repetitions per workload (default: 3)")
    ap.add_argument("--include-stretch", action="store_true",
                    help="Include 405B 8-GPU stretch workload (datacenter only)")
    ap.add_argument("--tapo", action="store_true", help="Enable Tapo P115 wall power")
    ap.add_argument("--no-dcgm", action="store_true", default=True,
                    help="Disable DCGM (default True)")
    ap.add_argument("--dry-run", action="store_true", help="Preview and exit")
    ap.add_argument("--reset", action="store_true", help="Clear progress file")
    args = ap.parse_args()

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
        print("Progress cleared.")
        return

    if args.workloads:
        workloads = list(args.workloads)
    else:
        workloads = workloads_for_tier(args.tier, args.include_stretch)

    # Expand to (label, rep) pairs
    jobs = [(w, r) for w in workloads for r in range(1, args.reps + 1)]

    progress = load_progress()
    done = {tuple(x) for x in progress["completed"]}
    remaining = [j for j in jobs if list(j) not in [list(x) for x in progress["completed"]]]

    if args.dry_run:
        print(f"Tier: {args.tier}  Reps: {args.reps}  Stretch: {args.include_stretch}")
        print(f"Workloads ({len(workloads)}):")
        for w in workloads:
            reps_done = sum(1 for (ww, _) in done if ww == w)
            print(f"  [{reps_done}/{args.reps} done] {w}")
        print(f"\n{len(remaining)} (label, rep) jobs remaining.")
        # Rough estimate: consumer ~5.5 min/run, datacenter big ~15 min/run.
        est_min = sum(15 if w.startswith("llm_infer_llama33") or
                              w.startswith("llm_infer_mixtral") or
                              w.startswith("llm_infer_qwen25_72b") or
                              w.startswith("llm_infer_llama31") else 5.5
                      for (w, _) in remaining)
        print(f"Estimated wall time: ~{est_min:.0f} minutes.")
        return

    if not remaining:
        print("All jobs already completed!")
        print(f"Progress file: {PROGRESS_FILE}")
        return

    print(f"Round 5b inference collection")
    print(f"  Tier:       {args.tier}")
    print(f"  Workloads:  {len(workloads)}")
    print(f"  Reps:       {args.reps}")
    print(f"  Stretch:    {args.include_stretch}")
    print(f"  Remaining:  {len(remaining)} (label,rep) jobs")
    print(f"  Tapo power: {'ON' if args.tapo else 'off'}")
    print()

    sweep_start = time.time()
    for i, (label, rep) in enumerate(remaining):
        print(f"\n[{i+1}/{len(remaining)}] {label} (rep {rep})")
        progress["started"] = [label, rep]
        save_progress(progress)

        rc = run_one(label, rep, use_tapo=args.tapo, no_dcgm=args.no_dcgm)

        if rc == 0:
            progress["completed"].append([label, rep])
            progress["started"] = None
            save_progress(progress)
            print(f"  {label} rep {rep} completed.")
        else:
            progress["failed"].append({
                "label": label, "rep": rep, "exit_code": rc,
                "time": datetime.now().isoformat(),
            })
            progress["started"] = None
            save_progress(progress)
            print(f"  WARNING: {label} rep {rep} failed (rc={rc}); continuing.")

    elapsed = time.time() - sweep_start
    print(f"\n{'='*60}")
    print("Round 5b inference collection done")
    print(f"  Wall time: {timedelta(seconds=int(elapsed))}")
    print(f"  Completed: {len(progress['completed'])}")
    print(f"  Failed:    {len(progress['failed'])}")
    print(f"  Data in:   data/")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
