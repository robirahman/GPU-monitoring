#!/usr/bin/env python3
"""Automated datacenter experiment collection — runs the full experiment plan.

Designed to maximize useful data per GPU-hour on expensive multi-GPU instances.
Crash-resilient with JSON progress tracking.

Phase 1: Single-GPU adversarial strategies (parallel across 8 GPUs)
Phase 2: DDP baseline training workloads (all 8 GPUs)
Phase 3: DDP adversarial strategies (all 8 GPUs)

Usage:
    python scripts/run_datacenter_collection.py                    # full plan
    python scripts/run_datacenter_collection.py --phase 1          # single-GPU only
    python scripts/run_datacenter_collection.py --phase 2          # DDP baseline only
    python scripts/run_datacenter_collection.py --phase 3          # DDP adversarial only
    python scripts/run_datacenter_collection.py --dry-run           # preview plan
    python scripts/run_datacenter_collection.py --num-gpus 2       # 2-GPU test
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
PY = sys.executable
PROGRESS_FILE = REPO_ROOT / "data" / "datacenter_collection_progress.json"

# ── Phase 1: Single-GPU adversarial strategies ───────────────────────────────
# Run in parallel across GPUs. 8 strategies per batch, ~300s each.

SINGLE_GPU_ADVERSARIAL = [
    # Strategies missing from datacenter (E-L)
    "adversarial_E_memory_minimal",
    "adversarial_F_interleave_30",
    "adversarial_F_interleave_50",
    "adversarial_G_clock_throttled",
    "adversarial_H_mimicry_mining",
    "adversarial_H_mimicry_cufft",
    "adversarial_I_stochastic_30",
    "adversarial_J_pid_75",
    # Round 3 (strongest attacks)
    "adversarial_K_online_learning",
    "adversarial_L_diluted_2",
    "adversarial_L_diluted_5",
    "adversarial_L_diluted_10",
    "adversarial_L_diluted_20",
]

# Also run clean baselines if this is B100 (no existing data)
BASELINE_WORKLOADS = [
    "idle",
    "pytorch_resnet_cifar10",
    "pytorch_resnet_cifar10_amp",
    "pytorch_mlp_cifar10",
    "gpt2_wikitext2",
    "gpt2_wikitext2_amp",
    "bert_sst2",
    "bert_sst2_amp",
    "resnet50_inference",
    "cufft_benchmark",
    "nbody_sim",
    "mining_ethash_proxy",
    "rendering_proxy",
]

# ── Phase 2: DDP baseline workloads ─────────────────────────────────────────

DDP_BASELINES = [
    {
        "label": "ddp_resnet18_cifar10",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "pytorch_training_ddp.py"),
                "--model", "resnet18", "--duration", "300"],
        "timeout": 360,
    },
    {
        "label": "ddp_resnet50_cifar10",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "pytorch_training_ddp.py"),
                "--model", "resnet50", "--duration", "300"],
        "timeout": 360,
    },
    {
        "label": "ddp_resnet18_cifar10_amp",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "pytorch_training_ddp.py"),
                "--model", "resnet18", "--duration", "300", "--amp"],
        "timeout": 360,
    },
]

# ── Phase 3: DDP adversarial strategies ──────────────────────────────────────

DDP_ADVERSARIAL = [
    {
        "label": "adversarial_K_ddp",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "adversarial_ddp.py"),
                "--strategy", "K", "--duration", "300"],
        "timeout": 360,
    },
    {
        "label": "adversarial_L_ddp_5",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "adversarial_ddp.py"),
                "--strategy", "L", "--dilution", "5", "--duration", "300"],
        "timeout": 360,
    },
    {
        "label": "adversarial_L_ddp_10",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "adversarial_ddp.py"),
                "--strategy", "L", "--dilution", "10", "--duration", "300"],
        "timeout": 360,
    },
    {
        "label": "adversarial_L_ddp_stagger_5",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "adversarial_ddp.py"),
                "--strategy", "L-stagger", "--dilution", "5", "--duration", "300"],
        "timeout": 360,
    },
    {
        "label": "adversarial_K_ddp_accum_4",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "adversarial_ddp.py"),
                "--strategy", "K-accum", "--accum", "4", "--duration", "300"],
        "timeout": 360,
    },
    {
        "label": "adversarial_K_ddp_accum_8",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "adversarial_ddp.py"),
                "--strategy", "K-accum", "--accum", "8", "--duration", "300"],
        "timeout": 360,
    },
]

# ── Phase 4: FSDP NVLink stress workloads ───────────────────────────────────
# Model-sharded training via FSDP FULL_SHARD. Heavy AllGather/ReduceScatter
# on every forward/backward step — maximizes NVLink interconnect traffic.

FSDP_NVLINK = [
    {
        "label": "fsdp_7b_nvlink",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "fsdp_training.py"),
                "--size", "7b", "--duration", "300"],
        "timeout": 420,
    },
    {
        "label": "fsdp_13b_nvlink",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "fsdp_training.py"),
                "--size", "13b", "--duration", "300"],
        "timeout": 420,
    },
    {
        "label": "fsdp_30b_nvlink",
        "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
                str(REPO_ROOT / "workloads" / "fsdp_training.py"),
                "--size", "30b", "--duration", "600",
                "--batch-size", "1", "--seq-len", "512"],
        "timeout": 720,
    },
    # 70B does not fit on 8x A100-80GB even with SGD (no momentum).
    # Needs H200 or instances with >80GB/GPU. Skipped here.
    # {
    #     "label": "fsdp_70b_nvlink",
    #     "cmd": ["torchrun", "--nproc_per_node={num_gpus}",
    #             str(REPO_ROOT / "workloads" / "fsdp_training.py"),
    #             "--size", "70b", "--duration", "600",
    #             "--batch-size", "1", "--seq-len", "256"],
    #     "timeout": 780,
    # },
]


def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE) as f:
            return json.load(f)
    return {"completed": [], "failed": [], "last_updated": None}


def save_progress(progress):
    PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)
    progress["last_updated"] = datetime.now().isoformat()
    with open(PROGRESS_FILE, "w") as f:
        json.dump(progress, f, indent=2)


def run_parallel_single_gpu(labels, num_gpus, progress, timeout=360, no_dcgm=False):
    """Run single-GPU workloads in parallel across GPUs, 8 at a time."""
    remaining = [l for l in labels if l not in progress["completed"]]
    if not remaining:
        print(f"  All {len(labels)} single-GPU workloads already completed.")
        return

    print(f"\n  Running {len(remaining)} single-GPU workloads across {num_gpus} GPUs")

    for batch_start in range(0, len(remaining), num_gpus):
        batch = remaining[batch_start:batch_start + num_gpus]
        print(f"\n  Batch: {batch}")

        cmd = [
            PY, str(REPO_ROOT / "scripts" / "run_workload_multi_gpu.py"),
            "--workload", batch[0],  # Primary label
            "--parallel",
            "--num-gpus", str(min(len(batch), num_gpus)),
            "--timeout", str(timeout),
        ]
        if no_dcgm:
            cmd.append("--no-dcgm")

        # For multiple different workloads in one batch, we need to run them
        # individually in parallel (the multi-gpu script handles one label)
        # So instead, launch them as separate processes
        import threading
        results = {}

        def run_one_gpu(gpu_idx, label):
            gpu_cmd = [
                PY, str(REPO_ROOT / "scripts" / "run_workload.py"),
                "--workload", label,
                "--gpu-index", str(gpu_idx),
                "--timeout", str(timeout),
            ]
            if no_dcgm:
                gpu_cmd.append("--no-dcgm")
            try:
                result = subprocess.run(gpu_cmd, cwd=str(REPO_ROOT),
                                        timeout=timeout + 60)
                results[label] = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
            except subprocess.TimeoutExpired:
                results[label] = "TIMEOUT"
            except Exception as e:
                results[label] = f"ERROR: {e}"

        threads = []
        for i, label in enumerate(batch):
            t = threading.Thread(target=run_one_gpu, args=(i, label))
            threads.append(t)
            t.start()

        for t in threads:
            t.join()

        for label, status in results.items():
            if status == "OK":
                progress["completed"].append(label)
            else:
                progress["failed"].append({"label": label, "status": status,
                                           "time": datetime.now().isoformat()})
            print(f"    {label}: {status}")

        save_progress(progress)
        time.sleep(10)  # Brief cooldown between batches


def run_multi_gpu_workload(entry, num_gpus, progress, no_dcgm=False):
    """Run a single multi-GPU (DDP) workload with per-GPU telemetry."""
    label = entry["label"]
    if label in progress["completed"]:
        print(f"  SKIP: {label} (already completed)")
        return

    # Substitute {num_gpus} in command
    cmd = [c.format(num_gpus=num_gpus) for c in entry["cmd"]]
    timeout = entry["timeout"]

    print(f"\n  Running: {label} ({num_gpus} GPUs)")
    print(f"    cmd: {' '.join(cmd)}")

    multi_cmd = [
        PY, str(REPO_ROOT / "scripts" / "run_workload_multi_gpu.py"),
        "--workload", label,
        "--num-gpus", str(num_gpus),
        "--timeout", str(timeout),
    ]
    if no_dcgm:
        multi_cmd.append("--no-dcgm")
    multi_cmd += ["--"] + cmd

    try:
        result = subprocess.run(multi_cmd, cwd=str(REPO_ROOT),
                                timeout=timeout + 120)
        status = "OK" if result.returncode == 0 else f"FAIL({result.returncode})"
    except subprocess.TimeoutExpired:
        status = "TIMEOUT"
    except Exception as e:
        status = f"ERROR: {e}"

    if status == "OK":
        progress["completed"].append(label)
    else:
        progress["failed"].append({"label": label, "status": status,
                                   "time": datetime.now().isoformat()})

    print(f"    {label}: {status}")
    save_progress(progress)
    time.sleep(10)


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Datacenter experiment collection")
    parser.add_argument("--phase", type=int, default=None,
                        help="Run specific phase only (1, 2, or 3)")
    parser.add_argument("--num-gpus", type=int, default=8)
    parser.add_argument("--no-dcgm", action="store_true")
    parser.add_argument("--include-baselines", action="store_true",
                        help="Also run clean baseline workloads (for new GPU types)")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--reset", action="store_true")
    args = parser.parse_args()

    phases = [args.phase] if args.phase else [1, 2, 3, 4]

    if args.dry_run:
        print(f"Datacenter collection plan ({args.num_gpus} GPUs)\n")
        if 1 in phases:
            labels = SINGLE_GPU_ADVERSARIAL[:]
            if args.include_baselines:
                labels += BASELINE_WORKLOADS
            print(f"Phase 1: {len(labels)} single-GPU workloads (parallel)")
            for i, l in enumerate(labels, 1):
                print(f"  {i:2d}. {l}")
            batches = (len(labels) + args.num_gpus - 1) // args.num_gpus
            print(f"  -> {batches} batches x ~5 min = ~{batches * 5} min\n")
        if 2 in phases:
            print(f"Phase 2: {len(DDP_BASELINES)} DDP baseline workloads")
            for entry in DDP_BASELINES:
                print(f"  - {entry['label']} ({entry['timeout']}s)")
            print(f"  -> ~{len(DDP_BASELINES) * 6} min\n")
        if 3 in phases:
            print(f"Phase 3: {len(DDP_ADVERSARIAL)} DDP adversarial workloads")
            for entry in DDP_ADVERSARIAL:
                print(f"  - {entry['label']} ({entry['timeout']}s)")
            print(f"  -> ~{len(DDP_ADVERSARIAL) * 6} min\n")
        if 4 in phases:
            print(f"Phase 4: {len(FSDP_NVLINK)} FSDP NVLink stress workloads")
            for entry in FSDP_NVLINK:
                print(f"  - {entry['label']} ({entry['timeout']}s)")
            print(f"  -> ~{sum(e['timeout'] for e in FSDP_NVLINK) // 60} min\n")
        return

    if args.reset:
        if PROGRESS_FILE.exists():
            PROGRESS_FILE.unlink()
            print(f"Cleared: {PROGRESS_FILE}")
        return

    progress = load_progress()
    start_time = time.time()

    print(f"{'='*60}")
    print(f"SPAR Datacenter Collection")
    print(f"{'='*60}")
    print(f"  GPUs:     {args.num_gpus}")
    print(f"  Phases:   {phases}")
    print(f"  DCGM:     {'disabled' if args.no_dcgm else 'enabled'}")
    print(f"  Started:  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"{'='*60}")

    try:
        if 1 in phases:
            print(f"\n{'='*60}")
            print(f"PHASE 1: Single-GPU Adversarial")
            print(f"{'='*60}")
            labels = SINGLE_GPU_ADVERSARIAL[:]
            if args.include_baselines:
                labels += BASELINE_WORKLOADS
            run_parallel_single_gpu(labels, args.num_gpus, progress,
                                    timeout=360, no_dcgm=args.no_dcgm)

        if 2 in phases:
            print(f"\n{'='*60}")
            print(f"PHASE 2: DDP Baseline Training")
            print(f"{'='*60}")
            for entry in DDP_BASELINES:
                run_multi_gpu_workload(entry, args.num_gpus, progress,
                                       no_dcgm=args.no_dcgm)

        if 3 in phases:
            print(f"\n{'='*60}")
            print(f"PHASE 3: DDP Adversarial Strategies")
            print(f"{'='*60}")
            for entry in DDP_ADVERSARIAL:
                run_multi_gpu_workload(entry, args.num_gpus, progress,
                                       no_dcgm=args.no_dcgm)

        if 4 in phases:
            print(f"\n{'='*60}")
            print(f"PHASE 4: FSDP NVLink Stress")
            print(f"{'='*60}")
            for entry in FSDP_NVLINK:
                run_multi_gpu_workload(entry, args.num_gpus, progress,
                                       no_dcgm=args.no_dcgm)

    except KeyboardInterrupt:
        print("\n\nInterrupted. Progress saved.")
        save_progress(progress)
        sys.exit(1)

    elapsed = time.time() - start_time
    print(f"\n{'='*60}")
    print(f"COLLECTION COMPLETE")
    print(f"{'='*60}")
    print(f"  Time:      {timedelta(seconds=int(elapsed))}")
    print(f"  Completed: {len(progress['completed'])}")
    print(f"  Failed:    {len(progress['failed'])}")
    if progress["failed"]:
        for f in progress["failed"]:
            print(f"    - {f['label']}: {f['status']}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
