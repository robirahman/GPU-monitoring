#!/usr/bin/env python3
"""GROMACS molecular dynamics simulation workload.

Runs the ADH benchmark (alcohol dehydrogenase) or a lysozyme simulation.
Scientific HPC signature: high GPU utilization, no tensor cores, heavy
FP32 compute and memory bandwidth.

Usage:
    python gromacs_sim.py --duration 300
    python gromacs_sim.py --benchmark adh --duration 300
"""

import argparse
import os
import signal
import subprocess
import sys
import time
import urllib.request


ADH_BENCHMARK_URL = "https://ftp.gromacs.org/pub/benchmarks/ADH_bench_systems.tar.gz"
ADH_DIR = "/tmp/gromacs_benchmark"


def check_gromacs():
    """Check that GROMACS is available with GPU support."""
    try:
        result = subprocess.run(
            ["gmx", "--version"],
            capture_output=True, text=True, timeout=10,
        )
        version_line = [l for l in result.stdout.splitlines() if "GROMACS version" in l]
        if version_line:
            print(f"  {version_line[0].strip()}")
        # Check for GPU support
        if "CUDA" in result.stdout or "GPU" in result.stdout:
            print("  GPU support: detected")
        else:
            print("  WARNING: GPU support not detected in GROMACS build")
    except FileNotFoundError:
        print("ERROR: gmx not found. Install with: apt-get install gromacs")
        sys.exit(1)


def download_adh_benchmark():
    """Download the ADH benchmark if not present."""
    tpr_path = os.path.join(ADH_DIR, "adh_cubic", "topol.tpr")
    if os.path.exists(tpr_path):
        print(f"  ADH benchmark already downloaded")
        return tpr_path

    os.makedirs(ADH_DIR, exist_ok=True)
    tar_path = os.path.join(ADH_DIR, "adh_bench.tar.gz")

    if not os.path.exists(tar_path):
        print(f"  Downloading ADH benchmark...")
        urllib.request.urlretrieve(ADH_BENCHMARK_URL, tar_path)

    print(f"  Extracting...")
    subprocess.run(
        ["tar", "xzf", tar_path, "-C", ADH_DIR],
        capture_output=True, check=True,
    )

    # Find the .tpr file
    for root, dirs, files in os.walk(ADH_DIR):
        for f in files:
            if f.endswith(".tpr"):
                tpr = os.path.join(root, f)
                print(f"  Found topology: {tpr}")
                return tpr

    print("ERROR: Could not find .tpr file in ADH benchmark")
    sys.exit(1)


def run_gromacs(tpr_path, duration, nsteps=None):
    """Run GROMACS MD simulation."""
    # If nsteps not specified, use a large number and rely on timeout
    if nsteps is None:
        nsteps = 1000000  # Will be killed by duration timeout

    cmd = [
        "gmx", "mdrun",
        "-s", tpr_path,
        "-nsteps", str(nsteps),
        "-nb", "gpu",           # Non-bonded on GPU
        "-pme", "gpu",          # PME on GPU
        "-bonded", "gpu",       # Bonded on GPU
        "-update", "gpu",       # Update on GPU
        "-ntomp", "4",          # OpenMP threads
        "-pin", "on",
        "-noconfout",           # Skip writing final coordinates
    ]

    return subprocess.Popen(
        cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        cwd=os.path.dirname(tpr_path),
    )


def main():
    parser = argparse.ArgumentParser(description="GROMACS MD simulation workload")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument(
        "--benchmark", type=str, default="adh",
        choices=["adh"], help="Benchmark system",
    )
    args = parser.parse_args()

    print(f"GROMACS MD simulation workload")
    print(f"  Benchmark: {args.benchmark}, Duration: {args.duration}s")

    check_gromacs()
    tpr_path = download_adh_benchmark()

    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    print(f"  Starting MD simulation...")
    wall_start = time.time()
    proc = run_gromacs(tpr_path, args.duration)

    try:
        # Wait for duration or process completion
        while (time.time() - wall_start) < args.duration and not stopped:
            if proc.poll() is not None:
                break
            time.sleep(5)
            elapsed = time.time() - wall_start
            print(f"  Running... {elapsed:.0f}s / {args.duration}s")
    finally:
        if proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()

    wall_time = time.time() - wall_start
    stdout = proc.stdout.read().decode(errors="replace") if proc.stdout else ""
    stderr = proc.stderr.read().decode(errors="replace") if proc.stderr else ""

    # Try to extract performance from GROMACS output
    for line in (stdout + stderr).splitlines():
        if "ns/day" in line.lower() or "Performance" in line:
            print(f"  {line.strip()}")

    print(f"GROMACS simulation complete. Wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
