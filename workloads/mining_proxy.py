#!/usr/bin/env python3
"""Crypto mining proxy workload for SPAR telemetry collection.

Simulates Ethash (Ethereum-style) mining patterns using PyTorch CUDA kernels.
Ethash is memory-hard: large random-access reads from a ~1GB DAG, then
a compute pass (Keccak-like hash). This proxy mimics those patterns:
  - Large working set (DAG-like tensor, 1+ GB)
  - Random gather reads into the DAG
  - XOR-heavy compute pass

This is used when T-Rex or actual mining software is unavailable/blocked.

Usage:
    python mining_proxy.py --duration 600
    python mining_proxy.py --dag-size-mb 1024 --duration 600
"""

import argparse
import time

import torch


def run_ethash_proxy(duration: int, device: torch.device, dag_size_mb: int):
    """Simulate Ethash memory-hard hashing pattern.

    Ethash characteristics:
    - ~1GB DAG (directed acyclic graph) in memory
    - 64 sequential random reads per hash (memory-bound)
    - XOR mixing between reads (light compute)
    - Target: ~30 MH/s on A100 (memory bandwidth bound)
    """
    dag_elements = (dag_size_mb * 1024 * 1024) // 64  # 64 bytes per DAG element
    print(f"  Ethash proxy: DAG={dag_size_mb}MB ({dag_elements:,} elements), {duration}s")

    # Allocate DAG as uint32 tensor (4 bytes * 16 = 64 bytes per element)
    dag = torch.randint(0, 2**31, (dag_elements, 16), dtype=torch.int32, device=device)

    # Nonce buffer for mixing (batch of 1024 hashes)
    batch = 1024
    seed_size = 8  # 8 uint32 per seed
    seeds = torch.randint(0, 2**31, (batch, seed_size), dtype=torch.int32, device=device)

    wall_start = time.time()
    iterations = 0

    while time.time() - wall_start < duration:
        # Phase 1: Random gather from DAG (64 accesses per hash, memory-bound)
        mix = seeds.clone()
        for _ in range(64):
            # Generate pseudo-random DAG indices from current mix state
            indices = (mix[:, 0].abs() % dag_elements).long()
            # Gather DAG elements and XOR into mix
            dag_row = dag[indices]  # (batch, 16) random memory read
            mix = mix ^ dag_row[:, :seed_size]

        # Phase 2: Final hash mixing (compute pass)
        result = mix
        for _ in range(4):
            result = result ^ (result >> 1)
            result = result * 0x5bd1e995  # Murmur-like hash step
            result = result ^ (result >> 16)

        # Prevent dead code elimination
        _ = result.sum()
        torch.cuda.synchronize()

        iterations += 1
        if iterations % 20 == 0:
            elapsed = time.time() - wall_start
            hashrate = (iterations * batch) / elapsed / 1e6
            print(f"  Iteration {iterations}: ~{hashrate:.2f} MH/s, {elapsed:.1f}s elapsed")

        # Refresh seeds for next iteration
        seeds = torch.randint(0, 2**31, (batch, seed_size), dtype=torch.int32, device=device)

    wall_time = time.time() - wall_start
    total_hashes = iterations * batch
    hashrate = total_hashes / wall_time / 1e6
    print(f"Ethash proxy complete. Iterations: {iterations}, ~{hashrate:.2f} MH/s, {wall_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Crypto mining proxy workload")
    parser.add_argument("--duration", type=int, default=600,
                        help="Run duration in seconds (default: 600)")
    parser.add_argument("--dag-size-mb", type=int, default=1024,
                        help="DAG size in MB (default: 1024)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Crypto mining proxy workload (Ethash-like)")
    print(f"  Device: {device}")

    run_ethash_proxy(args.duration, device, args.dag_size_mb)


if __name__ == "__main__":
    main()
