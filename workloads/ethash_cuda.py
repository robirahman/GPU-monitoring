#!/usr/bin/env python3
"""Custom Ethash-like CUDA kernel for crypto mining telemetry signature.

Mimics the memory-hard hashing pattern of Ethash without being actual mining
software — won't trigger cloud provider mining detection. Uses Keccak-like
hash mixing with a large DAG array to produce the characteristic memory-bound,
low-tensor-core pattern of crypto mining.

Usage:
    python ethash_cuda.py --duration 300 --dag-size 1024
"""

import argparse
import signal
import time

import torch


def ethash_round(dag, header, nonce_batch, mix_rounds=64):
    """Simulate one Ethash-like mixing round.

    Ethash is memory-hard: it reads pseudorandom locations from a large DAG.
    We simulate this with gather operations on a large GPU tensor.
    """
    batch_size = nonce_batch.shape[0]
    dag_size = dag.shape[0]

    # Initial mix: hash(header, nonce) — simulated with random init
    mix = torch.bitwise_xor(header.expand(batch_size, -1), nonce_batch)

    for _ in range(mix_rounds):
        # Pseudorandom DAG index from current mix state
        indices = (mix[:, 0].abs() % dag_size).long()
        # Fetch from DAG (memory-hard access pattern)
        dag_data = dag[indices]
        # Mix: XOR + bit rotation simulation
        mix = torch.bitwise_xor(mix, dag_data)
        # FNV-like mixing (multiply and XOR)
        mix = mix * 0x01000193 ^ (mix >> 8)

    return mix


def main():
    parser = argparse.ArgumentParser(description="Ethash-like CUDA workload")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument(
        "--dag-size", type=int, default=1024,
        help="DAG size in MB (real Ethash uses ~4GB)",
    )
    parser.add_argument("--batch-size", type=int, default=65536, help="Nonces per batch")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dag_elements = (args.dag_size * 1024 * 1024) // 4  # int32 elements
    print(f"Ethash-like CUDA workload")
    print(f"  Device: {device}, DAG: {args.dag_size} MB, Batch: {args.batch_size}")
    print(f"  Duration: {args.duration}s")

    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    # Allocate DAG (large memory region for random reads)
    print(f"  Allocating DAG ({dag_elements} int32 elements)...")
    dag = torch.randint(
        -(2**31), 2**31 - 1, (dag_elements, 16),
        dtype=torch.int32, device=device,
    )

    # Simulated block header
    header = torch.randint(
        -(2**31), 2**31 - 1, (1, 16), dtype=torch.int32, device=device
    )

    wall_start = time.time()
    total_hashes = 0
    nonce_offset = 0

    while (time.time() - wall_start) < args.duration and not stopped:
        # Generate nonce batch
        nonces = torch.arange(
            nonce_offset, nonce_offset + args.batch_size,
            device=device, dtype=torch.int32,
        ).unsqueeze(1).expand(-1, 16)

        result = ethash_round(dag, header, nonces)
        torch.cuda.synchronize()

        total_hashes += args.batch_size
        nonce_offset += args.batch_size

        if total_hashes % (args.batch_size * 10) == 0:
            elapsed = time.time() - wall_start
            hashrate = total_hashes / elapsed
            print(f"  {total_hashes} hashes, {elapsed:.0f}s, {hashrate:.0f} H/s")

    wall_time = time.time() - wall_start
    print(
        f"Ethash workload complete. {total_hashes} hashes in {wall_time:.1f}s "
        f"({total_hashes / wall_time:.0f} H/s)"
    )


if __name__ == "__main__":
    main()
