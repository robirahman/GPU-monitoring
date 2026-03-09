#!/usr/bin/env python3
"""Scientific HPC workloads for SPAR telemetry collection.

Implements two GPU workloads:
  - cufft: Repeated large 3D FFT benchmarks using torch.fft (memory + compute bandwidth)
  - nbody: Gravitational N-body simulation in PyTorch (compute-bound)

Usage:
    python scientific_hpc.py --workload cufft --duration 600
    python scientific_hpc.py --workload nbody --duration 600 --n-bodies 8192
"""

import argparse
import time

import torch


def run_cufft(duration: int, device: torch.device):
    """Repeated large 3D FFT benchmark. Primarily tests memory + cache bandwidth."""
    # Large 3D tensor: 512x512x64 complex float
    size = (512, 512, 64)
    print(f"  cuFFT benchmark: {size} complex float32 tensors, {duration}s")

    x = torch.randn(*size, dtype=torch.complex64, device=device)
    wall_start = time.time()
    iterations = 0

    while time.time() - wall_start < duration:
        # Forward and inverse FFT to stress memory bandwidth
        y = torch.fft.fftn(x)
        z = torch.fft.ifftn(y)
        # Prevent optimizer from eliminating the ops
        x = z.real.to(dtype=torch.complex64) + x.imag * 1j
        torch.cuda.synchronize()
        iterations += 1
        if iterations % 50 == 0:
            elapsed = time.time() - wall_start
            print(f"  Iteration {iterations}: {elapsed:.1f}s elapsed")

    wall_time = time.time() - wall_start
    print(f"cuFFT complete. Iterations: {iterations}, Wall time: {wall_time:.1f}s")


def run_nbody(duration: int, device: torch.device, n_bodies: int):
    """Direct-summation gravitational N-body simulation. Compute-bound O(N^2)."""
    print(f"  N-body simulation: {n_bodies} bodies, {duration}s")
    G = 6.674e-11
    dt = 1.0

    # Initialize random positions and velocities
    pos = torch.randn(n_bodies, 3, device=device) * 1e11
    vel = torch.randn(n_bodies, 3, device=device) * 1e3
    mass = torch.rand(n_bodies, device=device) * 1e30 + 1e29

    wall_start = time.time()
    step = 0

    while time.time() - wall_start < duration:
        # Compute pairwise displacement: (N, 1, 3) - (1, N, 3) -> (N, N, 3)
        delta = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3)
        dist_sq = (delta ** 2).sum(dim=-1) + 1e10   # (N, N) softened
        dist_cube = dist_sq ** 1.5                   # (N, N)

        # Acceleration: sum over all other bodies
        # F_ij = G * m_j * delta_ij / dist^3
        accel = G * (mass.unsqueeze(0).unsqueeze(-1) * delta / dist_cube.unsqueeze(-1)).sum(dim=1)

        vel = vel + accel * dt
        pos = pos + vel * dt
        torch.cuda.synchronize()
        step += 1

        if step % 20 == 0:
            elapsed = time.time() - wall_start
            print(f"  Step {step}: {elapsed:.1f}s elapsed")

    wall_time = time.time() - wall_start
    print(f"N-body complete. Steps: {step}, Wall time: {wall_time:.1f}s")


def main():
    parser = argparse.ArgumentParser(description="Scientific HPC GPU workload")
    parser.add_argument("--workload", type=str, default="cufft",
                        choices=["cufft", "nbody"],
                        help="Workload type (default: cufft)")
    parser.add_argument("--duration", type=int, default=600,
                        help="Run duration in seconds (default: 600)")
    parser.add_argument("--n-bodies", type=int, default=4096,
                        help="Number of bodies for N-body (default: 4096)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Scientific HPC workload: {args.workload}")
    print(f"  Device: {device}")

    if args.workload == "cufft":
        run_cufft(args.duration, device)
    else:
        run_nbody(args.duration, device, args.n_bodies)


if __name__ == "__main__":
    main()
