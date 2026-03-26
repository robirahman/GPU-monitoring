#!/usr/bin/env python3
"""GPU-accelerated gravitational N-body simulation in PyTorch.

Pure FP32 compute workload — high SM utilization, no tensor cores, distinct
from ML training. Uses direct O(N^2) pairwise force computation.

Usage:
    python nbody_sim.py --duration 300 --particles 16384
"""

import argparse
import signal
import time

import torch


def nbody_step(pos, vel, mass, dt=0.001, softening=0.01):
    """Compute one N-body timestep with direct pairwise forces."""
    # pos: (N, 3), vel: (N, 3), mass: (N,)
    N = pos.shape[0]
    # Pairwise displacement: (N, N, 3)
    diff = pos.unsqueeze(0) - pos.unsqueeze(1)  # (N, N, 3)
    # Pairwise distances: (N, N)
    dist_sq = (diff ** 2).sum(dim=-1) + softening ** 2
    inv_dist3 = dist_sq ** (-1.5)
    # Force per unit mass: (N, 3)
    # F_i = sum_j G * m_j * (r_j - r_i) / |r_j - r_i|^3
    # Using G=1 for simplicity
    acc = (inv_dist3.unsqueeze(-1) * diff * mass.unsqueeze(0).unsqueeze(-1)).sum(dim=1)
    # Leapfrog integration
    vel = vel + acc * dt
    pos = pos + vel * dt
    return pos, vel


def main():
    parser = argparse.ArgumentParser(description="N-body gravitational simulation")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--particles", type=int, default=16384, help="Number of particles")
    parser.add_argument("--dt", type=float, default=0.001, help="Timestep")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"N-body simulation: {args.particles} particles")
    print(f"  Device: {device}, Duration: {args.duration}s, dt: {args.dt}")

    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    # Initialize random particle distribution (Plummer sphere approximation)
    pos = torch.randn(args.particles, 3, device=device, dtype=torch.float32)
    vel = torch.randn(args.particles, 3, device=device, dtype=torch.float32) * 0.1
    mass = torch.ones(args.particles, device=device, dtype=torch.float32) / args.particles

    wall_start = time.time()
    steps = 0

    while (time.time() - wall_start) < args.duration and not stopped:
        pos, vel = nbody_step(pos, vel, mass, dt=args.dt)
        torch.cuda.synchronize()
        steps += 1
        if steps % 100 == 0:
            elapsed = time.time() - wall_start
            print(f"  Step {steps}, {elapsed:.0f}s elapsed")

    wall_time = time.time() - wall_start
    print(
        f"N-body simulation complete. {steps} steps in {wall_time:.1f}s "
        f"({steps / wall_time:.1f} steps/s)"
    )


if __name__ == "__main__":
    main()
