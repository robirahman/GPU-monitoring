"""Scientific HPC workloads — DirectML port of scientific_hpc.py.

Two GPU workloads:
  - cufft: Repeated 3D FFTs via torch.fft (memory-bandwidth bound)
  - nbody: Direct-summation gravitational N-body (compute bound, O(N^2))

Both are implemented in pure-PyTorch tensor ops, which DirectML supports.
torch.fft on torch-directml falls back to a slower path on some shapes;
the cufft workload still produces a distinct memory-bandwidth telemetry
signature suitable for non-ML labelling.

Usage:
    python scientific_hpc_directml.py --workload cufft --duration 600
    python scientific_hpc_directml.py --workload nbody --duration 600 --n-bodies 4096
"""

import argparse
import sys
import time

import torch

from _dml_device import describe, get_device, sync


def run_cufft(duration: int, device: torch.device):
    size = (256, 256, 64)  # smaller than the CUDA version to fit DML's stricter limits
    print(f"  3D FFT benchmark: {size} complex64, {duration}s")
    x = torch.randn(*size, dtype=torch.complex64, device=device)
    wall_start = time.time()
    last_print = wall_start
    iters = 0
    while time.time() - wall_start < duration:
        y = torch.fft.fftn(x)
        z = torch.fft.ifftn(y)
        x = z.real.to(dtype=torch.complex64) + x.imag * 1j
        sync(device)
        iters += 1
        now = time.time()
        if now - last_print >= 30:
            print(f"  iter {iters}: {now - wall_start:.1f}s")
            last_print = now
    print(f"FFT complete. iters={iters}, wall={time.time() - wall_start:.1f}s")


def run_nbody(duration: int, device: torch.device, n: int):
    print(f"  N-body sim: {n} bodies, {duration}s")
    G = 6.674e-11
    dt = 1.0
    pos = torch.randn(n, 3, device=device) * 1e11
    vel = torch.randn(n, 3, device=device) * 1e3
    mass = torch.rand(n, device=device) * 1e30 + 1e29
    wall_start = time.time()
    last_print = wall_start
    steps = 0
    while time.time() - wall_start < duration:
        # Pairwise distances and forces, O(N^2) compute
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)        # (n, n, 3)
        r2 = (diff * diff).sum(dim=-1) + 1e10              # softened distance squared
        inv_r3 = r2.pow(-1.5)
        force = (G * mass.view(-1, 1, 1) * diff * inv_r3.unsqueeze(-1)).sum(dim=0)
        accel = force / mass.unsqueeze(-1)
        vel = vel + accel * dt
        pos = pos + vel * dt
        sync(device)
        steps += 1
        now = time.time()
        if now - last_print >= 30:
            print(f"  step {steps}: {now - wall_start:.1f}s")
            last_print = now
    print(f"N-body complete. steps={steps}, wall={time.time() - wall_start:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="Scientific HPC (DirectML)")
    ap.add_argument("--workload", choices=["cufft", "nbody"], required=True)
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--n-bodies", type=int, default=4096)
    args = ap.parse_args()

    device = get_device("dml")
    print(f"Device: {describe(device)}")
    if args.workload == "cufft":
        run_cufft(args.duration, device)
    elif args.workload == "nbody":
        run_nbody(args.duration, device, args.n_bodies)


if __name__ == "__main__":
    import os
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
