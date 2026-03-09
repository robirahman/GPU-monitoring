#!/usr/bin/env python3
"""cuFFT benchmark via torch.fft.

Runs repeated large 2D FFTs on GPU. Scientific HPC signature — high SM
utilization, no tensor core activity, high memory bandwidth.

Usage:
    python cufft_benchmark.py --duration 300 --size 4096
"""

import argparse
import signal
import time

import torch


def main():
    parser = argparse.ArgumentParser(description="cuFFT benchmark")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--size", type=int, default=4096, help="Matrix size (NxN)")
    parser.add_argument("--batch", type=int, default=8, help="Batch of FFTs per iteration")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"cuFFT benchmark: {args.batch}x {args.size}x{args.size} complex FFTs")
    print(f"  Device: {device}, Duration: {args.duration}s")

    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)

    # Pre-allocate input tensor
    x = torch.randn(args.batch, args.size, args.size, dtype=torch.float32, device=device)

    wall_start = time.time()
    iterations = 0

    while (time.time() - wall_start) < args.duration and not stopped:
        # Forward FFT
        y = torch.fft.fft2(x)
        # Inverse FFT
        z = torch.fft.ifft2(y)
        torch.cuda.synchronize()
        iterations += 1
        if iterations % 50 == 0:
            elapsed = time.time() - wall_start
            print(f"  {iterations} iters, {elapsed:.0f}s elapsed")

    wall_time = time.time() - wall_start
    total_ffts = iterations * args.batch * 2  # forward + inverse
    print(
        f"cuFFT benchmark complete. {total_ffts} FFTs in {wall_time:.1f}s "
        f"({total_ffts / wall_time:.0f} FFTs/s)"
    )


if __name__ == "__main__":
    main()
