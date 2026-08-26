"""Steady DRAM-bandwidth cover workload for the co-execution evader.

Runs continuous large-tensor reads/writes that saturate the memory controller at
a roughly constant rate, so that when it co-executes with tiny-batch training on
the same GPU, the aggregate NVML `mem_utilization_pct` / DCGM `dram_active` and
SM-occupancy are pinned high and *flat* — burying the training's bursty,
compute-bound, low-DRAM signature under a steady HPC-like envelope.

It deliberately does memory-bound elementwise ops (not matmul), so it raises
DRAM activity without contributing a distinctive compute signature.

Usage: python scripts/cover_kernel.py --duration 320 --gb 3 --gpu 0
"""
import argparse
import time

import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=float, default=320.0)
    ap.add_argument("--gb", type=float, default=3.0, help="working-set GiB (must miss cache)")
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--intensity", type=float, default=1.0,
                    help="fraction of time spent churning (1.0 = continuous)")
    args = ap.parse_args()

    torch.cuda.set_device(args.gpu)
    dev = torch.device(f"cuda:{args.gpu}")
    # two buffers of gb/2 each (float32) so a+=b streams the full working set
    n = int(args.gb * (1024**3) / 2 / 4)
    a = torch.ones(n, device=dev, dtype=torch.float32)
    b = torch.ones(n, device=dev, dtype=torch.float32)
    torch.cuda.synchronize()
    print(f"cover: churning {args.gb} GiB working set on cuda:{args.gpu}", flush=True)

    period = 0.05
    end = time.time() + args.duration
    while time.time() < end:
        t0 = time.time()
        # memory-bound: elementwise, DRAM-bandwidth limited, minimal compute
        while time.time() - t0 < period * args.intensity:
            a.add_(b)
            b.mul_(1.0000001)
        torch.cuda.synchronize()
        rest = period - (time.time() - t0)
        if rest > 0:
            time.sleep(rest)


if __name__ == "__main__":
    main()
