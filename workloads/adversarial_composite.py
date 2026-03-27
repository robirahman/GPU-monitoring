#!/usr/bin/env python3
"""Composite adversarial training — maximally evasive strategy.

Combines all known evasion techniques into a single parameterized workload
to find the minimum throughput cost for evading the Tier 1 classifier.

Evasion techniques applied simultaneously:
  1. Memory pre-allocation: allocate a large buffer upfront to flatten
     the memory ramp (spoofs mem_slope, mem_slope_ratio features)
  2. Idle padding: sleep for fraction F of total time between batches
     (spoofs power, temperature, utilization features)
  3. Random timing noise: vary sleep durations + inject dummy CUDA kernels
     (spoofs autocorrelation, FFT periodicity features)
  4. Gradient accumulation: only call optimizer.step() every N batches
     (reduces periodic optimizer memory spikes)

Key parameter: --idle-fraction F (0.0 to 0.95)
  F=0.0: no evasion, full throughput (should be detected as training)
  F=0.5: sleep 50% of the time (50% throughput loss)
  F=0.95: sleep 95% of the time (95% throughput loss, may evade)

The sweep over F produces the evasion tax curve.

Usage:
    python adversarial_composite.py --idle-fraction 0.0 --duration 120
    python adversarial_composite.py --idle-fraction 0.5 --duration 120
    python adversarial_composite.py --idle-fraction 0.9 --duration 120
"""

import argparse
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


def build_resnet18_cifar(num_classes=10):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


def main():
    parser = argparse.ArgumentParser(description="Composite adversarial training")
    parser.add_argument("--idle-fraction", type=float, required=True,
                        help="Fraction of time to idle (0.0-0.95)")
    parser.add_argument("--duration", type=int, default=120,
                        help="Total run duration in seconds")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Training batch size")
    parser.add_argument("--accum-steps", type=int, default=4,
                        help="Gradient accumulation steps")
    parser.add_argument("--prealloc-mb", type=int, default=4096,
                        help="Memory to pre-allocate in MB (flattens memory ramp)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available")
        args.device = "cpu"
    device = torch.device(args.device)

    idle_frac = max(0.0, min(0.95, args.idle_fraction))

    print(f"Composite Adversarial Training")
    print(f"  Idle fraction: {idle_frac:.0%}")
    print(f"  Batch size: {args.batch_size}")
    print(f"  Grad accumulation: {args.accum_steps} steps")
    print(f"  Memory pre-alloc: {args.prealloc_mb} MB")
    print(f"  Duration: {args.duration}s")
    print(f"  Device: {device}")

    # ── EVASION 1: Memory pre-allocation ────────────────────────────────
    # Allocate a large buffer upfront to mask the memory ramp from
    # model + optimizer initialization. This makes mem_slope ≈ 0.
    if args.prealloc_mb > 0 and device.type == "cuda":
        prealloc_bytes = args.prealloc_mb * 1024 * 1024
        _prealloc_buffer = torch.empty(prealloc_bytes // 4, dtype=torch.float32, device=device)
        actual_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"  Pre-allocated: {actual_mb:.0f} MB on GPU")

    # ── Setup model and data ────────────────────────────────────────────
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=args.batch_size, shuffle=True,
        num_workers=2, pin_memory=True,
    )

    model = build_resnet18_cifar().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # ── Training loop with evasion ──────────────────────────────────────
    start = time.time()
    steps = 0
    total_samples = 0
    optimizer.zero_grad()

    while time.time() - start < args.duration:
        for data, target in loader:
            if time.time() - start >= args.duration:
                break

            data, target = data.to(device), target.to(device)

            # Forward + backward (always happens — real training)
            output = model(data)
            loss = criterion(output, target)
            loss = loss / args.accum_steps  # Scale for accumulation
            loss.backward()

            steps += 1
            total_samples += data.size(0)

            # ── EVASION 4: Gradient accumulation ────────────────────
            # Only step optimizer every N batches to reduce periodic
            # memory spikes from optimizer state updates
            if steps % args.accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            # ── EVASION 2 + 3: Idle padding with random timing ─────
            if idle_frac > 0:
                # Sync GPU before sleeping (so power drops during sleep)
                torch.cuda.synchronize()

                # Compute sleep duration to achieve target idle fraction
                # If we want to be idle F% of the time, and the batch took
                # T_compute seconds, we need to sleep T_compute * F / (1 - F)
                batch_time = 0.01  # Approximate compute time per batch
                target_sleep = batch_time * idle_frac / max(1 - idle_frac, 0.01)

                # ── EVASION 3: Randomize sleep to break autocorrelation
                # Add ±50% noise to the sleep duration
                jitter = random.uniform(0.5, 1.5)
                actual_sleep = target_sleep * jitter
                actual_sleep = min(actual_sleep, 2.0)  # Cap at 2s

                time.sleep(actual_sleep)

                # Occasionally inject a dummy CUDA kernel (10% chance)
                # to further randomize the power/util profile
                if random.random() < 0.1:
                    noise_size = random.randint(128, 1024)
                    a = torch.randn(noise_size, noise_size, device=device)
                    _ = torch.mm(a, a)
                    del a

        if steps % 50 == 0 and steps > 0:
            elapsed = time.time() - start
            throughput = total_samples / elapsed
            print(f"  Step {steps}: loss={loss.item()*args.accum_steps:.4f}, "
                  f"{throughput:.0f} samples/sec, {elapsed:.0f}s")

    elapsed = time.time() - start
    throughput = total_samples / elapsed

    # Free pre-allocated buffer
    if args.prealloc_mb > 0 and device.type == "cuda":
        del _prealloc_buffer
        torch.cuda.empty_cache()

    print(f"\nComposite adversarial complete:")
    print(f"  Steps: {steps}")
    print(f"  Samples: {total_samples}")
    print(f"  Throughput: {throughput:.0f} samples/sec")
    print(f"  Duration: {elapsed:.1f}s")
    print(f"  Idle fraction: {idle_frac:.0%}")
    # Print throughput stats for evasion tax computation
    print(f"  EVASION_TAX_DATA: idle_frac={idle_frac}, throughput={throughput:.1f}, "
          f"samples={total_samples}, steps={steps}, duration={elapsed:.1f}")


if __name__ == "__main__":
    main()
