#!/usr/bin/env python3
"""Round 4 adversarial strategies — targeting LOO classifier weaknesses.

LOO analysis revealed two exploitable gaps:
  - E_memory_minimal at 70% detection: gradient checkpointing masks memory
  - composite at 10-40% idle: idle padding + pre-alloc partially evades
  - L_diluted at N=5: 3.7% detection (but uses train/eval mode switching)

Round 4 combines the strongest elements of these into new strategies.

Strategy M: Composite + Memory Minimal
    Combines adversarial_composite's idle padding, memory pre-allocation,
    and timing noise with Strategy E's gradient checkpointing and
    aggressive GC. ResNet-18, batch 4, SGD no momentum. Parameterized
    by idle fraction.

Strategy N: Gradient Accumulation Dilution
    Unlike Strategy L (which alternates between train/eval mode), this
    always runs forward+backward on every batch but only calls
    optimizer.step() every N batches. No mode switching means no
    utilization transitions between train() and eval() — the temporal
    signature is more uniform than L's alternating pattern.
    ResNet-50, batch 128 (matching dilution sweep).

Usage:
    python adversarial_round4.py --strategy M --idle-fraction 0.2 --duration 300
    python adversarial_round4.py --strategy N --accum-steps 8 --duration 300
"""

import argparse
import gc
import random
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

try:
    from workloads._step import emit_step
except ImportError:  # launched as `python workloads/<script>.py`
    from _step import emit_step

from torch.utils.checkpoint import checkpoint_sequential


# ── Strategy M: Composite + Memory Minimal ──────────────────────────────────

def strategy_m_composite_memory_minimal(args, device):
    """Combine idle padding + memory pre-alloc + gradient checkpointing.

    From composite: memory pre-allocation (flatten mem_slope), idle padding
    (reduce power/util features), random timing noise (break autocorrelation).

    From E: gradient checkpointing (recompute activations instead of storing),
    tiny batch size (4), aggressive GC, SGD no momentum.

    The combination should mask both memory-based AND power-based features.
    """
    idle_frac = max(0.0, min(0.95, args.idle_fraction))
    prealloc_mb = args.prealloc_mb

    print(f"Strategy M: Composite + Memory Minimal")
    print(f"  Idle fraction: {idle_frac:.0%}")
    print(f"  Batch size: 4 (tiny, like E)")
    print(f"  Gradient checkpointing: ON")
    print(f"  Memory pre-alloc: {prealloc_mb} MB")
    print(f"  Duration: {args.duration}s")

    # Memory pre-allocation (from composite)
    if prealloc_mb > 0:
        prealloc_bytes = prealloc_mb * 1024 * 1024
        _prealloc_buffer = torch.empty(
            prealloc_bytes // 4, dtype=torch.float32, device=device)
        actual_mb = torch.cuda.memory_allocated(device) / (1024 * 1024)
        print(f"  Pre-allocated: {actual_mb:.0f} MB on GPU")

    # Data (same augmentation as E)
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=4, shuffle=True,
        num_workers=0, pin_memory=False, drop_last=True)

    # Model with checkpointing segments (from E)
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    segments = [
        nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]

    start = time.time()
    last_print = start
    steps = 0
    total_loss = 0

    while time.time() - start < args.duration:
        for data, target in loader:
            if time.time() - start >= args.duration:
                break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)

            # Forward with gradient checkpointing (from E)
            x = data
            for seg in segments:
                x = checkpoint_sequential(nn.Sequential(seg), 1, x,
                                          use_reentrant=False)
            x = model.avgpool(x)
            x = torch.flatten(x, 1)
            output = model.fc(x)
            loss = criterion(output, target)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            steps += 1
            emit_step("train", steps, tokens=data.size(0))

            # Aggressive cleanup (from E)
            del data, target, output, loss, x
            torch.cuda.empty_cache()
            gc.collect()

            # Idle padding with random timing (from composite)
            if idle_frac > 0:
                torch.cuda.synchronize()
                batch_time = 0.02  # approximate per-step time with checkpointing
                target_sleep = batch_time * idle_frac / max(1 - idle_frac, 0.01)
                jitter = random.uniform(0.5, 1.5)
                actual_sleep = min(target_sleep * jitter, 2.0)
                time.sleep(actual_sleep)
                emit_step("stall", steps)

                # Occasional dummy kernel (from composite)
                if random.random() < 0.1:
                    noise_size = random.randint(64, 256)
                    a = torch.randn(noise_size, noise_size, device=device)
                    _ = torch.mm(a, a)
                    del a

            now = time.time()
            if now - last_print >= 30:
                elapsed = now - start
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"  Step {steps}: loss={total_loss/steps:.4f}, "
                      f"mem={mem:.0f}MB, {elapsed:.0f}s")
                last_print = now

    # Free pre-alloc
    if prealloc_mb > 0:
        del _prealloc_buffer
        torch.cuda.empty_cache()

    elapsed = time.time() - start
    throughput = steps * 4 / elapsed  # samples/sec (batch_size=4)
    print(f"Strategy M complete. {steps} steps, {throughput:.1f} samples/sec, "
          f"{elapsed:.1f}s")


# ── Strategy N: Gradient Accumulation Dilution ──────────────────────────────

def strategy_n_grad_accum_dilution(args, device):
    """Always forward+backward, but optimizer.step() every N batches.

    Unlike Strategy L which alternates between model.train() and
    model.eval() (creating utilization transitions), this stays in
    train mode permanently. Every batch does:
      1. Forward pass
      2. loss.backward() (accumulate gradients)
      3. Only every N-th batch: optimizer.step() + zero_grad()

    The telemetry signature is more uniform than L's because there are
    no mode-switching transitions. The gradient accumulation means the
    optimizer state update (which can cause a memory/power spike) is
    spread across N batches.

    With N=16, the optimizer updates once every 16 batches — similar
    effective training rate to L_diluted_16, but with completely
    different temporal characteristics.
    """
    accum_steps = args.accum_steps
    batch_size = 128  # Match dilution sweep

    print(f"Strategy N: Gradient Accumulation Dilution")
    print(f"  Accumulation steps: {accum_steps}")
    print(f"  Model: resnet50, Batch: {batch_size}")
    print(f"  Duration: {args.duration}s")
    print(f"  Effective update rate: 1/{accum_steps} batches")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=False,
        num_workers=0, pin_memory=False)

    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    model.train()  # Stay in train mode always — no mode switching

    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    batch_idx = 0
    update_count = 0

    optimizer.zero_grad(set_to_none=True)

    while time.time() - wall_start < args.duration:
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break

            data, target = data.to(device), target.to(device)

            # Always forward + backward (no mode switching)
            output = model(data)
            loss = criterion(output, target) / accum_steps
            loss.backward()

            total_samples += data.size(0)
            batch_idx += 1

            # Only step optimizer every N batches
            if batch_idx % accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                update_count += 1
            # Strategy N never leaves train mode: every batch is a gradient
            # computation whether or not it triggers an optimizer update.
            emit_step("train", batch_idx, tokens=data.size(0),
                      phase="sync" if batch_idx % accum_steps == 0 else "accum")

            del output, loss, data, target

            now = time.time()
            if now - last_print >= 30:
                elapsed = now - wall_start
                throughput = total_samples / elapsed if elapsed > 0 else 0
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"  Batch {batch_idx}: {total_samples} samples, "
                      f"{throughput:.0f} img/s, updates={update_count}, "
                      f"mem={mem:.0f}MB, {elapsed:.0f}s")
                last_print = now

    elapsed = time.time() - wall_start
    throughput = total_samples / elapsed
    print(f"Strategy N complete. {total_samples} samples ({throughput:.0f} img/s), "
          f"batches={batch_idx}, updates={update_count}, "
          f"effective_update_rate=1/{accum_steps}, {elapsed:.1f}s")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Round 4 adversarial strategies")
    parser.add_argument("--strategy", type=str, required=True, choices=["M", "N"])
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--device", type=str, default="cuda")
    # Strategy M parameters
    parser.add_argument("--idle-fraction", type=float, default=0.2,
                        help="Strategy M: fraction of time to idle (default: 0.2)")
    parser.add_argument("--prealloc-mb", type=int, default=4096,
                        help="Strategy M: memory to pre-allocate in MB (default: 4096)")
    # Strategy N parameters
    parser.add_argument("--accum-steps", type=int, default=8,
                        help="Strategy N: gradient accumulation steps (default: 8)")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    {"M": strategy_m_composite_memory_minimal,
     "N": strategy_n_grad_accum_dilution}[args.strategy](args, device)


if __name__ == "__main__":
    main()
