#!/usr/bin/env python3
"""Advanced adversarial evasion strategies for SPAR telemetry classification.

These strategies address the fundamental physical signals that betray training:
- High sustained memory usage
- Elevated SM/memory clock frequencies
- Sustained power draw and temperature
- PCIe batch-loading periodicity

Strategy E: Memory-Minimal Training
    Uses gradient checkpointing + aggressive GC to minimize memory footprint.
    Goal: memory looks like inference, not training.

Strategy F: Workload Interleaving
    Alternates between real training and real non-training compute.
    Goal: 60s window features average between training and other.

Strategy G: Clock-Throttled Stealth Training
    Locks GPU to base clocks and low power limit via nvidia-smi.
    Goal: power/clock/temperature signatures match idle/light compute.

Usage:
    python adversarial_advanced.py --strategy E --duration 300
    python adversarial_advanced.py --strategy F --duration 300 --train-frac 0.3
    python adversarial_advanced.py --strategy G --duration 300 --power-limit 100
"""

import argparse
import gc
import os
import random
import subprocess
import sys
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.checkpoint import checkpoint_sequential


def get_cifar10_loader(batch_size, num_workers=0):
    transform = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=True, download=True, transform=transform
    )
    return torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=False, drop_last=True,
    )


# ── Strategy E: Memory-Minimal Training ─────────────────────────────────────

def strategy_e_memory_minimal(args, device):
    """Train with minimal memory footprint using gradient checkpointing.

    Key techniques:
    - Gradient checkpointing: recompute activations during backward pass
      instead of storing them. Trades compute for memory.
    - Tiny batch size (4): minimizes activation memory.
    - Aggressive garbage collection: torch.cuda.empty_cache() after every step.
    - No optimizer state accumulation: use SGD (no momentum buffers).
    - Delete all intermediates immediately.

    Physical effect: memory stays low (~500-1500 MB) like inference.
    Cost: ~3-5x slower than normal training (recomputation overhead).
    """
    print(f"Strategy E: Memory-Minimal Training")
    print(f"  Gradient checkpointing + tiny batch + aggressive GC")

    batch_size = 4  # Tiny batches to minimize activation memory
    loader = get_cifar10_loader(batch_size)

    # Use ResNet-18 with checkpoint_sequential for memory savings
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)

    # SGD with no momentum (avoids storing momentum buffers)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    # Split model into segments for checkpointing
    # ResNet-18 sequential segments
    segments = [
        nn.Sequential(model.conv1, model.bn1, model.relu, model.maxpool),
        model.layer1,
        model.layer2,
        model.layer3,
        model.layer4,
    ]

    wall_start = time.time()
    last_print = wall_start
    steps = 0
    total_loss = 0

    while time.time() - wall_start < args.duration:
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad(set_to_none=True)  # set_to_none saves memory

            # Forward with gradient checkpointing
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

            # Aggressive memory cleanup
            del data, target, output, loss, x
            torch.cuda.empty_cache()
            gc.collect()

            now = time.time()
            if now - last_print >= 30:
                elapsed = now - wall_start
                mem = torch.cuda.memory_allocated() / 1024**2
                print(f"  Step {steps}: loss={total_loss/steps:.4f}, "
                      f"mem={mem:.0f}MB, {elapsed:.0f}s")
                last_print = now

    elapsed = time.time() - wall_start
    print(f"Strategy E complete. {steps} steps in {elapsed:.0f}s, "
          f"avg loss={total_loss/max(steps,1):.4f}")


# ── Strategy F: Workload Interleaving ────────────────────────────────────────

def strategy_f_interleaving(args, device):
    """Alternate between real training and real non-training compute.

    Key idea: within each 60s window, spend only a fraction of time training
    and the rest doing genuine non-training work (mining-like memory-hard
    hashing or cuFFT). The 60s window features will be a weighted average.

    Unlike idle padding (which creates bimodal util), this fills idle time
    with genuine compute that has different telemetry characteristics.

    Physical effect: features blend between training and other.
    Cost: throughput reduced proportional to train_frac.
    """
    train_frac = args.train_frac
    train_window = args.interleave_sec * train_frac
    other_window = args.interleave_sec * (1 - train_frac)

    print(f"Strategy F: Workload Interleaving")
    print(f"  Train for {train_window:.1f}s, then non-training for {other_window:.1f}s")
    print(f"  Cycle: {args.interleave_sec}s, Train fraction: {train_frac:.0%}")

    # Set up training
    loader = get_cifar10_loader(batch_size=64)
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()
    loader_iter = iter(loader)

    # Set up non-training workloads
    # Mining-like: memory-hard random access
    dag_elements = (512 * 1024 * 1024) // 64  # 512MB DAG
    dag = torch.randint(0, 2**31, (dag_elements, 16), dtype=torch.int32, device=device)

    # cuFFT work
    fft_data = torch.randn(8, 512, 512, dtype=torch.complex64, device=device)

    wall_start = time.time()
    last_print = wall_start
    train_steps = 0
    other_iters = 0
    phase = "train"
    phase_start = wall_start

    while time.time() - wall_start < args.duration:
        now = time.time()
        phase_elapsed = now - phase_start

        if phase == "train" and phase_elapsed >= train_window:
            phase = "other"
            phase_start = now
            # Free training memory during other phase
            torch.cuda.empty_cache()
        elif phase == "other" and phase_elapsed >= other_window:
            phase = "train"
            phase_start = now

        if phase == "train":
            # Real training step
            try:
                data, target = next(loader_iter)
            except StopIteration:
                loader_iter = iter(loader)
                data, target = next(loader_iter)

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            train_steps += 1
            del data, target, output, loss

        else:
            # Non-training: alternate between mining-like and cuFFT
            if other_iters % 2 == 0:
                # Mining-like memory-hard access
                seeds = torch.randint(0, 2**31, (256, 8), dtype=torch.int32, device=device)
                mix = seeds.clone()
                for _ in range(16):
                    indices = (mix[:, 0].abs() % dag_elements).long()
                    dag_row = dag[indices]
                    mix = mix ^ dag_row[:, :8]
                _ = mix.sum()
                torch.cuda.synchronize()
                del seeds, mix
            else:
                # cuFFT
                y = torch.fft.fftn(fft_data)
                z = torch.fft.ifftn(y)
                _ = z.real.sum()
                torch.cuda.synchronize()
                del y, z

            other_iters += 1

        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            print(f"  Steps: train={train_steps}, other={other_iters}, "
                  f"phase={phase}, {elapsed:.0f}s")
            last_print = now

    elapsed = time.time() - wall_start
    print(f"Strategy F complete. train_steps={train_steps}, other_iters={other_iters}, "
          f"{elapsed:.0f}s")

    # Clean up DAG
    del dag, fft_data
    torch.cuda.empty_cache()


# ── Strategy G: Clock-Throttled Stealth Training ─────────────────────────────

def strategy_g_clock_throttled(args, device):
    """Train with GPU locked to low clocks and power limit.

    Key techniques:
    - Lock SM clocks to base frequency (not boost)
    - Set low power limit via nvidia-smi
    - Result: power/clock/temperature features match light compute

    Physical effect: GPU operates in a low-power regime that looks like
    idle or light inference. Training still happens but slowly.
    Cost: 5-10x throughput loss from clock/power throttling.
    """
    power_limit = args.power_limit  # watts
    print(f"Strategy G: Clock-Throttled Stealth Training")
    print(f"  Power limit: {power_limit}W")

    # Try to set power limit (requires admin on Windows, may fail)
    original_power = None
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=power.default_limit", "--format=csv,noheader,nounits"],
            capture_output=True, text=True
        )
        original_power = float(result.stdout.strip())
        print(f"  Original power limit: {original_power}W")

        subprocess.run(
            ["nvidia-smi", "-pl", str(power_limit)],
            capture_output=True, text=True
        )
        print(f"  Set power limit to {power_limit}W")
    except Exception as e:
        print(f"  WARNING: Could not set power limit: {e}")
        print(f"  Falling back to software throttling")

    # Try to lock clocks low
    try:
        # Lock SM clock to 600 MHz (well below boost)
        subprocess.run(
            ["nvidia-smi", "--lock-gpu-clocks=210,600"],
            capture_output=True, text=True
        )
        print(f"  Locked GPU clocks to 210-600 MHz")
    except Exception as e:
        print(f"  WARNING: Could not lock clocks: {e}")

    # Train with small batch (less work per step = lower sustained power)
    loader = get_cifar10_loader(batch_size=32)
    model = torchvision.models.resnet18(weights=None, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    steps = 0
    total_loss = 0

    try:
        while time.time() - wall_start < args.duration:
            for data, target in loader:
                if time.time() - wall_start >= args.duration:
                    break

                data, target = data.to(device), target.to(device)
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()

                total_loss += loss.item()
                steps += 1
                del data, target, output, loss

                # Software throttle: brief sleep between steps to further reduce power
                time.sleep(0.01)

                now = time.time()
                if now - last_print >= 30:
                    elapsed = now - wall_start
                    mem = torch.cuda.memory_allocated() / 1024**2
                    print(f"  Step {steps}: loss={total_loss/steps:.4f}, "
                          f"mem={mem:.0f}MB, {elapsed:.0f}s")
                    last_print = now
    finally:
        # ALWAYS restore power and clock settings
        try:
            if original_power:
                subprocess.run(
                    ["nvidia-smi", "-pl", str(int(original_power))],
                    capture_output=True, text=True
                )
                print(f"  Restored power limit to {int(original_power)}W")
            subprocess.run(
                ["nvidia-smi", "--reset-gpu-clocks"],
                capture_output=True, text=True
            )
            print(f"  Reset GPU clocks to default")
        except Exception:
            pass

    elapsed = time.time() - wall_start
    print(f"Strategy G complete. {steps} steps in {elapsed:.0f}s")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Advanced adversarial evasion strategies")
    parser.add_argument("--strategy", type=str, required=True, choices=["E", "F", "G"],
                        help="Evasion strategy: E=memory-minimal, F=interleaving, G=clock-throttled")
    parser.add_argument("--duration", type=int, default=300, help="Duration in seconds")
    parser.add_argument("--device", type=str, default="cuda")
    # Strategy F parameters
    parser.add_argument("--train-frac", type=float, default=0.3,
                        help="Fraction of time spent training (Strategy F, default: 0.3)")
    parser.add_argument("--interleave-sec", type=float, default=10.0,
                        help="Cycle length in seconds (Strategy F, default: 10)")
    # Strategy G parameters
    parser.add_argument("--power-limit", type=int, default=100,
                        help="GPU power limit in watts (Strategy G, default: 100)")

    args = parser.parse_args()
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    strategies = {
        "E": strategy_e_memory_minimal,
        "F": strategy_f_interleaving,
        "G": strategy_g_clock_throttled,
    }

    strategies[args.strategy](args, device)


if __name__ == "__main__":
    main()
