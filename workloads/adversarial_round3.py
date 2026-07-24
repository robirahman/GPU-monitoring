#!/usr/bin/env python3
"""Round 3 adversarial strategies — strongest possible attacks.

These target the fundamental training/inference boundary rather than trying
to disguise training as a non-ML workload. The key insight: the minimum
difference between training and inference is the backward pass (~2x FLOPS).

Strategy K: Online Learning Camouflage
    Identical to the inference workload (same model, batch size, data pipeline)
    except each batch also does loss.backward() + optimizer.step().
    Uses SGD with no momentum (zero persistent optimizer state).
    This is the MINIMUM possible delta from inference.

Strategy L: Diluted Training (parameterized)
    Inference loop where every N-th batch also does a gradient update.
    At N=20, 95% of compute is pure inference.
    Sweep N to find the detection threshold.

Usage:
    python adversarial_round3.py --strategy K --duration 300
    python adversarial_round3.py --strategy L --duration 300 --dilution 5
    python adversarial_round3.py --strategy L --duration 300 --dilution 20
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def get_cifar10_test_loader(batch_size):
    """Same data pipeline as inference.py — test set, no augmentation."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=False, download=True, transform=transform
    )
    return DataLoader(dataset, batch_size=batch_size, shuffle=False,
                      num_workers=0, pin_memory=False)


# ── Strategy K: Online Learning Camouflage ───────────────────────────────────

def strategy_k_online_learning(args, device):
    """Training disguised as inference — minimum delta from pure inference.

    This is structurally identical to workloads/inference.py except:
    1. torch.no_grad() is removed
    2. loss.backward() + optimizer.step() is added after each forward pass
    3. SGD with lr=0.001, no momentum (zero optimizer state)

    The telemetry should show:
    - Same memory as inference (gradients freed immediately, no optimizer state)
    - ~2-3x power/FLOPS due to backward pass
    - Same batch pattern, same data, same model
    """
    batch_size = args.batch_size
    print(f"Strategy K: Online Learning Camouflage")
    print(f"  Model: resnet50, Batch: {batch_size}, Duration: {args.duration}s")
    print(f"  Identical to inference except: backward pass + weight update per batch")

    loader = get_cifar10_test_loader(batch_size)

    # Same model as inference workload
    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    model.train()  # enable gradients (vs model.eval() in inference)

    # SGD with NO momentum — zero persistent optimizer state
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    passes = 0

    while time.time() - wall_start < args.duration:
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break
            data, target = data.to(device), target.to(device)

            # Forward pass (same as inference)
            output = model(data)
            total_samples += data.size(0)

            # Backward pass + update (the ONLY difference from inference)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

            # Immediately free gradient memory
            del output, loss

        passes += 1
        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            throughput = total_samples / elapsed if elapsed > 0 else 0
            mem = torch.cuda.memory_allocated() / 1024**2
            print(f"  Pass {passes}: {total_samples} samples, {throughput:.0f} img/s, "
                  f"mem={mem:.0f}MB, {elapsed:.1f}s")
            last_print = now

    wall_time = time.time() - wall_start
    throughput = total_samples / wall_time
    print(f"Strategy K complete. {total_samples} samples, {throughput:.0f} img/s, {wall_time:.1f}s")


# ── Strategy L: Diluted Training ─────────────────────────────────────────────

def strategy_l_diluted(args, device):
    """Inference loop with occasional gradient updates.

    Every N-th batch does forward + backward + update.
    All other batches do forward only (pure inference).
    At N=20, 95% of compute is indistinguishable from inference.
    """
    dilution = args.dilution
    batch_size = args.batch_size
    print(f"Strategy L: Diluted Training (1/{dilution} batches train)")
    print(f"  Model: resnet50, Batch: {batch_size}, Duration: {args.duration}s")
    print(f"  Training fraction: {1/dilution:.1%}")

    loader = get_cifar10_test_loader(batch_size)

    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    train_batches = 0
    infer_batches = 0
    batch_idx = 0

    while time.time() - wall_start < args.duration:
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break
            data, target = data.to(device), target.to(device)

            if batch_idx % dilution == 0:
                # Training batch: forward + backward + update
                model.train()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                del output, loss
                # Prevent GPU memory fragmentation from mode switching
                torch.cuda.empty_cache()
                train_batches += 1
            else:
                # Inference batch: forward only
                model.eval()
                with torch.no_grad():
                    _ = model(data)
                infer_batches += 1

            # Per-step ground truth for window re-scoring (see
            # scripts/label_windows_by_training.py). Deterministic schedule
            # (batch_idx % dilution), but per-batch wall-clock timing is not
            # otherwise recorded, so log it.
            print(f"STEP ts={time.time():.3f} "
                  f"kind={'train' if batch_idx % dilution == 0 else 'infer'} "
                  f"batch_idx={batch_idx}", flush=True)
            total_samples += data.size(0)
            batch_idx += 1
            del data, target

        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            throughput = total_samples / elapsed if elapsed > 0 else 0
            print(f"  {elapsed:.0f}s: {total_samples} samples, {throughput:.0f} img/s, "
                  f"train={train_batches}, infer={infer_batches}")
            last_print = now

    wall_time = time.time() - wall_start
    throughput = total_samples / wall_time
    print(f"Strategy L complete. {total_samples} samples ({throughput:.0f} img/s), "
          f"train_batches={train_batches}, infer_batches={infer_batches}, "
          f"train_frac={train_batches/(train_batches+infer_batches+1e-9):.1%}")


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Round 3 adversarial strategies")
    parser.add_argument("--strategy", type=str, required=True, choices=["K", "L"])
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256, same as inference)")
    parser.add_argument("--dilution", type=int, default=5,
                        help="Strategy L: train on 1/N batches (default: 5)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    {"K": strategy_k_online_learning, "L": strategy_l_diluted}[args.strategy](args, device)


if __name__ == "__main__":
    main()
