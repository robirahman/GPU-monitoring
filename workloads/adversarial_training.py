#!/usr/bin/env python3
"""Adversarial training workloads — disguised ML training for evasion testing.

Three strategies that modify real training to evade telemetry classifiers:

  Strategy A — Utilization modulation:
    Insert sync + sleep between batches to create bursty utilization pattern
    resembling rendering workloads. Disrupts steady-state power/util features.

  Strategy B — Low-utilization training:
    Tiny batch size (bs=2) + heavy CPU augmentation + throttled GPU clock.
    Makes training look like inference or idle. Targets util-based features.

  Strategy D — Temporal disruption:
    Randomize batch sizes each step, inject random no-op CUDA kernels between
    batches, break epoch periodicity. Targets autocorrelation/FFT features.

All strategies perform REAL training (gradients flow, loss decreases) so the
throughput penalty is genuine and measurable for evasion-tax analysis.

Usage:
    python adversarial_training.py --strategy A --duration 300
    python adversarial_training.py --strategy B --duration 300
    python adversarial_training.py --strategy D --duration 300
"""

import argparse
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



def build_resnet18_cifar(num_classes=10):
    model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    return model


# ── Strategy A: Utilization Modulation ──────────────────────────────────────

def strategy_a(args, device):
    """Insert sync+sleep between batches to mimic rendering's bursty pattern.

    The sleep duration creates idle gaps that:
    - Lower mean GPU utilization
    - Increase utilization CV (coefficient of variation)
    - Break power autocorrelation (ACF1)
    - Create an irregular duty cycle similar to frame rendering
    """
    print(f"Strategy A: Utilization Modulation")
    print(f"  Sleep range: {args.sleep_min:.2f}-{args.sleep_max:.2f}s between batches")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=256, shuffle=True, num_workers=2, pin_memory=True,
    )

    model = build_resnet18_cifar().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    start = time.time()
    steps = 0
    epoch = 0
    while time.time() - start < args.duration:
        epoch += 1
        running_loss = 0.0
        for data, target in loader:
            if time.time() - start >= args.duration:
                break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            emit_step("train", steps, tokens=data.size(0))

            # THE DISGUISE: sync GPU then sleep for a random interval
            torch.cuda.synchronize()
            sleep_time = random.uniform(args.sleep_min, args.sleep_max)
            time.sleep(sleep_time)
            # Closes the training slot at the gradient step. Without this the
            # sleep is folded into the step before it, because a STEP line
            # reports the work done since the previous line.
            emit_step("stall", steps)

        if epoch % 5 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch}: loss={running_loss/max(steps,1):.4f}, "
                  f"steps={steps}, elapsed={elapsed:.0f}s")

    elapsed = time.time() - start
    throughput = steps * 256 / elapsed
    print(f"Strategy A complete: {steps} steps, {elapsed:.1f}s, "
          f"{throughput:.0f} samples/sec")


# ── Strategy B: Low-Utilization Training ────────────────────────────────────

def strategy_b(args, device):
    """Tiny batch + heavy CPU augmentation to look like inference.

    Uses batch_size=2, aggressive random augmentations on CPU,
    and an artificially slow DataLoader to keep GPU utilization <20%.
    The GPU sees brief compute bursts separated by long data-loading gaps.
    """
    print(f"Strategy B: Low-Utilization Training")
    print(f"  Batch size: 2, heavy CPU augmentation")

    # Heavy CPU-side augmentation pipeline to bottleneck on CPU
    transform = transforms.Compose([
        transforms.RandomResizedCrop(32, scale=(0.2, 1.0)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        transforms.RandomGrayscale(p=0.2),
        transforms.RandomRotation(15),
        transforms.GaussianBlur(kernel_size=3),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        transforms.RandomErasing(p=0.5),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform
    )
    # Single worker + tiny batch = CPU-bottlenecked
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=2, shuffle=True, num_workers=0, pin_memory=False,
    )

    model = build_resnet18_cifar().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    start = time.time()
    steps = 0
    epoch = 0
    while time.time() - start < args.duration:
        epoch += 1
        running_loss = 0.0
        for data, target in loader:
            if time.time() - start >= args.duration:
                break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            steps += 1
            # Tiny batches and CPU-side augmentation make this look like
            # inference; it is a gradient step on every one of them.
            emit_step("train", steps, tokens=data.size(0))

        if epoch % 2 == 0:
            elapsed = time.time() - start
            print(f"  Epoch {epoch}: loss={running_loss/max(steps,1):.4f}, "
                  f"steps={steps}, elapsed={elapsed:.0f}s")

    elapsed = time.time() - start
    throughput = steps * 2 / elapsed
    print(f"Strategy B complete: {steps} steps, {elapsed:.1f}s, "
          f"{throughput:.0f} samples/sec")


# ── Strategy D: Temporal Disruption ─────────────────────────────────────────

def strategy_d(args, device):
    """Randomize batch sizes and inject noise to break periodicity.

    Each step uses a random batch size from [4, 512], and between steps
    a random CUDA kernel (matmul of random size) runs for a random duration.
    This destroys:
    - Epoch periodicity (FFT peak power)
    - Autocorrelation at all lags
    - Regular memory allocation patterns
    """
    print(f"Strategy D: Temporal Disruption")
    print(f"  Random batch sizes [4-512], random CUDA noise between steps")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="./data/cifar10", train=True, download=True, transform=transform
    )

    model = build_resnet18_cifar().to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    model.train()

    # Pre-create dataset tensor for random-batch slicing
    all_data = torch.stack([dataset[i][0] for i in range(len(dataset))])
    all_targets = torch.tensor([dataset[i][1] for i in range(len(dataset))])

    start = time.time()
    steps = 0
    total_samples = 0
    while time.time() - start < args.duration:
        # Random batch size each step
        bs = random.choice([4, 8, 16, 32, 64, 128, 256, 512])
        idx = torch.randint(0, len(dataset), (bs,))
        data = all_data[idx].to(device)
        target = all_targets[idx].to(device)

        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        steps += 1
        total_samples += bs
        emit_step("train", steps, tokens=bs)

        # THE DISGUISE: inject random CUDA noise between some steps
        if random.random() < 0.3:  # 30% chance of noise injection
            noise_size = random.randint(64, 2048)
            noise_iters = random.randint(1, 20)
            a = torch.randn(noise_size, noise_size, device=device)
            for _ in range(noise_iters):
                _ = torch.mm(a, a)
            del a
            # Real matmuls, but not ML work: labelling them `infer` would
            # assert a forward pass that never happened.
            emit_step("nonml", steps)

        # Random micro-sleep (0-50ms) to further break periodicity
        if random.random() < 0.2:
            time.sleep(random.uniform(0, 0.05))
            emit_step("stall", steps)

        if steps % 200 == 0:
            elapsed = time.time() - start
            print(f"  Step {steps}: loss={loss.item():.4f}, "
                  f"bs={bs}, elapsed={elapsed:.0f}s")

    elapsed = time.time() - start
    throughput = total_samples / elapsed
    print(f"Strategy D complete: {steps} steps, {total_samples} samples, "
          f"{elapsed:.1f}s, {throughput:.0f} samples/sec")


# ── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Adversarial training workloads")
    parser.add_argument("--strategy", type=str, required=True, choices=["A", "B", "D"],
                        help="Adversarial strategy: A (util modulation), B (low-util), D (temporal disruption)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Run duration in seconds (default: 300)")
    parser.add_argument("--device", type=str, default="cuda")
    # Strategy A parameters
    parser.add_argument("--sleep-min", type=float, default=0.1,
                        help="Strategy A: min sleep between batches (default: 0.1s)")
    parser.add_argument("--sleep-max", type=float, default=0.5,
                        help="Strategy A: max sleep between batches (default: 0.5s)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)
    print(f"Device: {device}")

    strategies = {"A": strategy_a, "B": strategy_b, "D": strategy_d}
    strategies[args.strategy](args, device)


if __name__ == "__main__":
    main()
