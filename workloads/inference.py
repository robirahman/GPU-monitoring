#!/usr/bin/env python3
"""ML inference workloads for SPAR telemetry collection.

Runs ResNet-50 batch inference (no gradients) on CIFAR-10 test set,
cycling repeatedly to fill the requested duration.

Usage:
    python inference.py --duration 600
    python inference.py --model resnet50 --batch-size 256 --duration 600
"""

import argparse
import time

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import DataLoader


def main():
    parser = argparse.ArgumentParser(description="ML inference workload")
    parser.add_argument("--model", type=str, default="resnet50",
                        choices=["resnet50", "resnet18"],
                        help="Model (default: resnet50)")
    parser.add_argument("--batch-size", type=int, default=256,
                        help="Batch size (default: 256)")
    parser.add_argument("--duration", type=int, default=600,
                        help="Run duration in seconds (default: 600)")
    parser.add_argument("--data-dir", type=str, default="./data/cifar10")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"Inference workload: {args.model}")
    print(f"  Device: {device}, Batch: {args.batch_size}, Duration: {args.duration}s")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                        num_workers=2, pin_memory=(args.device == "cuda"))

    if args.model == "resnet50":
        model = torchvision.models.resnet50(weights=None, num_classes=10)
    else:
        model = torchvision.models.resnet18(weights=None, num_classes=10)

    model = model.to(device)
    model.eval()

    wall_start = time.time()
    total_samples = 0
    passes = 0

    with torch.no_grad():
        while time.time() - wall_start < args.duration:
            for data, _ in loader:
                if time.time() - wall_start >= args.duration:
                    break
                data = data.to(device)
                _ = model(data)
                total_samples += data.size(0)
            passes += 1
            elapsed = time.time() - wall_start
            throughput = total_samples / elapsed if elapsed > 0 else 0
            print(f"  Pass {passes}: {total_samples} samples, {throughput:.0f} img/s, {elapsed:.1f}s elapsed")

    wall_time = time.time() - wall_start
    print(f"Inference complete. Samples: {total_samples}, Throughput: {total_samples/wall_time:.0f} img/s")


if __name__ == "__main__":
    main()
