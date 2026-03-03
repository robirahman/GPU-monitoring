#!/usr/bin/env python3
"""Simple PyTorch training workload for telemetry pipeline testing.

Supports ResNet-18 and a simple MLP on CIFAR-10. Configurable epochs,
batch size, learning rate, and mixed precision (AMP).

Usage:
    python pytorch_training.py --model resnet18 --epochs 5 --batch-size 128
    python pytorch_training.py --model resnet18 --epochs 5 --batch-size 128 --amp
    python pytorch_training.py --model simple_mlp --epochs 5 --batch-size 256
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms


class SimpleMLP(nn.Module):
    """3-layer MLP for CIFAR-10 (flattened 32x32x3 = 3072 input)."""

    def __init__(self, num_classes: int = 10):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3072, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        return self.layers(x)


def build_model(name: str, num_classes: int = 10) -> nn.Module:
    if name == "resnet18":
        model = torchvision.models.resnet18(weights=None, num_classes=num_classes)
        # Adapt first conv for CIFAR-10's 32x32 images (instead of ImageNet's 224x224)
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    elif name == "simple_mlp":
        return SimpleMLP(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}. Choose from: resnet18, simple_mlp")


def main():
    parser = argparse.ArgumentParser(description="PyTorch training workload")
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "simple_mlp"],
        help="Model architecture (default: resnet18)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10"],
        help="Dataset (default: cifar10)",
    )
    parser.add_argument("--epochs", type=int, default=5, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size")
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument(
        "--amp", action="store_true", help="Enable automatic mixed precision"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device: cuda or cpu (default: cuda)",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="./data/cifar10",
        help="Dataset download directory",
    )
    args = parser.parse_args()

    # Device setup
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)

    print(f"Training {args.model} on {args.dataset}")
    print(f"  Device: {device}")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  AMP: {args.amp}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
        pin_memory=(device.type == "cuda"),
    )

    # Model, optimizer, loss
    model = build_model(args.model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Training loop
    wall_start = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=args.amp):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        epoch_time = time.time() - epoch_start
        avg_loss = running_loss / len(train_loader)
        acc = 100.0 * correct / total
        print(
            f"  Epoch {epoch + 1}/{args.epochs}: "
            f"loss={avg_loss:.4f}, acc={acc:.1f}%, time={epoch_time:.1f}s"
        )

    wall_time = time.time() - wall_start
    print(f"Training complete. Total wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
