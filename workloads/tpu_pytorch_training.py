#!/usr/bin/env python3
"""TPU-adapted PyTorch training workload using PyTorch XLA.

Usage:
    python tpu_pytorch_training.py --model resnet18 --epochs 5 --batch-size 128
"""

import argparse
import time
import os

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

# PyTorch XLA imports
try:
    import torch_xla
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met
    import torch_xla.distributed.parallel_loader as pl
    XLA_AVAILABLE = True
except ImportError:
    XLA_AVAILABLE = False
    print("WARNING: torch_xla not found. Running on CPU/GPU fallback.")

class SimpleMLP(nn.Module):
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
        model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
        model.maxpool = nn.Identity()
        return model
    elif name == "simple_mlp":
        return SimpleMLP(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")

def main():
    parser = argparse.ArgumentParser(description="TPU PyTorch training workload")
    parser.add_argument("--model", type=str, default="resnet18", choices=["resnet18", "simple_mlp"])
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--data-dir", type=str, default="./data/cifar10")
    args = parser.parse_args()

    # Device setup for XLA
    if XLA_AVAILABLE:
        device = xm.xla_device()
        print(f"Using XLA device: {device}")
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"XLA not available, using device: {device}")

    # Dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform
    )
    
    # In XLA, we often use DistributedSampler even on a single TPU core
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=2,
    )

    # Wrap loader for XLA
    if XLA_AVAILABLE:
        train_loader = pl.ParallelLoader(train_loader, [device]).per_device_loader(device)

    model = build_model(args.model).to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    model.train()
    for epoch in range(args.epochs):
        epoch_start = time.time()
        running_loss = 0.0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            if not XLA_AVAILABLE:
                data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            
            if XLA_AVAILABLE:
                xm.optimizer_step(optimizer, barrier=True)
            else:
                optimizer.step()
                
            running_loss += loss.item()

        epoch_time = time.time() - epoch_start
        print(f"  Epoch {epoch + 1}/{args.epochs} completed in {epoch_time:.1f}s")

    wall_time = time.time() - wall_start
    print(f"Training complete. Total wall time: {wall_time:.1f}s")

if __name__ == "__main__":
    main()
