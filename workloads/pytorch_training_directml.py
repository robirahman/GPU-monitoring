"""PyTorch training workload — DirectML port of pytorch_training.py.

Identical model definitions and optimizer settings; only the device
selection and synchronization paths differ. Uses ``torch_directml`` to
target the DirectML adapter (the dGPU on Windows-AMD by default).

Usage:
    python pytorch_training_directml.py --model resnet18 --epochs 10 --batch-size 128
    python pytorch_training_directml.py --model resnet18 --epochs 10 --batch-size 128 --bf16
    python pytorch_training_directml.py --model simple_mlp --epochs 20 --batch-size 256

Note on AMP: torch-directml supports BF16 manual casts but does NOT have a
working ``torch.amp.autocast(device_type='privateuseone')`` path as of late
2025. We use ``--bf16`` to manually cast the model + inputs to bfloat16,
which is the closest analogue to the CUDA --amp path.
"""

import argparse
import time

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms

from _dml_device import describe, get_device, sync


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
    if name == "simple_mlp":
        return SimpleMLP(num_classes=num_classes)
    raise ValueError(f"Unknown model: {name}")


def main():
    ap = argparse.ArgumentParser(description="PyTorch training (DirectML)")
    ap.add_argument("--model", choices=["resnet18", "simple_mlp"], default="resnet18")
    ap.add_argument("--dataset", choices=["cifar10"], default="cifar10")
    ap.add_argument("--epochs", type=int, default=5)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--bf16", action="store_true",
                    help="Manual BF16 cast (DirectML AMP isn't supported)")
    ap.add_argument("--num-workers", type=int, default=2,
                    help="DataLoader workers (Windows: keep <= 4, fork is unavailable)")
    ap.add_argument("--data-dir", type=str, default="./data")
    args = ap.parse_args()

    device = get_device("dml")
    print(f"Device: {describe(device)}")
    print(f"Config: model={args.model} epochs={args.epochs} batch={args.batch_size} "
          f"lr={args.lr} bf16={args.bf16}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    trainset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=True, transform=transform,
    )
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, pin_memory=False,  # pin_memory is CUDA-only
    )

    model = build_model(args.model).to(device)
    if args.bf16:
        model = model.to(dtype=torch.bfloat16)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()

    print(f"\nTraining {args.epochs} epoch(s) on CIFAR-10 ({len(trainset)} samples)...")
    wall_start = time.time()
    for epoch in range(args.epochs):
        ep_start = time.time()
        running_loss = 0.0
        n_batches = 0
        for inputs, labels in trainloader:
            inputs = inputs.to(device)
            labels = labels.to(device)
            if args.bf16:
                inputs = inputs.to(dtype=torch.bfloat16)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.detach().to("cpu"))
            n_batches += 1
        sync(device)
        ep_time = time.time() - ep_start
        print(f"  epoch {epoch + 1}/{args.epochs}: loss={running_loss / n_batches:.4f} "
              f"time={ep_time:.1f}s ({n_batches} batches)")

    total = time.time() - wall_start
    print(f"\nTraining complete. Total time: {total:.1f}s")


if __name__ == "__main__":
    import os, sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
