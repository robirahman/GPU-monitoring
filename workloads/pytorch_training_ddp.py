#!/usr/bin/env python3
"""Distributed Data Parallel (DDP) training workloads for multi-GPU telemetry.

Wraps existing training workloads (ResNet, GPT-2, BERT) in DDP for 1-8 GPU
distributed training. Designed for datacenter experiments on A100/H100/B100.

Launch via torchrun (NOT directly with python):
    torchrun --nproc_per_node=8 workloads/pytorch_training_ddp.py --model resnet18 --epochs 5
    torchrun --nproc_per_node=8 workloads/pytorch_training_ddp.py --model resnet50 --duration 300
    torchrun --nproc_per_node=2 workloads/pytorch_training_ddp.py --model resnet18 --epochs 3  # 2-GPU test
"""

import argparse
import os
import time

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
except ImportError:  # launched as `torchrun workloads/pytorch_training_ddp.py`
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


class SimpleMLP(nn.Module):
    """3-layer MLP for CIFAR-10."""

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
    elif name == "resnet50":
        return torchvision.models.resnet50(weights=None, num_classes=num_classes)
    elif name == "simple_mlp":
        return SimpleMLP(num_classes=num_classes)
    else:
        raise ValueError(f"Unknown model: {name}")


def main():
    parser = argparse.ArgumentParser(description="DDP training workload")
    parser.add_argument("--model", type=str, default="resnet18",
                        choices=["resnet18", "resnet50", "simple_mlp"])
    parser.add_argument("--epochs", type=int, default=None,
                        help="Number of epochs (mutually exclusive with --duration)")
    parser.add_argument("--duration", type=int, default=None,
                        help="Training duration in seconds (mutually exclusive with --epochs)")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-GPU batch size")
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--amp", action="store_true",
                        help="Enable automatic mixed precision")
    parser.add_argument("--data-dir", type=str, default="./data/cifar10")
    args = parser.parse_args()

    if args.epochs is None and args.duration is None:
        args.epochs = 5

    # DDP setup — torchrun sets these env vars
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))  # torchrun: this node's index

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"DDP Training: {args.model} on {world_size} GPUs")
        print(f"  Per-GPU batch: {args.batch_size}, Global batch: {args.batch_size * world_size}")
        print(f"  AMP: {args.amp}")
        if args.duration:
            print(f"  Duration: {args.duration}s")
        else:
            print(f"  Epochs: {args.epochs}")

    # Dataset with DistributedSampler
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    if args.model == "resnet50":
        transform = transforms.Compose([
            transforms.Resize(224),
            transform,
        ])
    train_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=True, download=(rank == 0), transform=transform
    )
    # Wait for rank 0 to download
    dist.barrier()

    sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, sampler=sampler,
        num_workers=2, pin_memory=True,
    )

    # Model with DDP
    model = build_model(args.model).to(device)
    model = DDP(model, device_ids=[local_rank])
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
    criterion = nn.CrossEntropyLoss()
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    # Training loop
    wall_start = time.time()
    # Collective deadline: see workloads/dist_deadline.py. Exiting this loop at
    # different iterations per rank kills the 2-node cell with ncclRemoteError.
    deadline = CollectiveDeadline(args.duration, device)
    model.train()
    epoch = 0
    global_step = 0  # monotonic across epochs, for STEP batch_idx

    while True:
        sampler.set_epoch(epoch)
        epoch_start = time.time()
        running_loss = 0.0
        correct = 0
        total = 0

        for data, target in train_loader:
            if deadline.expired():
                break

            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()

            with torch.amp.autocast("cuda", enabled=args.amp):
                output = model(data)
                loss = criterion(output, target)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            global_step += 1

            # Per-window ground truth (rank 0 only; synchronous DDP step).
            if rank == 0:
                emit_step("train", global_step,
                          tokens=args.batch_size * world_size, passes=1,
                          node=node_rank, rank=rank, loss=f"{loss.item():.4f}")

            running_loss += loss.item()
            _, predicted = output.max(1)
            total += target.size(0)
            correct += predicted.eq(target).sum().item()

        epoch_time = time.time() - epoch_start
        epoch += 1

        if rank == 0:
            avg_loss = running_loss / max(len(train_loader), 1)
            acc = 100.0 * correct / max(total, 1)
            print(f"  Epoch {epoch}: loss={avg_loss:.4f}, acc={acc:.1f}%, "
                  f"time={epoch_time:.1f}s")

        # Check stopping conditions
        if deadline.expired():
            break
        if args.epochs and epoch >= args.epochs:
            break

    wall_time = time.time() - wall_start
    if rank == 0:
        print(f"DDP training complete. {epoch} epochs, {wall_time:.1f}s wall time")

    sync_teardown()


if __name__ == "__main__":
    main()
