#!/usr/bin/env python3
"""Multi-GPU adversarial evasion strategies for datacenter experiments.

These extend single-GPU strategies K and L to distributed settings, plus
introduce DDP-specific evasion techniques that exploit gradient communication
patterns.

Strategies:
    K-DDP:      Online learning with DDP (forward+backward+allreduce per batch)
    L-DDP:      Diluted training with DDP (all ranks train on same batches)
    L-stagger:  Staggered diluted training (each rank trains on different batches,
                spreading allreduce across time)
    K-accum:    Online learning with gradient accumulation (allreduce every M steps
                instead of every step, reducing communication frequency)

Launch via torchrun:
    torchrun --nproc_per_node=8 workloads/adversarial_ddp.py --strategy K --duration 300
    torchrun --nproc_per_node=8 workloads/adversarial_ddp.py --strategy L --dilution 5 --duration 300
    torchrun --nproc_per_node=8 workloads/adversarial_ddp.py --strategy L-stagger --dilution 5 --duration 300
    torchrun --nproc_per_node=8 workloads/adversarial_ddp.py --strategy K-accum --accum 4 --duration 300
"""

import argparse
import os
import time

try:
    from workloads._step import emit_step
except ImportError:  # launched as `torchrun workloads/adversarial_ddp.py`
    from _step import emit_step

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler


def get_cifar10_loader(batch_size, rank, world_size, train=False):
    """CIFAR-10 loader with DistributedSampler."""
    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    dataset = torchvision.datasets.CIFAR10(
        root="data/cifar10", train=train, download=(rank == 0), transform=transform
    )
    dist.barrier()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    return DataLoader(dataset, batch_size=batch_size, sampler=sampler,
                      num_workers=2, pin_memory=True), sampler


# ── Strategy K-DDP: Online Learning, Distributed ────────────────────────────

def strategy_k_ddp(args, device, rank, world_size):
    """Distributed online learning — identical to DDP training but disguised.

    Every rank does forward + backward + allreduce + optimizer step on every batch.
    Uses ResNet-50, SGD with no momentum. Structurally identical to
    pytorch_training_ddp.py except uses test set (like inference.py) and SGD
    with no momentum (zero optimizer state).
    """
    batch_size = args.batch_size
    if rank == 0:
        print(f"Strategy K-DDP: Online Learning, {world_size} GPUs")
        print(f"  Per-GPU batch: {batch_size}, Duration: {args.duration}s")

    loader, sampler = get_cifar10_loader(batch_size, rank, world_size, train=False)

    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    model = DDP(model, device_ids=[device.index])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    step = 0
    epoch = 0

    while time.time() - wall_start < args.duration:
        sampler.set_epoch(epoch)
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break
            data, target = data.to(device), target.to(device)

            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            total_samples += data.size(0) * world_size
            step += 1
            # Online learning: every batch is a real training step (rank 0).
            if rank == 0:
                emit_step("train", step, tokens=data.size(0) * world_size,
                          passes=1, node=args.node_rank, rank=rank)

        epoch += 1
        now = time.time()
        if rank == 0 and now - last_print >= 30:
            elapsed = now - wall_start
            throughput = total_samples / elapsed
            print(f"  {elapsed:.0f}s: {total_samples} samples, {throughput:.0f} img/s")
            last_print = now

    if rank == 0:
        wall_time = time.time() - wall_start
        print(f"K-DDP complete. {total_samples} samples, {wall_time:.1f}s")


# ── Strategy L-DDP: Diluted Training, Synchronized ──────────────────────────

def strategy_l_ddp(args, device, rank, world_size):
    """Diluted training — all ranks train on same batch indices.

    Every N-th batch: all ranks do forward + backward + allreduce + step.
    Other batches: all ranks do forward only (inference mode).
    """
    dilution = args.dilution
    batch_size = args.batch_size
    if rank == 0:
        print(f"Strategy L-DDP: Diluted Training (1/{dilution}), {world_size} GPUs")
        print(f"  Per-GPU batch: {batch_size}, Duration: {args.duration}s")

    loader, sampler = get_cifar10_loader(batch_size, rank, world_size, train=False)

    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    model = DDP(model, device_ids=[device.index])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    train_batches = 0
    infer_batches = 0
    batch_idx = 0
    epoch = 0

    while time.time() - wall_start < args.duration:
        sampler.set_epoch(epoch)
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break
            data, target = data.to(device), target.to(device)

            if batch_idx % dilution == 0:
                # Training batch: all ranks train together
                model.train()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()  # triggers allreduce via DDP
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_batches += 1
                kind = "train"
            else:
                # Inference batch: no gradients, no communication
                model.eval()
                with torch.no_grad():
                    _ = model(data)
                infer_batches += 1
                kind = "infer"

            # All ranks synchronized on the same batches, so rank 0's per-batch
            # train/infer label is representative of the whole job.
            if rank == 0:
                emit_step(kind, batch_idx, tokens=data.size(0) * world_size,
                          passes=1, node=args.node_rank, rank=rank)

            total_samples += data.size(0) * world_size
            batch_idx += 1

        epoch += 1
        now = time.time()
        if rank == 0 and now - last_print >= 30:
            elapsed = now - wall_start
            throughput = total_samples / elapsed
            print(f"  {elapsed:.0f}s: {total_samples} samples, {throughput:.0f} img/s, "
                  f"train={train_batches}, infer={infer_batches}")
            last_print = now

    if rank == 0:
        wall_time = time.time() - wall_start
        total_b = train_batches + infer_batches
        frac = train_batches / max(total_b, 1)
        print(f"L-DDP complete. train_frac={frac:.1%}, {wall_time:.1f}s")


# ── Strategy L-stagger: Staggered Diluted Training ──────────────────────────

def strategy_l_stagger(args, device, rank, world_size):
    """Staggered diluted training — each rank trains on different batches.

    Rank r trains on batch indices where (batch_idx % (dilution * world_size)) == r.
    This spreads allreduce communication across time steps rather than having
    synchronized bursts. A detector looking for correlated GPU activity sees
    noise instead of periodic allreduce patterns.

    Note: requires careful handling — DDP expects all ranks to participate in
    backward pass together. We use model.no_sync() for non-training batches
    and manually accumulate/synchronize.
    """
    dilution = args.dilution
    batch_size = args.batch_size
    if rank == 0:
        print(f"Strategy L-stagger: Staggered Dilution (1/{dilution}), {world_size} GPUs")
        print(f"  Each rank trains on different batch offsets")

    loader, sampler = get_cifar10_loader(batch_size, rank, world_size, train=False)

    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    model = DDP(model, device_ids=[device.index])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    train_batches = 0
    infer_batches = 0
    batch_idx = 0
    epoch = 0

    # Each rank's training offset: rank 0 trains batch 0, rank 1 trains batch 1, etc.
    my_train_offset = rank

    while time.time() - wall_start < args.duration:
        sampler.set_epoch(epoch)
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break
            data, target = data.to(device), target.to(device)

            # This rank trains if batch_idx matches its offset
            is_my_train_batch = (batch_idx % (dilution * world_size)) == my_train_offset

            if is_my_train_batch:
                # Training batch — but only this rank is training.
                # Use no_sync to avoid allreduce, apply gradients locally.
                model.train()
                with model.no_sync():
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                train_batches += 1
                kind = "train"
            else:
                model.eval()
                with torch.no_grad():
                    _ = model(data)
                infer_batches += 1
                kind = "infer"

            # Emit from EVERY rank: ranks train on staggered batches, so rank 0
            # is not representative -- the per-rank ground truth is the point.
            emit_step(kind, batch_idx, tokens=data.size(0) * world_size,
                      passes=1, node=args.node_rank, rank=rank)

            total_samples += data.size(0) * world_size
            batch_idx += 1

        epoch += 1
        now = time.time()
        if rank == 0 and now - last_print >= 30:
            elapsed = now - wall_start
            throughput = total_samples / elapsed
            print(f"  {elapsed:.0f}s: {total_samples} samples, {throughput:.0f} img/s, "
                  f"train={train_batches}, infer={infer_batches}")
            last_print = now

    if rank == 0:
        wall_time = time.time() - wall_start
        total_b = train_batches + infer_batches
        frac = train_batches / max(total_b, 1)
        print(f"L-stagger complete. per-rank train_frac={frac:.1%}, {wall_time:.1f}s")


# ── Strategy K-accum: Online Learning with Gradient Accumulation ─────────────

def strategy_k_accum(args, device, rank, world_size):
    """Online learning with gradient accumulation to hide allreduce frequency.

    Accumulates gradients over M microbatches before allreduce. This reduces
    the frequency of inter-GPU communication by M, making it harder for
    NVLink-based detectors to identify training.

    Normal DDP: allreduce every batch → periodic NVLink bursts.
    K-accum(M=4): allreduce every 4th batch → 4x less NVLink traffic.
    K-accum(M=8): allreduce every 8th batch → 8x less NVLink traffic.
    """
    accum_steps = args.accum
    batch_size = args.batch_size
    if rank == 0:
        print(f"Strategy K-accum: Online Learning + Accumulation (M={accum_steps}), "
              f"{world_size} GPUs")
        print(f"  Per-GPU batch: {batch_size}, Effective batch: "
              f"{batch_size * world_size * accum_steps}")

    loader, sampler = get_cifar10_loader(batch_size, rank, world_size, train=False)

    model = torchvision.models.resnet50(weights=None, num_classes=10).to(device)
    model = DDP(model, device_ids=[device.index])
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    micro_step = 0
    allreduce_count = 0
    epoch = 0

    while time.time() - wall_start < args.duration:
        sampler.set_epoch(epoch)
        for data, target in loader:
            if time.time() - wall_start >= args.duration:
                break
            data, target = data.to(device), target.to(device)

            is_sync_step = ((micro_step + 1) % accum_steps == 0)

            if is_sync_step:
                # Normal forward+backward with DDP allreduce
                output = model(data)
                loss = criterion(output, target) / accum_steps
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                allreduce_count += 1
                phase = "sync"
            else:
                # Accumulate without allreduce
                with model.no_sync():
                    output = model(data)
                    loss = criterion(output, target) / accum_steps
                    loss.backward()
                phase = "accum"

            total_samples += data.size(0) * world_size
            micro_step += 1
            # All micro-steps are training compute; phase marks which ones
            # trigger the cross-rank allreduce burst.
            if rank == 0:
                emit_step("train", micro_step, tokens=data.size(0) * world_size,
                          passes=1, node=args.node_rank, rank=rank, phase=phase)

        epoch += 1
        now = time.time()
        if rank == 0 and now - last_print >= 30:
            elapsed = now - wall_start
            throughput = total_samples / elapsed
            print(f"  {elapsed:.0f}s: {total_samples} samples, {throughput:.0f} img/s, "
                  f"allreduces={allreduce_count}")
            last_print = now

    if rank == 0:
        wall_time = time.time() - wall_start
        print(f"K-accum complete. {allreduce_count} allreduces in {micro_step} steps "
              f"({allreduce_count/max(micro_step,1):.1%} sync rate), {wall_time:.1f}s")


# ── Main ─────────────────────────────────────────────────────────────────────

STRATEGIES = {
    "K": strategy_k_ddp,
    "L": strategy_l_ddp,
    "L-stagger": strategy_l_stagger,
    "K-accum": strategy_k_accum,
}


def main():
    parser = argparse.ArgumentParser(description="Multi-GPU adversarial strategies")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=list(STRATEGIES.keys()))
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Per-GPU batch size")
    parser.add_argument("--dilution", type=int, default=5,
                        help="L/L-stagger: train on 1/N batches")
    parser.add_argument("--accum", type=int, default=4,
                        help="K-accum: accumulate over M microbatches")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    args.node_rank = int(os.environ.get("GROUP_RANK", 0))  # this node's index

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    STRATEGIES[args.strategy](args, device, rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
