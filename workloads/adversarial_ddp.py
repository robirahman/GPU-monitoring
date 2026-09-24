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

Interconnect ground truth (W15): add ``--comm-log data/x.jsonl`` to log every gradient
AllReduce (one JSONL line per DDP bucket: ``t, op, bytes_est, step, microstep,
is_sync_step``) and ``<stem>.steps.jsonl`` with one record per batch (``t, t_end, step,
microstep, is_sync_step, phase``; phase = train_sync | train_accum | train_local_nosync |
infer_forward). Without the flag nothing changes. Registry aliases:
``train_ddp_rare_sync_accum{4,8}``, ``train_ddp_dilute_1of{5,10}``.

CPU smoke (no CIFAR needed): ``--synthetic-data --backend gloo`` swaps the CIFAR loader
for random tensors and a small MLP so the comm log can be checked without GPUs.
"""

import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from interconnect_common import CommTrace, DistEnv, StopSignal, env_from_initialized


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


class _SyntheticLoader:
    """Random (image, label) batches; CPU-smoke replacement for the CIFAR loader."""

    def __init__(self, batch_size, n_batches=64):
        self.batch_size, self.n = batch_size, n_batches

    def __iter__(self):
        for _ in range(self.n):
            yield torch.randn(self.batch_size, 3, 32, 32), torch.randint(0, 10, (self.batch_size,))


class _Deadline:
    """Run-time budget for the training loops.

    Default (no --comm-log): the original per-rank ``time.time()`` checks, unchanged.
    With --comm-log: rank 0's clock decides and all ranks stop on the same batch
    (StopSignal). A per-rank check can straddle the deadline -- one rank exits while
    its peer waits in an AllReduce -- and that hang shows up on fast loops; interconnect
    runs are the ones we can't afford to lose.
    """

    def __init__(self, args):
        self.duration = args.duration
        self.t0 = time.time()
        self.sig = StopSignal(args.env, args.duration) if args.comm_log else None
        self.done = False

    def running(self):
        if self.sig is not None:
            return not self.done
        return time.time() - self.t0 < self.duration

    def hit(self, idx):
        if self.sig is not None:
            self.done = self.done or self.sig.should_stop(idx)
            return self.done
        return time.time() - self.t0 >= self.duration


class _NoSampler:
    def set_epoch(self, epoch):
        pass


def get_loader(args, batch_size, rank, world_size):
    if args.synthetic_data:
        return _SyntheticLoader(batch_size), _NoSampler()
    return get_cifar10_loader(batch_size, rank, world_size, train=False)


def build_model(args, device):
    """DDP-wrapped model + optional interconnect trace (see CommTrace)."""
    if args.synthetic_data:
        net = nn.Sequential(nn.Flatten(), nn.Linear(3072, 256), nn.ReLU(), nn.Linear(256, 10))
    else:
        net = torchvision.models.resnet50(weights=None, num_classes=10)
    net = net.to(device)
    if device.type == "cuda":
        model = DDP(net, device_ids=[device.index])
    else:
        model = DDP(net)
    trace = CommTrace(args.comm_log, args.env, model, step_path=args.step_log)
    return model, trace


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

    loader, sampler = get_loader(args, batch_size, rank, world_size)

    model, trace = build_model(args, device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    batch_idx = 0
    epoch = 0

    deadline = _Deadline(args)
    while deadline.running():
        sampler.set_epoch(epoch)
        for data, target in loader:
            if deadline.hit(batch_idx):
                break
            data, target = data.to(device), target.to(device)

            with trace.window(step=batch_idx, microstep=0, is_sync_step=True, phase="train_sync"):
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            total_samples += data.size(0) * world_size
            batch_idx += 1

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
    trace.close()


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

    loader, sampler = get_loader(args, batch_size, rank, world_size)

    model, trace = build_model(args, device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    train_batches = 0
    infer_batches = 0
    batch_idx = 0
    epoch = 0

    deadline = _Deadline(args)
    while deadline.running():
        sampler.set_epoch(epoch)
        for data, target in loader:
            if deadline.hit(batch_idx):
                break
            data, target = data.to(device), target.to(device)

            is_train = batch_idx % dilution == 0
            with trace.window(step=batch_idx, microstep=0, is_sync_step=is_train,
                              phase="train_sync" if is_train else "infer_forward"):
                if is_train:
                    # Training batch: all ranks train together
                    model.train()
                    output = model(data)
                    loss = criterion(output, target)
                    loss.backward()  # triggers allreduce via DDP
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    train_batches += 1
                else:
                    # Inference batch: no gradients, no communication
                    model.eval()
                    with torch.no_grad():
                        _ = model(data)
                    infer_batches += 1

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
    trace.close()


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

    loader, sampler = get_loader(args, batch_size, rank, world_size)

    model, trace = build_model(args, device)
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

    deadline = _Deadline(args)
    while deadline.running():
        sampler.set_epoch(epoch)
        for data, target in loader:
            if deadline.hit(batch_idx):
                break
            data, target = data.to(device), target.to(device)

            # This rank trains if batch_idx matches its offset
            is_my_train_batch = (batch_idx % (dilution * world_size)) == my_train_offset

            # is_sync_step is always False here: no_sync() means this strategy never
            # AllReduces, so its comm log is empty by construction.
            with trace.window(step=batch_idx, microstep=0, is_sync_step=False,
                              phase="train_local_nosync" if is_my_train_batch else "infer_forward"):
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
                else:
                    model.eval()
                    with torch.no_grad():
                        _ = model(data)
                    infer_batches += 1

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
    trace.close()


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

    loader, sampler = get_loader(args, batch_size, rank, world_size)

    model, trace = build_model(args, device)
    optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.0)
    criterion = nn.CrossEntropyLoss()

    wall_start = time.time()
    last_print = wall_start
    total_samples = 0
    micro_step = 0
    allreduce_count = 0
    epoch = 0

    deadline = _Deadline(args)
    while deadline.running():
        sampler.set_epoch(epoch)
        for data, target in loader:
            if deadline.hit(micro_step):
                break
            data, target = data.to(device), target.to(device)

            is_sync_step = ((micro_step + 1) % accum_steps == 0)

            with trace.window(step=micro_step, microstep=micro_step % accum_steps,
                              is_sync_step=is_sync_step,
                              phase="train_sync" if is_sync_step else "train_accum"):
                if is_sync_step:
                    # Normal forward+backward with DDP allreduce
                    output = model(data)
                    loss = criterion(output, target) / accum_steps
                    loss.backward()
                    optimizer.step()
                    optimizer.zero_grad(set_to_none=True)
                    allreduce_count += 1
                else:
                    # Accumulate without allreduce
                    with model.no_sync():
                        output = model(data)
                        loss = criterion(output, target) / accum_steps
                        loss.backward()

            total_samples += data.size(0) * world_size
            micro_step += 1

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
    trace.close()


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
    parser.add_argument("--comm-log", type=str, default="",
                        help="Rank0 JSONL: one line per gradient AllReduce bucket "
                             "(+ <stem>.steps.jsonl per-batch window truth)")
    parser.add_argument("--step-log", type=str, default="",
                        help="Override path for the per-batch step records")
    parser.add_argument("--backend", choices=["nccl", "gloo"], default="nccl",
                        help="gloo + --synthetic-data = CPU smoke test")
    parser.add_argument("--synthetic-data", action="store_true",
                        help="Random tensors + small MLP instead of CIFAR-10/ResNet-50")
    args = parser.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(backend=args.backend)
    if args.backend == "nccl":
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device("cpu")
    if args.comm_log:
        args.env = env_from_initialized(args.backend, device, local_rank)   # + gloo control group
    else:
        args.env = DistEnv(rank, world_size, local_rank, device, args.backend)

    STRATEGIES[args.strategy](args, device, rank, world_size)

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
