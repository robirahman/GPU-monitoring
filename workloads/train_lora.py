#!/usr/bin/env python3
"""HONEST training with LoRA adapters -- the quietest real training there is.

WHY THIS EXISTS. Held out by family, the detector recognises 0.5% of FSDP
training windows and 4% of DDP windows, because the corpus contains exactly two
training programs and holding one out leaves it with one. Every training cell we
own is also LOUD: full-parameter updates, so the gradient all-reduce moves the
whole model every step, and the project's own measurement is that FSDP's
inter-node traffic is ~1000x sharded inference's.

LoRA inverts that. The base weights are frozen, so only the adapters -- well
under 1% of the parameters -- have gradients, and the gradient all-reduce shrinks
by two or three orders of magnitude while the forward and backward FLOPs stay
almost unchanged. A LoRA fine-tune is therefore real training whose interconnect
signature looks like inference, which is exactly the region the detector must not
concede. If a detector leaning on NIC volume calls this "not training", that is a
finding about the detector, not a failure of the cell.

It is also the single most common real fine-tuning workload in the wild, so a
detector that misses it misses most of what it would actually meet.

LABELLING: cells are named s16_train_lora* -- contains "train", so
threeway_label() routes to ml_training, which is correct: gradients are computed
and an optimizer steps. No disguise token, so it stays in the honest pool.
Verified in classifier/rebuttal_analysis/test_cell_label_routing.py. STEP lines
carry kind="train".

This is NOT an evasion cell. Nothing here is trying to hide: LoRA is the standard
way to fine-tune, and the low traffic is a property of the method, not a choice
made to defeat a detector. It belongs in the honest corpus.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/train_lora.py \
        --size 7b --rank 16 --duration 600
"""
import argparse
import math
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads.fsdp_training import MODEL_CONFIGS, RandomTransformerLM
except ImportError:
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown
    from fsdp_training import MODEL_CONFIGS, RandomTransformerLM


class LoRALinear(nn.Module):
    """Frozen base Linear with a trainable low-rank update, as in Hu et al.

    Deliberately hand-rolled rather than pulled from peft: the base model here is
    the repo's RandomTransformerLM (see fsdp_training.py), not a HF model, so
    peft's injection machinery does not apply. The arithmetic is the same --
    y = W0 x + (alpha/r) B A x, with W0 frozen -- and keeping it local means this
    cell has no dependency that bootstrap might not have installed.
    """

    def __init__(self, base: nn.Linear, r: int, alpha: int):
        super().__init__()
        self.base = base
        for p in self.base.parameters():
            p.requires_grad_(False)
        self.A = nn.Parameter(torch.zeros(r, base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, r))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))   # B stays zero: the
        self.scale = alpha / r                             # adapter starts as a no-op

    def forward(self, x):
        return self.base(x) + torch.nn.functional.linear(
            torch.nn.functional.linear(x, self.A), self.B) * self.scale


def inject_lora(model, r, alpha):
    """Replace every MLP Linear with a LoRA-wrapped one. Returns (n_lora, n_total)."""
    for block in model.blocks:
        for i, child in enumerate(block.mlp):
            if isinstance(child, nn.Linear):
                block.mlp[i] = LoRALinear(child, r, alpha)
    for n, p in model.named_parameters():
        # Everything outside an adapter is frozen. `.A`/`.B` are the adapters.
        p.requires_grad_(n.endswith(".A") or n.endswith(".B"))
    train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    return train, total


def main():
    ap = argparse.ArgumentParser(description="Honest LoRA fine-tuning (real training)")
    ap.add_argument("--size", default="7b", choices=list(MODEL_CONFIGS.keys()))
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    ap.add_argument("--alpha", type=int, default=32, help="LoRA alpha")
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    # RandomTransformerLM takes the config NAME, not the unpacked dims.
    vocab = MODEL_CONFIGS[args.size][4]
    if args.seq_len > 2048:
        raise SystemExit(f"--seq-len {args.seq_len} exceeds the model's learned "
                         f"positional embedding table (2048 rows)")
    model = RandomTransformerLM(args.size)
    n_train, n_total = inject_lora(model, args.rank, args.alpha)
    model = model.to(device)

    if rank == 0:
        print(f"LoRA fine-tune (HONEST training): {args.size}, {world} ranks")
        print(f"  trainable {n_train/1e6:.2f}M of {n_total/1e9:.2f}B params "
              f"({100*n_train/n_total:.3f}%) -- that ratio IS the gradient "
              f"all-reduce volume relative to a full fine-tune")
        print(f"  rank={args.rank} alpha={args.alpha} batch={args.batch_size} "
              f"seq={args.seq_len} duration={args.duration}s")

    # DDP, not FSDP: with 99%+ of parameters frozen there is nothing to shard
    # that would pay for itself, and DDP is what a real LoRA job uses. Only the
    # adapters are in the bucket, which is the whole point.
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    opt = torch.optim.AdamW([p for p in ddp.parameters() if p.requires_grad],
                            lr=args.lr)
    lossf = nn.CrossEntropyLoss()

    deadline = CollectiveDeadline(args.duration, device)
    step = 0
    tokens = 0
    last = time.time()
    while not deadline.expired():
        x = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
        y = torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device)
        out = ddp(x)
        loss = lossf(out.reshape(-1, vocab).float(), y.reshape(-1))
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        tokens += args.batch_size * args.seq_len
        if rank == 0:
            emit_step("train", step, tokens=args.batch_size * args.seq_len,
                      passes=1, node=node_rank, rank=rank, method="lora")
            now = time.time()
            if now - last >= 30:
                el = deadline.elapsed
                print(f"  t={el:6.0f}s  step={step}  loss={loss.item():.3f}  "
                      f"{tokens*world/max(el,1):.0f} tok/s (global)", flush=True)
                last = now

    if rank == 0:
        print(f"LoRA done: {step} steps, {tokens*world} tokens, "
              f"{deadline.elapsed:.0f}s", flush=True)
    sync_teardown()


if __name__ == "__main__":
    main()
