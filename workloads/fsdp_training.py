#!/usr/bin/env python3
"""FSDP (Fully Sharded Data Parallel) training -- NVLink stress workload.

Trains a randomly initialized transformer with FSDP FULL_SHARD to maximize
inter-GPU NVLink traffic. Every forward pass triggers AllGather on all params;
every backward pass triggers ReduceScatter on all gradients.

Launch via torchrun:
    torchrun --nproc_per_node=8 workloads/fsdp_training.py --size 7b --duration 300
    torchrun --nproc_per_node=8 workloads/fsdp_training.py --size 30b --duration 600
    torchrun --nproc_per_node=8 workloads/fsdp_training.py --size 70b --duration 600
"""

import argparse
import functools
import os
import sys
import time

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
    CPUOffload,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy

# ── Model configurations ────────────────────────────────────────────────────
# (hidden_dim, num_layers, num_heads, ffn_dim, vocab_size)

MODEL_CONFIGS = {
    "1b":  (2048, 24, 16, 8192,  32000),   # ~1.3B
    "7b":  (4096, 32, 32, 11008, 32000),   # ~6.7B
    "13b": (5120, 40, 40, 13824, 32000),   # ~13B
    "30b": (6656, 60, 52, 17920, 32000),   # ~30B
    "70b": (8192, 80, 64, 28672, 32000),   # ~70B
}

BATCH_DEFAULTS = {"1b": 8, "7b": 4, "13b": 2, "30b": 1, "70b": 1}


# ── Model definition ────────────────────────────────────────────────────────

class TransformerBlock(nn.Module):
    """Single transformer decoder block (pre-norm, self-attention + MLP)."""

    def __init__(self, hidden_dim, num_heads, ffn_dim):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, ffn_dim, bias=False),
            nn.GELU(),
            nn.Linear(ffn_dim, hidden_dim, bias=False),
        )

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        x = x + self.mlp(self.norm2(x))
        return x


class RandomTransformerLM(nn.Module):
    """Decoder-only transformer language model with random weights."""

    def __init__(self, config_name):
        super().__init__()
        hidden, n_layers, n_heads, ffn, vocab = MODEL_CONFIGS[config_name]
        self.vocab_size = vocab
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(2048, hidden)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, n_heads, ffn) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.lm_head(x)


# ── FSDP wrapping ───────────────────────────────────────────────────────────

def param_init_fn(module: nn.Module, device: torch.device):
    """Materialize meta-device parameters with random weights."""
    module.to_empty(device=device)
    for p in module.parameters(recurse=False):
        if p.requires_grad:
            nn.init.normal_(p, mean=0.0, std=0.02)
    for b in module.buffers(recurse=False):
        b.zero_()


def build_fsdp_model(model, device, activation_ckpt=True, cpu_offload=False):
    """Wrap model in FSDP with FULL_SHARD + bf16 mixed precision."""
    mp_policy = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.bfloat16,
        buffer_dtype=torch.bfloat16,
    )

    wrap_policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    offload = CPUOffload(offload_params=True) if cpu_offload else None

    # When CPU offloading, materialize params on CPU; FSDP moves them as needed
    init_device = torch.device("cpu") if cpu_offload else device

    fsdp_model = FSDP(
        model,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        mixed_precision=mp_policy,
        auto_wrap_policy=wrap_policy,
        cpu_offload=offload,
        device_id=device if not cpu_offload else None,
        param_init_fn=functools.partial(param_init_fn, device=init_device),
        limit_all_gathers=True,
    )

    if activation_ckpt:
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )
        non_reentrant_wrapper = functools.partial(
            checkpoint_wrapper, checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            fsdp_model,
            checkpoint_wrapper_fn=non_reentrant_wrapper,
            check_fn=lambda submodule: isinstance(submodule, TransformerBlock),
        )

    return fsdp_model


# ── Training loop ────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="FSDP NVLink stress training")
    parser.add_argument("--size", type=str, default="7b",
                        choices=list(MODEL_CONFIGS.keys()),
                        help="Model size preset (default: 7b)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Training duration in seconds (default: 300)")
    parser.add_argument("--batch-size", type=int, default=None,
                        help="Per-GPU micro-batch size (auto if omitted)")
    parser.add_argument("--seq-len", type=int, default=1024,
                        help="Sequence length (default: 1024)")
    parser.add_argument("--optimizer", type=str, default=None,
                        choices=["adamw", "sgd", "sgd_momentum"],
                        help="Override optimizer (default: auto based on model size)")
    parser.add_argument("--cpu-offload", action="store_true",
                        help="Offload optimizer states to CPU")
    parser.add_argument("--no-activation-ckpt", action="store_true",
                        help="Disable activation checkpointing")
    args = parser.parse_args()

    # Distributed setup
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    batch_size = args.batch_size or BATCH_DEFAULTS[args.size]
    hidden, n_layers, n_heads, ffn, vocab = MODEL_CONFIGS[args.size]

    if rank == 0:
        print(f"FSDP Training: {args.size} model on {world_size} GPUs")
        print(f"  Hidden: {hidden}, Layers: {n_layers}, Heads: {n_heads}, FFN: {ffn}")
        print(f"  Per-GPU batch: {batch_size}, Seq len: {args.seq_len}")
        print(f"  Global batch: {batch_size * world_size}")
        print(f"  Mixed precision: bf16, Sharding: FULL_SHARD")
        print(f"  Activation checkpointing: {not args.no_activation_ckpt}")
        print(f"  CPU offload: {args.cpu_offload}")
        print(f"  Duration: {args.duration}s")

    # Build model on meta device (no memory until FSDP materializes shards)
    with torch.device("meta"):
        model = RandomTransformerLM(args.size)

    if rank == 0:
        param_count = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {param_count:,} ({param_count / 1e9:.1f}B)")

    # Wrap in FSDP -- materializes only local shard on each GPU
    fsdp_model = build_fsdp_model(
        model, device,
        activation_ckpt=not args.no_activation_ckpt,
        cpu_offload=args.cpu_offload,
    )

    # Select optimizer. --optimizer overrides the automatic selection.
    # Auto logic: AdamW for <=13b, SGD+momentum for 30b, plain SGD for 70b.
    # On GPUs with >80GB (e.g. H200-141GB, B200-183GB), callers can override
    # to AdamW/SGD+momentum for larger models that would OOM on 80GB cards.
    opt_choice = args.optimizer
    if opt_choice is None:
        if args.size == "70b":
            opt_choice = "sgd"
        elif args.size == "30b":
            opt_choice = "sgd_momentum"
        else:
            opt_choice = "adamw"

    if opt_choice == "adamw":
        optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
    elif opt_choice == "sgd_momentum":
        optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=1e-3, momentum=0.9)
    else:
        optimizer = torch.optim.SGD(fsdp_model.parameters(), lr=1e-3)

    if rank == 0:
        mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
        print(f"  Optimizer: {opt_choice}")
        print(f"  Post-init GPU memory: {mem_gb:.1f} GB")
        print(f"  Training started...")

    # Training loop with synthetic data
    wall_start = time.time()
    step = 0

    while time.time() - wall_start < args.duration:
        # Random token ids -- no real dataset needed
        input_ids = torch.randint(0, vocab, (batch_size, args.seq_len), device=device)

        try:
            # Forward: AllGather params for each FSDP unit
            logits = fsdp_model(input_ids)

            # Causal LM loss
            shift_logits = logits[:, :-1, :].contiguous()
            shift_labels = input_ids[:, 1:].contiguous()
            loss = nn.functional.cross_entropy(
                shift_logits.view(-1, shift_logits.size(-1)),
                shift_labels.view(-1),
            )

            # Backward: ReduceScatter grads for each FSDP unit
            loss.backward()
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1

        except torch.cuda.OutOfMemoryError:
            if rank == 0:
                mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
                print(f"OOM at step {step}! Peak memory: {mem_gb:.1f} GB")
                print(f"Try: smaller --size, --cpu-offload, or smaller --batch-size/--seq-len")
            dist.destroy_process_group()
            sys.exit(1)

        if rank == 0 and step % 5 == 0:
            elapsed = time.time() - wall_start
            tokens_per_sec = (step * batch_size * world_size * args.seq_len) / elapsed
            mem_gb = torch.cuda.max_memory_allocated(device) / 1e9
            print(f"  Step {step}: loss={loss.item():.4f}, "
                  f"{tokens_per_sec:.0f} tok/s, "
                  f"peak_mem={mem_gb:.1f}GB, "
                  f"elapsed={elapsed:.0f}s/{args.duration}s")

    wall_time = time.time() - wall_start
    if rank == 0:
        total_tokens = step * batch_size * world_size * args.seq_len
        print(f"\nFSDP training complete: {step} steps in {wall_time:.1f}s")
        print(f"  Total tokens: {total_tokens:,}")
        print(f"  Throughput: {total_tokens / wall_time:.0f} tok/s")
        print(f"  Peak GPU memory: {torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
