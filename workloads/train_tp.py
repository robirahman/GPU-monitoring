#!/usr/bin/env python3
"""Megatron-style tensor-parallel training -- many small collectives per step.

Why this workload exists
------------------------
The corpus's training families are separated by HOW MUCH they communicate; none
of them is separated by how OFTEN. Every one issues its collectives at
step granularity:

    FSDP          per-layer all-gather + reduce-scatter, sized by PARAMETERS
    DDP           one all-reduce per step, sized by PARAMETERS
    local SGD     one all-reduce per H steps, sized by PARAMETERS
    MoE           two all-to-alls per layer, sized by TOKENS

Tensor parallelism is the other axis. Megatron splits each weight matrix across
ranks, so a single forward pass needs an all-reduce after EVERY attention block
and EVERY MLP -- two per layer forward, two per layer backward -- and each one
is sized by the ACTIVATION (batch x seq x hidden), not by the model. At 24
layers that is ~96 collectives per step instead of one, each a few megabytes.

High message rate at small message size is a regime the corpus has never
sampled, and it is the one where the per-window aggregates most plausibly wash
out: a hundred small bursts inside a 30-second window average into something
much flatter than FSDP's periodic multi-GB spikes, which is exactly the shape
the shape-only Stage-2 mask is supposed to survive on.

The `f` and `g` operators below are the canonical Megatron pair: identity in one
direction, all-reduce in the other. Getting them backwards is not a crash, it is
a model that trains slightly wrong while producing entirely plausible telemetry,
so scripts/test_tp_layers.py checks the TP result against a dense reference.

    torchrun --nproc-per-node=8 workloads/train_tp.py --tp-size 16   # TP spans both nodes
    torchrun --nproc-per-node=8 workloads/train_tp.py --tp-size 8    # TP in-node, DP across

--tp-size 8 is the arrangement a real deployment uses (TP over NVLink, data
parallel over the NIC) and --tp-size 16 is the one that puts every activation
collective on the wire. They look completely different from outside, and the
corpus should contain both.
"""

import argparse
import os
import sys
import time

try:
    from workloads._step import emit_step
    from workloads._seed import add_seed_argument, record_seed, record_progress
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
except ImportError:  # launched as `torchrun workloads/train_tp.py`
    from _step import emit_step
    from _seed import add_seed_argument, record_seed, record_progress
    from dist_deadline import CollectiveDeadline, sync_teardown

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist

# Bytes pushed through TP collectives this process, accumulated by the autograd
# operators themselves so the backward pass is counted where it actually happens
# rather than assumed to mirror the forward.
_TP_BYTES = [0]


def tp_bytes():
    return _TP_BYTES[0]


class _CopyToTP(torch.autograd.Function):
    """Megatron's `f`: identity forward, all-reduce backward.

    Sits at the INPUT of a column-parallel layer. Each rank computes a different
    slice of the output from the same input, so in backward each rank holds a
    partial gradient with respect to that shared input and they must be summed.
    """

    @staticmethod
    def forward(ctx, x, group):
        ctx.group = group
        return x

    @staticmethod
    def backward(ctx, grad):
        grad = grad.contiguous()
        dist.all_reduce(grad, group=ctx.group)
        _TP_BYTES[0] += grad.numel() * grad.element_size()
        return grad, None


class _ReduceFromTP(torch.autograd.Function):
    """Megatron's `g`: all-reduce forward, identity backward.

    Sits at the OUTPUT of a row-parallel layer. Each rank has computed a partial
    sum over its slice of the contraction, so the forward must add them; the
    gradient with respect to a partial sum is the incoming gradient unchanged.
    """

    @staticmethod
    def forward(ctx, x, group):
        x = x.contiguous()
        dist.all_reduce(x, group=group)
        _TP_BYTES[0] += x.numel() * x.element_size()
        return x

    @staticmethod
    def backward(ctx, grad):
        return grad, None


class ColumnParallelLinear(nn.Module):
    """y = x A, with A split along its OUTPUT dimension. Output stays sharded."""

    def __init__(self, in_features, out_features, tp_size, tp_rank, group):
        super().__init__()
        assert out_features % tp_size == 0, \
            f"out_features {out_features} not divisible by tp_size {tp_size}"
        self.group = group
        self.out_per_rank = out_features // tp_size
        self.weight = nn.Parameter(torch.empty(self.out_per_rank, in_features))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return F.linear(_CopyToTP.apply(x, self.group), self.weight)


class RowParallelLinear(nn.Module):
    """y = x A, with A split along its INPUT dimension. Input must be sharded."""

    def __init__(self, in_features, out_features, tp_size, tp_rank, group):
        super().__init__()
        assert in_features % tp_size == 0, \
            f"in_features {in_features} not divisible by tp_size {tp_size}"
        self.group = group
        self.in_per_rank = in_features // tp_size
        self.weight = nn.Parameter(torch.empty(out_features, self.in_per_rank))
        nn.init.normal_(self.weight, std=0.02)

    def forward(self, x):
        return _ReduceFromTP.apply(F.linear(x, self.weight), self.group)


class TPAttention(nn.Module):
    """Self-attention with the heads split across the TP group."""

    def __init__(self, hidden, n_heads, tp_size, tp_rank, group):
        super().__init__()
        assert n_heads % tp_size == 0, \
            f"n_heads {n_heads} not divisible by tp_size {tp_size}"
        self.heads_per_rank = n_heads // tp_size
        self.head_dim = hidden // n_heads
        self.qkv = ColumnParallelLinear(hidden, 3 * hidden, tp_size, tp_rank, group)
        self.proj = RowParallelLinear(hidden, hidden, tp_size, tp_rank, group)

    def forward(self, x):
        B, T, _ = x.shape
        qkv = self.qkv(x)                       # (B, T, 3*hidden/tp)
        q, k, v = qkv.chunk(3, dim=-1)
        shape = (B, T, self.heads_per_rank, self.head_dim)
        q = q.view(shape).transpose(1, 2)
        k = k.view(shape).transpose(1, 2)
        v = v.view(shape).transpose(1, 2)
        o = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        o = o.transpose(1, 2).reshape(B, T, self.heads_per_rank * self.head_dim)
        return self.proj(o)


class TPMLP(nn.Module):
    def __init__(self, hidden, ffn, tp_size, tp_rank, group):
        super().__init__()
        self.fc1 = ColumnParallelLinear(hidden, ffn, tp_size, tp_rank, group)
        self.fc2 = RowParallelLinear(ffn, hidden, tp_size, tp_rank, group)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class TPBlock(nn.Module):
    def __init__(self, hidden, n_heads, ffn, tp_size, tp_rank, group):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = TPAttention(hidden, n_heads, tp_size, tp_rank, group)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = TPMLP(hidden, ffn, tp_size, tp_rank, group)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        return x + self.mlp(self.norm2(x))


class TPTransformerLM(nn.Module):
    def __init__(self, hidden, n_layers, n_heads, ffn, vocab, tp_size, tp_rank,
                 group):
        super().__init__()
        self.vocab_size = vocab
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(4096, hidden)
        self.blocks = nn.ModuleList([
            TPBlock(hidden, n_heads, ffn, tp_size, tp_rank, group)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        # Column-parallel head: the vocab dimension is the big one, and leaving
        # the logits sharded avoids an all-gather of (B, T, vocab). The loss is
        # computed on this rank's vocab slice, which is enough to drive the
        # backward pass -- this is a telemetry workload, not a model we ship.
        self.lm_head = ColumnParallelLinear(hidden, vocab, tp_size, tp_rank, group)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.norm(x))


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--hidden", type=int, default=2048)
    p.add_argument("--layers", type=int, default=24)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ffn", type=int, default=8192)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--tp-size", type=int, default=0,
                   help="ranks per tensor-parallel group; 0 means the whole "
                        "world. Anything smaller leaves the remaining ranks as "
                        "data-parallel replicas.")
    p.add_argument("--lr", type=float, default=1e-4)
    add_seed_argument(p)
    return p.parse_args()


def main():
    args = parse_args()
    record_seed(args.seed)
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    local_world = int(os.environ.get("LOCAL_WORLD_SIZE", 8)) or 8
    node_rank = int(os.environ.get("GROUP_RANK", os.environ.get("NODE_RANK", 0)))
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    tp_size = args.tp_size or world_size
    if world_size % tp_size:
        raise SystemExit(f"world_size {world_size} not divisible by tp_size {tp_size}")
    n_dp = world_size // tp_size
    tp_rank = rank % tp_size
    dp_rank = rank // tp_size

    # Contiguous ranks share a TP group, so with tp_size == local_world the group
    # is exactly one node and every TP collective stays on NVLink. That is the
    # arrangement a real deployment picks, and the reason --tp-size is the knob
    # that decides whether this cell is loud on the NIC or silent on it.
    tp_group, dp_group = None, None
    for i in range(n_dp):
        g = dist.new_group(ranks=list(range(i * tp_size, (i + 1) * tp_size)))
        if i == dp_rank:
            tp_group = g
    for j in range(tp_size):
        g = dist.new_group(ranks=[j + i * tp_size for i in range(n_dp)])
        if j == tp_rank:
            dp_group = g

    # The sharded weights must DIFFER across the TP group, or every rank holds
    # the same slice and the tensor parallelism is decorative -- the collectives
    # still fire at the right sizes, so the telemetry would look exactly right
    # while the model was a replicated rank-1 stub. Seeding by tp_rank (not by
    # global rank) gives each shard different values while keeping the two DP
    # replicas of a given tp_rank identical, which is what data parallelism
    # requires.
    torch.manual_seed(args.seed + tp_rank)
    model = TPTransformerLM(args.hidden, args.layers, args.heads, args.ffn,
                            args.vocab, tp_size, tp_rank, tp_group).to(device)
    # Replicated parameters (embeddings, norms) must AGREE across the TP group,
    # so broadcast just those back into line from tp_rank 0.
    for mod in model.modules():
        if isinstance(mod, (ColumnParallelLinear, RowParallelLinear)):
            continue
        for p in mod.parameters(recurse=False):
            dist.broadcast(p.data, src=dp_rank * tp_size, group=tp_group)
    torch.manual_seed(args.seed + rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"Megatron TP: shard params={n_par/1e6:.0f}M/rank hidden={args.hidden} "
              f"layers={args.layers} heads={args.heads} tp_size={tp_size} "
              f"dp={n_dp} world_size={world_size} "
              f"collectives/step~{4 * args.layers + 2}", flush=True)

    deadline = CollectiveDeadline(args.duration, device)
    wall_start = time.time()
    step = 0
    tokens = 0
    while not deadline.expired():
        input_ids = torch.randint(0, args.vocab, (args.batch_size, args.seq_len),
                                  device=device)
        logits = model(input_ids)
        # Loss on this rank's vocab shard. Targets outside the shard contribute
        # nothing, which is fine: the point is a real backward pass through every
        # TP collective, not a converged model.
        shard = logits.size(-1)
        target = (input_ids[:, 1:] % shard).reshape(-1)
        loss = F.cross_entropy(logits[:, :-1, :].reshape(-1, shard), target)
        loss.backward()

        if n_dp > 1:
            # Data-parallel gradient all-reduce across replicas. This is the
            # traffic that crosses the node boundary when tp_size == local_world.
            for p in model.parameters():
                if p.grad is not None:
                    dist.all_reduce(p.grad, group=dp_group)
                    p.grad /= n_dp

        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        step += 1
        tokens += args.batch_size * args.seq_len
        if rank == 0:
            emit_step("train", step, tokens=tokens, node=node_rank, rank=rank,
                      tp_mb=round(tp_bytes() / 2**20, 1))

    elapsed = time.time() - wall_start
    record_progress(args.seed, step, tokens, elapsed)
    # TP traffic leaves the node only when the TP group spans nodes.
    inter_node = tp_bytes() if tp_size > local_world else 0
    if rank == 0:
        print(f"\nMegatron TP complete: {step} steps in {elapsed:.1f}s", flush=True)
        print(f"  Useful tokens: {tokens:,} ({tokens/elapsed:.0f} tok/s)", flush=True)
        print(f"  TP collective volume (this rank): {tp_bytes()/1e9:.2f} GB "
              f"over ~{step * (4 * args.layers + 2):,} calls", flush=True)
        print(f"  Inter-node bytes (this rank): {inter_node/1e9:.2f} GB "
              f"= {inter_node/elapsed/1e6:.1f} MB/s "
              f"= {inter_node*8/elapsed/1e6:.0f} Mbps", flush=True)
        print(f"  EVASION_TAX_DATA: strategy=TP world_size={world_size} "
              f"train_frac=1.0 dilution= steps={step} train_steps={step} "
              f"useful_tokens={tokens} attributed_tokens={tokens} "
              f"duration={elapsed:.1f} tp_size={tp_size} dp={n_dp} "
              f"internode_mbps={inter_node*8/elapsed/1e6:.1f}", flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
