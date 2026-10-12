#!/usr/bin/env python3
"""Mixture-of-Experts training with expert parallelism -- the all-to-all family.

Why this workload exists
------------------------
Every training family in the corpus communicates with all-reduce, all-gather,
reduce-scatter or point-to-point send/recv. Not one uses ALL-TO-ALL, and that is
the collective MoE training is built on: each token is routed to its expert,
which lives on a different rank, so every MoE layer ships a slice of the batch
to every peer and gets a different slice back -- twice per layer, once out and
once home.

That produces a traffic shape none of the current families can teach a
classifier about:

  * volume scales with TOKENS and the router's choices, not with parameter
    count, so it is independent of model size in a way all-reduce never is;
  * it is bidirectional and asymmetric per peer, where all-reduce is symmetric;
  * it is load-IMBALANCED by construction. Router collapse -- a few experts
    taking most of the tokens -- is the normal failure mode of MoE training, and
    it makes some ranks much louder than others. Every other training family in
    the corpus is rank-symmetric, so `xr_level_spread` and the cross-rank
    correlation features have never seen a training job whose ranks legitimately
    disagree about how much work they are doing.

The last point is the one that matters for the disguise question. A detector
that has learned "training means every rank looks alike" will mistake MoE for
something else, and an adversary who knows that can reach for it deliberately.

    torchrun --nproc-per-node=8 workloads/train_moe.py --experts-per-rank 2
    torchrun --nproc-per-node=8 workloads/train_moe.py --top-k 2 --capacity-factor 2.0

Router skew is a knob (--router-skew) precisely so the corpus gets both the
balanced and the collapsed regime; they look quite different on the wire.
"""

import argparse
import os
import sys
import time

try:
    from workloads._step import emit_step
    from workloads._seed import add_seed_argument, record_seed, record_progress
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
except ImportError:  # launched as `torchrun workloads/train_moe.py`
    from _step import emit_step
    from _seed import add_seed_argument, record_seed, record_progress
    from dist_deadline import CollectiveDeadline, sync_teardown

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
# The nodes pin torch 2.5.1 (requirements_v2.txt), where this is the supported
# autograd-aware collective. torch 2.12 on a dev box warns that it is deprecated
# in favour of torch.distributed._functional_collectives -- do NOT switch: that
# module is private and its 2.5 signature differs. The warning is harmless and
# Python emits it once per call site.
import torch.distributed.nn.functional as dist_nn


class Expert(nn.Module):
    """One expert: a plain feed-forward block."""

    def __init__(self, hidden, ffn):
        super().__init__()
        self.fc1 = nn.Linear(hidden, ffn, bias=False)
        self.fc2 = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x):
        return self.fc2(F.gelu(self.fc1(x)))


class MoELayer(nn.Module):
    """Expert-parallel MoE layer: route, all-to-all, compute, all-to-all back.

    Each rank owns `experts_per_rank` experts, so the global expert count is
    world_size * experts_per_rank and no rank holds the whole layer. Tokens must
    therefore physically move to their expert and back, which is the whole point
    -- a "MoE" that keeps every expert on every rank would produce no
    distinctive traffic at all and would not be worth collecting.
    """

    def __init__(self, hidden, ffn, experts_per_rank, world_size, top_k,
                 capacity_factor, router_skew):
        super().__init__()
        self.experts_per_rank = experts_per_rank
        self.world_size = world_size
        self.n_experts = experts_per_rank * world_size
        self.top_k = top_k
        self.capacity_factor = capacity_factor
        self.router_skew = router_skew
        self.router = nn.Linear(hidden, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [Expert(hidden, ffn) for _ in range(experts_per_rank)])

    def forward(self, x):
        """x: (tokens, hidden) already flattened. Returns same shape.

        Returns (output, bytes_moved) so the caller can account for traffic.
        """
        n_tok, hidden = x.shape
        logits = self.router(x)
        if self.router_skew:
            # Bias the router toward the low-numbered experts. This is what
            # collapse looks like, and it is a REGIME we want in the corpus, not
            # a bug: it makes the ranks that own those experts visibly busier.
            bias = torch.linspace(self.router_skew, 0.0, self.n_experts,
                                  device=x.device, dtype=logits.dtype)
            logits = logits + bias
        gates = F.softmax(logits, dim=-1)
        topk_gate, topk_idx = gates.topk(self.top_k, dim=-1)

        # Capacity: tokens per expert this rank will accept. Fixed-size buffers
        # are what make the all-to-all a single symmetric-shaped call; tokens
        # over capacity are dropped, exactly as in a real MoE implementation.
        cap = max(1, int(self.capacity_factor * n_tok * self.top_k / self.n_experts))

        # Build the send buffer: (n_experts, cap, hidden), zero-padded.
        #
        # `filled` is load-bearing. With top_k > 1 the same expert receives
        # tokens from several k-slots, and writing each slot at offset 0 would
        # have the second overwrite the first while combine_idx still pointed at
        # both -- every token after the first k would train on another token's
        # expert output. The result still runs and still moves exactly the right
        # number of bytes, so telemetry would look perfect.
        send = x.new_zeros(self.n_experts, cap, hidden)
        filled = [0] * self.n_experts
        combine_idx = []
        for k in range(self.top_k):
            idx = topk_idx[:, k]                      # expert per token
            for e in range(self.n_experts):
                room = cap - filled[e]
                if room <= 0:
                    continue
                sel = (idx == e).nonzero(as_tuple=True)[0][:room]
                if sel.numel():
                    lo = filled[e]
                    send[e, lo:lo + sel.numel()] = x[sel]
                    combine_idx.append((e, lo, sel, topk_gate[sel, k]))
                    filled[e] = lo + sel.numel()

        # THE call this workload exists for. Every rank hands every other rank
        # the tokens bound for its experts and receives the tokens bound for
        # its own, in one shot.
        #
        # torch.distributed.nn.functional, NOT torch.distributed: the plain
        # collective is opaque to autograd, so gradients would stop dead at the
        # layer boundary. The experts would never receive a gradient and the
        # backward pass would issue no all-to-all at all -- half the traffic
        # this cell exists to record, silently missing, while the workload still
        # reported a full step rate.
        recv = dist_nn.all_to_all_single(torch.empty_like(send), send)
        moved = send.numel() * send.element_size()

        # Each rank now holds, for each of ITS experts, the tokens every rank
        # routed there: shape (world_size, experts_per_rank, cap, hidden).
        recv = recv.view(self.world_size, self.experts_per_rank, cap, hidden)
        out = torch.stack([expert(recv[:, e])
                           for e, expert in enumerate(self.experts)], dim=1)

        # ... and the same call in reverse to bring every token home.
        flat = out.reshape(self.n_experts, cap, hidden)
        back = dist_nn.all_to_all_single(torch.empty_like(flat), flat)
        moved += flat.numel() * flat.element_size()

        y = torch.zeros_like(x)
        for e, lo, sel, gate in combine_idx:
            y = y.index_add(0, sel, back[e, lo:lo + sel.numel()] * gate.unsqueeze(-1))
        return y, moved


class MoETransformerBlock(nn.Module):
    def __init__(self, hidden, n_heads, ffn, experts_per_rank, world_size,
                 top_k, capacity_factor, router_skew):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, n_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.moe = MoELayer(hidden, ffn, experts_per_rank, world_size, top_k,
                            capacity_factor, router_skew)

    def forward(self, x):
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        B, T, H = x.shape
        y, moved = self.moe(self.norm2(x).reshape(B * T, H))
        return x + y.view(B, T, H), moved


class MoELM(nn.Module):
    def __init__(self, hidden, n_layers, n_heads, ffn, vocab, experts_per_rank,
                 world_size, top_k, capacity_factor, router_skew):
        super().__init__()
        self.vocab_size = vocab
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(4096, hidden)
        self.blocks = nn.ModuleList([
            MoETransformerBlock(hidden, n_heads, ffn, experts_per_rank,
                                world_size, top_k, capacity_factor, router_skew)
            for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        moved = 0
        for block in self.blocks:
            x, m = block(x)
            moved += m
        return self.lm_head(self.norm(x)), moved


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--duration", type=int, default=600)
    p.add_argument("--hidden", type=int, default=1024)
    p.add_argument("--layers", type=int, default=12)
    p.add_argument("--heads", type=int, default=16)
    p.add_argument("--ffn", type=int, default=4096)
    p.add_argument("--vocab", type=int, default=32000)
    p.add_argument("--batch-size", type=int, default=4)
    p.add_argument("--seq-len", type=int, default=512)
    p.add_argument("--experts-per-rank", type=int, default=2)
    p.add_argument("--top-k", type=int, default=2)
    p.add_argument("--capacity-factor", type=float, default=1.25)
    p.add_argument("--router-skew", type=float, default=0.0,
                   help="logit bias toward low-numbered experts; >0 induces the "
                        "collapsed, rank-imbalanced regime on purpose")
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

    torch.manual_seed(args.seed)
    model = MoELM(args.hidden, args.layers, args.heads, args.ffn, args.vocab,
                  args.experts_per_rank, world_size, args.top_k,
                  args.capacity_factor, args.router_skew).to(device)
    # Non-expert parameters must stay identical across ranks (they are replicated
    # data-parallel); the EXPERTS must not be, since each rank owns different
    # ones. Broadcasting everything and then letting the experts diverge through
    # their own gradients is the standard arrangement.
    for p in model.parameters():
        dist.broadcast(p.data, src=0)
    torch.manual_seed(args.seed + rank)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    if rank == 0:
        n_par = sum(p.numel() for p in model.parameters())
        print(f"MoE expert-parallel: params={n_par/1e6:.0f}M "
              f"experts={args.experts_per_rank * world_size} "
              f"(x{args.experts_per_rank}/rank) top_k={args.top_k} "
              f"cap_factor={args.capacity_factor} skew={args.router_skew} "
              f"world_size={world_size}", flush=True)

    deadline = CollectiveDeadline(args.duration, device)
    wall_start = time.time()
    step = 0
    tokens = 0
    a2a_bytes = 0
    while not deadline.expired():
        input_ids = torch.randint(0, args.vocab, (args.batch_size, args.seq_len),
                                  device=device)
        logits, moved = model(input_ids)
        loss = F.cross_entropy(
            logits[:, :-1, :].reshape(-1, logits.size(-1)),
            input_ids[:, 1:].reshape(-1))
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        opt.zero_grad(set_to_none=True)

        step += 1
        tokens += args.batch_size * args.seq_len
        # The backward pass runs the same two all-to-alls again, in reverse.
        a2a_bytes += moved * 2
        if rank == 0:
            emit_step("train", step, tokens=tokens, node=node_rank, rank=rank,
                      a2a_mb=round(moved * 2 / 2**20, 1))

    elapsed = time.time() - wall_start
    record_progress(args.seed, step, tokens, elapsed)
    # Only the share of the all-to-all that crosses the node boundary is
    # inter-node: each rank talks to world_size-1 peers, local_world-1 of them
    # on this host.
    remote_frac = (world_size - local_world) / max(world_size - 1, 1)
    inter_node = int(a2a_bytes * max(remote_frac, 0.0))
    if rank == 0:
        print(f"\nMoE complete: {step} steps in {elapsed:.1f}s", flush=True)
        print(f"  Useful tokens: {tokens:,} ({tokens/elapsed:.0f} tok/s)", flush=True)
        print(f"  All-to-all volume (this rank): {a2a_bytes/1e9:.2f} GB", flush=True)
        print(f"  Inter-node bytes (this rank): {inter_node/1e9:.2f} GB "
              f"= {inter_node/elapsed/1e6:.1f} MB/s "
              f"= {inter_node*8/elapsed/1e6:.0f} Mbps", flush=True)
        print(f"  EVASION_TAX_DATA: strategy=MOE world_size={world_size} "
              f"train_frac=1.0 dilution= steps={step} train_steps={step} "
              f"useful_tokens={tokens} attributed_tokens={tokens} "
              f"duration={elapsed:.1f} experts={args.experts_per_rank * world_size} "
              f"top_k={args.top_k} skew={args.router_skew} internode_mbps="
              f"{inter_node*8/elapsed/1e6:.1f}", flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
