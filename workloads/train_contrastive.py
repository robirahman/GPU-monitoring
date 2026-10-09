#!/usr/bin/env python3
"""HONEST contrastive (CLIP-style) training -- a collective pattern the corpus lacks.

WHY THIS EXISTS. The two training programs in the corpus both communicate the same
way: a gradient reduction over parameters, once per step, whose size is set by the
model and never by the batch. FSDP adds parameter all-gathers, DDP does not, but
in both the traffic is a function of how big the model is.

Contrastive training is structurally different. The loss is defined over the whole
global batch, so before any loss can be computed every rank must all-gather the
other ranks' EMBEDDINGS -- and gradients must flow back through that gather. So:

  * there is a large collective in the FORWARD pass, not only the backward
  * its size scales with batch x embedding dim and with WORLD SIZE, not with the
    model, so adding nodes makes the message bigger rather than just spreading it
  * the similarity matmul is (global_batch x global_batch), which on 16 ranks is
    256x the per-rank matrix -- a compute spike with no analogue in a decoder LM
  * two towers run per step, so the forward is two disjoint subgraphs

That combination is what makes it worth collecting: a detector that has learned
"training = one big gradient reduction per step" has no reason to recognise it,
and CLIP-style training is a large fraction of real multi-node training jobs.

The all-gather is implemented with the standard differentiable trick: torch's
all_gather does not propagate gradients, so each rank splices its OWN slice back
into the gathered tensor, leaving a path to its local graph. Without that the
towers receive no gradient from the negatives and the cell would be quietly
training on almost nothing -- a failure that does not crash and barely changes
the telemetry.

LABELLING: cells are named s16_train_contrastive -- contains "train", routes to
ml_training, which is correct: real gradients, real optimizer step. No disguise
token. Verified in classifier/rebuttal_analysis/test_cell_label_routing.py.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/train_contrastive.py \
        --embed-dim 1024 --batch-size 64 --duration 600
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
except ImportError:
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown


class Tower(nn.Module):
    """One encoder tower: patch/token projection + transformer blocks + pooling."""

    def __init__(self, in_dim, hidden, layers, heads, embed_dim):
        super().__init__()
        self.proj = nn.Linear(in_dim, hidden)
        enc = nn.TransformerEncoderLayer(
            hidden, heads, dim_feedforward=4 * hidden, batch_first=True,
            norm_first=True)
        self.enc = nn.TransformerEncoder(enc, layers)
        self.head = nn.Linear(hidden, embed_dim)

    def forward(self, x):
        h = self.enc(self.proj(x))
        return F.normalize(self.head(h.mean(dim=1)), dim=-1)


class ContrastivePair(nn.Module):
    def __init__(self, img_dim, txt_dim, hidden, layers, heads, embed_dim):
        super().__init__()
        self.image = Tower(img_dim, hidden, layers, heads, embed_dim)
        self.text = Tower(txt_dim, hidden, layers, heads, embed_dim)
        self.logit_scale = nn.Parameter(torch.tensor(2.6592))   # ln(1/0.07)

    def forward(self, img, txt):
        return self.image(img), self.text(txt)


def gather_with_grad(t, world, rank):
    """all_gather that keeps a gradient path to this rank's own slice.

    dist.all_gather returns tensors detached from the autograd graph. If we used
    the result as-is, the negatives would contribute no gradient to this rank's
    tower and each rank would effectively train against its own batch only -- the
    loss would still fall, the step time would barely move, and the cell would be
    silently much weaker than a real contrastive job. Splicing the local slice
    back in restores the path.
    """
    buf = [torch.zeros_like(t) for _ in range(world)]
    dist.all_gather(buf, t.contiguous())
    buf[rank] = t                      # the one slice with a live graph
    return torch.cat(buf, dim=0)


def main():
    ap = argparse.ArgumentParser(description="Honest contrastive training (CLIP-style)")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=64, help="per rank")
    ap.add_argument("--img-tokens", type=int, default=196, help="14x14 patches")
    ap.add_argument("--txt-tokens", type=int, default=77)
    ap.add_argument("--img-dim", type=int, default=768)
    ap.add_argument("--txt-dim", type=int, default=512)
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=12)
    ap.add_argument("--heads", type=int, default=16)
    ap.add_argument("--embed-dim", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = ContrastivePair(args.img_dim, args.txt_dim, args.hidden,
                            args.layers, args.heads, args.embed_dim).to(device)
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr)

    gb = args.batch_size * world
    if rank == 0:
        nparam = sum(p.numel() for p in model.parameters())
        print(f"Contrastive training (HONEST): {world} ranks, {nparam/1e6:.0f}M params")
        print(f"  per-rank batch {args.batch_size} -> global {gb}; "
              f"similarity matrix {gb}x{gb} ({gb*gb*4/1e6:.1f} MB)")
        print(f"  embedding all-gather {args.batch_size*args.embed_dim*4/1e6:.2f} MB "
              f"per tower per rank, x2 towers, EVERY STEP IN THE FORWARD PASS")
        print(f"  duration {args.duration}s")

    deadline = CollectiveDeadline(args.duration, device)
    step = 0
    last = time.time()
    while not deadline.expired():
        img = torch.randn(args.batch_size, args.img_tokens, args.img_dim, device=device)
        txt = torch.randn(args.batch_size, args.txt_tokens, args.txt_dim, device=device)

        ie, te = ddp(img, txt)
        # The collective that defines this workload.
        all_ie = gather_with_grad(ie, world, rank)
        all_te = gather_with_grad(te, world, rank)

        scale = model.logit_scale.exp().clamp(max=100.0)
        logits = scale * all_ie @ all_te.t()
        labels = torch.arange(logits.shape[0], device=device)
        loss = 0.5 * (F.cross_entropy(logits, labels)
                      + F.cross_entropy(logits.t(), labels))

        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if rank == 0:
            emit_step("train", step, tokens=gb, passes=2, node=node_rank,
                      rank=rank, method="contrastive")
            now = time.time()
            if now - last >= 30:
                print(f"  t={deadline.elapsed:6.0f}s  step={step}  "
                      f"loss={loss.item():.3f}  {step*gb/max(deadline.elapsed,1):.0f} "
                      f"pairs/s (global)", flush=True)
                last = now

    if rank == 0:
        print(f"Contrastive done: {step} steps, {step*gb} pairs, "
              f"{deadline.elapsed:.0f}s", flush=True)
    sync_teardown()


if __name__ == "__main__":
    main()
