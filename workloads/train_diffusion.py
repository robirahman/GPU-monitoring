#!/usr/bin/env python3
"""HONEST diffusion (UNet denoising) training -- a step shape the corpus lacks.

WHY THIS EXISTS. Both training programs in the corpus are token models with a
flat step: one forward through a stack of identical transformer blocks, one
backward, one optimizer step, repeat. Every window looks like every other window,
which is exactly the periodicity a detector can lock onto -- and, per section 33d,
what it does NOT lock onto, preferring absolute memory level, partly because there
is no training program in the corpus whose memory moves.

A denoising UNet moves on both axes:

  * the encoder halves the spatial resolution and doubles the channels at each
    level, then the decoder reverses it, so activation memory rises and falls
    WITHIN a single step instead of being flat across it
  * skip connections keep every encoder activation alive until its decoder level
    consumes it, so peak memory lands mid-step, not at the end of the forward
  * convolution dominates the outer levels and attention the inner ones, so one
    step sweeps two different arithmetic-intensity regimes
  * the timestep embedding makes each sample's compute path depend on a sampled
    t, so consecutive steps are not numerically identical

The result is real training with a non-flat intra-step profile, which is the
control the corpus has never had: if the detector is reading "steady high
utilisation at a fixed memory level" as its training signal, this is the training
job it should miss, and we need to know that.

Built from torch primitives rather than diffusers' UNet2DModel on purpose: it
must run under venv-main with no new dependency and no weight download, and the
shape is what matters here, not the pretrained weights.

LABELLING: cells are named s16_train_diffusion -- contains "train", routes to
ml_training, correct. No disguise token. Verified in
classifier/rebuttal_analysis/test_cell_label_routing.py.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/train_diffusion.py \
        --resolution 256 --batch-size 16 --duration 600
"""
import argparse
import math
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


def timestep_embedding(t, dim):
    """Sinusoidal embedding of the diffusion timestep, as in Ho et al."""
    half = dim // 2
    freqs = torch.exp(-math.log(10000) * torch.arange(half, device=t.device) / half)
    a = t.float()[:, None] * freqs[None]
    return torch.cat([torch.cos(a), torch.sin(a)], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, cin, cout, tdim):
        super().__init__()
        self.n1 = nn.GroupNorm(8, cin)
        self.c1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.n2 = nn.GroupNorm(8, cout)
        self.c2 = nn.Conv2d(cout, cout, 3, padding=1)
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x, temb):
        h = self.c1(F.silu(self.n1(x)))
        h = h + self.temb(F.silu(temb))[:, :, None, None]
        h = self.c2(F.silu(self.n2(h)))
        return h + self.skip(x)


class SelfAttn2d(nn.Module):
    """Attention at the low-resolution levels, where it is affordable."""

    def __init__(self, c, heads=8):
        super().__init__()
        self.norm = nn.GroupNorm(8, c)
        self.attn = nn.MultiheadAttention(c, heads, batch_first=True)

    def forward(self, x):
        b, c, h, w = x.shape
        y = self.norm(x).flatten(2).transpose(1, 2)
        y, _ = self.attn(y, y, y, need_weights=False)
        return x + y.transpose(1, 2).reshape(b, c, h, w)


class UNet(nn.Module):
    def __init__(self, base=128, mults=(1, 2, 4, 4), tdim=512, attn_from=2):
        super().__init__()
        self.tdim = tdim
        self.tmlp = nn.Sequential(nn.Linear(tdim, tdim), nn.SiLU(),
                                  nn.Linear(tdim, tdim))
        self.stem = nn.Conv2d(3, base, 3, padding=1)
        chans = [base * m for m in mults]

        self.down, self.downsample = nn.ModuleList(), nn.ModuleList()
        cin = base
        for i, c in enumerate(chans):
            self.down.append(nn.ModuleList(
                [ResBlock(cin, c, tdim), ResBlock(c, c, tdim),
                 SelfAttn2d(c) if i >= attn_from else nn.Identity()]))
            self.downsample.append(
                nn.Conv2d(c, c, 3, stride=2, padding=1) if i < len(chans) - 1
                else nn.Identity())
            cin = c

        self.mid1 = ResBlock(cin, cin, tdim)
        self.midattn = SelfAttn2d(cin)
        self.mid2 = ResBlock(cin, cin, tdim)

        self.up, self.upsample = nn.ModuleList(), nn.ModuleList()
        for i, c in reversed(list(enumerate(chans))):
            # cin + c: the skip connection, which is what keeps every encoder
            # activation resident until its decoder level consumes it.
            self.up.append(nn.ModuleList(
                [ResBlock(cin + c, c, tdim), ResBlock(c, c, tdim),
                 SelfAttn2d(c) if i >= attn_from else nn.Identity()]))
            self.upsample.append(
                nn.Upsample(scale_factor=2, mode="nearest") if i > 0 else nn.Identity())
            cin = c

        self.out = nn.Sequential(nn.GroupNorm(8, cin), nn.SiLU(),
                                 nn.Conv2d(cin, 3, 3, padding=1))

    def forward(self, x, t):
        temb = self.tmlp(timestep_embedding(t, self.tdim))
        h = self.stem(x)
        skips = []
        for (r1, r2, at), ds in zip(self.down, self.downsample):
            h = r2(r1(h, temb), temb)
            h = at(h) if not isinstance(at, nn.Identity) else h
            skips.append(h)
            h = ds(h)
        h = self.mid2(self.midattn(self.mid1(h, temb)), temb)
        for (r1, r2, at), us in zip(self.up, self.upsample):
            h = torch.cat([h, skips.pop()], dim=1)
            h = r2(r1(h, temb), temb)
            h = at(h) if not isinstance(at, nn.Identity) else h
            h = us(h)
        return self.out(h)


def main():
    ap = argparse.ArgumentParser(description="Honest diffusion UNet training")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--resolution", type=int, default=256)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--base-channels", type=int, default=128)
    ap.add_argument("--timesteps", type=int, default=1000)
    ap.add_argument("--lr", type=float, default=1e-4)
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    model = UNet(base=args.base_channels).to(device)
    ddp = DDP(model, device_ids=[local_rank], find_unused_parameters=False)
    opt = torch.optim.AdamW(ddp.parameters(), lr=args.lr)

    # Cosine schedule (Nichol & Dhariwal), precomputed once.
    steps = torch.arange(args.timesteps + 1, device=device) / args.timesteps
    ac = torch.cos((steps + 0.008) / 1.008 * math.pi / 2) ** 2
    ac = (ac / ac[0]).clamp(1e-5, 1.0)

    if rank == 0:
        n = sum(p.numel() for p in model.parameters())
        print(f"Diffusion UNet training (HONEST): {world} ranks, {n/1e6:.0f}M params")
        print(f"  {args.resolution}x{args.resolution}, batch {args.batch_size}/rank "
              f"-> {args.batch_size*world} global, {args.timesteps}-step schedule")
        print(f"  duration {args.duration}s")

    deadline = CollectiveDeadline(args.duration, device)
    step = 0
    last = time.time()
    while not deadline.expired():
        x0 = torch.randn(args.batch_size, 3, args.resolution, args.resolution,
                         device=device)
        t = torch.randint(0, args.timesteps, (args.batch_size,), device=device)
        noise = torch.randn_like(x0)
        a = ac[t][:, None, None, None]
        xt = a.sqrt() * x0 + (1 - a).sqrt() * noise

        pred = ddp(xt, t)
        loss = F.mse_loss(pred, noise)
        loss.backward()
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if rank == 0:
            emit_step("train", step, tokens=args.batch_size, passes=1,
                      node=node_rank, rank=rank, method="diffusion")
            now = time.time()
            if now - last >= 30:
                print(f"  t={deadline.elapsed:6.0f}s  step={step}  "
                      f"loss={loss.item():.4f}  "
                      f"{step*args.batch_size*world/max(deadline.elapsed,1):.1f} img/s "
                      f"(global)  peak {torch.cuda.max_memory_allocated(device)/1e9:.1f} GB",
                      flush=True)
                last = now

    if rank == 0:
        print(f"Diffusion done: {step} steps, {step*args.batch_size*world} images, "
              f"{deadline.elapsed:.0f}s", flush=True)
    sync_teardown()


if __name__ == "__main__":
    main()
