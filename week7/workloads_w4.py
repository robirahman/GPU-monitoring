#!/usr/bin/env python3
"""
Week 4 standalone GPU workloads (no torchvision dependency).
Uses randomly generated tensors to exercise GPU in realistic patterns.
All workloads accept a --duration or --epochs flag.
"""

import argparse
import time
import sys
import signal

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np


# ── Model definitions ────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
        )
    def forward(self, x): return self.seq(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.seq = nn.Sequential(
            ConvBlock(ch, ch), ConvBlock(ch, ch)
        )
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.seq(x) + x)

def make_resnet18_like(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        ResBlock(64), ResBlock(64),
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        ResBlock(128), ResBlock(128),
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        ResBlock(256),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, num_classes),
    )

def make_mlp(input_dim=3072, hidden=512, num_classes=10):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(input_dim, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, num_classes),
    )

def make_resnet50_like(num_classes=1000):
    """Larger model for inference workload."""
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        ResBlock(64), ResBlock(64), ResBlock(64),
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        ResBlock(128), ResBlock(128), ResBlock(128), ResBlock(128),
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        ResBlock(256), ResBlock(256), ResBlock(256),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, num_classes),
    )


# ── Workload functions ────────────────────────────────────────────────────────

def workload_resnet_training(batch_size=512, epochs=3, amp=False,
                              image_size=32, num_classes=10, device="cuda"):
    device = torch.device(device)
    model = make_resnet18_like(num_classes).to(device)
    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)

    # Simulate dataset: 50000 random images
    n_train = 50000
    batches_per_epoch = max(1, n_train // batch_size)

    print(f"ResNet training: bs={batch_size}, epochs={epochs}, amp={amp}, device={device}")
    t0 = time.time()

    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for step in range(batches_per_epoch):
            data = torch.randn(batch_size, 3, image_size, image_size, device=device)
            target = torch.randint(0, num_classes, (batch_size,), device=device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=amp):
                out = model(data)
                loss = criterion(out, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
        avg_loss = running_loss / batches_per_epoch
        elapsed = time.time() - t0
        print(f"  Epoch {epoch+1}/{epochs}: loss={avg_loss:.4f}, elapsed={elapsed:.1f}s")

    print(f"Training complete in {time.time() - t0:.1f}s")


def workload_mlp_training(batch_size=512, epochs=5, image_size=32, device="cuda"):
    device = torch.device(device)
    model = make_mlp(3 * image_size * image_size, num_classes=10).to(device)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss().to(device)
    n_train = 50000
    batches = max(1, n_train // batch_size)

    print(f"MLP training: bs={batch_size}, epochs={epochs}, device={device}")
    t0 = time.time()
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        for _ in range(batches):
            data = torch.randn(batch_size, 3, image_size, image_size, device=device)
            target = torch.randint(0, 10, (batch_size,), device=device)
            optimizer.zero_grad()
            out = model(data)
            loss = criterion(out, target)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        print(f"  Epoch {epoch+1}/{epochs}: loss={running_loss/batches:.4f}")
    print(f"MLP training complete in {time.time() - t0:.1f}s")


def workload_inference(batch_size=256, duration=120, image_size=224, device="cuda"):
    """Pure forward pass — no gradients."""
    device = torch.device(device)
    model = make_resnet50_like(num_classes=1000).to(device)
    model.eval()

    print(f"ResNet50-like inference: bs={batch_size}, dur={duration}s, device={device}")
    t0 = time.time()
    total = 0
    with torch.no_grad():
        while time.time() - t0 < duration:
            data = torch.randn(batch_size, 3, image_size, image_size, device=device)
            _ = model(data)
            torch.cuda.synchronize()
            total += batch_size
            if total % (batch_size * 20) == 0:
                ips = total / (time.time() - t0)
                print(f"  {total:,} imgs, {ips:.0f} img/s")
    print(f"Inference complete: {total:,} images in {time.time()-t0:.1f}s")


def workload_cufft(duration=120, size=4096):
    """Repeated large 2D FFTs."""
    import signal as sig
    stopped = False
    def handler(s, f): nonlocal stopped; stopped = True
    sig.signal(sig.SIGTERM, handler)

    device = torch.device("cuda")
    x = torch.randn(8, size, size, dtype=torch.float32, device=device)
    print(f"cuFFT: 8×{size}×{size}, dur={duration}s")
    t0 = time.time(); iters = 0
    while (time.time() - t0) < duration and not stopped:
        y = torch.fft.fft2(x)
        z = torch.fft.ifft2(y)
        torch.cuda.synchronize()
        iters += 1
        if iters % 50 == 0:
            print(f"  {iters} iters, {time.time()-t0:.0f}s")
    print(f"cuFFT complete: {iters*16} FFTs in {time.time()-t0:.1f}s")


def workload_nbody(duration=120, particles=16384):
    """N-body gravitational simulation."""
    import signal as sig
    stopped = False
    def handler(s, f): nonlocal stopped; stopped = True
    sig.signal(sig.SIGTERM, handler)

    device = torch.device("cuda")
    pos = torch.randn(particles, 3, device=device)
    vel = torch.randn(particles, 3, device=device) * 0.1
    mass = torch.ones(particles, device=device) / particles
    print(f"N-body: {particles} particles, dur={duration}s")
    t0 = time.time(); steps = 0
    while (time.time() - t0) < duration and not stopped:
        diff = pos.unsqueeze(0) - pos.unsqueeze(1)
        dist_sq = (diff**2).sum(-1) + 0.01**2
        inv_dist3 = dist_sq**(-1.5)
        acc = (inv_dist3.unsqueeze(-1) * diff * mass.unsqueeze(0).unsqueeze(-1)).sum(1)
        vel = vel + acc * 0.001
        pos = pos + vel * 0.001
        torch.cuda.synchronize()
        steps += 1
        if steps % 100 == 0:
            print(f"  Step {steps}, {time.time()-t0:.0f}s")
    print(f"N-body complete: {steps} steps in {time.time()-t0:.1f}s")


def workload_idle(duration=120):
    """Idle GPU baseline."""
    print(f"Idle baseline: {duration}s")
    t0 = time.time()
    while time.time() - t0 < duration:
        time.sleep(1)
    print("Idle complete.")


# ── CLI dispatch ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--workload", required=True,
        choices=["resnet_train", "resnet_train_amp", "mlp_train", "inference",
                 "cufft", "nbody", "idle",
                 "resnet_small_batch", "resnet_large_batch", "resnet_short",
                 "mlp_small", "mlp_large",
                 "inference_small", "inference_large"])
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--duration", type=int, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    W = args.workload
    bs = args.batch_size
    ep = args.epochs
    dur = args.duration

    if W == "resnet_train":
        workload_resnet_training(batch_size=bs or 512, epochs=ep or 3, device=args.device)
    elif W == "resnet_train_amp":
        workload_resnet_training(batch_size=bs or 512, epochs=ep or 3, amp=True, device=args.device)
    elif W == "resnet_small_batch":
        workload_resnet_training(batch_size=bs or 8, epochs=ep or 2, device=args.device)
    elif W == "resnet_large_batch":
        workload_resnet_training(batch_size=bs or 2048, epochs=ep or 3, amp=True, device=args.device)
    elif W == "resnet_short":
        workload_resnet_training(batch_size=bs or 512, epochs=ep or 1, device=args.device)
    elif W == "mlp_train":
        workload_mlp_training(batch_size=bs or 512, epochs=ep or 5, device=args.device)
    elif W == "mlp_small":
        workload_mlp_training(batch_size=bs or 16, epochs=ep or 3, device=args.device)
    elif W == "mlp_large":
        workload_mlp_training(batch_size=bs or 4096, epochs=ep or 5, device=args.device)
    elif W == "inference":
        workload_inference(batch_size=bs or 256, duration=dur or 120, device=args.device)
    elif W == "inference_small":
        workload_inference(batch_size=bs or 32, duration=dur or 120, device=args.device)
    elif W == "inference_large":
        workload_inference(batch_size=bs or 1024, duration=dur or 120, device=args.device)
    elif W == "cufft":
        workload_cufft(duration=dur or 120)
    elif W == "nbody":
        workload_nbody(duration=dur or 120)
    elif W == "idle":
        workload_idle(duration=dur or 120)
