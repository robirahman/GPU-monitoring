#!/usr/bin/env python3
"""Adversarial workload: real training disguised as steady HPC, to evade the
PHYSICAL-sensor classifier (acoustic mic + GPU power-rail current clamps).

Threat model
------------
The physical classifier detects training from the *bursty, periodic* current
draw of the forward->backward->optimizer step loop: the per-second current-clamp
RMS fluctuates (high std/CV) with an FFT periodicity peak, plus a matching
acoustic coil-whine modulation. Steady compute (cuFFT) shows high-but-FLAT RMS;
idle/mining show low RMS.

Evasion: "ballast masking". We do genuine ResNet-18 training (real gradients +
optimizer steps), but interleave a continuously-running large GEMM ballast that
pins the GPU at a constant high power level. The ballast fills the power troughs
between training steps, so:
  - per-second current RMS becomes ~constant  -> low std / low CV
  - the step cadence is buried under steady GEMM -> no FFT periodicity peak
  - coil whine flattens -> mic looks like steady compute
The aggregate physical signature mimics a steady HPC/GEMM workload ("other"),
while real training still progresses underneath.

Usage:
    python adversarial_physical_stealth.py --duration 150 --ballast-size 8192 --steps-per-cycle 1
"""
import argparse
import time
import torch
import torch.nn as nn
import torchvision


def build_loader(data_dir, batch_size):
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,) * 3, (0.5,) * 3),
    ])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True,
                                       num_workers=4, drop_last=True, persistent_workers=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=150)
    ap.add_argument("--data-dir", default="./data/cifar10")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--ballast-mode", choices=["gemm", "fft"], default="fft",
                    help="fft = memory-bound cuFFT (HIGH steady current ripple, mimics cufft_benchmark); "
                         "gemm = compute-bound (low ripple)")
    ap.add_argument("--ballast-size", type=int, default=8192,
                    help="square GEMM dim (gemm mode)")
    ap.add_argument("--ballast-iters", type=int, default=20,
                    help="ballast ops issued per training micro-step (higher=flatter/more masking)")
    ap.add_argument("--steps-per-cycle", type=int, default=1,
                    help="real training steps per cycle")
    ap.add_argument("--train-every", type=int, default=1,
                    help="run a training step only every Nth ballast cycle (dilution); "
                         "jittered +/-50%% to avoid a periodic cadence")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import random
    random.seed(args.seed)

    dev = "cuda"
    torch.backends.cuda.matmul.allow_tf32 = True

    # Real training components
    model = torchvision.models.resnet18(num_classes=10).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    loader = build_loader(args.data_dir, args.batch_size)
    it = iter(loader)

    # Steady ballast.
    bs = args.ballast_size
    A = torch.randn(bs, bs, device=dev)
    B = torch.randn(bs, bs, device=dev)
    C = torch.empty(bs, bs, device=dev)
    # cuFFT ballast tensor (mirrors workloads cufft_benchmark: memory-bound 3D FFT)
    fft_data = torch.randn(8, 512, 512, dtype=torch.complex64, device=dev)

    def run_ballast():
        if args.ballast_mode == "fft":
            for _ in range(args.ballast_iters):
                r = torch.fft.fftn(fft_data, dim=(1, 2))
                fft_data.copy_(torch.fft.ifftn(r, dim=(1, 2)))  # keep it churning memory
        else:
            for _ in range(args.ballast_iters):
                torch.matmul(A, B, out=C)
                A.mul_(0.999999).add_(C, alpha=1e-12)

    print(f"[stealth] ResNet18 training masked by {args.ballast_mode.upper()} ballast "
          f"({args.ballast_iters} ballast ops / train step)", flush=True)

    import random
    t0 = time.time()
    train_steps = 0
    ballast_ops = 0
    cycle = 0
    next_train = max(1, args.train_every)
    model.train()
    while time.time() - t0 < args.duration:
        # --- steady ballast (dominant, pins a flat current floor) ---
        run_ballast()
        ballast_ops += args.ballast_iters
        cycle += 1

        # --- rare, jittered, hidden training step(s) (dilution) ---
        if cycle < next_train:
            continue
        # schedule next training cycle with +/-50% jitter (no clean periodicity)
        next_train = cycle + max(1, int(args.train_every * random.uniform(0.5, 1.5)))
        for _ in range(args.steps_per_cycle):
            try:
                x, y = next(it)
            except StopIteration:
                it = iter(loader); x, y = next(it)
            x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            train_steps += 1

    torch.cuda.synchronize()
    dur = time.time() - t0
    print(f"[stealth] done: {train_steps} real training steps, {ballast_ops} ballast GEMMs "
          f"in {dur:.1f}s (final loss={loss.item():.3f})", flush=True)


if __name__ == "__main__":
    main()
