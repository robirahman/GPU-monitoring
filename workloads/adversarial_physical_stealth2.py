#!/usr/bin/env python3
"""Red-team round 2: evade the HARDENED physical classifier by mimicking cufft.

Round 1's winning attack used a compute-bound GEMM ballast -> high power but LOW
current ripple. The hardened classifier has now learned that "low-RMS + low-
periodicity + high-power" region as training. So round 2 moves to a DIFFERENT
"other" region the hardened model cannot reclaim without breaking its cufft
label: we mimic cufft directly with a CONTINUOUS, memory-bound (cuFFT) ballast
running in a background thread -> high, steady current RMS with low cv/periodicity
(cufft's signature). Real training is heavily diluted + jittered underneath.

Usage:
  python adversarial_physical_stealth2.py --duration 70 --fft-batch 16 --train-every 14
"""
import argparse, random, threading, time
import torch, torch.nn as nn, torchvision


def make_loader(data_dir, bs):
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=bs, shuffle=True, num_workers=8,
                                       drop_last=True, persistent_workers=True, pin_memory=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=70)
    ap.add_argument("--data-dir", default="./data/cifar10")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--fft-batch", type=int, default=16, help="3D-FFT batch (sets ballast RMS level)")
    ap.add_argument("--fft-size", type=int, default=512)
    ap.add_argument("--train-every", type=int, default=14, help="train one cycle per ~N (jittered) ballast ticks")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"

    model = torchvision.models.resnet18(num_classes=10).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(args.data_dir, args.batch_size); it = iter(loader)

    # Continuous memory-bound cuFFT ballast on a background thread/stream -> steady HIGH ripple
    # (mimics workloads/cufft_benchmark). No alternation with training -> low periodicity.
    stop = threading.Event()
    def ballast():
        s = torch.cuda.Stream()
        data = torch.randn(args.fft_batch, args.fft_size, args.fft_size, dtype=torch.complex64, device=dev)
        with torch.cuda.stream(s):
            while not stop.is_set():
                r = torch.fft.fftn(data, dim=(1, 2))
                data.copy_(torch.fft.ifftn(r, dim=(1, 2)))
    th = threading.Thread(target=ballast, daemon=True); th.start()

    model.train()
    t0 = time.time(); steps = 0; cycle = 0; nxt = args.train_every
    while time.time() - t0 < args.duration:
        cycle += 1
        time.sleep(0.001)  # let the ballast dominate the timeline
        if cycle < nxt:
            continue
        nxt = cycle + max(1, int(args.train_every * random.uniform(0.5, 1.5)))
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y); loss.backward(); opt.step(); steps += 1

    stop.set(); th.join(timeout=5); torch.cuda.synchronize()
    print(f"[stealth2] {steps} real training steps in {time.time()-t0:.1f}s "
          f"(final loss={loss.item():.3f})", flush=True)


if __name__ == "__main__":
    main()
