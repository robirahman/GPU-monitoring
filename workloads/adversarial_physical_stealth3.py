#!/usr/bin/env python3
"""Red-team round 3: evade the HARDENED classifier by faithfully mimicking cufft.

Round 2 failed because the FFT ballast (512x512) only reached RMS ~14. The real
cufft workload (workloads/scientific_hpc.py::run_cufft) runs a (512,512,64)
complex64 3D fftn/ifftn loop -> RMS ~31, cv ~0.19, FFT-peak ~0.10. Here we run
that EXACT loop as a continuous background ballast and dilute real training to a
tunable rate, so the aggregate current signature sits squarely in cufft's "other"
region that the hardened model cannot reclaim without breaking its cufft label.

Usage:
  python adversarial_physical_stealth3.py --duration 70 --train-period 2.0
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
    ap.add_argument("--train-period", type=float, default=2.0,
                    help="seconds between real training steps (jittered +/-50%); higher = stealthier/slower")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = torch.device("cuda")

    model = torchvision.models.resnet18(num_classes=10).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(args.data_dir, args.batch_size); it = iter(loader)

    # EXACT cufft ballast (mirrors scientific_hpc.run_cufft) on a background stream.
    stop = threading.Event()
    def ballast():
        s = torch.cuda.Stream()
        x = torch.randn(512, 512, 64, dtype=torch.complex64, device=dev)
        with torch.cuda.stream(s):
            while not stop.is_set():
                y = torch.fft.fftn(x)
                z = torch.fft.ifftn(y)
                x = z.real.to(dtype=torch.complex64) + x.imag * 1j
                s.synchronize()
    th = threading.Thread(target=ballast, daemon=True); th.start()

    model.train()
    t0 = time.time(); steps = 0
    next_t = t0 + args.train_period
    while time.time() - t0 < args.duration:
        now = time.time()
        if now < next_t:
            time.sleep(min(0.02, next_t - now))  # idle (ballast runs) until next diluted step
            continue
        next_t = now + args.train_period * random.uniform(0.5, 1.5)
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y); loss.backward(); opt.step(); steps += 1

    stop.set(); th.join(timeout=5); torch.cuda.synchronize()
    dur = time.time() - t0
    print(f"[stealth3] {steps} real training steps in {dur:.1f}s "
          f"({steps/dur:.1f} steps/s, final loss={loss.item():.3f})", flush=True)


if __name__ == "__main__":
    main()
