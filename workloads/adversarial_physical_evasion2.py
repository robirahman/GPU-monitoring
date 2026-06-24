#!/usr/bin/env python3
"""Throughput-preserving evasions of the PHYSICAL-sensor classifier (v2).

The classifier separates training from steady HPC almost entirely on per-second
current-clamp *burstiness* (cv) and *periodicity* (FFT peak). These strategies
keep training at ~full throughput and reshape its signature instead of diluting.

Modes (all do REAL ResNet-18 training; report samples/sec):
  baseline  : plain fixed-batch loop (control; should be CAUGHT)
  gradaccum : gradient accumulation over K micro-batches -> the big periodic
              optimizer-step current spike is removed; continuous fwd/bwd stream.
              ~zero throughput cost.
  aperiodic : randomize batch size each step -> smears the cadence.
  cofill    : full-speed training + a CONTINUOUS background GEMM ballast on its
              own CUDA stream/thread (no per-step alternation). Fills idle gaps
              -> raises+flattens the current floor, masks training bursts.
  combo     : gradaccum + cofill.

Usage:
  python adversarial_physical_evasion2.py --mode cofill --cofill-gemm 4096 --duration 65
  python adversarial_physical_evasion2.py --mode gradaccum --accum-steps 8 --duration 65
"""
import argparse, random, threading, time
import torch, torch.nn as nn, torchvision


def make_loader(data_dir, batch_size):
    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = torchvision.datasets.CIFAR10(root=data_dir, train=True, download=True, transform=tf)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=True, num_workers=8,
                                       drop_last=True, persistent_workers=True, pin_memory=True, prefetch_factor=4)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["baseline", "gradaccum", "aperiodic", "cofill", "combo", "gapfill"], required=True)
    ap.add_argument("--gap-gemm", type=int, default=2048, help="gapfill ballast GEMM dim (small=fine-grained)")
    ap.add_argument("--gap-sleep", type=float, default=0.004, help="sleep between gapfill GEMMs (leaves GPU headroom)")
    ap.add_argument("--duration", type=int, default=65)
    ap.add_argument("--data-dir", default="./data/cifar10")
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--accum-steps", type=int, default=8, help="micro-batches per optimizer step (gradaccum/combo)")
    ap.add_argument("--cofill-gemm", type=int, default=4096, help="background ballast GEMM dim (cofill/combo)")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    random.seed(args.seed); torch.manual_seed(args.seed)
    dev = "cuda"; torch.backends.cuda.matmul.allow_tf32 = True

    model = torchvision.models.resnet18(num_classes=10).to(dev)
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()
    loader = make_loader(args.data_dir, 256 if args.mode == "aperiodic" else args.batch_size)
    it = iter(loader)

    aperiodic = args.mode == "aperiodic"
    accum = args.accum_steps if args.mode in ("gradaccum", "combo") else 1
    sizes = [64, 96, 128, 160, 192, 224, 256]

    # Continuous background ballast (cofill/combo): a daemon thread issuing GEMMs
    # on its own stream, overlapping training to fill idle gaps -> steady floor.
    stop_evt = threading.Event()
    def ballast_loop():
        s = torch.cuda.Stream()
        # gapfill: small GEMM + sleep throttle -> fills the data-stall idle gaps
        # without monopolizing the GPU during training kernels.
        g = args.gap_gemm if args.mode == "gapfill" else args.cofill_gemm
        sleep = args.gap_sleep if args.mode == "gapfill" else 0.0
        A = torch.randn(g, g, device=dev); B = torch.randn(g, g, device=dev); C = torch.empty(g, g, device=dev)
        with torch.cuda.stream(s):
            while not stop_evt.is_set():
                torch.matmul(A, B, out=C); A.mul_(0.999999).add_(C, alpha=1e-12)
                if sleep:
                    s.synchronize(); time.sleep(sleep)
    bal_thread = None
    if args.mode in ("cofill", "combo", "gapfill"):
        bal_thread = threading.Thread(target=ballast_loop, daemon=True); bal_thread.start()

    model.train()
    t0 = time.time(); steps = 0; samples = 0; micro = 0
    opt.zero_grad(set_to_none=True)
    while time.time() - t0 < args.duration:
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        if aperiodic:
            bs = random.choice(sizes); x, y = x[:bs], y[:bs]
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        loss = crit(model(x), y) / accum
        loss.backward()
        micro += 1; samples += x.size(0)
        if micro % accum == 0:
            opt.step(); opt.zero_grad(set_to_none=True); steps += 1

    stop_evt.set()
    if bal_thread: bal_thread.join(timeout=5)
    torch.cuda.synchronize()
    dur = time.time() - t0
    print(f"[evasion2:{args.mode}] {steps} opt-steps, {micro} micro-batches, {samples} samples in {dur:.1f}s "
          f"-> {samples/dur:.0f} samples/s (final loss={loss.item()*accum:.3f})", flush=True)


if __name__ == "__main__":
    main()
