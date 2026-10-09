#!/usr/bin/env python3
"""Multi-GPU ResNet training across all visible CUDA devices (nn.DataParallel).

Used to add multi-GPU and other-than-GPU0 training to the physical-sensor dataset,
so the classifier can detect training that runs on GPU1/GPU2 or spans several GPUs
(otherwise an adversary trivially evades a GPU0-only detector by training elsewhere).

Set which GPUs via CUDA_VISIBLE_DEVICES (the run_workload_with_sensors --cuda-devices
flag does this). Uses real CIFAR-10 + gradient updates.

Usage:
  CUDA_VISIBLE_DEVICES=0,1,2 python multigpu_training.py --duration 150 --batch-size 384
"""
import argparse, time
import torch, torch.nn as nn, torchvision


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--duration", type=int, default=150)
    ap.add_argument("--data-dir", default="./data/cifar10")
    ap.add_argument("--batch-size", type=int, default=384, help="global batch (split across GPUs)")
    ap.add_argument("--model", default="resnet18")
    args = ap.parse_args()

    n = torch.cuda.device_count()
    dev = torch.device("cuda")
    print(f"[multigpu] {n} visible GPU(s): {[torch.cuda.get_device_name(i) for i in range(n)]}", flush=True)

    tf = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize((0.5,) * 3, (0.5,) * 3)])
    ds = torchvision.datasets.CIFAR10(root=args.data_dir, train=True, download=True, transform=tf)
    loader = torch.utils.data.DataLoader(ds, batch_size=args.batch_size, shuffle=True, num_workers=8,
                                         drop_last=True, persistent_workers=True, pin_memory=True)

    model = getattr(torchvision.models, args.model)(num_classes=10).to(dev)
    if n > 1:
        model = nn.DataParallel(model, device_ids=list(range(n)))  # replicate across all visible GPUs
    opt = torch.optim.SGD(model.parameters(), lr=0.05, momentum=0.9)
    crit = nn.CrossEntropyLoss()

    model.train()
    t0 = time.time(); steps = 0; it = iter(loader)
    while time.time() - t0 < args.duration:
        try: x, y = next(it)
        except StopIteration: it = iter(loader); x, y = next(it)
        x, y = x.to(dev, non_blocking=True), y.to(dev, non_blocking=True)
        opt.zero_grad(set_to_none=True)
        loss = crit(model(x), y); loss.backward(); opt.step(); steps += 1
    torch.cuda.synchronize()
    print(f"[multigpu] {steps} steps across {n} GPU(s) in {time.time()-t0:.1f}s "
          f"(final loss={loss.item():.3f})", flush=True)


if __name__ == "__main__":
    main()
