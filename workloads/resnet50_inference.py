#!/usr/bin/env python3
"""ResNet-50 batch inference on CIFAR-10 test set.

No gradient computation — pure forward pass. Should produce a distinct
telemetry signature: lower GPU util, no backward pass, constant memory,
lower power than training.

Usage:
    python resnet50_inference.py --batch-size 256 --loops 10
"""

import argparse
import time

import torch
import torchvision
import torchvision.transforms as transforms


def main():
    parser = argparse.ArgumentParser(description="ResNet-50 batch inference")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument(
        "--loops", type=int, default=10,
        help="Number of passes over the test set",
    )
    parser.add_argument(
        "--data-dir", type=str, default="./data/cifar10", help="Dataset directory",
    )
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"ResNet-50 inference on CIFAR-10 test set")
    print(f"  Device: {device}, Batch size: {args.batch_size}, Loops: {args.loops}")

    transform = transforms.Compose([
        transforms.Resize(224),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
    ])
    test_dataset = torchvision.datasets.CIFAR10(
        root=args.data_dir, train=False, download=True, transform=transform,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False,
        num_workers=2, pin_memory=True,
    )

    model = torchvision.models.resnet50(weights="IMAGENET1K_V1").to(device)
    model.eval()

    wall_start = time.time()
    total_images = 0

    with torch.no_grad():
        for loop in range(args.loops):
            loop_start = time.time()
            correct = 0
            count = 0
            for data, target in test_loader:
                data = data.to(device)
                output = model(data)
                pred = output.argmax(dim=1).cpu()
                correct += pred.eq(target).sum().item()
                count += target.size(0)
            total_images += count
            loop_time = time.time() - loop_start
            print(
                f"  Loop {loop + 1}/{args.loops}: "
                f"{count} images, {loop_time:.1f}s, "
                f"{count / loop_time:.0f} img/s"
            )

    wall_time = time.time() - wall_start
    print(
        f"Inference complete. {total_images} images in {wall_time:.1f}s "
        f"({total_images / wall_time:.0f} img/s)"
    )


if __name__ == "__main__":
    main()
