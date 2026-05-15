#!/usr/bin/env python3
"""YOLOv8 object-detection training workload.

Trains YOLOv8n (the nano variant, ~3 MB weights) on the bundled COCO128
dataset for a fixed number of iterations. Adds object detection — a
materially different gradient pattern from classification — to the corpus.

Usage:
    python yolo_train.py --duration 300
    python yolo_train.py --duration 300 --imgsz 320 --batch 16
"""

import argparse
import time

import torch
from ultralytics import YOLO


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--duration", type=int, default=300)
    p.add_argument("--model", default="yolov8n.pt")
    p.add_argument("--data", default="coco128.yaml")
    p.add_argument("--imgsz", type=int, default=320)
    p.add_argument("--batch", type=int, default=16)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU"); return 1
    print(f"YOLOv8 training on {torch.cuda.get_device_name(0)}, imgsz={args.imgsz}, bs={args.batch}")

    # Estimate epochs from duration — coco128 is 128 imgs, ~10s per epoch at bs=16
    epochs = max(1, args.duration // 10)
    model = YOLO(args.model)
    start = time.time()
    model.train(
        data=args.data,
        epochs=epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        device=0,
        workers=2,
        verbose=False,
        save=False,
        plots=False,
        amp=True,
    )
    elapsed = time.time() - start
    print(f"YOLO training done. {epochs} epochs in {elapsed:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
