#!/usr/bin/env python3
"""Pre-download CIFAR-10 + HF datasets used by extended-epoch training."""

import os
import sys
import time

CIFAR_ROOT = os.environ.get("CIFAR_ROOT", "/workspace/.cache/cifar")
os.makedirs(CIFAR_ROOT, exist_ok=True)

print("CIFAR-10 →", CIFAR_ROOT, flush=True)
t0 = time.time()
try:
    import torchvision
    torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=True, download=True)
    torchvision.datasets.CIFAR10(root=CIFAR_ROOT, train=False, download=True)
    print(f"  OK in {time.time()-t0:.0f}s", flush=True)
except Exception as e:
    print(f"  FAIL: {e}", flush=True)

print("\nHF datasets...", flush=True)
from datasets import load_dataset

for spec in [
    ("wikitext", "wikitext-2-raw-v1"),
    ("glue", "sst2"),
]:
    print(f"  {spec} ...", flush=True)
    t0 = time.time()
    try:
        load_dataset(*spec)
        print(f"    OK in {time.time()-t0:.0f}s", flush=True)
    except Exception as e:
        print(f"    FAIL: {e}", flush=True)

print("\nDone.", flush=True)
