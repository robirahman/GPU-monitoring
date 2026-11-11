#!/usr/bin/env python3
"""Common physical markers for clock alignment, and a sustained full-load probe.

A marker is a short, distinctive GPU load pattern (dense bf16 matmul bursts of
0.3 / 0.6 / 1.2 s separated by 1.0 s idle, repeated twice). Each burst edge is
timestamped twice: host wall/mono ns when enqueued, and the GPU execution time
via CUDA events (events.py). The scope, NVML, DCGM and BMC streams all see the
same physical edges, so offsets are estimated from repeated markers at both
ends of an acquisition instead of from logger start times.

  python -m h200_campaign.markers --events E.jsonl --tag start
  python -m h200_campaign.markers --events E.jsonl --sustain 60   # saturation probe
"""
from __future__ import annotations

import argparse
import time

import torch

from h200_campaign.events import EventLog

PATTERN = (0.3, 0.6, 1.2)
GAP = 1.0
REPEAT = 2


def burst(a, b, seconds, log, name, **kv):
    log.emit(f"{name}_start", gpu_mark=True, _depth=2, **kv)
    t_end = time.monotonic() + seconds
    n = 0
    while time.monotonic() < t_end:
        torch.matmul(a, b)
        n += 1
        if n % 8 == 0:
            # bound the queue so the GPU burst ends close to the host deadline
            torch.cuda.current_stream().synchronize()
    log.emit(f"{name}_end", gpu_mark=True, _depth=2, matmuls=n, **kv)
    torch.cuda.synchronize()
    return n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--events", required=True)
    ap.add_argument("--tag", default="marker")
    ap.add_argument("--sustain", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=8192)
    args = ap.parse_args()
    log = EventLog(args.events, stream=f"marker_{args.tag}", gpu=True)
    a = torch.randn(args.n, args.n, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(args.n, args.n, device="cuda", dtype=torch.bfloat16)
    torch.matmul(a, b)
    torch.cuda.synchronize()
    time.sleep(GAP)
    log.emit("marker_sequence_start", tag=args.tag, pattern=PATTERN, gap=GAP, repeat=REPEAT, sustain=args.sustain)
    if args.sustain > 0:
        burst(a, b, args.sustain, log, "sustain", tag=args.tag)
    else:
        for r in range(REPEAT):
            for i, d in enumerate(PATTERN):
                burst(a, b, d, log, "marker_burst", tag=args.tag, rep=r, idx=i, dur=d)
                time.sleep(GAP)
    log.emit("marker_sequence_end", tag=args.tag)
    log.close()


if __name__ == "__main__":
    main()
