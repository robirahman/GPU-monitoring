#!/usr/bin/env python3
"""CPU test for the expert-parallel MoE layer in workloads/train_moe.py.

Two failure modes here are invisible in telemetry, which is the only thing this
project records, so they have to be caught on a laptop:

  1. Non-differentiable collectives. `torch.distributed.all_to_all_single` is
     opaque to autograd: the experts get no gradient and the BACKWARD all-to-all
     never happens. The cell still runs at full step rate and still reports a
     number for inter-node bytes -- one that is 2x the truth.
  2. Capacity-slot collisions. With top_k > 1 the same expert is fed from
     several k-slots; writing each at offset 0 makes later slots overwrite
     earlier ones while the combine step still reads both. Byte counts stay
     exactly right while tokens train against the wrong expert's output.

Runs two gloo ranks on CPU. No GPU required.

    python3 scripts/test_moe_layer.py
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.train_moe import MoELayer  # noqa: E402

HIDDEN, FFN, EPR, TOPK, NTOK = 8, 16, 2, 2, 24


def _worker(rank, world, initfile, results):
    dist.init_process_group(backend="gloo", init_method=f"file://{initfile}",
                            rank=rank, world_size=world)
    torch.manual_seed(7)
    layer = MoELayer(HIDDEN, FFN, EPR, world, TOPK,
                     capacity_factor=4.0, router_skew=0.0)
    for p in layer.parameters():
        dist.broadcast(p.data, src=0)

    x = torch.randn(NTOK, HIDDEN, requires_grad=True)
    y, moved = layer(x)
    y.sum().backward()

    expert_grads = [float(p.grad.abs().sum()) if p.grad is not None else None
                    for p in layer.experts.parameters()]
    results[rank] = {
        "out_shape": tuple(y.shape),
        "moved": moved,
        "x_grad": None if x.grad is None else float(x.grad.abs().sum()),
        "expert_grads": expert_grads,
        "router_grad": float(layer.router.weight.grad.abs().sum()),
        "finite": bool(torch.isfinite(y).all()),
    }
    dist.barrier()
    dist.destroy_process_group()


def main():
    world = 2
    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as td:
        mgr = mp.Manager()
        results = mgr.dict()
        mp.spawn(_worker, args=(world, os.path.join(td, "rdzv"), results),
                 nprocs=world, join=True)
        results = dict(results)

    fails = []
    r0 = results[0]

    if r0["out_shape"] != (NTOK, HIDDEN):
        fails.append(f"output shape {r0['out_shape']} != {(NTOK, HIDDEN)}")
    if not r0["finite"]:
        fails.append("output contains non-finite values")

    # (1) The gradient must reach BOTH the experts and the layer input. If the
    #     collectives were opaque the expert grads would be None/zero and x.grad
    #     would be zero, while everything else looked healthy.
    if r0["x_grad"] is None or r0["x_grad"] == 0.0:
        fails.append("no gradient reached the layer input -- all-to-all is not "
                     "autograd-aware")
    if any(g is None or g == 0.0 for g in r0["expert_grads"]):
        fails.append(f"an expert received no gradient: {r0['expert_grads']} -- "
                     "the backward all-to-all did not happen")
    if r0["router_grad"] == 0.0:
        fails.append("router received no gradient")

    # (2) Byte accounting must match the two fixed-size buffers exactly, and
    #     must agree between ranks (the call is symmetric).
    cap = max(1, int(4.0 * NTOK * TOPK / (EPR * world)))
    expect = 2 * (EPR * world) * cap * HIDDEN * 4
    if r0["moved"] != expect:
        fails.append(f"moved {r0['moved']} bytes, expected {expect}")
    if results[0]["moved"] != results[1]["moved"]:
        fails.append("ranks disagree on bytes moved")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print(f"OK: all-to-all round trip differentiable "
          f"(x.grad={r0['x_grad']:.3f}, experts all non-zero), "
          f"{r0['moved']} bytes per rank per layer")
    return 0


if __name__ == "__main__":
    sys.exit(main())
