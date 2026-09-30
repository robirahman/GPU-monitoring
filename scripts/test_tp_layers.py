#!/usr/bin/env python3
"""CPU test for the Megatron f/g operators in workloads/train_tp.py.

`_CopyToTP` (identity forward, all-reduce backward) and `_ReduceFromTP`
(all-reduce forward, identity backward) are duals, and swapping them does not
crash: the shapes still line up, the collectives still fire at the same sizes,
the step rate is unchanged, and the telemetry -- the only thing this project
records -- is indistinguishable. The model just trains wrong. So the test checks
the NUMBERS, against a dense reference assembled from the shards themselves.

Runs two gloo ranks on CPU. No GPU required.

    python3 scripts/test_tp_layers.py
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.nn.functional as F
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.train_tp import TPMLP, TPAttention, tp_bytes  # noqa: E402

HIDDEN, FFN, HEADS, B, T = 16, 32, 4, 2, 8


def _worker(rank, world, initfile, results):
    dist.init_process_group(backend="gloo", init_method=f"file://{initfile}",
                            rank=rank, world_size=world)
    group = dist.new_group(ranks=list(range(world)))

    torch.manual_seed(100 + rank)                 # different shards per rank
    mlp = TPMLP(HIDDEN, FFN, world, rank, group)
    attn = TPAttention(HIDDEN, HEADS, world, rank, group)

    torch.manual_seed(0)                          # same input on every rank
    x = torch.randn(B, T, HIDDEN, requires_grad=True)

    y = mlp(x)
    y.sum().backward()
    x_grad = x.grad.clone()

    x2 = x.detach().clone().requires_grad_(True)
    a = attn(x2)
    a.sum().backward()

    # Gather the shards so rank 0 can rebuild the equivalent dense MLP.
    w1 = [torch.zeros_like(mlp.fc1.weight) for _ in range(world)]
    w2 = [torch.zeros_like(mlp.fc2.weight) for _ in range(world)]
    dist.all_gather(w1, mlp.fc1.weight.detach())
    dist.all_gather(w2, mlp.fc2.weight.detach())

    results[rank] = {
        "mlp_out": y.detach().clone(),
        "attn_out": a.detach().clone(),
        "x_grad": x_grad,
        "attn_x_grad": x2.grad.clone(),
        "bytes": tp_bytes(),
        "w1": torch.cat(w1, dim=0),               # (FFN, HIDDEN)
        "w2": torch.cat(w2, dim=1),               # (HIDDEN, FFN)
        "x": x.detach().clone(),
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
    r0, r1 = results[0], results[1]

    # 1. After the row-parallel all-reduce every rank must hold the SAME output.
    #    If g were missing or replaced by identity, each rank would keep only its
    #    own partial sum and they would differ.
    if not torch.allclose(r0["mlp_out"], r1["mlp_out"], atol=1e-5):
        fails.append("MLP output differs across ranks -- the row-parallel "
                     "all-reduce did not happen")
    if not torch.allclose(r0["attn_out"], r1["attn_out"], atol=1e-5):
        fails.append("attention output differs across ranks")

    # 2. The numbers must match a dense MLP built from the gathered shards.
    #    This is what catches an f/g swap: the shapes and the byte counts survive
    #    it, the values do not.
    dense = F.linear(F.gelu(F.linear(r0["x"], r0["w1"])), r0["w2"])
    if not torch.allclose(dense, r0["mlp_out"], atol=1e-4):
        err = (dense - r0["mlp_out"]).abs().max().item()
        fails.append(f"TP MLP != dense equivalent (max abs err {err:.2e})")

    # 3. Both ranks must agree on the input gradient too -- _CopyToTP's backward
    #    all-reduce is the only thing that makes that true.
    if not torch.allclose(r0["x_grad"], r1["x_grad"], atol=1e-5):
        fails.append("input gradients differ across ranks -- _CopyToTP's "
                     "backward all-reduce did not happen")
    if r0["x_grad"].abs().sum() == 0:
        fails.append("input gradient is zero")
    if r0["attn_x_grad"].abs().sum() == 0:
        fails.append("attention input gradient is zero")

    # 4. Bytes must be counted, and counted the same on both ranks.
    if r0["bytes"] == 0:
        fails.append("no TP bytes accounted -- the counters never fired")
    if r0["bytes"] != r1["bytes"]:
        fails.append(f"byte counts disagree: {r0['bytes']} vs {r1['bytes']}")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print(f"OK: TP MLP matches its dense equivalent "
          f"(max err {(dense - r0['mlp_out']).abs().max():.2e}), ranks agree on "
          f"output and input gradient, {r0['bytes']} bytes counted per rank")
    return 0


if __name__ == "__main__":
    sys.exit(main())
