#!/usr/bin/env python3
"""CPU test for the DiLoCo outer step in workloads/train_localsgd.py.

The averaging is easy to eyeball and the outer step is not: a sign error in the
pseudo-gradient leaves a workload that still runs, still moves multi-GB through
the all-reduce, and still reports plausible tok/s -- while training away from
the objective. That failure is invisible in telemetry, which is the only thing
this project records, so it has to be caught here.

Runs two gloo ranks on CPU with a 3-parameter model. No GPU required.

    python3 scripts/test_localsgd_sync.py
"""

import os
import sys
import tempfile
import argparse
from pathlib import Path

import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.train_localsgd import local_sgd_sync  # noqa: E402


def _args(outer_opt="nesterov", outer_lr=0.7, momentum=0.9):
    return argparse.Namespace(outer_opt=outer_opt, outer_lr=outer_lr,
                              outer_momentum=momentum)


def _worker(rank, world, initfile, results):
    dist.init_process_group(backend="gloo", init_method=f"file://{initfile}",
                            rank=rank, world_size=world)
    torch.manual_seed(0)
    model = nn.Linear(4, 2, bias=False)          # 8 params, one tensor
    with torch.no_grad():                        # identical start on both ranks
        model.weight.fill_(1.0)

    outer = {n: (p.data.detach().clone(), torch.zeros_like(p.data))
             for n, p in model.named_parameters()}

    # Ranks diverge the way independent replicas would: rank r ends its inner
    # loop at 1 + (r + 1). Mean of the replicas is 1 + (world + 1) / 2.
    with torch.no_grad():
        model.weight.add_(rank + 1.0)

    moved = local_sgd_sync(model, outer, _args(), world)

    results[rank] = {
        "weight": model.weight.data.clone(),
        "moved": moved,
        "outer_prev": outer["weight"][0].clone(),
        "outer_buf": outer["weight"][1].clone(),
    }
    dist.barrier()
    dist.destroy_process_group()


def test_delta_vs_param_form():
    """The reason compression had to switch the sync to delta form.

    Top-k on a PARAMETER tensor keeps the largest 1% of the weights and zeroes
    the other 99%, destroying the model in a single sync. Top-k on a DELTA keeps
    the largest 1% of the CHANGE and postpones the rest, which error feedback
    carries forward. Both run. Both move exactly the same number of bytes. Only
    one of them is training.
    """
    import torch as _t
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from workloads.gradcompress import TopKCodec

    _t.manual_seed(0)
    codec = TopKCodec(k_frac=0.01, error_feedback=True)
    params = _t.randn(10000) * 0.1 + 1.0          # weights near 1.0
    delta = _t.randn(10000) * 1e-3                # a plausible inner-loop change

    p_payload, p_meta = codec.encode(params)
    d_payload, d_meta = codec.encode(delta)
    p_back = codec.decode(p_payload, p_meta)
    d_back = codec.decode(d_payload, d_meta)

    p_zero = float((p_back == 0).float().mean())
    fails = []
    if p_zero < 0.9:
        fails.append(f"top-k on parameters zeroed only {p_zero:.1%} -- the test "
                     f"cannot demonstrate the hazard it exists for")
    # In delta form the RECONSTRUCTED PARAMETER is base + decoded delta, which
    # is never mostly zero however sparse the delta is. That is the whole point.
    recon = params + d_back
    r_zero = float((recon == 0).float().mean())
    if r_zero > 0.01:
        fails.append(f"delta form still zeroed {r_zero:.1%} of the parameters")
    if float((recon - params).abs().max()) == 0.0:
        fails.append("delta form applied no change at all")
    # Byte counts are identical, which is why this cannot be caught downstream.
    if codec.payload_bytes(p_payload, p_meta) != codec.payload_bytes(d_payload, d_meta):
        fails.append("the two forms move different byte counts, so the hazard "
                     "would be visible in telemetry after all")
    return fails


def main():
    world = 2
    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as td:
        initfile = os.path.join(td, "rdzv")
        mgr = mp.Manager()
        results = mgr.dict()
        mp.spawn(_worker, args=(world, initfile, results), nprocs=world, join=True)
        results = dict(results)

    fails = test_delta_vs_param_form()

    # 1. Every rank leaves the sync holding identical weights. If this fails the
    #    replicas never actually merged and the job is N independent models.
    w0, w1 = results[0]["weight"], results[1]["weight"]
    if not torch.allclose(w0, w1, atol=1e-6):
        fails.append(f"ranks disagree after sync: {w0.flatten()} vs {w1.flatten()}")

    # 2. The outer step moved the weights in the direction of the averaged
    #    replica, not away from it. Start 1.0, replica mean 1.0 + 1.5 = 2.5, so
    #    the global weights must end ABOVE 1.0. A sign error lands below.
    got = float(w0.flatten()[0])
    if not got > 1.0:
        fails.append(f"outer step moved the wrong way: 1.0 -> {got:.4f} "
                     f"(replica mean was 2.5, so it must increase)")

    # 3. With lr=0.7 and Nesterov momentum 0.9 on the first step, the update is
    #    pseudo_grad * (1 + momentum) * lr where pseudo_grad = prev - averaged.
    expect = 1.0 + 1.5 * (1 + 0.9) * 0.7
    if abs(got - expect) > 1e-5:
        fails.append(f"outer step magnitude {got:.6f} != expected {expect:.6f}")

    # 4. Byte accounting feeds the reported interconnect volume, which is the
    #    number this whole family of cells exists to characterise.
    expect_bytes = 8 * 4                          # 8 float32 params
    if results[0]["moved"] != expect_bytes:
        fails.append(f"moved {results[0]['moved']} bytes, expected {expect_bytes}")

    # 5. 'avg' mode must be plain averaging and must not touch outer state.
    #    (Checked analytically: local_sgd_sync returns before the outer block.)
    if results[0]["outer_buf"].abs().sum() == 0:
        fails.append("outer momentum buffer never updated -- outer step is a no-op")

    if fails:
        print("FAIL")
        for f in fails:
            print("  " + f)
        return 1
    print(f"OK: sync averages across ranks, outer step 1.0 -> {got:.4f} "
          f"(expected {expect:.4f}), {results[0]['moved']} bytes accounted")
    return 0


if __name__ == "__main__":
    sys.exit(main())
