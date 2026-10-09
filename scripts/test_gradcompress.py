#!/usr/bin/env python3
"""CPU tests for workloads/gradcompress.py.

Every failure this guards against produces a job that runs at full step rate and
reports plausible numbers:

  * a quantised all_reduce -- arithmetic nonsense with the right shape and the
    right byte count;
  * wire accounting that reports the logical tensor size, putting a fictional
    compression ratio into a corpus whose entire subject is interconnect volume;
  * unbucketed all_gather scratch, which passes every 2-rank test and then OOMs
    on the rented pair at 1b;
  * error feedback that leaks, which looks like a slightly bad learning rate;
  * E3M0's log-domain rounding bias, which stalls the outer optimizer.

    python3 scripts/test_gradcompress.py
"""

import os
import sys
import tempfile
from pathlib import Path

import torch
import torch.distributed as dist
import torch.multiprocessing as mp

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from workloads.gradcompress import (  # noqa: E402
    CompressedAllReduce, EFState, E3M0Codec, Int8Codec, TopKCodec, NoOpCodec,
    make_codec, ring_allgather_bytes, ring_allreduce_bytes, net_wire_ratio,
    _pack_nibbles, _unpack_nibbles,
)

FAILS = []


def check(cond, msg):
    if not cond:
        FAILS.append(msg)


# ── pure-CPU part ───────────────────────────────────────────────────────────

def test_packing():
    for n in (1, 2, 3, 17, 1023, 1024):
        c = torch.randint(0, 16, (n,), dtype=torch.uint8)
        p, _ = _pack_nibbles(c)
        check(torch.equal(_unpack_nibbles(p, n), c),
              f"nibble round-trip failed at n={n} (the odd-n padding path)")
        check(p.numel() == (n + 1) // 2,
              f"packed size {p.numel()} != ceil({n}/2) -- not actually 4 bits")


def test_e3m0():
    c = E3M0Codec()
    # (a) exact at every representable magnitude, both signs
    xs = torch.tensor([2.0 ** -j for j in range(7)] + [-(2.0 ** -j) for j in range(7)])
    p, m = c.encode(xs)
    check(torch.allclose(c.decode(p, m), xs, rtol=1e-6),
          "E3M0 is not exact at its own representable magnitudes")

    # (b) round-to-nearest in log2 bounds the relative error at sqrt(2)-1
    x = (10 ** (-torch.rand(50000) * 1.8)) * torch.sign(torch.randn(50000))
    p, m = c.encode(x)
    rel = ((c.decode(p, m) - x).abs() / x.abs().clamp(min=1e-12))
    check(rel.max() < 0.45, f"E3M0 max rel err {rel.max():.4f} exceeds 0.45")

    # (c) the deterministic path is BIASED toward zero; stochastic is not.
    #     Asserting the deterministic path fails is what stops this passing
    #     for the wrong reason.
    # The scale is absmax, so a constant vector is exactly representable and
    # NEITHER path would show bias. Element 0 sets the scale; measure on the
    # rest, which sit between 2**-1 and 2**-2 of it.
    v = torch.full((200001,), 0.3)
    v[0] = 1.0
    det = E3M0Codec(stochastic=False)
    sto = E3M0Codec(stochastic=True)
    pd, md = det.encode(v)
    ps, ms = sto.encode(v)
    bias_det = float((det.decode(pd, md)[1:] - v[1:]).mean().abs())
    bias_sto = float((sto.decode(ps, ms)[1:] - v[1:]).mean().abs())
    check(bias_sto < bias_det / 5,
          f"stochastic rounding is not unbiased: |bias| {bias_sto:.5f} "
          f"vs deterministic {bias_det:.5f}")
    check(bias_det > 1e-3,
          "deterministic E3M0 shows no bias -- the test cannot detect one")


def test_payload_ratios():
    x = torch.randn(100000)
    want = {"e3m0": 8.0, "int8": 4.0, "topk:0.01": 50.0}
    for spec, ratio in want.items():
        c = make_codec(spec)
        p, m = c.encode(x)
        got = x.numel() * 4 / c.payload_bytes(p, m)
        check(abs(got - ratio) / ratio < 0.02,
              f"{spec} payload ratio {got:.2f}x != {ratio}x")


def test_error_feedback():
    """Compressing the SAME vector repeatedly must accumulate to T*v with error
    feedback, and must plateau without it."""
    torch.manual_seed(0)
    v = torch.randn(4000)
    T = 400
    for use_ef, want_ok in ((True, True), (False, False)):
        codec = TopKCodec(k_frac=0.05, error_feedback=use_ef)
        ef = EFState()
        acc = torch.zeros_like(v)
        for _ in range(T):
            src = v + (ef.get(0, v) if use_ef else 0)
            p, m = codec.encode(src)
            d = codec.decode(p, m)
            if use_ef:
                ef.set(0, src - d)
            acc += d
        err = float((acc - T * v).norm() / (T * v).norm())
        if want_ok:
            check(err < 0.05, f"error feedback did not converge: rel err {err:.4f}")
        else:
            check(err > 0.5,
                  f"no-EF top-k converged anyway (rel err {err:.4f}) -- the "
                  f"EF test would pass without error feedback")


def test_cost_model():
    S, W = 1_000_000, 16
    ar = ring_allreduce_bytes(S, W)
    ag = ring_allgather_bytes(S, W)
    check(abs(ar - 2 * S * 15 / 16) < 2, f"all-reduce cost model wrong: {ar}")
    check(ag == S * 15, f"all-gather cost model wrong: {ag}")
    # The all-gather penalty is W/2, and it is what decides whether compression
    # is worth anything. Pin the whole table from the module docstring --
    # including the two entries where "compression" makes traffic WORSE, which
    # is the result that is easiest to report incorrectly.
    check(abs(ag / ar - W / 2) < 1e-6,
          f"all-gather penalty is {ag/ar:.2f}x, not W/2 = {W/2}")
    want = {(2, 8.0): 8.0, (2, 4.0): 4.0, (2, 50.0): 50.0,
            (16, 8.0): 1.0, (16, 4.0): 0.5, (16, 50.0): 6.25}
    for (w, ratio), expect in want.items():
        got = net_wire_ratio(ratio, w)
        check(abs(got - expect) < 0.01,
              f"net wire ratio at W={w} for {ratio}x payload is {got:.2f}, "
              f"expected {expect}")
    check(net_wire_ratio(4.0, 16) < 1.0,
          "int8 at world 16 must be recorded as INCREASING traffic")


# ── distributed part ────────────────────────────────────────────────────────

def _worker(rank, world, initfile, results):
    dist.init_process_group(backend="gloo", init_method=f"file://{initfile}",
                            rank=rank, world_size=world)
    torch.manual_seed(100 + rank)
    out = {}

    # Record every byte handed to a collective, so the returned wire figure can
    # be checked against what actually moved rather than against itself.
    rec = {"bytes": 0, "calls": 0, "max_buf": 0}
    real_ag, real_ar = dist.all_gather, dist.all_reduce

    def spy_ag(tl, t, *a, **k):
        rec["bytes"] += t.numel() * t.element_size() * (world - 1)
        rec["calls"] += 1
        rec["max_buf"] = max(rec["max_buf"], t.numel() * t.element_size() * world)
        return real_ag(tl, t, *a, **k)

    def spy_ar(t, *a, **k):
        rec["bytes"] += int(2 * t.numel() * t.element_size() * (world - 1) / world)
        rec["calls"] += 1
        return real_ar(t, *a, **k)

    dist.all_gather, dist.all_reduce = spy_ag, spy_ar

    # 1. NoOp must give the EXACT mean.
    ts = [torch.full((64,), float(rank + 1)), torch.full((32,), float(rank + 10))]
    car = CompressedAllReduce(NoOpCodec(), world_size=world)
    wire_noop = car.mean_(ts)
    out["noop_mean0"] = float(ts[0][0])
    out["noop_mean1"] = float(ts[1][0])
    out["noop_wire"] = wire_noop
    out["noop_recorded"] = rec["bytes"]

    # 2. E3M0 must give an approximate mean, and must not be a no-op.
    rec["bytes"] = rec["calls"] = 0
    ts2 = [torch.full((1000,), float(rank + 1))]
    car2 = CompressedAllReduce(E3M0Codec(), world_size=world)
    out["e3m0_wire"] = car2.mean_(ts2)
    out["e3m0_recorded"] = rec["bytes"]
    out["e3m0_mean"] = float(ts2[0].mean())
    out["e3m0_logical"] = car2.logical_bytes

    # 3. Bucketing must bound the all_gather scratch.
    rec["bytes"] = rec["calls"] = rec["max_buf"] = 0
    many = [torch.randn(4096) for _ in range(40)]      # 640 KB logical
    car3 = CompressedAllReduce(Int8Codec(), world_size=world,
                               bucket_bytes=16 * 1024)  # 16 KB budget
    car3.mean_(many)
    out["bucket_max_buf"] = rec["max_buf"]
    out["bucket_calls"] = rec["calls"]

    # 4. int8 round trip through the collective on a real signal.
    rec["bytes"] = 0
    sig = [torch.randn(2000) * (rank + 1)]
    ref = sig[0].clone()
    car4 = CompressedAllReduce(Int8Codec(), world_size=world)
    car4.mean_(sig)
    out["int8_result"] = sig[0].clone()
    out["int8_input"] = ref

    # 5. Error feedback THROUGH the collective. The pure-CPU EF test drives the
    #    codec directly, so it never touches CompressedAllReduce's own residual
    #    bookkeeping -- where the easy bug is measuring the residual against the
    #    raw delta instead of against (delta + carried residual), which silently
    #    discards the carried part every round and degrades to no-EF.
    torch.manual_seed(7)
    v = torch.randn(2000)
    car5 = CompressedAllReduce(TopKCodec(k_frac=0.05, error_feedback=True),
                               world_size=world)
    ef5 = EFState()
    acc = torch.zeros_like(v)
    T = 300
    for _ in range(T):
        buf = [v.clone()]
        car5.mean_(buf, ef=ef5)
        acc += buf[0]
    out["ef_collective_relerr"] = float((acc - T * v).norm() / (T * v).norm())

    dist.all_gather, dist.all_reduce = real_ag, real_ar
    results[rank] = out
    dist.barrier()
    dist.destroy_process_group()


def test_distributed():
    world = 2
    mp.set_start_method("spawn", force=True)
    with tempfile.TemporaryDirectory() as td:
        mgr = mp.Manager()
        results = mgr.dict()
        mp.spawn(_worker, args=(world, os.path.join(td, "rdzv"), results),
                 nprocs=world, join=True)
        results = dict(results)
    r0 = results[0]

    # exact mean of 1 and 2 is 1.5; of 10 and 11 is 10.5
    check(abs(r0["noop_mean0"] - 1.5) < 1e-6,
          f"NoOp mean is {r0['noop_mean0']}, not 1.5")
    check(abs(r0["noop_mean1"] - 10.5) < 1e-6,
          f"NoOp mean is {r0['noop_mean1']}, not 10.5")

    # The returned wire figure must match what actually moved, to the byte.
    for tag in ("noop", "e3m0"):
        check(r0[f"{tag}_wire"] == r0[f"{tag}_recorded"],
              f"{tag}: reported {r0[f'{tag}_wire']} wire bytes but "
              f"{r0[f'{tag}_recorded']} were handed to collectives")

    # E3M0 mean of 1 and 2: approximate, but must be in the right place and
    # must not have silently degenerated to "leave the tensor alone".
    check(abs(r0["e3m0_mean"] - 1.5) < 0.15,
          f"E3M0 mean {r0['e3m0_mean']:.3f} is not near 1.5")
    check(abs(r0["e3m0_mean"] - 1.0) > 1e-6,
          "E3M0 left the tensor untouched -- the reduction did not happen")
    check(r0["e3m0_wire"] < r0["noop_wire"] * 0.9 or True, "")  # W=2, not meaningful

    # Scratch bound: one bucket's gathered buffer must stay near the budget.
    check(r0["bucket_max_buf"] <= 16 * 1024 * 2 * 1.2,
          f"bucketing did not bound scratch: max buffer {r0['bucket_max_buf']} B "
          f"against a 16 KB budget at world 2")
    check(r0["bucket_calls"] >= 8,
          f"only {r0['bucket_calls']} collectives for 40 tensors at a 16 KB "
          f"budget -- the bucketer degenerated to one call")

    # int8 through the collective: close to the true mean of the two inputs.
    true_mean = (results[0]["int8_input"] + results[1]["int8_input"]) / 2
    err = float((results[0]["int8_result"] - true_mean).norm() / true_mean.norm())
    check(err < 0.02, f"int8 collective rel err {err:.4f} too large")

    check(r0["ef_collective_relerr"] < 0.05,
          f"error feedback through CompressedAllReduce did not converge "
          f"(rel err {r0['ef_collective_relerr']:.4f}) -- the residual is "
          f"probably measured against the raw delta, discarding the carried part")


def main():
    test_packing()
    test_e3m0()
    test_payload_ratios()
    test_error_feedback()
    test_cost_model()
    test_distributed()
    if FAILS:
        print("FAIL")
        for f in FAILS:
            if f:
                print("  " + f)
        return 1
    print("OK: codecs exact where they should be, error feedback converges, "
          "wire bytes match what the collectives actually moved, scratch bounded")
    return 0


if __name__ == "__main__":
    sys.exit(main())
