#!/usr/bin/env python3
"""Prove the slab-transpose algebra in hpc_distributed.py --mode fftshard.

There is no GPU and no torch on the analysis box, and a wrong transpose does not
crash -- it produces a plausible-looking field that is simply not the transform
of the input. On a rented pair that costs a cell to discover and might never be
noticed at all, because the workload exists to move bytes and would move exactly
as many bytes while computing nonsense.

So the index algebra is checked here, in numpy, by simulating all_to_all_single
across `world` virtual ranks and comparing the distributed result against a
single-process numpy fftn. What this does NOT check is the torch API surface
(dtype handling, contiguity, NCCL itself); that is what the 1-GPU smoke run is
for. What it does check is the only part that is easy to get wrong and hard to
see: which block goes to which rank, and how the axes fold back.

    python3 workloads/test_hpc_distributed_transpose.py
"""
import sys

import numpy as np


def all_to_all(sendbufs):
    """Simulate dist.all_to_all_single across `world` ranks.

    Semantics being modelled: each rank's input is split along dim 0 into
    `world` equal chunks, chunk p is delivered to rank p, and on the receiving
    side chunk q is whatever rank q sent. So out[r][q] = sendbufs[q][r].
    """
    world = len(sendbufs)
    return [np.stack([sendbufs[q][r] for q in range(world)]) for r in range(world)]


def distributed_fftn(full, world):
    """Run the fft_slab_step index dance over `world` virtual ranks."""
    n = full.shape[0]
    nloc, nyloc = n // world, n // world

    # Scatter into x-slabs, the decomposition the solver starts and ends in.
    a = [full[r * nloc:(r + 1) * nloc].copy() for r in range(world)]

    # 1. Local transforms over the two resident axes.
    buf = [np.fft.fftn(a[r], axes=(1, 2)) for r in range(world)]

    # 2. Forward transpose: x-slabs -> y-slabs.
    send = [buf[r].reshape(nloc, world, nyloc, n).transpose(1, 0, 2, 3)
            for r in range(world)]
    work = all_to_all(send)

    # 3. The transform that required the exchange.
    ypencil = [np.fft.fft(work[r].reshape(world * nloc, nyloc, n), axis=0)
               for r in range(world)]

    # 4. Inverse transpose: y-slabs -> x-slabs.
    back = all_to_all([ypencil[r].reshape(world, nloc, nyloc, n) for r in range(world)])
    out = [back[r].transpose(1, 0, 2, 3).reshape(nloc, n, n) for r in range(world)]

    return np.concatenate(out, axis=0)


def main():
    rng = np.random.default_rng(0)
    failures = []
    for n, world in ((8, 2), (8, 4), (16, 4), (16, 8), (12, 4)):
        full = (rng.standard_normal((n, n, n))
                + 1j * rng.standard_normal((n, n, n))).astype(np.complex64)
        want = np.fft.fftn(full, axes=(0, 1, 2))
        got = distributed_fftn(full, world)
        err = np.abs(got - want).max() / max(np.abs(want).max(), 1e-12)
        ok = err < 1e-4          # complex64 over an n^3 reduction
        print(f"  grid {n}^3, world {world:2d}: max rel err {err:.2e}  "
              f"{'OK' if ok else 'MISMATCH'}")
        if not ok:
            failures.append((n, world, err))

    # The decomposition must also round-trip the ORIGINAL layout: rank r has to
    # end holding x-range r, or the next timestep reads someone else's subdomain.
    n, world = 16, 4
    full = (rng.standard_normal((n, n, n))
            + 1j * rng.standard_normal((n, n, n))).astype(np.complex64)
    got = distributed_fftn(full, world)
    if got.shape != full.shape:
        failures.append(("shape", got.shape, full.shape))
        print(f"  layout: MISMATCH {got.shape} != {full.shape}")
    else:
        print(f"  layout: result is {got.shape}, same decomposition as the input  OK")

    if failures:
        print(f"\nFAILED: {failures}")
        return 1
    print("\nOK: the distributed transpose reproduces a single-process fftn, and "
          "returns to the input decomposition.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
