#!/usr/bin/env python3
"""JAX non-ML filler for the TPU rung -- the disguise's cover work.

Ported from `NonMLFiller` in workloads/adversarial_multinode.py, carrying forward
the lesson that invalidated the first GPU ladder: **a filler that does not
saturate the accelerator is not a disguise.** The original hashed a 256x8 tensor
per round and reallocated its seeds every step, so it was kernel-launch bound and
ran at 55% utilisation / 160 W while the honest non-ML class it was imitating sat
at 91.7% / 509 W (cufft_benchmark). Being detected under those conditions measured
nothing about evasion. After resizing to ~2000x the work per launch it reached
96.6% / 541 W and the disguise started evading.

Two TPU-specific constraints shape this port:

1. **Static shapes or nothing.** XLA compiles per shape. Any data-dependent shape
   triggers a recompilation, and a 5-10 minute XLA compile in the middle of a
   600 s cell would dominate the telemetry we are trying to measure. Every buffer
   here is allocated once at fixed shape and every kernel is jitted once.
2. **Power is invisible.** The GPU sizing target (96.6% util, 541 W) is not
   transferable: `scripts/collect_tpu_telemetry.py:74-78` hard-wires power,
   temperature and both clocks to -1. Size against duty cycle and HBM-bandwidth
   utilisation instead, and calibrate empirically against
   `tpu_jax_other.py --mode fft --sharded`, which is the honest non-ML cell this
   filler is imitating.

Kernel modes exist because on GPU the kernel axis mattered more than duty or
allocation: at identical duty, hash-only vs fft-only moved held-out p_train 3.5x
(0.671 vs 0.212), making memory-bound periodic work the strongest false positive.

CPU gate (no TPU, $0):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python workloads/tpu_nonml_filler.py --self-test
"""

import argparse
import sys
import time

try:
    import jax
    import jax.numpy as jnp
    _HAS_JAX = True
except ImportError:  # importable on a laptop for --help / self-test skip
    _HAS_JAX = False

MODES = ("mixed", "hash", "fft")


class JaxNonMLFiller:
    """Sharded, saturating, non-ML compute. No model, no gradients.

    `table` is the memory-hard gather target (the ethash "DAG" analogue) and is
    sharded over the mesh axis so every chip owns a slice and the gather is a
    real HBM-bandwidth load rather than a broadcast read. `fft_buf` is the
    compute-bound half, shaped like a batched 2-D transform, which is what
    cufft_benchmark does on the GPU side.
    """

    def __init__(self, mesh, axis, *, table_mb=512, hash_rows=1 << 20,
                 fft_batch=32, fft_size=1024, mode="mixed", seed=0):
        assert mode in MODES, mode
        self.mesh, self.axis, self.mode = mesh, axis, mode
        P, NS = jax.sharding.PartitionSpec, jax.sharding.NamedSharding
        n_dev = len(mesh.devices.flatten())

        # 16 int32 per row => 64 B per row, matching the GPU DAG layout.
        rows = max(n_dev, (table_mb * 1024 * 1024) // 64)
        rows -= rows % n_dev                      # divisible: keeps sharding even
        self.rows = rows
        k = jax.random.PRNGKey(seed)
        k_t, k_m, k_f = jax.random.split(k, 3)

        row_shard = NS(mesh, P(axis))             # shard dim 0 across chips
        repl = NS(mesh, P())
        self.table = jax.device_put(
            jax.random.randint(k_t, (rows, 16), 0, 2**31 - 1, dtype=jnp.int32),
            row_shard)
        # Mix buffer is preallocated and carried between steps. The GPU version's
        # per-step randint was pure launch overhead.
        hash_rows -= hash_rows % n_dev
        self.hash_rows = max(n_dev, hash_rows)
        self.mix = jax.device_put(
            jax.random.randint(k_m, (self.hash_rows, 8), 0, 2**31 - 1,
                               dtype=jnp.int32),
            row_shard)
        fft_batch -= fft_batch % n_dev
        fft_batch = max(n_dev, fft_batch)
        self.fft_buf = jax.device_put(
            jax.random.normal(k_f, (fft_batch, fft_size, fft_size),
                              dtype=jnp.float32).astype(jnp.complex64),
            row_shard)
        self.iters = 0

        # Both kernels run as PER-SHARD LOCAL work via shard_map, with NO
        # cross-device collective. The first version sharded the table on dim 0
        # and gathered with globally-sharded indices, which forces an all-gather:
        # on the CPU backend that deadlocked outright ("Expected 8 threads to
        # join the rendezvous, but only 7 arrived"), and on a real slice it would
        # have put ICI/DCN traffic inside the FILLER -- contaminating the very
        # channel the scope ladder measures. The GPU filler was per-rank local
        # work with no NCCL, and this now matches that semantics.
        # Reductions are also local: a global .sum() is itself a collective, so
        # each shard returns its own partial and the host just blocks on it.
        try:
            from jax.experimental.shard_map import shard_map
        except ImportError:      # relocated in newer jax
            shard_map = jax.shard_map
        from functools import partial
        spec = P(axis)

        def _hash_local(table_s, mix_s):
            # `& 0x7FFFFFFF` rather than abs(): abs(INT32_MIN) overflows and
            # would index negatively. Modulo the LOCAL shard length, so every
            # gather stays on-chip.
            for _ in range(8):
                idx = jnp.bitwise_and(mix_s[:, 0], 0x7FFFFFFF) % table_s.shape[0]
                mix_s = jnp.bitwise_xor(mix_s,
                                        jnp.take(table_s, idx, axis=0)[:, :8])
            return mix_s

        def _fft_local(buf_s):
            y = jnp.fft.fft2(buf_s)
            z = jnp.fft.ifft2(y)
            # per-shard partial sum, keeps a real data dependency, no collective
            return z.real.sum(axis=(1, 2), keepdims=False)[:, None]

        self._hash = jax.jit(shard_map(
            _hash_local, mesh=mesh, in_specs=(spec, spec), out_specs=spec,
            check_rep=False))
        self._fft = jax.jit(shard_map(
            _fft_local, mesh=mesh, in_specs=(spec,), out_specs=spec,
            check_rep=False))

        # Warm up BOTH kernels now. XLA compiles on first call, and an
        # uncompiled first iteration would (a) land a multi-second stall inside
        # the first duty slot and (b) poison the iter_cost EMA -- measured 176 ms
        # for a kernel whose steady-state cost is 0.58 ms. Compile once here,
        # outside the measured region, then reset the estimate.
        with mesh:
            jax.block_until_ready(self._hash(self.table, self.mix))
            jax.block_until_ready(self._fft(self.fft_buf))
        self._iter_est = None

    def bytes_resident(self) -> int:
        return (self.rows * 16 * 4 + self.hash_rows * 8 * 4
                + self.fft_buf.size * 8)

    def step(self):
        """One filler unit of work. Alternates kernels in 'mixed' mode."""
        do_hash = (self.mode == "hash"
                   or (self.mode == "mixed" and self.iters % 2 == 0))
        with self.mesh:
            if do_hash:
                self.mix = self._hash(self.table, self.mix)
                acc = self.mix
            else:
                acc = self._fft(self.fft_buf)
        self.iters += 1
        return acc

    def run_for(self, seconds: float) -> int:
        """Fill `seconds` of wall clock, without overshooting. Returns iters.

        Deadline-aware on purpose. The naive form -- `while time.time() < end:
        step()` -- checks the clock BEFORE starting an iteration, so one slow
        kernel blows straight through the budget: asked 0.30 s, got 1.61 s on the
        CPU gate. Duty cycle is the axis that produced the GPU false positives
        (an honest duty-cycled job scored 0.323 against the F=0.05 disguise's
        0.295), so a duty knob that silently does not hold would corrupt the
        honest controls rather than merely blur them.

        So: block after every iteration to know when work actually landed, keep
        an EMA of iteration cost, and decline to start an iteration that would
        not fit. Always runs at least one, so duty>0 always does something --
        which also means achievable duty precision is bounded by one iteration.
        Size the kernels so one iteration is well under `--slot-sec`; the
        self-test reports per-mode iteration cost for exactly this reason.
        """
        if seconds <= 0:
            return 0
        n0 = self.iters
        end = time.time() + seconds
        est = getattr(self, "_iter_est", None)
        while True:
            remaining = end - time.time()
            if self.iters > n0 and (remaining <= 0 or
                                    (est is not None and remaining < est * 0.75)):
                break
            t0 = time.time()
            acc = self.step()
            jax.block_until_ready(acc)
            dt = time.time() - t0
            est = dt if est is None else 0.7 * est + 0.3 * dt
            self._iter_est = est
            if self.iters > n0 and end - time.time() <= 0:
                break
        return self.iters - n0

    def iter_cost(self) -> float:
        """EMA seconds per filler iteration; duty precision is bounded by this."""
        return float(getattr(self, "_iter_est", 0.0) or 0.0)


def self_test() -> bool:
    """Tiny-shape correctness + saturation smoke on whatever backend exists."""
    if not _HAS_JAX:
        print("filler self-test: jax absent, not applicable")
        return True
    try:
        from workloads import tpu_transformer_common as T
    except ImportError:
        import tpu_transformer_common as T
    mesh, axis = T.make_mesh("fsdp")
    ok = True
    for mode in MODES:
        f = JaxNonMLFiller(mesh, axis, table_mb=4, hash_rows=1 << 12,
                           fft_batch=8, fft_size=64, mode=mode)
        n = f.run_for(0.5)
        # Per-mode iteration cost bounds achievable duty precision. Warmed up,
        # the CPU gate gives fft 0.25 ms and hash 0.61 ms at toy shapes, so the
        # kernels are comparable and "mixed" is a genuine alternation. Re-measure
        # with --calibrate at production sizes on the TPU smoke; TPU decomposes
        # FFT rather than running it natively, so the ratio need not transfer.
        print(f"    iter_cost={f.iter_cost()*1e3:8.2f} ms", flush=True)
        finite = bool(jnp.isfinite(jnp.asarray(f.step(), dtype=jnp.float64)).all())
        good = n > 0 and finite
        ok &= good
        print(f"  mode={mode:5s} iters={n:5d} resident={f.bytes_resident()/1e6:7.1f}MB "
              f"finite={finite} {'ok' if good else 'FAIL'}")
    print(f"filler self-test {'PASS' if ok else 'FAIL'}")
    return ok


def calibrate(table_mb, hash_rows, fft_batch, fft_size, secs=3.0):
    """Report steady-state cost per kernel at these sizes.

    The kernel axis only means something if the two kernels cost COMPARABLE time
    per iteration, and duty control only works if one iteration is well under a
    slot. Both hold comfortably on the CPU gate once the kernels are warmed up
    (fft 0.25 ms, hash 0.61 ms at toy shapes) -- but ONLY after warmup: the same
    fft measured 1423 ms/iter cold, which is all XLA compile and nothing else.
    That is why JaxNonMLFiller warms both kernels in __init__. TPU has no
    dedicated FFT unit (XLA decomposes it), so re-measure here at production
    sizes rather than assuming these ratios transfer.
    """
    try:
        from workloads import tpu_transformer_common as T
    except ImportError:
        import tpu_transformer_common as T
    mesh, axis = T.make_mesh("fsdp")
    print(f"  table_mb={table_mb} hash_rows={hash_rows} "
          f"fft_batch={fft_batch} fft_size={fft_size}")
    cost = {}
    for mode in ("hash", "fft"):
        f = JaxNonMLFiller(mesh, axis, table_mb=table_mb, hash_rows=hash_rows,
                           fft_batch=fft_batch, fft_size=fft_size, mode=mode)
        n = f.run_for(secs)
        cost[mode] = f.iter_cost()
        print(f"    {mode:5s} iters={n:6d} iter_cost={cost[mode]*1e3:9.3f} ms "
              f"resident={f.bytes_resident()/1e6:8.1f} MB")
    if min(cost.values()) > 0:
        r = max(cost.values()) / min(cost.values())
        print(f"    cost ratio fft/hash = {cost['fft']/max(cost['hash'],1e-12):.1f}x "
              f"(want ~1x for a meaningful 'mixed' mode)")
        print(f"    duty is controllable to ~{max(cost.values())*1e3:.0f} ms; "
              f"keep that well under --slot-sec")
    return cost


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="JAX non-ML filler")
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--calibrate", action="store_true",
                    help="measure steady-state per-kernel iteration cost")
    ap.add_argument("--table-mb", type=int, default=512)
    ap.add_argument("--hash-rows", type=int, default=1 << 20)
    ap.add_argument("--fft-batch", type=int, default=32)
    ap.add_argument("--fft-size", type=int, default=1024)
    ap.add_argument("--secs", type=float, default=3.0)
    a = ap.parse_args()
    if a.self_test:
        sys.exit(0 if (self_test() or not _HAS_JAX) else 1)
    if a.calibrate:
        if not _HAS_JAX:
            print("calibrate: jax absent"); sys.exit(2)
        calibrate(a.table_mb, a.hash_rows, a.fft_batch, a.fft_size, a.secs)
        sys.exit(0)
    ap.print_help()
