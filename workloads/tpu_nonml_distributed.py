#!/usr/bin/env python3
"""Honest non-ML distributed cells for the TPU rung. No model, no gradients.

Ported from workloads/nonml_distributed.py. These are the controls that made the
GPU result interpretable, and each exists because of a specific finding:

* `--mode hpc` with a resident dummy allocation is the LIKE-FOR-LIKE partner of
  the F=0 disguise: same kernels, same footprint, no model. On GPU the pair came
  out 135 MB / 0.2 points of utilisation / 3 W apart, which is what let us
  attribute the difference to the apparatus and nothing else -- and the answer
  was that the apparatus is worth ~0.04 of p_train, not the ~0.49 originally
  claimed. That retraction is the reason this cell is not optional.

* `--duty` gives the partial-utilisation profile the honest class was missing
  entirely. Real clusters run half-idle HPC jobs and the detector had never seen
  one. This is where the GPU false positives live: an honest duty-cycled job
  scored 0.323 against the F=0.05 disguise's 0.295 -- the false positive fired
  HARDER than the true positive, which is what killed tau=0.10.

* `--filler-mode` turned out to matter more than duty or allocation: at identical
  duty, hash-only vs fft-only moved held-out p_train 3.5x (0.671 vs 0.212). Any
  honest class that fixes the kernel mix will mis-estimate the false-alarm rate.

* `--mode comm` is slot-bounded. The GPU version emitted one STEP line per
  all_reduce, and at world_size=1 the collective is a local no-op, so the cell
  spun at interpreter speed and wrote a 1.44 GB log -- measuring Python, not
  communication. Bounding by slot fixes it at any size. On TPU the analogue is
  `jax.lax.psum` over the mesh, which on a multi-host slice rides ICI/DCN.

Launch (per host, via run_workload_tpu.py):
    python workloads/tpu_nonml_distributed.py --mode hpc --alloc-gb 2 \
        --duty 0.3 --filler-mode hash --duration 600

CPU gate (no TPU, $0):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python workloads/tpu_nonml_distributed.py --self-test
"""

import argparse
import os
import sys
import time

try:
    from workloads import tpu_transformer_common as T
    from workloads.tpu_nonml_filler import JaxNonMLFiller, MODES
except ImportError:
    import tpu_transformer_common as T
    from tpu_nonml_filler import JaxNonMLFiller, MODES


def _detect_multihost(mode: str) -> bool:
    if mode == "on":
        return True
    if mode == "off":
        return False
    for var in ("MEGASCALE_NUM_SLICES", "TPU_WORKER_HOSTNAMES"):
        val = os.environ.get(var)
        if val and ("," in val or val.strip() not in ("", "1")):
            return True
    return os.environ.get("TPU_WORKER_HOSTNAMES", "").count(",") >= 1


def self_test() -> bool:
    if not T._HAS_JAX:
        print("nonml self-test: jax absent, not applicable")
        return True
    import jax
    import jax.numpy as jnp
    mesh, axis = T.make_mesh("fsdp")
    ok = True
    # comm: a slot-bounded psum must complete and stay finite
    n_dev = len(mesh.devices.flatten())
    P, NS = jax.sharding.PartitionSpec, jax.sharding.NamedSharding
    buf = jax.device_put(jnp.ones((n_dev * 256, 64), jnp.float32), NS(mesh, P(axis)))

    @jax.jit
    def allreduce(x):
        return jax.lax.with_sharding_constraint(x * 1.0 + x.sum() * 0.0,
                                               NS(mesh, P(axis)))
    with mesh:
        r = allreduce(buf)
        jax.block_until_ready(r)
    finite = bool(jnp.isfinite(r).all())
    print(f"  comm psum finite={finite}")
    ok &= finite
    # duty: 0.3 duty over 1.0 s must spend ~0.3 s busy
    # hash mode for the duty assertion: sub-millisecond iterations mean the test
    # measures the DUTY MECHANISM rather than kernel granularity. The tolerance is
    # still expressed in terms of iter_cost, because achievable duty precision is
    # bounded by one iteration whatever the kernel.
    f = JaxNonMLFiller(mesh, axis, table_mb=4, hash_rows=1 << 12,
                       fft_batch=8, fft_size=64, mode="hash")
    t0 = time.time(); f.run_for(0.3); busy = time.time() - t0
    tol = 0.3 + max(0.05, 1.2 * f.iter_cost())
    close = 0.15 < busy <= tol
    ok &= close
    print(f"  duty slot: asked 0.30s got {busy:.2f}s "
          f"(tol {tol:.2f}s, iter={f.iter_cost()*1e3:.2f}ms) "
          f"{'ok' if close else 'FAIL'}")
    print(f"nonml self-test {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="TPU honest non-ML distributed cell")
    ap.add_argument("--mode", choices=("hpc", "comm"), default="hpc")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--alloc-gb", type=float, default=0.0,
                    help="resident dummy allocation, the model-footprint analogue")
    ap.add_argument("--duty", type=float, default=1.0,
                    help="hpc: fraction of each slot spent computing, rest idle")
    ap.add_argument("--filler-mode", choices=MODES, default="mixed")
    ap.add_argument("--filler-table-mb", type=int, default=512)
    ap.add_argument("--slot-sec", type=float, default=0.5)
    ap.add_argument("--comm-mb", type=float, default=256.0)
    ap.add_argument("--multihost", type=str, default="auto",
                    choices=["auto", "on", "off"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        ok = self_test()
        sys.exit(0 if (ok or not T._HAS_JAX) else 1)
    if not T._HAS_JAX:
        print("ERROR: jax not installed -- runs on a TPU host.", file=sys.stderr)
        sys.exit(2)

    import jax
    import jax.numpy as jnp

    if _detect_multihost(args.multihost):
        try:
            jax.distributed.initialize()
        except Exception as e:
            print(f"[nonml] distributed.initialize skipped: {e}", flush=True)

    pidx, n_host = T.process_index(), T.process_count()
    n_dev = jax.device_count()
    mesh, axis = T.make_mesh("fsdp")
    P, NS = jax.sharding.PartitionSpec, jax.sharding.NamedSharding
    row_shard = NS(mesh, P(axis))

    # Resident dummy allocation: occupies HBM exactly like a model would, while
    # containing nothing. This is the control that refuted "the apparatus is the
    # detection channel".
    dummy = None
    if args.alloc_gb > 0:
        n = max(n_dev, int(args.alloc_gb * (1 << 30) / 4))
        n -= n % n_dev
        dummy = jax.device_put(jnp.zeros((n // n_dev, n_dev), jnp.float32).reshape(-1, n_dev),
                               row_shard)

    if pidx == 0:
        print(f"TPU non-ML: mode={args.mode} on {n_dev} chips ({n_host} hosts)")
        if args.mode == "hpc":
            print(f"  alloc={args.alloc_gb}GB duty={args.duty:.0%} "
                  f"kernel={args.filler_mode} slot={args.slot_sec}s")
        else:
            print(f"  comm buffer={args.comm_mb}MB slot={args.slot_sec}s")

    filler = comm_buf = allreduce = None
    if args.mode == "hpc":
        filler = JaxNonMLFiller(mesh, axis, table_mb=args.filler_table_mb,
                                mode=args.filler_mode)
        if pidx == 0:
            print(f"  filler resident={filler.bytes_resident()/1e6:.0f}MB")
    else:
        elems = max(n_dev, int(args.comm_mb * (1 << 20) / 4))
        elems -= elems % n_dev
        comm_buf = jax.device_put(
            jnp.ones((elems // n_dev, n_dev), jnp.float32).reshape(-1, n_dev),
            row_shard)

        @jax.jit
        def _ar(x):
            # Cross-chip reduction then re-broadcast: on a multi-host slice this
            # is real ICI/DCN traffic, the TPU analogue of NCCL all_reduce.
            s = jax.lax.with_sharding_constraint(x.sum(axis=0), NS(mesh, P()))
            return jax.lax.with_sharding_constraint(x + s * 0.0, row_shard)
        allreduce = _ar

    t0 = time.time()
    slot = iters = collectives = 0
    with mesh:
        while time.time() - t0 < args.duration:
            slot += 1
            if args.mode == "hpc":
                busy = args.slot_sec * max(0.0, min(1.0, args.duty))
                iters += filler.run_for(busy)
                if busy < args.slot_sec:
                    time.sleep(args.slot_sec - busy)
            else:
                # Slot-bounded: one STEP line per slot, never one per collective.
                end = time.time() + args.slot_sec
                r = None
                while time.time() < end:
                    r = allreduce(comm_buf)
                    collectives += 1
                if r is not None:
                    jax.block_until_ready(r)
            T.emit_tpu_step("other", slot, tokens=0, passes=1)
            if pidx == 0 and slot % 40 == 0:
                print(f"  slot {slot}: iters={iters} collectives={collectives} "
                      f"elapsed={time.time()-t0:.0f}s/{args.duration}s", flush=True)

    if pidx == 0:
        print(f"TPU non-ML complete: {slot} slots in {time.time()-t0:.1f}s "
              f"iters={iters} collectives={collectives} "
              f"alloc_gb={args.alloc_gb} duty={args.duty} "
              f"kernel={args.filler_mode}")
    del dummy


if __name__ == "__main__":
    main()
