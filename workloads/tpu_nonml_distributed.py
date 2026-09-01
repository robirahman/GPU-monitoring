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
import threading
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
    ap.add_argument("--match-mem-mb", type=float, default=0.0,
                    help="CLOSED-LOOP footprint match: allocate until this chip "
                         "reports this many MB in use. Overrides --alloc-gb. Use "
                         "this for any cell whose purpose is to match another "
                         "cell's measured footprint.")
    ap.add_argument("--match-chunk-mb", type=float, default=256.0)
    ap.add_argument("--match-tol-mb", type=float, default=128.0)
    ap.add_argument("--alloc-chunk-gb", type=float, default=1.0,
                    help="allocate the dummy in chunks this large; a single huge "
                         "buffer hits the TPU allocator's bottom-of-memory limit")
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

    _mh = _detect_multihost(args.multihost)
    if _mh:
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
    #
    # It used to be one jnp.zeros() followed by a device_put into the sharding,
    # which allocates the whole thing TWICE -- once host-shaped on device, once as
    # the sharded copy -- and asks the allocator for a single buffer that large.
    # On a v6e-1 (32 GB HBM, mem_total_mb=31996 measured) that failed at 8 GB:
    #   RESOURCE_EXHAUSTED: Attempting to reserve 8.00G at the bottom of memory.
    #   There are 7.25G free, 0B reserved, and 7.25G reservable.
    # so tpuTMS_nonml_sat (12 GB) and tpuTMS_nonml_duty50_hash (8 GB) died in ~55s
    # on 2026-09-01 while duty30 (2 GB) survived. Allocate in chunks, each built
    # directly at its sharding by a jitted zeros so nothing is ever copied, and
    # report what was actually realised -- a matched-footprint control that
    # silently allocated less than asked would quietly break the pairing that the
    # apparatus retraction depends on.
    def _mem_stats():
        try:
            return jax.local_devices()[0].memory_stats() or {}
        except Exception:
            return {}

    if pidx == 0:
        # One-shot dump of EVERY key. Needed because bytes_in_use is not what the
        # collector reports: on a v6e running MN-I at F=0.05 the cell logged
        # bytes_in_use=4694 MB/chip while collect_tpu_telemetry recorded
        # mem_used_mb=16495 -- a 11.8 GB gap, because bytes_in_use counts live
        # tensors (params + optimizer + filler) whereas the collector reports the
        # runtime's reserved arena, which grew to hold the train step's transient
        # workspace and never shrank. A control matched on bytes_in_use would
        # leave the collector-visible footprint -- the actual feature the detector
        # sees -- unmatched, so the match must close on whichever key tracks the
        # arena. This print identifies it.
        print("  MEM_STATS_KEYS=" + ",".join(
            f"{k}={v/(1<<20):.0f}MB" if isinstance(v, (int, float)) else f"{k}={v}"
            for k, v in sorted(_mem_stats().items())), flush=True)

    def _chip_used_mb():
        """MB in use on this host's first chip, from the runtime allocator."""
        try:
            ms = jax.local_devices()[0].memory_stats() or {}
            return float(ms.get("bytes_in_use", 0)) / (1 << 20)
        except Exception:
            return float("nan")

    dummy = []
    if args.alloc_gb > 0:
        chunk_gb = min(args.alloc_chunk_gb, args.alloc_gb)
        want = int(args.alloc_gb * (1 << 30))
        per = max(n_dev * 4, int(chunk_gb * (1 << 30)))
        got = 0
        while got < want:
            this = min(per, want - got)
            n = max(n_dev, this // 4)
            n -= n % n_dev
            try:
                z = jax.jit(lambda: jnp.zeros((n // n_dev, n_dev), jnp.float32),
                            out_shardings=row_shard)()
                jax.block_until_ready(z)
            except Exception as e:
                if pidx == 0:
                    print(f"  alloc stopped at {got/(1<<30):.2f}GB of "
                          f"{args.alloc_gb}GB: {type(e).__name__}", flush=True)
                break
            dummy.append(z)
            got += n * 4
        if pidx == 0:
            print(f"  ALLOC_REALISED_GB={got/(1<<30):.3f} requested={args.alloc_gb} "
                  f"chunks={len(dummy)}", flush=True)

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
        # Same single-shot rule as the dummy above: build at the sharding.
        comm_buf = jax.jit(lambda: jnp.ones((elems // n_dev, n_dev), jnp.float32),
                           out_shardings=row_shard)()
        jax.block_until_ready(comm_buf)

        @jax.jit
        def _ar(x):
            # Cross-chip reduction then re-broadcast: on a multi-host slice this
            # is real ICI/DCN traffic, the TPU analogue of NCCL all_reduce.
            s = jax.lax.with_sharding_constraint(x.sum(axis=0), NS(mesh, P()))
            return jax.lax.with_sharding_constraint(x + s * 0.0, row_shard)
        allreduce = _ar

    # Ordered AFTER the filler and comm buffer are resident, which is the whole
    # point: when this ran first, it filled to the target and the filler then
    # added ~1.3 GB on top -- nonml_sat_rep3 logged
    #   MEM_MATCH target=5196 realised=5120 MB/chip (baseline 0, 20 x 256MB)
    # and the collector reported 6464 MB against a 5196 MB target, overshooting by
    # 24% where the open-loop version had undershot by 45%. `baseline 0` was the
    # tell. Topping up LAST makes the match hold whatever else the cell allocates.
    # ── Closed-loop footprint match ──────────────────────────────────────────
    # An open-loop `--alloc-gb` CANNOT match a measured footprint, because the
    # dummy is sharded across the mesh while mem_used_mb is per-chip, and the
    # non-alloc baseline is not a constant either. Measured 2026-09-01 on 2x
    # v6e-1: nonml_sat asked for 3.65 GB, realised 3.0 GB globally = 1.5 GB/chip,
    # and reported mem_used_mb 2880 against the F=0 cell's 5196 -- 45% short, on a
    # cell whose ONLY job is to hold the same footprint as that cell. Worse, the
    # two available data points imply different baselines (1344 vs 2480 MB/chip),
    # so no better constant would fix it.
    #
    # So close the loop on the measurement: add per-LOCAL-DEVICE chunks until the
    # allocator reports the target. Allocated straight onto each device with a
    # SingleDeviceSharding (never jnp.zeros-then-copy, which would double the
    # peak), and no collective is involved, so hosts that need a different number
    # of chunks cannot deadlock each other.
    if args.match_mem_mb > 0:
        devs = jax.local_devices()
        n_el = max(1, int(args.match_chunk_mb * (1 << 20) / 4))
        chunks = 0
        start_mb = _chip_used_mb()
        while chunks < 4096:
            used = _chip_used_mb()
            if not (used == used) or used >= args.match_mem_mb - args.match_tol_mb:
                break
            try:
                for dv in devs:
                    z = jax.jit(lambda: jnp.zeros((n_el,), jnp.float32),
                                out_shardings=jax.sharding.SingleDeviceSharding(dv))()
                    jax.block_until_ready(z)
                    dummy.append(z)
                chunks += 1
            except Exception as e:
                print(f"  match stopped at {used:.0f}MB/chip: {type(e).__name__}",
                      flush=True)
                break
        if pidx == 0:
            print(f"  MEM_MATCH target={args.match_mem_mb:.0f} "
                  f"realised={_chip_used_mb():.0f} MB/chip "
                  f"(baseline {start_mb:.0f}, {chunks} x {args.match_chunk_mb:.0f}MB)",
                  flush=True)

    t0 = time.time()
    slot = iters = collectives = 0

    # Both loop bounds must be rank-uniform, not wall-clock. Every slot of comm
    # mode is a barrier, so if host A fits 100 all_reduces into its slot and host
    # B fits 98, A's surplus two have no partner and the slice wedges; and when
    # the outer deadline fires first on one host it leaves the other blocked in a
    # collective. tpuTMS_nonml_comm died exactly this way on all three reps --
    # 1078 slots of real DCN traffic, then a MegaScale abort-on-hang. So: a slot
    # count both hosts derive identically from the same arguments, and a
    # per-slot collective count measured once and broadcast from host 0.
    n_slots = max(1, int(args.duration / max(args.slot_sec, 1e-3)))
    per_slot = 0
    if args.mode == "comm":
        with mesh:                       # warm up (compiles), then time one
            jax.block_until_ready(allreduce(comm_buf))
            _t = time.time()
            jax.block_until_ready(allreduce(comm_buf))
            _one = time.time() - _t
        per_slot = T.broadcast_int(max(1, int(args.slot_sec / max(_one, 1e-6))))
        if pidx == 0:
            print(f"  slots={n_slots} collectives/slot={per_slot} "
                  f"(one all_reduce = {_one*1e3:.2f}ms)", flush=True)
    elif pidx == 0:
        print(f"  slots={n_slots}", flush=True)

    _rebudgeted = False
    with mesh:
        while slot < n_slots:
            slot += 1
            # comm mode's a-priori slot count is derived from ONE synchronous
            # all_reduce, but the loop issues its collectives asynchronously and
            # blocks only once per slot, so the real throughput is higher and the
            # cell finishes early: nonml_comm_rep1 returned OK after 406 s of a
            # requested 600 s. Re-agree the budget from the achieved rate after a
            # few slots. The extra broadcast is free of confound here -- this cell
            # is nothing but collectives -- and it is the same rank-uniform
            # agreement used everywhere else, so both hosts still leave together.
            if args.mode == "comm" and slot == 4 and not _rebudgeted:
                _el = time.time() - t0
                n_slots = T.agreed_step_budget(slot, _el, args.duration - _el, slot)
                _rebudgeted = True
                if pidx == 0:
                    print(f"  slot budget re-agreed at slot {slot}: {n_slots} "
                          f"({_el/slot:.3f}s/slot achieved)", flush=True)
            if args.mode == "hpc":
                busy = args.slot_sec * max(0.0, min(1.0, args.duty))
                iters += filler.run_for(busy)
                if busy < args.slot_sec:
                    time.sleep(args.slot_sec - busy)
            else:
                # Slot-bounded: one STEP line per slot, never one per collective.
                r = None
                for _ in range(per_slot):
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

    # Multislice teardown can hang, and it costs a whole cell when it does:
    # tpuTMS_nonml_comm ran its full 600 s (1078 psum slots, last STEP ~19:00Z),
    # then sat idle until the 1000 s cell timeout killed it at 19:12:49Z -- rc=1
    # for a cell whose science had completed, on all three reps. Once a collective
    # has been in flight, jax.distributed's coordinator connection can block at
    # interpreter exit, and with two slices whichever process leaves first strands
    # the other. So: flush, ask for an orderly shutdown with a bounded wait, then
    # leave hard. The telemetry collector is a separate process, so exiting this
    # way loses nothing we own.
    sys.stdout.flush()
    sys.stderr.flush()
    if _mh:
        _t = threading.Thread(target=jax.distributed.shutdown, daemon=True)
        _t.start()
        _t.join(20)
    os._exit(0)

if __name__ == "__main__":
    main()
