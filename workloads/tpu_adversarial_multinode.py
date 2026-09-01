#!/usr/bin/env python3
"""MN-I on TPU: duty-cycle interleave disguise. The cell the TPU rung exists for.

Ported from workloads/adversarial_multinode.py (strategy MN-I). A real sharded
transformer stays RESIDENT for the whole cell while only a fraction `F` of steps
actually train; the rest run genuine non-ML filler. The GPU result this is here to
test on different silicon:

    held out by workload, run-level k=3-of-5, tau=0.75
      F=0.5  -> 100% of runs detected at 0.6% false-alarm rate
      F=0.2  -> 0%
      F=0.05 -> 0%
      F=0    -> 0%   (a competent filler evades outright)

so the claim is a *cost*: evading requires capping real training near half duty,
i.e. roughly a 2x slowdown. Phase 0 established this survives on TPU-available
metrics at 3.0% FPR instead of 0.6% -- power carries most of the high-precision
discrimination -- so the TPU question is whether the native comm channels
(dcn_collective_lat_us_*, hlo_exec_us_*) recover what power provided.

RANK-UNIFORMITY IS NOT OPTIONAL. On GPU, workloads/dist_deadline.py already
required every rank to call expired() the same number of times or the job hangs.
JAX SPMD makes this stricter: a per-host branch that changes the traced
computation forces a recompilation, and a 5-10 minute XLA compile mid-cell would
dominate the telemetry. So the schedule is derived, not communicated: host 0
draws a seed, broadcasts it ONCE, and every host computes the identical
train/filler decision locally from (seed, step). No per-step collective, no
data-dependent tracing, two jitted functions compiled once each.

MN-O (the oracle veto) is deliberately NOT ported: every veto changes the traced
computation, and the recompilation stalls would contaminate the signal.

Launch (per host, via run_workload_tpu.py which wraps the collector around it):
    python workloads/tpu_adversarial_multinode.py --size 1b --train-frac 0.5 \
        --duration 600 --multihost auto

CPU gate (no TPU, $0):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python workloads/tpu_adversarial_multinode.py --self-test
"""

import argparse
import os
import sys
import threading
import time

try:
    from workloads import tpu_transformer_common as T
    from workloads.tpu_nonml_filler import JaxNonMLFiller, MODES
except ImportError:  # launched as `python workloads/tpu_adversarial_multinode.py`
    import tpu_transformer_common as T
    from tpu_nonml_filler import JaxNonMLFiller, MODES

_MASK63 = (1 << 63) - 1


def is_train_step(seed: int, step: int, train_frac: float) -> bool:
    """Identical decision on every host, derived not communicated.

    A 64-bit LCG mix of (seed, step) in pure Python: same inputs -> same bool on
    every process, so hosts stay rank-uniform with zero per-step communication.
    Deterministic in `seed`, so a cell is reproducible from its logged seed.
    """
    if train_frac >= 1.0:
        return True
    if train_frac <= 0.0:
        return False
    h = (seed * 6364136223846793005 + (step + 1) * 1442695040888963407) & _MASK63
    h ^= h >> 31
    h = (h * 0x9E3779B97F4A7C15) & _MASK63
    return ((h >> 17) % 1_000_000) < int(train_frac * 1_000_000)


def _detect_multihost(mode: str) -> bool:
    """Mirrors tpu_fsdp_training._detect_multihost so both cells agree."""
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
    """CPU gate: schedule is host-uniform and hits the requested fraction."""
    ok = True
    for frac in (0.0, 0.05, 0.2, 0.5, 1.0):
        n = sum(is_train_step(12345, s, frac) for s in range(20000))
        got = n / 20000
        close = abs(got - frac) < 0.02
        ok &= close
        print(f"  train_frac={frac:<5} realised={got:.4f} "
              f"{'ok' if close else 'FAIL'}")
    # determinism: the same (seed, step) must agree across "hosts"
    same = all(is_train_step(7, s, 0.3) == is_train_step(7, s, 0.3)
               for s in range(1000))
    # and differ with the seed, or every cell would share one schedule
    differs = any(is_train_step(7, s, 0.3) != is_train_step(8, s, 0.3)
                  for s in range(1000))
    ok &= same and differs
    print(f"  deterministic={same} seed-sensitive={differs}")
    print(f"MN-I schedule self-test {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="TPU MN-I duty-cycle interleave")
    ap.add_argument("--size", type=str, default="1b",
                    choices=list(T.MODEL_CONFIGS.keys()))
    ap.add_argument("--train-frac", type=float, default=0.5,
                    help="F: fraction of steps that really train (0 => filler only)")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--optimizer", type=str, default=None,
                    choices=["adamw", "sgd", "sgd_momentum"])
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--remat", action="store_true")
    ap.add_argument("--filler-mode", choices=MODES, default="mixed")
    ap.add_argument("--filler-table-mb", type=int, default=512)
    ap.add_argument("--slot-sec", type=float, default=0.5,
                    help="wall-clock granularity of one schedule slot")
    ap.add_argument("--seed", type=int, default=0,
                    help="0 => derive from the clock on host 0 and broadcast")
    ap.add_argument("--multihost", type=str, default="auto",
                    choices=["auto", "on", "off"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        ok = self_test()
        if T._HAS_JAX:
            try:
                from workloads.tpu_nonml_filler import self_test as fst
            except ImportError:
                from tpu_nonml_filler import self_test as fst
            ok &= fst()
        sys.exit(0 if (ok or not T._HAS_JAX) else 1)

    if not T._HAS_JAX:
        print("ERROR: jax not installed -- this cell runs on a TPU host.",
              file=sys.stderr)
        sys.exit(2)

    import jax

    _mh = _detect_multihost(args.multihost)
    if _mh:
        try:
            jax.distributed.initialize()
        except Exception as e:
            print(f"[mn-i] distributed.initialize skipped: {e}", flush=True)

    pidx, n_host = T.process_index(), T.process_count()
    n_dev = jax.device_count()

    # ── Seed: drawn once on host 0, broadcast once, never again ──────────────
    seed = args.seed
    if seed == 0:
        seed = int(time.time() * 1000) & 0xFFFFFFFF
        if n_host > 1:
            from jax.experimental import multihost_utils
            import numpy as _np
            seed = int(multihost_utils.broadcast_one_to_all(
                _np.asarray(seed, dtype=_np.uint32)))
    if pidx == 0:
        print(f"TPU MN-I: size={args.size} F={args.train_frac} "
              f"on {n_dev} chips ({n_host} hosts) seed={seed}")

    _, _, n_heads, _, vocab = T.MODEL_CONFIGS[args.size]
    batch = args.batch_size or T.BATCH_DEFAULTS[args.size]
    optimizer = T.auto_optimizer(args.size, args.optimizer)

    # ── The apparatus: resident for the WHOLE cell, trained only F of the time ─
    mesh, axis = T.make_mesh("fsdp")
    params = T.init_params(args.size, jax.random.PRNGKey(0))
    with mesh:
        params = T.shard_params(params, mesh, "fsdp", axis)
        opt_state = T.init_opt_state(params, optimizer)
        opt_state = T.shard_params(opt_state, mesh, "fsdp", axis) if opt_state else opt_state

    filler = JaxNonMLFiller(mesh, axis, table_mb=args.filler_table_mb,
                            mode=args.filler_mode, seed=seed & 0xFFFF)
    if pidx == 0:
        print(f"  params={T.param_count(params):,} optimizer={optimizer}")
        print(f"  filler mode={args.filler_mode} "
              f"resident={filler.bytes_resident()/1e6:.0f}MB slot={args.slot_sec}s")

    def train_step(params, opt_state, input_ids):
        loss, grads = jax.value_and_grad(T.loss_fn)(
            params, input_ids, n_heads, args.remat)
        new_params, new_state = T.opt_update(optimizer, params, grads,
                                             opt_state, args.lr)
        return new_params, new_state, loss
    step_fn = jax.jit(train_step, donate_argnums=(0, 1))

    # Multi-host: the batch is placed with a REPLICATED PartitionSpec(), and
    # jax.device_put asserts that a replicated value is identical on every
    # process. Seeding per-process (PRNGKey(base + pidx)) therefore aborted the
    # coordinator on the first train step of every multi-host cell -- measured on
    # a 2-slice v6e-1 multislice, 2026-09-01: "ArrayImpl passed to device_put is
    # not the same on each process". It never fired on tpuT8 because a v6e-8 is a
    # single process. Keep the data key process-uniform: the tokens are synthetic
    # and carry no information, so identical batches across hosts change no
    # timing, memory or collective behaviour -- i.e. no telemetry feature -- while
    # per-process seeding would need make_array_from_process_local_data and a
    # batch-sharded spec, which is a different experiment.
    def _chip_used_mb():
        """MB in use on this host's first chip, from the runtime allocator.

        Printed so the honest matched control can be targeted in the SAME units
        it measures. The control closes its loop on
        memory_stats()['bytes_in_use'], while the footprint target was taken from
        the collector's mem_used_mb -- and the two differ by roughly 900 MB of
        runtime overhead on a v6e. Matching across that gap by arithmetic is
        guesswork: nonml_sat_rep3 filled to bytes_in_use=5120 and the collector
        then reported 6464 against a 5196 target. So both cells now report the
        allocator's own number, and the control is targeted on the disguise
        cell's reading rather than on a converted one.
        """
        try:
            ms = jax.local_devices()[0].memory_stats() or {}
            return float(ms.get("bytes_in_use", 0)) / (1 << 20)
        except Exception:
            return float("nan")

    if pidx == 0:
        print(f"  MEM_BYTES_IN_USE_START={_chip_used_mb():.0f} MB/chip", flush=True)

    data_key = jax.random.PRNGKey(1000)
    t0 = time.time()
    step = train_steps = filler_steps = filler_iters = 0

    # Rank-uniform exit. See T.agreed_step_budget: a wall-clock loop condition
    # deadlocks the slice once the body runs collectives, and MegaScale then
    # aborts the coordinator on hang. At F=0 there are no train steps, so there
    # are no collectives and the per-step cost is exactly one filler slot -- the
    # budget is then known up front and no collective is needed to agree on it.
    n_budget = None
    _ref_step = _ref_t = None
    _b2_step = _b2_t = None      # steady-state reference, set when the budget is fixed
    _rebudgeted = False
    if args.train_frac <= 0.0:
        n_budget = max(1, int(args.duration / max(args.slot_sec, 1e-3)))
        if pidx == 0:
            print(f"  step budget (F=0, no collectives): {n_budget}", flush=True)
    _reported_steady = False
    with mesh:
        while True:
            if n_budget is not None:
                if step >= n_budget:
                    break
            elif time.time() - t0 >= 3 * args.duration:
                # The budget is set a few train steps in. If we somehow never got
                # there, stop rather than run until the cell timeout.
                print("[mn-i] step budget never agreed; stopping", flush=True)
                break
            step += 1
            if is_train_step(seed, step, args.train_frac):
                data_key, sub = jax.random.split(data_key)
                ids = T.synthetic_batch(sub, batch, args.seq_len, vocab)
                ids = jax.device_put(ids, jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec()))
                params, opt_state, loss = step_fn(params, opt_state, ids)
                jax.block_until_ready(loss)
                train_steps += 1
                # `tokens=` is USEFUL work; the evasion-cost metric is derived
                # from kind=train lines, so filler must never inflate it.
                T.emit_tpu_step("train", step, tokens=batch * args.seq_len,
                                passes=1, loss=f"{float(loss):.4f}")
            else:
                filler_iters += filler.run_for(args.slot_sec)
                filler_steps += 1
                T.emit_tpu_step("filler", step, tokens=0, passes=1)

            # Reference point taken AFTER the first train step, so the budget
            # is not extrapolated from XLA compile time.
            if _ref_t is None and train_steps >= 1:
                _ref_t, _ref_step = time.time(), step
            elif n_budget is None and train_steps >= 4:
                n_budget = T.agreed_step_budget(
                    step - _ref_step, time.time() - _ref_t,
                    args.duration - (time.time() - t0), step)
                _b2_step, _b2_t = step, time.time()
                if pidx == 0:
                    print(f"  step budget agreed at step {step}: {n_budget} "
                          f"(F={args.train_frac})", flush=True)
                    # Post-first-train-step, so this includes the training
                    # workspace -- the figure the 16 GB control must match.
                    print(f"  MEM_BYTES_IN_USE_STEADY={_chip_used_mb():.0f} MB/chip",
                          flush=True)

            # ONE steady-state correction. The first budget is fixed only a few
            # train steps in, and at F>0 that early window still contains the
            # FILLER's first compile as well as the train step's, so the measured
            # rate is far too slow and the budget far too short: f50_rep2 agreed
            # 101 steps at step 7 and finished in 268 s of a requested 600 s, at a
            # true steady-state rate of ~2.2 s/step (600 s needs ~270 steps). The
            # realised fraction suffers too -- 0.4455 against a requested 0.5,
            # since it converges only with step count. Re-agreeing once at 40% of
            # the duration, from the rate observed since the budget was fixed,
            # costs one broadcast in a cell that already runs gradient collectives
            # every train step. F=0 never reaches here: its budget is exact.
            if (n_budget is not None and not _rebudgeted and _b2_t is not None
                    and step > _b2_step + 20
                    and time.time() - t0 >= 0.4 * args.duration):
                n_budget = T.agreed_step_budget(
                    step - _b2_step, time.time() - _b2_t,
                    args.duration - (time.time() - t0), step)
                _rebudgeted = True
                if pidx == 0:
                    print(f"  step budget re-agreed at step {step}: {n_budget} "
                          f"({(time.time()-_b2_t)/max(step-_b2_step,1):.2f}s/step "
                          f"steady state)", flush=True)

            # F=0 skips the budget block entirely, so report the steady-state
            # footprint here as well -- it is the target for nonml_sat.
            if pidx == 0 and not _reported_steady and step >= 30:
                _reported_steady = True
                print(f"  MEM_BYTES_IN_USE_STEADY={_chip_used_mb():.0f} MB/chip",
                      flush=True)

            if pidx == 0 and step % 20 == 0:
                el = time.time() - t0
                print(f"  step {step}: train={train_steps} filler={filler_steps} "
                      f"realised_F={train_steps/max(step,1):.3f} "
                      f"elapsed={el:.0f}s/{args.duration}s", flush=True)

    if pidx == 0:
        wall = time.time() - t0
        useful = train_steps * batch * args.seq_len
        print(f"TPU MN-I complete: {step} steps in {wall:.1f}s "
              f"(train={train_steps} filler={filler_steps} "
              f"filler_iters={filler_iters})")
        print(f"EVASION_TAX_DATA: strategy=MN-I train_frac={args.train_frac} "
              f"realised_frac={train_steps/max(step,1):.4f} "
              f"useful_tokens={useful} wall_s={wall:.1f} "
              f"chips={n_dev} hosts={n_host} seed={seed}")

    # Same bounded teardown as tpu_nonml_distributed: a multislice process that
    # has run collectives can block at interpreter exit, and the cell then burns
    # billed time until its timeout fires instead of returning OK. Flush first,
    # give an orderly shutdown 20 s, then leave hard -- the collector is a
    # separate process.
    sys.stdout.flush()
    sys.stderr.flush()
    if _mh:
        _t = threading.Thread(target=jax.distributed.shutdown, daemon=True)
        _t.start()
        _t.join(20)
    os._exit(0)


if __name__ == "__main__":
    main()
