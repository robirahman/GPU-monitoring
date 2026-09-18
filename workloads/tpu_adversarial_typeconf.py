#!/usr/bin/env python3
"""MN-T: real multi-slice TRAINING shaped to be misread as INFERENCE.

MN-I (tpu_adversarial_multinode.py) attacks Stage 1 -- it dilutes training with
non-ML filler so the run looks like it is not ML at all. This attacks Stage 2
instead: keep the run obviously "ML", but make the train-vs-infer discriminator
say INFER. That is a type misclassification rather than an evasion, and it is a
strictly harder target because Stage 2 sees only SHAPE features (the WB-O level
drop), so an attacker cannot simply move a level.

Every knob here targets a channel MEASURED to separate the two classes on real
hardware (v5litepod-8, 2026-09-02, medians over full 600 s cells):

    channel                training   inference    knob
    tensorcore_util_pct       17          88        --tc-fill
    mem_used_mb             7013        2668        --optimizer sgd (no moments)
    cross-slice collectives  every step  none       --accum K  (local SGD)
    d2h transfer latency     idle        streaming  --d2h-trickle

It stays REAL training throughout: a genuine forward/backward and a genuine
parameter update on the same 1b model as every other training cell, with useful
tokens reported so the evasion tax is measurable. An attack that stops training
is not an attack.

HONEST LIMITATION, and it is deliberate. Under --accum K the parameters are
replicated per slice and averaged across slices only every K steps -- classic
local SGD -- which is what cuts the DCN event rate to 1/K. But the batch is
process-uniform (jax.device_put asserts a replicated value is identical on every
process; see the 2026-09-01 abort), so every host trains on the SAME data and the
averaging is numerically redundant. The TELEMETRY -- compute, footprint,
collective cadence, transfer pattern -- is identical to true local SGD, and the
telemetry is what the detector sees, so the evasion question is answered
faithfully. The learning-efficiency question is not, and this cell should never be
cited for throughput-per-token.

CPU gate (no TPU, $0):
    JAX_PLATFORMS=cpu XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python workloads/tpu_adversarial_typeconf.py --self-test
"""

import argparse
import os
import sys
import threading
import time

try:
    from workloads import tpu_transformer_common as T
except ImportError:
    import tpu_transformer_common as T




def self_test() -> bool:
    """Assert the mechanisms do what they claim, on CPU."""
    if not T._HAS_JAX:
        print("typeconf self-test: jax absent, not applicable")
        return True
    import jax
    import jax.numpy as jnp
    ok = True
    # 1. sgd carries no optimizer state, adamw carries two moments per leaf
    p = {"w": jnp.ones((4, 4))}
    n_sgd = len(jax.tree_util.tree_leaves(T.init_opt_state(p, "sgd")))
    n_adam = len(jax.tree_util.tree_leaves(T.init_opt_state(p, "adamw")))
    print(f"  opt state leaves: sgd={n_sgd} adamw={n_adam}")
    ok &= (n_sgd == 0 and n_adam > 0)
    # 2. the sync cadence: with accum=K, syncs happen on 1/K of steps
    for K in (1, 4, 16):
        steps = 64
        syncs = sum(1 for s in range(1, steps + 1) if s % K == 0)
        want = steps // K
        print(f"  accum={K:2d}: {syncs} sync(s) in {steps} steps (want {want})")
        ok &= (syncs == want)
    # 3. the tc filler actually runs and returns finite work
    mesh, axis = T.make_mesh("fsdp")
    with mesh:
        a = jnp.ones((256, 256))
        r = float(jnp.sum(a @ a))
    ok &= (r > 0)
    print(f"  tc filler matmul finite={r > 0}")
    print(f"typeconf self-test {'PASS' if ok else 'FAIL'}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="MN-T: training shaped to look like inference")
    ap.add_argument("--size", default="1b", choices=list(T.MODEL_CONFIGS))
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--batch", type=int, default=0, help="0 = BATCH_DEFAULTS")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--optimizer", default="sgd",
                    choices=("sgd", "sgd_momentum", "adamw"),
                    help="sgd keeps no moments, so the HBM arena stays near the "
                         "inference footprint instead of the training one")
    ap.add_argument("--accum", type=int, default=1,
                    help="local steps between cross-slice parameter averages; "
                         "K>1 cuts the DCN event rate to 1/K (local SGD)")
    ap.add_argument("--tc-fill", type=float, default=0.0,
                    help="0..1 -- dense matmul work added per step to raise "
                         "tensorcore utilisation toward the inference level")
    ap.add_argument("--tc-dim", type=int, default=2048)
    ap.add_argument("--d2h-trickle", type=int, default=0,
                    help="floats copied device->host each step, mimicking token "
                         "streaming on the d2h transfer channel")
    ap.add_argument("--remat", action="store_true")
    ap.add_argument("--multihost", default="auto", choices=T.MULTIHOST_MODES)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()

    if args.self_test:
        sys.exit(0 if (self_test() or not T._HAS_JAX) else 1)
    if not T._HAS_JAX:
        print("ERROR: jax not installed -- runs on a TPU host.", file=sys.stderr)
        sys.exit(2)

    import jax
    import jax.numpy as jnp

    _mh = T.detect_multihost(args.multihost)
    T.init_distributed(args.multihost, tag="tpu_typeconf",
                       expect_processes=int(os.environ.get("SPAR_EXPECT_PROCESSES", 0) or 0))

    pidx, n_host = T.process_index(), T.process_count()
    mesh, axis = T.make_mesh("fsdp")
    n_dev = jax.device_count()
    d_model, n_layers, n_heads, d_ff, vocab = T.MODEL_CONFIGS[args.size]
    batch = args.batch or T.BATCH_DEFAULTS[args.size]

    key = jax.random.PRNGKey(0)
    params = T.init_params(args.size, key)
    # REPLICATED, not sharded: local SGD needs a full copy per slice so the
    # micro-steps carry no cross-slice traffic at all. Sharding the parameters
    # would put a collective inside every step and defeat the --accum knob.
    params = T.replicate(params, mesh)
    opt_state = T.init_opt_state(params, args.optimizer)

    def _chip_used_mb():
        try:
            ms = jax.local_devices()[0].memory_stats() or {}
            for k in ("bytes_reserved", "peak_bytes_reserved", "bytes_in_use"):
                if ms.get(k):
                    return float(ms[k]) / (1 << 20)
        except Exception:
            pass
        return float("nan")

    if pidx == 0:
        print(f"TPU MN-T: {args.size} on {n_dev} chips ({n_host} hosts) "
              f"params={T.param_count(params):,}")
        print(f"  optimizer={args.optimizer} accum={args.accum} "
              f"tc_fill={args.tc_fill} d2h_trickle={args.d2h_trickle}")
        print(f"  MEM_BYTES_IN_USE_START={_chip_used_mb():.0f} MB/chip", flush=True)

    def local_step(params, opt_state, ids):
        loss, grads = jax.value_and_grad(T.loss_fn)(params, ids, n_heads, args.remat)
        new_params, new_state = T.opt_update(args.optimizer, params, grads,
                                             opt_state, args.lr)
        return new_params, new_state, loss
    step_fn = jax.jit(local_step, donate_argnums=(0, 1))

    # The only cross-slice collective in the cell, fired once per --accum steps.
    #
    # pmean needs its axis name BOUND, which a plain jax.jit does not do -- the
    # first version raised "NameError: unbound axis name: fsdp" on the CPU gate.
    # shard_map binds it. in_specs/out_specs are both replicated, so each device
    # gets the whole leaf and pmean averages it across the mesh: a genuine
    # parameter average, and on a multislice group a genuine DCN transfer.
    try:
        from jax.experimental.shard_map import shard_map
    except ImportError:
        shard_map = jax.shard_map
    _P = jax.sharding.PartitionSpec

    @jax.jit
    def sync_params(p):
        avg = shard_map(lambda y: jax.lax.pmean(y, axis), mesh=mesh,
                        in_specs=_P(), out_specs=_P(), check_rep=False)
        return jax.tree_util.tree_map(avg, p)

    tc_a = jnp.ones((args.tc_dim, args.tc_dim), jnp.bfloat16) if args.tc_fill > 0 else None

    @jax.jit
    def tc_burst(a, n):
        def body(i, x):
            return (x @ a).astype(jnp.bfloat16) * 0.5
        return jax.lax.fori_loop(0, n, body, a)

    data_key = jax.random.PRNGKey(1000)   # process-uniform; see the docstring
    t0 = time.time()
    step = syncs = tc_iters = 0
    n_budget = None
    _ref_step = _ref_t = None
    _rebudgeted = False
    _b2_step = _b2_t = None

    with mesh:
        while True:
            if n_budget is not None:
                if step >= n_budget:
                    break
            elif time.time() - t0 >= 3 * args.duration:
                print("[mn-t] step budget never agreed; stopping", flush=True)
                break
            step += 1
            data_key, sub = jax.random.split(data_key)
            ids = T.synthetic_batch(sub, batch, args.seq_len, vocab)
            ids = jax.device_put(ids, jax.sharding.NamedSharding(
                mesh, jax.sharding.PartitionSpec()))
            params, opt_state, loss = step_fn(params, opt_state, ids)
            jax.block_until_ready(loss)

            if args.tc_fill > 0:
                n_it = max(1, int(args.tc_fill * 16))
                jax.block_until_ready(tc_burst(tc_a, n_it))
                tc_iters += n_it

            if args.d2h_trickle > 0:
                # device -> host copy, the channel inference exercises and
                # training normally does not
                _ = jax.device_get(loss)

            if args.accum > 1 and step % args.accum == 0:
                params = sync_params(params)
                jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
                syncs += 1
            elif args.accum == 1:
                params = sync_params(params)
                jax.block_until_ready(jax.tree_util.tree_leaves(params)[0])
                syncs += 1

            T.emit_tpu_step("train", step, tokens=batch * args.seq_len, passes=1,
                            loss=f"{float(loss):.4f}")

            if _ref_t is None and step >= 1:
                _ref_t, _ref_step = time.time(), step
            elif n_budget is None and step >= 4:
                n_budget = T.agreed_step_budget(
                    step - _ref_step, time.time() - _ref_t,
                    args.duration - (time.time() - t0), step)
                _b2_step, _b2_t = step, time.time()
                if pidx == 0:
                    print(f"  step budget agreed at step {step}: {n_budget}", flush=True)
                    print(f"  MEM_BYTES_IN_USE_STEADY={_chip_used_mb():.0f} MB/chip",
                          flush=True)
            # STEP-based trigger, not time-based. This condition gates a
            # broadcast_one_to_all inside agreed_step_budget, and a collective
            # must be reached at the same point in program order on every host.
            # A wall-clock test is per-host: the two slices can cross
            # 0.4*duration at different step indices, and then one host is in the
            # broadcast while the other is still training. Measured 2026-09-03:
            # typeconf_off_rep2 ran 550 steps to 587s of its 600s target and then
            # died with megascale_error_reporter_abort_on_hang -- the same
            # signature as the original wall-clock deadlock, reintroduced by the
            # fix for it. n_budget is broadcast and _b2_step is set at a
            # deterministic step, so this trigger is identical on both hosts.
            if (n_budget is not None and not _rebudgeted
                    and _b2_step is not None
                    and step >= _b2_step + max(20, n_budget // 4)):
                n_budget = T.agreed_step_budget(
                    step - _b2_step, time.time() - _b2_t,
                    args.duration - (time.time() - t0), step)
                _rebudgeted = True
                if pidx == 0:
                    print(f"  step budget re-agreed at step {step}: {n_budget}",
                          flush=True)

            if pidx == 0 and step % 25 == 0:
                print(f"  step {step}: syncs={syncs} tc_iters={tc_iters} "
                      f"elapsed={time.time()-t0:.0f}s/{args.duration}s", flush=True)

    if pidx == 0:
        wall = time.time() - t0
        useful = step * batch * args.seq_len
        print(f"TPU MN-T complete: {step} steps in {wall:.1f}s "
              f"(syncs={syncs} tc_iters={tc_iters})")
        print(f"EVASION_TAX_DATA: strategy=MN-T optimizer={args.optimizer} "
              f"accum={args.accum} tc_fill={args.tc_fill} "
              f"d2h_trickle={args.d2h_trickle} steps={step} syncs={syncs} "
              f"useful_tokens={useful} wall_s={wall:.1f} "
              f"chips={n_dev} hosts={n_host}")

    sys.stdout.flush()
    sys.stderr.flush()
    if _mh:
        _t = threading.Thread(target=jax.distributed.shutdown, daemon=True)
        _t.start()
        _t.join(20)
    os._exit(0)


if __name__ == "__main__":
    main()
