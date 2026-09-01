#!/usr/bin/env python3
"""TPU FSDP-style training cell -- the TRAINING rung of the comm-scope ladder.

Trains the shared pure-JAX decoder transformer (workloads/tpu_transformer_common)
with parameters sharded across the whole slice ('fsdp' mesh). Every jitted step
all-gathers params in the forward and reduce-scatters gradients in the backward:
on a multi-host slice (v6e-8 / v6e-16) those collectives ride ICI/DCN across
servers -- the inter-server training signature the ladder is built to detect.
There are NO fake sleeps (the toy tpu_jax_training.py had `time.sleep(0.5)`);
throughput is set by real compute + collectives, exactly like the GPU FSDP cell.

Launch (per host, via run_workload_tpu.py which starts the collector around it):
    python workloads/tpu_fsdp_training.py --size 7b --duration 600 --multihost auto

Multi-host: `jax.distributed.initialize()` is called when >1 host is detected
(or forced with --multihost on) so jax.devices() spans all chips of the slice.

CPU gate (no TPU, $0):
    python workloads/tpu_fsdp_training.py --self-test
    XLA_FLAGS=--xla_force_host_platform_device_count=8 \
        python workloads/tpu_fsdp_training.py --self-test
"""

import argparse
import os
import sys
import time

try:
    from workloads import tpu_transformer_common as T
except ImportError:  # launched as `python workloads/tpu_fsdp_training.py`
    import tpu_transformer_common as T


def _detect_multihost(mode: str) -> bool:
    """Decide whether to bring up the cross-host JAX cluster.

    'on'/'off' force it; 'auto' initialises when the TPU env advertises more
    than one worker (megascale / multi-slice or a >1-host slice).
    """
    if mode == "on":
        return True
    if mode == "off":
        return False
    # auto: infer from the standard TPU worker env vars.
    for var in ("MEGASCALE_NUM_SLICES", "TPU_WORKER_HOSTNAMES"):
        val = os.environ.get(var)
        if val and ("," in val or val.strip() not in ("", "1")):
            return True
    # TPU_WORKER_HOSTNAMES is a comma-separated host list on multi-host slices.
    hosts = os.environ.get("TPU_WORKER_HOSTNAMES", "")
    if hosts.count(",") >= 1:
        return True
    return False


def main():
    ap = argparse.ArgumentParser(description="TPU FSDP-style training cell")
    ap.add_argument("--size", type=str, default="7b",
                    choices=list(T.MODEL_CONFIGS.keys()),
                    help="Model size preset (default: 7b)")
    ap.add_argument("--duration", type=int, default=600,
                    help="Training duration in seconds (default: 600)")
    ap.add_argument("--optimizer", type=str, default=None,
                    choices=["adamw", "sgd", "sgd_momentum"],
                    help="Override optimizer (default: auto by size)")
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Per-step global batch (auto by size if omitted)")
    ap.add_argument("--remat", action="store_true",
                    help="Activation checkpointing (recompute to save HBM)")
    ap.add_argument("--multihost", type=str, default="auto",
                    choices=["auto", "on", "off"],
                    help="Bring up the cross-host JAX cluster (default: auto)")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--self-test", action="store_true",
                    help="CPU gradient check (G1 gate); no TPU needed")
    args = ap.parse_args()

    if args.self_test:
        ok = T.self_test()
        # No jax on this box => "not applicable" (exit 0), not a failure. Where
        # jax exists the real check runs and only a true pass exits 0; the queue
        # G1 gate greps for the literal "self-test PASS" line, which only a real
        # pass prints.
        sys.exit(0 if (ok or not T._HAS_JAX) else 1)

    if not T._HAS_JAX:
        print("ERROR: jax not installed -- this cell runs on a TPU host.",
              file=sys.stderr)
        sys.exit(2)

    import jax
    import jax.numpy as jnp

    # ── Cross-host cluster ───────────────────────────────────────────────────
    if _detect_multihost(args.multihost):
        try:
            jax.distributed.initialize()
        except Exception as e:  # already initialised, or single-host after all
            print(f"[tpu_fsdp] distributed.initialize skipped: {e}", flush=True)

    n_dev = jax.device_count()
    pidx = T.process_index()
    if pidx == 0:
        hidden, n_layers, n_heads, ffn, vocab = T.MODEL_CONFIGS[args.size]
        print(f"TPU FSDP training: {args.size} on {n_dev} chips "
              f"({T.process_count()} hosts)")
        print(f"  hidden={hidden} layers={n_layers} heads={n_heads} ffn={ffn}")
        print(f"  seq_len={args.seq_len} remat={args.remat}")

    _, _, n_heads, _, vocab = T.MODEL_CONFIGS[args.size]
    batch = args.batch_size or T.BATCH_DEFAULTS[args.size]
    optimizer = T.auto_optimizer(args.size, args.optimizer)
    lr = args.lr

    # ── Build + shard the model across the whole slice (FSDP) ────────────────
    mesh, axis = T.make_mesh("fsdp")
    key = jax.random.PRNGKey(0)
    params = T.init_params(args.size, key)
    with mesh:
        params = T.shard_params(params, mesh, "fsdp", axis)
        opt_state = T.init_opt_state(params, optimizer)
        opt_state = T.shard_params(opt_state, mesh, "fsdp", axis) if opt_state else opt_state

    if pidx == 0:
        print(f"  params={T.param_count(params):,} optimizer={optimizer} lr={lr}")

    def train_step(params, opt_state, input_ids):
        loss, grads = jax.value_and_grad(T.loss_fn)(
            params, input_ids, n_heads, args.remat)
        new_params, new_state = T.opt_update(optimizer, params, grads, opt_state, lr)
        return new_params, new_state, loss

    # donate params + opt_state so XLA reuses their buffers in place.
    step_fn = jax.jit(train_step, donate_argnums=(0, 1))

    # ── Training loop (real compute + collectives, no sleeps) ────────────────
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
    data_key = jax.random.PRNGKey(1000)
    wall_start = time.time()
    step = 0
    # Rank-uniform exit: see T.agreed_step_budget. Every step here is a
    # collective, so a wall-clock loop condition wedges the slice the moment one
    # host's deadline fires while another is still inside. The budget is set from
    # a reference taken after step 1 so XLA compile time is excluded.
    n_budget = None
    _ref_step = _ref_t = None
    with mesh:
        while True:
            if n_budget is not None and step >= n_budget:
                break
            if n_budget is None and time.time() - wall_start >= 3 * args.duration:
                print('[fsdp] budget never agreed; stopping', flush=True)
                break
            data_key, sub = jax.random.split(data_key)
            input_ids = T.synthetic_batch(sub, batch, args.seq_len, vocab)
            input_ids = jax.device_put(
                input_ids, jax.sharding.NamedSharding(
                    mesh, jax.sharding.PartitionSpec()))
            params, opt_state, loss = step_fn(params, opt_state, input_ids)
            jax.block_until_ready(loss)
            step += 1

            if _ref_t is None and step >= 1:
                _ref_t, _ref_step = time.time(), step
            elif n_budget is None and step >= 4:
                n_budget = T.agreed_step_budget(
                    step - _ref_step, time.time() - _ref_t,
                    args.duration - (time.time() - wall_start), step)
                if pidx == 0:
                    print(f'  step budget agreed at step={step}: {n_budget}', flush=True)

            # Per-window ground truth from EVERY host (multi-host = all workers).
            T.emit_tpu_step("train", step,
                            tokens=batch * args.seq_len,
                            passes=1, loss=f"{float(loss):.4f}")

            if pidx == 0 and step % 10 == 0:
                elapsed = time.time() - wall_start
                tok_s = step * batch * args.seq_len / elapsed
                print(f"  step {step}: loss={float(loss):.4f} "
                      f"{tok_s:.0f} tok/s elapsed={elapsed:.0f}s/{args.duration}s",
                      flush=True)

    if pidx == 0:
        wall = time.time() - wall_start
        print(f"TPU FSDP training complete: {step} steps in {wall:.1f}s")


if __name__ == "__main__":
    main()
