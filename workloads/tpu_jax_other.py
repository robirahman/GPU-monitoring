#!/usr/bin/env python3
"""TPU 'other' cell -- the non-ML guard rung of the comm-scope ladder.

Two non-ML GPU-bound workloads, so the classifier's "other" class has a TPU
representative and, crucially, so we can test the failure mode "the model just
learned that communication => training":

  --mode fft --sharded   A large multi-axis FFT with the data sharded across the
                         whole slice. The transform along the sharded axis forces
                         an all-to-all every pass -- heavy CROSS-HOST collective
                         traffic that is NOT training. If the ladder's S16 gain
                         survives this negative, the signal is genuinely the
                         gradient-sync pattern and not merely "comm is present."

  --mode nbody           An O(N^2) gravitational n-body step -- compute-bound,
                         little communication -- the contrasting non-ML negative.

No STEP lines are emitted: this cell is neither train nor infer, and its label
(`tpuT*_other_*`) already routes to the "other" class via threeway_label.

Launch (per host, via run_workload_tpu.py):
    python workloads/tpu_jax_other.py --mode fft --sharded --duration 600
    python workloads/tpu_jax_other.py --mode nbody --duration 600
"""

import argparse
import sys
import time

try:
    from workloads import tpu_transformer_common as T
except ImportError:
    import tpu_transformer_common as T


def run_fft(args):
    import jax
    import jax.numpy as jnp

    sharded = args.sharded
    if sharded:
        try:
            jax.distributed.initialize()
        except Exception as e:
            print(f"[tpu_other] distributed.initialize skipped: {e}", flush=True)
        mesh, axis = T.make_mesh("fsdp")   # 1-D mesh over the whole slice
    else:
        mesh, axis = T.make_mesh("local")

    pidx = T.process_index()
    n_dev = int(mesh.devices.size)
    # Leading axis divisible by device count so it shards cleanly.
    rows = n_dev * args.fft_rows
    cols = args.fft_cols

    key = jax.random.PRNGKey(0)
    base = (jax.random.normal(key, (rows, cols), dtype=jnp.float32)
            .astype(jnp.complex64))

    if sharded:
        spec = jax.sharding.PartitionSpec(axis, None)   # shard the leading axis
    else:
        spec = jax.sharding.PartitionSpec()
    shard = jax.sharding.NamedSharding(mesh, spec)
    base = jax.device_put(base, shard)

    def fft_pass(x):
        # FFT along the *non*-sharded axis is local; the FFT along the sharded
        # (leading) axis needs an all-to-all -> cross-host collective traffic.
        x = jnp.fft.fft(x, axis=1)
        x = jnp.fft.fft(x, axis=0)
        return jnp.fft.ifft(x, axis=0)

    fft_fn = jax.jit(fft_pass)

    if pidx == 0:
        print(f"TPU FFT ({'sharded' if sharded else 'local'}): "
              f"{rows}x{cols} complex64 on {n_dev} chips")

    wall_start = time.time()
    rnd = 0
    x = base
    with mesh:
        while time.time() - wall_start < args.duration:
            for _ in range(args.inner):
                x = fft_fn(x)
            jax.block_until_ready(x)
            # Renormalise so magnitudes don't blow up over many passes.
            x = base
            rnd += 1
            if pidx == 0 and rnd % 20 == 0:
                elapsed = time.time() - wall_start
                print(f"  fft round {rnd}: elapsed={elapsed:.0f}s/{args.duration}s",
                      flush=True)
    if pidx == 0:
        print(f"TPU FFT complete: {rnd} rounds in {time.time() - wall_start:.1f}s")


def run_nbody(args):
    import jax
    import jax.numpy as jnp

    # Compute-bound negative: single host is enough (little comm to speak of).
    mesh, axis = T.make_mesh("local")
    pidx = T.process_index()
    n = args.n_bodies
    key = jax.random.PRNGKey(0)
    k1, k2 = jax.random.split(key)
    pos = jax.random.normal(k1, (n, 3), dtype=jnp.float32)
    vel = jax.random.normal(k2, (n, 3), dtype=jnp.float32) * 0.01
    dt, soft = 0.001, 1e-3

    def step(pos, vel):
        # O(N^2) pairwise gravity.
        d = pos[None, :, :] - pos[:, None, :]           # (N, N, 3)
        r2 = (d * d).sum(-1) + soft                     # (N, N)
        inv = r2 ** -1.5
        acc = (d * inv[:, :, None]).sum(1)              # (N, 3)
        vel = vel + dt * acc
        pos = pos + dt * vel
        return pos, vel

    step_fn = jax.jit(step)
    if pidx == 0:
        print(f"TPU n-body: {n} bodies, O(N^2) gravity on {int(mesh.devices.size)} chips")

    wall_start = time.time()
    it = 0
    with mesh:
        while time.time() - wall_start < args.duration:
            for _ in range(args.inner):
                pos, vel = step_fn(pos, vel)
            jax.block_until_ready(pos)
            it += args.inner
            if pidx == 0 and it % (args.inner * 20) == 0:
                elapsed = time.time() - wall_start
                print(f"  nbody step {it}: elapsed={elapsed:.0f}s/{args.duration}s",
                      flush=True)
    if pidx == 0:
        print(f"TPU n-body complete: {it} steps in {time.time() - wall_start:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="TPU non-ML 'other' cell")
    ap.add_argument("--mode", type=str, default="fft", choices=["fft", "nbody"])
    ap.add_argument("--sharded", action="store_true",
                    help="Shard the FFT across the slice (comm-heavy non-ML guard)")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--inner", type=int, default=8,
                    help="Inner iterations per block_until_ready round")
    ap.add_argument("--fft-rows", type=int, default=512,
                    help="Per-chip FFT rows (total rows = chips x this)")
    ap.add_argument("--fft-cols", type=int, default=4096)
    ap.add_argument("--n-bodies", type=int, default=8192)
    args = ap.parse_args()

    if not T._HAS_JAX:
        print("ERROR: jax not installed -- this cell runs on a TPU host.",
              file=sys.stderr)
        sys.exit(2)

    if args.mode == "fft":
        run_fft(args)
    else:
        run_nbody(args)


if __name__ == "__main__":
    main()
