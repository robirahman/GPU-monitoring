#!/usr/bin/env python3
"""TPU sharded-inference cell -- the INFERENCE rungs of the comm-scope ladder.

Serves the SAME shared transformer as the training cell, closed-loop and
saturating (no arrival process -- matching every GPU inference loop in the
repo), so train-vs-infer differs only in communication regime:

  --mode tp          full-mesh tensor-parallel serving. Params are sharded
                     across ALL chips of the slice, so each decode step emits
                     cross-host activation psums (all-reduce over the model
                     axis) -- realistic-adversary "sharded serving" that has
                     inter-server traffic but NO gradient sync. Brings up the
                     cross-host cluster (jax.distributed.initialize()).

  --mode replicated  independent per-host replicas. Uses jax.local_devices()
                     ONLY and never calls jax.distributed.initialize(), so there
                     is ZERO cross-host traffic -- the clean negative. (If the
                     host carve-out is unavailable, the fallback is 4x
                     sequential v6e-4; see plan gate G4.)

Launch (per host, via run_workload_tpu.py):
    python workloads/tpu_sharded_inference.py --size 7b --mode tp --duration 600
    python workloads/tpu_sharded_inference.py --size 7b --mode replicated --duration 600
"""

import argparse
import sys
import time

try:
    from workloads import tpu_transformer_common as T
except ImportError:
    import tpu_transformer_common as T


def main():
    ap = argparse.ArgumentParser(description="TPU sharded / replicated inference cell")
    ap.add_argument("--size", type=str, default="7b",
                    choices=list(T.MODEL_CONFIGS.keys()))
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--mode", type=str, default="tp", choices=["tp", "replicated"],
                    help="tp = full-mesh sharded (cross-host psums); "
                         "replicated = local devices only, no cross-host sync")
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Concurrent sequences per decode round")
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="Tokens generated per closed-loop round")
    args = ap.parse_args()

    if not T._HAS_JAX:
        print("ERROR: jax not installed -- this cell runs on a TPU host.",
              file=sys.stderr)
        sys.exit(2)

    import jax
    import jax.numpy as jnp

    # ── Communication regime ────────────────────────────────────────────────
    if args.mode == "tp":
        # Full-slice tensor parallel: bring up the cross-host cluster.
        try:
            jax.distributed.initialize()
        except Exception as e:
            print(f"[tpu_infer] distributed.initialize skipped: {e}", flush=True)
        mesh, axis = T.make_mesh("tp")
        shard_kind = "tp"
    else:
        # Replicated: LOCAL devices only, NO distributed.initialize() -- this is
        # what guarantees the cell has zero cross-host traffic.
        mesh, axis = T.make_mesh("local")
        shard_kind = "local"

    pidx = T.process_index()
    _, _, n_heads, _, vocab = T.MODEL_CONFIGS[args.size]
    total_len = args.prompt_len + args.gen_tokens

    key = jax.random.PRNGKey(0)
    params = T.init_params(args.size, key)
    with mesh:
        params = T.shard_params(params, mesh, shard_kind, axis)

    if pidx == 0:
        n_dev = mesh.devices.size
        print(f"TPU {args.mode} inference: {args.size} on {n_dev} chips "
              f"({'global slice' if args.mode == 'tp' else 'local host'})")
        print(f"  batch={args.batch_size} prompt_len={args.prompt_len} "
              f"gen_tokens={args.gen_tokens} params={T.param_count(params):,}")

    def generate(params, tokens, start, n_gen):
        """Closed-loop greedy decode: fill positions [start, start+n_gen)."""
        def body(i, toks):
            logits = T.forward(params, toks, n_heads)
            pos = start + i
            nxt = jnp.argmax(logits[:, pos - 1, :], axis=-1).astype(toks.dtype)
            return toks.at[:, pos].set(nxt)
        return jax.lax.fori_loop(0, n_gen, body, tokens)

    gen_fn = jax.jit(generate, static_argnums=(2, 3))

    # ── Saturating closed-loop serving ──────────────────────────────────────
    data_key = jax.random.PRNGKey(2000 + pidx)
    wall_start = time.time()
    rnd = 0
    with mesh:
        while time.time() - wall_start < args.duration:
            data_key, sub = jax.random.split(data_key)
            prompt = T.synthetic_batch(sub, args.batch_size, args.prompt_len, vocab)
            # Pad the prompt out to the full generation buffer (static shape).
            tokens = jnp.zeros((args.batch_size, total_len), dtype=jnp.int32)
            tokens = tokens.at[:, :args.prompt_len].set(prompt)
            out = gen_fn(params, tokens, args.prompt_len, args.gen_tokens)
            jax.block_until_ready(out)
            rnd += 1

            # One STEP per decode round, from every worker.
            T.emit_tpu_step("infer", rnd,
                            tokens=args.batch_size * args.gen_tokens,
                            passes=1)

            if pidx == 0 and rnd % 10 == 0:
                elapsed = time.time() - wall_start
                tok_s = rnd * args.batch_size * args.gen_tokens / elapsed
                print(f"  round {rnd}: {tok_s:.0f} tok/s "
                      f"elapsed={elapsed:.0f}s/{args.duration}s", flush=True)

    if pidx == 0:
        wall = time.time() - wall_start
        print(f"TPU {args.mode} inference complete: {rnd} rounds in {wall:.1f}s")


if __name__ == "__main__":
    main()
