#!/usr/bin/env python3
"""Shared pure-JAX decoder transformer for the TPU communication-scope ladder.

One model source drives BOTH the training cell (`tpu_fsdp_training.py`) and the
inference cells (`tpu_sharded_inference.py`), so train-vs-infer is a genuinely
matched pair: same architecture, same parameter count, same optimizer family --
only the communication regime (gradient collectives vs. activation psums vs.
none) differs. That is exactly the independent variable the ladder isolates.

Design constraints (from the approved plan, section 3c):
  * `MODEL_CONFIGS` is copied VERBATIM from `workloads/fsdp_training.py` so a
    `7b` TPU run is directly comparable to a `7b` GPU FSDP run.
  * `make_mesh(kind)` builds an fsdp / tp / local device mesh via
    `jax.sharding.Mesh` + `NamedSharding` + `PartitionSpec`.
  * Parameters live as a bf16 pytree with random init.
  * Optimizers (sgd / sgd_momentum / adamw) are hand-rolled over the pytree
    with `jax.tree_util` -- no optax dependency.
  * STEP ground truth is emitted through the shared superset dialect with
    `worker=jax.process_index()`.

ALL jax imports are deferred/guarded so this module byte-compiles and imports on
a machine without jax (the dev box); the jax-touching functions are only called
at run time on the TPU host. Nothing at import/definition time evaluates jax.
"""

import math

# numpy is always available (scientific stack); jax is TPU-only.
try:
    import numpy as np
except Exception:  # pragma: no cover - numpy is a hard dep of the repo
    np = None

try:
    import jax
    import jax.numpy as jnp
    from jax.sharding import Mesh, NamedSharding, PartitionSpec
    _HAS_JAX = True
except Exception:  # dev box: jax not installed -- keep names bound so defs compile
    jax = None
    jnp = None
    Mesh = NamedSharding = PartitionSpec = None
    _HAS_JAX = False

# STEP emitter: workloads run as scripts (cwd on path) or as a package module.
try:
    from workloads._step import emit_step
except ImportError:  # launched as `python workloads/tpu_*.py`
    from _step import emit_step


# ── Model configurations ─────────────────────────────────────────────────────
# Copied VERBATIM from workloads/fsdp_training.py:39 for cross-accelerator
# comparability. (hidden_dim, num_layers, num_heads, ffn_dim, vocab_size)

MODEL_CONFIGS = {
    "1b":  (2048, 24, 16, 8192,  32000),   # ~1.3B
    "7b":  (4096, 32, 32, 11008, 32000),   # ~6.7B
    "13b": (5120, 40, 40, 13824, 32000),   # ~13B
    "30b": (6656, 60, 52, 17920, 32000),   # ~30B
    "70b": (8192, 80, 64, 28672, 32000),   # ~70B
}

BATCH_DEFAULTS = {"1b": 8, "7b": 4, "13b": 2, "30b": 1, "70b": 1}

MAX_POS = 2048  # positional-embedding table size (matches fsdp_training.py)


# ── Process / host identity ──────────────────────────────────────────────────

def process_index() -> int:
    """This JAX process' index (0 on a single host, 0..H-1 across hosts)."""
    if _HAS_JAX:
        try:
            return int(jax.process_index())
        except Exception:
            return 0
    return 0


def process_count() -> int:
    if _HAS_JAX:
        try:
            return int(jax.process_count())
        except Exception:
            return 1
    return 1


def emit_tpu_step(kind, batch_idx, *, tokens=0, passes=1, **kv):
    """Emit one STEP line in the superset dialect from THIS worker.

    Every JAX process emits its own STEP around block_until_ready(), so a
    multi-host job labels all of its streams. node/rank/worker all carry the
    process index -- the physical host that produced the step -- which is what
    window_labels needs to attribute a window to a server.
    """
    pidx = process_index()
    emit_step(kind, batch_idx, tokens=tokens, passes=passes,
              node=pidx, rank=pidx, worker=pidx, **kv)


# ── Device mesh ──────────────────────────────────────────────────────────────

def make_mesh(kind="fsdp"):
    """Build a 1-D device mesh for the requested communication regime.

    kind:
      * "fsdp"  -- all *global* devices under axis 'fsdp'; params are sharded so
                   the forward all-gathers and the backward reduce-scatters
                   (gradient collectives = the training signature).
      * "tp"    -- all *global* devices under axis 'tp'; model dims are sharded
                   so matmuls emit cross-host activation psums (sharded-serving
                   signature, no gradient sync).
      * "local" -- this host's *local* devices only, under axis 'data'; used by
                   replicated inference where there is ZERO cross-host traffic.

    Returns (mesh, axis_name).
    """
    if not _HAS_JAX:
        raise RuntimeError("make_mesh requires jax (TPU host only)")
    devs = jax.local_devices() if kind == "local" else jax.devices()
    n = len(devs)
    axis = "data" if kind == "local" else kind
    arr = np.asarray(devs, dtype=object).reshape((n,))
    return Mesh(arr, (axis,)), axis


def _spec_for_leaf(x, kind, axis_name, n_devices):
    """PartitionSpec for one parameter leaf under the chosen regime.

    * local / single device / scalars  -> fully replicated
    * fsdp -> shard leading axis when divisible by device count (FSDP-style),
              else replicate; GSPMD inserts the all-gather / reduce-scatter.
    * tp   -> shard the LAST axis of >=2-D leaves when divisible (column
              parallel), forcing activation psums across the mesh.
    """
    if PartitionSpec is None:
        raise RuntimeError("jax required")
    if kind == "local" or n_devices <= 1 or getattr(x, "ndim", 0) == 0:
        return PartitionSpec()
    if kind == "fsdp":
        if x.ndim >= 1 and x.shape[0] % n_devices == 0:
            return PartitionSpec(axis_name)
        return PartitionSpec()
    if kind == "tp":
        if x.ndim >= 2 and x.shape[-1] % n_devices == 0:
            parts = [None] * x.ndim
            parts[-1] = axis_name
            return PartitionSpec(*parts)
        return PartitionSpec()
    return PartitionSpec()


def shard_params(params, mesh, kind, axis_name):
    """device_put every leaf onto its NamedSharding; GSPMD propagates from here.

    Sharding the parameters (rather than annotating every op) is enough to make
    XLA emit the collectives: a sharded matmul all-gathers/psums as needed and
    the backward inserts the matching reduce-scatter. That is the genuine
    cross-host communication the ladder is built to detect.
    """
    n = int(np.asarray(mesh.devices).size)

    def put(x):
        spec = _spec_for_leaf(x, kind, axis_name, n)
        return jax.device_put(x, NamedSharding(mesh, spec))

    return jax.tree_util.tree_map(put, params)


def replicate(tree, mesh):
    """Put a pytree fully-replicated across the mesh (batches, opt hyperparams)."""
    rep = NamedSharding(mesh, PartitionSpec())
    return jax.tree_util.tree_map(lambda x: jax.device_put(x, rep), tree)


# ── Parameter initialisation (bf16 pytree, random) ───────────────────────────

def init_params(config_name, key, dtype=None):
    """Random bf16 parameter pytree for the given model size.

    Layout mirrors RandomTransformerLM in fsdp_training.py: token + positional
    embeddings, N pre-norm decoder blocks (fused QKV projection), final norm,
    and an untied LM head.
    """
    if not _HAS_JAX:
        raise RuntimeError("init_params requires jax (TPU host only)")
    if dtype is None:
        dtype = jnp.bfloat16
    hidden, n_layers, n_heads, ffn, vocab = MODEL_CONFIGS[config_name]

    keys = jax.random.split(key, n_layers * 6 + 8)
    ki = iter(keys)

    def randn(k, shape, scale):
        return (jax.random.normal(k, shape, dtype=jnp.float32) * scale).astype(dtype)

    params = {
        "tok_emb": randn(next(ki), (vocab, hidden), 0.02),
        "pos_emb": randn(next(ki), (MAX_POS, hidden), 0.02),
        "blocks": [],
        "norm_w": jnp.ones((hidden,), dtype=dtype),
        "norm_b": jnp.zeros((hidden,), dtype=dtype),
        "lm_head": randn(next(ki), (hidden, vocab), 0.02),
    }
    scale = 0.02
    for _ in range(n_layers):
        params["blocks"].append({
            "ln1_w": jnp.ones((hidden,), dtype=dtype),
            "ln1_b": jnp.zeros((hidden,), dtype=dtype),
            # fused QKV: (hidden, 3*hidden)
            "wqkv": randn(next(ki), (hidden, 3 * hidden), scale),
            "wo":   randn(next(ki), (hidden, hidden), scale),
            "ln2_w": jnp.ones((hidden,), dtype=dtype),
            "ln2_b": jnp.zeros((hidden,), dtype=dtype),
            "w_in":  randn(next(ki), (hidden, ffn), scale),
            "w_out": randn(next(ki), (ffn, hidden), scale),
        })
    return params


def param_count(params):
    """Total number of scalars in the pytree (for logging)."""
    if not _HAS_JAX:
        return 0
    return int(sum(x.size for x in jax.tree_util.tree_leaves(params)))


# ── Forward / loss ───────────────────────────────────────────────────────────

def _layernorm(x, w, b, eps=1e-5):
    x32 = x.astype(jnp.float32)
    mu = x32.mean(-1, keepdims=True)
    var = x32.var(-1, keepdims=True)
    y = (x32 - mu) / jnp.sqrt(var + eps)
    return (y * w.astype(jnp.float32) + b.astype(jnp.float32)).astype(x.dtype)


def _attention(x, wqkv, wo, n_heads):
    B, T, H = x.shape
    dh = H // n_heads
    qkv = x @ wqkv                       # (B, T, 3H)
    q, k, v = jnp.split(qkv, 3, axis=-1)  # each (B, T, H)

    def heads(t):
        return t.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)  # (B, nh, T, dh)

    q, k, v = heads(q), heads(k), heads(v)
    scores = (q @ k.transpose(0, 1, 3, 2)) / math.sqrt(dh)  # (B, nh, T, T)
    # causal mask
    mask = jnp.tril(jnp.ones((T, T), dtype=bool))
    scores = jnp.where(mask, scores, jnp.finfo(jnp.float32).min).astype(jnp.float32)
    attn = jax.nn.softmax(scores, axis=-1).astype(x.dtype)
    out = attn @ v                       # (B, nh, T, dh)
    out = out.transpose(0, 2, 1, 3).reshape(B, T, H)
    return out @ wo


def _apply_block(blk, x, n_heads):
    """One pre-norm decoder block: attention + MLP residuals."""
    h = _layernorm(x, blk["ln1_w"], blk["ln1_b"])
    x = x + _attention(h, blk["wqkv"], blk["wo"], n_heads)
    h = _layernorm(x, blk["ln2_w"], blk["ln2_b"])
    h = jax.nn.gelu(h @ blk["w_in"]) @ blk["w_out"]
    return x + h


def forward(params, input_ids, n_heads, remat=False):
    """Decoder-only transformer forward. Returns logits (B, T, vocab).

    remat=True wraps each block in jax.checkpoint (activation checkpointing),
    trading recompute for memory so the larger sizes fit HBM -- the JAX analogue
    of fsdp_training.py's activation checkpointing.
    """
    T = input_ids.shape[1]
    pos = jnp.arange(T)
    x = params["tok_emb"][input_ids] + params["pos_emb"][pos][None, :, :]
    block_fn = jax.checkpoint(_apply_block, static_argnums=(2,)) if remat else _apply_block
    for blk in params["blocks"]:
        x = block_fn(blk, x, n_heads)
    x = _layernorm(x, params["norm_w"], params["norm_b"])
    return x @ params["lm_head"]


def loss_fn(params, input_ids, n_heads, remat=False):
    """Next-token cross-entropy over synthetic tokens (scalar)."""
    logits = forward(params, input_ids, n_heads, remat).astype(jnp.float32)
    shift_logits = logits[:, :-1, :]
    shift_labels = input_ids[:, 1:]
    logp = jax.nn.log_softmax(shift_logits, axis=-1)
    B, Tm1, V = logp.shape
    tgt = jax.nn.one_hot(shift_labels, V, dtype=logp.dtype)
    return -(logp * tgt).sum(-1).mean()


def synthetic_batch(key, batch, seq, vocab):
    """Random token ids -- no dataset needed (matches fsdp_training.py)."""
    return jax.random.randint(key, (batch, seq), 0, vocab, dtype=jnp.int32)


# ── Hand-rolled optimizers over the pytree (jax.tree_util) ───────────────────

def init_opt_state(params, optimizer):
    if not _HAS_JAX:
        return {}
    tm = jax.tree_util.tree_map
    if optimizer == "adamw":
        return {"m": tm(jnp.zeros_like, params),
                "v": tm(jnp.zeros_like, params),
                "t": jnp.zeros((), dtype=jnp.int32)}
    if optimizer == "sgd_momentum":
        return {"v": tm(jnp.zeros_like, params)}
    return {}


def opt_update(optimizer, params, grads, state, lr):
    """Return (new_params, new_state) for one optimizer step, purely functional."""
    tm = jax.tree_util.tree_map
    if optimizer == "adamw":
        b1, b2, eps, wd = 0.9, 0.999, 1e-8, 0.01
        t = state["t"] + 1
        m = tm(lambda m_, g: b1 * m_ + (1 - b1) * g, state["m"], grads)
        v = tm(lambda v_, g: b2 * v_ + (1 - b2) * (g * g), state["v"], grads)
        bc1 = 1.0 - b1 ** t.astype(jnp.float32)
        bc2 = 1.0 - b2 ** t.astype(jnp.float32)

        def upd(p, m_, v_):
            mhat = (m_.astype(jnp.float32)) / bc1
            vhat = (v_.astype(jnp.float32)) / bc2
            step = mhat / (jnp.sqrt(vhat) + eps) + wd * p.astype(jnp.float32)
            return (p.astype(jnp.float32) - lr * step).astype(p.dtype)

        new = tm(upd, params, m, v)
        return new, {"m": m, "v": v, "t": t}

    if optimizer == "sgd_momentum":
        mom = 0.9
        v = tm(lambda v_, g: mom * v_ + g, state["v"], grads)
        new = tm(lambda p, v_: (p.astype(jnp.float32) - lr * v_.astype(jnp.float32)).astype(p.dtype),
                 params, v)
        return new, {"v": v}

    # plain sgd
    new = tm(lambda p, g: (p.astype(jnp.float32) - lr * g.astype(jnp.float32)).astype(p.dtype),
             params, grads)
    return new, state


def auto_optimizer(size, override=None):
    """Match fsdp_training.py's auto rule: adamw <=13b, sgd_momentum 30b, sgd 70b."""
    if override:
        return override
    if size == "70b":
        return "sgd"
    if size == "30b":
        return "sgd_momentum"
    return "adamw"


# ── CPU gradient self-test (G1 gate) ─────────────────────────────────────────

def self_test(verbose=True):
    """CPU-only correctness gate: analytic gradient vs. finite difference on a
    tiny model, plus a one-step loss-decrease check. Runs with no distribution
    and no TPU -- set XLA_FLAGS=--xla_force_host_platform_device_count=8 to also
    exercise the mesh path on CPU. Prints 'self-test PASS' on success so a queue
    can gate on it (grep, exactly like the Round 6 G1 gate).
    """
    if not _HAS_JAX:
        print("self-test SKIP: jax not installed (dev box); byte-compile only")
        return False

    # A deliberately tiny config so the check is instant on CPU.
    tiny = {"tiny": (32, 2, 4, 64, 97)}
    global MODEL_CONFIGS
    saved = MODEL_CONFIGS
    MODEL_CONFIGS = {**MODEL_CONFIGS, **tiny}
    try:
        _, _, n_heads, _, vocab = MODEL_CONFIGS["tiny"]
        key = jax.random.PRNGKey(0)
        params = init_params("tiny", key, dtype=jnp.float32)
        batch = synthetic_batch(jax.random.PRNGKey(1), 2, 8, vocab)

        loss0, grads = jax.value_and_grad(lambda p: loss_fn(p, batch, n_heads))(params)
        loss0 = float(loss0)

        # Finite-difference check on a single scalar of the LM head.
        eps = 1e-3
        g = np.asarray(grads["lm_head"])[0, 0]
        base = np.asarray(params["lm_head"])
        pp = params.copy(); pm = params.copy()
        up = base.copy(); up[0, 0] += eps
        dn = base.copy(); dn[0, 0] -= eps
        pp["lm_head"] = jnp.asarray(up)
        pm["lm_head"] = jnp.asarray(dn)
        lp = float(loss_fn(pp, batch, n_heads))
        lm = float(loss_fn(pm, batch, n_heads))
        fd = (lp - lm) / (2 * eps)
        rel = abs(fd - g) / (abs(fd) + abs(g) + 1e-8)

        # One SGD step must not increase the loss (gradient points downhill).
        new, _ = opt_update("adamw", params, grads, init_opt_state(params, "adamw"), 1e-3)
        loss1 = float(loss_fn(new, batch, n_heads))

        ok = math.isfinite(loss0) and rel < 5e-2 and loss1 <= loss0 + 1e-3
        if verbose:
            print(f"  loss0={loss0:.5f} loss1={loss1:.5f} "
                  f"grad={g:.5f} fd={fd:.5f} rel_err={rel:.2e}")

        # If >1 device is visible (e.g. under
        # XLA_FLAGS=--xla_force_host_platform_device_count=8), also exercise the
        # fsdp mesh / shard_params / sharded jitted train-step path on CPU so a
        # sharding bug surfaces here at V0 rather than on the TPU.
        if jax.device_count() > 1:
            try:
                mesh, axis = make_mesh("fsdp")
                with mesh:
                    p = shard_params(params, mesh, "fsdp", axis)
                    st = init_opt_state(p, "adamw")

                    def _step(p, st, x):
                        l, g2 = jax.value_and_grad(
                            lambda pp: loss_fn(pp, x, n_heads))(p)
                        p2, st2 = opt_update("adamw", p, g2, st, 1e-3)
                        return p2, st2, l

                    step_fn = jax.jit(_step)
                    xb = jax.device_put(
                        batch, NamedSharding(mesh, PartitionSpec()))
                    p, st, lmesh = step_fn(p, st, xb)
                    jax.block_until_ready(lmesh)
                mesh_ok = math.isfinite(float(lmesh))
                if verbose:
                    print(f"  mesh({jax.device_count()} dev) step loss={float(lmesh):.5f}")
                ok = ok and mesh_ok
            except Exception as e:
                print(f"  mesh self-test FAILED: {e}")
                ok = False

        print("self-test PASS" if ok else "self-test FAIL")
        return ok
    finally:
        MODEL_CONFIGS = saved


if __name__ == "__main__":
    self_test()
