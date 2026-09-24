#!/usr/bin/env python3
"""Shared plumbing for the SPAR interconnect workloads (W03-W05, W10, W15-W19).

Not a workload itself. Import from a sibling script with a bare import
(``from interconnect_common import ...``) -- same convention as
``diloco_production.py`` importing ``diloco_core``.

What lives here
  * ``dist_setup``      -- NCCL on GPU boxes, gloo on CPU (so every workload can be
                           smoke-tested with ``torchrun --nproc_per_node=2 ... --backend gloo``)
  * ``JsonlLog``        -- thread-safe rank0 JSONL collective log (+ optional per-rank files)
  * ``StopSignal``      -- all ranks stop on the same step (a per-rank ``time.time()``
                           check can straddle the deadline and hang the last collective)
  * ``CausalLM``        -- sized decoder LM, same size family as fsdp_training / diloco_production
  * KV geometry helpers -- shared by W10 (disagg KV) and W18 (KV-shaped grads) so both
                           use one size schedule
  * chunked collectives, logging DDP comm hook

``bytes_est`` convention in every comm log (per rank, send side):
  all_reduce   ring estimate 2*(N-1)/N * payload   (same as diloco_core.CommEvent)
  all_to_all   payload = whole send buffer          (same as moe_ep_synthetic.py)
  p2p_*        payload
``payload_bytes`` is always the raw buffer size, so either convention can be recomputed.
"""
from __future__ import annotations

import contextlib
import json
import os
import socket
import threading
import time
import warnings
from dataclasses import dataclass
from typing import Callable, Sequence

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

# torch >= 2.13 warns that DDP(broadcast_buffers=...) is deprecated; the replacement
# (forward_sync_buffers) does not exist on the older torch these run on, so keep the old
# spelling and silence the notice.
warnings.filterwarnings("ignore", message=".*broadcast_buffers.*", category=FutureWarning)


# ── Distributed setup ────────────────────────────────────────────────────────

@dataclass
class DistEnv:
    rank: int
    world: int
    local_rank: int
    device: torch.device
    backend: str
    ctrl_group: object | None = None   # gloo group for control-plane flags (NCCL runs)

    @property
    def is_main(self) -> bool:
        return self.rank == 0

    def sync(self) -> None:
        """Block until queued device work (incl. NCCL ops on this stream) is done."""
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)


def _free_port() -> int:
    with socket.socket() as s:
        s.bind(("", 0))
        return s.getsockname()[1]


def dist_setup(backend: str = "auto") -> DistEnv:
    """Init the default process group.

    ``auto`` -> nccl if CUDA is visible else gloo. Without torchrun env vars we fall
    back to a single-process world so ``python workload.py --help``-style smoke runs work.
    """
    if "RANK" not in os.environ:
        os.environ.update(RANK="0", WORLD_SIZE="1", LOCAL_RANK="0")
        os.environ.setdefault("MASTER_ADDR", "127.0.0.1")
        os.environ.setdefault("MASTER_PORT", str(_free_port()))
    rank = int(os.environ["RANK"])
    world = int(os.environ["WORLD_SIZE"])
    local = int(os.environ.get("LOCAL_RANK", 0))

    has_cuda = torch.cuda.is_available()
    if backend == "auto":
        backend = "nccl" if has_cuda else "gloo"
    if backend == "nccl" and not has_cuda:
        raise SystemExit("--backend nccl requested but CUDA is not available (use --backend gloo)")

    if backend == "nccl":
        torch.cuda.set_device(local)
        device = torch.device("cuda", local)
    else:
        device = torch.device("cpu")
    dist.init_process_group(backend=backend)
    return env_from_initialized(backend, device, local)


def env_from_initialized(backend: str, device: torch.device, local_rank: int | None = None) -> DistEnv:
    """DistEnv for a process group the caller already initialised (adds the gloo control group).

    Collective: with NCCL every rank must call this at the same point.
    """
    rank, world = dist.get_rank(), dist.get_world_size()
    ctrl = None
    if backend == "nccl" and world > 1:
        try:
            ctrl = dist.new_group(backend="gloo")
        except Exception:                            # noqa: BLE001 - fall back to NCCL flag
            ctrl = None
    return DistEnv(rank, world, local_rank if local_rank is not None else (device.index or 0),
                   device, backend, ctrl)


def dist_cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class StopSignal:
    """Rank-consistent stop condition (duration and/or max steps).

    Rank 0 owns the clock and broadcasts the flag over the (gloo) control group, so
    the tiny flag never shows up as fabric traffic on NCCL runs. Every rank must call
    ``should_stop`` at the same points in the loop.
    """

    def __init__(self, env: DistEnv, duration: float | None, max_steps: int | None = None,
                 check_every: int = 1) -> None:
        self.env = env
        self.duration = duration
        self.max_steps = max_steps
        self.check_every = max(1, check_every)
        self.t0 = time.time()

    def should_stop(self, step: int) -> bool:
        if self.max_steps is not None and step >= self.max_steps:
            return True
        if self.duration is None or step % self.check_every != 0:
            return False
        flag = torch.zeros(1, dtype=torch.int32)
        if self.env.is_main and time.time() - self.t0 >= self.duration:
            flag.fill_(1)
        if self.env.world > 1:
            if self.env.ctrl_group is None and self.env.device.type == "cuda":
                flag = flag.to(self.env.device)
            dist.broadcast(flag, src=0, group=self.env.ctrl_group)
        return bool(flag.item())


# ── Byte accounting + JSONL logs ─────────────────────────────────────────────

def ar_wire_bytes(payload: int, n: int) -> int:
    return int(2 * (n - 1) / n * payload) if n > 1 else 0


def tensor_bytes(t: torch.Tensor) -> int:
    return t.numel() * t.element_size()


def _rank_path(path: str, rank: int) -> str:
    root, ext = os.path.splitext(path)
    return f"{root}.rank{rank}{ext or '.jsonl'}"


def role_tags_from_env() -> dict[str, str]:
    """W25 role tags set by ``scripts/run_colocate_serve_train.py`` for its children (empty otherwise).

    ``workload_role`` = serve|train, ``role_label`` = that role's workload label, ``run_id`` = the
    shared colocation run id. Stamped on every JsonlLog event and meta sidecar so comm/step logs
    stay attributable when two workloads share a node.
    """
    m = {"workload_role": "SPAR_WORKLOAD_ROLE", "role_label": "SPAR_ROLE_LABEL", "run_id": "SPAR_RUN_ID"}
    return {k: os.environ[v] for k, v in m.items() if os.environ.get(v)}


class JsonlLog:
    """Append-only JSONL log, one event per line.

    Rank 0 writes ``path``; other ranks write ``<stem>.rank<N>.jsonl`` only when
    ``all_ranks`` is set (needed to see both ends of P2P traffic). ``step`` /
    ``microstep`` are plain attributes the training loop updates; ``emit`` stamps them
    on every event so hook callbacks running on autograd threads need no plumbing.
    """

    def __init__(self, path: str | None, env: DistEnv, all_ranks: bool = False) -> None:
        self.rank = env.rank
        self.world = env.world
        self.step = 0
        self.microstep = 0
        self.ctx: dict = dict(role_tags_from_env())   # stamped on every event (+ is_sync_step, ...)
        self.path: str | None = None
        self._fh = None
        self._lock = threading.Lock()
        if path:
            if env.rank == 0:
                self.path = path
            elif all_ranks:
                self.path = _rank_path(path, env.rank)
        if self.path:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)

    @property
    def enabled(self) -> bool:
        return self._fh is not None

    def emit(self, **kw) -> None:
        if self._fh is None:
            return
        kw.setdefault("t", time.time())
        kw.setdefault("step", self.step)
        kw.setdefault("rank", self.rank)
        kw.setdefault("world_size", self.world)
        for k, v in self.ctx.items():
            kw.setdefault(k, v)
        line = json.dumps(kw, separators=(",", ":"), default=str)
        with self._lock:
            self._fh.write(line + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def sibling_path(path: str, suffix: str) -> str:
    """``data/x.jsonl`` + ``.steps`` -> ``data/x.steps.jsonl``."""
    root, ext = os.path.splitext(path)
    return f"{root}{suffix}{ext or '.jsonl'}"


def write_meta(comm_log_path: str, meta: dict, env: DistEnv) -> None:
    """Rank0 sidecar ``<stem>.meta.json`` with config / topology. Keeps the JSONL pure."""
    if not comm_log_path or not env.is_main:
        return
    root, _ = os.path.splitext(comm_log_path)
    os.makedirs(os.path.dirname(root) or ".", exist_ok=True)
    with open(root + ".meta.json", "w") as f:
        json.dump({**meta, **role_tags_from_env()}, f, indent=2, default=str)


def add_common_args(ap, *, duration: float = 300.0) -> None:
    ap.add_argument("--backend", choices=["auto", "nccl", "gloo"], default="auto",
                    help="auto = nccl when CUDA is visible, else gloo (CPU smoke tests)")
    ap.add_argument("--duration", type=float, default=duration,
                    help="Wall-clock seconds (rank0 clock; all ranks stop together)")
    ap.add_argument("--max-steps", type=int, default=None,
                    help="Also stop after this many steps/rounds")
    ap.add_argument("--comm-log", type=str, default="",
                    help="Rank0 JSONL path, one collective per line (+ <stem>.meta.json)")
    ap.add_argument("--comm-log-all-ranks", action="store_true",
                    help="Also write <stem>.rank<N>.jsonl on every other rank")
    ap.add_argument("--seed", type=int, default=0)


def make_logging_allreduce_hook(log: JsonlLog, *, phase: str, group_name: str = "dp"):
    """DDP comm hook = stock allreduce_hook + one JSONL event per bucket.

    DDP's default C++ reducer is invisible from Python; routing it through this hook
    keeps the bytes on the wire identical and makes every bucket loggable. ``state``
    may be a process group (or None for WORLD). ``t`` is the *issue* time (async op).
    """
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    def hook(state, bucket):
        buf = bucket.buffer()
        n = state.size() if state is not None else dist.get_world_size()
        payload = tensor_bytes(buf)
        log.emit(op="all_reduce", phase=phase, group=group_name, group_size=n,
                 bytes_est=ar_wire_bytes(payload, n), payload_bytes=payload,
                 bucket_index=bucket.index(), last_bucket=bool(bucket.is_last()),
                 async_op=True, microstep=log.microstep)
        return default_hooks.allreduce_hook(state, bucket)

    return hook


class AllReduceTap:
    """Context manager: wrap ``dist.all_reduce`` so every Python-level AllReduce is logged.

    Sees everything issued from Python -- DDP comm hooks (PowerSGD, allreduce_hook) and
    manual calls -- with exact per-collective sizes. Does NOT see DDP's default C++
    reducer path (register a Python hook for that, e.g. ``make_logging_allreduce_hook``).
    Per-step totals for compression-ratio reporting: ``pop_step()``.
    """

    def __init__(self, log: JsonlLog, phase_fn: Callable[[], str | None], group_name: str = "dp") -> None:
        self.log = log
        self.phase_fn = phase_fn
        self.group_name = group_name
        self._orig = None
        self._lock = threading.Lock()
        self._bytes = 0
        self._count = 0

    def __enter__(self) -> "AllReduceTap":
        self._orig = dist.all_reduce
        dist.all_reduce = self._tapped
        return self

    def __exit__(self, *exc) -> None:
        dist.all_reduce = self._orig

    def _tapped(self, tensor, *args, **kwargs):
        # all_reduce(tensor, op=SUM, group=None, async_op=False)
        phase = self.phase_fn()
        if phase is None:                   # e.g. cover traffic: logged by its owner, not a grad sync
            return self._orig(tensor, *args, **kwargs)
        group = kwargs.get("group", args[1] if len(args) > 1 else None)
        n = dist.get_world_size(group)
        payload = tensor_bytes(tensor)
        est = ar_wire_bytes(payload, n)
        with self._lock:
            self._bytes += est
            self._count += 1
        self.log.emit(op="all_reduce", phase=phase, group=self.group_name, group_size=n,
                      bytes_est=est, payload_bytes=payload, numel=tensor.numel(),
                      dtype=str(tensor.dtype).replace("torch.", ""),
                      async_op=bool(kwargs.get("async_op", args[2] if len(args) > 2 else False)))
        return self._orig(tensor, *args, **kwargs)

    def pop_step(self) -> tuple[int, int]:
        """(estimated wire bytes, collective count) since the last call."""
        with self._lock:
            out = (self._bytes, self._count)
            self._bytes = self._count = 0
        return out


class CommTrace:
    """--comm-log support for an existing DDP training loop (W15).

    * Registers ``make_logging_allreduce_hook`` on the DDP model -> one JSONL event per
      gradient-bucket AllReduce, stamped with ``step`` / ``microstep`` / ``is_sync_step``.
      The hook goes on *every* rank whenever a path is given (a hook on some ranks and
      the C++ reducer on others would average gradients inconsistently); only rank 0
      writes. ``model.no_sync()`` steps never reach the hook, so accumulation
      micro-steps correctly log nothing.
    * ``window(...)`` writes one record per batch to ``<stem>.steps.jsonl`` with
      ``t``/``t_end`` and the batch phase (``train_sync`` | ``train_accum`` |
      ``train_local_nosync`` | ``infer_forward``): window truth for rare-sync regimes.

    With no path both pieces are no-ops, so existing runs are unchanged.
    """

    def __init__(self, comm_path: str, env: DistEnv, ddp_model=None, step_path: str = "",
                 phase: str = "grad_sync") -> None:
        self.env = env
        self.on = bool(comm_path)
        self.comm = JsonlLog(comm_path, env)
        self.steps = JsonlLog(step_path or (sibling_path(comm_path, ".steps") if comm_path else ""), env)
        if self.on and ddp_model is not None:
            ddp_model.register_comm_hook(None, make_logging_allreduce_hook(self.comm, phase=phase))

    @contextlib.contextmanager
    def window(self, *, step: int, microstep: int, is_sync_step: bool, phase: str, **extra):
        if not self.comm.enabled:
            yield
            return
        self.comm.step = self.steps.step = step
        self.comm.microstep = microstep
        self.comm.ctx = {"is_sync_step": is_sync_step, "batch_phase": phase}
        t0 = time.time()
        yield
        self.env.sync()
        self.steps.emit(t=t0, t_end=time.time(), step=step, microstep=microstep,
                        is_sync_step=is_sync_step, phase=phase, **extra)

    def close(self) -> None:
        self.comm.close()
        self.steps.close()


def make_cover_group(env: DistEnv, tp: int):
    """TP-like subgroups of ``tp`` consecutive ranks -> (my group or None=WORLD, my ranks).

    Collective: every rank creates every group, in the same order.
    """
    if tp == env.world:
        return None, list(range(env.world))
    if env.world % tp:
        raise SystemExit(f"--cover-tp {tp} must divide world size {env.world}")
    mine = None
    for start in range(0, env.world, tp):
        ranks = list(range(start, start + tp))
        g = dist.new_group(ranks)
        if env.rank in ranks:
            mine = (g, ranks)
    return mine


class CoverTraffic:
    """Decode-shaped cover traffic (W19 mechanism, W24 threat cell).

    Per call to ``emit``: ``layers * ars_per_layer * steps_per_call`` small AllReduces of a
    ``tokens x hidden`` activation-sized buffer on a TP-like subgroup -- what TP decode serving
    puts on the wire. Defaults match W08 (Qwen2.5-7B: hidden 3584, 28 layers, 2 AR/layer,
    4 decode rows). The collectives are real NCCL AllReduces of zeros; only their *purpose*
    is cover. ``phase=decode_shape`` in the comm log is the ground-truth label for them.

    ``in_cover`` is True while ``emit`` runs so an :class:`AllReduceTap` can exclude cover
    from gradient-sync accounting (``phase_fn`` returning None).
    """

    def __init__(self, env: DistEnv, comm: JsonlLog, *, tp: int = 0, hidden: int = 3584,
                 layers: int = 28, ars_per_layer: int = 2, tokens: int = 4, dtype: str = "bf16",
                 steps_per_call: int = 4, log_mode: str = "aggregate") -> None:
        self.env, self.comm, self.log_mode = env, comm, log_mode
        self.tp = tp or env.world
        self.group, self.ranks = make_cover_group(env, self.tp)
        # CPU/gloo: keep bf16 when asked (fp16 collectives are not universally supported)
        dt = dtype_from_name(dtype) if env.device.type == "cuda" else (
            torch.bfloat16 if dtype == "bf16" else torch.float32)
        self.buf = torch.zeros(tokens, hidden, device=env.device, dtype=dt)
        self.msg_bytes = tensor_bytes(self.buf)
        self.ars_per_call = layers * ars_per_layer * steps_per_call
        self.wire_bytes_per_call = self.ars_per_call * ar_wire_bytes(self.msg_bytes, self.tp)
        self.in_cover = False

    def emit(self) -> None:
        if self.tp <= 1:
            return
        self.in_cover = True
        try:
            t0 = time.time()
            for _ in range(self.ars_per_call):
                dist.all_reduce(self.buf, group=self.group)
                if self.log_mode == "full":
                    self.comm.emit(op="all_reduce", phase="decode_shape", group="tp", group_size=self.tp,
                                   bytes_est=ar_wire_bytes(self.msg_bytes, self.tp),
                                   payload_bytes=self.msg_bytes)
            if self.log_mode == "aggregate":
                self.comm.emit(t=t0, dur_s=time.time() - t0, op="all_reduce", phase="decode_shape",
                               group="tp", group_size=self.tp, count=self.ars_per_call,
                               msg_bytes=self.msg_bytes, bytes_est=self.wire_bytes_per_call,
                               payload_bytes=self.msg_bytes * self.ars_per_call)
            self.buf.zero_()               # keep values bounded across thousands of AllReduces
        finally:
            self.in_cover = False

    def meta(self) -> dict:
        return {"cover_tp": self.tp, "cover_group_ranks": self.ranks, "cover_msg_bytes": self.msg_bytes,
                "cover_collectives_per_step": self.ars_per_call,
                "cover_wire_bytes_est_per_step": self.wire_bytes_per_call}


def add_cover_args(ap) -> None:
    """CLI for :class:`CoverTraffic` (defaults ~ Qwen2.5-7B decode: the W08 scale)."""
    ap.add_argument("--cover-tp", type=int, default=0, help="TP subgroup size (0 = whole world)")
    ap.add_argument("--decode-hidden", type=int, default=3584)
    ap.add_argument("--decode-layers", type=int, default=28)
    ap.add_argument("--ars-per-layer", type=int, default=2, help="TP: attn-out + mlp-out")
    ap.add_argument("--decode-tokens", type=int, default=4, help="Batch rows per decode AllReduce")
    ap.add_argument("--decode-dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--cover-tokens", type=int, default=4,
                    help="Decode steps' worth of cover traffic emitted per training step")
    ap.add_argument("--log-decode", choices=["aggregate", "full", "off"], default="aggregate")


def cover_from_args(env, comm, args) -> "CoverTraffic":
    return CoverTraffic(env, comm, tp=args.cover_tp, hidden=args.decode_hidden, layers=args.decode_layers,
                        ars_per_layer=args.ars_per_layer, tokens=args.decode_tokens, dtype=args.decode_dtype,
                        steps_per_call=args.cover_tokens, log_mode=args.log_decode)


def duty_fractions(sync_every: int, sync_wire_bytes: int, cover_wire_bytes_per_step: int) -> dict:
    """The 'how diluted' numbers for the meta sidecar (W24: >=95% infer-shaped is a checkable claim).

    ``train_step_fraction`` = share of steps that carry a gradient sync (window-label truth);
    ``expected_train_wire_byte_fraction`` = share of wire bytes that are sync, not cover.
    Local forward/backward compute happens EVERY step -- the dilution is on the wire, so an NVML
    layer still sees training compute.
    """
    h = max(1, sync_every)
    total = sync_wire_bytes + h * cover_wire_bytes_per_step
    return {"sync_period_steps": h, "train_step_fraction": 1.0 / h, "infer_shaped_step_fraction": 1.0 - 1.0 / h,
            "expected_train_wire_byte_fraction": (sync_wire_bytes / total) if total else None}


# ── Flat buffers / chunked collectives ───────────────────────────────────────

def flatten_into(tensors: Sequence[torch.Tensor], out: torch.Tensor) -> None:
    off = 0
    for t in tensors:
        n = t.numel()
        out[off:off + n].copy_(t.reshape(-1))
        off += n


def unflatten_from(flat: torch.Tensor, tensors: Sequence[torch.Tensor]) -> None:
    off = 0
    for t in tensors:
        n = t.numel()
        t.copy_(flat[off:off + n].view_as(t))
        off += n


def ensure_grads(params: Sequence[nn.Parameter]) -> list[torch.Tensor]:
    """``p.grad`` for every param, materialising zeros for any that got no gradient."""
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    return [p.grad for p in params]


def chunked_allreduce(flat: torch.Tensor, block_bytes: int, env: DistEnv, *,
                      group=None, log: JsonlLog | None = None, phase: str = "grad_sync",
                      group_name: str = "world", gap_s: float = 0.0, average: bool = True,
                      sizes: list[int] | None = None, gaps: list[float] | None = None,
                      **extra) -> int:
    """AllReduce ``flat`` in blocks of ``block_bytes``. Returns number of blocks.

    Synchronous per block (device sync after each) so ``dur_s`` in the log is real
    and ``gap_s`` pacing produces a genuine burst/idle duty cycle. ``sizes`` (elements per
    block, summing to ``flat.numel()``) and ``gaps`` (seconds slept before each block)
    override the uniform ``block_bytes`` / ``gap_s`` schedule -- used by W18's
    ``--schedule-from`` to follow an empirical KV-handoff profile.
    """
    n = dist.get_world_size(group)
    block_elems = max(1, block_bytes // flat.element_size())
    if sizes is None:
        sizes = split_blocks(flat.numel(), block_elems)
    off = 0
    for i, sz in enumerate(sizes):
        if gaps is not None and gaps[i] > 0:
            time.sleep(gaps[i])
        blk = flat[off:off + sz]
        t0 = time.time()
        dist.all_reduce(blk, group=group)
        env.sync()
        payload = tensor_bytes(blk)
        if log is not None:
            log.emit(t=t0, dur_s=time.time() - t0, op="all_reduce", phase=phase,
                     group=group_name, group_size=n, block=i, n_blocks=len(sizes),
                     bytes_est=ar_wire_bytes(payload, n), payload_bytes=payload, **extra)
        off += sz
        if gaps is None and gap_s and i + 1 < len(sizes):
            time.sleep(gap_s)
    if average:
        flat.div_(n)
    return len(sizes)


# ── KV-cache geometry (W10 handoff size == W18 block size) ───────────────────
# Pure-python module so the serving orchestrators need no torch; re-exported here.
from kv_geometry import KV_PRESETS, kv_bytes_per_token, kv_handoff_bytes, split_blocks  # noqa: E402,F401


# ── Sized decoder LM (same family as fsdp_training / diloco_production) ──────

# (hidden, layers, heads, ffn, vocab). "tiny" exists for CPU/gloo smoke tests only.
MODEL_CONFIGS = {
    "tiny": (128, 4, 4, 512, 2048),
    "150m": (768, 12, 12, 3072, 32000),
    "1b":   (2048, 24, 16, 8192, 32000),   # ~1.3B
    "7b":   (4096, 32, 32, 11008, 32000),
}


class Block(nn.Module):
    """Pre-norm decoder block with fused QKV and causal SDPA."""

    def __init__(self, hidden: int, heads: int, ffn: int) -> None:
        super().__init__()
        self.heads = heads
        self.norm1 = nn.LayerNorm(hidden)
        self.qkv = nn.Linear(hidden, 3 * hidden, bias=False)
        self.proj = nn.Linear(hidden, hidden, bias=False)
        self.norm2 = nn.LayerNorm(hidden)
        self.up = nn.Linear(hidden, ffn, bias=False)
        self.down = nn.Linear(ffn, hidden, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, t, h = x.shape
        q, k, v = self.qkv(self.norm1(x)).view(b, t, 3, self.heads, h // self.heads).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self.proj(a.transpose(1, 2).reshape(b, t, h))
        return x + self.down(F.gelu(self.up(self.norm2(x))))


class CausalLM(nn.Module):
    """Decoder LM. Blocks are named ``blocks.N`` so name-based fragmenting works."""

    def __init__(self, size: str, max_pos: int = 4096) -> None:
        super().__init__()
        hidden, n_layers, heads, ffn, vocab = MODEL_CONFIGS[size]
        self.vocab_size = vocab
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(max_pos, hidden)
        self.blocks = nn.ModuleList([Block(hidden, heads, ffn) for _ in range(n_layers)])
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)
        self.apply(init_weights)

    def forward(self, input_ids: torch.Tensor) -> torch.Tensor:
        pos = torch.arange(input_ids.size(1), device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))


def init_weights(m: nn.Module) -> None:
    if isinstance(m, (nn.Linear, nn.Embedding)):
        nn.init.normal_(m.weight, mean=0.0, std=0.02)


def lm_loss(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits[:, :-1].reshape(-1, logits.size(-1)).float(),
                           input_ids[:, 1:].reshape(-1))


def count_params(m: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in m.parameters() if p.requires_grad or not trainable_only)


def synthetic_tokens(batch: int, seq: int, vocab: int, device: torch.device) -> torch.Tensor:
    return torch.randint(0, vocab, (batch, seq), device=device)


def autocast_ctx(env: DistEnv, enabled: bool = True):
    """bf16 autocast on GPU, no-op on CPU (keeps gloo smoke tests exact)."""
    on = enabled and env.device.type == "cuda"
    return torch.autocast(device_type=env.device.type, dtype=torch.bfloat16, enabled=on)


def make_optimizer(name: str, params, lr: float = 1e-4):
    params = [p for p in params if p.requires_grad]
    if name == "adamw":
        return torch.optim.AdamW(params, lr=lr)
    if name == "sgd_momentum":
        return torch.optim.SGD(params, lr=lr, momentum=0.9)
    return torch.optim.SGD(params, lr=lr)


BusyFn = Callable[[], None]


def make_gemm_busywork(env: DistEnv, hidden: int, rows: int, dtype: torch.dtype) -> BusyFn:
    """One decode-like MLP step: (rows,h) @ (h,4h) -> gelu -> @ (4h,h). Keeps the GPU busy."""
    w1 = torch.randn(hidden, 4 * hidden, device=env.device, dtype=dtype) * 0.02
    w2 = torch.randn(4 * hidden, hidden, device=env.device, dtype=dtype) * 0.02
    x = torch.randn(rows, hidden, device=env.device, dtype=dtype)

    def step() -> None:
        nonlocal x
        x = F.gelu(x @ w1) @ w2
        x = x / (x.norm(dim=-1, keepdim=True) + 1e-3)   # keep values bounded
        if env.device.type == "cuda":
            torch.cuda.current_stream().synchronize()

    return step


def dtype_from_name(name: str) -> torch.dtype:
    return {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[name]
