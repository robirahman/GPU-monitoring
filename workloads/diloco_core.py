#!/usr/bin/env python3
"""Production DiLoCo primitives (OpenDiLoCo + Streaming DiLoCo).

Algorithm (Douillard et al. 2311.08105 / OpenDiLoCo 2407.07852):
  inner: AdamW for H local steps (no cross-replica traffic)
  outer: Δ = θ_shadow − θ_local; AllReduceMean(Δ); Nesterov-SGD on shadow; copy → local

Streaming (2501.18512 / torchft):
  partition params into F fragments; stagger sync; async AllReduce overlapped with compute.

This is NCCL / torch.distributed — not the ICTD per-parameter Python toy.
"""
from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from typing import Callable, Iterable, Literal, Sequence

import torch
import torch.distributed as dist
from torch import nn, optim
from torch.distributed.distributed_c10d import Work

ReduceDType = Literal["fp32", "fp16", "bf16", "fp8"]


def _resolve_dtype(name: ReduceDType) -> torch.dtype:
    return {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp8": getattr(torch, "float8_e4m3fn", torch.float16),
    }[name]


def _param_bytes(params: Iterable[torch.Tensor]) -> int:
    return sum(p.numel() * p.element_size() for p in params)


@dataclass
class CommEvent:
    """One collective for interconnect STEP logs / detectors."""

    t: float
    op: str
    phase: str  # inner | outer_sync | fragment_prepare | fragment_wait
    bytes_est: int
    reduce_dtype: str
    fragment: int | None = None
    step: int = 0
    async_op: bool = False
    world_size: int = 1

    def to_json(self) -> str:
        return json.dumps(self.__dict__, separators=(",", ":"))


@dataclass
class CommLogger:
    path: str | None = None
    rank: int = 0
    events: list[CommEvent] = field(default_factory=list)
    _fh: object | None = field(default=None, repr=False)

    def open(self) -> None:
        if self.path and self.rank == 0:
            os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
            self._fh = open(self.path, "a", buffering=1)

    def emit(self, ev: CommEvent) -> None:
        self.events.append(ev)
        if self._fh is not None:
            self._fh.write(ev.to_json() + "\n")

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None


def partition_named_params(
    model: nn.Module,
    num_fragments: int,
) -> list[list[tuple[str, nn.Parameter]]]:
    """Split trainable params into F contiguous fragments (by name order).

    Prefer splitting on transformer block boundaries when names contain `.layers.`
    or `.blocks.` — otherwise even numel chunks.
    """
    named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    if num_fragments <= 1:
        return [named]

    block_keys = [".layers.", ".blocks.", ".h.", ".layer."]
    groups: dict[str, list[tuple[str, nn.Parameter]]] = {}
    order: list[str] = []
    for n, p in named:
        key = None
        for bk in block_keys:
            if bk in n:
                # e.g. model.layers.12.mlp.up_proj.weight → model.layers.12
                parts = n.split(bk, 1)
                head = parts[0] + bk + parts[1].split(".", 1)[0]
                key = head
                break
        if key is None:
            key = "__other__"
        if key not in groups:
            groups[key] = []
            order.append(key)
        groups[key].append((n, p))

    if len(order) >= num_fragments:
        # round-robin block-groups into F fragments (keeps layer locality)
        chunks: list[list[tuple[str, nn.Parameter]]] = [[] for _ in range(num_fragments)]
        for i, k in enumerate(order):
            chunks[i % num_fragments].extend(groups[k])
        chunks = [c for c in chunks if c]
        while len(chunks) < num_fragments and chunks:
            i = max(range(len(chunks)), key=lambda j: sum(p.numel() for _, p in chunks[j]))
            mid = len(chunks[i]) // 2
            if mid == 0:
                break
            chunks.append(chunks[i][mid:])
            chunks[i] = chunks[i][:mid]
        return chunks[:num_fragments] if chunks else [named]

    # fallback: even by parameter count
    total = sum(p.numel() for _, p in named)
    target = math.ceil(total / num_fragments)
    frags: list[list[tuple[str, nn.Parameter]]] = [[]]
    running = 0
    for n, p in named:
        if running >= target and len(frags) < num_fragments:
            frags.append([])
            running = 0
        frags[-1].append((n, p))
        running += p.numel()
    return frags


class _BucketAllReduce:
    """Coalesced flatten → AllReduceMean → unpack. Optional low-precision wire dtype."""

    def __init__(
        self,
        reduce_dtype: ReduceDType = "fp16",
        bucket_cap_mb: float = 256.0,
        logger: CommLogger | None = None,
    ) -> None:
        self.reduce_dtype_name = reduce_dtype
        self.reduce_dtype = _resolve_dtype(reduce_dtype)
        self.bucket_cap_bytes = int(bucket_cap_mb * 1024 * 1024)
        self.logger = logger

    def _cast_for_wire(self, flat: torch.Tensor) -> tuple[torch.Tensor, torch.dtype]:
        """Return (wire_tensor, restore_dtype). FP8 falls back to fp16 if NCCL rejects it."""
        if self.reduce_dtype_name == "fp32":
            return flat.float(), torch.float32
        if self.reduce_dtype_name in ("fp16", "bf16"):
            return flat.to(self.reduce_dtype), self.reduce_dtype
        # fp8: try native; else fp16 wire (still record intent in logger)
        try:
            wire = flat.to(torch.float8_e4m3fn)
            # smoke: some builds can't all_reduce fp8 — probe with empty async on clone
            return wire, torch.float8_e4m3fn
        except (TypeError, RuntimeError, AttributeError):
            return flat.to(torch.float16), torch.float16

    @torch.no_grad()
    def allreduce_mean_async(
        self,
        tensors: Sequence[torch.Tensor],
        *,
        phase: str,
        step: int,
        fragment: int | None = None,
    ) -> tuple[list[Work], Callable[[], None]]:
        """Start bucketed async all-reduces. Returns (works, finalize_fn).

        finalize_fn waits, divides by world_size, copies back into `tensors` (in-place).
        """
        if not tensors:
            return [], lambda: None
        if not dist.is_initialized() or dist.get_world_size() == 1:
            return [], lambda: None

        world = dist.get_world_size()
        works: list[Work] = []
        pending: list[tuple[torch.Tensor, list[tuple[torch.Tensor, int, int]], torch.dtype]] = []

        # pack into buckets
        i = 0
        n = len(tensors)
        while i < n:
            bucket: list[torch.Tensor] = []
            nbytes = 0
            while i < n:
                t = tensors[i]
                need = t.numel() * max(t.element_size(), 2)  # wire at least 2B/elem for fp16
                if bucket and nbytes + need > self.bucket_cap_bytes:
                    break
                bucket.append(t)
                nbytes += need
                i += 1

            flat = torch.empty(sum(t.numel() for t in bucket), device=bucket[0].device, dtype=torch.float32)
            off = 0
            meta: list[tuple[torch.Tensor, int, int]] = []
            for t in bucket:
                numel = t.numel()
                flat[off : off + numel].copy_(t.detach().float().reshape(-1))
                meta.append((t, off, numel))
                off += numel

            wire, wire_dtype = self._cast_for_wire(flat)
            # If fp8 path produced float8, NCCL may fail — fall back once
            try:
                work = dist.all_reduce(wire, op=dist.ReduceOp.SUM, async_op=True)
            except RuntimeError:
                wire = flat.to(torch.float16)
                wire_dtype = torch.float16
                work = dist.all_reduce(wire, op=dist.ReduceOp.SUM, async_op=True)

            works.append(work)
            pending.append((wire, meta, wire_dtype))

            if self.logger is not None:
                # ring all-reduce ≈ 2*(N-1)/N * size ≈ 2 * size for large N
                payload = wire.numel() * wire.element_size()
                self.logger.emit(
                    CommEvent(
                        t=time.time(),
                        op="AllReduceMean",
                        phase=phase,
                        bytes_est=int(payload * 2 * (world - 1) / world) if world > 1 else 0,
                        reduce_dtype=self.reduce_dtype_name,
                        fragment=fragment,
                        step=step,
                        async_op=True,
                        world_size=world,
                    )
                )

        def finalize() -> None:
            for w in works:
                w.wait()
            for wire, meta, _wd in pending:
                wire.div_(world)
                # dequant / cast back
                host = wire.float()
                for t, off, numel in meta:
                    t.copy_(host[off : off + numel].view_as(t))

        return works, finalize


class DiLoCoState:
    """Shadow (outer) params + optional per-fragment outer optimizers."""

    def __init__(
        self,
        named_params: Sequence[tuple[str, nn.Parameter]],
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        backup_device: torch.device | None = None,
    ) -> None:
        self.backup_device = backup_device or torch.device("cpu")
        self.shadow: dict[str, torch.Tensor] = {}
        self._params = list(named_params)
        for name, p in self._params:
            t = torch.empty_like(p.data, device=self.backup_device)
            if self.backup_device.type == "cpu" and torch.cuda.is_available():
                t = t.pin_memory()
            t.copy_(p.data, non_blocking=False)
            self.shadow[name] = t

        # Outer opt over a ParameterList that mirrors shadow on CUDA for SGD.step
        self._outer_params = nn.ParameterList(
            [nn.Parameter(self.shadow[n].to(p.device).clone(), requires_grad=True) for n, p in self._params]
        )
        self._name_to_outer = {n: self._outer_params[i] for i, (n, _) in enumerate(self._params)}
        for n, op in self._name_to_outer.items():
            op.data.copy_(self.shadow[n].to(op.device))

        self.outer_opt = optim.SGD(
            self._outer_params.parameters(),
            lr=outer_lr,
            momentum=outer_momentum,
            nesterov=True,
        )

    @torch.no_grad()
    def sync_outer_from_shadow(self) -> None:
        for n, op in self._name_to_outer.items():
            op.data.copy_(self.shadow[n].to(op.device, non_blocking=True))

    @torch.no_grad()
    def save_shadow_from_outer(self) -> None:
        for n, op in self._name_to_outer.items():
            self.shadow[n].copy_(op.data, non_blocking=False)

    @torch.no_grad()
    def copy_shadow_to_model(self) -> None:
        for n, p in self._params:
            p.data.copy_(self.shadow[n].to(p.device, non_blocking=False))


class DiLoCoController:
    """Full-sync or streaming DiLoCo over an NCCL process group."""

    def __init__(
        self,
        model: nn.Module,
        inner_opt: optim.Optimizer,
        *,
        local_steps: int = 500,
        outer_lr: float = 0.7,
        outer_momentum: float = 0.9,
        reduce_dtype: ReduceDType = "fp16",
        bucket_cap_mb: float = 256.0,
        streaming: bool = False,
        num_fragments: int = 4,
        fragment_sync_delay: int = 0,
        backup_device: str = "cpu",
        logger: CommLogger | None = None,
    ) -> None:
        if local_steps < 1:
            raise ValueError("local_steps must be >= 1")
        self.model = model
        self.inner_opt = inner_opt
        self.local_steps = local_steps
        self.streaming = streaming
        self.fragment_sync_delay = fragment_sync_delay
        self.logger = logger or CommLogger()
        self._step_in_period = 0
        self.global_inner_step = 0

        named = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
        self.fragments = partition_named_params(model, num_fragments if streaming else 1)
        if streaming:
            if local_steps < len(self.fragments):
                raise ValueError("local_steps must be >= num_fragments in streaming mode")
            if local_steps % len(self.fragments) != 0:
                raise ValueError("local_steps must be divisible by num_fragments (streaming)")
            self._period = local_steps // len(self.fragments)
            if fragment_sync_delay >= self._period:
                raise ValueError("fragment_sync_delay must be < local_steps/num_fragments")
        else:
            self._period = local_steps

        bdev = torch.device(backup_device)
        self.states = [
            DiLoCoState(frag, outer_lr, outer_momentum, backup_device=bdev) for frag in self.fragments
        ]
        self.bucket = _BucketAllReduce(reduce_dtype, bucket_cap_mb, self.logger)

        # pending async outer for streaming
        self._pending_finalize: Callable[[], None] | None = None
        self._pending_fragment: int | None = None
        self._pending_pseudos: list[torch.Tensor] | None = None
        self._periods_done = 0

        self.comm_stream = (
            torch.cuda.Stream() if torch.cuda.is_available() else None
        )

    @property
    def num_fragments(self) -> int:
        return len(self.fragments)

    def _current_fragment(self) -> int:
        return self._periods_done % self.num_fragments

    @torch.no_grad()
    def _build_pseudo_grads(self, frag_idx: int) -> list[torch.Tensor]:
        state = self.states[frag_idx]
        outs: list[torch.Tensor] = []
        for name, p in self.fragments[frag_idx]:
            shadow = state.shadow[name].to(p.device, non_blocking=True)
            # Δ = θ_shadow − θ_local  (OpenDiLoCo / torchft)
            outs.append(shadow - p.data)
        return outs

    def _apply_outer(self, frag_idx: int, averaged_delta: list[torch.Tensor]) -> None:
        state = self.states[frag_idx]
        state.sync_outer_from_shadow()
        state.outer_opt.zero_grad(set_to_none=True)
        for (name, p), delta, op in zip(
            self.fragments[frag_idx], averaged_delta, state._outer_params
        ):
            # outer step on shadow: treat averaged Δ as gradient
            op.grad = delta.to(op.device)
        state.outer_opt.step()
        state.save_shadow_from_outer()
        # write updated shadow into model
        for name, p in self.fragments[frag_idx]:
            p.data.copy_(state.shadow[name].to(p.device, non_blocking=False))

    def _start_fragment_allreduce(self, frag_idx: int) -> None:
        if self._pending_finalize is not None:
            raise RuntimeError("previous fragment allreduce still pending")
        pseudos = self._build_pseudo_grads(frag_idx)
        # clone buffers that allreduce will write into
        bufs = [d.clone() for d in pseudos]
        if self.comm_stream is not None:
            self.comm_stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(self.comm_stream):
                _works, finalize = self.bucket.allreduce_mean_async(
                    bufs,
                    phase="fragment_prepare",
                    step=self.global_inner_step,
                    fragment=frag_idx,
                )
        else:
            _works, finalize = self.bucket.allreduce_mean_async(
                bufs,
                phase="fragment_prepare",
                step=self.global_inner_step,
                fragment=frag_idx,
            )
        self._pending_finalize = finalize
        self._pending_fragment = frag_idx
        self._pending_pseudos = bufs

    def _finish_fragment_allreduce(self) -> None:
        if self._pending_finalize is None:
            return
        if self.comm_stream is not None:
            torch.cuda.current_stream().wait_stream(self.comm_stream)
        self._pending_finalize()
        assert self._pending_fragment is not None and self._pending_pseudos is not None
        self.logger.emit(
            CommEvent(
                t=time.time(),
                op="OuterStep",
                phase="fragment_wait",
                bytes_est=0,
                reduce_dtype=self.bucket.reduce_dtype_name,
                fragment=self._pending_fragment,
                step=self.global_inner_step,
                world_size=dist.get_world_size() if dist.is_initialized() else 1,
            )
        )
        self._apply_outer(self._pending_fragment, self._pending_pseudos)
        self._pending_finalize = None
        self._pending_fragment = None
        self._pending_pseudos = None

    @torch.no_grad()
    def _full_outer_sync(self) -> None:
        """Classic DiLoCo: sync all fragments (usually one) synchronously."""
        for fi in range(self.num_fragments):
            bufs = [d.clone() for d in self._build_pseudo_grads(fi)]
            _works, finalize = self.bucket.allreduce_mean_async(
                bufs,
                phase="outer_sync",
                step=self.global_inner_step,
                fragment=fi if self.num_fragments > 1 else None,
            )
            finalize()
            self._apply_outer(fi, bufs)

    def after_inner_step(self) -> str:
        """Call once after every inner optimizer.step().

        Returns current phase label for STEP markers: 'inner' | 'outer_sync' | ...
        """
        self.global_inner_step += 1
        self._step_in_period += 1
        phase = "inner"

        if not self.streaming:
            if self._step_in_period >= self.local_steps:
                self._full_outer_sync()
                self._step_in_period = 0
                phase = "outer_sync"
            return phase

        # Streaming schedule (torchft-style, NCCL world):
        # at (period - delay): kick async allreduce for current fragment
        # at period: wait + outer step
        delay = self.fragment_sync_delay
        fi = self._current_fragment()
        if delay > 0 and self._step_in_period == self._period - delay:
            self._start_fragment_allreduce(fi)
            phase = "fragment_prepare"
        if self._step_in_period >= self._period:
            if delay == 0:
                self._start_fragment_allreduce(fi)
            self._finish_fragment_allreduce()
            self._step_in_period = 0
            self._periods_done += 1
            phase = "outer_sync"
        return phase

    def force_sync(self) -> None:
        """Barrier outer sync (e.g. end of run)."""
        if self._pending_finalize is not None:
            self._finish_fragment_allreduce()
        self._full_outer_sync()
        self._step_in_period = 0
