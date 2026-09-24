#!/usr/bin/env python3
"""W04 -- Pipeline-parallel 1F1B training (P2P activation traffic, no DP AllReduce).

**SCHEDULE MICROBENCH -- not SPAR primary corpus.** Hand-rolled 1F1B (numerically verified against a
reference; good schedule unit test). Primary W04: real torchtitan pipeline
(``train_pp4_1f1b_qwen3_1p7b_titan``, torch.distributed.pipelining).

Training whose fabric fingerprint is *point-to-point activations and activation
gradients between adjacent stages* -- no gradient AllReduce at all. The shape is
neither "big periodic AllReduce" (DDP) nor "AllReduce per layer" (TP): world_size = PP
stages, each rank holds a contiguous slice of a causal LM, and per step ``M``
microbatches flow through a one-forward-one-backward (1F1B) schedule::

    stage s:  warmup  min(P-s-1, M) forwards
              steady  (M - warmup) x [forward, backward]
              cooldown warmup backwards

Hand-rolled on ``dist.batch_isend_irecv`` (Megatron's non-interleaved structure, incl.
the combined send_forward_recv_backward / send_backward_recv_forward exchanges that
keep it deadlock-free) rather than ``torch.distributed.pipelining``: identical wire
behaviour, but the same code runs on gloo for CPU checks and the schedule phases can be
logged exactly. Synthetic tokens are generated deterministically per (step, microbatch)
on every stage, so the last stage has its labels without extra traffic.

Per step, per stage boundary: M forward activations (mb x T x hidden) one way and M
activation-gradients the other. Nothing else crosses the fabric.

Comm log (JSONL). Rank0 = stage 0 only; pass ``--comm-log-all-ranks`` for the whole
pipeline (``<stem>.rank<N>.jsonl``)::

    op=p2p_send|p2p_recv  phase=pp_fwd_act|pp_bwd_grad  microbatch  stage  peer
        sched=warmup|steady|cooldown  bytes_est payload_bytes  step  group=pp
    op=pp_phase           phase=pp_warmup|pp_steady|pp_cooldown  t t_end dur_s
        blocked_s (time stalled in P2P waits)  -- the bubble markers
    op=pp_step            phase=pp_bubble  bubble_frac_ideal=(P-1)/(M+P-1)
        blocked_s step_s measured_bubble_frac

``--verify`` (step 0) recomputes the whole model in-process on identical weights and
checks each stage's accumulated gradients against it -- validates the schedule/P2P math.

Launch::

    torchrun --nproc_per_node=2 workloads/pp_1f1b_train.py --size 150m --microbatches 8 \\
        --micro-batch-size 4 --seq-len 512 --comm-log data/pp2_1f1b_comm.jsonl --comm-log-all-ranks
    torchrun --nproc_per_node=4 workloads/pp_1f1b_train.py --size 1b --microbatches 16 ...
    python scripts/run_workload_multi_gpu.py --workload train_pp4_1f1b_1b --num-gpus 4

CPU smoke (includes the numerical check)::

    torchrun --nproc_per_node=4 workloads/pp_1f1b_train.py --backend gloo --size tiny \\
        --microbatches 6 --micro-batch-size 2 --seq-len 16 --max-steps 3 --verify \\
        --comm-log /tmp/pp.jsonl --comm-log-all-ranks
"""
from __future__ import annotations

import argparse
import time
from collections import deque

import torch
import torch.distributed as dist
import torch.nn as nn

from interconnect_common import (
    Block,
    JsonlLog,
    MODEL_CONFIGS,
    StopSignal,
    add_common_args,
    dist_cleanup,
    dist_setup,
    dtype_from_name,
    init_weights,
    lm_loss,
    make_optimizer,
    tensor_bytes,
    write_meta,
)


class Stage(nn.Module):
    """Layers [lo, hi) of the LM, plus embeddings (first) / final norm + head (last).

    Weights are seeded per *global* layer index, so a single-stage instance covering all
    layers reproduces exactly the concatenation of the pipeline stages (used by --verify).
    """

    def __init__(self, size: str, lo: int, hi: int, first: bool, last: bool, seed: int,
                 max_pos: int = 4096) -> None:
        super().__init__()
        hidden, _, heads, ffn, vocab = MODEL_CONFIGS[size]
        self.first, self.last = first, last
        if first:
            torch.manual_seed(seed)
            self.tok_emb = nn.Embedding(vocab, hidden)
            self.pos_emb = nn.Embedding(max_pos, hidden)
            self.tok_emb.apply(init_weights)
            self.pos_emb.apply(init_weights)
        self.blocks = nn.ModuleList()
        for g in range(lo, hi):
            torch.manual_seed(seed + 1 + g)
            blk = Block(hidden, heads, ffn)
            blk.apply(init_weights)
            self.blocks.append(blk)
        self.lo = lo
        if last:
            torch.manual_seed(seed + 100_000)
            self.norm = nn.LayerNorm(hidden)
            self.lm_head = nn.Linear(hidden, vocab, bias=False)
            self.lm_head.apply(init_weights)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.first:
            pos = torch.arange(x.size(1), device=x.device).unsqueeze(0)
            x = self.tok_emb(x) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        if self.last:
            x = self.lm_head(self.norm(x))
        return x

    def global_names(self) -> dict[str, nn.Parameter]:
        """Parameter dict keyed by the *full-model* name (blocks.<global layer>...)."""
        out = {}
        for n, p in self.named_parameters():
            if n.startswith("blocks."):
                _, idx, rest = n.split(".", 2)
                n = f"blocks.{self.lo + int(idx)}.{rest}"
            out[n] = p
        return out


def stage_bounds(n_layers: int, stages: int) -> list[tuple[int, int]]:
    base, rem = divmod(n_layers, stages)
    out, lo = [], 0
    for i in range(stages):
        hi = lo + base + (1 if i < rem else 0)
        out.append((lo, hi))
        lo = hi
    return out


class Pipeline:
    def __init__(self, env, stage: Stage, args, comm: JsonlLog, act_shape, dtype) -> None:
        self.env, self.stage, self.args, self.comm = env, stage, args, comm
        self.P, self.s, self.M = env.world, env.rank, args.microbatches
        self.first, self.last = self.s == 0, self.s == self.P - 1
        self.act_shape, self.dtype = act_shape, dtype
        self.vocab = MODEL_CONFIGS[args.size][4]
        self.sched = "warmup"
        self.blocked = 0.0
        self.step = 0

    # ── P2P plumbing ─────────────────────────────────────────────────────────
    def _exchange(self, sends, recvs) -> None:
        """One batched isend/irecv group; logs every op and accumulates stall time."""
        ops = ([dist.P2POp(dist.isend, t, peer) for t, peer, _, _ in sends]
               + [dist.P2POp(dist.irecv, t, peer) for t, peer, _, _ in recvs])
        if not ops:
            return
        self.env.sync()
        t0 = time.time()
        for r in dist.batch_isend_irecv(ops):
            r.wait()
        self.env.sync()
        dur = time.time() - t0
        self.blocked += dur
        for kind, items in (("p2p_send", sends), ("p2p_recv", recvs)):
            for t, peer, phase, mb in items:
                nbytes = tensor_bytes(t)
                self.comm.emit(t=t0, dur_s=dur, op=kind, phase=phase, microbatch=mb, stage=self.s,
                               peer=peer, sched=self.sched, group="pp", step=self.step,
                               bytes_est=nbytes, payload_bytes=nbytes)

    def _buf(self) -> torch.Tensor:
        return torch.empty(self.act_shape, device=self.env.device, dtype=self.dtype)

    def recv_forward(self, mb):
        if self.first:
            return None
        x = self._buf()
        self._exchange([], [(x, self.s - 1, "pp_fwd_act", mb)])
        return x.requires_grad_(True)

    def send_forward(self, out, mb) -> None:
        if not self.last:
            self._exchange([(out.detach().contiguous(), self.s + 1, "pp_fwd_act", mb)], [])

    def recv_backward(self, mb):
        if self.last:
            return None
        g = self._buf()
        self._exchange([], [(g, self.s + 1, "pp_bwd_grad", mb)])
        return g

    def send_backward(self, g, mb) -> None:
        if not self.first:
            self._exchange([(g.contiguous(), self.s - 1, "pp_bwd_grad", mb)], [])

    def send_forward_recv_backward(self, out, mb_f, mb_b):
        if self.last:
            return None
        g = self._buf()
        self._exchange([(out.detach().contiguous(), self.s + 1, "pp_fwd_act", mb_f)],
                       [(g, self.s + 1, "pp_bwd_grad", mb_b)])
        return g

    def send_backward_recv_forward(self, g, mb_b, mb_f):
        if self.first:
            return None
        x = self._buf()
        self._exchange([(g.contiguous(), self.s - 1, "pp_bwd_grad", mb_b)],
                       [(x, self.s - 1, "pp_fwd_act", mb_f)])
        return x.requires_grad_(True)

    # ── data ─────────────────────────────────────────────────────────────────
    def make_ids(self, step: int, mb: int) -> torch.Tensor:
        g = torch.Generator(device=self.env.device)
        g.manual_seed(self.args.seed * 1_000_003 + step * 1009 + mb)
        return torch.randint(0, self.vocab, (self.args.micro_batch_size, self.args.seq_len),
                             device=self.env.device, generator=g)

    # ── one 1F1B step ────────────────────────────────────────────────────────
    def train_step(self, step: int) -> float:
        self.step = step
        self.comm.step = step
        M, P, s = self.M, self.P, self.s
        warm = min(P - s - 1, M)
        remaining = M - warm
        pending: deque = deque()
        loss_sum = torch.zeros((), device=self.env.device)
        fwd_i = bwd_i = 0
        t_step = time.time()
        self.blocked = 0.0

        def forward(x_in, i):
            ids = self.make_ids(step, i)
            out = self.stage(ids if self.first else x_in)
            if self.last:
                loss = lm_loss(out, ids) / M
                out = loss
            pending.append((x_in, out))
            return out

        def backward(g_out):
            x_in, out = pending.popleft()
            if self.last:
                out.backward()
                loss_sum.add_(out.detach())
            else:
                torch.autograd.backward(out, g_out)
            return None if self.first else x_in.grad

        def phase_begin(name):
            self.sched = name
            self.blocked_at = self.blocked
            return time.time()

        def phase_end(name, t0, n_fwd, n_bwd):
            self.env.sync()
            self.comm.emit(t=t0, t_end=time.time(), dur_s=time.time() - t0, op="pp_phase",
                           phase=f"pp_{name}", stage=s, step=step, n_fwd=n_fwd, n_bwd=n_bwd,
                           blocked_s=self.blocked - self.blocked_at)

        # warmup: forwards only
        t0 = phase_begin("warmup")
        for _ in range(warm):
            x_in = self.recv_forward(fwd_i)
            out = forward(x_in, fwd_i)
            self.send_forward(out, fwd_i)
            fwd_i += 1
        phase_end("warmup", t0, warm, 0)

        # steady: one forward, one backward
        t0 = phase_begin("steady")
        x_in = self.recv_forward(fwd_i) if remaining > 0 else None
        for k in range(remaining):
            last_iter = k == remaining - 1
            mb_f = fwd_i
            out = forward(x_in, mb_f)
            fwd_i += 1
            g_out = self.send_forward_recv_backward(out, mb_f, bwd_i)
            mb_b = bwd_i
            g_in = backward(g_out)
            bwd_i += 1
            if last_iter:
                self.send_backward(g_in, mb_b)
            else:
                x_in = self.send_backward_recv_forward(g_in, mb_b, fwd_i)
        phase_end("steady", t0, remaining, remaining)

        # cooldown: remaining backwards
        t0 = phase_begin("cooldown")
        for _ in range(warm):
            g_out = self.recv_backward(bwd_i)
            g_in = backward(g_out)
            self.send_backward(g_in, bwd_i)
            bwd_i += 1
        phase_end("cooldown", t0, 0, warm)

        step_s = time.time() - t_step
        self.comm.emit(t=t_step, t_end=time.time(), dur_s=step_s, op="pp_step", phase="pp_bubble",
                       stage=s, step=step, microbatches=M, stages=P,
                       bubble_frac_ideal=(P - 1) / (M + P - 1), blocked_s=self.blocked,
                       measured_bubble_frac=self.blocked / max(step_s, 1e-9))
        return float(loss_sum) if self.last else float("nan")


def verify_against_reference(pipe: Pipeline, stage: Stage, args, step: int) -> None:
    """Whole model in-process, same weights, same microbatches -> compare stage grads."""
    n_layers = MODEL_CONFIGS[args.size][1]
    ref = Stage(args.size, 0, n_layers, True, True, args.seed).to(pipe.env.device, pipe.dtype)
    for i in range(args.microbatches):
        ids = pipe.make_ids(step, i)
        (lm_loss(ref(ids), ids) / args.microbatches).backward()
    ref_grads = {n: p.grad for n, p in ref.named_parameters()}
    worst = 0.0
    for n, p in stage.global_names().items():
        assert p.grad is not None, f"{n}: no grad on stage {pipe.s}"
        denom = ref_grads[n].abs().max().clamp_min(1e-12)
        worst = max(worst, float((p.grad - ref_grads[n]).abs().max() / denom))
    print(f"  [verify] stage {pipe.s}: max relative grad error vs single-process reference = {worst:.2e}")
    tol = 1e-3 if pipe.dtype == torch.float32 else 5e-2      # bf16/fp16 accumulate differently
    if worst > tol:
        raise SystemExit(f"verify FAILED on stage {pipe.s}: {worst} > {tol}")


def main() -> None:
    ap = argparse.ArgumentParser(description="W04 pipeline-parallel 1F1B training",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--size", choices=list(MODEL_CONFIGS), default="150m")
    ap.add_argument("--microbatches", "-M", type=int, default=8, dest="microbatches")
    ap.add_argument("--micro-batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--optimizer", choices=["adamw", "sgd", "sgd_momentum"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--verify", action="store_true",
                    help="Step 0: compare pipelined grads with a single-process reference (small models)")
    args = ap.parse_args()

    env = dist_setup(args.backend)
    P = env.world
    hidden, n_layers = MODEL_CONFIGS[args.size][:2]
    if P > n_layers:
        raise SystemExit(f"{P} stages but the {args.size} model has only {n_layers} layers")
    dtype = dtype_from_name(args.dtype) if env.device.type == "cuda" else torch.float32
    bounds = stage_bounds(n_layers, P)
    lo, hi = bounds[env.rank]

    stage = Stage(args.size, lo, hi, env.rank == 0, env.rank == P - 1, args.seed).to(env.device, dtype)
    opt = make_optimizer(args.optimizer, stage.parameters(), args.lr)
    act_shape = (args.micro_batch_size, args.seq_len, hidden)
    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    pipe = Pipeline(env, stage, args, comm, act_shape, dtype)

    act_bytes = args.micro_batch_size * args.seq_len * hidden * torch.empty(0, dtype=dtype).element_size()
    n_params = sum(p.numel() for p in stage.parameters())
    write_meta(args.comm_log, {
        "workload": "pp_1f1b_train", "wid": "W04", "world_size": P, "backend": env.backend,
        "model": args.size, "stage_layer_ranges": bounds, "microbatches": args.microbatches,
        "micro_batch_size": args.micro_batch_size, "seq_len": args.seq_len,
        "activation_bytes_per_p2p": act_bytes, "p2p_per_boundary_per_step": 2 * args.microbatches,
        "bubble_frac_ideal": (P - 1) / (args.microbatches + P - 1),
        "interconnect_truth": "training (pipeline P2P; no gradient AllReduce)",
    }, env)
    if env.is_main:
        print(f"pp_1f1b_train model={args.size} stages={P} layers/stage={[h - l for l, h in bounds]} "
              f"M={args.microbatches} mb={args.micro_batch_size} T={args.seq_len} dtype={dtype}")
        print(f"  activation P2P = {act_bytes / 1e6:.2f} MB; ideal bubble = "
              f"{(P - 1) / (args.microbatches + P - 1):.1%}; stage0 params {n_params / 1e6:.0f}M")

    stop = StopSignal(env, args.duration, args.max_steps)
    dist.barrier()
    stop.t0 = t_start = time.time()
    step = 0
    while not stop.should_stop(step):
        t_s = time.time()
        loss = pipe.train_step(step)
        if args.verify and step == 0:
            verify_against_reference(pipe, stage, args, step)
        opt.step()
        opt.zero_grad(set_to_none=True)
        env.sync()
        step += 1
        if env.rank == P - 1 and (step <= 3 or step % 10 == 0):
            tok = args.microbatches * args.micro_batch_size * args.seq_len
            print(f"  step {step}: loss={loss:.4f} {time.time() - t_s:.3f}s "
                  f"({tok / (time.time() - t_s):.0f} tok/s, blocked {pipe.blocked / (time.time() - t_s):.0%})")

    dist.barrier()
    if env.is_main:
        print(f"pp_1f1b_train done: {step} steps, {time.time() - t_start:.1f}s")
    comm.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
