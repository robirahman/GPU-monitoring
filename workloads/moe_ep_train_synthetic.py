#!/usr/bin/env python3
"""W05 -- MoE expert-parallel *training* fingerprint (synthetic; AllToAll incl. backward).

**MICROBENCH ONLY -- not SPAR primary corpus.** Hand-written A2A gym with shared expert weights. The
primary W05 workload is real torchtitan MoE training (``scripts/run_titan_job.py``, registry
``moe_ep_train_dsv3_16b_ep8``).

Counterpart of ``moe_ep_synthetic.py`` (W12/W20, infer-shaped: forward-only A2A pairs).
Here the same dispatch -> expert GEMM -> combine layer runs inside a real training
loop, so the fabric sees the *backward* AllToAlls too, plus optional gradient AllReduce:

    per layer, forward :  A2A dispatch  ->  local expert FFN  ->  A2A combine
    per layer, backward:  A2A combine^T ->  expert backward   ->  A2A dispatch^T
    per step, optional :  AllReduce(dense grads, world) [--dense-sync]
                          AllReduce(expert grads, DP group) [--ep-size < world]

so per step there are 4*layers AllToAlls (2x the infer shape) with expert grads staying
local -- the "loud but not AllReduce-shaped" training fingerprint DeepSeek-V3/DeepEP-style
MoE training produces. Not a real Mixtral/DeepSeek train: expert weights are *shared
across layers* (like moe_ep_synthetic) so memory stays bounded; only the communication
shape and per-layer compute are faithful. Do not report it as real MoE training.

Layout: world = EP x DP. ``--ep-size`` ranks form each expert-parallel group (A2A runs
inside it, consecutive ranks); ranks with the same EP index form DP groups (expert-grad
AllReduce). Default ``--ep-size`` = world (pure EP, DP=1).

Comm log (rank0 JSONL, mirrors moe_ep_synthetic): ``op=all_to_all`` with
``phase=dispatch_fwd|combine_fwd|combine_bwd|dispatch_bwd`` (``bytes_est`` = send buffer,
same convention as moe_ep_synthetic, so infer and train logs compare directly) and
``op=all_reduce`` with ``phase=grad_sync_dense|grad_sync_expert``; ``step``, ``layer``,
``group``.

Launch::

    torchrun --nproc_per_node=8 workloads/moe_ep_train_synthetic.py \\
        --duration 300 --layers 16 --hidden 4096 --tokens-per-step 4096 \\
        --experts-per-rank 2 --comm-log data/moe_ep_train_comm.jsonl
    # EP=4 x DP=2 with dense-param AllReduce:
    torchrun --nproc_per_node=8 workloads/moe_ep_train_synthetic.py --ep-size 4 --dense-sync ...
    python scripts/run_workload_multi_gpu.py --workload moe_ep_train_synthetic --num-gpus 8

CPU smoke::

    torchrun --nproc_per_node=4 workloads/moe_ep_train_synthetic.py --backend gloo --ep-size 2 \\
        --dense-sync --layers 2 --hidden 64 --tokens-per-step 16 --experts-per-rank 2 \\
        --max-steps 4 --comm-log /tmp/moe_train.jsonl
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from interconnect_common import (
    JsonlLog,
    StopSignal,
    add_common_args,
    chunked_allreduce,
    dist_cleanup,
    dist_setup,
    dtype_from_name,
    ensure_grads,
    flatten_into,
    tensor_bytes,
    unflatten_from,
    write_meta,
)

MIB = 1024 * 1024


class _A2A(torch.autograd.Function):
    """Equal-split all_to_all_single whose backward is the (self-adjoint) A2A on the grad."""

    @staticmethod
    def forward(ctx, x, group, log, name, layer):
        ctx.group, ctx.log, ctx.name, ctx.layer = group, log, name, layer
        x = x.contiguous()
        out = torch.empty_like(x)
        _run_a2a(out, x, group, log, f"{name}_fwd", layer)
        return out

    @staticmethod
    def backward(ctx, g):
        g = g.contiguous()
        out = torch.empty_like(g)
        _run_a2a(out, g, ctx.group, ctx.log, f"{ctx.name}_bwd", ctx.layer)
        return out, None, None, None, None


def _run_a2a(out, inp, group, log, phase, layer):
    t0 = time.time()
    dist.all_to_all_single(out, inp, group=group)
    payload = tensor_bytes(inp)
    log.emit(t=t0, op="all_to_all", phase=phase, layer=layer, group="ep",
             group_size=dist.get_world_size(group), bytes_est=payload, payload_bytes=payload)


def make_groups(env, ep_size: int):
    """(ep_group, dp_group, ep_rank, dp_rank). Collective over all ranks."""
    if env.world % ep_size:
        raise SystemExit(f"--ep-size {ep_size} must divide world size {env.world}")
    dp_size = env.world // ep_size
    ep_group = dp_group = None            # None = default (world) group
    if ep_size < env.world:
        for start in range(0, env.world, ep_size):
            g = dist.new_group(list(range(start, start + ep_size)))
            if start <= env.rank < start + ep_size:
                ep_group = g
    if dp_size > 1:
        for r in range(ep_size):
            g = dist.new_group(list(range(r, env.world, ep_size)))
            if env.rank % ep_size == r:
                dp_group = g
    return ep_group, dp_group, env.rank % ep_size, env.rank // ep_size, dp_size


class MoETrainStack(nn.Module):
    def __init__(self, layers, hidden, inter, experts, dense, dtype, device, seed, ep_rank, log,
                 ep_group):
        super().__init__()
        self.layers, self.log, self.ep_group, self.experts_n = layers, log, ep_group, experts
        torch.manual_seed(seed)                                   # dense params: identical on all ranks
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dense = nn.ModuleList([nn.Linear(hidden, hidden, bias=False) for _ in range(layers)]) if dense else None
        torch.manual_seed(seed + 1 + ep_rank)                     # experts: differ across EP ranks only
        self.w1 = nn.Parameter(torch.randn(experts, hidden, inter) * 0.02)
        self.w2 = nn.Parameter(torch.randn(experts, inter, hidden) * 0.02)
        self.to(device=device, dtype=dtype)

    def dense_params(self):
        return list(self.norms.parameters()) + (list(self.dense.parameters()) if self.dense is not None else [])

    def expert_params(self):
        return [self.w1, self.w2]

    def forward(self, x):
        for layer in range(self.layers):
            h = self.norms[layer](x)
            if self.dense is not None:
                h = h + self.dense[layer](h)
            disp = _A2A.apply(h, self.ep_group, self.log, "dispatch", layer)
            chunks = disp.chunk(self.experts_n, dim=0)             # local experts share the tokens
            outs = [F.gelu(c @ self.w1[e]) @ self.w2[e] for e, c in enumerate(chunks)]
            comb = _A2A.apply(torch.cat(outs, dim=0), self.ep_group, self.log, "combine", layer)
            x = x + comb
        return x


def main() -> None:
    ap = argparse.ArgumentParser(description="W05 synthetic MoE EP training (AllToAll fwd+bwd)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--layers", type=int, default=16, help="MoE layers (4 A2A per layer per step)")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--tokens-per-step", type=int, default=4096,
                    help="Tokens per rank per step (must be divisible by --ep-size)")
    ap.add_argument("--experts-per-rank", type=int, default=2)
    ap.add_argument("--ffn-mult", type=int, default=2, help="Expert intermediate = hidden * ffn_mult")
    ap.add_argument("--ep-size", type=int, default=0, help="Expert-parallel group size (0 = world)")
    ap.add_argument("--dense-sync", action="store_true",
                    help="Add per-layer dense params and AllReduce their grads across the world")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--bucket-mb", type=float, default=256.0)
    args = ap.parse_args()

    env = dist_setup(args.backend)
    ep_size = args.ep_size or env.world
    ep_group, dp_group, ep_rank, dp_rank, dp_size = make_groups(env, ep_size)
    if args.tokens_per_step % ep_size:
        raise SystemExit(f"--tokens-per-step {args.tokens_per_step} must be divisible by ep-size {ep_size}")
    if args.tokens_per_step % args.experts_per_rank:
        raise SystemExit("--tokens-per-step must be divisible by --experts-per-rank")
    dtype = dtype_from_name(args.dtype) if env.device.type == "cuda" else torch.float32

    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    net = MoETrainStack(args.layers, args.hidden, args.hidden * args.ffn_mult, args.experts_per_rank,
                        args.dense_sync, dtype, env.device, args.seed, ep_rank, comm, ep_group)
    torch.manual_seed(args.seed + 100 + env.rank)
    x0 = torch.randn(args.tokens_per_step, args.hidden, device=env.device, dtype=dtype)
    target = torch.randn_like(x0)

    dense_ps, expert_ps = net.dense_params(), net.expert_params()
    do_dense = args.dense_sync and env.world > 1
    do_expert_dp = dp_size > 1
    flat_dense = torch.zeros(sum(p.numel() for p in dense_ps), dtype=dtype, device=env.device) if do_dense else None
    flat_exp = torch.zeros(sum(p.numel() for p in expert_ps), dtype=dtype, device=env.device) if do_expert_dp else None
    opt = torch.optim.SGD(list(net.parameters()), lr=args.lr)

    a2a_bytes = args.tokens_per_step * args.hidden * x0.element_size()
    write_meta(args.comm_log, {
        "workload": "moe_ep_train_synthetic", "wid": "W05", "world_size": env.world, "backend": env.backend,
        "ep_size": ep_size, "dp_size": dp_size, "layers": args.layers, "hidden": args.hidden,
        "tokens_per_step": args.tokens_per_step, "experts_per_rank": args.experts_per_rank,
        "a2a_payload_bytes": a2a_bytes, "a2a_per_step": 4 * args.layers, "dense_sync": args.dense_sync,
        "dense_grad_bytes": sum(tensor_bytes(p) for p in dense_ps) if do_dense else 0,
        "expert_grad_bytes": sum(tensor_bytes(p) for p in expert_ps) if do_expert_dp else 0,
        "note": "expert weights shared across layers (comm shape stand-in); not real MoE training",
        "interconnect_truth": "training (AllToAll dispatch/combine fwd+bwd)",
    }, env)
    if env.is_main:
        print(f"moe_ep_train_synthetic world={env.world} EP={ep_size} DP={dp_size} layers={args.layers} "
              f"hidden={args.hidden} tokens/step={args.tokens_per_step} experts/rank={args.experts_per_rank} "
              f"dtype={dtype} dense_sync={args.dense_sync}")
        print(f"  {4 * args.layers} A2A/step x {a2a_bytes / MIB:.1f} MiB")

    stop = StopSignal(env, args.duration, args.max_steps)
    dist.barrier()
    stop.t0 = t_start = time.time()
    step = 0
    while not stop.should_stop(step):
        comm.step = step
        t_step = time.time()
        out = net(x0)
        loss = F.mse_loss(out.float(), target.float())
        loss.backward()                                   # backward A2As are logged from autograd
        t_bwd = time.time()

        if do_dense:
            grads = ensure_grads(dense_ps)
            flatten_into(grads, flat_dense)
            chunked_allreduce(flat_dense, int(args.bucket_mb * MIB), env, log=comm,
                              phase="grad_sync_dense", group_name="world")
            unflatten_from(flat_dense, grads)
        if do_expert_dp:
            grads = ensure_grads(expert_ps)
            flatten_into(grads, flat_exp)
            chunked_allreduce(flat_exp, int(args.bucket_mb * MIB), env, group=dp_group, log=comm,
                              phase="grad_sync_expert", group_name="dp")
            unflatten_from(flat_exp, grads)
        opt.step()
        opt.zero_grad(set_to_none=True)
        env.sync()
        step += 1
        if env.is_main and (step <= 3 or step % 10 == 0):
            print(f"  step {step}: loss={float(loss.detach()):.4f} fwd+bwd={t_bwd - t_step:.3f}s "
                  f"total={time.time() - t_step:.3f}s")

    dist.barrier()
    if env.is_main:
        wall = time.time() - t_start
        print(f"moe_ep_train_synthetic done: {step} steps, {wall:.1f}s, {step * 4 * args.layers} A2A ops")
    comm.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
