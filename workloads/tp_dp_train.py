#!/usr/bin/env python3
"""W03 -- Tensor-parallel + data-parallel training (Megatron-lite).

**PARALLELISM UNIT / ABLATION -- not SPAR primary corpus** (replicated vocab/lm_head understates TP bytes;
random-init sized LM). Numerically verified against an unsharded reference, so it is a good unit test of the
TP/DP collective shapes. Primary W03: real torchtitan (``train_tp2_*_titan``).

Training with *two* collective types on two rank groups -- the fingerprint neither DDP
(one AllReduce family) nor FSDP (AG/RS) has:

  TP group (``--tp`` consecutive ranks, intra-node friendly): per transformer layer
      forward : AllReduce of the activations after attention and after the MLP
                (row-parallel outputs)                    -> tp_fwd_attn, tp_fwd_mlp
      backward: AllReduce of the activation gradient at the input of each
                column-parallel matmul                    -> tp_bwd_attn, tp_bwd_mlp
  DP group (ranks with the same TP index, stride ``--tp``): bucketed AllReduce of the
      gradients once per step, overlapped with backward (DDP over the DP subgroup)
                                                          -> grad_sync_dp

world = TP x DP. Self-contained torch.distributed, no Megatron-LM: column-parallel QKV /
MLP-up, row-parallel proj / MLP-down, conjugate ``f`` (identity fwd, AllReduce bwd) and
``g`` (AllReduce fwd, identity bwd) operators. Embeddings, LayerNorms and lm_head are
replicated inside a TP group (real Megatron shards the vocab; that would only add
AllReduces on the logits, so this understates TP traffic slightly). TP shards are literal
slices of one deterministically generated full weight set, so ``--verify`` can compare the
sharded model against an unsharded reference.

Comm log (rank0 JSONL): one line per collective -- ``op=all_reduce``, ``phase=tp_fwd_attn|
tp_fwd_mlp|tp_bwd_attn|tp_bwd_mlp`` (``group=tp``, ``layer``) or ``phase=grad_sync_dp``
(``group=dp``, per bucket), ``bytes_est``, ``payload_bytes``, ``step``.

Launch::

    torchrun --nproc_per_node=8 workloads/tp_dp_train.py --tp 2 --size 1b \\
        --batch-size 4 --seq-len 1024 --duration 300 --comm-log data/tp2_dp4_1b_comm.jsonl
    python scripts/run_workload_multi_gpu.py --workload train_tp2_dp4_1b --num-gpus 8

CPU smoke (checks shards == unsharded reference; needs heads % tp == 0)::

    torchrun --nproc_per_node=4 workloads/tp_dp_train.py --backend gloo --tp 2 --size tiny \\
        --batch-size 2 --seq-len 16 --max-steps 3 --verify --comm-log /tmp/tpdp.jsonl
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from interconnect_common import (
    CausalLM,
    JsonlLog,
    MODEL_CONFIGS,
    StopSignal,
    add_common_args,
    ar_wire_bytes,
    dist_cleanup,
    dist_setup,
    dtype_from_name,
    lm_loss,
    make_logging_allreduce_hook,
    make_optimizer,
    tensor_bytes,
    write_meta,
)

MAX_POS = 4096


# ── TP conjugate operators ───────────────────────────────────────────────────

def _tp_allreduce(t: torch.Tensor, group, log: JsonlLog, phase: str, layer: int) -> None:
    t0 = time.time()
    dist.all_reduce(t, group=group)
    n = dist.get_world_size(group)
    payload = tensor_bytes(t)
    log.emit(t=t0, op="all_reduce", phase=phase, layer=layer, group="tp", group_size=n,
             bytes_est=ar_wire_bytes(payload, n), payload_bytes=payload)


class _CopyToTP(torch.autograd.Function):
    """``f``: identity forward, AllReduce of the gradient in backward."""

    @staticmethod
    def forward(ctx, x, group, log, tag, layer):
        ctx.group, ctx.log, ctx.tag, ctx.layer = group, log, tag, layer
        return x.view_as(x)

    @staticmethod
    def backward(ctx, g):
        g = g.clone()
        _tp_allreduce(g, ctx.group, ctx.log, f"tp_bwd_{ctx.tag}", ctx.layer)
        return g, None, None, None, None


class _ReduceFromTP(torch.autograd.Function):
    """``g``: AllReduce forward (sum partial products), identity backward."""

    @staticmethod
    def forward(ctx, x, group, log, tag, layer):
        out = x.clone()
        _tp_allreduce(out, group, log, f"tp_fwd_{tag}", layer)
        return out

    @staticmethod
    def backward(ctx, g):
        return g, None, None, None, None


# ── Deterministic full weights, sliced into TP shards ────────────────────────

def _randn(shape, gen, device):
    return torch.randn(shape, generator=gen, device=device) * 0.02


def layer_full(size: str, g: int, seed: int, device) -> dict[str, torch.Tensor]:
    h, _, _, ffn, _ = MODEL_CONFIGS[size]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed + 1 + g)
    return {"qkv": _randn((3 * h, h), gen, device), "proj": _randn((h, h), gen, device),
            "up": _randn((ffn, h), gen, device), "down": _randn((h, ffn), gen, device)}


def head_full(size: str, seed: int, device) -> dict[str, torch.Tensor]:
    h, _, _, _, vocab = MODEL_CONFIGS[size]
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)
    return {"tok_emb": _randn((vocab, h), gen, device), "pos_emb": _randn((MAX_POS, h), gen, device),
            "lm_head": _randn((vocab, h), gen, device)}


def shard(kind: str, full: torch.Tensor, r: int, tp: int, size: str) -> torch.Tensor:
    """TP-rank ``r``'s slice of a full weight (nn.Linear layout: [out, in])."""
    h, _, heads, ffn, _ = MODEL_CONFIGS[size]
    hl = h // tp                      # == heads_local * head_dim (heads split evenly)
    fl = ffn // tp
    if kind == "qkv":                 # column-parallel; keep [3, heads_local*hd, h] ordering
        return full.view(3, h, h)[:, r * hl:(r + 1) * hl, :].reshape(3 * hl, h).contiguous()
    if kind == "proj":                # row-parallel: split input columns
        return full[:, r * hl:(r + 1) * hl].contiguous()
    if kind == "up":                  # column-parallel
        return full[r * fl:(r + 1) * fl].contiguous()
    if kind == "down":                # row-parallel
        return full[:, r * fl:(r + 1) * fl].contiguous()
    raise KeyError(kind)


class TPBlock(nn.Module):
    def __init__(self, size, tp, tp_rank, group, log, layer, w):
        super().__init__()
        h, _, heads, ffn, _ = MODEL_CONFIGS[size]
        self.tp, self.group, self.log, self.layer = tp, group, log, layer
        self.heads_l, self.hd = heads // tp, h // heads
        self.norm1, self.norm2 = nn.LayerNorm(h), nn.LayerNorm(h)
        self.qkv = nn.Linear(h, 3 * self.heads_l * self.hd, bias=False)
        self.proj = nn.Linear(self.heads_l * self.hd, h, bias=False)
        self.up = nn.Linear(h, ffn // tp, bias=False)
        self.down = nn.Linear(ffn // tp, h, bias=False)
        for kind in ("qkv", "proj", "up", "down"):
            getattr(self, kind).weight.data.copy_(shard(kind, w[kind], tp_rank, tp, size))

    def _f(self, x, tag):
        return _CopyToTP.apply(x, self.group, self.log, tag, self.layer) if self.tp > 1 else x

    def _g(self, x, tag):
        return _ReduceFromTP.apply(x, self.group, self.log, tag, self.layer) if self.tp > 1 else x

    def forward(self, x):
        b, t, h = x.shape
        y = self._f(self.norm1(x), "attn")
        q, k, v = self.qkv(y).view(b, t, 3, self.heads_l, self.hd).permute(2, 0, 3, 1, 4)
        a = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        x = x + self._g(self.proj(a.transpose(1, 2).reshape(b, t, self.heads_l * self.hd)), "attn")
        y = self._f(self.norm2(x), "mlp")
        return x + self._g(self.down(F.gelu(self.up(y))), "mlp")


class TPModel(nn.Module):
    """Same module names as interconnect_common.CausalLM so params map 1:1 to a reference."""

    def __init__(self, size, tp, tp_rank, group, log, seed, device):
        super().__init__()
        h, n_layers, _, _, vocab = MODEL_CONFIGS[size]
        self.vocab_size = vocab
        hf = head_full(size, seed, device)
        self.tok_emb, self.pos_emb = nn.Embedding(vocab, h), nn.Embedding(MAX_POS, h)
        self.tok_emb.weight.data.copy_(hf["tok_emb"])
        self.pos_emb.weight.data.copy_(hf["pos_emb"])
        self.blocks = nn.ModuleList(
            [TPBlock(size, tp, tp_rank, group, log, g, layer_full(size, g, seed, device))
             for g in range(n_layers)])
        self.norm = nn.LayerNorm(h)
        self.lm_head = nn.Linear(h, vocab, bias=False)
        self.lm_head.weight.data.copy_(hf["lm_head"])

    def forward(self, ids):
        pos = torch.arange(ids.size(1), device=ids.device).unsqueeze(0)
        x = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.lm_head(self.norm(x))


def build_reference(size: str, seed: int, device, dtype) -> CausalLM:
    ref = CausalLM(size, max_pos=MAX_POS).to(device)
    hf = head_full(size, seed, device)
    ref.tok_emb.weight.data.copy_(hf["tok_emb"])
    ref.pos_emb.weight.data.copy_(hf["pos_emb"])
    ref.lm_head.weight.data.copy_(hf["lm_head"])
    for g, blk in enumerate(ref.blocks):
        w = layer_full(size, g, seed, device)
        for kind in ("qkv", "proj", "up", "down"):
            getattr(blk, kind).weight.data.copy_(w[kind])
        for ln in (blk.norm1, blk.norm2):
            nn.init.ones_(ln.weight)
            nn.init.zeros_(ln.bias)
    nn.init.ones_(ref.norm.weight)
    nn.init.zeros_(ref.norm.bias)
    return ref.to(dtype)


def make_ids(step: int, dp_rank: int, args, vocab: int, device) -> torch.Tensor:
    g = torch.Generator(device=device)
    g.manual_seed(args.seed * 1_000_003 + step * 1009 + dp_rank)
    return torch.randint(0, vocab, (args.batch_size, args.seq_len), device=device, generator=g)


def verify(model: TPModel, args, tp_rank: int, dp: int, step: int, dtype, device) -> None:
    """Step 0: loss + gradients of this TP shard vs an unsharded model averaged over DP batches."""
    ref = build_reference(args.size, args.seed, device, dtype)
    loss_ref = 0.0
    for d in range(dp):
        ids = make_ids(step, d, args, model.vocab_size, device)
        loss = lm_loss(ref(ids), ids) / dp
        loss.backward()
        loss_ref += float(loss.detach())
    ref_grads = dict(ref.named_parameters())
    worst = 0.0
    for n, p in model.named_parameters():
        rg = ref_grads[n].grad
        kind = n.rsplit(".", 2)[-2] if n.startswith("blocks.") else ""
        if kind in ("qkv", "proj", "up", "down"):
            rg = shard(kind, rg, tp_rank, args.tp, args.size)
        worst = max(worst, float((p.grad - rg).abs().max() / rg.abs().max().clamp_min(1e-12)))
    print(f"  [verify] tp_rank {tp_rank}: ref loss(avg over DP)={loss_ref:.5f}, "
          f"max relative grad error vs unsharded = {worst:.2e}")
    tol = 1e-3 if dtype == torch.float32 else 5e-2
    if worst > tol:
        raise SystemExit(f"verify FAILED (tp_rank {tp_rank}): {worst} > {tol}")


def main() -> None:
    ap = argparse.ArgumentParser(description="W03 TP + DP training (Megatron-lite)",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--tp", type=int, default=2, help="Tensor-parallel group size")
    ap.add_argument("--dp", type=int, default=0, help="Data-parallel size (0 = world/tp; checked)")
    ap.add_argument("--size", choices=list(MODEL_CONFIGS), default="150m")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-DP-replica micro-batch")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--optimizer", choices=["adamw", "sgd", "sgd_momentum"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--bucket-mb", type=float, default=25.0, help="DDP bucket size for the DP AllReduce")
    ap.add_argument("--verify", action="store_true",
                    help="Step 0: check shards against an unsharded reference (small models)")
    args = ap.parse_args()

    env = dist_setup(args.backend)
    tp = args.tp
    if env.world % tp:
        raise SystemExit(f"world size {env.world} not divisible by --tp {tp}")
    dp = env.world // tp
    if args.dp and args.dp != dp:
        raise SystemExit(f"--dp {args.dp} inconsistent with world {env.world} / tp {tp} = {dp}")
    h, n_layers, heads, ffn, vocab = MODEL_CONFIGS[args.size]
    if heads % tp or ffn % tp or h % tp:
        raise SystemExit(f"--tp {tp} must divide heads ({heads}), hidden ({h}) and ffn ({ffn})")
    dtype = dtype_from_name(args.dtype) if env.device.type == "cuda" else torch.float32

    tp_rank, dp_rank = env.rank % tp, env.rank // tp
    tp_group = dp_group = None            # None = default (world) group
    # Collective: every rank creates every group.
    if 1 < tp < env.world:
        for d in range(dp):
            g = dist.new_group(list(range(d * tp, (d + 1) * tp)))
            if d == dp_rank:
                tp_group = g
    if dp > 1:
        for t in range(tp):
            g = dist.new_group(list(range(t, env.world, tp)))
            if t == tp_rank:
                dp_group = g

    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    model = TPModel(args.size, tp, tp_rank, tp_group, comm, args.seed, env.device).to(dtype)
    n_params = sum(p.numel() for p in model.parameters())
    param_bytes = sum(tensor_bytes(p) for p in model.parameters())
    trainable = model
    if dp > 1:
        ddp = DDP(model, device_ids=[env.local_rank] if env.device.type == "cuda" else None,
                  process_group=dp_group, broadcast_buffers=False, find_unused_parameters=False,
                  bucket_cap_mb=args.bucket_mb)
        ddp.register_comm_hook(dp_group, make_logging_allreduce_hook(comm, phase="grad_sync_dp",
                                                                     group_name="dp"))
        trainable = ddp
    opt = make_optimizer(args.optimizer, model.parameters(), args.lr)

    act_bytes = args.batch_size * args.seq_len * h * torch.empty(0, dtype=dtype).element_size()
    write_meta(args.comm_log, {
        "workload": "tp_dp_train", "wid": "W03", "world_size": env.world, "backend": env.backend,
        "tp": tp, "dp": dp, "model": args.size, "batch_size": args.batch_size, "seq_len": args.seq_len,
        "tp_allreduce_payload_bytes": act_bytes, "tp_allreduce_per_step": 4 * n_layers if tp > 1 else 0,
        "params_per_rank": n_params, "dp_grad_bytes_per_rank": param_bytes if dp > 1 else 0,
        "tp_groups": [list(range(d * tp, (d + 1) * tp)) for d in range(dp)],
        "dp_groups": [list(range(t, env.world, tp)) for t in range(tp)],
        "interconnect_truth": "training (TP AllReduce on activations + DP AllReduce on grads)",
    }, env)
    if env.is_main:
        print(f"tp_dp_train model={args.size} world={env.world} TP={tp} DP={dp} dtype={dtype} "
              f"params/rank={n_params / 1e6:.1f}M")
        print(f"  TP: {4 * n_layers if tp > 1 else 0} x {act_bytes / 1e6:.2f} MB AllReduce/step; "
              f"DP: ~{param_bytes / 1e6:.0f} MB grads/step over {dp} replicas")

    stop = StopSignal(env, args.duration, args.max_steps)
    trainable.train()
    dist.barrier()
    stop.t0 = t_start = time.time()
    step = 0
    while not stop.should_stop(step):
        comm.step = step
        t_s = time.time()
        ids = make_ids(step, dp_rank, args, vocab, env.device)
        loss = lm_loss(trainable(ids), ids)
        loss.backward()                 # TP AllReduces in autograd; DP buckets via the DDP hook
        if args.verify and step == 0:
            verify(model, args, tp_rank, dp, step, dtype, env.device)
        opt.step()
        opt.zero_grad(set_to_none=True)
        env.sync()
        step += 1
        if env.is_main and (step <= 3 or step % 10 == 0):
            print(f"  step {step}: loss={float(loss.detach()):.4f} {time.time() - t_s:.3f}s")

    dist.barrier()
    if env.is_main:
        print(f"tp_dp_train done: {step} steps, {time.time() - t_start:.1f}s")
    comm.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
