#!/usr/bin/env python3
"""Production DiLoCo training workload for multi-node interconnect red-team.

Replaces the ICTD toy (`diloco_train.py`): real NCCL, bucketed outer AllReduce,
OpenDiLoCo defaults (H=500, FP16 outer), optional Streaming DiLoCo, HF or
sized transformers, STEP JSONL for detectors.

Launch (2 nodes × 8 GPU example — VCF / any torchrun fabric)::

    # node 0
    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=0 \\
      --master_addr=$MASTER --master_port=29500 \\
      workloads/diloco_production.py \\
      --model-size 1b --local-steps 500 --reduce-dtype fp16 \\
      --max-steps 2000 --comm-log runs/diloco_comm.jsonl

    # streaming (crush peak / periodicity)
    torchrun ... workloads/diloco_production.py \\
      --model-size 1b --streaming --num-fragments 8 \\
      --local-steps 512 --fragment-sync-delay 4 --reduce-dtype fp16

See also: scripts/run_diloco_vcf.sh
"""
from __future__ import annotations

import argparse
import json
import os
import time
from typing import Iterator

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, IterableDataset

from diloco_core import CommLogger, DiLoCoController, ReduceDType

# ── Model sizes (same family as workloads/fsdp_training.py) ──────────────────
# (hidden, layers, heads, ffn, vocab) ≈ params
MODEL_CONFIGS = {
    "150m": (768, 12, 12, 3072, 32000),    # ~150M — OpenDiLoCo smoke
    "1b":   (2048, 24, 16, 8192, 32000),   # ~1.3B
    "7b":   (4096, 32, 32, 11008, 32000),
}


class TransformerBlock(nn.Module):
    def __init__(self, hidden: int, heads: int, ffn: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(hidden)
        self.attn = nn.MultiheadAttention(hidden, heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden)
        self.mlp = nn.Sequential(
            nn.Linear(hidden, ffn, bias=False),
            nn.GELU(),
            nn.Linear(ffn, hidden, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm1(x)
        h, _ = self.attn(h, h, h, need_weights=False)
        x = x + h
        return x + self.mlp(self.norm2(x))


class CausalLM(nn.Module):
    """Decoder LM with `.blocks.N` names so streaming splits on layer boundaries."""

    def __init__(self, size: str):
        super().__init__()
        hidden, n_layers, heads, ffn, vocab = MODEL_CONFIGS[size]
        self.vocab_size = vocab
        self.tok_emb = nn.Embedding(vocab, hidden)
        self.pos_emb = nn.Embedding(8192, hidden)
        self.blocks = nn.ModuleList(
            [TransformerBlock(hidden, heads, ffn) for _ in range(n_layers)]
        )
        self.norm = nn.LayerNorm(hidden)
        self.lm_head = nn.Linear(hidden, vocab, bias=False)

    def forward(self, input_ids: torch.Tensor, labels: torch.Tensor | None = None):
        B, T = input_ids.shape
        pos = torch.arange(T, device=input_ids.device).unsqueeze(0)
        x = self.tok_emb(input_ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        logits = self.lm_head(self.norm(x))
        loss = None
        if labels is not None:
            loss = F.cross_entropy(
                logits[:, :-1].reshape(-1, logits.size(-1)),
                labels[:, 1:].reshape(-1),
            )
        return logits, loss


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())


# ── Data ─────────────────────────────────────────────────────────────────────

class SyntheticTokens(IterableDataset):
    """Infinite random tokens — real matmuls, no download. Prefer --dataset c4 on fabric."""

    def __init__(self, seq_len: int, vocab: int, seed: int = 0):
        self.seq_len = seq_len
        self.vocab = vocab
        self.seed = seed

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        rank = int(os.environ.get("RANK", 0))
        g = torch.Generator()
        g.manual_seed(self.seed + rank * 10007)
        while True:
            x = torch.randint(0, self.vocab, (self.seq_len,), generator=g)
            yield {"input_ids": x, "labels": x.clone()}


class HFStreamC4(IterableDataset):
    def __init__(self, seq_len: int, tokenizer_name: str, rank: int, world: int):
        from datasets import load_dataset
        from transformers import AutoTokenizer

        self.seq_len = seq_len
        self.tok = AutoTokenizer.from_pretrained(tokenizer_name, use_fast=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        ds = load_dataset("allenai/c4", "en", split="train", streaming=True)
        self.ds = ds.shard(num_shards=world, index=rank)

    def __iter__(self) -> Iterator[dict[str, torch.Tensor]]:
        buf: list[int] = []
        for row in self.ds:
            ids = self.tok(row["text"], add_special_tokens=False)["input_ids"]
            buf.extend(ids)
            while len(buf) >= self.seq_len:
                chunk = buf[: self.seq_len]
                buf = buf[self.seq_len :]
                t = torch.tensor(chunk, dtype=torch.long)
                yield {"input_ids": t, "labels": t.clone()}


def build_loader(args, vocab: int, rank: int, world: int) -> DataLoader:
    if args.dataset == "c4":
        ds: Dataset = HFStreamC4(args.seq_len, args.tokenizer, rank, world)
    else:
        ds = SyntheticTokens(args.seq_len, vocab, seed=args.seed)
    return DataLoader(ds, batch_size=args.batch_size, num_workers=args.num_workers)


# ── Dist / STEP ──────────────────────────────────────────────────────────────

def dist_setup() -> tuple[int, int, int, torch.device]:
    if "RANK" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = int(os.environ["RANK"])
        world = int(os.environ["WORLD_SIZE"])
        local = int(os.environ["LOCAL_RANK"])
    else:
        # single-process fallback (still real DiLoCo codepath, world=1)
        rank, world, local = 0, 1, 0
    torch.cuda.set_device(local)
    return rank, world, local, torch.device(f"cuda:{local}")


def dist_cleanup() -> None:
    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


class StepLogger:
    def __init__(self, path: str | None, rank: int):
        self.rank = rank
        self._fh = None
        if path and rank == 0:
            os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
            self._fh = open(path, "a", buffering=1)

    def log(self, **kw) -> None:
        if self._fh is None:
            return
        kw.setdefault("t", time.time())
        self._fh.write(json.dumps(kw, separators=(",", ":")) + "\n")

    def close(self) -> None:
        if self._fh:
            self._fh.close()


def maybe_hf_model(args, device: torch.device) -> nn.Module:
    if not args.hf_model:
        m = CausalLM(args.model_size).to(device)
        return m
    from transformers import AutoModelForCausalLM

    m = AutoModelForCausalLM.from_pretrained(
        args.hf_model,
        torch_dtype=torch.bfloat16 if args.precision == "bf16" else torch.float32,
        attn_implementation="sdpa",
    )
    return m.to(device)


def train(args: argparse.Namespace) -> None:
    rank, world, local, device = dist_setup()
    is_main = rank == 0

    if is_main:
        print(
            f"DiLoCo production | world={world} H={args.local_steps} "
            f"streaming={args.streaming} frags={args.num_fragments} "
            f"reduce={args.reduce_dtype} model={args.hf_model or args.model_size}"
        )

    model = maybe_hf_model(args, device)
    nparams = count_params(model)
    if is_main:
        print(f"  params={nparams/1e6:.1f}M  device={device}")

    # Intra-node: each DiLoCo *replica* is one process. For multi-GPU-per-replica
    # wrap with FSDP yourself; here every rank is a DiLoCo worker (OpenDiLoCo
    # "galaxy" member) — matching multi-node fabric experiments.
    vocab = getattr(model, "vocab_size", None) or MODEL_CONFIGS[args.model_size][4]
    if args.hf_model:
        vocab = model.config.vocab_size

    loader = build_loader(args, vocab, rank, world)
    data_iter = iter(loader)

    inner = torch.optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )

    comm = CommLogger(path=args.comm_log, rank=rank)
    comm.open()
    step_log = StepLogger(args.step_log, rank)

    ctrl = DiLoCoController(
        model,
        inner,
        local_steps=args.local_steps,
        outer_lr=args.outer_lr,
        outer_momentum=args.outer_momentum,
        reduce_dtype=args.reduce_dtype,  # type: ignore[arg-type]
        bucket_cap_mb=args.bucket_cap_mb,
        streaming=args.streaming,
        num_fragments=args.num_fragments,
        fragment_sync_delay=args.fragment_sync_delay,
        backup_device=args.backup_device,
        logger=comm,
    )

    use_bf16 = args.precision == "bf16"
    scaler = torch.cuda.amp.GradScaler(enabled=(args.precision == "fp16"))

    step_log.log(
        event="start",
        world_size=world,
        local_steps=args.local_steps,
        streaming=args.streaming,
        num_fragments=ctrl.num_fragments,
        reduce_dtype=args.reduce_dtype,
        nparams=nparams,
        model=args.hf_model or args.model_size,
    )

    model.train()
    t0 = time.time()
    tokens = 0

    for step in range(1, args.max_steps + 1):
        batch = next(data_iter)
        input_ids = batch["input_ids"].to(device, non_blocking=True)
        labels = batch["labels"].to(device, non_blocking=True)

        inner.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(dtype=torch.bfloat16 if use_bf16 else torch.float16, enabled=args.precision != "fp32"):
            if args.hf_model:
                out = model(input_ids=input_ids, labels=labels)
                loss = out.loss
            else:
                _, loss = model(input_ids, labels)

        if args.precision == "fp16":
            scaler.scale(loss).backward()
            scaler.unscale_(inner)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(inner)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            inner.step()

        phase = ctrl.after_inner_step()
        tokens += input_ids.numel()

        if step % args.log_every == 0 or phase != "inner":
            step_log.log(
                event="step",
                step=step,
                sync_phase=phase,
                loss=float(loss.detach().float().item()),
                tokens=tokens,
                elapsed_s=time.time() - t0,
            )
            if is_main:
                tok_s = tokens / max(time.time() - t0, 1e-6)
                print(
                    f"step={step} phase={phase} loss={loss.item():.4f} "
                    f"tok/s={tok_s:.0f} H_rem={ctrl.local_steps - ctrl._step_in_period}"
                )

    ctrl.force_sync()
    step_log.log(event="end", steps=args.max_steps, elapsed_s=time.time() - t0)
    if is_main:
        print(f"done in {time.time() - t0:.1f}s")

    step_log.close()
    comm.close()
    dist_cleanup()


def build_argparser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model-size", choices=list(MODEL_CONFIGS), default="150m")
    p.add_argument("--hf-model", default=None, help="HF id (e.g. meta-llama/Llama-3.2-1B). Overrides --model-size.")
    p.add_argument("--tokenizer", default="meta-llama/Llama-3.2-1B")
    p.add_argument("--dataset", choices=["synthetic", "c4"], default="synthetic")
    p.add_argument("--seq-len", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=2, help="Per-rank microbatch")
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=0)

    # DiLoCo — OpenDiLoCo defaults
    p.add_argument("--local-steps", type=int, default=500, help="H inner steps (OpenDiLoCo default 500)")
    p.add_argument("--outer-lr", type=float, default=0.7)
    p.add_argument("--outer-momentum", type=float, default=0.9)
    p.add_argument("--lr", type=float, default=4e-4)
    p.add_argument("--weight-decay", type=float, default=0.1)
    p.add_argument(
        "--reduce-dtype",
        choices=["fp32", "fp16", "bf16", "fp8"],
        default="fp16",
        help="Outer pseudo-grad AllReduce dtype (OpenDiLoCo: fp16)",
    )
    p.add_argument("--bucket-cap-mb", type=float, default=256.0)
    p.add_argument("--streaming", action="store_true", help="Streaming DiLoCo fragment schedule")
    p.add_argument("--num-fragments", type=int, default=4)
    p.add_argument("--fragment-sync-delay", type=int, default=0, help="Overlap: prepare AR this many steps before wait")
    p.add_argument("--backup-device", default="cpu", help="Shadow param device (cpu pin-memory or cuda)")

    p.add_argument("--precision", choices=["fp32", "fp16", "bf16"], default="bf16")
    p.add_argument("--max-steps", type=int, default=2000)
    p.add_argument("--log-every", type=int, default=20)
    p.add_argument("--step-log", default="runs/diloco_steps.jsonl")
    p.add_argument("--comm-log", default="runs/diloco_comm.jsonl")
    return p


if __name__ == "__main__":
    torch.set_float32_matmul_precision("high")
    train(build_argparser().parse_args())
