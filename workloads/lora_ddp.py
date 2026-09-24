#!/usr/bin/env python3
"""W16 -- LoRA / PEFT-only cross-rank sync (interconnect FN).

Multi-GPU (or multi-node) DDP fine-tuning where *only the LoRA adapter parameters*
are in the AllReduce. The base model is frozen, identical on every rank (each rank
loads the same checkpoint) and never synced, so per-step fabric traffic is MB-scale
(Qwen2.5-1.5B, r=16, q/v: ~2.2M adapter params ~ 9 MB) rather than |theta| (~3 GB).
Interconnect twin of Rahman's white-box LoRA family (``adversarial_whitebox.py``
WB-L), which was single-GPU only.

How base traffic is avoided
  * DDP's reducer only buckets ``requires_grad`` params -> frozen base is never
    AllReduced.
  * DDP's constructor normally broadcasts *all* module state from rank 0 once. We pass
    ``init_sync=False`` (default) so even that one |theta| broadcast is skipped, and
    broadcast just the adapters ourselves (seeded identically as well). ``--init-sync``
    restores stock behaviour ("broadcast the base once").

LoRA wiring is ``lora_training.apply_lora`` (same PEFT config as the matched-pair
LoRA runs). Synthetic tokens, forward+backward+step every batch (no dilution), so the
telemetry is unambiguously training.

**W24 variant (``train_serve_cover_lora_H20``)**: ``--sync-every H`` accumulates gradients locally
(``ddp.no_sync()``) and AllReduces the adapters only on every H-th step; ``--cover`` adds W08-scale
decode-shaped cover AllReduces (``interconnect_common.CoverTraffic``, same mechanism as W19
``hybrid_wire``) every step. Because the synced payload is MB-scale, this is the W24 cell where
the *byte* share of training stays in the low single digits (H=20: ~6% of wire bytes), i.e. the
mean interconnect rate genuinely sits in the serving band -- unlike full-|theta| sync (see
``hybrid_wire``). Step windows: ``phase = decode_shape | train_sync`` every step.

Comm log: one line per AllReduce (Python hook path, exact bucket sizes),
``phase=grad_sync_lora``; ``<stem>.steps.jsonl`` has per-step ``bytes_est`` vs
``dense_bytes_est`` (what full-parameter DDP of this model would send).

Launch::

    torchrun --nproc_per_node=8 workloads/lora_ddp.py \\
        --model Qwen/Qwen2.5-1.5B-Instruct --duration 300 --batch-size 2 --seq-len 256 \\
        --comm-log data/lora_ddp_qwen15_comm.jsonl
    python scripts/run_workload_multi_gpu.py --workload train_lora_only_ddp_qwen15 --num-gpus 8

CPU smoke (random tiny Llama, no download)::

    torchrun --nproc_per_node=2 workloads/lora_ddp.py --backend gloo --model random-tiny \\
        --batch-size 2 --seq-len 32 --max-steps 5 --comm-log /tmp/lora_ddp.jsonl
"""
from __future__ import annotations

import argparse
import contextlib
import io
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

from interconnect_common import (
    AllReduceTap,
    CoverTraffic,
    JsonlLog,
    StopSignal,
    add_common_args,
    add_cover_args,
    ar_wire_bytes,
    dist_cleanup,
    dist_setup,
    cover_from_args,
    duty_fractions,
    make_optimizer,
    sibling_path,
    synthetic_tokens,
    tensor_bytes,
    write_meta,
)
from lora_training import apply_lora


def load_base(model_id: str, dtype: torch.dtype, device: torch.device):
    from transformers import AutoModelForCausalLM

    if model_id == "random-tiny":     # smoke-test path: random Llama, no download
        from transformers import LlamaConfig

        cfg = LlamaConfig(vocab_size=1024, hidden_size=64, intermediate_size=128,
                          num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
                          max_position_embeddings=256)
        model = AutoModelForCausalLM.from_config(cfg)
    else:
        model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype,
                                                     trust_remote_code=False)
    return model.to(device=device, dtype=dtype)


def main() -> None:
    ap = argparse.ArgumentParser(description="W16 LoRA-only DDP",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--model", required=True,
                    help="HF id (Qwen/Qwen2.5-1.5B-Instruct, TinyLlama/TinyLlama-1.1B-Chat-v1.0) "
                         "or 'random-tiny' for CPU smoke tests")
    ap.add_argument("--rank", type=int, default=16, help="LoRA rank r")
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument("--target-modules", default="q_proj,v_proj",
                    help="Comma list of module-name substrings, or 'all-linear'")
    ap.add_argument("--batch-size", type=int, default=2, help="Per-rank micro-batch")
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--dtype", choices=["auto", "bf16", "fp16", "fp32"], default="auto",
                    help="Base weight dtype (auto: bf16 on GPU, fp32 on CPU)")
    ap.add_argument("--grad-ckpt", action="store_true", help="Non-reentrant gradient checkpointing")
    ap.add_argument("--init-sync", action="store_true",
                    help="Let DDP broadcast the whole base model once at construction")
    ap.add_argument("--sync-every", "-H", type=int, default=1, dest="sync_every",
                    help="W24: AllReduce adapters every H steps (local accumulation in between)")
    ap.add_argument("--cover", action="store_true",
                    help="W24: emit decode-shaped cover AllReduces every step (see --cover-tp ...)")
    add_cover_args(ap)
    args = ap.parse_args()

    env = dist_setup(args.backend)
    if args.dtype == "auto":
        dtype = torch.bfloat16 if env.device.type == "cuda" else torch.float32
    else:
        dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}[args.dtype]

    torch.manual_seed(args.seed)            # identical base (random-tiny) and LoRA-A init on all ranks
    base = load_base(args.model, dtype, env.device)
    n_total_base = sum(p.numel() for p in base.parameters())
    dense_bytes = sum(tensor_bytes(p) for p in base.parameters())     # full-FT grads ~ param dtype
    if args.grad_ckpt:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        base.enable_input_require_grads()

    tm = args.target_modules.strip()
    targets = tm if tm == "all-linear" else [s.strip() for s in tm.split(",") if s.strip()]
    # apply_lora prints; keep only rank 0's line.
    quiet = contextlib.redirect_stdout(io.StringIO()) if not env.is_main else contextlib.nullcontext()
    with quiet:
        model = apply_lora(base, rank=args.rank, alpha=args.lora_alpha, target_modules=targets)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_train = sum(p.numel() for p in trainable)
    train_bytes = sum(tensor_bytes(p) for p in trainable)
    for p in trainable:                     # adapters only: MB-scale one-time broadcast
        dist.broadcast(p.data, src=0)

    ddp_kwargs = dict(device_ids=[env.local_rank] if env.device.type == "cuda" else None,
                      broadcast_buffers=False, find_unused_parameters=False)
    if not args.init_sync:
        ddp_kwargs["init_sync"] = False
    try:
        ddp = DDP(model, **ddp_kwargs)
    except TypeError:                     # torch without DDP(init_sync=...): stock init broadcast
        ddp_kwargs.pop("init_sync", None)
        ddp = DDP(model, **ddp_kwargs)
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    ddp.register_comm_hook(None, default_hooks.allreduce_hook)   # Python path -> loggable

    opt = make_optimizer("adamw", trainable, args.lr)
    torch.manual_seed(args.seed + 1 + env.rank)
    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    steps = JsonlLog(sibling_path(args.comm_log, ".steps") if args.comm_log else "", env)
    dense_est = ar_wire_bytes(dense_bytes, env.world)
    lora_est = ar_wire_bytes(train_bytes, env.world)
    if args.sync_every < 1:
        raise SystemExit("--sync-every must be >= 1")
    cover: CoverTraffic | None = cover_from_args(env, comm, args) if args.cover else None
    windowed = cover is not None or args.sync_every > 1      # W24 mode: per-step window phases
    w24 = {**cover.meta(), **duty_fractions(args.sync_every, lora_est, cover.wire_bytes_per_call)} \
        if cover is not None else {}
    write_meta(args.comm_log, {
        "workload": "lora_ddp", "wid": "W24" if windowed else "W16", "world_size": env.world,
        "backend": env.backend, "sync_every": args.sync_every, "cover": cover is not None, **w24,
        "model": args.model, "base_params": n_total_base, "trainable_params": n_train,
        "trainable_bytes": train_bytes, "full_theta_bytes": dense_bytes,
        "trainable_fraction": n_train / (n_total_base + n_train), "lora_rank": args.rank,
        "targets": targets, "init_sync": args.init_sync,
        "dense_bytes_est_per_step": dense_est, "lora_bytes_est_per_step": lora_est,
        "interconnect_truth": "training (FN: PEFT-only sync, MB-scale adapters)",
    }, env)
    if env.is_main:
        print(f"lora_ddp model={args.model} world={env.world} dtype={dtype} "
              f"trainable={n_train / 1e6:.2f}M ({train_bytes / 1e6:.1f} MB) of "
              f"{n_total_base / 1e6:.0f}M base ({dense_bytes / 1e9:.2f} GB); "
              f"sync ~{lora_est / 1e6:.1f} MB/step vs ~{dense_est / 1e6:.0f} MB full-FT "
              f"(x{dense_est / max(lora_est, 1):.0f} smaller)")
        if cover is not None:
            print(f"  W24: adapters synced every {args.sync_every} steps; cover {cover.ars_per_call} x "
                  f"{cover.msg_bytes / 1024:.0f} KiB AllReduce/step; train share of wire bytes "
                  f"~{w24['expected_train_wire_byte_fraction']:.1%}")

    stop = StopSignal(env, args.duration, args.max_steps)
    vocab = base.config.vocab_size
    ddp.train()
    dist.barrier()
    stop.t0 = t_start = time.time()
    step = 0
    tot = 0

    # cover AllReduces are logged by CoverTraffic (phase=decode_shape), not counted as grad sync
    phase_fn = lambda: None if (cover is not None and cover.in_cover) else "grad_sync_lora"  # noqa: E731
    with AllReduceTap(comm, phase_fn) as tap:
        while not stop.should_stop(step):
            comm.step = steps.step = step
            t_step = time.time()
            is_sync = (step + 1) % args.sync_every == 0
            ids = synthetic_tokens(args.batch_size, args.seq_len, vocab, env.device)
            with (contextlib.nullcontext() if is_sync else ddp.no_sync()):
                out = ddp(input_ids=ids, attention_mask=torch.ones_like(ids), labels=ids, use_cache=False)
                (out.loss / args.sync_every).backward()      # AllReduce fires only when not in no_sync
            if is_sync:
                opt.step()
                opt.zero_grad(set_to_none=True)
            if cover is not None:
                cover.emit()

            actual, count = tap.pop_step()
            tot += actual
            phase = ("train_sync" if is_sync else "decode_shape") if windowed else "grad_sync_lora"
            env.sync()
            steps.emit(t=t_step, t_end=time.time(), op="grad_sync_summary", phase=phase, is_sync_step=is_sync,
                       bytes_est=actual, dense_bytes_est=dense_est, n_collectives=count,
                       compression_ratio=(dense_est / actual) if actual else None,
                       loss=float(out.loss.detach()))
            step += 1
            if env.is_main and (step <= 3 or step % 20 == 0):
                print(f"  step {step}: loss={float(out.loss.detach()):.3f} "
                      f"sync={actual / 1e6:.2f} MB in {count} collective(s)")

    dist.barrier()
    if env.is_main:
        print(f"lora_ddp done: {step} steps in {time.time() - t_start:.1f}s, "
              f"total sync {tot / 1e6:.1f} MB (full-FT would be {dense_est * step / 1e6:.0f} MB)")
    comm.close()
    steps.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
