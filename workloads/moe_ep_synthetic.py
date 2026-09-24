#!/usr/bin/env python3
"""Synthetic MoE expert-parallel inference fingerprint (SPAR M2).

**MICROBENCH ONLY -- not SPAR primary corpus.** The primary W12 / W20 workloads are real vLLM serving
(``llm_moe_serving.py``: ``moe_serve_*`` and ``moe_serve_*_fake_ar``).

Not a real model — reproduces the *communication shape* of decode-experts:
  per layer:  all_to_all (dispatch) → local expert GEMM → all_to_all (combine)

Optional rare AllReduce mimics hybrid duty-cycle / fake-train pollution.

Launch::

    torchrun --nproc_per_node=8 workloads/moe_ep_synthetic.py \\
        --duration 300 --layers 32 --hidden 4096 --experts-per-rank 4 \\
        --comm-log data/moe_ep_comm.jsonl

STEP JSONL (rank0): one line per collective for interconnect detectors.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import torch
import torch.distributed as dist


def _emit(fh, **kw) -> None:
    if fh is None:
        return
    fh.write(json.dumps(kw, separators=(",", ":")) + "\n")


def main() -> None:
    ap = argparse.ArgumentParser(description="Synthetic MoE EP AllToAll workload")
    ap.add_argument("--duration", type=float, default=300.0)
    ap.add_argument("--layers", type=int, default=32,
                    help="Transformer layers per forward (A2A pairs per token step).")
    ap.add_argument("--hidden", type=int, default=4096)
    ap.add_argument("--tokens-per-step", type=int, default=8,
                    help="Tokens per rank per step (batch-ish decode).")
    ap.add_argument("--experts-per-rank", type=int, default=4,
                    help="Local expert FFN count; GEMM width = hidden.")
    ap.add_argument("--ffn-mult", type=int, default=4,
                    help="Expert intermediate = hidden * ffn_mult.")
    ap.add_argument("--fake-ar-every", type=int, default=0,
                    help="If >0, AllReduce a grad-sized buffer every N steps (FP noise).")
    ap.add_argument("--fake-ar-mb", type=float, default=64.0,
                    help="Fake AllReduce payload size in MiB.")
    ap.add_argument("--comm-log", type=str, default="",
                    help="Rank0 JSONL path for STEP collective events.")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world = dist.get_world_size()
    torch.cuda.set_device(int(os.environ.get("LOCAL_RANK", rank % torch.cuda.device_count())))
    device = torch.device("cuda")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    h = args.hidden
    t = args.tokens_per_step
    inter = h * args.ffn_mult

    # Local experts: (E, h, inter) and (E, inter, h) — decode-expert FFN stand-in.
    E = args.experts_per_rank
    w1 = torch.randn(E, h, inter, device=device, dtype=dtype)
    w2 = torch.randn(E, inter, h, device=device, dtype=dtype)

    # A2A buffers: token-major (t, h); equal-split requires t % world == 0.
    if t % world != 0:
        raise SystemExit(
            f"--tokens-per-step={t} must be divisible by world_size={world}"
        )
    send = torch.randn(t, h, device=device, dtype=dtype)
    recv = torch.empty_like(send)

    item = send.element_size()
    fake_ar = None
    if args.fake_ar_every > 0:
        n_elem = int(args.fake_ar_mb * 1024 * 1024 / item)
        fake_ar = torch.zeros(n_elem, device=device, dtype=dtype)

    log_fh = None
    if args.comm_log and rank == 0:
        os.makedirs(os.path.dirname(args.comm_log) or ".", exist_ok=True)
        log_fh = open(args.comm_log, "a", buffering=1)

    if rank == 0:
        print(
            f"moe_ep_synthetic world={world} layers={args.layers} hidden={h} "
            f"tokens/step={t} experts/rank={E} dtype={args.dtype} "
            f"fake_ar_every={args.fake_ar_every}"
        )

    a2a_bytes = t * h * item  # total payload leaving this rank per A2A

    dist.barrier()
    t0 = time.time()
    step = 0
    while time.time() - t0 < args.duration:
        for layer in range(args.layers):
            # Dispatch
            dist.all_to_all_single(recv, send)
            _emit(
                log_fh,
                t=time.time(),
                step=step,
                layer=layer,
                op="all_to_all",
                phase="dispatch",
                bytes_est=a2a_bytes,
                world_size=world,
            )

            # Local expert compute (all experts see a slice of recv — shape stand-in)
            # recv: (t, h) → treat as batch; cycle experts
            x = recv
            e = layer % E
            # (t, h) @ (h, inter) @ (inter, h)
            y = x @ w1[e]
            y = torch.nn.functional.gelu(y)
            y = y @ w2[e]
            send = y  # combine input

            # Combine
            dist.all_to_all_single(recv, send)
            _emit(
                log_fh,
                t=time.time(),
                step=step,
                layer=layer,
                op="all_to_all",
                phase="combine",
                bytes_est=a2a_bytes,
                world_size=world,
            )
            send = recv

        if args.fake_ar_every > 0 and step > 0 and step % args.fake_ar_every == 0:
            dist.all_reduce(fake_ar, op=dist.ReduceOp.SUM)
            _emit(
                log_fh,
                t=time.time(),
                step=step,
                layer=-1,
                op="all_reduce",
                phase="fake_train",
                bytes_est=int(args.fake_ar_mb * 1024 * 1024),
                world_size=world,
            )

        step += 1
        if rank == 0 and step % 10 == 0:
            elapsed = time.time() - t0
            print(f"  step {step}: {elapsed:.0f}s, "
                  f"{step * args.layers * 2 / elapsed:.0f} A2A/s")

    dist.barrier()
    if rank == 0:
        elapsed = time.time() - t0
        print(f"moe_ep_synthetic done: {step} steps, {elapsed:.1f}s, "
              f"{step * args.layers * 2} A2A ops")
    if log_fh is not None:
        log_fh.close()
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
