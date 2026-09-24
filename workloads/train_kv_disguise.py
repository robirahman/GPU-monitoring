#!/usr/bin/env python3
"""W18 -- KV-shaped gradient disguise (interconnect FN twin of W10).

Real data-parallel training whose gradient sync is re-cut so the wire looks like a
disaggregated-serving KV handoff (W10) instead of a dense AllReduce of |theta|.
Interconnect analogue of ICTD ``kv_disguise``; not an NVML-util trick.

Nothing is faked: every rank trains, gradients are genuinely averaged across all
ranks, and the optimizer steps. Only the *schedule* of the sync differs from DDP.

  --mode ddp   AllReduce in KV-sized blocks (``--block-mb``), optionally paced
               (``--gap-s``) and batched behind ``--accum-steps`` local micro-steps,
               so the fabric sees bursts of equal-size transfers with idle gaps
               (KV-like duty cycle) rather than one bucket stream per step.
  --mode p2p   No AllReduce op at all. Recursive-doubling ("butterfly") allreduce built
               from batched isend/irecv between paired ranks, in the same KV-sized
               blocks. Round 0 pairs rank r with r^1 -- the even/odd pairing that W10's
               ``--split evenodd --pattern pair`` uses -- so op mix, size distribution
               and pair graph look like KV traffic. Needs a power-of-two world.
               Cost: log2(N) * |g| per rank per sync (vs ~2|g| for ring): louder than
               DDP, on purpose.

Size schedule, two ways:
  * ``--block-mb`` (default): one uniform block size; use W10's ``--kv-mb /
    --chunks-per-handoff`` to make it match the *synthetic* W10 microbench. Hand constants.
  * ``--schedule-from <profile.json>``: block sizes AND inter-block gaps sampled from an
    empirical profile of a REAL disaggregated vLLM run (``scripts/build_kv_profile.py`` turns
    ``disagg_kv_vllm_*`` handoff logs into the profile). Blocks follow the empirical size
    distribution and burst structure (ordered walk from a seeded offset); the last block is
    truncated to fit |g|. ``--schedule-gap-scale`` / ``--schedule-max-gap-s`` scale/cap the
    real inter-arrival gaps so a sync does not stall for minutes. If the profile file is
    missing, falls back to ``--block-mb`` with a warning (smoke-safe) rather than failing hard.

Comm log: ground truth stays honest -- ``phase=grad_kv_block`` (ddp) /
``grad_sync_p2p`` (p2p) with ``mimics=kv_handoff``. A detector sees the wire; the label
here says what it really was. Same JSONL fields as W10 (``op``, ``bytes_est``,
``payload_bytes``, ``step``, ``microstep``, ...).

Launch::

    torchrun --nproc_per_node=8 workloads/train_kv_disguise.py --mode ddp --size 150m \\
        --block-mb 256 --accum-steps 4 --comm-log data/kv_disguise_ddp_comm.jsonl
    torchrun --nproc_per_node=8 workloads/train_kv_disguise.py --mode p2p --size 150m \\
        --block-mb 256 --comm-log data/kv_disguise_p2p_comm.jsonl
    python scripts/run_workload_multi_gpu.py --workload train_kv_disguise_p2p --num-gpus 8
    # calibrated to a real disagg run (after scripts/build_kv_profile.py):
    python scripts/run_workload_multi_gpu.py --workload train_kv_disguise_p2p_realprof --num-gpus 8

CPU smoke (no GPU; --verify cross-checks the P2P sum against dist.all_reduce)::

    torchrun --nproc_per_node=4 workloads/train_kv_disguise.py --backend gloo --mode p2p \\
        --size tiny --batch-size 2 --seq-len 32 --block-mb 0.25 --max-steps 3 --verify
"""
from __future__ import annotations

import argparse
import os
import time

import torch
import torch.distributed as dist

from kv_profile import load_profile, sample_schedule
from interconnect_common import (
    CausalLM,
    JsonlLog,
    MODEL_CONFIGS,
    StopSignal,
    add_common_args,
    autocast_ctx,
    chunked_allreduce,
    count_params,
    dist_cleanup,
    dist_setup,
    ensure_grads,
    flatten_into,
    lm_loss,
    make_optimizer,
    split_blocks,
    synthetic_tokens,
    tensor_bytes,
    unflatten_from,
    write_meta,
)

MIB = 1024 * 1024


def butterfly_allreduce_p2p(flat: torch.Tensor, recv_buf: torch.Tensor, block_elems: int,
                            env, comm: JsonlLog, gap_s: float, mimics: str = "kv_handoff",
                            sizes: list[int] | None = None, gaps: list[float] | None = None) -> int:
    """In-place SUM over all ranks via recursive doubling on batched isend/irecv.

    Round k exchanges the full (accumulated) buffer with partner rank^(1<<k), block by
    block. Pairwise addition is commutative and every rank follows the same tree, so all
    ranks end bit-identical. Returns blocks moved per round.
    """
    rounds = env.world.bit_length() - 1
    if sizes is None:
        sizes = split_blocks(flat.numel(), block_elems)
    item = flat.element_size()
    for k in range(rounds):
        partner = env.rank ^ (1 << k)
        off = 0
        for i, sz in enumerate(sizes):
            if gaps is not None and gaps[i] > 0:
                time.sleep(gaps[i])
            snd = flat[off:off + sz]
            rcv = recv_buf[:sz]
            t0 = time.time()
            reqs = dist.batch_isend_irecv([dist.P2POp(dist.isend, snd, partner),
                                           dist.P2POp(dist.irecv, rcv, partner)])
            for r in reqs:
                r.wait()
            env.sync()
            dur = time.time() - t0
            snd.add_(rcv)
            nbytes = sz * item
            common = dict(t=t0, dur_s=dur, phase="grad_sync_p2p", mimics=mimics,
                          bfly_round=k, block=i, n_blocks=len(sizes),
                          bytes_est=nbytes, payload_bytes=nbytes, microstep=comm.microstep)
            comm.emit(op="p2p_send", src_rank=env.rank, dst_rank=partner, **common)
            comm.emit(op="p2p_recv", src_rank=partner, dst_rank=env.rank, **common)
            off += sz
            if gaps is None and gap_s and i + 1 < len(sizes):
                time.sleep(gap_s)
    return len(sizes)


def main() -> None:
    ap = argparse.ArgumentParser(description="W18 KV-shaped gradient disguise",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--mode", choices=["ddp", "p2p"], default="ddp")
    ap.add_argument("--size", choices=list(MODEL_CONFIGS), default="150m")
    ap.add_argument("--batch-size", type=int, default=4, help="Per-rank micro-batch")
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--block-mb", type=float, default=256.0,
                    help="KV-block size in MiB (match W10 kv-mb / chunks-per-handoff)")
    ap.add_argument("--schedule-from", default="",
                    help="Empirical KV-handoff profile JSON from a real disagg run "
                         "(scripts/build_kv_profile.py); overrides --block-mb / --gap-s")
    ap.add_argument("--schedule-gap-scale", type=float, default=1.0)
    ap.add_argument("--schedule-max-gap-s", type=float, default=2.0)
    ap.add_argument("--accum-steps", type=int, default=1,
                    help="Local micro-steps per sync (sets the burst period / duty cycle)")
    ap.add_argument("--gap-s", type=float, default=0.0,
                    help="Sleep between blocks within one sync burst")
    ap.add_argument("--optimizer", choices=["adamw", "sgd", "sgd_momentum"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast on GPU")
    ap.add_argument("--verify", action="store_true",
                    help="Step 0: cross-check the disguised sync against dist.all_reduce "
                         "(adds one real AllReduce -- testing only)")
    args = ap.parse_args()

    env = dist_setup(args.backend)
    if args.mode == "p2p" and env.world & (env.world - 1):
        raise SystemExit(f"--mode p2p needs a power-of-two world (got {env.world})")

    torch.manual_seed(args.seed)                       # identical init on every rank
    model = CausalLM(args.size).to(env.device)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:                                   # one-time init broadcast (not steady state)
        dist.broadcast(p.data, src=0)
    opt = make_optimizer(args.optimizer, params, args.lr)
    torch.manual_seed(args.seed + 1 + env.rank)        # different data per rank

    n_params = count_params(model)
    flat = torch.zeros(n_params, dtype=params[0].dtype, device=env.device)
    item = flat.element_size()
    block_bytes = max(item, int(args.block_mb * MIB))
    block_elems = block_bytes // item
    sizes = split_blocks(n_params, block_elems)
    profile = None
    if args.schedule_from:
        if not os.path.exists(args.schedule_from):
            if env.is_main:
                print(
                    f"WARNING: --schedule-from {args.schedule_from} missing; "
                    f"falling back to hand schedule --block-mb={args.block_mb}. "
                    "Run disagg_kv_vllm_qwen7b + scripts/build_kv_profile.py for a real profile.",
                    flush=True,
                )
            args.schedule_from = ""
        else:
            profile = load_profile(args.schedule_from)
            max_block = max(block_elems, int(profile["size_bytes"]["max"]) // item)
            block_elems = min(n_params, max_block)
            sizes = split_blocks(n_params, block_elems)
    recv_buf = torch.empty(block_elems, dtype=flat.dtype, device=env.device) if args.mode == "p2p" else None
    rounds = env.world.bit_length() - 1
    sync_bytes = tensor_bytes(flat) * (rounds if args.mode == "p2p" else 1)

    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    write_meta(args.comm_log, {
        "workload": "train_kv_disguise", "wid": "W18", "mode": args.mode, "world_size": env.world,
        "backend": env.backend, "model": args.size, "params": n_params,
        "grad_bytes": tensor_bytes(flat), "block_bytes": block_bytes, "blocks_per_sync": len(sizes),
        "accum_steps": args.accum_steps, "gap_s": args.gap_s,
        "schedule_from": args.schedule_from or None,
        "schedule_source": (profile["source"] if profile else "hand constants (--block-mb)"),
        "schedule_provenance": (profile["provenance"] if profile else None),
        "profile_size_bytes": (profile["size_bytes"] if profile else None),
        "interconnect_truth": "training (FN: gradient sync disguised as KV handoff)",
    }, env)
    if env.is_main:
        print(f"train_kv_disguise mode={args.mode} world={env.world} backend={env.backend} "
              f"model={args.size} params={n_params / 1e6:.1f}M grads={tensor_bytes(flat) / MIB:.1f} MiB")
        if profile:
            sz = profile["size_bytes"]
            print(f"  schedule from {args.schedule_from}: {profile['n_handoffs']} real handoffs, size "
                  f"p10/p50/p90 = {sz['p10'] / MIB:.1f}/{sz['p50'] / MIB:.1f}/{sz['p90'] / MIB:.1f} MiB")
        else:
            print(f"  block={block_bytes / MIB:.2f} MiB x {len(sizes)}/sync, accum={args.accum_steps}, "
                  f"gap={args.gap_s}s, est {sync_bytes / MIB:.0f} MiB sent per rank per sync")

    stop = StopSignal(env, args.duration, args.max_steps)
    vocab = model.vocab_size
    model.train()
    dist.barrier()
    stop.t0 = t_start = time.time()
    step = 0
    verified = not args.verify

    while not stop.should_stop(step):
        comm.step = step
        t_step = time.time()
        loss_val = 0.0
        for micro in range(args.accum_steps):
            comm.microstep = micro
            ids = synthetic_tokens(args.batch_size, args.seq_len, vocab, env.device)
            with autocast_ctx(env, not args.no_amp):
                loss = lm_loss(model(ids), ids) / args.accum_steps
            loss.backward()
            loss_val += float(loss.detach())
        t_compute = time.time() - t_step

        grads = ensure_grads(params)
        flatten_into(grads, flat)
        ref = flat.clone() if not verified else None

        t_sync = time.time()
        sched_sizes = sched_gaps = None
        if profile is not None:            # same (seed, step) on every rank -> identical schedule
            byte_sizes, sched_gaps = sample_schedule(
                profile, tensor_bytes(flat), itemsize=item, seed=args.seed * 100003 + step,
                gap_scale=args.schedule_gap_scale, max_gap_s=args.schedule_max_gap_s)
            sched_sizes = [b // item for b in byte_sizes]
        if env.world > 1:
            if args.mode == "ddp":
                chunked_allreduce(flat, block_bytes, env, log=comm, phase="grad_kv_block",
                                  group_name="world", gap_s=args.gap_s, average=True,
                                  mimics="kv_handoff", microstep=args.accum_steps - 1,
                                  sizes=sched_sizes, gaps=sched_gaps)
            else:
                butterfly_allreduce_p2p(flat, recv_buf, block_elems, env, comm, args.gap_s,
                                        sizes=sched_sizes, gaps=sched_gaps)
                flat.div_(env.world)
        sync_s = time.time() - t_sync

        if ref is not None:
            dist.all_reduce(ref)
            ref.div_(env.world)
            err = float((flat - ref).abs().max())
            if env.is_main:
                print(f"  verify: max|disguised - all_reduce| = {err:.3e}")
            if err > 1e-4:
                raise SystemExit(f"verify FAILED: disguised sync deviates by {err}")
            verified = True

        unflatten_from(flat, grads)
        opt.step()
        opt.zero_grad(set_to_none=True)
        step += 1
        if env.is_main and (step <= 3 or step % 10 == 0):
            print(f"  step {step}: loss={loss_val:.3f} compute={t_compute:.2f}s sync={sync_s:.2f}s "
                  f"({sync_bytes / max(sync_s, 1e-9) / 1e9:.2f} GB/s)")

    dist.barrier()
    if env.is_main:
        print(f"train_kv_disguise done: {step} steps in {time.time() - t_start:.1f}s")
    comm.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
