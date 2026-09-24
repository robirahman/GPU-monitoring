#!/usr/bin/env python3
"""W19 -- Hybrid duty-cycle on the wire (interconnect FN).

Real training whose fabric traffic is *mostly* what TP decode serving looks like --
small, latency-bound AllReduces every step, sized like W08 (Qwen2.5-7B decode-heavy)
-- with a full-|theta| gradient sync only every H steps. A detector averaging bytes
over a window, or keying on "small messages => inference", is pulled toward
"inference" between syncs and only sees training in the rare burst.

Unlike Rahman's ``adversarial_F_interleave`` / K / L (compute/mode switching judged by
NVML), this shapes the *collectives*: the cover traffic is real NCCL AllReduce on a
TP-like subgroup, and the rare sync moves genuine gradients.

**W24 (temporal serve-cover) is this workload at large H**: ``train_serve_cover_H20`` (sync on 1/20 = 5%
of steps -> 95% of steps look like decode serving on the wire) and ``_H100`` (99/1). The dilution is on
the WIRE only: local forward/backward runs every step, so an NVML layer still sees training compute.
The meta sidecar records ``train_step_fraction`` / ``expected_train_wire_byte_fraction`` so the "95%"
claim is checkable. SPAR must score **train-active windows** (``phase=train_sync`` rows of
``<stem>.steps.jsonl``) and run-level statistics separately -- pooled window accuracy is dominated by
the 95% cover and will lie.

Per step (all ranks in lock-step)::

    local forward/backward on synthetic tokens            (no comm)
    decode-shaped cover: layers * ars_per_layer AllReduces of tokens*hidden*dtype
                         on the TP subgroup, ``--cover-tokens`` times per step
    every H-th step:  train_sync
        --sync-mode grad     H micro-batches of local grad accumulation, then one
                             bucketed AllReduce of the whole gradient (|theta|) + step
        --sync-mode diloco   local AdamW every step, DiLoCo outer step (diloco_core:
                             AllReduce of pseudo-gradients) every H steps

``--cover-tp N`` makes TP groups of N consecutive ranks (default = whole world). With
N = GPUs per node the cover stays intra-node and the only cross-node traffic is the rare
sync -- the sharpest version of the disguise.

Window truth (critical for scoring): every step is written to ``<stem>.steps.jsonl``
with ``phase = decode_shape | train_sync``, ``t``, ``t_end``, ``is_sync_step``. The
comm log (``--comm-log``) has one aggregated ``phase=decode_shape`` line per step
(``count`` collectives of ``msg_bytes`` each; ``--log-decode full`` for one line per
collective) and one ``phase=train_sync`` line per gradient bucket (DiLoCo mode: the
controller's own ``outer_sync`` events, same file).

Launch::

    torchrun --nproc_per_node=8 workloads/hybrid_wire.py --size 1b --sync-every 50 \\
        --duration 300 --comm-log data/hybrid_wire_H50_comm.jsonl
    python scripts/run_workload_multi_gpu.py --workload train_hybrid_wire_H50 --num-gpus 8

CPU smoke::

    torchrun --nproc_per_node=4 workloads/hybrid_wire.py --backend gloo --size tiny \\
        --batch-size 2 --seq-len 32 --sync-every 5 --max-steps 12 --cover-tp 2 \\
        --decode-hidden 64 --decode-layers 3 --comm-log /tmp/hybrid.jsonl
    (add --sync-mode diloco --reduce-dtype fp32 for the DiLoCo variant)
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist

from diloco_core import CommLogger, DiLoCoController
from interconnect_common import (
    CausalLM,
    JsonlLog,
    MODEL_CONFIGS,
    StopSignal,
    add_common_args,
    add_cover_args,
    ar_wire_bytes,
    autocast_ctx,
    chunked_allreduce,
    count_params,
    cover_from_args,
    dist_cleanup,
    dist_setup,
    duty_fractions,
    ensure_grads,
    flatten_into,
    lm_loss,
    make_optimizer,
    sibling_path,
    synthetic_tokens,
    unflatten_from,
    write_meta,
)

MIB = 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser(description="W19 hybrid duty-cycle on the wire",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--size", choices=list(MODEL_CONFIGS), default="150m")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--sync-every", "-H", type=int, default=50, dest="sync_every",
                    help="H: steps between full-|theta| syncs")
    ap.add_argument("--sync-mode", choices=["grad", "diloco"], default="grad")
    ap.add_argument("--bucket-mb", type=float, default=256.0, help="grad-sync AllReduce bucket")
    ap.add_argument("--optimizer", choices=["adamw", "sgd", "sgd_momentum"], default="adamw")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--no-amp", action="store_true")
    add_cover_args(ap)     # decode-shaped cover (defaults ~ Qwen2.5-7B decode, the W08 scale)
    # diloco outer step
    ap.add_argument("--outer-lr", type=float, default=0.7)
    ap.add_argument("--outer-momentum", type=float, default=0.9)
    ap.add_argument("--reduce-dtype", choices=["fp32", "fp16", "bf16", "fp8"], default="fp16")
    args = ap.parse_args()
    if args.sync_every < 1:
        raise SystemExit("--sync-every must be >= 1")

    env = dist_setup(args.backend)
    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    cover = cover_from_args(env, comm, args)

    torch.manual_seed(args.seed)
    model = CausalLM(args.size).to(env.device)
    params = [p for p in model.parameters() if p.requires_grad]
    for p in params:
        dist.broadcast(p.data, src=0)
    opt = make_optimizer(args.optimizer, params, args.lr)
    torch.manual_seed(args.seed + 1 + env.rank)

    n_params = count_params(model)
    theta_bytes = n_params * params[0].element_size()
    sync_wire = ar_wire_bytes(theta_bytes, env.world)

    steps = JsonlLog(sibling_path(args.comm_log, ".steps") if args.comm_log else "", env)
    flat = None
    ctrl = None
    if args.sync_mode == "grad":
        flat = torch.zeros(n_params, dtype=params[0].dtype, device=env.device)
    else:
        dl = CommLogger(path=args.comm_log or None, rank=env.rank)
        dl.open()
        ctrl = DiLoCoController(model, opt, local_steps=args.sync_every, outer_lr=args.outer_lr,
                                outer_momentum=args.outer_momentum, reduce_dtype=args.reduce_dtype,
                                bucket_cap_mb=args.bucket_mb, logger=dl)

    write_meta(args.comm_log, {
        "workload": "hybrid_wire", "wid": "W19", "world_size": env.world, "backend": env.backend,
        "model": args.size, "params": n_params, "theta_bytes": theta_bytes,
        "sync_every": args.sync_every, "sync_mode": args.sync_mode,
        "sync_wire_bytes_est": sync_wire, **cover.meta(),
        **duty_fractions(args.sync_every, sync_wire, cover.wire_bytes_per_call),
        "interconnect_truth": "training (FN: decode-shaped small AllReduces + rare |theta| sync)",
    }, env)
    if env.is_main:
        print(f"hybrid_wire model={args.size} params={n_params / 1e6:.1f}M world={env.world} "
              f"H={args.sync_every} mode={args.sync_mode} cover_tp={cover.tp}")
        print(f"  cover: {cover.ars_per_call} x {cover.msg_bytes / 1024:.0f} KiB AllReduce/step "
              f"(~{cover.wire_bytes_per_call / 1e6:.2f} MB); sync: ~{sync_wire / MIB:.0f} MiB every "
              f"{args.sync_every} steps -> {1 - 1 / args.sync_every:.1%} of steps infer-shaped")

    stop = StopSignal(env, args.duration, args.max_steps)
    model.train()
    dist.barrier()
    stop.t0 = t_start = time.time()
    step = 0
    n_syncs = 0
    inner_scale = 1.0 / args.sync_every if args.sync_mode == "grad" else 1.0

    while not stop.should_stop(step):
        comm.step = steps.step = step
        t0 = time.time()
        is_sync = (step + 1) % args.sync_every == 0

        ids = synthetic_tokens(args.batch_size, args.seq_len, model.vocab_size, env.device)
        with autocast_ctx(env, not args.no_amp):
            loss = lm_loss(model(ids), ids)
        (loss * inner_scale).backward()
        if args.sync_mode == "diloco":
            opt.step()
            opt.zero_grad(set_to_none=True)

        cover.emit()          # decode-shaped cover traffic on the TP-like subgroup

        if args.sync_mode == "grad":
            if is_sync:
                grads = ensure_grads(params)
                flatten_into(grads, flat)
                chunked_allreduce(flat, int(args.bucket_mb * MIB), env, log=comm, phase="train_sync",
                                  group_name="world", average=True, sync_index=n_syncs)
                unflatten_from(flat, grads)
                opt.step()
                opt.zero_grad(set_to_none=True)
        else:
            ctrl.after_inner_step()      # outer sync (and its CommEvents) every H inner steps
        if is_sync:
            n_syncs += 1

        env.sync()
        phase = "train_sync" if is_sync else "decode_shape"
        steps.emit(t=t0, t_end=time.time(), phase=phase, is_sync_step=is_sync,
                   loss=float(loss.detach()), sync_index=(n_syncs - 1) if is_sync else None)
        step += 1
        if env.is_main and (is_sync or step <= 2 or step % 20 == 0):
            print(f"  step {step}: phase={phase} loss={float(loss.detach()):.3f} "
                  f"{(time.time() - t0) * 1000:.0f} ms")

    dist.barrier()
    if env.is_main:
        print(f"hybrid_wire done: {step} steps, {n_syncs} full syncs in {time.time() - t_start:.1f}s")
    if ctrl is not None:
        dl.close()
    comm.close()
    steps.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
