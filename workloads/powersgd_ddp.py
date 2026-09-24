#!/usr/bin/env python3
"""W17 -- PowerSGD / Top-k DDP training: synced payload << |theta| (interconnect FN).

Ordinary data-parallel training, but gradients are compressed before they hit the
fabric. With PowerSGD (PyTorch's DDP comm hook) AllReduce volume per step drops by
roughly ``min(m, n) / (2 * rank)`` on the big matrices; with Top-k it drops to
``ratio`` of the elements. The model still trains; a detector keyed on
"|theta|-sized AllReduce every step" sees far less.

  --hook powersgd          torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook
  --hook batched_powersgd  batched_powerSGD_hook (one flat P/Q per bucket)
  --hook topk              Top-k sparsification with error feedback (residual carried to
                           the next step). Each rank sends its top ``--topk-ratio`` of
                           gradient elements (int32 index + fp32 value, packed in ONE
                           all_gather per bucket) and every rank scatter-adds the result.
                           No AllReduce op in the sparse phase -- the op mix changes too.
                           Unlike PowerSGD its bytes grow with world size (all_gather
                           sends (N-1) x payload per rank), so it is a small-N disguise.
  --hook allreduce         dense baseline through the same Python path (byte-identical
                           to stock DDP) -- run it to get the "before" number

  --powersgd-rank R            matrix_approximation_rank (default 4)
  --start-powerSGD-iter N      iterations of *vanilla* AllReduce before compression kicks
                               in (PyTorch default 1000; 10 here so short runs compress).
                               Log ``phase`` is ``grad_sync_dense_warmup`` until then,
                               ``grad_sync_powersgd`` after -- use it for window truth.
  --min-compression-rate X     tensors that can't be compressed >= X are sent dense
  --topk-ratio R               fraction of elements kept per bucket (default 0.01; 1.0 == dense)
  --topk-start-iter N          dense warm-up steps before Top-k starts (phase
                               ``grad_sync_dense_warmup`` -> ``grad_sync_topk``)

Models: ``resnet18`` (CIFAR-style, synthetic 3x32x32 images, no dataset download) or a
sized CausalLM (``tiny`` | ``150m`` | ``1b``) on synthetic tokens.

Comm log (JSONL, rank0): one line per Python-level AllReduce, exact sizes
(PowerSGD sends P then Q per bucket, plus dense stragglers) -- ``op=all_reduce``,
``bytes_est``, ``payload_bytes``, ``phase``, ``step``; Top-k logs ``op=all_gather``
(``k``, ``numel``, ``ratio``, ``bucket_index``). A second file
``<stem>.steps.jsonl`` has one line per step: ``bytes_est`` (actual estimated wire
bytes), ``dense_bytes_est`` (what plain DDP would send) and ``compression_ratio``.

Launch::

    torchrun --nproc_per_node=8 workloads/powersgd_ddp.py --model resnet18 \\
        --powersgd-rank 4 --duration 300 --comm-log data/powersgd_resnet18_comm.jsonl
    torchrun --nproc_per_node=8 workloads/powersgd_ddp.py --model 150m --hook topk \\
        --topk-ratio 0.01 --comm-log data/topk_150m_comm.jsonl
    python scripts/run_workload_multi_gpu.py --workload ddp_powersgd_150m --num-gpus 8

CPU smoke::

    torchrun --nproc_per_node=2 workloads/powersgd_ddp.py --backend gloo --model tiny \\
        --batch-size 2 --seq-len 32 --start-powerSGD-iter 3 --max-steps 8 --comm-log /tmp/psgd.jsonl
"""
from __future__ import annotations

import argparse
import threading
import time

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP

from interconnect_common import (
    AllReduceTap,
    CausalLM,
    JsonlLog,
    MODEL_CONFIGS,
    StopSignal,
    add_common_args,
    ar_wire_bytes,
    autocast_ctx,
    count_params,
    dist_cleanup,
    dist_setup,
    lm_loss,
    make_optimizer,
    sibling_path,
    synthetic_tokens,
    tensor_bytes,
    write_meta,
)


def build_resnet18(num_classes: int = 10) -> nn.Module:
    import torchvision   # lazy: LM-only runs don't need torchvision

    m = torchvision.models.resnet18(weights=None, num_classes=num_classes)
    m.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)   # CIFAR stem
    m.maxpool = nn.Identity()
    return m


class TopKState:
    """State for the Top-k sparsified-gradient DDP hook (with error feedback).

    Per bucket: g += residual; keep the k largest-|g| elements; residual = g minus the
    kept part; all_gather the packed (index, value) pairs; every rank scatter-adds all
    ranks' pairs and divides by world size. Runs on DDP's autograd-thread hook path,
    so counters are locked. ``iter`` is bumped by the last bucket of each iteration
    (same convention as PowerSGDState).
    """

    def __init__(self, ratio: float, start_iter: int, log: JsonlLog,
                 error_feedback: bool = True) -> None:
        if not 0.0 < ratio <= 1.0:
            raise SystemExit("--topk-ratio must be in (0, 1]")
        self.ratio, self.start_iter, self.log = ratio, start_iter, log
        self.error_feedback = error_feedback
        self.iter = 0
        self.phase = "grad_sync_dense_warmup"
        self.memory: dict[int, torch.Tensor] = {}   # error-feedback residual per bucket
        self._lock = threading.Lock()
        self._bytes = self._count = 0

    def pop_step(self) -> tuple[int, int]:
        with self._lock:
            out = (self._bytes, self._count)
            self._bytes = self._count = 0
        return out

    def _account(self, nbytes: int) -> None:
        with self._lock:
            self._bytes += nbytes
            self._count += 1


def topk_hook(state: TopKState, bucket):
    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks

    warm = state.iter < state.start_iter
    state.phase = "grad_sync_dense_warmup" if warm else "grad_sync_topk"
    last = bool(bucket.is_last())
    if warm:
        if last:
            state.iter += 1
        return default_hooks.allreduce_hook(None, bucket)   # dense; logged by AllReduceTap

    buf = bucket.buffer()
    world = dist.get_world_size()
    n = buf.numel()
    k = max(1, int(n * state.ratio))
    key = bucket.index()
    if state.error_feedback:
        mem = state.memory.get(key)
        if mem is not None and mem.numel() == n:    # DDP rebuilds buckets once after iteration 0
            buf.add_(mem)                           # error feedback
    idx = buf.abs().topk(k, sorted=False).indices
    vals = buf[idx].float()
    if state.error_feedback:
        resid = buf.clone()
        resid[idx] = 0
        state.memory[key] = resid

    packed = torch.cat([idx.to(torch.int32).view(torch.float32), vals])       # 2k x 4 B
    gathered = torch.empty(world * 2 * k, dtype=torch.float32, device=buf.device)
    work = dist.all_gather_into_tensor(gathered, packed, async_op=True)
    payload = tensor_bytes(packed)
    wire = (world - 1) * payload             # ring all_gather: each rank forwards N-1 chunks
    state.log.emit(op="all_gather", phase=state.phase, group="dp", group_size=world,
                   bytes_est=wire, payload_bytes=payload, k=k, numel=n, ratio=k / n,
                   bucket_index=key, async_op=True)
    state._account(wire)
    if last:
        state.iter += 1

    def finish(_fut):
        g = gathered.view(world, 2 * k)
        all_idx = g[:, :k].contiguous().view(torch.int32).long().reshape(-1)
        out = torch.zeros(n, dtype=torch.float32, device=buf.device)
        out.index_add_(0, all_idx, g[:, k:].reshape(-1))
        return out.div_(world).to(buf.dtype)

    return work.get_future().then(finish)


def main() -> None:
    ap = argparse.ArgumentParser(description="W17 PowerSGD / Top-k DDP",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--model", choices=["resnet18"] + list(MODEL_CONFIGS), default="resnet18")
    ap.add_argument("--hook", choices=["powersgd", "batched_powersgd", "topk", "allreduce"],
                    default="powersgd")
    ap.add_argument("--powersgd-rank", type=int, default=4)
    ap.add_argument("--start-powerSGD-iter", type=int, default=10)
    ap.add_argument("--min-compression-rate", type=float, default=2.0)
    ap.add_argument("--topk-ratio", type=float, default=0.01)
    ap.add_argument("--topk-start-iter", type=int, default=10)
    ap.add_argument("--no-error-feedback", action="store_true")
    ap.add_argument("--no-warm-start", action="store_true")
    ap.add_argument("--batch-size", type=int, default=None,
                    help="Per-rank batch (default 128 for resnet18, 4 for LMs)")
    ap.add_argument("--seq-len", type=int, default=512, help="LM sequence length")
    ap.add_argument("--optimizer", choices=["adamw", "sgd", "sgd_momentum"], default="sgd_momentum")
    ap.add_argument("--lr", type=float, default=0.01)
    ap.add_argument("--no-amp", action="store_true", help="Disable bf16 autocast (LM on GPU)")
    args = ap.parse_args()

    env = dist_setup(args.backend)
    is_lm = args.model != "resnet18"
    batch = args.batch_size or (4 if is_lm else 128)

    torch.manual_seed(args.seed)
    model = (CausalLM(args.model) if is_lm else build_resnet18()).to(env.device)
    ddp = DDP(model, device_ids=[env.local_rank] if env.device.type == "cuda" else None)
    torch.manual_seed(args.seed + 1 + env.rank)

    n_params = count_params(model)
    dense_est = ar_wire_bytes(n_params * 4, env.world)     # fp32 grads, ring estimate

    from torch.distributed.algorithms.ddp_comm_hooks import default_hooks
    from torch.distributed.algorithms.ddp_comm_hooks import powerSGD_hook as psgd

    state = None
    topk = None
    phase_now = ["grad_sync_dense"]
    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    if args.hook == "allreduce":
        ddp.register_comm_hook(None, default_hooks.allreduce_hook)
    elif args.hook == "topk":
        topk = TopKState(args.topk_ratio, args.topk_start_iter, comm,
                         error_feedback=not args.no_error_feedback)
        ddp.register_comm_hook(topk, topk_hook)
    else:
        state = psgd.PowerSGDState(
            process_group=None,
            matrix_approximation_rank=args.powersgd_rank,
            start_powerSGD_iter=args.start_powerSGD_iter,
            min_compression_rate=args.min_compression_rate,
            use_error_feedback=not args.no_error_feedback,
            warm_start=not args.no_warm_start,
        )
        fn = psgd.powerSGD_hook if args.hook == "powersgd" else psgd.batched_powerSGD_hook

        def hook(st, bucket):
            # state.iter is bumped *inside* fn on the last bucket, before its collective
            # is issued -- record the phase first so that bucket isn't mislabeled.
            phase_now[0] = ("grad_sync_dense_warmup" if st.iter < args.start_powerSGD_iter
                            else "grad_sync_powersgd")
            return fn(st, bucket)

        ddp.register_comm_hook(state, hook)

    def phase() -> str:
        return topk.phase if topk is not None else phase_now[0]

    opt = make_optimizer(args.optimizer, ddp.parameters(), args.lr)
    steps = JsonlLog(sibling_path(args.comm_log, ".steps") if args.comm_log else "", env)
    write_meta(args.comm_log, {
        "workload": "powersgd_ddp", "wid": "W17", "world_size": env.world, "backend": env.backend,
        "model": args.model, "params": n_params, "dense_bytes_est_per_step": dense_est,
        "hook": args.hook, "powersgd_rank": args.powersgd_rank,
        "topk_ratio": args.topk_ratio if args.hook == "topk" else None,
        "topk_start_iter": args.topk_start_iter if args.hook == "topk" else None,
        "start_powerSGD_iter": args.start_powerSGD_iter,
        "min_compression_rate": args.min_compression_rate,
        "interconnect_truth": "training (FN: compressed gradient sync, payload << |theta|)",
    }, env)
    if env.is_main:
        print(f"powersgd_ddp model={args.model} params={n_params / 1e6:.1f}M world={env.world} "
              f"hook={args.hook} " + (f"ratio={args.topk_ratio} start_iter={args.topk_start_iter}"
                                      if args.hook == "topk" else
                                      f"rank={args.powersgd_rank} start_iter={args.start_powerSGD_iter}")
              + f" dense~{dense_est / 1e6:.1f} MB/step")

    stop = StopSignal(env, args.duration, args.max_steps)
    ddp.train()
    step = 0
    t_start = time.time()
    tot_actual = tot_dense = 0
    dist.barrier()
    stop.t0 = time.time()

    with AllReduceTap(comm, phase) as tap:
        while not stop.should_stop(step):
            comm.step = steps.step = step
            if is_lm:
                ids = synthetic_tokens(batch, args.seq_len, model.vocab_size, env.device)
                with autocast_ctx(env, not args.no_amp):
                    loss = lm_loss(ddp(ids), ids)
            else:
                x = torch.randn(batch, 3, 32, 32, device=env.device)
                y = torch.randint(0, 10, (batch,), device=env.device)
                loss = F.cross_entropy(ddp(x), y)
            opt.zero_grad(set_to_none=True)
            loss.backward()                  # hook futures are finalized before this returns
            opt.step()

            actual, count = tap.pop_step()
            if topk is not None:
                b, c = topk.pop_step()
                actual, count = actual + b, count + c
            tot_actual += actual
            tot_dense += dense_est
            steps.emit(op="grad_sync_summary", phase=phase(), bytes_est=actual,
                       dense_bytes_est=dense_est, n_collectives=count,
                       compression_ratio=(dense_est / actual) if actual else None,
                       loss=float(loss.detach()))
            step += 1
            if env.is_main and (step <= 3 or step % 20 == 0):
                print(f"  step {step}: loss={float(loss.detach()):.3f} phase={phase()} "
                      f"sync={actual / 1e6:.2f} MB (dense {dense_est / 1e6:.2f} MB, "
                      f"x{(dense_est / actual) if actual else float('inf'):.1f}) collectives={count}")

    dist.barrier()
    if env.is_main:
        stats = ""
        if state is not None:
            rate, before, after = state.compression_stats()
            stats = f", PowerSGD stats: rate={rate:.1f}x numel {before}->{after}"
        print(f"powersgd_ddp done: {step} steps in {time.time() - t_start:.1f}s; "
              f"total sync {tot_actual / 1e6:.1f} MB vs dense {tot_dense / 1e6:.1f} MB"
              f" (x{(tot_dense / tot_actual) if tot_actual else float('inf'):.1f}){stats}")
    comm.close()
    steps.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
