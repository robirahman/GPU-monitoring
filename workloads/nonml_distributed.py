#!/usr/bin/env python3
"""HONEST non-ML controls: distributed work with no model anywhere.

Two modes, each isolating one candidate explanation for why the `filler_only`
cell scores p_train 0.44-0.56 despite never training, when the honest H100
non-ML corpus scores 0.001.

  --mode hpc   THE DECISIVE ONE. Byte-for-byte the same as filler_only except the
               sharded 1.3B model is replaced by a DUMMY allocation of the same
               size: same filler kernels, same slot length, same deadline
               collective cadence, same resident GB. So:
                 ~0.49  -> the tell is not ML-specific at all, it is "large
                           allocation + busy GPU", and we have a SPECIFICITY
                           problem rather than an anti-evasion result.
                 ~0.001 -> something about the model specifically (parameter
                           layout, allocator pattern, access shape) is the tell,
                           and the anti-evasion claim stands.
  --mode comm  Heavy cross-rank all_reduce, no model, negligible compute. The
               corpus's non-ML comm workloads (nvlink_bandwidth 0.0006,
               nvlink_latency 0.0015) are single-node bandwidth tests; there is
               nothing multi-node and collective-heavy in it. This is also the
               class most likely to produce genuine false positives in
               deployment (a distributed HPC solver), so it is worth having in
               the benign corpus on its own merits.

LABELLING: neither label contains "train" or "infer", so threeway_label() routes
both to `other`, which is correct -- there is no model and no gradient anywhere.
Neither matches DISGUISED_FAMILIES, so both legitimately join the honest corpus.

Note on the deadline: CollectiveDeadline uses check_every=5, so its collective is
one 4-byte all_reduce per five iterations (~1 per 2.5 s at the default slot), not
one per iteration. `--mode hpc` inherits exactly that, which is the point.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/nonml_distributed.py \
        --mode hpc --alloc-gb 1.4 --duration 600
    torchrun --nnodes=2 --nproc-per-node=8 workloads/nonml_distributed.py \
        --mode comm --comm-mb 256 --duration 600
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    # Same kernels as the disguise, deliberately: a control that ran DIFFERENT
    # kernels could not attribute the score gap to the harness.
    from workloads.adversarial_multinode import NonMLFiller
except ImportError:  # launched as `torchrun workloads/nonml_distributed.py`
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown
    from adversarial_multinode import NonMLFiller


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=["hpc", "comm"])
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--slot-sec", type=float, default=0.5,
                    help="hpc: wall time per filler slot. Must match the "
                         "filler slot length used by the cell being compared.")
    ap.add_argument("--alloc-gb", type=float, default=0.0,
                    help="hpc: dummy resident allocation standing in for the "
                         "sharded model (measured: 1.4 GB at S16, 1.8 at S8, "
                         "10.7 at S1).")
    ap.add_argument("--comm-mb", type=float, default=256.0,
                    help="comm: all_reduce buffer size per iteration")
    ap.add_argument("--filler-mode", choices=("mixed", "hash", "fft"),
                    default="mixed",
                    help="kernel mix for the honest non-ML profile")
    ap.add_argument("--duty", type=float, default=1.0,
                    help="hpc: fraction of each slot spent computing, the rest "
                         "idle. duty<1 gives the partial-utilisation / low-power "
                         "profile that the honest `other` class lacks -- real "
                         "clusters run half-idle HPC jobs and the detector has "
                         "never seen one.")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Non-ML distributed (HONEST control): mode={args.mode}, "
              f"{world_size} GPUs")
        print(f"  No model, no gradients, no optimizer anywhere")
        if args.mode == "hpc":
            print(f"  Dummy resident alloc: {args.alloc_gb} GB, "
                  f"filler slot {args.slot_sec}s, duty {args.duty:.0%}")
        else:
            print(f"  all_reduce buffer: {args.comm_mb} MB/iter")
        print(f"  Duration: {args.duration}s")

    filler = None
    dummy = None
    comm_buf = None
    if args.mode == "hpc":
        filler = NonMLFiller(device, mode=args.filler_mode)
        if args.alloc_gb > 0:
            # Held for the whole run, like a model would be. float32 = 4 B.
            n = int(args.alloc_gb * 1e9 / 4)
            dummy = torch.empty(n, dtype=torch.float32, device=device)
            dummy.fill_(1.0)
    else:
        comm_buf = torch.ones(int(args.comm_mb * 1e6 / 4), dtype=torch.float32,
                              device=device)

    if rank == 0:
        print(f"  Post-init GPU memory: "
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB")

    deadline = CollectiveDeadline(args.duration, device)
    wall_start = time.time()
    step = 0
    filler_iters = 0
    allreduces = 0
    last_print = wall_start

    while not deadline.expired():
        if args.mode == "hpc":
            busy = args.slot_sec * max(0.0, min(1.0, args.duty))
            filler_iters += filler.run_for(busy)
            if busy < args.slot_sec:
                time.sleep(args.slot_sec - busy)
        else:
            # All-reduce for a whole slot, then emit ONE step line. The first
            # version emitted one line per all_reduce, and at world_size=1 the
            # collective is a local no-op -- so s1_nonml_comm span the loop at
            # interpreter speed and wrote a 1.44 GB log, measuring Python rather
            # than communication. Slot-bounding fixes both.
            # Synchronise EVERY iteration, not once at the end of the slot.
            # dist.all_reduce only ENQUEUES, so a slot that loops on wall-clock
            # alone bounds the enqueue rate rather than the work: measured on
            # the 2x A100 pair (13.6 Gbit/s inter-node), 0.5 s of enqueueing
            # queued 1026 collectives = 263 GB, which then took 155 s to drain.
            # One "0.5 s slot" therefore ran 155 s, the 60 s cell ran 461.6 s,
            # and the amount of communication per slot scaled with how SLOW the
            # fabric was -- so the cell was not the same workload on two
            # machines. On H100 (~100 GB/s) the same backlog drains in a couple
            # of seconds, which is why this went unnoticed. The per-iteration
            # sync costs tens of microseconds against a transfer of 150 ms here
            # and ~2.5 ms on H100, so it is free in both regimes.
            t_end = time.time() + args.slot_sec
            while time.time() < t_end:
                dist.all_reduce(comm_buf, op=dist.ReduceOp.SUM)
                torch.cuda.synchronize(device)
                allreduces += 1
        if rank == 0:
            emit_step("nonml", step, tokens=0, passes=1,
                      node=node_rank, rank=rank)
        step += 1

        now = time.time()
        if rank == 0 and now - last_print >= 30:
            extra = (f"filler_iters={filler_iters}" if args.mode == "hpc"
                     else f"allreduces={allreduces}")
            print(f"  {now - wall_start:.0f}s: iters={step} {extra}", flush=True)
            last_print = now

    if rank == 0:
        el = time.time() - wall_start
        print(f"  Done: {step} iterations in {el:.1f}s")
        print(f"  NONML_CONTROL_DATA: mode={args.mode} world_size={world_size} "
              f"iters={step} filler_iters={filler_iters} "
              f"allreduces={allreduces} alloc_gb={args.alloc_gb} "
              f"duty={args.duty} fmode={args.filler_mode} "
              f"duration={el:.1f} train_steps=0 useful_tokens=0", flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
