#!/usr/bin/env python3
"""HONEST control: a training job whose optimizer never steps.

Why this exists
---------------
adversarial_multinode.py's `filler_only` cell (MN-I at --train-frac 0) never
trains, yet the honest-only detector scores it p_train 0.44-0.56 and alerts on
100% of its windows -- while honest H100 non-ML runs score 0.001, INCLUDING the
very kernels the filler runs (cufft_concurrent 0.0007, cufft_short 0.0006). The
~500x gap has to come from the harness rather than the workload, because the
workload is the same.

But `filler_only` confounds three things at once: a resident sharded 1.3B model,
the filler's compute, and the deadline's collectives. This file isolates the
FIRST: model resident, process group live, deadline collectives at the same
cadence, and NO compute at all.

  filler_only     model + filler compute + deadline comm  -> p_train ~0.49
  THIS FILE       model +      (idle)     + deadline comm  -> ?
  nonml_distributed --mode hpc
                  dummy alloc + filler compute + deadline comm -> ?
  corpus cufft/ethash          filler-like compute only    -> ~0.0006

LABELLING: the label carries "train", so threeway_label() routes this to
ml_training -- which is CORRECT and deliberate. A stalled or data-starved
training job is still a training job, and it is a real production state (loader
starvation, checkpoint stalls, long eval phases). It is emphatically NOT labelled
`other`: teaching the detector that "apparatus present + no optimizer step =
benign" would hand an evader the exact gap it wants, since it could then keep the
apparatus resident permanently and train in short bursts. This file is also not
in DISGUISED_FAMILIES, so it legitimately joins the honest corpus.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/stalled_training.py \
        --size 1b --duration 600
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
    from workloads.fsdp_training import (
        MODEL_CONFIGS, BATCH_DEFAULTS, RandomTransformerLM, build_fsdp_model,
    )
except ImportError:  # launched as `torchrun workloads/stalled_training.py`
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown
    from fsdp_training import (
        MODEL_CONFIGS, BATCH_DEFAULTS, RandomTransformerLM, build_fsdp_model,
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", default="1b", choices=list(MODEL_CONFIGS))
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--seq-len", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=None)
    ap.add_argument("--slot-sec", type=float, default=0.5,
                    help="wall time per idle iteration. Matches the filler slot "
                         "length in adversarial_multinode.py so the deadline's "
                         "collective cadence is identical to filler_only's.")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    batch_size = args.batch_size or BATCH_DEFAULTS[args.size]

    if rank == 0:
        print(f"Stalled training (HONEST control): {args.size} on {world_size} GPUs")
        print(f"  Model resident, optimizer NEVER steps, GPUs otherwise idle")
        print(f"  Duration: {args.duration}s, idle slot {args.slot_sec}s")

    with torch.device("meta"):
        model = RandomTransformerLM(args.size)
    fsdp_model = build_fsdp_model(model, device, activation_ckpt=True,
                                 cpu_offload=False)
    # Built and held, exactly as a real training job holds it. Never stepped.
    optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)

    if rank == 0:
        print(f"  Post-init GPU memory: "
              f"{torch.cuda.max_memory_allocated(device) / 1e9:.1f} GB")

    deadline = CollectiveDeadline(args.duration, device)
    wall_start = time.time()
    step = 0
    last_print = wall_start

    while not deadline.expired():
        # No compute. The apparatus alone: sharded params + optimizer state
        # resident, process group alive, deadline collectives ticking.
        time.sleep(args.slot_sec)
        if rank == 0:
            emit_step("stall", step, tokens=0, passes=1,
                      node=node_rank, rank=rank)
        step += 1

        now = time.time()
        if rank == 0 and now - last_print >= 30:
            print(f"  {now - wall_start:.0f}s: idle_iters={step}", flush=True)
            last_print = now

    if rank == 0:
        el = time.time() - wall_start
        print(f"  Done: {step} idle iterations in {el:.1f}s")
        print(f"  STALLED_CONTROL_DATA: world_size={world_size} "
              f"idle_iters={step} duration={el:.1f} train_steps=0 "
              f"useful_tokens=0", flush=True)

    sync_teardown()


if __name__ == "__main__":
    main()
