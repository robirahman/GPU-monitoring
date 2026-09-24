#!/usr/bin/env python3
"""W20 -- fake-collective padding co-tenant for a REAL serving job.

The FP weapon "periodic all_reduce(zeros) on a serving box": while real vLLM serves real
traffic (``llm_moe_serving.py --cotenant-fake-ar``), this process group runs alongside on the
same GPUs and issues a periodic NCCL AllReduce of a zero buffer, so the fabric shows
training-shaped AllReduce bursts that belong to no training job. Breaks volume-only detectors.

Honest by construction: the serving stack is untouched (no hook, no fake events inside vLLM).
The collectives logged here are ones THIS process really issues -- ``source=cotenant_fake_ar``,
``phase=fake_ar``, ``interconnect_truth=padding traffic, not training`` -- with wall-clock
``t``/``t_end`` on the same clock as the serving traffic log (``--traffic-log``), so a
detector's alerts can be scored against both. Real-serve collectives (vLLM's own NCCL) are NOT
in this log; capture them with NCCL_DEBUG COLL + ``scripts/parse_nccl_coll_log.py`` if needed.

One rank per GPU (``torchrun --nproc_per_node=<gpus>``); the serve must leave a little GPU
memory free (``--gpu-memory-utilization 0.85``): CUDA context + NCCL buffers + the pad buffer.

Launched by ``llm_moe_serving.py`` after the server is healthy; standalone::

    torchrun --standalone --nproc_per_node=8 workloads/fake_ar_cotenant.py \\
        --every-s 2 --mb 64 --duration 300 --comm-log data/fake_ar.jsonl

CPU smoke::

    torchrun --standalone --nproc_per_node=2 workloads/fake_ar_cotenant.py --backend gloo \\
        --every-s 0.05 --mb 1 --burst 2 --max-steps 5 --comm-log /tmp/fake_ar.jsonl
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist

from interconnect_common import (
    JsonlLog,
    StopSignal,
    add_common_args,
    ar_wire_bytes,
    dist_cleanup,
    dist_setup,
    dtype_from_name,
    write_meta,
)

MIB = 1024 * 1024


def main() -> None:
    ap = argparse.ArgumentParser(description="W20 fake-AllReduce co-tenant",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap, duration=0.0)
    ap.add_argument("--every-s", type=float, default=2.0, help="seconds between padding bursts")
    ap.add_argument("--mb", type=float, default=64.0, help="AllReduce payload MiB")
    ap.add_argument("--burst", type=int, default=1, help="back-to-back AllReduces per burst")
    ap.add_argument("--dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    ap.add_argument("--start-at", type=float, default=0.0,
                    help="epoch second of the first burst (default: now + one period)")
    args = ap.parse_args()

    env = dist_setup(args.backend)
    dtype = dtype_from_name(args.dtype) if env.device.type == "cuda" else torch.float32
    buf = torch.zeros(int(args.mb * MIB) // torch.empty(0, dtype=dtype).element_size(),
                      dtype=dtype, device=env.device)
    payload = buf.numel() * buf.element_size()
    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    write_meta(args.comm_log, {
        "workload": "fake_ar_cotenant", "wid": "W20", "world_size": env.world, "backend": env.backend,
        "every_s": args.every_s, "payload_bytes": payload, "burst": args.burst,
        "interconnect_truth": "padding traffic (all_reduce of zeros) co-tenant with real serving; not training",
    }, env)
    if env.is_main:
        print(f"fake_ar_cotenant world={env.world} {payload / MIB:.1f} MiB AllReduce x{args.burst} "
              f"every {args.every_s}s")

    dist.barrier()
    # rank 0's clock decides the schedule; every rank follows the same absolute times
    t0 = torch.tensor([args.start_at or (time.time() + args.every_s)], dtype=torch.float64)
    if env.world > 1:
        if env.ctrl_group is None and env.device.type == "cuda":
            t0 = t0.to(env.device)               # no gloo control group: fall back to NCCL
        dist.broadcast(t0, src=0, group=env.ctrl_group)
    first = float(t0.item())
    stop = StopSignal(env, args.duration if args.duration > 0 else None, args.max_steps)
    step = 0
    while not stop.should_stop(step):
        due = first + step * args.every_s
        while time.time() < due:
            time.sleep(min(0.05, max(0.0, due - time.time())))
        for b in range(args.burst):
            t_a = time.time()
            dist.all_reduce(buf)
            env.sync()
            comm.emit(t=t_a, t_end=time.time(), dur_s=time.time() - t_a, step=step, burst_idx=b,
                      op="all_reduce", phase="fake_ar", source="cotenant_fake_ar", group="world",
                      group_size=env.world, bytes_est=ar_wire_bytes(payload, env.world),
                      payload_bytes=payload)
        step += 1
    comm.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
