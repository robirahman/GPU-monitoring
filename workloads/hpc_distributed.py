#!/usr/bin/env python3
"""HONEST non-ML HPC with REAL cross-node data dependency.

WHY THIS EXISTS. The whole `other` class of the multi-node corpus comes from one
script, `nonml_distributed.py`, and its `--mode hpc` is embarrassingly parallel:
every rank runs identical kernels on identical local data, and the only traffic
crossing the node boundary is CollectiveDeadline's 4-byte vote, about one per
2.5 s. It is "sixteen independent single-GPU jobs launched by torchrun", not a
distributed solver. `--mode comm` is genuinely collective but is a fixed-size
all_reduce with negligible compute, so between them the corpus has no workload
that both computes hard AND talks hard without being training.

That gap matters more than any other. Leave-one-family-out (2026-09-22) put
held-out training recall at 0.005-0.04, and the project's own finding is that
inter-node NIC traffic is what most separates training from sharded inference --
FSDP is roughly 1000x louder than sharded inference. A detector that has never
seen a non-training job with FSDP-like interconnect traffic has not been tested
on the case that would fool it. These three modes are that case.

  --mode fftshard  Slab-decomposed 3-D FFT. The transform along the distributed
                   axis is impossible without redistributing the grid, so every
                   pass performs a genuine all-to-all of the entire field. This
                   is the heaviest honest cross-node traffic we can generate, and
                   it is the classic HPC communication pattern (parallel spectral
                   solvers, PIC codes, molecular dynamics with PME).
  --mode stencil   7-point Jacobi on a domain decomposed along x, with halo
                   exchange to the two neighbouring ranks each sweep. The
                   nearest-neighbour pattern: much less volume than an
                   all-to-all, but latency-bound and continuous, which is the
                   shape most CFD and finite-difference codes actually have.
  --mode nccl      Collective bandwidth sweep over message sizes (all_reduce,
                   all_gather, reduce_scatter, all_to_all). The nccl-tests shape.
                   Distinct from nonml_distributed's `comm`, which is one fixed
                   256 MB all_reduce: sweeping the size is what exercises the
                   latency-bound and bandwidth-bound regimes in one run.

LABELLING: no label for these cells contains "train" or "infer", so
threeway_label() routes them to `other` -- correct, since there is no model, no
gradient and no optimizer anywhere in this file. No label matches
DISGUISED_FAMILIES, so they legitimately join the honest corpus. Like
nonml_distributed.py, STEP lines are emitted with kind="nonml", which
window_labels.STEP_RE deliberately does not match (it accepts only train|infer),
so no window is marked train_active. Verified by
classifier/rebuttal_analysis/test_cell_label_routing.py.

MEMORY IS AN AXIS, NOT AN ACCIDENT. tpu_multislice_findings.md section 33d found
the detector reads absolute memory level rather than periodicity, and section 33c
that every vLLM cell sits at 74-79 GB because of --gpu-memory-utilization. So
--grid and --fields exist to place this family at several resident footprints;
collecting all three modes at one footprint would relabel that confound instead
of removing it.

    torchrun --nnodes=2 --nproc-per-node=8 workloads/hpc_distributed.py \
        --mode fftshard --grid 512 --duration 600
    torchrun --nnodes=2 --nproc-per-node=8 workloads/hpc_distributed.py \
        --mode stencil --grid 768 --duration 600
    torchrun --nnodes=2 --nproc-per-node=8 workloads/hpc_distributed.py \
        --mode nccl --max-mb 512 --duration 600
"""

import argparse
import os
import time

import torch
import torch.distributed as dist

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
except ImportError:  # launched as `torchrun workloads/hpc_distributed.py`
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown


def fft_slab_step(a, work, world, nloc, n, fields):
    """One 3-D FFT over a slab-decomposed field, and back. Returns bytes moved.

    Each rank owns `nloc = n // world` planes of x and the whole of y and z. The
    y and z transforms are local. The x transform is not: rank r holds only its
    own planes, so before it can transform along x the grid has to be
    redistributed into y-slabs. That redistribution is an all-to-all of the
    entire field, and it is the reason this workload exists -- it is real
    algorithmic necessity, not synthetic traffic.

    TWO exchanges per pass, not one, because that is what a spectral solver
    actually does: forward transform, operate in k-space, transform back, and the
    result has to land in the original decomposition or the next timestep cannot
    use it. Returning with a local permute instead would be silently wrong -- it
    reinterprets y-slab data as x-slab data, giving a field nobody computed.

    The reshape/permute dance is the standard slab transpose. all_to_all_single
    splits along dim 0 and sends chunk p to rank p, so:

      forward   (nloc, n, n) -> (nloc, world, nyloc, n) -> (world, nloc, nyloc, n)
                after exchange dim 0 indexes the SOURCE rank, and since rank p
                owns x-planes [p*nloc, (p+1)*nloc), folding dim 0 into dim 1
                reassembles the full x axis in the right order.
      inverse   exactly the mirror; the two are checked against a single-process
                numpy fftn in test_hpc_distributed_transpose.py.
    """
    nyloc = n // world
    moved = 0
    for f in range(fields):
        # Local transforms along the two resident axes.
        buf = torch.fft.fftn(a[f], dim=(1, 2))
        # Redistribute: x-slabs -> y-slabs.
        send = (buf.reshape(nloc, world, nyloc, n)
                   .permute(1, 0, 2, 3)
                   .contiguous())
        dist.all_to_all_single(work[f], send)
        moved += send.element_size() * send.nelement()
        # dim0 was the source rank; folding it into dim1 restores the full x axis.
        ypencil = torch.fft.fft(work[f].reshape(world * nloc, nyloc, n), dim=0)
        # Redistribute back: y-slabs -> x-slabs. Chunk p is rank p's x-range.
        back = torch.empty_like(work[f])          # (world, nloc, nyloc, n)
        dist.all_to_all_single(back, ypencil.reshape(world, nloc, nyloc, n).contiguous())
        moved += back.element_size() * back.nelement()
        # dim0 now indexes the y-block each peer returned; fold it into y.
        a[f].copy_(back.permute(1, 0, 2, 3).reshape(nloc, n, n))
    return moved


def run_fftshard(args, device, rank, world, node_rank, deadline):
    n = args.grid
    if n % world:
        raise SystemExit(f"--grid {n} must be divisible by world size {world}")
    nloc = n // world
    # complex64: 8 bytes. Per rank we hold `fields` slabs plus an equal-sized
    # exchange buffer, so resident bytes = 2 * fields * nloc * n * n * 8.
    a = [torch.randn(nloc, n, n, dtype=torch.complex64, device=device)
         for _ in range(args.fields)]
    # The exchange buffers hold the TRANSPOSE layout (world, nloc, nyloc, n), not
    # the field layout (nloc, n, n). Same element count, different rank -- and
    # allocating them with empty_like(a[0]) gave a 3-D buffer that then failed on
    # `back.permute(1, 0, 2, 3)`. The numpy proof could not catch this: it builds
    # its arrays by explicit reshape, so it never exercised the allocation.
    nyloc = n // world
    work = [torch.empty(world, nloc, nyloc, n, dtype=torch.complex64, device=device)
            for _ in range(args.fields)]
    if rank == 0:
        gb = 2 * args.fields * nloc * n * n * 8 / 1e9
        print(f"  grid {n}^3 over {world} ranks -> local slab {nloc}x{n}x{n}, "
              f"{args.fields} field(s), ~{gb:.2f} GB/rank resident", flush=True)

    step = 0
    moved_total = 0
    last = time.time()
    while not deadline.expired():
        moved_total += fft_slab_step(a, work, world, nloc, n, args.fields)
        step += 1
        if rank == 0:
            emit_step("nonml", step, tokens=0, passes=args.fields,
                      node=node_rank, rank=rank, mode="fftshard")
            now = time.time()
            if now - last >= 30:
                el = deadline.elapsed
                print(f"  t={el:6.0f}s  passes={step}  "
                      f"all-to-all {moved_total / 1e9:.1f} GB "
                      f"({moved_total / 1e9 / max(el, 1):.2f} GB/s/rank)", flush=True)
                last = now
    return step


def run_stencil(args, device, rank, world, node_rank, deadline):
    """7-point Jacobi with halo exchange to the two x-neighbours.

    The domain is split along x and each rank carries one ghost plane on each
    side. Every sweep needs its neighbours' boundary planes before it can update
    its own edges, so the exchange is on the critical path -- unlike a filler
    loop, a rank genuinely cannot proceed alone.

    isend/irecv are posted before the interior is computed so the transfer
    overlaps the bulk of the arithmetic, which is what a real solver does and
    what makes the resulting telemetry a fair negative.
    """
    n = args.grid
    if n % world:
        raise SystemExit(f"--grid {n} must be divisible by world size {world}")
    nloc = n // world
    # +2 ghost planes. float32.
    u = torch.randn(nloc + 2, n, n, device=device)
    v = torch.empty_like(u)
    up = (rank + 1) % world
    dn = (rank - 1) % world
    if rank == 0:
        gb = 2 * (nloc + 2) * n * n * 4 / 1e9
        halo_mb = n * n * 4 / 1e6
        print(f"  grid {n}^3 over {world} ranks -> local {nloc}+2 x {n} x {n}, "
              f"~{gb:.2f} GB/rank, halo {halo_mb:.1f} MB x2 per sweep", flush=True)

    step = 0
    last = time.time()
    while not deadline.expired():
        for _ in range(args.sweeps_per_step):
            # Post the halo exchange first so it overlaps the interior update.
            #
            # batch_isend_irecv, not bare isend/irecv: NCCL point-to-point ops are
            # enqueued on the device stream and a ring of ranks each posting
            # send-up/send-down/recv-down/recv-up in program order deadlocks --
            # which is exactly what the first version did (rc=124, killed at the
            # 240 s timeout). Batching lets NCCL group them into one coalesced
            # exchange that cannot self-deadlock.
            #
            # The .contiguous() copies are bound to names as well. As bare
            # temporaries inside the op list they had no owning reference and
            # could be freed before wait() returned, which is a use-after-free
            # that shows up only under memory pressure.
            send_up = u[nloc].contiguous()
            send_dn = u[1].contiguous()
            top = torch.empty(n, n, device=device)
            bot = torch.empty(n, n, device=device)
            ops = [dist.P2POp(dist.isend, send_up, up),
                   dist.P2POp(dist.isend, send_dn, dn),
                   dist.P2POp(dist.irecv, bot, dn),
                   dist.P2POp(dist.irecv, top, up)]
            reqs = dist.batch_isend_irecv(ops)
            # Interior: needs no ghost data, so it runs while the halos fly.
            v[2:nloc] = (u[1:nloc - 1] + u[3:nloc + 1]
                         + torch.roll(u[2:nloc], 1, dims=1)
                         + torch.roll(u[2:nloc], -1, dims=1)
                         + torch.roll(u[2:nloc], 1, dims=2)
                         + torch.roll(u[2:nloc], -1, dims=2)
                         - 6.0 * u[2:nloc]) * 0.1 + u[2:nloc]
            for r in reqs:
                r.wait()
            u[0].copy_(bot)
            u[nloc + 1].copy_(top)
            # Edges, now that the ghosts have landed.
            for i in (1, nloc):
                v[i] = (u[i - 1] + u[i + 1]
                        + torch.roll(u[i], 1, dims=0) + torch.roll(u[i], -1, dims=0)
                        + torch.roll(u[i], 1, dims=1) + torch.roll(u[i], -1, dims=1)
                        - 6.0 * u[i]) * 0.1 + u[i]
            u, v = v, u
        step += 1
        if rank == 0:
            emit_step("nonml", step, tokens=0, passes=args.sweeps_per_step,
                      node=node_rank, rank=rank, mode="stencil")
            now = time.time()
            if now - last >= 30:
                print(f"  t={deadline.elapsed:6.0f}s  steps={step} "
                      f"({step * args.sweeps_per_step} sweeps)", flush=True)
                last = now
    return step


COLLECTIVES = ("all_reduce", "all_gather", "reduce_scatter", "all_to_all")


def run_nccl(args, device, rank, world, node_rank, deadline):
    """Collective bandwidth sweep -- the nccl-tests shape.

    Every rank walks the identical (collective, size) schedule, which keeps the
    loop rank-uniform so CollectiveDeadline's vote stays valid. Each op is
    synchronised before the next is enqueued: without that the loop becomes
    enqueue-rate bound and queues far more work than it can drain, which is a
    failure nonml_distributed.py already hit once (263 GB queued, 155 s to drain
    on a 13.6 Gbit/s A100 pair).
    """
    sizes = []
    mb = args.min_kb / 1024.0
    while mb <= args.max_mb:
        sizes.append(mb)
        mb *= 4
    if rank == 0:
        print(f"  sweep {args.min_kb} KB -> {args.max_mb} MB in {len(sizes)} steps "
              f"x {len(COLLECTIVES)} collectives, {world} ranks", flush=True)

    biggest = int(args.max_mb * 1e6 / 4)
    buf = torch.ones(biggest, dtype=torch.float32, device=device)
    out = torch.empty(biggest, dtype=torch.float32, device=device)
    # all_gather output is world x input, so the input must be sized down to keep
    # the output inside `out`.
    step = 0
    last = time.time()
    while not deadline.expired():
        for mb in sizes:
            nel = max(world, int(mb * 1e6 / 4) // world * world)
            for op in COLLECTIVES:
                src = buf[:nel]
                if op == "all_reduce":
                    dist.all_reduce(src, op=dist.ReduceOp.SUM)
                elif op == "all_gather":
                    per = nel // world
                    dist.all_gather_into_tensor(out[:per * world], src[:per])
                elif op == "reduce_scatter":
                    per = nel // world
                    dist.reduce_scatter_tensor(out[:per], src[:per * world])
                else:
                    dist.all_to_all_single(out[:nel], src)
                torch.cuda.synchronize()
            step += 1
            if rank == 0:
                emit_step("nonml", step, tokens=0, passes=len(COLLECTIVES),
                          node=node_rank, rank=rank, mode="nccl", mb=f"{mb:.3f}")
            if deadline.expired():
                break
        if rank == 0 and time.time() - last >= 30:
            print(f"  t={deadline.elapsed:6.0f}s  size-steps={step}", flush=True)
            last = time.time()
    return step


def main():
    ap = argparse.ArgumentParser(description="Honest distributed HPC, no model anywhere")
    ap.add_argument("--mode", required=True, choices=["fftshard", "stencil", "nccl"])
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--grid", type=int, default=512,
                    help="fftshard/stencil: cube edge. Must be divisible by the "
                         "world size. Drives the resident footprint, which is a "
                         "deliberate axis -- see the module docstring.")
    ap.add_argument("--fields", type=int, default=1,
                    help="fftshard: independent fields held resident, a second "
                         "footprint knob that does not change the message size")
    ap.add_argument("--sweeps-per-step", type=int, default=4,
                    help="stencil: Jacobi sweeps between deadline checks")
    ap.add_argument("--min-kb", type=float, default=4.0, help="nccl: smallest message")
    ap.add_argument("--max-mb", type=float, default=256.0, help="nccl: largest message")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if rank == 0:
        print(f"Distributed HPC (HONEST non-ML): mode={args.mode}, {world} ranks")
        print("  No model, no gradients, no optimizer anywhere")
        print(f"  Duration: {args.duration}s")

    # check_every=1: these iterations each carry a multi-hundred-MB collective,
    # so one extra 4-byte all_reduce is free, and the default of 5 would let the
    # run overshoot its deadline by up to four full passes.
    deadline = CollectiveDeadline(args.duration, device, check_every=1)
    t0 = time.time()
    fn = {"fftshard": run_fftshard, "stencil": run_stencil, "nccl": run_nccl}[args.mode]
    steps = fn(args, device, rank, world, node_rank, deadline)

    if rank == 0:
        print(f"{args.mode} done: {steps} steps in {time.time() - t0:.0f}s, "
              f"peak GPU mem {torch.cuda.max_memory_allocated(device) / 1e9:.2f} GB",
              flush=True)
    sync_teardown()


if __name__ == "__main__":
    main()
