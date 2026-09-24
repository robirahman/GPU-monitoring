#!/usr/bin/env python3
"""W10 -- Disaggregated prefill<->decode KV handoff (interconnect FP weapon).

**MICROBENCH ONLY -- not SPAR primary corpus.** Random uint8 buffers over hand-rolled isend/irecv; useful
for detector unit tests and fabric calibration. The primary W10 workload is the real vLLM NixlConnector
run, ``workloads/disagg_kv_vllm.py`` (registry ``disagg_kv_vllm_qwen7b*``).

Inference that *looks like training on the wire*: GB-scale bulk P2P between two rank
pools (prefill -> decode), the traffic shape of llm-d / NIXL / Mooncake style
disaggregated serving. There is no gradient anywhere; a volume-only or
"large periodic transfer => training" detector should fire falsely on this.

Synthetic: uint8 buffers sized like a real KV handoff, moved with batched isend/irecv.
Depends on no external KV server, which is why it stays useful as a unit test. Real
vLLM+NIXL disaggregation is ``disagg_kv_vllm.py`` (primary).

Handoff size (per request)::

    kv_bytes = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes * seq * batch

``--kv-model llama70b --seq-len 4096`` -> 1.25 GiB. ``--kv-mb`` overrides the total
directly. ``--chunks-per-handoff N`` splits one handoff into N blocks (layer-wise
streaming). W18 (train_kv_disguise.py) reuses the same block-size schedule.

Rank pools
  --split evenodd   prefill = even ranks, decode = odd ranks (default; on a 2-node job
                    these pairs are intra-node)
  --split half      prefill = first half, decode = second half (cross-node on 2 nodes)
  --prefill-ranks 0,1,2,3   explicit list, everything else is decode
Pairing
  --pattern pair        i-th prefill -> i-th decode, every round
  --pattern roundrobin  i-th prefill -> decode (i + round) % |D|, so peers rotate

Duty cycle: ``--burst N`` handoffs back-to-back, then ``--gap-s`` idle. Decode ranks
run small GEMMs while a transfer is in flight and through the gap (GPU not idle);
prefill ranks run ``--prefill-ms`` of GEMM before each send.

Comm log (JSONL; rank0, or every rank with --comm-log-all-ranks): one line per
handoff per rank, ``phase=kv_handoff``, ``op=p2p_send|p2p_recv``, ``src_group``,
``dst_group``, ``src_rank``, ``dst_rank``, ``bytes_est``, plus ``round_pairs`` (the
global bipartite graph of that round) so the rank0 log alone reconstructs the topology.

Launch (8 GPUs, one node)::

    torchrun --nproc_per_node=8 workloads/disagg_kv_synth.py \\
        --split half --kv-mb 1024 --gap-s 1 --duration 300 \\
        --comm-log data/disagg_kv_1gb_comm.jsonl
    python scripts/run_workload_multi_gpu.py --workload disagg_kv_synth_1gb --num-gpus 8

Two nodes (same flags on both)::

    torchrun --nnodes=2 --nproc_per_node=8 --node_rank=$N --master_addr=$MASTER \\
        workloads/disagg_kv_synth.py --split half --kv-mb 1024

CPU smoke (no GPU)::

    torchrun --nproc_per_node=4 workloads/disagg_kv_synth.py --backend gloo \\
        --kv-mb 1 --chunks-per-handoff 4 --burst 2 --gap-s 0.1 --max-steps 6 \\
        --compute-hidden 64 --decode-batch 4 --prefill-ms 2 --comm-log /tmp/kv.jsonl
"""
from __future__ import annotations

import argparse
import time

import torch
import torch.distributed as dist

from interconnect_common import (
    KV_PRESETS,
    JsonlLog,
    StopSignal,
    add_common_args,
    dist_cleanup,
    dist_setup,
    dtype_from_name,
    kv_handoff_bytes,
    make_gemm_busywork,
    split_blocks,
    write_meta,
)

MIB = 1024 * 1024
OVERLAP_CAP_S = 30.0   # stop overlapping decode compute with one transfer after this


def split_ranks(world: int, split: str, prefill_arg: str) -> tuple[list[int], list[int]]:
    if prefill_arg:
        prefill = sorted({int(x) for x in prefill_arg.split(",") if x.strip()})
        bad = [r for r in prefill if not 0 <= r < world]
        if bad:
            raise SystemExit(f"--prefill-ranks {bad} outside world of {world}")
    elif split == "evenodd":
        prefill = list(range(0, world, 2))
    else:
        prefill = list(range(world // 2))
    decode = [r for r in range(world) if r not in prefill]
    if not prefill or not decode:
        raise SystemExit(f"need >=1 prefill and >=1 decode rank (world={world}, prefill={prefill})")
    return prefill, decode


def make_pairs(prefill: list[int], decode: list[int], pattern: str, rnd: int) -> list[tuple[int, int]]:
    shift = rnd if pattern == "roundrobin" else 0
    return [(p, decode[(i + shift) % len(decode)]) for i, p in enumerate(prefill)]


def max_fan_in(prefill: list[int], decode: list[int], pattern: str) -> int:
    period = len(decode) if pattern == "roundrobin" else 1
    worst = 1
    for rnd in range(period):
        counts: dict[int, int] = {}
        for _, d in make_pairs(prefill, decode, pattern, rnd):
            counts[d] = counts.get(d, 0) + 1
        worst = max(worst, max(counts.values()))
    return worst


def main() -> None:
    ap = argparse.ArgumentParser(description="W10 synthetic disaggregated KV handoff",
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    add_common_args(ap)
    ap.add_argument("--split", choices=["evenodd", "half"], default="evenodd")
    ap.add_argument("--prefill-ranks", default="", help="Comma list; overrides --split")
    ap.add_argument("--pattern", choices=["pair", "roundrobin"], default="pair")
    ap.add_argument("--kv-model", choices=list(KV_PRESETS), default="llama70b",
                    help="KV geometry preset (layers, kv_heads, head_dim)")
    ap.add_argument("--seq-len", type=int, default=4096, help="Prompt tokens per request")
    ap.add_argument("--batch", type=int, default=1, help="Requests per handoff")
    ap.add_argument("--kv-dtype-bytes", type=int, default=2, help="2 = bf16/fp16 KV, 1 = fp8 KV")
    ap.add_argument("--kv-mb", type=float, default=None,
                    help="Total MiB per handoff (overrides the geometry formula)")
    ap.add_argument("--chunks-per-handoff", type=int, default=1,
                    help="Blocks per handoff (e.g. = layers for layer-wise streaming)")
    ap.add_argument("--burst", type=int, default=1, help="Handoff rounds back-to-back per burst")
    ap.add_argument("--gap-s", type=float, default=2.0, help="Idle seconds after each burst")
    ap.add_argument("--prefill-ms", type=float, default=50.0,
                    help="GEMM busy-time on prefill ranks before each send")
    ap.add_argument("--no-compute", action="store_true", help="Pure transfer, no GEMM busy work")
    ap.add_argument("--compute-hidden", type=int, default=4096)
    ap.add_argument("--decode-batch", type=int, default=32, help="Rows per decode-like GEMM step")
    ap.add_argument("--compute-dtype", choices=["fp16", "bf16", "fp32"], default="bf16")
    args = ap.parse_args()

    env = dist_setup(args.backend)
    rank = env.rank
    prefill, decode = split_ranks(env.world, args.split, args.prefill_ranks)
    role = "prefill" if rank in prefill else "decode"

    layers, kv_heads, head_dim = KV_PRESETS[args.kv_model]
    if args.kv_mb is not None:
        total = int(args.kv_mb * MIB)
    else:
        total = kv_handoff_bytes(layers, kv_heads, head_dim, args.seq_len, args.batch,
                                 args.kv_dtype_bytes)
    n_chunks = max(1, args.chunks_per_handoff)
    blocks = split_blocks(total, -(-total // n_chunks))
    fan_in = max_fan_in(prefill, decode, args.pattern)

    send_buf = recv_buf = None
    if role == "prefill":
        send_buf = torch.empty(total, dtype=torch.uint8, device=env.device).random_(0, 256)
    else:
        recv_buf = torch.empty(total * fan_in, dtype=torch.uint8, device=env.device)

    busy = None
    if not args.no_compute:
        cdt = dtype_from_name(args.compute_dtype) if env.device.type == "cuda" else torch.float32
        busy = make_gemm_busywork(env, args.compute_hidden, args.decode_batch, cdt)

    comm = JsonlLog(args.comm_log, env, args.comm_log_all_ranks)
    write_meta(args.comm_log, {
        "workload": "disagg_kv_synth", "wid": "W10", "world_size": env.world, "backend": env.backend,
        "prefill_ranks": prefill, "decode_ranks": decode, "pattern": args.pattern,
        "kv_model": args.kv_model, "seq_len": args.seq_len, "batch": args.batch,
        "kv_dtype_bytes": args.kv_dtype_bytes, "handoff_bytes": total,
        "n_blocks": len(blocks), "block_bytes": blocks, "burst": args.burst,
        "gap_s": args.gap_s, "interconnect_truth": "inference_kv_handoff (FP: loud, non-training)",
    }, env)

    if env.is_main:
        print(f"disagg_kv_synth world={env.world} backend={env.backend} pattern={args.pattern} "
              f"prefill={prefill} decode={decode}")
        print(f"  handoff={total / MIB:.1f} MiB in {len(blocks)} block(s) "
              f"(geometry {args.kv_model}: 2*{layers}L*{kv_heads}kvh*{head_dim}d*{args.kv_dtype_bytes}B "
              f"*seq {args.seq_len}*batch {args.batch}); burst={args.burst} gap={args.gap_s}s")

    stop = StopSignal(env, args.duration, args.max_steps)
    dist.barrier()
    stop.t0 = time.time()
    t_start = stop.t0
    t_next = time.time()
    rnd = 0
    moved = 0   # bytes this rank has sent+received

    while not stop.should_stop(rnd):
        # Wait for the next round: decode ranks keep decoding, prefill ranks idle.
        while time.time() < t_next:
            if role == "decode" and busy is not None:
                busy()
            else:
                time.sleep(min(0.005, max(0.0, t_next - time.time())))

        pairs = make_pairs(prefill, decode, args.pattern, rnd)
        my_send = [d for s, d in pairs if s == rank]
        my_recv = [s for s, d in pairs if d == rank]

        if my_send and busy is not None and args.prefill_ms > 0:
            t_end = time.time() + args.prefill_ms / 1000.0
            while time.time() < t_end:
                busy()

        ops: list[dist.P2POp] = []
        for dst in my_send:
            off = 0
            for sz in blocks:
                ops.append(dist.P2POp(dist.isend, send_buf[off:off + sz], dst))
                off += sz
        for j, src in enumerate(my_recv):
            off = j * total
            for sz in blocks:
                ops.append(dist.P2POp(dist.irecv, recv_buf[off:off + sz], src))
                off += sz

        env.sync()                       # prefill compute must not count as transfer time
        t_issue = time.time()
        reqs = dist.batch_isend_irecv(ops) if ops else []
        # NCCL Work.is_completed() polls a CUDA event; gloo's only flips inside wait(),
        # so on gloo (CPU smoke tests) there is no overlap and we just wait.
        while reqs and env.backend == "nccl" and not all(r.is_completed() for r in reqs):
            if time.time() - t_issue > OVERLAP_CAP_S:
                break
            if role == "decode" and busy is not None:
                busy()                   # decode overlaps the in-flight transfer
            else:
                time.sleep(0.0005)
        for r in reqs:
            r.wait()
        env.sync()
        t_done = time.time()

        dur = max(t_done - t_issue, 1e-9)
        rank_bytes = total * (len(my_send) + len(my_recv))
        moved += rank_bytes
        common = dict(t=t_issue, t_end=t_done, dur_s=dur, step=rnd, round=rnd,
                      burst_idx=rnd % args.burst, phase="kv_handoff",
                      src_group="prefill", dst_group="decode",
                      bytes_est=total, payload_bytes=total,
                      n_blocks=len(blocks), block_bytes=max(blocks),
                      rank_gb_per_s=rank_bytes / dur / 1e9,
                      round_pairs=[list(p) for p in pairs], round_bytes_total=len(pairs) * total)
        for dst in my_send:
            comm.emit(op="p2p_send", src_rank=rank, dst_rank=dst, **common)
        for src in my_recv:
            comm.emit(op="p2p_recv", src_rank=src, dst_rank=rank, **common)

        rnd += 1
        if env.is_main and (rnd <= 3 or rnd % 10 == 0):
            print(f"  round {rnd}: {len(pairs)} handoff(s) x {total / MIB:.1f} MiB in {dur:.3f}s "
                  f"({rank_bytes / dur / 1e9:.2f} GB/s on rank0)")
        t_next = time.time() + (args.gap_s if rnd % args.burst == 0 else 0.0)

    dist.barrier()
    if env.is_main:
        wall = time.time() - t_start
        print(f"disagg_kv_synth done: {rnd} rounds, {wall:.1f}s, "
              f"rank0 moved {moved / 1e9:.2f} GB, fabric total "
              f"{rnd * len(make_pairs(prefill, decode, args.pattern, 0)) * total / 1e9:.2f} GB")
    comm.close()
    dist_cleanup()


if __name__ == "__main__":
    main()
