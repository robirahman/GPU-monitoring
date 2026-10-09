#!/usr/bin/env python3
"""DLRM-style recommender inference -- the one workload class the corpus has none of.

WHY THIS EXISTS. Every inference stream in the corpus is a dense model doing dense
arithmetic: LLM decode, ViT, CLIP, SDXL. Recommenders are the other half of what
production inference fleets actually run, and they are shaped nothing like the
dense half:

  * the parameters are EMBEDDING TABLES, tens of GB of them, and a step touches a
    vanishing fraction -- so the job is enormous in memory and tiny in FLOPs, the
    opposite corner from a 72B LLM that is large in both
  * the work is sparse GATHER, not GEMM: random rows pulled from a huge table,
    which is memory-latency bound and leaves the SMs mostly idle
  * the tables do not fit on one GPU, so they are sharded by table and each step
    does an ALL-TO-ALL to exchange the looked-up vectors -- a collective driven by
    the batch, not by the model, and one no other inference cell in the corpus
    performs
  * only the small dense MLP on top is compute-bound

That profile -- huge resident memory, low utilisation, all-to-all every step --
is close to what an evasive training job would try to look like, and the detector
has never seen an honest example of it. Section 33d found the detector reads
absolute memory level; a legitimate inference job holding 30 GB of tables is the
direct test of whether that is a real signal or a shortcut.

MODEL PARALLEL, NOT DATA PARALLEL. Each rank owns a disjoint subset of the tables
and looks up the WHOLE global batch in them, then the all-to-all redistributes so
each rank ends with all features for its own slice of the batch. That is the real
DLRM layout, and it is the reason the collective exists.

LABELLING: cells are named s16_inferrec_* -- contains "infer" and none of the
training keywords, so threeway_label() routes to ml_inference. Correct: forward
only, no gradients, no optimizer. Verified in
classifier/rebuttal_analysis/test_cell_label_routing.py. STEP lines carry
kind="infer".

    torchrun --nnodes=2 --nproc-per-node=8 workloads/inference_recsys.py \
        --table-gb 2 --batch-size 4096 --duration 600
"""
import argparse
import os
import time

import torch
import torch.distributed as dist
import torch.nn as nn

try:
    from workloads._step import emit_step
    from workloads.dist_deadline import CollectiveDeadline, sync_teardown
except ImportError:
    from _step import emit_step
    from dist_deadline import CollectiveDeadline, sync_teardown


class DenseMLP(nn.Module):
    """The small compute-bound part: bottom MLP on dense features, top MLP on the
    concatenation of pooled embeddings and the bottom output."""

    def __init__(self, dense_dim, embed_dim, n_tables, hidden=512):
        super().__init__()
        self.bottom = nn.Sequential(
            nn.Linear(dense_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, embed_dim), nn.ReLU())
        top_in = embed_dim * (n_tables + 1)
        self.top = nn.Sequential(
            nn.Linear(top_in, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden // 2), nn.ReLU(),
            nn.Linear(hidden // 2, 1))

    def forward(self, dense, pooled):
        b = self.bottom(dense)
        return torch.sigmoid(self.top(torch.cat([b] + pooled, dim=1)))


def main():
    ap = argparse.ArgumentParser(description="DLRM-style sharded recommender inference")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=4096, help="global batch")
    ap.add_argument("--table-gb", type=float, default=2.0,
                    help="embedding table bytes PER RANK. The footprint knob: "
                         "recommenders are defined by holding far more memory than "
                         "they compute on, and section 33d makes memory level an "
                         "axis we must sweep rather than fix.")
    ap.add_argument("--n-tables", type=int, default=8, help="tables per rank")
    ap.add_argument("--embed-dim", type=int, default=128)
    ap.add_argument("--dense-dim", type=int, default=13)
    ap.add_argument("--lookups-per-table", type=int, default=30,
                    help="sparse ids per sample per table (pooled by sum)")
    args = ap.parse_args()

    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    world = int(os.environ.get("WORLD_SIZE", 1))
    rank = int(os.environ.get("RANK", 0))
    node_rank = int(os.environ.get("GROUP_RANK", 0))

    dist.init_process_group(backend="nccl")
    torch.cuda.set_device(local_rank)
    device = torch.device(f"cuda:{local_rank}")

    if args.batch_size % world:
        raise SystemExit(f"--batch-size {args.batch_size} must be divisible by "
                         f"world size {world}")
    local_batch = args.batch_size // world

    # Tables sized to hit the requested footprint. fp16 = 2 bytes.
    rows = max(1024, int(args.table_gb * 1e9 / (args.n_tables * args.embed_dim * 2)))
    tables = [torch.randn(rows, args.embed_dim, dtype=torch.float16, device=device)
              for _ in range(args.n_tables)]
    mlp = DenseMLP(args.dense_dim, args.embed_dim,
                   args.n_tables * world).to(device).half().eval()

    if rank == 0:
        gb = args.n_tables * rows * args.embed_dim * 2 / 1e9
        print(f"DLRM-style recommender inference: {world} ranks")
        print(f"  {args.n_tables} tables x {rows:,} rows x {args.embed_dim}d fp16 "
              f"= {gb:.2f} GB/rank ({gb*world:.1f} GB global)")
        print(f"  global batch {args.batch_size}, {args.lookups_per_table} lookups/"
              f"table/sample, all-to-all "
              f"{local_batch*args.n_tables*world*args.embed_dim*2/1e6:.1f} MB/step")
        print(f"  duration {args.duration}s")

    deadline = CollectiveDeadline(args.duration, device)
    step = 0
    last = time.time()
    while not deadline.expired():
        with torch.inference_mode():
            # Every rank looks the GLOBAL batch up in the tables it owns.
            pooled_local = []
            for t in tables:
                idx = torch.randint(0, rows,
                                    (args.batch_size * args.lookups_per_table,),
                                    device=device)
                v = t[idx].view(args.batch_size, args.lookups_per_table, -1).sum(1)
                pooled_local.append(v)
            # (n_tables, global_batch, d) -> all-to-all so each rank ends with the
            # whole feature set for its own slice of the batch.
            send = torch.stack(pooled_local, dim=1)            # (B, T, d)
            send = send.view(world, local_batch, args.n_tables, args.embed_dim)
            recv = torch.empty_like(send)
            dist.all_to_all_single(recv, send.contiguous())
            # recv[p] is rank p's tables for MY batch slice.
            feats = [recv[p, :, t] for p in range(world)
                     for t in range(args.n_tables)]

            dense = torch.randn(local_batch, args.dense_dim,
                                dtype=torch.float16, device=device)
            out = mlp(dense, feats)
        step += 1
        if rank == 0:
            emit_step("infer", step, tokens=args.batch_size, passes=1,
                      node=node_rank, rank=rank, model="dlrm")
            now = time.time()
            if now - last >= 30:
                el = deadline.elapsed
                print(f"  t={el:6.0f}s  step={step}  "
                      f"{step*args.batch_size/max(el,1):.0f} samples/s  "
                      f"last p={float(out.float().mean()):.4f}  "
                      f"peak {torch.cuda.max_memory_allocated(device)/1e9:.1f} GB",
                      flush=True)
                last = now

    if rank == 0:
        print(f"Recsys done: {step} steps, {step*args.batch_size} samples, "
              f"{deadline.elapsed:.0f}s", flush=True)
    sync_teardown()


if __name__ == "__main__":
    main()
