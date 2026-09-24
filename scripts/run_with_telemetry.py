#!/usr/bin/env python3
"""Run any command with SPAR telemetry ON (all local GPUs + fabric + DCGM).

For first-time / RunPod use: you should not have to know which collectors to enable.
This wrapper starts ``TelemetryCollector`` on every visible GPU (shared IB/eth cache),
runs your command, then saves per-GPU parquet + fabric summary.

2-node example (run the SAME wrapper on each node with matching NODE_RANK)::

    export MASTER_ADDR=<node0_ip> MASTER_PORT=29500 NNODES=2 NPROC_PER_NODE=8
    export NODE_RANK=0   # or 1 on the other node

    python scripts/run_with_telemetry.py \\
      --label hour1_diloco --output-dir data/hour1 --nnodes 2 --node-rank $NODE_RANK -- \\
      torchrun --nnodes=2 --nproc_per_node=$NPROC_PER_NODE --node_rank=$NODE_RANK \\
        --master_addr=$MASTER_ADDR --master_port=$MASTER_PORT \\
        workloads/diloco_production.py --model-size 150m --local-steps 20 --max-steps 60 \\
        --batch-size 1 --seq-len 512 --dataset synthetic --reduce-dtype fp16 --precision bf16 \\
        --step-log runs/hour1_diloco_steps.jsonl --comm-log runs/hour1_diloco_comm.jsonl
"""
from __future__ import annotations

import argparse
import logging
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.collect_telemetry import TelemetryCollector  # noqa: E402

log = logging.getLogger("spar.run_with_telemetry")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--label", required=True, help="workload_label stamped on every parquet row")
    ap.add_argument("--output-dir", default="data/")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--no-dcgm", action="store_true")
    ap.add_argument("--no-fabric", action="store_true")
    ap.add_argument("--nnodes", type=int, default=None)
    ap.add_argument("--node-rank", type=int, default=None)
    ap.add_argument("--nproc-per-node", type=int, default=None,
                    help="default: torch.cuda.device_count() / nvidia-smi count")
    ap.add_argument("--baseline-sec", type=float, default=2.0)
    ap.add_argument("--cooldown-sec", type=float, default=2.0)
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="command after --")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    cmd = list(args.cmd)
    if cmd and cmd[0] == "--":
        cmd = cmd[1:]
    if not cmd:
        ap.error("pass a command after --")

    # GPU count
    nproc = args.nproc_per_node
    if nproc is None:
        try:
            import torch
            nproc = torch.cuda.device_count() if torch.cuda.is_available() else 0
        except Exception:
            nproc = 0
        if not nproc:
            import subprocess as sp
            r = sp.run(["nvidia-smi", "-L"], capture_output=True, text=True)
            nproc = len([ln for ln in (r.stdout or "").splitlines() if ln.strip().startswith("GPU ")])
    if not nproc:
        sys.exit("no GPUs visible")

    nnodes = args.nnodes if args.nnodes is not None else int(os.environ.get("NNODES", "1"))
    node_rank = args.node_rank if args.node_rank is not None else int(os.environ.get("NODE_RANK", "0"))

    fabric_cache = None
    topology = None
    if not args.no_fabric:
        try:
            from scripts.interconnect_collect import SharedFabricCache, infer_topology
        except ImportError:
            from interconnect_collect import SharedFabricCache, infer_topology  # type: ignore
        topology = infer_topology(nnodes=nnodes, node_rank=node_rank, nproc_per_node=nproc)
        fabric_cache = SharedFabricCache(interval_sec=args.interval)
        log.info("Fabric ON topology=%s", topology)
    else:
        log.info("Fabric OFF (--no-fabric)")

    run_id = uuid.uuid4().hex
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    log.info("Starting telemetry on %d GPUs label=%s run_id=%s", nproc, args.label, run_id[:12])

    collectors = []
    for i in range(nproc):
        c = TelemetryCollector(
            gpu_index=i,
            interval_sec=args.interval,
            workload_label=args.label,
            run_id=run_id,
            enable_dcgm=not args.no_dcgm,
            enable_fabric=not args.no_fabric,
            fabric_cache=fabric_cache,
            topology=topology,
        )
        c.start()
        collectors.append(c)
        log.info("  GPU %d: %s", i, c.metadata.gpu_name)

    time.sleep(args.baseline_sec)

    env = os.environ.copy()
    env["SPAR_TELEMETRY_OWNER"] = "run_with_telemetry"
    env.setdefault("NNODES", str(nnodes))
    env.setdefault("NODE_RANK", str(node_rank))
    env.setdefault("NPROC_PER_NODE", str(nproc))
    if topology:
        env.setdefault("SPAR_NNODES", str(topology.get("nnodes", nnodes)))
        env.setdefault("SPAR_NODE_RANK", str(topology.get("node_rank", node_rank)))
        env.setdefault("SPAR_NPROC_PER_NODE", str(nproc))
        if topology.get("world_size", -1) > 0:
            env.setdefault("SPAR_WORLD_SIZE", str(topology["world_size"]))

    log.info("Launching: %s", " ".join(cmd))
    try:
        rc = subprocess.call(cmd, cwd=str(REPO), env=env)
    except KeyboardInterrupt:
        rc = 130
    time.sleep(args.cooldown_sec)

    for i, c in enumerate(collectors):
        c.stop()
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in c.metadata.gpu_name)[:40]
        path = out_dir / f"{args.label}_{safe}_gpu{i}_{run_id[:8]}.parquet"
        c.save(str(path))
        log.info("  saved %s (%d rows)", path, c.sample_count)
        c.cleanup()

    if fabric_cache is not None:
        summary = fabric_cache.summary()
        log.info(
            "Fabric totals ib_bytes=%s eth_tx+rx=%s samples=%s",
            summary.get("ib_total_bytes"),
            (summary.get("eth_total_tx_bytes", 0) or 0) + (summary.get("eth_total_rx_bytes", 0) or 0),
            summary.get("samples"),
        )

    sys.exit(rc)


if __name__ == "__main__":
    main()
