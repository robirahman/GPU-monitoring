#!/usr/bin/env python3
"""Convert NCCL's own collective debug log into JSONL -- a TRUTHFUL per-collective record of
what a real framework (torchtitan, Megatron, vLLM, ...) issued, with no instrumentation of the
framework and no synthetic events.

Capture (env vars, all read from NCCL's source; older NCCL may lack the timestamp ones, in
which case lines simply have no timestamp)::

    NCCL_DEBUG=INFO  NCCL_DEBUG_SUBSYS=COLL
    NCCL_DEBUG_FILE=/path/nccl.%h.%p.log            # one file per host/pid
    NCCL_DEBUG_TIMESTAMP_LEVELS=INFO  NCCL_DEBUG_TIMESTAMP_FORMAT="[%F %T.%6f] "

(``scripts/run_titan_job.py --nccl-collective-log`` sets them.) NCCL logs each collective at
enqueue with this line (src/enqueue/enqueue.cc)::

    <op>: opCount %lx sendbuff %p recvbuff %p count %zu datatype %d op %d root %d comm %p [nranks=%d] stream %p

Timestamps are NCCL's ``[%F %T.%6f]`` in the *local time of the logging host*; they are converted
with this machine's timezone, so parse on the same host (or pass ``--tz-offset-hours``).

Byte fields (``payload_bytes`` = count x datatype size; ``bytes_est`` = per-rank send-side ring
estimate, same convention as workloads/interconnect_common):
  AllReduce                 count = total elements          bytes_est = 2(N-1)/N * payload
  AllGather / AllToAll      count = elements per rank       bytes_est = (N-1) * payload
  ReduceScatter             count = output elements/rank    bytes_est = (N-1) * payload
  Broadcast / Reduce        count = total elements          bytes_est = payload
  Send / Recv               count = elements                bytes_est = payload
Unknown op names are kept with ``bytes_est = payload_bytes``. These are model-based estimates
of wire bytes, not measurements; fabric truth is NVLink/IB counters.
"""
from __future__ import annotations

import argparse
import glob
import json
import re
import sys
from datetime import datetime

# ncclDataType_t (src/nccl.h.in) -> (name, bytes)
DATATYPES = {0: ("int8", 1), 1: ("uint8", 1), 2: ("int32", 4), 3: ("uint32", 4), 4: ("int64", 8),
             5: ("uint64", 8), 6: ("fp16", 2), 7: ("fp32", 4), 8: ("fp64", 8), 9: ("bf16", 2),
             10: ("fp8e4m3", 1), 11: ("fp8e5m2", 1)}

LINE = re.compile(
    r"^(?:\[(?P<ts>[^\]]+)\]\s+)?(?P<host>\S+?):(?P<pid>\d+):(?P<tid>\d+)\s+\[(?P<dev>\d+)\]\s+NCCL INFO\s+"
    r"(?P<op>\w+): opCount (?P<opcount>[0-9a-fA-F]+) sendbuff \S+ recvbuff \S+ count (?P<count>\d+) "
    r"datatype (?P<dt>\d+) op (?P<redop>-?\d+) root (?P<root>-?\d+) comm (?P<comm>\S+) "
    r"\[nranks=(?P<nranks>\d+)\]")


def ts_to_epoch(ts: str | None, tz_offset_hours: float | None) -> float | None:
    if not ts:
        return None
    for fmt in ("%Y-%m-%d %H:%M:%S.%f", "%Y-%m-%d %H:%M:%S"):
        try:
            dt = datetime.strptime(ts.strip(), fmt)
        except ValueError:
            continue
        if tz_offset_hours is None:
            return dt.timestamp()                                   # this machine's local tz
        return dt.timestamp() - (datetime.now().astimezone().utcoffset().total_seconds()
                                 - tz_offset_hours * 3600)
    return None


def wire_bytes_est(op: str, payload: int, n: int) -> int:
    if n <= 1:
        return 0
    if op == "AllReduce":
        return int(2 * (n - 1) / n * payload)
    if op in ("AllGather", "ReduceScatter", "AllToAll", "Alltoall"):
        return (n - 1) * payload
    return payload


def parse_line(line: str, tz_offset_hours: float | None = None) -> dict | None:
    m = LINE.match(line)
    if not m:
        return None
    name, size = DATATYPES.get(int(m["dt"]), (f"dt{m['dt']}", 1))
    payload = int(m["count"]) * size
    nranks = int(m["nranks"])
    return {"t": ts_to_epoch(m["ts"], tz_offset_hours), "source": "nccl_debug_coll", "op": m["op"],
            "count": int(m["count"]), "datatype": name, "payload_bytes": payload,
            "bytes_est": wire_bytes_est(m["op"], payload, nranks), "nranks": nranks,
            "comm": m["comm"], "op_count": int(m["opcount"], 16), "root": int(m["root"]),
            "cuda_dev": int(m["dev"]), "pid": int(m["pid"]), "host": m["host"]}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("logs", nargs="+", help="NCCL_DEBUG_FILE outputs (globs ok)")
    ap.add_argument("--out", default="-", help="JSONL path (default stdout)")
    ap.add_argument("--cuda-dev", type=int, default=None, help="keep only this CUDA device's lines "
                    "(every rank logs every collective it joins; one device = one rank's view)")
    ap.add_argument("--min-payload-bytes", type=int, default=0)
    ap.add_argument("--tz-offset-hours", type=float, default=None)
    args = ap.parse_args()

    files = sorted({f for pat in args.logs for f in glob.glob(pat)})
    if not files:
        sys.exit("no log files matched")
    events = []
    for path in files:
        with open(path, errors="replace") as fh:
            for line in fh:
                ev = parse_line(line, args.tz_offset_hours)
                if ev and (args.cuda_dev is None or ev["cuda_dev"] == args.cuda_dev) \
                        and ev["payload_bytes"] >= args.min_payload_bytes:
                    events.append(ev)
    events.sort(key=lambda e: (e["t"] if e["t"] is not None else 0.0, e["pid"], e["op_count"]))
    out = sys.stdout if args.out == "-" else open(args.out, "w")
    for ev in events:
        out.write(json.dumps(ev, separators=(",", ":")) + "\n")
    if out is not sys.stdout:
        out.close()
    print(f"{len(events)} collectives from {len(files)} file(s)"
          + (f" -> {args.out}" if args.out != "-" else ""), file=sys.stderr)


if __name__ == "__main__":
    main()
