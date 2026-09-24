#!/usr/bin/env python3
"""W23 -- intra-node-only training with an explicit IB-quiet harness.

Wraps an unmodified single-node DDP / FSDP command (``pytorch_training_ddp.py``,
``fsdp_training.py``, ...) and turns the *emergent* "NVLink busy, IB idle" property into
something we can assert and label:

  1. Pins NCCL to NVLink / away from the network (env table below).
  2. Writes a sidecar note ``{"nvlink_expected": true, "ib_expected": false, ...}``.
  3. Samples ``/sys/class/infiniband/*/ports/*/counters/{port_xmit_data,port_rcv_data}``
     around the run (passive IB-quiet proof) and records the byte deltas + a verdict.
  4. Optionally (``--nccl-evidence``) turns on NCCL INFO logging and counts the
     transports NCCL actually chose (``via P2P`` vs ``via NET/IB``) -- active proof.

Why this exists: the SPAR threat is "train on NVLink, leave the fabric quiet so a
cross-node monitor sees nothing" (P2 *intra-node-only train*; needs the NVML/DCGM layer to
catch). To score that family you need runs *known* to be IB-quiet, not just hoped to be.
Pair the sidecar's [t_start, t_end] with the NVLink counters collect_telemetry.py already
samples (DCGM fields 1011/1012).

NCCL environment applied (all overridable with ``--nccl-env KEY=VAL``; ``--no-pin`` skips
1-3 and only records/measures):

  NCCL_IB_DISABLE=1       disable the InfiniBand/RoCE transport entirely
  NCCL_P2P_LEVEL=NVL      GPU<->GPU P2P only where an NVLink path exists; without it NCCL
                          may pick PCIe P2P/SHM between non-NVLink pairs
  NCCL_NET_GDR_LEVEL=LOC  never use GPUDirect RDMA (LOC = local only)

Guard: refuses ``--nnodes > 1`` in the wrapped command (multi-node traffic *must* cross
the fabric); ``--allow-multinode`` overrides for control experiments.

IB counter units: ``port_*_data`` are octets / 4 (IB spec), 32-bit on many drivers, so the
sampler (default 0.5 s) accumulates per-sample deltas and corrects one wrap per sample.
Fabric management / NFS-over-IB can add background bytes, hence a threshold
(``--ib-quiet-threshold-bytes``, default 10 MiB) instead of "== 0". No IB devices ->
verdict ``unknown`` (never a false "quiet").

Usage::

    python scripts/run_nvl_only_ibquiet.py --sidecar data/ddp_nvl_ibquiet.json --nccl-evidence -- \\
        torchrun --nproc_per_node=8 workloads/pytorch_training_ddp.py --model resnet50 --duration 300
    python scripts/run_workload_multi_gpu.py --workload train_ddp_nvl_only_ibquiet --num-gpus 8
    python scripts/run_workload_multi_gpu.py --workload train_fsdp7b_nvl_only_ibquiet --num-gpus 8

Sidecar (JSON): ``nvlink_expected``, ``ib_expected``, ``cmd``, ``env_applied``,
``t_start``/``t_end``, ``returncode``, ``ib_counters`` (per device/port deltas, totals,
``available``), ``ib_quiet_verdict`` (quiet | NOT_QUIET | unknown), optional
``nccl_transport_evidence``. Optional ``--ib-series PATH`` writes the raw JSONL samples.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import re
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone

DEFAULT_IB_ROOT = "/sys/class/infiniband"
COUNTER_FILES = {"xmit": "port_xmit_data", "rcv": "port_rcv_data"}
WRAP = 2 ** 32           # 32-bit counters; port_*_data unit = 4 octets

PIN_ENV = {
    "NCCL_IB_DISABLE": "1",
    "NCCL_P2P_LEVEL": "NVL",
    "NCCL_NET_GDR_LEVEL": "LOC",
}


def read_ib_counters(root: str = DEFAULT_IB_ROOT) -> dict[str, dict[str, int]] | None:
    """{"mlx5_0/1": {"xmit": raw, "rcv": raw}} or None when no readable counters exist."""
    out: dict[str, dict[str, int]] = {}
    for xmit_path in sorted(glob.glob(os.path.join(root, "*", "ports", "*", "counters", COUNTER_FILES["xmit"]))):
        parts = xmit_path.split(os.sep)
        key = f"{parts[-5]}/{parts[-3]}"
        cdir = os.path.dirname(xmit_path)
        try:
            vals = {k: int(open(os.path.join(cdir, fn)).read().strip()) for k, fn in COUNTER_FILES.items()}
        except (OSError, ValueError):
            continue
        out[key] = vals
    return out or None


class IbSampler:
    """Wrap-safe accumulator of IB byte deltas (background thread)."""

    def __init__(self, root: str, interval: float, series_path: str | None) -> None:
        self.root = root
        self.interval = interval
        self.series_path = series_path
        self.prev = read_ib_counters(root)
        self.total: dict[str, dict[str, int]] = {
            k: {"xmit": 0, "rcv": 0} for k in (self.prev or {})
        }
        self.wraps = 0
        self.samples = 0
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._series = open(series_path, "w") if (series_path and self.prev) else None

    @property
    def available(self) -> bool:
        return self.prev is not None

    def start(self) -> None:
        if self.available:
            self._thread.start()

    def _sample(self) -> None:
        cur = read_ib_counters(self.root)
        if cur is None or self.prev is None:
            return
        row = {"t": time.time()}
        for key, vals in cur.items():
            if key not in self.prev:
                continue
            for k in ("xmit", "rcv"):
                d = vals[k] - self.prev[key][k]
                if d < 0:               # counter wrapped (or was reset) between samples
                    d += WRAP
                    self.wraps += 1
                self.total[key][k] += d
                row[f"{key}:{k}_octets"] = d * 4
        self.prev = cur
        self.samples += 1
        if self._series:
            self._series.write(json.dumps(row) + "\n")

    def _run(self) -> None:
        while not self._stop.wait(self.interval):
            self._sample()

    def stop(self) -> dict:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join()
        if self.available:
            self._sample()              # final delta up to process exit
        if self._series:
            self._series.close()
        per_port = {k: {"xmit_bytes": v["xmit"] * 4, "rcv_bytes": v["rcv"] * 4}
                    for k, v in self.total.items()}
        total = sum(v["xmit_bytes"] + v["rcv_bytes"] for v in per_port.values())
        return {"available": self.available, "root": self.root, "samples": self.samples,
                "wraps_corrected": self.wraps, "per_port": per_port, "total_bytes": total}


def scan_nccl_logs(pattern: str) -> dict:
    """Count transports NCCL reported in its INFO logs (evidence, not proof of absence)."""
    counts = {"net_ib": 0, "net_socket": 0, "p2p": 0, "shm": 0, "nvls": 0}
    files = glob.glob(pattern)
    for f in files:
        try:
            for line in open(f, errors="replace"):
                if "via NET/IB" in line:
                    counts["net_ib"] += 1
                elif "via NET/Socket" in line:
                    counts["net_socket"] += 1
                elif "via P2P" in line:
                    counts["p2p"] += 1
                elif "via SHM" in line:
                    counts["shm"] += 1
                if "NVLS" in line:
                    counts["nvls"] += 1
        except OSError:
            pass
    return {"log_files": len(files), **counts}


def multinode_requested(cmd: list[str]) -> bool:
    for i, tok in enumerate(cmd):
        m = re.match(r"--nnodes(?:=(\S+))?$", tok)
        if m:
            val = m.group(1) or (cmd[i + 1] if i + 1 < len(cmd) else "1")
            if not re.fullmatch(r"1(:1)?", val.strip()):
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="W23 intra-node train + explicit IB-quiet harness",
                                 formatter_class=argparse.RawDescriptionHelpFormatter,
                                 epilog="Command to wrap goes after '--'.")
    ap.add_argument("--sidecar", default="", help="Sidecar JSON path (default data/ibquiet_<ts>.json)")
    ap.add_argument("--no-pin", action="store_true", help="Do not set the NCCL NVLink-pinning env")
    ap.add_argument("--nccl-env", action="append", default=[], metavar="KEY=VAL",
                    help="Extra/override NCCL env (repeatable)")
    ap.add_argument("--nccl-evidence", action="store_true",
                    help="NCCL_DEBUG=INFO to a per-process log; count transports afterwards")
    ap.add_argument("--ib-root", default=DEFAULT_IB_ROOT, help="IB sysfs root (override for tests)")
    ap.add_argument("--ib-sample-interval", type=float, default=0.5)
    ap.add_argument("--ib-series", default="", help="Also write raw per-sample IB deltas (JSONL)")
    ap.add_argument("--ib-quiet-threshold-bytes", type=int, default=10 * 1024 * 1024)
    ap.add_argument("--allow-multinode", action="store_true")
    ap.add_argument("cmd", nargs=argparse.REMAINDER, help="-- <command to run>")
    args = ap.parse_args()

    cmd = args.cmd[1:] if args.cmd[:1] == ["--"] else args.cmd
    if not cmd:
        ap.error("no command given (put it after '--')")
    if multinode_requested(cmd) and not args.allow_multinode:
        print("run_nvl_only_ibquiet: refusing multi-node command (traffic must cross the fabric); "
              "pass --allow-multinode for a control run", file=sys.stderr)
        return 2

    ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    sidecar = args.sidecar or os.path.join("data", f"ibquiet_{ts}.json")
    os.makedirs(os.path.dirname(sidecar) or ".", exist_ok=True)
    stem = os.path.splitext(sidecar)[0]

    env = os.environ.copy()
    applied: dict[str, str] = {}
    if not args.no_pin:
        applied.update(PIN_ENV)
    for kv in args.nccl_env:
        k, _, v = kv.partition("=")
        applied[k] = v
    nccl_pattern = ""
    if args.nccl_evidence:
        nccl_pattern = f"{stem}.nccl.*.log"
        applied.update(NCCL_DEBUG="INFO", NCCL_DEBUG_SUBSYS="INIT,NET",
                       NCCL_DEBUG_FILE=f"{stem}.nccl.%h.%p.log")
        for old in glob.glob(nccl_pattern):
            os.remove(old)
    env.update(applied)

    sampler = IbSampler(args.ib_root, args.ib_sample_interval, args.ib_series or None)
    note = {
        "kind": "intranode_train_ibquiet", "wid": "W23",
        "nvlink_expected": True, "ib_expected": False,
        "host": socket.gethostname(), "cmd": cmd, "env_applied": applied,
        "pinned": not args.no_pin, "t_start": time.time(),
        "ib_quiet_threshold_bytes": args.ib_quiet_threshold_bytes,
    }
    with open(sidecar, "w") as f:                       # intent note exists even if the run dies
        json.dump(note, f, indent=2)

    print(f"[ibquiet] NVLink-only run: {' '.join(cmd)}")
    print(f"[ibquiet] env: {applied or '(none, --no-pin)'}")
    print(f"[ibquiet] IB counters: {'sampling ' + str(len(sampler.prev)) + ' port(s)' if sampler.available else 'not readable -> verdict will be unknown'}")
    sampler.start()
    rc = 1
    try:
        rc = subprocess.run(cmd, env=env).returncode
    except KeyboardInterrupt:
        rc = 130
    finally:
        ib = sampler.stop()
        note.update(t_end=time.time(), returncode=rc, ib_counters=ib)
        if not ib["available"]:
            verdict = "unknown"
        else:
            verdict = "quiet" if ib["total_bytes"] <= args.ib_quiet_threshold_bytes else "NOT_QUIET"
        note["ib_quiet_verdict"] = verdict
        if args.nccl_evidence:
            note["nccl_transport_evidence"] = scan_nccl_logs(nccl_pattern)
        with open(sidecar, "w") as f:
            json.dump(note, f, indent=2)

    print(f"[ibquiet] rc={rc} ib_quiet_verdict={verdict} "
          f"ib_bytes={ib['total_bytes'] if ib['available'] else 'n/a'} -> {sidecar}")
    if verdict == "NOT_QUIET":
        print("[ibquiet] WARNING: IB carried traffic during an 'IB-quiet' run "
              "(background NFS/mgmt, or NCCL used the network) -- check nccl_transport_evidence",
              file=sys.stderr)
    return rc


if __name__ == "__main__":
    sys.exit(main())
