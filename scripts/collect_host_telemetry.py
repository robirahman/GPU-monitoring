#!/usr/bin/env python3
"""SPAR host-telemetry sidecar — per-NIC bytes, CPU, and host memory at ~1 Hz.

The NVML/DCGM GPU collector cannot see the inter-server communication channel:
on a multi-node job, NCCL traffic between two servers leaves each GPU over the
NIC, and NVML reports only the GPU's own PCIe link. This sidecar records the
host network counters (from /proc/net/dev), CPU utilisation (/proc/stat), and
host memory (/proc/meminfo) so an all-reduce shows up as node0.tx ~= node1.rx
-- a clean signature that distinguishes cross-server training from replicated
inference (which has no cross-node traffic at all).

Mirrors the TelemetryCollector / *PowerLogger interface (start/stop/save) and
its conventions: 1 Hz background thread, timestamp_utc/timestamp_epoch columns,
shared run_id for joining to GPU telemetry, -1 per-field sentinel on a read
error, whole-row NaN on a catastrophic sample, atomic .partial checkpoints.

Standalone usage:
    python scripts/collect_host_telemetry.py --duration 600 \
        --output data/nic_test.parquet --run-id abc123 --label s16_train_fsdp7b

As a library / sidecar (started by the multinode queue alongside the GPU run):
    c = HostTelemetryCollector(run_id=RID, workload_label=LABEL, node_rank=R)
    c.start(); ...; c.stop(); c.save("data/nic_...parquet")
"""

import argparse
import logging
import os
import socket
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("spar.host")

# Bytes -> MiB, matching the GPU collector's *_mbps convention (it divides
# NVML KB/s by 1024, i.e. reports MiB/s). Keeping the same unit means NIC and
# PCIe rates are directly comparable in a feature vector.
_MIB = 1024.0 * 1024.0
# UNITS. Every `nic_*_mbps` column below is bytes/2**20/dt -- MEBIBYTES PER
# SECOND, not megabits. The name is wrong and has been since the collector was
# written; it is kept because 423 nic_*.parquet in the corpus carry it and a
# rename would split the schema for no analytical gain.
#
# Nothing published so far is affected: every comparison drawn from these
# columns is a RATIO (FSDP ~440x sharded inference), and ratios do not care
# about the unit. What it does affect is any ABSOLUTE comparison -- e.g. against
# the 100 Mbps consumer link in arXiv 2605.29359. Read straight off the column,
# `train_pipe 513` was reported as 513 Mbps; the true figure is 513 MiB/s, i.e.
# 4302 Mbps, 8.4x higher.
#
# Use MBPS_PER_MIBPS for anything that compares against a bitrate.
MBPS_PER_MIBPS = 8.0 * 1024.0 * 1024.0 / 1e6      # 8.389
def mibps_to_mbps(v):
    """MiB/s (what these columns hold) -> Mbit/s (what link rates are quoted in)."""
    return v * MBPS_PER_MIBPS
def mbps_to_mibps(v):
    """Mbit/s -> MiB/s. 100 Mbps -> 11.92, the shaping verification threshold."""
    return v / MBPS_PER_MIBPS


def _read_proc_net_dev() -> dict:
    """iface -> (rx_bytes, rx_packets, tx_bytes, tx_packets) from /proc/net/dev."""
    out = {}
    with open("/proc/net/dev") as fh:
        for line in fh:
            if ":" not in line:
                continue  # header rows
            name, _, rest = line.partition(":")
            name = name.strip()
            f = rest.split()
            # columns: rx: bytes packets errs drop fifo frame compressed multicast
            #          tx: bytes packets errs drop fifo colls carrier compressed
            out[name] = (int(f[0]), int(f[1]), int(f[8]), int(f[9]))
    return out


def _read_cpu_jiffies() -> tuple:
    """(idle_jiffies, total_jiffies) from the aggregate 'cpu' line of /proc/stat."""
    with open("/proc/stat") as fh:
        for line in fh:
            if line.startswith("cpu "):
                vals = [int(x) for x in line.split()[1:]]
                idle = vals[3] + (vals[4] if len(vals) > 4 else 0)  # idle + iowait
                return idle, sum(vals)
    return 0, 0


def _read_meminfo() -> tuple:
    """(used_mb, total_mb) from /proc/meminfo (used = total - available)."""
    total_kb = avail_kb = None
    with open("/proc/meminfo") as fh:
        for line in fh:
            if line.startswith("MemTotal:"):
                total_kb = int(line.split()[1])
            elif line.startswith("MemAvailable:"):
                avail_kb = int(line.split()[1])
            if total_kb is not None and avail_kb is not None:
                break
    if total_kb is None or avail_kb is None:
        return -1.0, -1.0
    return (total_kb - avail_kb) / 1024.0, total_kb / 1024.0


def _default_iface() -> str | None:
    """The default-route interface (gVNIC eth0/ens4 on GCE), or the busiest NIC."""
    try:
        with open("/proc/net/route") as fh:
            next(fh)  # header
            for line in fh:
                f = line.split()
                if len(f) > 1 and f[1] == "00000000":  # destination 0.0.0.0
                    return f[0]
    except OSError:
        pass
    # Fall back to the non-loopback iface with the most rx bytes.
    try:
        stats = _read_proc_net_dev()
        cand = {k: v for k, v in stats.items() if k != "lo"}
        if cand:
            return max(cand, key=lambda k: cand[k][0])
    except OSError:
        pass
    return None


class HostTelemetryCollector:
    """Polls host NIC/CPU/memory counters at a fixed interval in a daemon thread."""

    def __init__(
        self,
        run_id: str | None = None,
        workload_label: str = "unknown",
        node_rank: int = 0,
        iface: str | None = None,
        interval_sec: float = 1.0,
        checkpoint_sec: float = 300.0,
        checkpoint_path: str | None = None,
    ):
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._workload_label = workload_label
        self._node_rank = node_rank
        self._hostname = socket.gethostname()
        self._iface = iface or _default_iface()
        self._interval = interval_sec
        self._checkpoint_sec = checkpoint_sec
        self.checkpoint_path = checkpoint_path

        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        # Previous cumulative counters, seeded on the first sample so that the
        # first row reports rate = 0 rather than a huge bytes-since-boot spike.
        self._prev_net: dict | None = None
        self._prev_cpu: tuple | None = None
        self._prev_t: float | None = None
        self._last_checkpoint = 0.0
        self._checkpoint_count = 0

    # ---- sampling -----------------------------------------------------------

    def _sample(self) -> dict:
        now = time.time()
        row = {
            "timestamp_utc": datetime.now(timezone.utc).isoformat(),
            "timestamp_epoch": now,
            "hostname": self._hostname,
            "run_id": self._run_id,
            "workload_label": self._workload_label,
            "node_rank": self._node_rank,
            "iface": self._iface if self._iface else "none",
        }

        # dt for rate calculations; None on the seeding sample.
        dt = (now - self._prev_t) if self._prev_t else None
        self._prev_t = now

        # --- network counters ---
        try:
            net = _read_proc_net_dev()
        except OSError:
            net = None

        if net is None or dt is None or self._prev_net is None:
            # Seeding sample (or read failure): emit 0 rates, don't sentinel --
            # a rate genuinely is undefined until we have two readings.
            for k in ("nic_rx_mbps", "nic_tx_mbps", "nic_rx_pps", "nic_tx_pps",
                      "nic_total_rx_mbps", "nic_total_tx_mbps"):
                row[k] = 0.0 if net is not None else -1.0
        else:
            def rate(cur, prev, scale):
                return max(0.0, (cur - prev) / scale / dt)  # counters can wrap/reset

            nic = self._iface
            if nic in net and nic in self._prev_net:
                c, p = net[nic], self._prev_net[nic]
                row["nic_rx_mbps"] = rate(c[0], p[0], _MIB)
                row["nic_tx_mbps"] = rate(c[2], p[2], _MIB)
                row["nic_rx_pps"] = rate(c[1], p[1], 1.0)
                row["nic_tx_pps"] = rate(c[3], p[3], 1.0)
            else:
                row["nic_rx_mbps"] = row["nic_tx_mbps"] = -1.0
                row["nic_rx_pps"] = row["nic_tx_pps"] = -1.0
            # Sum over every non-loopback iface, so GPUDirect-TCPX / RDMA NICs
            # are captured even when the default route is the gVNIC.
            trx = ttx = 0.0
            for name, c in net.items():
                if name == "lo" or name not in self._prev_net:
                    continue
                p = self._prev_net[name]
                trx += rate(c[0], p[0], _MIB)
                ttx += rate(c[2], p[2], _MIB)
            row["nic_total_rx_mbps"] = trx
            row["nic_total_tx_mbps"] = ttx
        if net is not None:
            self._prev_net = net

        # --- CPU utilisation ---
        try:
            cpu = _read_cpu_jiffies()
        except OSError:
            cpu = None
        if cpu is None:
            row["cpu_util_pct"] = -1.0
        elif self._prev_cpu is None:
            row["cpu_util_pct"] = 0.0
        else:
            d_idle = cpu[0] - self._prev_cpu[0]
            d_total = cpu[1] - self._prev_cpu[1]
            row["cpu_util_pct"] = (100.0 * (1.0 - d_idle / d_total)
                                   if d_total > 0 else 0.0)
        if cpu is not None:
            self._prev_cpu = cpu

        # --- host memory ---
        try:
            used_mb, total_mb = _read_meminfo()
        except OSError:
            used_mb, total_mb = -1.0, -1.0
        row["mem_used_mb"] = used_mb
        row["mem_total_mb"] = total_mb

        return row

    def _poll_loop(self):
        self._last_checkpoint = time.monotonic()
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                row = self._sample()
            except Exception as e:  # never let one bad read kill the thread
                log.warning("host sample error: %s", e)
                row = {"timestamp_epoch": time.time(), "run_id": self._run_id,
                       "workload_label": self._workload_label,
                       "node_rank": self._node_rank, "hostname": self._hostname}
            self._records.append(row)

            if (self.checkpoint_path
                    and time.monotonic() - self._last_checkpoint >= self._checkpoint_sec):
                self._write_checkpoint()
                self._last_checkpoint = time.monotonic()

            sleep_time = max(0.0, self._interval - (time.monotonic() - t0))
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    # ---- lifecycle ----------------------------------------------------------

    def start(self):
        if self._thread is not None:
            return
        log.info("Host telemetry started (host=%s, iface=%s, interval=%.1fs, run_id=%s)",
                 self._hostname, self._iface, self._interval, self._run_id[:12])
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def _write_checkpoint(self):
        if not self._records:
            return
        try:
            path = Path(self.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            self.to_dataframe().to_parquet(tmp, index=False)
            os.replace(tmp, path)
            self._checkpoint_count += 1
        except Exception as e:
            log.warning("host checkpoint failed (continuing): %s", e)

    def save(self, output_path: str, fmt: str = "parquet"):
        df = self.to_dataframe()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet":
            df.to_parquet(str(path), index=False)
        else:
            df.to_csv(str(path), index=False)
        log.info("Saved %d host rows to %s", len(df), path)
        if self.checkpoint_path:
            try:
                Path(self.checkpoint_path).unlink(missing_ok=True)
            except OSError:
                pass
        return str(path)


def main():
    ap = argparse.ArgumentParser(description="Host NIC/CPU/memory telemetry sidecar")
    ap.add_argument("--duration", type=int, default=60, help="Seconds to collect")
    ap.add_argument("--output", type=str, required=True)
    ap.add_argument("--run-id", type=str, default=None,
                    help="Shared run_id joining this to the job's GPU telemetry")
    ap.add_argument("--label", type=str, default="unknown")
    ap.add_argument("--node-rank", type=int,
                    default=int(os.environ.get("GROUP_RANK",
                                os.environ.get("NODE_RANK", 0))))
    ap.add_argument("--iface", type=str, default=None,
                    help="NIC to report as the primary (default: default route)")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--checkpoint-sec", type=float, default=60.0)
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    ckpt = str(Path(args.output).with_suffix(".parquet.partial"))
    c = HostTelemetryCollector(
        run_id=args.run_id, workload_label=args.label, node_rank=args.node_rank,
        iface=args.iface, interval_sec=args.interval,
        checkpoint_sec=args.checkpoint_sec, checkpoint_path=ckpt,
    )
    c.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        log.warning("interrupted -- saving what we have")
    finally:
        c.stop()
        c.save(args.output)


if __name__ == "__main__":
    main()
