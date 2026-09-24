#!/usr/bin/env python3
"""Node + GPU interconnect counters for SPAR train/infer classification.

Fuses into TelemetryCollector rows (see collect_telemetry.py). Designed for both
single-node and multi-node: each node runs its own collector process and stamps
topology metadata from SLURM / torchrun / CLI.

Signals (best-effort; missing hardware -> NaN / available=False, never crash):

  * InfiniBand / RoCE sysfs  ``/sys/class/infiniband/*/ports/*/counters``
  * Ethernet NIC stats        ``/sys/class/net/*/statistics/{tx,rx}_bytes``
    (skips lo / docker / veth by default — background noise floor)
  * NVLink via NVML field IDs (per-GPU TX/RX KiB counters → MB/s deltas)
  * Topology stamp            hostname, nnodes, node_rank, world_size, …

Multi-GPU launchers MUST share one ``SharedFabricCache`` across GPU collectors so
IB/eth deltas are not double-consumed by concurrent poll threads.
"""

from __future__ import annotations

import glob
import logging
import os
import socket
import threading
import time
from dataclasses import dataclass, field
from typing import Any

log = logging.getLogger("spar.interconnect")

DEFAULT_IB_ROOT = "/sys/class/infiniband"
DEFAULT_NET_ROOT = "/sys/class/net"
IB_COUNTER_FILES = {"xmit": "port_xmit_data", "rcv": "port_rcv_data"}
IB_WRAP = 2**32  # many drivers expose 32-bit port_*_data; unit = 4 octets
# Skip virtual / loopback NICs for eth rates (mgmt still leaks via physical NIC).
_ETH_SKIP_PREFIXES = ("lo", "docker", "veth", "br-", "virbr", "tun", "tap", "cni", "flannel", "cali")


# ── topology ────────────────────────────────────────────────────────────────

def _env_int(*keys: str, default: int | None = None) -> int | None:
    for k in keys:
        v = os.environ.get(k)
        if v is None or v == "":
            continue
        try:
            return int(v)
        except ValueError:
            continue
    return default


def infer_topology(
    nnodes: int | None = None,
    node_rank: int | None = None,
    nproc_per_node: int | None = None,
    world_size: int | None = None,
) -> dict[str, Any]:
    """Best-effort multi-node topology from CLI overrides + launcher env."""
    nnodes = nnodes if nnodes is not None else _env_int(
        "SPAR_NNODES", "SLURM_NNODES", "NNODES", "GROUP_WORLD_SIZE", default=1
    )
    node_rank = node_rank if node_rank is not None else _env_int(
        "SPAR_NODE_RANK", "SLURM_NODEID", "SLURM_PROCID", "NODE_RANK",
        "GROUP_RANK", default=0,
    )
    nproc_per_node = nproc_per_node if nproc_per_node is not None else _env_int(
        "SPAR_NPROC_PER_NODE", "SLURM_GPUS_ON_NODE", "LOCAL_WORLD_SIZE",
        "NPROC_PER_NODE", default=None,
    )
    world_size = world_size if world_size is not None else _env_int(
        "SPAR_WORLD_SIZE", "WORLD_SIZE", "SLURM_NTASKS", default=None,
    )
    if world_size is None and nnodes and nproc_per_node:
        world_size = int(nnodes) * int(nproc_per_node)

    return {
        "hostname": socket.gethostname(),
        "nnodes": int(nnodes) if nnodes is not None else 1,
        "node_rank": int(node_rank) if node_rank is not None else 0,
        "nproc_per_node": int(nproc_per_node) if nproc_per_node is not None else -1,
        "world_size": int(world_size) if world_size is not None else -1,
        "job_id": os.environ.get("SLURM_JOB_ID")
        or os.environ.get("SPAR_JOB_ID")
        or "",
    }


# ── InfiniBand / RoCE ───────────────────────────────────────────────────────

def read_ib_counters(root: str = DEFAULT_IB_ROOT) -> dict[str, dict[str, int]] | None:
    """``{"mlx5_0/1": {"xmit": raw, "rcv": raw}}`` or None if nothing readable."""
    out: dict[str, dict[str, int]] = {}
    pattern = os.path.join(root, "*", "ports", "*", "counters", IB_COUNTER_FILES["xmit"])
    for xmit_path in sorted(glob.glob(pattern)):
        parts = xmit_path.split(os.sep)
        key = f"{parts[-5]}/{parts[-3]}"
        cdir = os.path.dirname(xmit_path)
        try:
            vals = {
                k: int(open(os.path.join(cdir, fn)).read().strip())
                for k, fn in IB_COUNTER_FILES.items()
            }
        except (OSError, ValueError):
            continue
        out[key] = vals
    return out or None


@dataclass
class IbDeltaSampler:
    """Wrap-safe IB byte-delta sampler (port_*_data unit = 4 octets)."""

    root: str = DEFAULT_IB_ROOT
    prev: dict[str, dict[str, int]] | None = field(default=None, init=False)
    wraps: int = field(default=0, init=False)

    def __post_init__(self) -> None:
        self.prev = read_ib_counters(self.root)

    @property
    def available(self) -> bool:
        return self.prev is not None

    def sample(self) -> dict[str, Any]:
        """Return rates + absolute deltas since last call. Empty/NaN if unavailable."""
        nan = float("nan")
        if not self.available:
            return {
                "ib_available": False,
                "ib_n_ports": 0,
                "ib_tx_bytes": 0,
                "ib_rx_bytes": 0,
                "ib_tx_mbps": nan,
                "ib_rx_mbps": nan,
                "ib_ports_json": "{}",
            }

        t0 = time.monotonic()
        cur = read_ib_counters(self.root)
        dt = max(time.monotonic() - t0, 1e-6)
        # Prefer wall interval from caller via SharedFabricCache; local dt is
        # only the read latency. Real dt is applied by SharedFabricCache.
        if cur is None or self.prev is None:
            return {
                "ib_available": False,
                "ib_n_ports": 0,
                "ib_tx_bytes": 0,
                "ib_rx_bytes": 0,
                "ib_tx_mbps": nan,
                "ib_rx_mbps": nan,
                "ib_ports_json": "{}",
            }

        per_port: dict[str, dict[str, int]] = {}
        tx_b = rx_b = 0
        for key, vals in cur.items():
            if key not in self.prev:
                continue
            port_tx = port_rx = 0
            for k, out_key in (("xmit", "tx"), ("rcv", "rx")):
                d = vals[k] - self.prev[key][k]
                if d < 0:
                    d += IB_WRAP
                    self.wraps += 1
                nbytes = d * 4
                if out_key == "tx":
                    port_tx = nbytes
                    tx_b += nbytes
                else:
                    port_rx = nbytes
                    rx_b += nbytes
            per_port[key] = {"tx_bytes": port_tx, "rx_bytes": port_rx}
        self.prev = cur

        # Rates filled by SharedFabricCache with true inter-sample dt.
        import json

        return {
            "ib_available": True,
            "ib_n_ports": len(per_port),
            "ib_tx_bytes": tx_b,
            "ib_rx_bytes": rx_b,
            "ib_tx_mbps": nan,  # placeholder; overwritten with real dt
            "ib_rx_mbps": nan,
            "ib_ports_json": json.dumps(per_port, sort_keys=True),
            "_dt_read": dt,  # internal
        }


# ── Ethernet (fallback fabric when no IB) ───────────────────────────────────

def _eth_ifaces(root: str = DEFAULT_NET_ROOT) -> list[str]:
    ifaces = []
    try:
        names = sorted(os.listdir(root))
    except OSError:
        return []
    for name in names:
        if any(name == p or name.startswith(p) for p in _ETH_SKIP_PREFIXES):
            continue
        tx = os.path.join(root, name, "statistics", "tx_bytes")
        if os.path.isfile(tx):
            ifaces.append(name)
    return ifaces


def read_eth_counters(root: str = DEFAULT_NET_ROOT) -> dict[str, dict[str, int]] | None:
    out: dict[str, dict[str, int]] = {}
    for name in _eth_ifaces(root):
        try:
            tx = int(open(os.path.join(root, name, "statistics", "tx_bytes")).read().strip())
            rx = int(open(os.path.join(root, name, "statistics", "rx_bytes")).read().strip())
        except (OSError, ValueError):
            continue
        out[name] = {"tx": tx, "rx": rx}
    return out or None


@dataclass
class EthDeltaSampler:
    root: str = DEFAULT_NET_ROOT
    prev: dict[str, dict[str, int]] | None = field(default=None, init=False)

    def __post_init__(self) -> None:
        self.prev = read_eth_counters(self.root)

    @property
    def available(self) -> bool:
        return self.prev is not None

    def sample(self) -> dict[str, Any]:
        nan = float("nan")
        if not self.available:
            return {
                "eth_available": False,
                "eth_n_ifaces": 0,
                "eth_tx_bytes": 0,
                "eth_rx_bytes": 0,
                "eth_tx_mbps": nan,
                "eth_rx_mbps": nan,
            }
        cur = read_eth_counters(self.root)
        if cur is None or self.prev is None:
            return {
                "eth_available": False,
                "eth_n_ifaces": 0,
                "eth_tx_bytes": 0,
                "eth_rx_bytes": 0,
                "eth_tx_mbps": nan,
                "eth_rx_mbps": nan,
            }
        tx_b = rx_b = 0
        for name, vals in cur.items():
            if name not in self.prev:
                continue
            dtx = vals["tx"] - self.prev[name]["tx"]
            drx = vals["rx"] - self.prev[name]["rx"]
            if dtx < 0:
                dtx = vals["tx"]  # counter reset
            if drx < 0:
                drx = vals["rx"]
            tx_b += dtx
            rx_b += drx
        self.prev = cur
        return {
            "eth_available": True,
            "eth_n_ifaces": len(cur),
            "eth_tx_bytes": tx_b,
            "eth_rx_bytes": rx_b,
            "eth_tx_mbps": nan,
            "eth_rx_mbps": nan,
        }


# ── NVLink (per-GPU, NVML) ──────────────────────────────────────────────────

class GpuNvlinkSampler:
    """Per-GPU NVLink TX/RX via NVML field values (KiB counters → byte deltas)."""

    def __init__(self, gpu_index: int = 0):
        self.gpu_index = gpu_index
        self._handle = None
        self._prev_tx: int | None = None
        self._prev_rx: int | None = None
        self._available = False
        self._n_links = 0
        try:
            import pynvml

            pynvml.nvmlInit()
            self._pynvml = pynvml
            self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
            # Probe link count
            for i in range(18):
                try:
                    st = pynvml.nvmlDeviceGetNvLinkState(self._handle, i)
                    if st == pynvml.NVML_FEATURE_ENABLED:
                        self._n_links += 1
                except pynvml.NVMLError:
                    break
            tx, rx = self._read_raw()
            if tx is not None:
                self._prev_tx, self._prev_rx = tx, rx
                self._available = True
        except Exception as e:
            log.debug("NVLink sampler unavailable on GPU %d: %s", gpu_index, e)
            self._available = False

    def _read_raw(self) -> tuple[int | None, int | None]:
        pynvml = self._pynvml
        try:
            fields = pynvml.nvmlDeviceGetFieldValues(
                self._handle,
                [
                    pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_TX,
                    pynvml.NVML_FI_DEV_NVLINK_THROUGHPUT_DATA_RX,
                ],
            )
            tx = fields[0].value.ullVal if fields[0].nvmlReturn == 0 else None
            rx = fields[1].value.ullVal if fields[1].nvmlReturn == 0 else None
            if tx is None and rx is None:
                return None, None
            # NVML documents these as KiB; treat as cumulative KiB.
            return int(tx or 0) * 1024, int(rx or 0) * 1024
        except (AttributeError, pynvml.NVMLError, Exception):
            return None, None

    @property
    def available(self) -> bool:
        return self._available

    def sample(self, dt_sec: float) -> dict[str, Any]:
        nan = float("nan")
        base = {
            "nvlink_available": self._available,
            "nvlink_n_links": self._n_links,
            "nvlink_tx_bytes": 0,
            "nvlink_rx_bytes": 0,
            "nvlink_tx_mbps": nan,
            "nvlink_rx_mbps": nan,
        }
        if not self._available:
            return base
        tx, rx = self._read_raw()
        if tx is None or self._prev_tx is None:
            return base
        dtx = tx - self._prev_tx
        drx = rx - (self._prev_rx or 0)
        if dtx < 0:
            dtx = tx  # reset
        if drx < 0:
            drx = rx
        self._prev_tx, self._prev_rx = tx, rx
        dt = max(dt_sec, 1e-6)
        return {
            "nvlink_available": True,
            "nvlink_n_links": self._n_links,
            "nvlink_tx_bytes": dtx,
            "nvlink_rx_bytes": drx,
            "nvlink_tx_mbps": (dtx / dt) / (1024 * 1024),
            "nvlink_rx_mbps": (drx / dt) / (1024 * 1024),
        }


# ── Shared node fabric cache (one per process) ──────────────────────────────

class SharedFabricCache:
    """Thread-safe node-level IB+eth sampler. One instance shared by all GPU collectors."""

    def __init__(
        self,
        interval_sec: float = 1.0,
        ib_root: str = DEFAULT_IB_ROOT,
        net_root: str = DEFAULT_NET_ROOT,
        enable_ib: bool = True,
        enable_eth: bool = True,
    ):
        self._interval = interval_sec
        self._lock = threading.Lock()
        self._ib = IbDeltaSampler(ib_root) if enable_ib else None
        self._eth = EthDeltaSampler(net_root) if enable_eth else None
        self._last: dict[str, Any] = {}
        self._last_mono = 0.0
        self._series: list[dict[str, Any]] = []
        # Prime so first snapshot after ~interval reports a real delta window.
        self._prime()

    def _prime(self) -> None:
        if self._ib and self._ib.available:
            self._ib.sample()  # discard bootstrap delta
        if self._eth and self._eth.available:
            self._eth.sample()
        self._last_mono = time.monotonic()
        self._last = self._empty_snapshot(available_probe=True)

    def _empty_snapshot(self, available_probe: bool = False) -> dict[str, Any]:
        nan = float("nan")
        ib_ok = bool(self._ib and self._ib.available) if available_probe else False
        eth_ok = bool(self._eth and self._eth.available) if available_probe else False
        return {
            "ib_available": ib_ok,
            "ib_n_ports": 0,
            "ib_tx_bytes": 0,
            "ib_rx_bytes": 0,
            "ib_tx_mbps": nan,
            "ib_rx_mbps": nan,
            "ib_ports_json": "{}",
            "eth_available": eth_ok,
            "eth_n_ifaces": 0,
            "eth_tx_bytes": 0,
            "eth_rx_bytes": 0,
            "eth_tx_mbps": nan,
            "eth_rx_mbps": nan,
        }

    def snapshot(self) -> dict[str, Any]:
        """Refresh at most once per ~interval; concurrent callers share the same sample."""
        with self._lock:
            now = time.monotonic()
            # Allow refresh if ≥ 40% of interval elapsed (multi-GPU staggered polls).
            if self._last and (now - self._last_mono) < self._interval * 0.4:
                return dict(self._last)

            dt = max(now - self._last_mono, 1e-6)
            row = self._empty_snapshot()
            if self._ib is not None:
                ib = self._ib.sample()
                ib.pop("_dt_read", None)
                row.update(ib)
                if row.get("ib_available"):
                    row["ib_tx_mbps"] = (row["ib_tx_bytes"] / dt) / (1024 * 1024)
                    row["ib_rx_mbps"] = (row["ib_rx_bytes"] / dt) / (1024 * 1024)
            if self._eth is not None:
                eth = self._eth.sample()
                row.update(eth)
                if row.get("eth_available"):
                    row["eth_tx_mbps"] = (row["eth_tx_bytes"] / dt) / (1024 * 1024)
                    row["eth_rx_mbps"] = (row["eth_rx_bytes"] / dt) / (1024 * 1024)

            row["fabric_dt_sec"] = dt
            row["fabric_timestamp_epoch"] = time.time()
            self._last = row
            self._last_mono = now
            self._series.append(dict(row))
            return dict(row)

    @property
    def series(self) -> list[dict[str, Any]]:
        with self._lock:
            return list(self._series)

    def summary(self) -> dict[str, Any]:
        ser = self.series
        if not ser:
            return {"samples": 0}
        ib_tx = sum(r.get("ib_tx_bytes", 0) or 0 for r in ser)
        ib_rx = sum(r.get("ib_rx_bytes", 0) or 0 for r in ser)
        eth_tx = sum(r.get("eth_tx_bytes", 0) or 0 for r in ser)
        eth_rx = sum(r.get("eth_rx_bytes", 0) or 0 for r in ser)
        return {
            "samples": len(ser),
            "ib_available": any(r.get("ib_available") for r in ser),
            "eth_available": any(r.get("eth_available") for r in ser),
            "ib_total_tx_bytes": ib_tx,
            "ib_total_rx_bytes": ib_rx,
            "ib_total_bytes": ib_tx + ib_rx,
            "eth_total_tx_bytes": eth_tx,
            "eth_total_rx_bytes": eth_rx,
            "ib_wraps_corrected": getattr(self._ib, "wraps", 0) if self._ib else 0,
        }
