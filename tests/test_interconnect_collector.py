"""Unit tests for scripts/interconnect_collect.py fabric sampling (no GPU needed)."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from scripts.interconnect_collect import (  # noqa: E402
    IB_WRAP,
    EthDeltaSampler,
    IbDeltaSampler,
    SharedFabricCache,
    infer_topology,
    read_ib_counters,
)


def _fake_ib(root: Path, xmit: int, rcv: int, dev: str = "mlx5_0", port: str = "1"):
    d = root / dev / "ports" / port / "counters"
    d.mkdir(parents=True, exist_ok=True)
    (d / "port_xmit_data").write_text(str(xmit))
    (d / "port_rcv_data").write_text(str(rcv))
    return d


def _fake_eth(root: Path, name: str, tx: int, rx: int):
    d = root / name / "statistics"
    d.mkdir(parents=True, exist_ok=True)
    (d / "tx_bytes").write_text(str(tx))
    (d / "rx_bytes").write_text(str(rx))
    return d


def test_read_ib_none_without_sysfs(tmp_path):
    assert read_ib_counters(str(tmp_path / "missing")) is None


def test_ib_delta_wraparound(tmp_path):
    d = _fake_ib(tmp_path, IB_WRAP - 10, 100)
    s = IbDeltaSampler(str(tmp_path))
    assert s.available
    # Bootstrap sample discards first delta when used via SharedFabricCache;
    # here we call sample() directly after priming prev in __post_init__.
    (d / "port_xmit_data").write_text("6")  # +16 units = 64 B
    (d / "port_rcv_data").write_text("1000100")  # +1e6 units = 4 MB
    out = s.sample()
    assert out["ib_available"] is True
    assert out["ib_tx_bytes"] == 64
    assert out["ib_rx_bytes"] == 4_000_000
    assert s.wraps == 1
    ports = json.loads(out["ib_ports_json"])
    assert ports["mlx5_0/1"] == {"tx_bytes": 64, "rx_bytes": 4_000_000}


def test_shared_fabric_cache_rates_and_no_double_count(tmp_path):
    d = _fake_ib(tmp_path, 1000, 2000)
    cache = SharedFabricCache(interval_sec=0.05, ib_root=str(tmp_path), enable_eth=False)
    time.sleep(0.06)
    (d / "port_xmit_data").write_text(str(1000 + 2500))  # +2500 units = 10_000 B
    (d / "port_rcv_data").write_text(str(2000 + 2500))
    a = cache.snapshot()
    b = cache.snapshot()  # within 40% interval → same sample, no extra delta
    assert a["ib_tx_bytes"] == 10_000
    assert b["ib_tx_bytes"] == 10_000  # shared, not re-sampled
    assert a["ib_tx_mbps"] > 0
    # After enough time, next refresh sees zero delta
    time.sleep(0.06)
    c = cache.snapshot()
    assert c["ib_tx_bytes"] == 0
    summary = cache.summary()
    assert summary["ib_total_tx_bytes"] == 10_000
    assert summary["samples"] >= 2


def test_eth_sampler_skips_lo(tmp_path):
    _fake_eth(tmp_path, "lo", 999, 999)
    _fake_eth(tmp_path, "eth0", 100, 200)
    s = EthDeltaSampler(str(tmp_path))
    assert s.available
    _fake_eth(tmp_path, "eth0", 100 + 1_048_576, 200 + 2_097_152)
    out = s.sample()
    assert out["eth_tx_bytes"] == 1_048_576
    assert out["eth_rx_bytes"] == 2_097_152
    assert out["eth_n_ifaces"] == 1


def test_infer_topology_from_env(monkeypatch):
    monkeypatch.setenv("SPAR_NNODES", "4")
    monkeypatch.setenv("SPAR_NODE_RANK", "2")
    monkeypatch.setenv("SPAR_NPROC_PER_NODE", "8")
    topo = infer_topology()
    assert topo["nnodes"] == 4
    assert topo["node_rank"] == 2
    assert topo["nproc_per_node"] == 8
    assert topo["world_size"] == 32
    assert topo["hostname"]


def test_infer_topology_cli_overrides_env(monkeypatch):
    monkeypatch.setenv("SPAR_NNODES", "99")
    topo = infer_topology(nnodes=2, node_rank=0, nproc_per_node=8, world_size=16)
    assert topo["nnodes"] == 2
    assert topo["world_size"] == 16
