#!/usr/bin/env python3
"""SPAR TPU Telemetry Collector -- GPU-schema-compatible, one parquet per chip.

Rewrite of the old single-chip scaffold. It writes the SAME 20 column names the
NVIDIA collector writes (scripts/collect_telemetry.py) so the classifier ingests
TPU parquets unchanged, plus the TPU-native communication columns that are the
whole point of the TPU rung -- HLO exec timing and DCN/collective latency, the
cross-server channel NVIDIA lacks.

Column contract (plan section 2.4):
  GPU-schema (always present):
    gpu_utilization_pct  <- duty_cycle_pct           (0.0 when dark, NOT -1)
    mem_utilization_pct  <- HBM bandwidth util        (0.0 when dark)
    mem_used_mb / mem_total_mb <- HBM capacity         (0.0 when dark)
    power_draw_w, temperature_c, sm_clock_mhz, mem_clock_mhz,
    pcie_tx_mbps, pcie_rx_mbps, encoder_util_pct, decoder_util_pct,
    fan_speed_pct  =  -1 CONSTANT   (never exposed on TPU; matches AMD-port)
    workload_label, run_id
    gpu_name = "Google TPU v6e", gpu_uuid, driver_version (= libtpu version)
  Identity (strings -> auto-excluded from features by get_feature_cols):
    hostname, worker_id, chip_index, tpu_name, tpu_type
  TPU-native comm/telemetry (numeric -> auto-picked by get_feature_cols;
  -1 when the metric is unsupported/dark, a filterable level channel):
    hlo_exec_us_p50/p90/p99, hlo_queue_size,
    dcn_collective_lat_us_p50/p90/p99, buffer_xfer_lat_us_p90,
    tpu_duty_cycle_pct, tensorcore_util_pct, hbm_bw_util_pct,
    tpu_metrics_live (0/1)
  Host (psutil): host_cpu_util_pct, host_mem_used_mb, host_mem_total_mb.

Sources, tried in priority order (all guarded; availability recorded in the
parquet footer):
  1. libtpu.sdk.tpumonitoring  -- list_supported_metrics() + get_metric(n).data()
  2. tpu_info (python modules)
  3. tpu-info (CLI)
  4. psutil                    -- host cpu / memory only

HONESTY: inside one slice, cross-host collectives ride ICI and the DCN-specific
metrics may stay dark (-1). hlo_exec_timing is the ALWAYS-ON comm channel
(collective time is inside HLO exec). The optional tpuTMS (2x v6e-8 multislice)
rung guarantees the DCN metrics fire. power/temp/clock/pcie are never exposed
and are hard-wired to -1.

Standalone probe:
    python scripts/collect_tpu_telemetry.py --duration 60 --label tpuT4_idle \
        --worker-id 0 --output-dir data/
"""

import argparse
import json
import logging
import os
import socket
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

try:
    import numpy as np
except Exception:  # pragma: no cover
    np = None

log = logging.getLogger("spar.tpu_telemetry")

# Hardware identity. Was hardcoded to v6e, which would have mislabelled every row
# of a v5e run -- and gpu_name is the column the vendor/hardware split keys on
# downstream (GPU_NAME_ALIASES, per-hardware grouping, the _vendor_of helpers), so
# a wrong value here silently pools two different accelerators. Overridable
# because which chip we get depends on quota: v6e-16 was refused in this project
# ("User does not have permission to submit requests into this queue for
# accelerator type v6e-16"), so the rung actually runs on v5litepod-16.
_ACCEL = os.environ.get("SPAR_TPU_ACCEL", "v6e")
GPU_NAME = os.environ.get("SPAR_TPU_GPU_NAME", f"Google TPU {_ACCEL}")
GPU_SHORT = GPU_NAME.replace(" ", "_")

# -1 constant columns: physical channels TPU never exposes at the host level.
_NEG1_CONST = [
    "power_draw_w", "temperature_c", "sm_clock_mhz", "mem_clock_mhz",
    "pcie_tx_mbps", "pcie_rx_mbps", "encoder_util_pct", "decoder_util_pct",
    "fan_speed_pct",
]

# TPU-native comm/telemetry columns (numeric features; -1 when dark).
_TPU_NATIVE = [
    "hlo_exec_us_p50", "hlo_exec_us_p90", "hlo_exec_us_p99",
    "hlo_queue_size",
    "dcn_collective_lat_us_p50", "dcn_collective_lat_us_p90",
    "dcn_collective_lat_us_p99",
    "buffer_xfer_lat_us_p90",
    "tpu_duty_cycle_pct", "tensorcore_util_pct", "hbm_bw_util_pct",
]


# ── Scalarisation helpers ────────────────────────────────────────────────────

def _to_array(val):
    """Best-effort coerce a libtpu metric payload to a 1-D float array.

    Handles: numbers; lists/tuples/ndarrays of numbers; objects exposing a
    numeric .value/.data/.mean attribute; dicts with a 'value' key. Returns
    None when nothing numeric can be extracted.
    """
    if val is None:
        return None
    if isinstance(val, (int, float)):
        return np.array([float(val)], dtype=float)
    if isinstance(val, dict):
        for k in ("value", "values", "data", "p50", "mean"):
            if k in val:
                return _to_array(val[k])
        return None
    if isinstance(val, (list, tuple)):
        out = []
        for v in val:
            a = _to_array(v)
            if a is not None:
                out.extend(a.tolist())
        return np.array(out, dtype=float) if out else None
    # numpy array?
    if np is not None and isinstance(val, np.ndarray):
        try:
            return val.astype(float).ravel()
        except Exception:
            return None
    # object with a numeric attribute
    for attr in ("value", "mean", "data"):
        if hasattr(val, attr):
            a = getattr(val, attr)
            if callable(a):
                try:
                    a = a()
                except Exception:
                    continue
            r = _to_array(a)
            if r is not None:
                return r
    # last resort: float()
    try:
        return np.array([float(val)], dtype=float)
    except Exception:
        return None


def _percentiles(val):
    """(p50, p90, p99) from a distribution payload; (-1,-1,-1) if empty/dark."""
    a = _to_array(val)
    if a is None or a.size == 0:
        return (-1.0, -1.0, -1.0)
    return (float(np.percentile(a, 50)),
            float(np.percentile(a, 90)),
            float(np.percentile(a, 99)))


def _scalar(val, default=-1.0):
    a = _to_array(val)
    if a is None or a.size == 0:
        return default
    return float(a.mean())


# ── Metric-name classification (libtpu spellings vary across versions) ───────

def _classify_metric(name: str):
    """Map a libtpu metric name to an internal key, by lowercase substring.

    Returns one of: duty, hbm_total, hbm_used, hbm_bw, tc_util, hlo_exec,
    hlo_queue, dcn_lat, buf_xfer -- or None if we don't consume it.
    Order matters: more specific spellings are checked first.
    """
    n = name.lower()
    if "duty_cycle" in n:
        return "duty"
    if "hlo_exec" in n or ("hlo" in n and "timing" in n):
        return "hlo_exec"
    if "hlo_queue" in n or ("hlo" in n and "queue" in n):
        return "hlo_queue"
    if ("collective" in n and ("latency" in n or "e2e" in n)) or "dcn" in n:
        return "dcn_lat"
    if "buffer_transfer" in n or ("buffer" in n and "latency" in n):
        return "buf_xfer"
    if "tensorcore" in n or "tensor_core" in n:
        return "tc_util"
    if "hbm" in n and ("bandwidth" in n or "bw" in n) and "util" in n:
        return "hbm_bw"
    if "hbm" in n and ("total" in n or "capacity_total" in n):
        return "hbm_total"
    if "hbm" in n and ("usage" in n or "used" in n or "capacity_usage" in n):
        return "hbm_used"
    return None


def _index_per_chip(arr, n_chips, chip):
    """Pick chip `chip`'s slice of a per-chip payload.

    If the payload has exactly one entry per local chip, index it; otherwise
    treat the whole payload as shared across chips (return it unchanged).
    """
    if isinstance(arr, (list, tuple)) and len(arr) == n_chips:
        return arr[chip]
    return arr


# ── Metric source (probes libtpu / tpu_info / tpu-info CLI once) ─────────────

class TpuMetricSource:
    """Owns the live metric handles for one host; sampled by all local chips."""

    def __init__(self, chip_count_hint: int | None = None):
        self.availability = {"libtpu": False, "tpu_info": False,
                             "tpu_info_cli": False, "psutil": False}
        self.libtpu_version = "unknown"
        self._tpumon = None
        self._libtpu_metrics: list[str] = []
        self._tpu_info = None
        self._chip_type = None
        self.n_chips = chip_count_hint or 0

        self._init_libtpu()
        self._init_grpc_metrics()
        self._init_tpu_info()
        if self.n_chips == 0:
            self.n_chips = chip_count_hint or self._probe_chip_count() or 1
        try:
            import psutil  # noqa: F401
            self.availability["psutil"] = True
        except Exception:
            pass

    # -- source 1: libtpu.sdk monitoring ------------------------------------
    def _init_libtpu(self):
        """Import the libtpu monitoring module, whatever it is called here.

        MEASURED on libtpu 0.0.17 (v5e-16, v2-alpha-tpuv5-lite): `libtpu.sdk`
        exists and exposes attributes ['monitoring', 'slice'] -- so the submodule
        is `monitoring`, NOT `tpumonitoring`. This function only ever tried
        `from libtpu.sdk import tpumonitoring`, which raises
        "cannot import name 'tpumonitoring' from 'libtpu.sdk'", and because every
        source here is guarded the collector fell silently through to psutil.
        Consequence: a full 60 s MN-I run collected 245 rows per chip in which
        EVERY accelerator metric was 0 or -1. Try both names, newest first.
        """
        tpumonitoring = None
        for modname, attr in (("libtpu.sdk", "monitoring"),
                              ("libtpu.sdk", "tpumonitoring"),
                              ("libtpu.sdk.monitoring", None),
                              ("libtpu.sdk.tpumonitoring", None)):
            try:
                import importlib
                mod = importlib.import_module(modname)
                cand = getattr(mod, attr) if attr else mod
                if cand is not None:
                    tpumonitoring = cand
                    log.info("libtpu monitoring found at %s%s", modname,
                             f".{attr}" if attr else "")
                    break
            except Exception:
                continue
        if tpumonitoring is None:
            log.info("libtpu monitoring unavailable under any known module name")
            return
        self._tpumon = tpumonitoring
        try:
            names = list(tpumonitoring.list_supported_metrics())
            # names may be enums; normalise to str
            self._libtpu_metrics = [str(getattr(n, "name", n)) for n in names]
            self.availability["libtpu"] = True
            log.info("libtpu monitoring: %d supported metrics", len(self._libtpu_metrics))
        except Exception as e:
            log.warning("libtpu list_supported_metrics failed: %s", e)
            self._tpumon = None
            return
        # libtpu version -> driver_version column
        try:
            import libtpu
            self.libtpu_version = str(getattr(libtpu, "__version__", "unknown"))
        except Exception:
            pass
        # Diagnostic: log one raw repr(data()) per metric so a human on the TPU
        # can correct the name->column mapping if a version differs.
        for name in self._libtpu_metrics:
            try:
                raw = self._tpumon.get_metric(name).data()
                log.info("libtpu metric %-32s raw=%r", name, raw)
            except Exception as e:
                log.info("libtpu metric %-32s (data() failed: %s)", name, e)

    # -- source 1b: the libtpu metrics gRPC server (the one that works) ------
    def _init_grpc_metrics(self):
        """Bind the tpu_info.metrics accessors, which all read localhost:8431.

        This is the source that actually carries TPU telemetry, and the reason
        nothing worked before. Every function in tpu_info.metrics takes
        `addr='localhost:8431'` -- the libtpu metrics gRPC server -- and that
        server is OPT-IN on the workload: tpu_info's own error text says
        "metrics unavailable. Did you start a MULTI_SLICE workload with
        TPU_RUNTIME_METRICS_PORTS=8431,8432,8433,8434?". No run of ours had ever
        set it, so the port was CONNECTION REFUSED even during a live 60 s JAX
        workload. Setting it is the job of whatever launches the workload
        (scripts/run_workload_tpu.py); this method only consumes the result.

        Metric names come from tpu_info.metrics.MetricName, read off the
        installed package rather than guessed:
            tpu.runtime.hbm.memory.usage.bytes        -> mem_used_mb
            tpu.runtime.hbm.memory.total.bytes        -> mem_total_mb
            tpu.runtime.tensorcore.dutycycle.percent  -> gpu_utilization_pct
            tpu.runtime.hbm.utilization.percent       -> hbm_bw_util_pct
            hlo.execution.timing.distribution.microseconds
            megascale.collective_end_to_end_latencies.microsecond...
        """
        self._grpc = None
        self._grpc_chip_type = None
        addr = os.environ.get("SPAR_TPU_METRICS_ADDR", "localhost:8431")
        try:
            from tpu_info import device as _dev, metrics as _met
        except Exception as e:
            log.info("tpu_info unavailable for gRPC metrics (%s)", e)
            return
        try:
            chip_type, n = _dev.get_local_chips()
        except Exception as e:
            log.info("get_local_chips failed (%s)", e)
            return
        if chip_type is None:
            log.info("no local TPU chips reported")
            return
        # Liveness: one real call. If the server is not up this raises, and we
        # want that recorded loudly rather than silently degrading to zeros.
        try:
            usage = _met.get_chip_usage(chip_type, addr=addr)
            self._grpc = _met
            self._grpc_chip_type = chip_type
            self._grpc_addr = addr
            if n:
                self.n_chips = int(n)
            self.availability["grpc_metrics"] = True
            log.info("libtpu gRPC metrics LIVE at %s: chip=%s n=%s usage_rows=%d",
                     addr, chip_type, n, len(usage or []))
        except Exception as e:
            log.warning("libtpu gRPC metrics at %s NOT available: %s: %s -- "
                        "is TPU_RUNTIME_METRICS_PORTS set on the workload?",
                        addr, type(e).__name__, e)

    # -- source 2: tpu_info python modules ----------------------------------
    def _init_tpu_info(self):
        try:
            from tpu_info import device, metrics  # noqa: F401
            self._tpu_info = (device, metrics)
            try:
                chip_type, count = device.get_local_chips()
                self._chip_type = chip_type
                if count:
                    self.n_chips = int(count)
            except Exception:
                # older tpu_info returns just chip_type
                try:
                    self._chip_type = device.get_local_chips()[0]
                except Exception:
                    self._chip_type = None
            self.availability["tpu_info"] = True
        except Exception as e:
            log.info("tpu_info modules unavailable (%s)", e)

    def _probe_chip_count(self):
        # tpu-info CLI as a last resort for chip count.
        try:
            out = subprocess.run(["tpu-info"], capture_output=True, text=True,
                                 timeout=10)
            self.availability["tpu_info_cli"] = out.returncode == 0
            # count "chip" rows heuristically
            n = sum(1 for ln in out.stdout.splitlines()
                    if "TPU" in ln and "%" in ln)
            return n or None
        except Exception:
            return None

    # -- per-sample readout --------------------------------------------------
    def sample_all(self) -> dict:
        """Return {chip_index: metric_dict} for every local chip this instant.

        Each metric_dict may contain any of: duty, hbm_used, hbm_total, hbm_bw,
        tc_util, hlo_exec(list), hlo_queue(scalar), dcn_lat(list),
        buf_xfer(list). Missing keys mean the metric was dark.
        """
        chips = {c: {} for c in range(self.n_chips)}

        # libtpu first (richest -- has the comm channels)
        if self._tpumon is not None:
            for name in self._libtpu_metrics:
                key = _classify_metric(name)
                if key is None:
                    continue
                try:
                    data = self._tpumon.get_metric(name).data()
                except Exception:
                    continue
                for c in range(self.n_chips):
                    chips[c][key] = _index_per_chip(data, self.n_chips, c)

        # gRPC metrics server: the TPU-native comm channels. The pre-existing
        # tpu_info block below already fetched duty/hbm via get_chip_usage, but
        # nothing ever fetched HBM bandwidth, HLO timing or DCN collective
        # latency -- and those are the channels Phase 0 identified as the only
        # plausible substitute for the power signal TPU does not expose. All of
        # these read localhost:8431, which is why TPU_RUNTIME_METRICS_PORTS must
        # be set on the workload (run_workload_tpu.py does that now).
        if getattr(self, "_grpc", None) is not None:
            g, ct, addr = self._grpc, self._grpc_chip_type, self._grpc_addr
            def _try(fn, *a, **kw):
                try:
                    return fn(*a, **kw)
                except Exception as e:
                    if not getattr(self, "_grpc_warned", False):
                        log.warning("gRPC metric read failed (%s: %s); further "
                                    "failures silent", type(e).__name__, e)
                        self._grpc_warned = True
                    return None
            hbm = _try(g.get_runtime_hbm_utilization, addr=addr)
            if hbm:
                for dev_id, val in hbm:
                    c = int(dev_id) % max(self.n_chips, 1)
                    chips[c]["hbm_bw"] = val
            timing = _try(g.get_hlo_exec_timing, ct, addr=addr)
            if timing:
                for t in timing:
                    c = int(getattr(t, "device_id", 0)) % max(self.n_chips, 1)
                    chips[c]["hlo_exec"] = [getattr(t, "p50", None),
                                            getattr(t, "p90", None),
                                            getattr(t, "p999", None)]
            q = _try(g.get_hlo_queue_size, ct, addr=addr)
            if q:
                for x in q:
                    c = int(getattr(x, "device_id", 0)) % max(self.n_chips, 1)
                    chips[c]["hlo_queue"] = getattr(x, "queue_size", None)
            # DCN collective latency -- the cross-host channel with no NVIDIA
            # counterpart, and the reason the TPU rung was worth running.
            for metric_arg, key in (
                    ("collective_e2e_latency", "dcn_lat"),
                    ("buffer_transfer_latency", "buf_xfer")):
                dist = _try(g.get_transfer_latency, metric_arg, addr=addr)
                if dist:
                    vals = []
                    for d in dist:
                        for attr in ("p50", "p90", "p99", "p999", "mean"):
                            v = getattr(d, attr, None)
                            if v is not None:
                                vals.append(v)
                    if vals:
                        for c in range(self.n_chips):
                            chips[c].setdefault(key, vals)
            tc = _try(g.get_tensorcore_idle_duration, addr=addr)
            if tc:
                for dev_id, idle in tc:
                    c = int(dev_id) % max(self.n_chips, 1)
                    # idle duration -> utilisation proxy, recorded as tc_util
                    chips[c].setdefault("tc_util", idle)

        # tpu_info fills duty / hbm when libtpu didn't
        if self._tpu_info is not None and self._chip_type is not None:
            try:
                _device, metrics = self._tpu_info
                usage = metrics.get_chip_usage(self._chip_type)
                for c in range(self.n_chips):
                    if c >= len(usage):
                        break
                    u = usage[c]
                    chips[c].setdefault("duty", getattr(u, "duty_cycle_pct", None))
                    mu = getattr(u, "memory_usage", None)
                    tm = getattr(u, "total_memory", None)
                    if mu is not None:
                        chips[c].setdefault("hbm_used", mu)
                    if tm is not None:
                        chips[c].setdefault("hbm_total", tm)
            except Exception as e:
                log.debug("tpu_info sample failed: %s", e)

        return chips

    def host_sample(self) -> dict:
        out = {"host_cpu_util_pct": -1.0, "host_mem_used_mb": -1.0,
               "host_mem_total_mb": -1.0}
        try:
            import psutil
            out["host_cpu_util_pct"] = float(psutil.cpu_percent())
            vm = psutil.virtual_memory()
            out["host_mem_used_mb"] = float(vm.used) / (1024 * 1024)
            out["host_mem_total_mb"] = float(vm.total) / (1024 * 1024)
        except Exception:
            pass
        return out


# ── Collector (one per host; manages all local chips; per-chip parquet) ─────

def chip_filename(label, worker_id, chip_index, run_id, ext="parquet"):
    """Per-stream filename token _w{W}c{C} (plan section 2.3). The base label
    (no token) is what lands in the workload_label column; the token only
    disambiguates the 16 streams on disk / in GCS."""
    return f"{label}_w{worker_id}c{chip_index}_{GPU_SHORT}_{str(run_id)[:12]}.{ext}"


class TpuTelemetryCollector:
    """Polls all local TPU chips at a fixed interval in a background thread and
    writes one parquet per chip. Mirrors collect_telemetry.TelemetryCollector."""

    def __init__(
        self,
        worker_id: int = 0,
        interval_sec: float = 1.0,
        workload_label: str = "unknown",
        run_id: str | None = None,
        output_dir: str = "data",
        chip_indices=None,
        checkpoint_sec: float = 60.0,
    ):
        self.worker_id = worker_id
        self._interval = interval_sec
        self._workload_label = workload_label
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._output_dir = output_dir
        self._checkpoint_sec = checkpoint_sec

        self.hostname = socket.gethostname()
        self.tpu_name = os.environ.get("TPU_NAME", "local-tpu")
        self.tpu_type = os.environ.get(
            "TPU_ACCELERATOR_TYPE", os.environ.get("TPU_TYPE", "v6e"))

        # Probe the host's metric sources once.
        self.source = TpuMetricSource()
        if chip_indices is not None:
            self.chip_indices = list(chip_indices)
        else:
            self.chip_indices = list(range(max(1, self.source.n_chips)))
        self.driver_version = self.source.libtpu_version

        self._records = {c: [] for c in self.chip_indices}
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._last_checkpoint = 0.0

    # -- lifecycle -----------------------------------------------------------
    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("TPU telemetry started (worker=%s, chips=%s, sources=%s, run=%s)",
                 self.worker_id, self.chip_indices,
                 {k: v for k, v in self.source.availability.items() if v},
                 self._run_id)

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        log.info("TPU telemetry stopped. %d samples/chip collected.",
                 len(self._records[self.chip_indices[0]]) if self.chip_indices else 0)

    def _poll_loop(self):
        self._last_checkpoint = time.monotonic()
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                chip_samples = self.source.sample_all()
                host = self.source.host_sample()
            except Exception as e:
                log.warning("TPU sample error: %s", e)
                chip_samples, host = {}, {"host_cpu_util_pct": -1.0,
                                         "host_mem_used_mb": -1.0,
                                         "host_mem_total_mb": -1.0}
            now = datetime.now(timezone.utc)
            for c in self.chip_indices:
                row = self._build_row(now, chip_samples.get(c, {}), host)
                self._records[c].append(row)

            if (self._checkpoint_sec > 0
                    and time.monotonic() - self._last_checkpoint >= self._checkpoint_sec):
                self._write_checkpoints()
                self._last_checkpoint = time.monotonic()

            sleep_time = max(0.0, self._interval - (time.monotonic() - t0))
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _build_row(self, now, m: dict, host: dict) -> dict:
        """Assemble one fully-populated row (stable schema) for one chip."""
        # device level metrics (may be dark)
        duty = _scalar(m.get("duty"), default=None) if "duty" in m else None
        hbm_bw = _scalar(m.get("hbm_bw"), default=None) if "hbm_bw" in m else None
        hbm_used_b = _scalar(m.get("hbm_used"), default=None) if "hbm_used" in m else None
        hbm_total_b = _scalar(m.get("hbm_total"), default=None) if "hbm_total" in m else None
        tc = _scalar(m.get("tc_util"), default=None) if "tc_util" in m else None

        live = 1 if (duty is not None or hbm_used_b is not None or tc is not None) else 0

        # GPU-schema mapped columns: 0.0 when dark (always a valid level)
        row = {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            "gpu_utilization_pct": float(duty) if duty is not None else 0.0,
            "mem_utilization_pct": float(hbm_bw) if hbm_bw is not None else 0.0,
            "mem_used_mb": (float(hbm_used_b) / (1024 * 1024)) if hbm_used_b is not None else 0.0,
            "mem_total_mb": (float(hbm_total_b) / (1024 * 1024)) if hbm_total_b is not None else 0.0,
        }
        for col in _NEG1_CONST:
            row[col] = -1

        # TPU-native comm columns: -1 when dark (filterable level channel)
        p50, p90, p99 = _percentiles(m.get("hlo_exec"))
        row["hlo_exec_us_p50"], row["hlo_exec_us_p90"], row["hlo_exec_us_p99"] = p50, p90, p99
        row["hlo_queue_size"] = _scalar(m.get("hlo_queue"), default=-1.0)
        d50, d90, d99 = _percentiles(m.get("dcn_lat"))
        (row["dcn_collective_lat_us_p50"], row["dcn_collective_lat_us_p90"],
         row["dcn_collective_lat_us_p99"]) = d50, d90, d99
        _, b90, _ = _percentiles(m.get("buf_xfer"))
        row["buffer_xfer_lat_us_p90"] = b90
        # native duty/tc/bw: -1 when dark (distinct sentinel from the mapped cols)
        row["tpu_duty_cycle_pct"] = float(duty) if duty is not None else -1.0
        row["tensorcore_util_pct"] = float(tc) if tc is not None else -1.0
        row["hbm_bw_util_pct"] = float(hbm_bw) if hbm_bw is not None else -1.0
        row["tpu_metrics_live"] = live  # int 0/1 (numeric -> picked as a feature)

        # host psutil
        row.update(host)

        # per-sample workload identity (static cols added in to_dataframe)
        row["workload_label"] = self._workload_label
        row["run_id"] = self._run_id
        return row

    # -- output --------------------------------------------------------------
    def to_dataframe(self, chip_index) -> pd.DataFrame:
        df = pd.DataFrame(self._records[chip_index])
        # static identity columns (strings -> excluded from features)
        df["gpu_name"] = GPU_NAME
        df["gpu_uuid"] = f"tpu-{self.tpu_name}-w{self.worker_id}-c{chip_index}"
        df["driver_version"] = self.driver_version
        df["hostname"] = self.hostname
        df["worker_id"] = str(self.worker_id)
        df["chip_index"] = str(chip_index)
        df["tpu_name"] = self.tpu_name
        df["tpu_type"] = self.tpu_type
        return df

    def _metadata(self, chip_index) -> dict:
        return {
            "worker_id": self.worker_id,
            "chip_index": chip_index,
            "tpu_name": self.tpu_name,
            "tpu_type": self.tpu_type,
            "gpu_name": GPU_NAME,
            "libtpu_version": self.driver_version,
            "hostname": self.hostname,
            "sources": self.source.availability,
            "n_chips": self.source.n_chips,
        }

    def _write_parquet(self, df, path, chip_index):
        import pyarrow as pa
        import pyarrow.parquet as pq
        table = pa.Table.from_pandas(df)
        meta = table.schema.metadata or {}
        meta[b"tpu_metadata"] = json.dumps(self._metadata(chip_index)).encode("utf-8")
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, str(path))

    def _final_path(self, chip_index) -> Path:
        return Path(self._output_dir) / chip_filename(
            self._workload_label, self.worker_id, chip_index, self._run_id)

    def _write_checkpoints(self):
        for c in self.chip_indices:
            if not self._records[c]:
                continue
            try:
                final = self._final_path(c)
                final.parent.mkdir(parents=True, exist_ok=True)
                tmp = final.with_suffix(final.suffix + ".partial.tmp")
                part = final.with_suffix(final.suffix + ".partial")
                self._write_parquet(self.to_dataframe(c), tmp, c)
                os.replace(tmp, part)  # atomic
            except Exception as e:
                log.warning("TPU checkpoint (chip %s) failed: %s", c, e)

    def save(self) -> list[str]:
        """Write one final parquet per chip; drop the .partial snapshots."""
        paths = []
        for c in self.chip_indices:
            final = self._final_path(c)
            final.parent.mkdir(parents=True, exist_ok=True)
            self._write_parquet(self.to_dataframe(c), final, c)
            part = final.with_suffix(final.suffix + ".partial")
            try:
                part.unlink(missing_ok=True)
            except OSError:
                pass
            paths.append(str(final))
            log.info("Saved chip %s: %d rows -> %s", c, len(self._records[c]), final)
        return paths


# ── CLI (standalone probe / smoke) ───────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(description="Collect TPU telemetry (per-chip parquet).")
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--label", type=str, default="tpu_probe")
    ap.add_argument("--run-id", type=str, default=None)
    ap.add_argument("--worker-id", type=int, default=0)
    ap.add_argument("--checkpoint-sec", type=float, default=60.0)
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    collector = TpuTelemetryCollector(
        worker_id=args.worker_id, interval_sec=args.interval,
        workload_label=args.label, run_id=args.run_id,
        output_dir=args.output_dir, checkpoint_sec=args.checkpoint_sec)

    print(f"Collecting TPU telemetry for {args.duration}s "
          f"(chips={collector.chip_indices}, "
          f"sources={ {k for k,v in collector.source.availability.items() if v} })...")
    collector.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted; saving what we have.")
    collector.stop()
    for p in collector.save():
        print(f"  saved {p}")


if __name__ == "__main__":
    main()
