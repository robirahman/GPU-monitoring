#!/usr/bin/env python3
"""SPAR GPU Telemetry Collector (AMD/ROCm) — Tier 1 (amdsmi) + Tier 2 (RDC).

AMD analogue of scripts/collect_telemetry.py. Polls GPU metrics via the
`amdsmi` Python bindings (shipped with ROCm at /opt/rocm/share/amd_smi)
and optionally via RDC (the ROCm Data Center tool, AMD's DCGM analogue).

Standalone usage:
    python collect_amdsmi_telemetry.py --duration 60 --output data/test.parquet

As a library (used by run_workload_amd.py):
    collector = AmdTelemetryCollector(workload_label="resnet_cifar10")
    collector.start()
    # ... run workload ...
    collector.stop()
    collector.save("data/output.parquet")

Schema notes (vs. the NVIDIA collector):
  - `gpu_utilization_pct`            -> amdsmi gfx_activity (graphics engine busy %)
  - `mem_utilization_pct`            -> amdsmi umc_activity (memory controller busy %)
  - `mem_used_mb` / `mem_total_mb`   -> VRAM, identical semantics
  - `power_draw_w`                   -> current_socket_power (W)
  - `temperature_c`                  -> EDGE temp (closest analogue to NVML's GPU temp)
  - `temperature_hotspot_c`          -> HOTSPOT (junction) temp; AMD-only
  - `temperature_vram_c`             -> VRAM/HBM temp; AMD-only
  - `sm_clock_mhz`                   -> GFX clock (kept under the NVIDIA name for cross-vendor joins)
  - `mem_clock_mhz`                  -> MEM clock
  - `pcie_tx_mbps` / `pcie_rx_mbps`  -> derived from pcie_info if available; -1 otherwise
  - `encoder_util_pct`               -> mm_activity (combined VCN encode+decode); decoder_util_pct
                                        is set to the same value with a flag column
  - `decoder_util_pct`               -> same as encoder_util_pct on AMD; see `mm_combined` flag
  - `fan_speed_pct`                  -> fan RPM converted to % of max; -1 if no fan sensor

Tier 2 (RDC) is best-effort. RDNA 4 consumer cards expose a smaller field set
than CDNA datacenter cards; profiling counters analogous to DCGM's
tensor_active / fp32_pipe_active are typically unavailable. The collector
falls back to RDC's basic activity fields and logs which were obtained.
"""

import argparse
import json
import logging
import os
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# amdsmi ships with ROCm, not pip. Add the standard install path before import.
_AMDSMI_PYTHON = "/opt/rocm/share/amd_smi"
if os.path.isdir(_AMDSMI_PYTHON) and _AMDSMI_PYTHON not in sys.path:
    sys.path.insert(0, _AMDSMI_PYTHON)

try:
    import amdsmi
except ImportError:
    print(
        "ERROR: amdsmi not found. It ships with ROCm; ensure ROCm 6.4+ is "
        "installed and /opt/rocm/share/amd_smi is on PYTHONPATH "
        "(or run: pip install amdsmi if your distribution provides it).",
        file=sys.stderr,
    )
    sys.exit(1)

log = logging.getLogger("spar.telemetry_amd")


# ---------------------------------------------------------------------------
# Helpers — amdsmi has version-to-version churn; wrap calls defensively
# ---------------------------------------------------------------------------

def _safe(call, default=None):
    """Run an amdsmi call; on AmdSmiException or AttributeError return default."""
    try:
        return call()
    except Exception as e:  # amdsmi.AmdSmiException, attr errors, etc.
        log.debug("amdsmi call failed: %s", e)
        return default


def _resolve_clk_type(name: str):
    """Resolve a clock-type enum across amdsmi versions.

    Newer amdsmi exposes ``AmdSmiClkType.GFX`` etc.; older builds use
    module-level ints like ``AMDSMI_CLK_TYPE_GFX``. Try both.
    """
    enum = getattr(amdsmi, "AmdSmiClkType", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    return getattr(amdsmi, f"AMDSMI_CLK_TYPE_{name}", None)


def _resolve_temp_type(name: str):
    enum = getattr(amdsmi, "AmdSmiTemperatureType", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    return getattr(amdsmi, f"AMDSMI_TEMPERATURE_TYPE_{name}", None)


def _resolve_temp_metric(name: str = "CURRENT"):
    enum = getattr(amdsmi, "AmdSmiTemperatureMetric", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    return getattr(amdsmi, f"AMDSMI_TEMP_{name}", None)


def _resolve_mem_type(name: str = "VRAM"):
    enum = getattr(amdsmi, "AmdSmiMemoryType", None)
    if enum is not None and hasattr(enum, name):
        return getattr(enum, name)
    return getattr(amdsmi, f"AMDSMI_MEM_TYPE_{name}", None)


# ---------------------------------------------------------------------------
# GpuMetadata — static GPU info captured once at startup
# ---------------------------------------------------------------------------

class GpuMetadata:
    """Captures static GPU properties via amdsmi."""

    def __init__(self, gpu_index: int = 0):
        amdsmi.amdsmi_init()
        handles = amdsmi.amdsmi_get_processor_handles()
        if gpu_index >= len(handles):
            raise IndexError(
                f"gpu_index={gpu_index} out of range; "
                f"amdsmi reports {len(handles)} processor(s)."
            )
        self._handle = handles[gpu_index]
        self.gpu_index = gpu_index

        asic = _safe(lambda: amdsmi.amdsmi_get_gpu_asic_info(self._handle), {}) or {}
        board = _safe(lambda: amdsmi.amdsmi_get_gpu_board_info(self._handle), {}) or {}
        driver = _safe(lambda: amdsmi.amdsmi_get_gpu_driver_info(self._handle), {}) or {}

        # Prefer the human-readable market name; fall back to product_name.
        self.gpu_name = (
            asic.get("market_name")
            or board.get("product_name")
            or asic.get("vendor_name")
            or "unknown_amd_gpu"
        )

        self.gpu_uuid = (
            _safe(lambda: amdsmi.amdsmi_get_gpu_device_uuid(self._handle))
            or asic.get("asic_serial")
            or board.get("product_serial")
            or ""
        )

        self.driver_version = driver.get("driver_version") or driver.get("driver_name", "unknown")

        # ROCm runtime version analogue of nvmlSystemGetCudaDriverVersion.
        # amdsmi doesn't expose a single ROCm version function reliably, so
        # parse /opt/rocm/.info/version if present.
        self.rocm_version = "unknown"
        for p in ("/opt/rocm/.info/version", "/opt/rocm/.info/version-utils"):
            if os.path.isfile(p):
                try:
                    with open(p) as f:
                        self.rocm_version = f.read().strip()
                    break
                except OSError:
                    pass

        mem_total = _safe(
            lambda: amdsmi.amdsmi_get_gpu_memory_total(
                self._handle, _resolve_mem_type("VRAM")
            ),
            0,
        ) or 0
        self.total_memory_mb = int(mem_total) // (1024 * 1024)

        pcie = _safe(lambda: amdsmi.amdsmi_get_pcie_info(self._handle), {}) or {}
        # Schema varies: newer amdsmi returns {"pcie_static": {...}, "pcie_metric": {...}}.
        static = pcie.get("pcie_static", {}) if isinstance(pcie, dict) else {}
        metric = pcie.get("pcie_metric", {}) if isinstance(pcie, dict) else {}
        self.pcie_gen = static.get("max_pcie_speed", -1) or metric.get("pcie_speed", -1)
        self.pcie_width = static.get("max_pcie_width", -1) or metric.get("pcie_width", -1)

        self.asic_target = asic.get("target_graphics_version") or asic.get("asic_family", "")
        self.vbios_version = _safe(
            lambda: amdsmi.amdsmi_get_gpu_vbios_info(self._handle).get("version"),
            "unknown",
        ) or "unknown"

    @property
    def handle(self):
        return self._handle

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "driver_version": self.driver_version,
            "rocm_version": self.rocm_version,
            "total_memory_mb": self.total_memory_mb,
            "pcie_gen": self.pcie_gen,
            "pcie_width": self.pcie_width,
            "asic_target": self.asic_target,
            "vbios_version": self.vbios_version,
            "vendor": "AMD",
        }

    def cleanup(self):
        try:
            amdsmi.amdsmi_shut_down()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# RdcCollector — optional Tier 2 RDC profiling metrics
# ---------------------------------------------------------------------------

class RdcCollector:
    """Collects Tier 2 RDC metrics. Raises ImportError or RuntimeError if RDC
    is unavailable, letting the caller skip Tier 2.

    On CDNA datacenter cards (MI200/MI300): may expose profiling counters
    (gfx busy, valu/salu pipe issue util, occupancy, memory busy).

    On RDNA consumer cards (RX 9070 XT etc.): typically only basic device
    activity fields are available; profiling counters return empty.
    """

    # RDC field IDs — values match rdc_fields.h. Names mirror what DCGM exposes
    # so downstream feature builders can join cross-vendor.
    BASIC_FIELDS = {
        # field_id: column_name
        "RDC_FI_GPU_BUSY_PERCENT":        "rdc_gpu_busy",
        "RDC_FI_GPU_MEMORY_USAGE":        "rdc_mem_used_bytes",
        "RDC_FI_GPU_TEMP":                "rdc_gpu_temp",
        "RDC_FI_POWER_USAGE":             "rdc_power_usage",
        "RDC_FI_GPU_CLOCK":               "rdc_gpu_clock",
        "RDC_FI_MEM_CLOCK":               "rdc_mem_clock",
    }

    PROF_FIELDS = {
        # On RDNA 4 these will mostly be unavailable; we try anyway.
        "RDC_FI_PROF_GUI_UTIL":           "rdc_gr_engine_active",
        "RDC_FI_PROF_OCCUPANCY_PERCENT":  "rdc_sm_occupancy",
        "RDC_FI_PROF_VALU_PIPE_ISSUE_UTIL": "rdc_fp32_pipe_active",
        "RDC_FI_PROF_SALU_UTIL":          "rdc_salu_active",
        "RDC_FI_PROF_ACTIVE_WAVES":       "rdc_active_waves",
        "RDC_FI_PROF_ELAPSED_CYCLES":     "rdc_elapsed_cycles",
        "RDC_FI_PROF_TENSOR_ACTIVE_PERCENT": "rdc_tensor_active",
    }

    # Basic fields arrive in raw RDC units (see `rdci dmon -l`). We disable
    # RdcReader's partial default unit conversion and scale explicitly here so
    # column names stay honest. Profiling fields (percentages, FLOPS, cycles,
    # waves) are kept raw. mem_used_bytes stays in bytes.
    RDC_UNIT_SCALE = {
        "rdc_power_usage": 1e-6,   # microwatts -> watts
        "rdc_gpu_temp":    1e-3,   # millidegrees C -> C
        "rdc_gpu_clock":   1e-6,   # Hz -> MHz
        "rdc_mem_clock":   1e-6,   # Hz -> MHz
    }

    @staticmethod
    def _discover_rdc_python_path() -> str | None:
        """Locate the RDC Python bindings across ROCm layouts.

        ROCm <=6 shipped them at /opt/rocm/share/rdc/python; ROCm 7 moved them
        to <rocm>/libexec/rdc/python_binding. Return the first dir that holds
        RdcReader.py, or None.
        """
        import glob
        candidates = [
            "/opt/rocm/share/rdc/python",            # ROCm <= 6 (and symlinks)
            "/opt/rocm/libexec/rdc/python_binding",  # ROCm 7 via /opt/rocm
        ]
        # Versioned install roots, newest first.
        for root in sorted(glob.glob("/opt/rocm-*"), reverse=True):
            candidates.append(os.path.join(root, "libexec/rdc/python_binding"))
            candidates.append(os.path.join(root, "share/rdc/python"))
        for c in candidates:
            if os.path.isfile(os.path.join(c, "RdcReader.py")):
                return c
        return None

    def __init__(self, gpu_index: int = 0):
        rdc_python = self._discover_rdc_python_path()
        if rdc_python and rdc_python not in sys.path:
            sys.path.insert(0, rdc_python)

        try:
            from RdcReader import RdcReader  # type: ignore
            import rdc_bootstrap  # noqa: F401  type: ignore
            from rdc_bootstrap import rdc_field_t  # type: ignore
            from rdc_bootstrap import rdc_field_type_t  # type: ignore
        except ImportError as e:
            raise ImportError(
                f"RDC Python bindings not found ({e}). Install with "
                "`sudo apt install rdc` and ensure rdcd is running "
                "(`sudo systemctl enable --now rdc`). If installed, ensure "
                "librdc_bootstrap.so is on LD_LIBRARY_PATH (e.g. "
                "/opt/rocm/lib)."
            )

        self._RdcReader = RdcReader
        self._rdc_field_t = rdc_field_t
        self._rdc_field_type_t = rdc_field_type_t
        self._gpu_index = gpu_index
        # Field-id -> latest value, populated by the reader's handle_field
        # callback (RdcReader.process() returns None and dispatches per-field).
        self._captured: dict[int, float] = {}

        # Resolve field-id constants; some names may not exist in older RDC.
        def resolve(name):
            return getattr(rdc_field_t, name, None)

        self._field_map = {}
        # Always include the basic set
        for key, col in self.BASIC_FIELDS.items():
            fid = resolve(key)
            if fid is not None:
                self._field_map[int(fid)] = col

        # Try profiling fields; record which ones the device actually accepts
        self._using_profiling = False
        prof_ids = {}
        for key, col in self.PROF_FIELDS.items():
            fid = resolve(key)
            if fid is not None:
                prof_ids[int(fid)] = col

        # RdcReader.process() returns None and dispatches each field to
        # handle_field(); worse, the base process() does an in-place
        # `value.value.l_int = int(value.value.dbl)` that clobbers the shared
        # ctypes union, truncating DOUBLE fields (tensor %, occupancy) to ints
        # before handle_field ever sees them. We override process() to read the
        # raw double *before* any conversion and capture into self._captured.
        import rdc_bootstrap as _rdcb
        capture = self._captured
        _rdc = _rdcb.rdc
        _rdc_field_value = _rdcb.rdc_field_value
        _rdc_status_t = _rdcb.rdc_status_t
        double_type = int(self._rdc_field_type_t.DOUBLE)
        # Kept for cleanup(): RdcReader has no stop()/destructor, so the field
        # group and GPU group it creates on rdcd leak unless we destroy them.
        # After enough runs rdcd returns RDC_ST_MAX_LIMIT and refuses new
        # groups, silently disabling Tier-2 for the rest of a long sweep.
        self._rdc = _rdc
        self._rdc_status_t = _rdc_status_t

        class _CapturingRdcReader(self._RdcReader):  # type: ignore[misc, valid-type]
            def process(self_inner):
                has_succeed = False
                for gindex in self_inner.gpu_indexes:
                    for fid in self_inner.field_ids:
                        value = _rdc_field_value()
                        result = _rdc.rdc_field_get_latest_value(
                            self_inner.rdc_handle, gindex, fid, value)
                        if _rdc_status_t(result) != _rdc_status_t.RDC_ST_OK:
                            continue
                        try:
                            key = int(value.field_id.value)
                        except AttributeError:
                            key = int(value.field_id)
                        if int(value.type.value) == double_type:
                            capture[key] = float(value.value.dbl)
                        else:
                            capture[key] = float(value.value.l_int)
                        has_succeed = True
                if (self_inner.gpu_indexes and self_inner.field_ids
                        and not has_succeed):
                    self_inner.try_reconnect()

        def _make_reader(field_ids):
            # unit_converter={} disables RdcReader's partial default scaling;
            # we scale explicitly in sample() via RDC_UNIT_SCALE.
            return _CapturingRdcReader(
                ip_port="localhost:50051",
                field_ids=field_ids,
                unit_converter={},
                update_freq=1_000_000,   # microseconds (1s)
                max_keep_age=30,
                max_keep_samples=30,
                gpu_indexes=[gpu_index],
            )

        # Build the reader. RdcReader handles the host-engine connection and
        # field-watch lifecycle. ip="localhost" assumes rdcd is running.
        all_field_ids = list(self._field_map.keys()) + list(prof_ids.keys())
        try:
            self._reader = _make_reader(all_field_ids)
            # If the reader accepted profiling fields, include them in the map.
            self._field_map.update(prof_ids)
            self._using_profiling = bool(prof_ids)
            log.info(
                "RDC Tier 2 collector initialized (gpu_index=%d, "
                "profiling_fields=%s, total_fields=%d)",
                gpu_index, self._using_profiling, len(self._field_map),
            )
        except Exception as e:
            # Fall back: try with basic fields only.
            log.info(
                "RDC profiling fields unsupported on this GPU (%s); "
                "retrying with basic fields only.", e,
            )
            self._reader = _make_reader(list(self._field_map.keys()))
            log.info(
                "RDC Tier 2 collector initialized with BASIC fields "
                "(gpu_index=%d, fields=%d)", gpu_index, len(self._field_map),
            )

    def sample(self) -> dict:
        """Read latest RDC field values. Returns dict with rdc_* keys.

        process() refreshes self._captured (field_id -> raw value) via the
        handle_field callback; we map those onto column names and apply
        explicit unit scaling.
        """
        result = {col: float("nan") for col in self._field_map.values()}
        # A field that fails this poll must remain missing, not repeat a value
        # retained from an earlier successful poll.
        self._captured.clear()
        try:
            self._reader.process()
            for field_id, col in self._field_map.items():
                if field_id in self._captured:
                    val = self._captured[field_id]
                    result[col] = val * self.RDC_UNIT_SCALE.get(col, 1.0)
        except Exception as e:
            log.warning("RDC sample error: %s", e)
        return result

    def cleanup(self):
        # RdcReader has no stop()/destructor. Release the watch and destroy the
        # field + GPU groups it created, then disconnect — otherwise each run
        # leaks two groups on rdcd until it hits RDC_ST_MAX_LIMIT.
        r = getattr(self, "_reader", None)
        if r is None:
            return
        handle = getattr(r, "rdc_handle", None)
        ok = self._rdc_status_t.RDC_ST_OK
        for fn, args in (
            (self._rdc.rdc_field_unwatch,
             (handle, getattr(r, "gpu_group_id", None), getattr(r, "field_group_id", None))),
            (self._rdc.rdc_group_field_destroy,
             (handle, getattr(r, "field_group_id", None))),
            (self._rdc.rdc_group_gpu_destroy,
             (handle, getattr(r, "gpu_group_id", None))),
            (self._rdc.rdc_disconnect, (handle,)),
        ):
            try:
                if any(a is None for a in args):
                    continue
                rc = fn(*args)
                if self._rdc_status_t(rc) != ok:
                    log.debug("RDC cleanup: %s returned %s", fn.__name__, rc)
            except Exception as e:
                log.debug("RDC cleanup: %s raised %s", getattr(fn, "__name__", fn), e)


# ---------------------------------------------------------------------------
# AmdTelemetryCollector — main polling engine
# ---------------------------------------------------------------------------

class AmdTelemetryCollector:
    """Polls GPU metrics at a fixed interval via amdsmi (Tier 1) and
    optionally RDC (Tier 2). Runs in a background daemon thread."""

    def __init__(
        self,
        gpu_index: int = 0,
        interval_sec: float = 1.0,
        workload_label: str = "unknown",
        run_id: str | None = None,
        enable_rdc: bool = True,
    ):
        self.metadata = GpuMetadata(gpu_index)
        self._handle = self.metadata.handle
        self._interval = interval_sec
        self._workload_label = workload_label
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        # Pre-resolve enum constants once (cheaper than doing it per-sample).
        self._clk_gfx = _resolve_clk_type("GFX")
        self._clk_mem = _resolve_clk_type("MEM")
        self._temp_edge = _resolve_temp_type("EDGE")
        self._temp_hotspot = _resolve_temp_type("HOTSPOT")
        self._temp_vram = _resolve_temp_type("VRAM")
        self._temp_metric_current = _resolve_temp_metric("CURRENT")
        self._mem_vram = _resolve_mem_type("VRAM")

        # Tier 2 (RDC) — optional
        self._rdc: RdcCollector | None = None
        if enable_rdc:
            try:
                self._rdc = RdcCollector(gpu_index)
            except ImportError as e:
                log.info(
                    "RDC bindings not available; Tier 2 metrics will be omitted. (%s)", e
                )
            except Exception as e:
                log.info("RDC initialization failed (%s); Tier 2 metrics will be omitted.", e)

    def start(self):
        """Start the background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info(
            "Telemetry started (gpu=%s, interval=%.1fs, rdc=%s, label=%s, run=%s)",
            self.metadata.gpu_name,
            self._interval,
            "enabled" if self._rdc else "disabled",
            self._workload_label,
            self._run_id,
        )

    def stop(self):
        """Stop the polling thread and wait for it to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        log.info("Telemetry stopped. %d samples collected.", len(self._records))

    def _poll_loop(self):
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                row = self._sample_tier1()
            except Exception as e:
                log.warning("amdsmi sample error: %s", e)
                row = self._nan_row()

            if self._rdc is not None:
                row.update(self._rdc.sample())

            row["workload_label"] = self._workload_label
            row["run_id"] = self._run_id
            self._records.append(row)

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _sample_tier1(self) -> dict:
        """Sample all Tier 1 amdsmi metrics. Returns a dict matching the
        NVIDIA collector's schema where the semantic equivalent exists."""
        now = datetime.now(timezone.utc)
        h = self._handle

        # --- Activity (gfx / umc / mm) ---
        activity = _safe(lambda: amdsmi.amdsmi_get_gpu_activity(h), {}) or {}
        gfx_pct = activity.get("gfx_activity", -1)
        umc_pct = activity.get("umc_activity", -1)
        mm_pct = activity.get("mm_activity", -1)

        # --- Memory ---
        mem_used = _safe(lambda: amdsmi.amdsmi_get_gpu_memory_usage(h, self._mem_vram), 0) or 0
        mem_total_b = self.metadata.total_memory_mb * 1024 * 1024

        # --- Power ---
        power_info = _safe(lambda: amdsmi.amdsmi_get_power_info(h), {}) or {}
        power_w = (
            power_info.get("current_socket_power")
            if power_info.get("current_socket_power") not in (None, "N/A")
            else power_info.get("average_socket_power", -1)
        )
        try:
            power_w = float(power_w)
        except (TypeError, ValueError):
            power_w = -1.0
        power_limit_w = power_info.get("power_limit", -1)
        try:
            # Some amdsmi versions report power_limit in microwatts.
            if power_limit_w and power_limit_w > 1000:
                power_limit_w = float(power_limit_w) / 1_000_000.0
        except (TypeError, ValueError):
            power_limit_w = -1.0

        # --- Temperature (multiple sensors) ---
        def _temp(sensor):
            if sensor is None or self._temp_metric_current is None:
                return -1
            return _safe(
                lambda: amdsmi.amdsmi_get_temp_metric(h, sensor, self._temp_metric_current),
                -1,
            )

        temp_edge = _temp(self._temp_edge)
        temp_hotspot = _temp(self._temp_hotspot)
        temp_vram = _temp(self._temp_vram)

        # --- Clocks ---
        def _clock(clk_type):
            if clk_type is None:
                return -1
            info = _safe(lambda: amdsmi.amdsmi_get_clock_info(h, clk_type), {}) or {}
            # amdsmi returns clk in MHz already
            return info.get("clk", info.get("cur_clk", -1))

        gfx_clock = _clock(self._clk_gfx)
        mem_clock = _clock(self._clk_mem)

        # --- PCIe throughput (newer amdsmi) ---
        pcie_info = _safe(lambda: amdsmi.amdsmi_get_pcie_info(h), {}) or {}
        pcie_metric = pcie_info.get("pcie_metric", {}) if isinstance(pcie_info, dict) else {}
        pcie_tx = pcie_metric.get("pcie_tx", -1)
        pcie_rx = pcie_metric.get("pcie_rx", -1)
        # Newer amdsmi reports bandwidth in MB/s already; older reports bytes/s.
        try:
            pcie_tx = float(pcie_tx) if pcie_tx is not None else -1.0
            pcie_rx = float(pcie_rx) if pcie_rx is not None else -1.0
            if pcie_tx > 1e6:  # heuristic: bytes/s -> MB/s
                pcie_tx /= (1024 * 1024)
                pcie_rx /= (1024 * 1024)
        except (TypeError, ValueError):
            pcie_tx, pcie_rx = -1.0, -1.0

        # --- Fan (RPM -> %) ---
        fan_rpm = _safe(lambda: amdsmi.amdsmi_get_gpu_fan_speed(h, 0), -1)
        fan_max = _safe(lambda: amdsmi.amdsmi_get_gpu_fan_speed_max(h, 0), 0) or 0
        if fan_rpm and fan_max and fan_max > 0:
            fan_pct = round(100.0 * float(fan_rpm) / float(fan_max), 1)
        else:
            fan_pct = -1

        # --- Throttle status (AMD-specific, useful signal) ---
        gpu_metrics = _safe(lambda: amdsmi.amdsmi_get_gpu_metrics_info(h), {}) or {}
        throttle_status = gpu_metrics.get("throttle_status", -1) if isinstance(gpu_metrics, dict) else -1

        return {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            # Cross-vendor names (matching the NVIDIA schema):
            "gpu_utilization_pct": gfx_pct,
            "mem_utilization_pct": umc_pct,
            "mem_used_mb": int(mem_used) // (1024 * 1024) if mem_used else -1,
            "mem_total_mb": self.metadata.total_memory_mb,
            "power_draw_w": power_w,
            "temperature_c": temp_edge,
            "sm_clock_mhz": gfx_clock,           # AMD: GFX clock
            "mem_clock_mhz": mem_clock,
            "pcie_tx_mbps": pcie_tx,
            "pcie_rx_mbps": pcie_rx,
            "encoder_util_pct": mm_pct,          # AMD: combined VCN encode+decode
            "decoder_util_pct": mm_pct,          # AMD: same as encoder; flag below
            "fan_speed_pct": fan_pct,
            # AMD-only extensions:
            "temperature_hotspot_c": temp_hotspot,
            "temperature_vram_c": temp_vram,
            "power_limit_w": power_limit_w,
            "throttle_status": throttle_status,
            "mm_combined": True,                 # encoder/decoder columns are the same
        }

    def _nan_row(self) -> dict:
        now = datetime.now(timezone.utc)
        nan = float("nan")
        return {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            "gpu_utilization_pct": nan,
            "mem_utilization_pct": nan,
            "mem_used_mb": nan,
            "mem_total_mb": self.metadata.total_memory_mb,
            "power_draw_w": nan,
            "temperature_c": nan,
            "sm_clock_mhz": nan,
            "mem_clock_mhz": nan,
            "pcie_tx_mbps": nan,
            "pcie_rx_mbps": nan,
            "encoder_util_pct": nan,
            "decoder_util_pct": nan,
            "fan_speed_pct": nan,
            "temperature_hotspot_c": nan,
            "temperature_vram_c": nan,
            "power_limit_w": nan,
            "throttle_status": nan,
            "mm_combined": True,
        }

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._records)
        df["gpu_name"] = self.metadata.gpu_name
        df["gpu_uuid"] = self.metadata.gpu_uuid
        df["driver_version"] = self.metadata.driver_version
        df["vendor"] = "AMD"
        return df

    def save(self, output_path: str, fmt: str = "parquet"):
        df = self.to_dataframe()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df)
            meta = table.schema.metadata or {}
            meta[b"gpu_metadata"] = json.dumps(self.metadata.to_dict()).encode("utf-8")
            table = table.replace_schema_metadata(meta)
            pq.write_table(table, str(path))
        else:
            df.to_csv(str(path), index=False)

        log.info("Saved %d rows to %s", len(df), path)
        return str(path)

    @property
    def sample_count(self) -> int:
        return len(self._records)

    def cleanup(self):
        if self._rdc is not None:
            self._rdc.cleanup()
        self.metadata.cleanup()


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect AMD GPU telemetry (Tier 1 amdsmi + Tier 2 RDC) for a fixed duration."
    )
    parser.add_argument("--duration", type=int, default=60,
                        help="Collection duration in seconds (default: 60)")
    parser.add_argument("--output", type=str, default="data/telemetry_amd_test.parquet",
                        help="Output file path")
    parser.add_argument("--format", type=str, choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--interval", type=float, default=1.0,
                        help="Polling interval in seconds (default: 1.0)")
    parser.add_argument("--gpu-index", type=int, default=0)
    parser.add_argument("--label", type=str, default="standalone_test_amd")
    parser.add_argument("--no-rdc", action="store_true",
                        help="Disable RDC Tier 2 collection")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    collector = AmdTelemetryCollector(
        gpu_index=args.gpu_index,
        interval_sec=args.interval,
        workload_label=args.label,
        enable_rdc=not args.no_rdc,
    )

    print(f"Collecting AMD telemetry for {args.duration}s (interval={args.interval}s)...")
    collector.start()

    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted by user.")

    collector.stop()
    out = collector.save(args.output, fmt=args.format)
    print(f"Saved {collector.sample_count} samples to {out}")
    collector.cleanup()


if __name__ == "__main__":
    main()
