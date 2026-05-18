"""SPAR GPU Telemetry Collector (AMD / Windows) — amd-smi.exe polling.

Windows analogue of scripts/collect_amdsmi_simple.py. The Linux collector
imports the in-process `amdsmi` Python bindings shipped with ROCm at
/opt/rocm/share/amd_smi. Those bindings do not exist on Windows; instead
the AMD HIP SDK / AMD Software PRO Edition installs `amd-smi.exe` and we
poll it via subprocess, parsing its JSON output.

Schema mirrors the Linux collector so feature extraction is vendor-
agnostic. Where amd-smi.exe omits a field on this RDNA 4 consumer card,
the column is filled with NaN.

Usage (standalone):
    python collect_amdsmi_windows.py --duration 60 --output data/test.parquet

As a library (called from run_workload_*.py):
    c = AmdSmiWindowsCollector(workload_label="resnet_cifar10")
    c.start(); ...; c.stop(); c.save("data/out.parquet")

Requires:
    - amd-smi.exe on PATH (HIP SDK or AMD Software PRO Edition installer)
    - python -m pip install pandas pyarrow
"""

import argparse
import json
import logging
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("spar.telemetry_amd_windows")

# amd-smi.exe is dropped here by recent HIP SDK / AMD Software PRO Edition installers.
_FALLBACK_AMDSMI_PATHS = [
    r"C:\Program Files\AMD\ROCm\6.4\bin\amd-smi.exe",
    r"C:\Program Files\AMD\ROCm\6.3\bin\amd-smi.exe",
    r"C:\Program Files\AMD\ROCm\6.2\bin\amd-smi.exe",
    r"C:\Program Files\AMD\ROCm\6.1\bin\amd-smi.exe",
    r"C:\Program Files\AMD\AMD-SMI\amd-smi.exe",
]


def _resolve_amdsmi() -> str | None:
    found = shutil.which("amd-smi") or shutil.which("amd-smi.exe")
    if found:
        return found
    for p in _FALLBACK_AMDSMI_PATHS:
        if Path(p).is_file():
            return p
    return None


_NUM_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _num(val, default=float("nan")):
    """Parse '75 W', '2400 MHz', '20 %', 75, '75.5', None into a float.

    amd-smi includes units in stringified values; some versions return raw
    numbers. 'N/A' and missing keys map to NaN.
    """
    if val is None:
        return default
    if isinstance(val, bool):
        return float(val)
    if isinstance(val, (int, float)):
        return float(val)
    if isinstance(val, str):
        s = val.strip()
        if s.upper() in ("N/A", "NA", "", "NONE"):
            return default
        m = _NUM_RE.search(s)
        if m:
            try:
                return float(m.group(0))
            except ValueError:
                return default
    return default


def _first(d: dict, *keys, default=None):
    """Return the first present key's value from a dict (case-insensitive)."""
    if not isinstance(d, dict):
        return default
    lowered = {k.lower(): v for k, v in d.items()}
    for k in keys:
        if k.lower() in lowered:
            return lowered[k.lower()]
    return default


def _walk(d, path):
    """Walk nested dicts by a dotted/sub-key path, returning None if missing."""
    cur = d
    for part in path:
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return None
    return cur


class AmdSmiWindowsCollector:
    """Poll amd-smi.exe at a fixed interval, in a background thread."""

    def __init__(
        self,
        gpu_index: int = 0,
        interval_sec: float = 1.0,
        workload_label: str = "unknown",
        run_id: str | None = None,
        amd_smi_path: str | None = None,
    ):
        self._gpu_index = gpu_index
        self._interval = interval_sec
        self._workload_label = workload_label
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        self._amd_smi = amd_smi_path or _resolve_amdsmi()
        if self._amd_smi is None:
            raise FileNotFoundError(
                "amd-smi.exe not found. Install the AMD HIP SDK for Windows "
                "(or AMD Software PRO Edition) and ensure amd-smi.exe is on "
                "PATH. As a fallback, use scripts/collect_pdh_windows.py."
            )
        log.info("Using amd-smi at %s", self._amd_smi)

        self._gpu_metadata = self._snapshot_static()

    # ------------------------------------------------------------------
    # amd-smi subprocess wrappers
    # ------------------------------------------------------------------

    def _run_smi(self, *args, timeout: float = 4.0) -> dict | list | None:
        """Run amd-smi with --json and return parsed JSON, or None on error."""
        cmd = [self._amd_smi, *args, "--json"]
        try:
            res = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=0x08000000 if sys.platform == "win32" else 0,  # CREATE_NO_WINDOW
            )
        except subprocess.TimeoutExpired:
            log.warning("amd-smi %s timed out after %ss", args, timeout)
            return None
        if res.returncode != 0:
            log.debug("amd-smi %s rc=%s stderr=%s", args, res.returncode, res.stderr.strip()[:200])
        if not res.stdout.strip():
            return None
        try:
            return json.loads(res.stdout)
        except json.JSONDecodeError as e:
            log.warning("amd-smi %s JSON parse error: %s", args, e)
            return None

    def _gpu_record(self, payload, gpu_index: int) -> dict:
        """Pull the dict for the requested GPU out of an amd-smi response.

        amd-smi returns either a list of per-GPU dicts, a single dict, or
        a dict keyed by 'gpu_<n>'. Be tolerant.
        """
        if payload is None:
            return {}
        if isinstance(payload, list):
            for entry in payload:
                idx = _first(entry, "gpu", "gpu_id", "gpu_index")
                if idx is not None and int(_num(idx, -1)) == gpu_index:
                    return entry
            return payload[gpu_index] if gpu_index < len(payload) else {}
        if isinstance(payload, dict):
            for key in (f"gpu_{gpu_index}", f"GPU{gpu_index}", str(gpu_index)):
                if key in payload:
                    return payload[key]
            return payload
        return {}

    def _snapshot_static(self) -> dict:
        """Capture GPU metadata once at startup."""
        static = self._run_smi("static", "--gpu", str(self._gpu_index)) or {}
        rec = self._gpu_record(static, self._gpu_index)

        asic = _first(rec, "asic", default={}) or {}
        board = _first(rec, "board", default={}) or {}
        driver = _first(rec, "driver", default={}) or {}
        vbios = _first(rec, "vbios", default={}) or {}
        ras = _first(rec, "ras", default={}) or {}
        limit = _first(rec, "limit", default={}) or {}
        del ras  # unused but kept for documentation

        gpu_name = (
            _first(asic, "market_name", "vendor_name")
            or _first(board, "product_name", "model_number")
            or "AMD GPU"
        )
        meta = {
            "vendor": "AMD",
            "platform": "Windows",
            "gpu_index": self._gpu_index,
            "gpu_name": gpu_name,
            "gpu_uuid": _first(asic, "asic_serial") or _first(board, "product_serial") or "",
            "driver_version": _first(driver, "driver_version", "version", default="unknown"),
            "vbios_version": _first(vbios, "version", default="unknown"),
            "asic_target": _first(asic, "target_graphics_version", "asic_family", default=""),
            "power_limit_w": _num(_first(limit, "power_limit", "max_power_limit")),
        }
        return meta

    # ------------------------------------------------------------------
    # Per-sample extraction (mirrors the Linux _sample_tier1 schema)
    # ------------------------------------------------------------------

    def _sample(self) -> dict:
        now = datetime.now(timezone.utc)
        metric = self._run_smi("metric", "--gpu", str(self._gpu_index)) or {}
        rec = self._gpu_record(metric, self._gpu_index)

        usage = _first(rec, "usage", "activity", default={}) or {}
        gfx_pct = _num(_first(usage, "gfx_activity", "gfx", "graphics_activity"))
        umc_pct = _num(_first(usage, "umc_activity", "umc", "mem_activity", "memory_activity"))
        mm_pct = _num(_first(usage, "mm_activity", "mm", "multimedia_activity"))

        power = _first(rec, "power", default={}) or {}
        power_w = _num(_first(power, "current_socket_power", "socket_power", "average_socket_power"))
        power_limit_w = _num(_first(power, "power_limit", "current_power_limit"),
                             default=self._gpu_metadata.get("power_limit_w", float("nan")))

        temp = _first(rec, "temperature", default={}) or {}
        temp_edge = _num(_first(temp, "edge", "current"))
        temp_hotspot = _num(_first(temp, "hotspot", "junction"))
        temp_vram = _num(_first(temp, "vram", "memory", "mem"))

        clock = _first(rec, "clock", default={}) or {}
        gfx_clk_block = _walk(clock, ["gfx_0"]) or _walk(clock, ["gfx"]) or _first(clock, "gfx_clk", default={})
        mem_clk_block = _walk(clock, ["mem_0"]) or _walk(clock, ["mem"]) or _first(clock, "mem_clk", default={})
        gfx_clock = _num(_first(gfx_clk_block if isinstance(gfx_clk_block, dict) else {},
                                "clk", "cur_clk", "current",
                                default=gfx_clk_block if not isinstance(gfx_clk_block, dict) else None))
        mem_clock = _num(_first(mem_clk_block if isinstance(mem_clk_block, dict) else {},
                                "clk", "cur_clk", "current",
                                default=mem_clk_block if not isinstance(mem_clk_block, dict) else None))

        memu = _first(rec, "memory_usage", "memory", default={}) or {}
        mem_used_mb = _num(_first(memu, "vram_used", "used"))
        mem_total_mb = _num(_first(memu, "vram_total", "total"))

        pcie = _first(rec, "pcie", "pcie_metric", default={}) or {}
        pcie_tx_mbps = _num(_first(pcie, "pcie_current_bandwidth_sent", "pcie_tx", "tx"))
        pcie_rx_mbps = _num(_first(pcie, "pcie_current_bandwidth_received", "pcie_rx", "rx"))

        fan = _first(rec, "fan", "fan_speed", default={}) or {}
        if isinstance(fan, dict):
            fan_pct = _num(_first(fan, "speed", "fan_speed", "percent"))
        else:
            fan_pct = _num(fan)

        throttle = _num(_first(rec, "throttle_status", default=float("nan")))

        return {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            # Cross-vendor schema (column names match the NVML/Linux collector)
            "gpu_utilization_pct": gfx_pct,
            "mem_utilization_pct": umc_pct,
            "mem_used_mb": mem_used_mb,
            "mem_total_mb": mem_total_mb,
            "power_draw_w": power_w,
            "temperature_c": temp_edge,
            "sm_clock_mhz": gfx_clock,
            "mem_clock_mhz": mem_clock,
            "pcie_tx_mbps": pcie_tx_mbps,
            "pcie_rx_mbps": pcie_rx_mbps,
            "encoder_util_pct": mm_pct,
            "decoder_util_pct": mm_pct,
            "fan_speed_pct": fan_pct,
            # AMD-specific extensions
            "temperature_hotspot_c": temp_hotspot,
            "temperature_vram_c": temp_vram,
            "power_limit_w": power_limit_w,
            "throttle_status": throttle,
            "mm_combined": True,
            # Bookkeeping
            "workload_label": self._workload_label,
            "run_id": self._run_id,
        }

    # ------------------------------------------------------------------
    # Threading + I/O
    # ------------------------------------------------------------------

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info(
            "Telemetry started (gpu=%s, interval=%.1fs, label=%s, run=%s)",
            self._gpu_metadata.get("gpu_name"),
            self._interval,
            self._workload_label,
            self._run_id,
        )

    def stop(self):
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
                row = self._sample()
            except Exception as e:
                log.warning("amd-smi sample error: %s", e)
                row = self._nan_row()
            self._records.append(row)
            elapsed = time.monotonic() - t0
            self._stop_event.wait(max(0.0, self._interval - elapsed))

    def _nan_row(self) -> dict:
        now = datetime.now(timezone.utc)
        nan = float("nan")
        cols = [
            "gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb", "mem_total_mb",
            "power_draw_w", "temperature_c", "sm_clock_mhz", "mem_clock_mhz",
            "pcie_tx_mbps", "pcie_rx_mbps", "encoder_util_pct", "decoder_util_pct",
            "fan_speed_pct", "temperature_hotspot_c", "temperature_vram_c",
            "power_limit_w", "throttle_status",
        ]
        row = {c: nan for c in cols}
        row.update({
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            "mm_combined": True,
            "workload_label": self._workload_label,
            "run_id": self._run_id,
        })
        return row

    def to_dataframe(self) -> pd.DataFrame:
        df = pd.DataFrame(self._records)
        for k, v in self._gpu_metadata.items():
            df[k] = v
        return df

    def save(self, output_path: str, fmt: str = "parquet") -> str:
        df = self.to_dataframe()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "parquet" or path.suffix == ".parquet":
            df.to_parquet(path, index=False)
        else:
            df.to_csv(path, index=False)
        log.info("Saved %d rows to %s", len(df), path)
        return str(path)

    @property
    def sample_count(self) -> int:
        return len(self._records)


def main():
    ap = argparse.ArgumentParser(
        description="Poll AMD GPU telemetry on Windows via amd-smi.exe."
    )
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--output", type=str,
                    default="data/raw_telemetry/amdsmi_windows_test.parquet")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--gpu-index", type=int, default=0)
    ap.add_argument("--label", type=str, default="standalone_test_amd_windows")
    ap.add_argument("--amd-smi-path", type=str, default=None,
                    help="Override amd-smi.exe location.")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    c = AmdSmiWindowsCollector(
        gpu_index=args.gpu_index,
        interval_sec=args.interval,
        workload_label=args.label,
        amd_smi_path=args.amd_smi_path,
    )
    print(f"Collecting amd-smi telemetry for {args.duration}s "
          f"(interval={args.interval}s, gpu={c._gpu_metadata.get('gpu_name')})")
    c.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    c.stop()
    out = c.save(args.output, fmt=args.format)
    print(f"Saved {c.sample_count} samples to {out}")


if __name__ == "__main__":
    main()
