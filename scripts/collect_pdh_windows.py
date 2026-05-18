"""SPAR GPU Telemetry Collector (AMD / Windows) — PDH + LibreHardwareMonitor.

Fallback path that does NOT require amd-smi.exe. Combines:

  1. Windows Performance Data Helper (PDH) counters — always available,
     no install. Provides per-engine GPU utilization (3D, Compute, Copy,
     Video, Video Decode), VRAM dedicated/shared usage, and per-process
     local-memory totals. Strictly richer than NVML's single utilization %.

  2. LibreHardwareMonitor HTTP/JSON server (default http://localhost:8085/
     data.json) when LHM is running with 'Run Web Server' enabled. Provides
     GPU power, temperatures (core/hotspot/VRAM), clocks (core/memory), fan
     RPM/% — fields that PDH cannot expose on AMD without a vendor SDK.

     The HTTP server path is preferred over LHM's WMI provider because the
     .NET 10 build of LHM ships without the WMI provider; the HTTP server
     works on every build and is easier to debug.

If LHM is not running, the sensor columns are filled with NaN but the PDH
columns still populate. Output schema mirrors collect_amdsmi_windows.py so
downstream feature extraction is identical.

Usage:
    python collect_pdh_windows.py --duration 60 --output data/test.parquet

LibreHardwareMonitor setup (one-time):
    1. Extract LibreHardwareMonitor.zip from
       https://github.com/LibreHardwareMonitor/LibreHardwareMonitor/releases
       to tools/LibreHardwareMonitor/ (the .NET 10 build also works).
    2. Launch LibreHardwareMonitor.exe AS ADMIN (sensor access requires it).
    3. Options -> Run Web Server (default port 8085).
    4. Options -> Run On Windows Startup (optional but recommended).
"""

import argparse
import json
import logging
import threading
import time
import urllib.error
import urllib.request
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
import win32pdh

log = logging.getLogger("spar.telemetry_pdh_windows")

# Canonical engine roles we surface as columns, and the literal engtype
# prefixes PDH reports for each role on AMD-Windows. PDH instance names
# look like '...engtype_Video Codec Engine' (with embedded spaces and
# numeric suffixes); we match by lowercase prefix.
#
# On RDNA 4 dGPU, video encode+decode share one 'Video Codec Engine'
# (matches the report's mm_combined=True case). On the iGPU they are
# split into 'Video Codec 0' (encode) and 'Video Decode 1' (decode).
# 'High Priority *' queues exist for preemptable workloads.
_ENGINE_ROLES: dict[str, list[str]] = {
    "3d":           ["3D", "High Priority 3D"],
    "compute":      ["Compute", "High Priority Compute"],  # Compute 0, Compute 1, ...
    "copy":         ["Copy"],
    "video":        ["Video Codec", "Video JPEG"],          # encode (and JPEG)
    "video_decode": ["Video Decode"],                       # explicit decode (iGPU only)
}
_ENGINE_TYPES = list(_ENGINE_ROLES.keys())


@dataclass
class _Adapter:
    luid: str         # 'luid_0x00000000_0x0001087c_phys_0'
    dedicated_total_b: int = 0  # capacity (Total Committed)
    shared_total_b: int = 0
    name: str = ""    # populated from WMI if available

    @property
    def short(self) -> str:
        return self.luid


def _enum_pdh_instances(counter_set: str) -> list[str]:
    try:
        _, instances = win32pdh.EnumObjectItems(
            None, None, counter_set, win32pdh.PERF_DETAIL_WIZARD
        )
        return list(instances)
    except Exception as e:
        log.debug("EnumObjectItems(%s) failed: %s", counter_set, e)
        return []


def _parse_luid(instance: str) -> str | None:
    """Extract 'luid_HI_LO_phys_N' substring from a PDH instance name."""
    parts = instance.split("_")
    try:
        i = parts.index("luid")
    except ValueError:
        return None
    # luid_<HI>_<LO>_phys_<N>
    if i + 4 >= len(parts):
        return None
    return "_".join(parts[i:i + 5])


def _parse_engtype(instance: str) -> str | None:
    """Map a PDH GPU Engine instance name to a canonical role key.

    PDH instance names are like '..._engtype_Video Codec Engine' (note the
    embedded spaces). We compare the substring after 'engtype_' against the
    role-prefix table. 'video_decode' is checked before 'video' because
    'Video Decode' would otherwise match the broader 'video' role first if
    its prefix list grew.
    """
    idx = instance.find("engtype_")
    if idx < 0:
        return None
    raw = instance[idx + len("engtype_"):].strip()
    raw_lower = raw.lower()
    # Check more specific roles before more general ones
    role_order = ["video_decode", "video", "compute", "3d", "copy"]
    for role in role_order:
        for prefix in _ENGINE_ROLES[role]:
            if raw_lower.startswith(prefix.lower()):
                return role
    return None


def discover_adapters() -> list[_Adapter]:
    """Enumerate visible GPU adapters by LUID, ordered largest dedicated VRAM first."""
    adapters: dict[str, _Adapter] = {}

    for counter_name, attr in [
        ("Dedicated Usage", "dedicated_total_b"),
        ("Shared Usage", "shared_total_b"),
    ]:
        path = f"\\GPU Adapter Memory(*)\\{counter_name}"
        try:
            hq = win32pdh.OpenQuery()
            hc = win32pdh.AddEnglishCounter(hq, path)
            win32pdh.CollectQueryData(hq)
            time.sleep(0.05)
            win32pdh.CollectQueryData(hq)
            items = win32pdh.GetFormattedCounterArray(hc, win32pdh.PDH_FMT_LARGE)
            for inst, val in items.items():
                if not inst:
                    continue
                ad = adapters.setdefault(inst, _Adapter(luid=inst))
                if val and val > getattr(ad, attr):
                    setattr(ad, attr, int(val))
            win32pdh.CloseQuery(hq)
        except Exception as e:
            log.warning("PDH discover (%s) failed: %s", counter_name, e)

    # Sort by dedicated capacity descending — discrete GPU first, iGPU last
    return sorted(adapters.values(), key=lambda a: -a.dedicated_total_b)


# ---------------------------------------------------------------------------
# LibreHardwareMonitor sensor reader — HTTP/JSON server
# ---------------------------------------------------------------------------

import re as _re
_LHM_NUM = _re.compile(r"-?\d+(?:\.\d+)?")


def _parse_lhm_value(s: str) -> float:
    """Parse '19.0 W', '32.0 °C', '124.0 MHz', '0 RPM', '978.0 MB' to float."""
    if s is None:
        return float("nan")
    if isinstance(s, (int, float)):
        return float(s)
    m = _LHM_NUM.search(str(s))
    return float(m.group(0)) if m else float("nan")


@dataclass
class _LhmState:
    available: bool = False
    error: str = ""
    base_url: str = ""
    gpu_text: str = ""


class LhmHttpReader:
    """Poll LibreHardwareMonitor's HTTP/JSON server for GPU sensors.

    LHM exposes the full sensor tree at /data.json when 'Run Web Server' is
    enabled (Options menu). The tree is keyed by Text labels; we walk it
    once at startup to locate the discrete GPU node (preferring 'RX 9070
    XT' / explicit dGPU name over the integrated Radeon), then refetch the
    full payload each sample. The payload is small (~30 KB) and parsing is
    cheap — single-digit ms per sample.

    Sensor layout (per [scripts/collect_pdh_windows.py probe of LHM v0.9.6]):
        Voltages/      'GPU Core'
        Powers/        'GPU Package'                            -> power_w
        Clocks/        'GPU Core', 'GPU Memory'                 -> *_clock_*_mhz
        Temperatures/  'GPU Core', 'GPU Hot Spot', 'GPU Memory' -> temp_*_c
        Load/          'GPU Core', 'GPU Memory', 'D3D <eng>'    -> load_*_pct
        Fans/          'GPU Fan'  (RPM)                         -> fan_rpm
        Controls/      'GPU Fan'  (% duty)                      -> fan_pct
        Data/          'GPU Memory Used/Total', 'D3D Dedicated/Shared *'

    On RDNA 4 the discrete dGPU has all of these; the integrated AMD Radeon
    Graphics is missing Fans/Controls and many sensor reads return -1.
    """

    DEFAULT_URL = "http://localhost:8085/data.json"

    def __init__(self, base_url: str | None = None, gpu_text_hint: str | None = None,
                 timeout_sec: float = 2.0):
        self.state = _LhmState(base_url=base_url or self.DEFAULT_URL)
        self._timeout = timeout_sec

        # Probe once to locate the dGPU node and confirm the server is up
        try:
            payload = self._fetch()
        except Exception as e:
            self.state.error = f"HTTP probe failed: {e}"
            return

        gpu = self._locate_gpu(payload, gpu_text_hint)
        if gpu is None:
            self.state.error = "no AMD GPU node found in LHM tree"
            return
        self.state.gpu_text = gpu["Text"]
        self.state.available = True
        log.info("LHM HTTP reader: %s @ %s", self.state.gpu_text, self.state.base_url)

    def _fetch(self) -> dict:
        with urllib.request.urlopen(self.state.base_url, timeout=self._timeout) as r:
            return json.loads(r.read())

    @staticmethod
    def _locate_gpu(root: dict, hint: str | None) -> dict | None:
        """Find the GPU node whose Text best matches the hint.

        Default: prefer a node containing 'RX 9070' or 'Radeon RX' (discrete
        AMD card) over 'Radeon(TM) Graphics' (integrated). When `hint` is
        given, prefer nodes whose Text contains it (case-insensitive).
        """
        gpu_nodes: list[dict] = []
        def walk(n):
            text = (n.get("Text", "") or "")
            tl = text.lower()
            if "radeon" in tl or "geforce" in tl or "intel(r) graphics" in tl or " gpu" in tl:
                # filter: must have a Load or Powers child to count as a GPU node
                if any(c.get("Text") in ("Load", "Powers", "Clocks") for c in n.get("Children", [])):
                    gpu_nodes.append(n)
            for c in n.get("Children", []):
                walk(c)
        walk(root)
        if not gpu_nodes:
            return None

        if hint:
            h = hint.lower()
            for n in gpu_nodes:
                if h in n["Text"].lower():
                    return n
        # Default ranking: RX 9070 / explicit dGPU before integrated
        def rank(n):
            t = n["Text"].lower()
            if "rx 9070" in t or "rx 7900" in t:
                return 0           # explicit dGPU
            if "(tm) graphics" in t or "integrated" in t:
                return 2           # iGPU
            return 1
        gpu_nodes.sort(key=rank)
        return gpu_nodes[0]

    def sample(self) -> dict:
        out = {
            "lhm_power_w":         float("nan"),
            "lhm_temp_core_c":     float("nan"),
            "lhm_temp_hotspot_c":  float("nan"),
            "lhm_temp_vram_c":     float("nan"),
            "lhm_clock_core_mhz":  float("nan"),
            "lhm_clock_mem_mhz":   float("nan"),
            "lhm_fan_pct":         float("nan"),
            "lhm_fan_rpm":         float("nan"),
            "lhm_load_core_pct":   float("nan"),
            "lhm_load_mem_pct":    float("nan"),
            "lhm_mem_used_mb":     float("nan"),
            "lhm_mem_total_mb":    float("nan"),
        }
        if not self.state.available:
            return out

        try:
            payload = self._fetch()
            gpu = self._locate_gpu(payload, self.state.gpu_text)
        except Exception as e:
            log.debug("LHM HTTP sample failed: %s", e)
            return out
        if gpu is None:
            return out

        for cat in gpu.get("Children", []):
            cat_name = cat.get("Text", "")
            for s in cat.get("Children", []):
                name = (s.get("Text", "") or "").lower()
                val = _parse_lhm_value(s.get("Value", ""))
                if cat_name == "Powers" and ("package" in name or out["lhm_power_w"] != out["lhm_power_w"]):
                    out["lhm_power_w"] = val
                elif cat_name == "Temperatures":
                    if "hot spot" in name or "hotspot" in name:
                        out["lhm_temp_hotspot_c"] = val
                    elif "memory" in name or "vram" in name:
                        out["lhm_temp_vram_c"] = val
                    elif "core" in name or name == "gpu":
                        out["lhm_temp_core_c"] = val
                elif cat_name == "Clocks":
                    if "memory" in name or "mem" in name:
                        out["lhm_clock_mem_mhz"] = val
                    elif "core" in name:
                        out["lhm_clock_core_mhz"] = val
                elif cat_name == "Fans" and "gpu" in name:
                    out["lhm_fan_rpm"] = val
                elif cat_name == "Controls" and "gpu" in name:
                    out["lhm_fan_pct"] = val
                elif cat_name == "Load":
                    if "memory" in name and "d3d" not in name:
                        out["lhm_load_mem_pct"] = val
                    elif name == "gpu core":
                        out["lhm_load_core_pct"] = val
                elif cat_name == "Data":
                    # Prefer 'GPU Memory Used/Total' (driver-reported VRAM) over
                    # 'D3D Dedicated Memory *' (Direct3D's accounting)
                    if name == "gpu memory used":
                        out["lhm_mem_used_mb"] = val
                    elif name == "gpu memory total":
                        out["lhm_mem_total_mb"] = val
        return out


# ---------------------------------------------------------------------------
# PDH GPU counter reader
# ---------------------------------------------------------------------------

class PdhGpuReader:
    """Read GPU Engine + GPU Adapter Memory counters from PDH."""

    def __init__(self, target_luid: str):
        # Match case-insensitively — PDH returns uppercase hex, but PowerShell
        # and amd-smi often surface lowercase. We compare on the lowercase form.
        self.target_luid = target_luid.lower()

        self._hq = win32pdh.OpenQuery()
        self._eng_path = "\\GPU Engine(*)\\Utilization Percentage"
        self._eng_handle = win32pdh.AddEnglishCounter(self._hq, self._eng_path)
        self._mem_dedicated_path = "\\GPU Adapter Memory(*)\\Dedicated Usage"
        self._mem_shared_path = "\\GPU Adapter Memory(*)\\Shared Usage"
        self._mem_dedicated_handle = win32pdh.AddEnglishCounter(self._hq, self._mem_dedicated_path)
        self._mem_shared_handle = win32pdh.AddEnglishCounter(self._hq, self._mem_shared_path)
        self._proc_local_path = "\\GPU Process Memory(*)\\Local Usage"
        self._proc_local_handle = win32pdh.AddEnglishCounter(self._hq, self._proc_local_path)

        # Prime the query — % counters need two samples to compute deltas
        win32pdh.CollectQueryData(self._hq)
        time.sleep(0.05)

    @staticmethod
    def _array(handle, fmt) -> dict:
        """GetFormattedCounterArray returns dict[instance_name -> value] in pywin32."""
        try:
            return win32pdh.GetFormattedCounterArray(handle, fmt) or {}
        except Exception as e:
            log.debug("GetFormattedCounterArray failed: %s", e)
            return {}

    def sample(self) -> dict:
        win32pdh.CollectQueryData(self._hq)
        out = {f"gpu_engine_{role}_pct": float("nan") for role in _ENGINE_TYPES}
        out["gpu_engine_total_pct"] = float("nan")
        out["mem_used_dedicated_mb"] = float("nan")
        out["mem_used_shared_mb"] = float("nan")
        out["mem_used_proc_local_mb"] = float("nan")

        # --- Engine util: sum across PIDs and per-role queues for our LUID ---
        items = self._array(self._eng_handle, win32pdh.PDH_FMT_DOUBLE)
        engsum = {role: 0.0 for role in _ENGINE_TYPES}
        engfound = {role: False for role in _ENGINE_TYPES}
        for inst, val in items.items():
            if self.target_luid not in inst.lower():
                continue
            role = _parse_engtype(inst)
            if role in engsum and val is not None:
                engsum[role] += float(val)
                engfound[role] = True
        for role in _ENGINE_TYPES:
            if engfound[role]:
                # Cap at 100 — multiple PIDs / numbered queues sharing a role can sum > 100
                out[f"gpu_engine_{role}_pct"] = min(100.0, engsum[role])

        # Cross-vendor "primary" GPU utilization: max(3D, Compute, Video).
        # ML compute surfaces in 3D (DirectX), Compute (DirectML/HIP/Vulkan
        # compute), or Video (encode-driven workloads). Take the max because
        # only one is typically the bottleneck for any given workload.
        def _nz(v):
            return v if v == v else 0.0  # NaN-safe
        out["gpu_engine_total_pct"] = max(
            _nz(out["gpu_engine_3d_pct"]),
            _nz(out["gpu_engine_compute_pct"]),
            _nz(out["gpu_engine_video_pct"]),
        )

        # --- Dedicated VRAM ---
        items = self._array(self._mem_dedicated_handle, win32pdh.PDH_FMT_LARGE)
        for inst, val in items.items():
            if inst.lower() == self.target_luid and val is not None:
                out["mem_used_dedicated_mb"] = float(val) / (1024 * 1024)

        # --- Shared (sysmem) VRAM ---
        items = self._array(self._mem_shared_handle, win32pdh.PDH_FMT_LARGE)
        for inst, val in items.items():
            if inst.lower() == self.target_luid and val is not None:
                out["mem_used_shared_mb"] = float(val) / (1024 * 1024)

        # --- Process-local VRAM sum ---
        items = self._array(self._proc_local_handle, win32pdh.PDH_FMT_LARGE)
        total = 0
        saw_match = False
        for inst, val in items.items():
            if self.target_luid in inst.lower() and val is not None:
                total += int(val)
                saw_match = True
        if saw_match:
            out["mem_used_proc_local_mb"] = total / (1024 * 1024)

        return out

    def close(self):
        try:
            win32pdh.CloseQuery(self._hq)
        except Exception:
            pass


# ---------------------------------------------------------------------------
# Combined collector
# ---------------------------------------------------------------------------

class PdhCollector:
    """Polls PDH + LHM at a fixed interval. Mirrors AmdSmiWindowsCollector
    interface so workload runners can swap them transparently."""

    def __init__(
        self,
        adapter_luid: str | None = None,
        interval_sec: float = 1.0,
        workload_label: str = "unknown",
        run_id: str | None = None,
        lhm_url: str | None = None,
        gpu_text_hint: str | None = None,
    ):
        self._interval = interval_sec
        self._workload_label = workload_label
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

        if adapter_luid is None:
            adapters = discover_adapters()
            if not adapters:
                raise RuntimeError("No GPU adapters discoverable via PDH.")
            self._adapter = adapters[0]
            log.info(
                "Auto-selected adapter %s (dedicated_max=%.1f MB) — alternates: %s",
                self._adapter.luid,
                self._adapter.dedicated_total_b / (1024 * 1024),
                [a.luid for a in adapters[1:]],
            )
        else:
            self._adapter = _Adapter(luid=adapter_luid)

        self._pdh = PdhGpuReader(self._adapter.luid)
        self._lhm = LhmHttpReader(base_url=lhm_url, gpu_text_hint=gpu_text_hint)
        if not self._lhm.state.available:
            log.info("LHM sensors unavailable: %s", self._lhm.state.error)

        self._gpu_metadata = {
            "vendor": "AMD",
            "platform": "Windows",
            "gpu_index": 0,
            "gpu_name": self._lhm.state.gpu_text or "Unknown (PDH-only)",
            "adapter_luid": self._adapter.luid,
            "lhm_available": self._lhm.state.available,
            "lhm_url": self._lhm.state.base_url,
            "lhm_error": self._lhm.state.error,
        }

    def start(self):
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("PDH telemetry started (label=%s, run=%s)",
                 self._workload_label, self._run_id)

    def stop(self):
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        log.info("PDH telemetry stopped. %d samples collected.", len(self._records))

    def _poll_loop(self):
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            row = self._sample()
            self._records.append(row)
            elapsed = time.monotonic() - t0
            self._stop_event.wait(max(0.0, self._interval - elapsed))

    def _sample(self) -> dict:
        now = datetime.now(timezone.utc)
        pdh = self._pdh.sample()
        lhm = self._lhm.sample()

        # Build cross-vendor schema row. Values come from PDH for util/memory
        # and LHM for sensor signals where available.
        row = {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            # Cross-vendor (matching the NVML/Linux schema):
            "gpu_utilization_pct": pdh["gpu_engine_total_pct"],
            "mem_utilization_pct": lhm["lhm_load_mem_pct"],
            "mem_used_mb": pdh["mem_used_dedicated_mb"],
            "mem_total_mb": lhm["lhm_mem_total_mb"],  # LHM reports VRAM capacity
            "power_draw_w": lhm["lhm_power_w"],
            "temperature_c": lhm["lhm_temp_core_c"],
            "sm_clock_mhz": lhm["lhm_clock_core_mhz"],
            "mem_clock_mhz": lhm["lhm_clock_mem_mhz"],
            "pcie_tx_mbps": float("nan"),
            "pcie_rx_mbps": float("nan"),
            # On RDNA 4 dGPU, encode and decode share 'Video Codec Engine',
            # so the cross-vendor encoder/decoder columns get the same value
            # (mm_combined=False here means: not the AMD-amdsmi 'mm' counter,
            # but PDH-derived). On the iGPU, decode has its own counter.
            "encoder_util_pct": pdh["gpu_engine_video_pct"],
            "decoder_util_pct": (pdh["gpu_engine_video_decode_pct"]
                                 if pdh["gpu_engine_video_decode_pct"] == pdh["gpu_engine_video_decode_pct"]
                                 else pdh["gpu_engine_video_pct"]),
            "fan_speed_pct": lhm["lhm_fan_pct"],
            "fan_speed_rpm": lhm["lhm_fan_rpm"],
            # AMD-specific extensions
            "temperature_hotspot_c": lhm["lhm_temp_hotspot_c"],
            "temperature_vram_c": lhm["lhm_temp_vram_c"],
            "power_limit_w": float("nan"),
            "throttle_status": float("nan"),
            "mm_combined": False,
            # Windows-PDH-specific extensions: full per-engine breakdown
            # (strictly richer than NVML; analogous to DCGM's per-pipeline split)
            "gpu_engine_3d_pct": pdh["gpu_engine_3d_pct"],
            "gpu_engine_compute_pct": pdh["gpu_engine_compute_pct"],
            "gpu_engine_copy_pct": pdh["gpu_engine_copy_pct"],
            "gpu_engine_video_pct": pdh["gpu_engine_video_pct"],
            "gpu_engine_video_decode_pct": pdh["gpu_engine_video_decode_pct"],
            "mem_used_dedicated_mb": pdh["mem_used_dedicated_mb"],
            "mem_used_shared_mb": pdh["mem_used_shared_mb"],
            "mem_used_proc_local_mb": pdh["mem_used_proc_local_mb"],
            # Bookkeeping
            "workload_label": self._workload_label,
            "run_id": self._run_id,
        }
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
        description="Poll AMD GPU telemetry on Windows via PDH + LibreHardwareMonitor."
    )
    ap.add_argument("--duration", type=int, default=60)
    ap.add_argument("--output", type=str,
                    default="data/raw_telemetry/pdh_windows_test.parquet")
    ap.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--label", type=str, default="standalone_test_pdh")
    ap.add_argument("--adapter-luid", type=str, default=None,
                    help="LUID like 'luid_0x00000000_0x0001087c_phys_0'. "
                         "Default: auto-detect largest dedicated-VRAM adapter.")
    ap.add_argument("--list-adapters", action="store_true",
                    help="List visible GPU adapters by LUID and exit.")
    ap.add_argument("--lhm-url", type=str, default=None,
                    help="LibreHardwareMonitor JSON endpoint "
                         "(default: http://localhost:8085/data.json).")
    ap.add_argument("--gpu-text", type=str, default=None,
                    help="Substring to prefer when multiple GPUs are visible "
                         "to LHM (default: prefer 'RX 9070'/'RX 7900').")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    if args.list_adapters:
        for a in discover_adapters():
            print(f"{a.luid}\tdedicated_max_mb={a.dedicated_total_b / (1024*1024):.1f}")
        return

    c = PdhCollector(
        adapter_luid=args.adapter_luid,
        interval_sec=args.interval,
        workload_label=args.label,
        lhm_url=args.lhm_url,
        gpu_text_hint=args.gpu_text,
    )
    print(f"Collecting PDH+LHM telemetry for {args.duration}s "
          f"(interval={args.interval}s, luid={c._adapter.luid}, "
          f"lhm={'on:'+c._lhm.state.gpu_text if c._lhm.state.available else 'off'})")
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
