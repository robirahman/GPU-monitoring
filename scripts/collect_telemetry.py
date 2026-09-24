#!/usr/bin/env python3
"""SPAR GPU Telemetry Collector — Tier 1 (pynvml) + Tier 2 (DCGM) + fabric.

Polls GPU metrics at a configurable interval (default 1 Hz) in a background
thread and writes labeled data to Parquet or CSV.

Also fuses interconnect counters when available (default on):
  * NVLink TX/RX via NVML field values (per-GPU)
  * InfiniBand / RoCE sysfs counters (node-level, shared across GPUs)
  * Ethernet NIC byte rates (node-level fallback)
  * Topology stamp (hostname / nnodes / node_rank / world_size)

Standalone usage:
    python collect_telemetry.py --duration 60 --output data/test.parquet

As a library (used by run_workload.py):
    collector = TelemetryCollector(workload_label="resnet_cifar10")
    collector.start()
    # ... run workload ...
    collector.stop()
    collector.save("data/output.parquet")
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

try:
    import pynvml
except ImportError:
    print(
        "ERROR: pynvml not found. Install with: pip install nvidia-ml-py3",
        file=sys.stderr,
    )
    sys.exit(1)

log = logging.getLogger("spar.telemetry")


# ---------------------------------------------------------------------------
# GpuMetadata — static GPU information captured once at startup
# ---------------------------------------------------------------------------

class GpuMetadata:
    """Captures static GPU properties via pynvml."""

    def __init__(self, gpu_index: int = 0):
        pynvml.nvmlInit()
        self._handle = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self.gpu_index = gpu_index
        self.gpu_name = pynvml.nvmlDeviceGetName(self._handle)
        if isinstance(self.gpu_name, bytes):
            self.gpu_name = self.gpu_name.decode("utf-8")
        self.gpu_uuid = pynvml.nvmlDeviceGetUUID(self._handle)
        if isinstance(self.gpu_uuid, bytes):
            self.gpu_uuid = self.gpu_uuid.decode("utf-8")
        self.driver_version = pynvml.nvmlSystemGetDriverVersion()
        if isinstance(self.driver_version, bytes):
            self.driver_version = self.driver_version.decode("utf-8")

        try:
            cuda_ver_raw = pynvml.nvmlSystemGetCudaDriverVersion_v2()
            self.cuda_version = f"{cuda_ver_raw // 1000}.{(cuda_ver_raw % 1000) // 10}"
        except AttributeError:
            # Older pynvml builds lack this function; extract from nvidia-smi output
            try:
                import subprocess
                out = subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
                    text=True,
                ).strip()
                self.cuda_version = "unknown"
            except Exception:
                self.cuda_version = "unknown"

        mem_info = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
        self.total_memory_mb = mem_info.total // (1024 * 1024)

        try:
            self.pcie_gen = pynvml.nvmlDeviceGetCurrPcieLinkGeneration(self._handle)
            self.pcie_width = pynvml.nvmlDeviceGetCurrPcieLinkWidth(self._handle)
        except pynvml.NVMLError:
            self.pcie_gen = -1
            self.pcie_width = -1

    @property
    def handle(self):
        return self._handle

    def to_dict(self) -> dict:
        return {
            "gpu_index": self.gpu_index,
            "gpu_name": self.gpu_name,
            "gpu_uuid": self.gpu_uuid,
            "driver_version": self.driver_version,
            "cuda_version": self.cuda_version,
            "total_memory_mb": self.total_memory_mb,
            "pcie_gen": self.pcie_gen,
            "pcie_width": self.pcie_width,
        }

    def cleanup(self):
        try:
            pynvml.nvmlShutdown()
        except pynvml.NVMLError:
            pass


# ---------------------------------------------------------------------------
# DcgmCollector — optional Tier 2 DCGM profiling metrics
# ---------------------------------------------------------------------------

class DcgmCollector:
    """Collects Tier 2 DCGM profiling metrics. Raises ImportError or
    RuntimeError if DCGM is not available, letting the caller skip Tier 2.

    On data-center GPUs (A100, H100): uses profiling fields (1001-1008) for
    tensor core utilization, pipe activity, SM occupancy, DRAM bandwidth.

    On consumer GPUs (RTX 3090, 5080) or when profiling module is unavailable:
    falls back to basic device fields that overlap with Tier 1.
    """

    # DCGM profiling field IDs (Tier 2 — data-center GPUs only)
    PROF_FIELDS = {
        1001: "dcgm_gr_engine_active",
        1002: "dcgm_sm_active",
        1003: "dcgm_sm_occupancy",
        1004: "dcgm_tensor_active",
        1005: "dcgm_dram_active",
        1006: "dcgm_fp64_pipe_active",
        1007: "dcgm_fp32_pipe_active",
        1008: "dcgm_fp16_pipe_active",
    }

    # NVLink fields (added if GPU supports NVLink)
    NVLINK_FIELDS = {
        1011: "dcgm_nvlink_tx_bytes",
        1012: "dcgm_nvlink_rx_bytes",
    }

    # Basic device fields (fallback — available on all GPUs)
    BASIC_FIELDS = {
        203: "dcgm_gpu_util",
        155: "dcgm_power_usage",
        150: "dcgm_gpu_temp",
        100: "dcgm_sm_clock",
        101: "dcgm_mem_clock",
    }

    def __init__(self, gpu_index: int = 0):
        import sys as _sys
        # DCGM Python bindings ship with the system package, not pip
        dcgm_path = "/usr/local/dcgm/bindings/python3"
        if dcgm_path not in _sys.path:
            _sys.path.insert(0, dcgm_path)

        import dcgm_structs
        import dcgm_agent
        import dcgm_fields  # noqa: F811
        import pydcgm

        self._dcgm_agent = dcgm_agent
        self._dcgm_structs = dcgm_structs
        self._pydcgm = pydcgm

        # Load the DCGM shared library
        for lib_path in [
            "/usr/lib/x86_64-linux-gnu/libdcgm.so",
            "/usr/lib/x86_64-linux-gnu/libdcgm.so.3",
        ]:
            if os.path.exists(lib_path):
                dcgm_structs._dcgmInit(lib_path)
                break

        # Connect to DCGM host engine; try standalone first, fall back to embedded
        try:
            self._dcgm_handle = pydcgm.DcgmHandle(ipAddress="127.0.0.1")
        except Exception:
            self._dcgm_handle = pydcgm.DcgmHandle(
                opMode=dcgm_structs.DCGM_OPERATION_MODE_AUTO
            )
        self._group = pydcgm.DcgmGroup(
            self._dcgm_handle,
            groupName="spar_telemetry",
            groupType=dcgm_structs.DCGM_GROUP_DEFAULT,
        )
        self._gpu_index = gpu_index

        # Try profiling fields first, fall back to basic if unsupported
        self._using_profiling = False
        try:
            all_field_ids = list(self.PROF_FIELDS.keys()) + list(self.NVLINK_FIELDS.keys())
            self._field_map = {**self.PROF_FIELDS, **self.NVLINK_FIELDS}
            self._field_group = pydcgm.DcgmFieldGroup(
                self._dcgm_handle, name="spar_tier2_prof", fieldIds=all_field_ids
            )
            self._group.samples.WatchFields(
                self._field_group,
                updateFreq=1000000,
                maxKeepAge=30.0,
                maxKeepSamples=30,
            )
            self._using_profiling = True
            log.info(
                "DCGM Tier 2 collector initialized with PROFILING fields (gpu_index=%d)",
                gpu_index,
            )
        except Exception as e:
            log.info(
                "DCGM profiling fields unavailable (%s); using basic device fields. "
                "Profiling fields require data-center GPUs (A100, H100).",
                e,
            )
            # Fall back to basic device fields
            self._field_map = dict(self.BASIC_FIELDS)
            self._field_group = pydcgm.DcgmFieldGroup(
                self._dcgm_handle,
                name="spar_tier2_basic",
                fieldIds=list(self.BASIC_FIELDS.keys()),
            )
            self._group.samples.WatchFields(
                self._field_group,
                updateFreq=1000000,
                maxKeepAge=30.0,
                maxKeepSamples=30,
            )
            log.info(
                "DCGM Tier 2 collector initialized with BASIC fields (gpu_index=%d)",
                gpu_index,
            )

    def sample(self) -> dict:
        """Read latest DCGM field values. Returns dict with dcgm_* keys."""
        result = {}
        try:
            values = self._group.samples.GetLatest(self._field_group).values
            gpu_values = values.get(self._gpu_index, {})
            for field_id, col_name in self._field_map.items():
                field_vals = gpu_values.get(field_id, [])
                if field_vals:
                    val = field_vals[-1].value
                    result[col_name] = float(val) if val is not None else float("nan")
                else:
                    result[col_name] = float("nan")
        except Exception as e:
            log.warning("DCGM sample error: %s", e)
            for col_name in self._field_map.values():
                result[col_name] = float("nan")
        return result

    def cleanup(self):
        try:
            self._group.samples.UnwatchFields(self._field_group)
        except Exception:
            pass
        # Prevent noisy __del__ errors from DCGM's DcgmFieldGroup by
        # marking it as already deleted (its Delete() checks fieldGroupId)
        try:
            self._field_group.fieldGroupId = None
        except Exception:
            pass
        try:
            self._dcgm_handle.Shutdown()
        except Exception:
            pass


# ---------------------------------------------------------------------------
# TelemetryCollector — main polling engine
# ---------------------------------------------------------------------------

class TelemetryCollector:
    """Polls GPU metrics at a fixed interval via pynvml (Tier 1),
    optionally DCGM (Tier 2), and optionally fabric counters (NVLink/IB/eth).

    Runs in a background daemon thread. For multi-GPU on one node, pass the
    same SharedFabricCache instance to every collector so IB/eth deltas are
    not double-consumed.
    """

    def __init__(
        self,
        gpu_index: int = 0,
        interval_sec: float = 1.0,
        workload_label: str = "unknown",
        run_id: str | None = None,
        enable_dcgm: bool = True,
        checkpoint_sec: float = 300.0,
        checkpoint_path: str | None = None,
        enable_fabric: bool = True,
        fabric_cache: "SharedFabricCache | None" = None,
        topology: dict | None = None,
        ib_root: str = "/sys/class/infiniband",
        net_root: str = "/sys/class/net",
        workload_role: str | None = None,
        role_label: str | None = None,
    ):
        self.metadata = GpuMetadata(gpu_index)
        self._handle = self.metadata.handle
        self._interval = interval_sec
        self._workload_label = workload_label
        # W25 colocation: several concurrent workloads share one node/run. ``workload_role``
        # (serve|train) and ``role_label`` (that role's own workload label) are stamped on every
        # row when set, so per-GPU rows stay attributable; None (default) leaves rows untouched.
        self._workload_role = workload_role
        self._role_label = role_label
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._gpu_index = gpu_index

        # Periodic crash-safety checkpoint. Samples otherwise live only in
        # self._records until save() runs after the workload exits, so a crash
        # in hour 8 of an 8 h run would lose all 28,800 rows. Set
        # checkpoint_path (or let run_workload.py set it) to flush a rolling
        # snapshot; save() removes it on a clean finish.
        self._checkpoint_sec = checkpoint_sec
        self.checkpoint_path: str | None = checkpoint_path
        self._last_checkpoint = 0.0
        self._checkpoint_count = 0

        # Try to initialize DCGM (Tier 2)
        self._dcgm: DcgmCollector | None = None
        if enable_dcgm:
            try:
                self._dcgm = DcgmCollector(gpu_index)
            except ImportError:
                log.info(
                    "DCGM Python bindings not available; Tier 2 metrics will be omitted. "
                    "To enable: apt-get install datacenter-gpu-manager && nv-hostengine -d"
                )
            except Exception as e:
                log.info("DCGM initialization failed (%s); Tier 2 metrics will be omitted.", e)

        # Fabric / interconnect (Tier interconnect): NVLink per-GPU + IB/eth node-level
        self._fabric_cache = None
        self._nvlink = None
        self._topology: dict = {}
        self._last_sample_mono = 0.0
        if enable_fabric:
            try:
                from scripts.interconnect_collect import (
                    SharedFabricCache,
                    GpuNvlinkSampler,
                    infer_topology,
                )
            except ImportError:
                # Allow running as ``python scripts/collect_telemetry.py`` without package path
                from interconnect_collect import (  # type: ignore
                    SharedFabricCache,
                    GpuNvlinkSampler,
                    infer_topology,
                )
            self._fabric_cache = fabric_cache or SharedFabricCache(
                interval_sec=interval_sec, ib_root=ib_root, net_root=net_root
            )
            self._nvlink = GpuNvlinkSampler(gpu_index)
            self._topology = dict(topology or infer_topology())
            ib_ok = bool(self._fabric_cache._ib and self._fabric_cache._ib.available)
            eth_ok = bool(self._fabric_cache._eth and self._fabric_cache._eth.available)
            log.info(
                "Fabric enabled (gpu=%d, nvlink=%s, ib=%s, eth=%s, host=%s, nnodes=%s)",
                gpu_index,
                "yes" if self._nvlink.available else "no",
                "yes" if ib_ok else "no",
                "yes" if eth_ok else "no",
                self._topology.get("hostname"),
                self._topology.get("nnodes"),
            )

    def start(self):
        """Start the background polling thread."""
        self._stop_event.clear()
        self._last_sample_mono = time.monotonic()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info(
            "Telemetry started (gpu=%s, interval=%.1fs, dcgm=%s, fabric=%s, label=%s, run=%s)",
            self.metadata.gpu_name,
            self._interval,
            "enabled" if self._dcgm else "disabled",
            "enabled" if self._fabric_cache else "disabled",
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
        """Main polling loop running in the background thread."""
        self._last_checkpoint = time.monotonic()
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            try:
                row = self._sample_tier1()
            except pynvml.NVMLError as e:
                log.warning("pynvml sample error: %s", e)
                row = self._nan_row()

            if self._dcgm is not None:
                dcgm_data = self._dcgm.sample()
                row.update(dcgm_data)

            # Interconnect / fabric fusion (NVLink per-GPU + shared IB/eth)
            dt = max(t0 - self._last_sample_mono, 1e-6)
            self._last_sample_mono = t0
            if self._nvlink is not None:
                try:
                    row.update(self._nvlink.sample(dt))
                except Exception as e:
                    log.warning("NVLink sample error: %s", e)
            if self._fabric_cache is not None:
                try:
                    row.update(self._fabric_cache.snapshot())
                except Exception as e:
                    log.warning("Fabric sample error: %s", e)
            if self._topology:
                row.update(self._topology)

            row["gpu_index"] = self._gpu_index
            row["workload_label"] = self._workload_label
            if self._workload_role is not None:
                row["workload_role"] = self._workload_role
                row["role_label"] = self._role_label
            row["run_id"] = self._run_id
            self._records.append(row)

            # Flush a crash-safety snapshot every checkpoint_sec. Runs in this
            # thread (the only writer of _records), so no locking is needed.
            if (
                self.checkpoint_path
                and self._checkpoint_sec > 0
                and time.monotonic() - self._last_checkpoint >= self._checkpoint_sec
            ):
                self._write_checkpoint()
                self._last_checkpoint = time.monotonic()

            # Sleep for the remainder of the interval
            elapsed = time.monotonic() - t0
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _sample_tier1(self) -> dict:
        """Sample all Tier 1 NVML metrics. Returns a dict."""
        now = datetime.now(timezone.utc)
        h = self._handle

        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem = pynvml.nvmlDeviceGetMemoryInfo(h)

        row = {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            "gpu_utilization_pct": util.gpu,
            "mem_utilization_pct": util.memory,
            "mem_used_mb": mem.used // (1024 * 1024),
            "mem_total_mb": mem.total // (1024 * 1024),
            "power_draw_w": pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "temperature_c": pynvml.nvmlDeviceGetTemperature(
                h, pynvml.NVML_TEMPERATURE_GPU
            ),
            "sm_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_mhz": pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
        }

        # PCIe throughput (returns KB/s, convert to MB/s)
        try:
            row["pcie_tx_mbps"] = (
                pynvml.nvmlDeviceGetPcieThroughput(
                    h, pynvml.NVML_PCIE_UTIL_TX_BYTES
                )
                / 1024
            )
            row["pcie_rx_mbps"] = (
                pynvml.nvmlDeviceGetPcieThroughput(
                    h, pynvml.NVML_PCIE_UTIL_RX_BYTES
                )
                / 1024
            )
        except pynvml.NVMLError:
            row["pcie_tx_mbps"] = -1
            row["pcie_rx_mbps"] = -1

        # Encoder/decoder utilization
        try:
            row["encoder_util_pct"] = pynvml.nvmlDeviceGetEncoderUtilization(h)[0]
        except pynvml.NVMLError:
            row["encoder_util_pct"] = -1
        try:
            row["decoder_util_pct"] = pynvml.nvmlDeviceGetDecoderUtilization(h)[0]
        except pynvml.NVMLError:
            row["decoder_util_pct"] = -1

        # Fan speed (unsupported on A100/H100 — no fans in data-center GPUs)
        try:
            row["fan_speed_pct"] = pynvml.nvmlDeviceGetFanSpeed(h)
        except pynvml.NVMLError:
            row["fan_speed_pct"] = -1

        return row

    def _nan_row(self) -> dict:
        """Return a row with NaN for all Tier 1 metrics (used on sample error)."""
        now = datetime.now(timezone.utc)
        nan = float("nan")
        return {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            "gpu_utilization_pct": nan,
            "mem_utilization_pct": nan,
            "mem_used_mb": nan,
            "mem_total_mb": nan,
            "power_draw_w": nan,
            "temperature_c": nan,
            "sm_clock_mhz": nan,
            "mem_clock_mhz": nan,
            "pcie_tx_mbps": nan,
            "pcie_rx_mbps": nan,
            "encoder_util_pct": nan,
            "decoder_util_pct": nan,
            "fan_speed_pct": nan,
        }

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected records to a pandas DataFrame."""
        df = pd.DataFrame(self._records)
        # Add static metadata columns
        df["gpu_name"] = self.metadata.gpu_name
        df["gpu_uuid"] = self.metadata.gpu_uuid
        df["driver_version"] = self.metadata.driver_version
        return df

    def _write_parquet(self, df: pd.DataFrame, path: Path):
        """Write df to a Parquet file, embedding GPU metadata in the schema."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        table = pa.Table.from_pandas(df)
        meta = table.schema.metadata or {}
        meta[b"gpu_metadata"] = json.dumps(self.metadata.to_dict()).encode("utf-8")
        table = table.replace_schema_metadata(meta)
        pq.write_table(table, str(path))

    def _write_checkpoint(self):
        """Atomically write a rolling snapshot of the samples collected so far.

        Best-effort: a checkpoint failure must never kill the polling thread,
        so all exceptions are caught and logged.
        """
        if not self._records:
            return
        try:
            path = Path(self.checkpoint_path)
            path.parent.mkdir(parents=True, exist_ok=True)
            tmp = path.with_suffix(path.suffix + ".tmp")
            self._write_parquet(self.to_dataframe(), tmp)
            os.replace(tmp, path)  # atomic — readers never see a partial file
            self._checkpoint_count += 1
            log.info(
                "Checkpoint %d: %d samples -> %s",
                self._checkpoint_count,
                len(self._records),
                path,
            )
        except Exception as e:
            log.warning("Checkpoint write failed (continuing): %s", e)

    def save(self, output_path: str, fmt: str = "parquet"):
        """Save collected data to Parquet or CSV.

        Also writes a sibling ``*_fabric_summary.json`` with node-level IB/eth
        totals when a fabric cache is attached.
        """
        df = self.to_dataframe()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            self._write_parquet(df, path)
        else:
            df.to_csv(str(path), index=False)

        log.info("Saved %d rows to %s", len(df), path)

        if self._fabric_cache is not None:
            summary_path = path.with_name(path.stem + "_fabric_summary.json")
            try:
                summary = {
                    "topology": self._topology,
                    "gpu_index": self._gpu_index,
                    "workload_label": self._workload_label,
                    "workload_role": self._workload_role,
                    "role_label": self._role_label,
                    "run_id": self._run_id,
                    **self._fabric_cache.summary(),
                }
                summary_path.write_text(json.dumps(summary, indent=2))
                log.info("Fabric summary -> %s", summary_path)
            except Exception as e:
                log.warning("Fabric summary write failed: %s", e)

        # The run finished and the real output is on disk, so the crash-safety
        # snapshot is now redundant.
        if self.checkpoint_path:
            try:
                Path(self.checkpoint_path).unlink(missing_ok=True)
            except OSError as e:
                log.warning("Could not remove checkpoint %s: %s", self.checkpoint_path, e)

        return str(path)

    @property
    def sample_count(self) -> int:
        return len(self._records)

    def cleanup(self):
        """Release resources."""
        if self._dcgm is not None:
            self._dcgm.cleanup()
        self.metadata.cleanup()


# ---------------------------------------------------------------------------
# CLI entrypoint — standalone collection for a fixed duration
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Collect GPU telemetry (Tier 1 + Tier 2) for a fixed duration."
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        help="Collection duration in seconds (default: 60)",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/telemetry_test.parquet",
        help="Output file path (default: data/telemetry_test.parquet)",
    )
    parser.add_argument(
        "--format",
        type=str,
        choices=["parquet", "csv"],
        default="parquet",
        help="Output format (default: parquet)",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=1.0,
        help="Polling interval in seconds (default: 1.0)",
    )
    parser.add_argument(
        "--gpu-index",
        type=int,
        default=0,
        help="GPU device index (default: 0)",
    )
    parser.add_argument(
        "--label",
        type=str,
        default="standalone_test",
        help="Workload label (default: standalone_test)",
    )
    parser.add_argument(
        "--no-dcgm",
        action="store_true",
        help="Disable DCGM Tier 2 collection",
    )
    parser.add_argument(
        "--no-fabric",
        action="store_true",
        help="Disable interconnect fusion (NVLink NVML + IB/eth sysfs)",
    )
    parser.add_argument("--nnodes", type=int, default=None, help="Topology: number of nodes")
    parser.add_argument("--node-rank", type=int, default=None, help="Topology: this node rank")
    parser.add_argument(
        "--nproc-per-node",
        type=int,
        default=None,
        help="Topology: GPUs/processes per node",
    )
    parser.add_argument(
        "--checkpoint-sec",
        type=float,
        default=300.0,
        help="Flush a crash-safety snapshot every N seconds (0 to disable; default: 300)",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    topology = None
    if not args.no_fabric:
        try:
            from scripts.interconnect_collect import infer_topology
        except ImportError:
            from interconnect_collect import infer_topology  # type: ignore
        topology = infer_topology(
            nnodes=args.nnodes,
            node_rank=args.node_rank,
            nproc_per_node=args.nproc_per_node,
        )

    collector = TelemetryCollector(
        gpu_index=args.gpu_index,
        interval_sec=args.interval,
        workload_label=args.label,
        enable_dcgm=not args.no_dcgm,
        enable_fabric=not args.no_fabric,
        topology=topology,
        checkpoint_sec=args.checkpoint_sec,
        checkpoint_path=f"{args.output}.partial" if args.checkpoint_sec > 0 else None,
    )

    print(f"Collecting telemetry for {args.duration}s (interval={args.interval}s)...")
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
