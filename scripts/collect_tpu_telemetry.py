#!/usr/bin/env python3
"""SPAR TPU Telemetry Collector.

Polls TPU metrics at a configurable interval (default 1 Hz) in a background
thread and writes labeled data to Parquet or CSV.

This is an initial implementation that will be expanded to include
TPU-specific metrics from Google Cloud Monitoring or local TPU APIs.
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

log = logging.getLogger("spar.tpu_telemetry")

class TpuMetadata:
    """Captures static TPU properties."""

    def __init__(self, tpu_index: int = 0):
        self.tpu_index = tpu_index
        # In a real GCP environment, we'd use metadata server or local env vars
        self.tpu_name = os.environ.get("TPU_NAME", "local-tpu")
        self.tpu_type = os.environ.get("TPU_TYPE", "v2-8") # Example default
        self.accelerator_type = "TPU"

    def to_dict(self) -> dict:
        return {
            "tpu_index": self.tpu_index,
            "tpu_name": self.tpu_name,
            "tpu_type": self.tpu_type,
            "accelerator_type": self.accelerator_type,
        }

class TpuTelemetryCollector:
    """Polls TPU metrics at a fixed interval. Runs in a background daemon thread."""

    def __init__(
        self,
        tpu_index: int = 0,
        interval_sec: float = 1.0,
        workload_label: str = "unknown",
        run_id: str | None = None,
    ):
        self.metadata = TpuMetadata(tpu_index)
        self._interval = interval_sec
        self._workload_label = workload_label
        self._run_id = run_id or uuid.uuid4().hex[:12]
        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self):
        """Start the background polling thread."""
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info(
            "TPU Telemetry started (tpu=%s, interval=%.1fs, label=%s, run=%s)",
            self.metadata.tpu_name,
            self._interval,
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
        log.info("TPU Telemetry stopped. %d samples collected.", len(self._records))

    def _poll_loop(self):
        """Main polling loop running in the background thread."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            
            row = self._sample_metrics()
            row["workload_label"] = self._workload_label
            row["run_id"] = self._run_id
            self._records.append(row)

            # Sleep for the remainder of the interval
            elapsed = time.monotonic() - t0
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _sample_metrics(self) -> dict:
        """Sample metrics using tpu_info if available."""
        now = datetime.now(timezone.utc)
        
        row = {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
        }

        # System metrics
        try:
            import psutil
            row["cpu_utilization_pct"] = psutil.cpu_percent()
            row["sys_mem_used_gb"] = psutil.virtual_memory().used / (1024**3)
        except ImportError:
            pass

        # TPU-specific metrics
        try:
            from tpu_info import metrics, device
            chip_type, _ = device.get_local_chips()
            usage = metrics.get_chip_usage(chip_type)
            # usage is typically a list of Usage objects (one per chip)
            if usage and len(usage) > 0:
                u = usage[0]
                row["tpu_utilization_pct"] = getattr(u, "duty_cycle_pct", 0.0)
                row["tpu_mem_used_gb"] = getattr(u, "memory_usage", 0.0) / (1024**3)
                row["tpu_mem_total_gb"] = getattr(u, "total_memory", 0.0) / (1024**3)
            else:
                row["tpu_utilization_pct"] = 0.0
                row["tpu_mem_used_gb"] = 0.0
        except Exception:
            # Fallback to 0.0 if tpu_info fails (e.g. connection refused)
            row["tpu_utilization_pct"] = 0.0
            row["tpu_mem_used_gb"] = 0.0

        return row

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected records to a pandas DataFrame."""
        df = pd.DataFrame(self._records)
        df["tpu_name"] = self.metadata.tpu_name
        df["tpu_type"] = self.metadata.tpu_type
        return df

    def save(self, output_path: str, fmt: str = "parquet"):
        """Save collected data to Parquet or CSV."""
        df = self.to_dataframe()
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            import pyarrow as pa
            import pyarrow.parquet as pq

            table = pa.Table.from_pandas(df)
            meta = table.schema.metadata or {}
            meta[b"tpu_metadata"] = json.dumps(self.metadata.to_dict()).encode("utf-8")
            table = table.replace_schema_metadata(meta)
            pq.write_table(table, str(path))
        else:
            df.to_csv(str(path), index=False)

        log.info("Saved %d rows to %s", len(df), path)
        return str(path)

    def cleanup(self):
        """Release resources."""
        pass

def main():
    parser = argparse.ArgumentParser(description="Collect TPU telemetry.")
    parser.add_argument("--duration", type=int, default=60)
    parser.add_argument("--output", type=str, default="data/tpu_telemetry_test.parquet")
    parser.add_argument("--interval", type=float, default=1.0)
    parser.add_argument("--label", type=str, default="tpu_test")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    collector = TpuTelemetryCollector(
        interval_sec=args.interval,
        workload_label=args.label,
    )

    print(f"Collecting TPU telemetry for {args.duration}s...")
    collector.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        pass
    collector.stop()
    collector.save(args.output)
    collector.cleanup()

if __name__ == "__main__":
    main()
