#!/usr/bin/env python3
"""SPAR Tier 5 Wall-Power Logger — polls Shelly Plus Plug US via REST API.

Collects wall-power measurements at ~1 Hz from a Shelly Plus Plug US smart plug.
Mirrors the TelemetryCollector interface (start/stop/save) so it can run as a
sidecar alongside GPU telemetry collection.

Standalone usage:
    python shelly_power_logger.py --shelly-ip 192.168.1.100 --duration 60 --output data/power.parquet

As a library (used by run_workload_with_power.py):
    logger = ShellyPowerLogger(shelly_ip="192.168.1.100")
    logger.start()
    # ... run workload ...
    logger.stop()
    logger.save("data/power.parquet")
"""

import argparse
import json
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path
from urllib.request import urlopen
from urllib.error import URLError

import pandas as pd

log = logging.getLogger("spar.shelly")


class ShellyPowerLogger:
    """Polls a Shelly Plus Plug US at ~1 Hz and records wall power data."""

    def __init__(
        self,
        shelly_ip: str,
        interval_sec: float = 1.0,
        run_id: str = "",
        workload_label: str = "unknown",
        timeout_sec: float = 2.0,
    ):
        self._url = f"http://{shelly_ip}/rpc/Switch.GetStatus?id=0"
        self._interval = interval_sec
        self._timeout = timeout_sec
        self._run_id = run_id
        self._workload_label = workload_label
        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_errors = 0

    def start(self):
        """Start background polling thread."""
        self._stop_event.clear()
        self._consecutive_errors = 0
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("Shelly power logger started (url=%s, interval=%.1fs)", self._url, self._interval)

    def stop(self):
        """Stop the polling thread and wait for it to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        log.info("Shelly power logger stopped. %d samples collected.", len(self._records))

    def _poll_loop(self):
        """Main polling loop running in a background thread."""
        while not self._stop_event.is_set():
            t0 = time.monotonic()
            row = self._sample()
            if row is not None:
                row["workload_label"] = self._workload_label
                row["run_id"] = self._run_id
                self._records.append(row)
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1
                if self._consecutive_errors >= 10:
                    log.error("10 consecutive Shelly poll failures; stopping logger.")
                    break

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

    def _sample(self) -> dict | None:
        """Poll the Shelly REST API once and return a measurement dict."""
        now = datetime.now(timezone.utc)
        try:
            with urlopen(self._url, timeout=self._timeout) as resp:
                data = json.loads(resp.read().decode("utf-8"))
        except (URLError, OSError, json.JSONDecodeError) as e:
            log.warning("Shelly poll error: %s", e)
            return None

        row = {
            "timestamp_utc": now.isoformat(),
            "timestamp_epoch": now.timestamp(),
            "wall_power_w": data.get("apower", float("nan")),
            "voltage_v": data.get("voltage", float("nan")),
            "current_a": data.get("current", float("nan")),
        }

        # Energy accumulator (watt-hours since plug powered on)
        aenergy = data.get("aenergy", {})
        row["cumulative_wh"] = aenergy.get("total", float("nan"))

        # Plug temperature
        temp = data.get("temperature", {})
        row["plug_temp_c"] = temp.get("tC", float("nan"))

        return row

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected records to a pandas DataFrame."""
        return pd.DataFrame(self._records)

    def save(self, output_path: str, fmt: str = "parquet"):
        """Save collected data to Parquet or CSV."""
        df = self.to_dataframe()
        if df.empty:
            log.warning("No Shelly data to save (0 records).")
            return None

        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if fmt == "parquet":
            df.to_parquet(str(path), index=False)
        else:
            df.to_csv(str(path), index=False)

        log.info("Saved %d wall-power rows to %s", len(df), path)
        return str(path)

    @property
    def sample_count(self) -> int:
        return len(self._records)

    def test_connection(self) -> bool:
        """Test that the Shelly plug is reachable. Returns True on success."""
        row = self._sample()
        if row is not None:
            log.info(
                "Shelly connection OK: %.1f W, %.1f V, %.3f A",
                row["wall_power_w"], row["voltage_v"], row["current_a"],
            )
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="Shelly Plus Plug US wall-power logger")
    parser.add_argument("--shelly-ip", type=str, required=True, help="Shelly plug IP address")
    parser.add_argument("--duration", type=int, default=60, help="Recording duration in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Polling interval in seconds")
    parser.add_argument("--output", type=str, default="data/wall_power.parquet", help="Output file path")
    parser.add_argument("--format", choices=["parquet", "csv"], default="parquet")
    parser.add_argument("--label", type=str, default="manual", help="Workload label")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    )

    logger = ShellyPowerLogger(
        shelly_ip=args.shelly_ip,
        interval_sec=args.interval,
        workload_label=args.label,
    )

    print(f"Testing connection to Shelly at {args.shelly_ip}...")
    if not logger.test_connection():
        print("ERROR: Cannot reach Shelly plug. Check IP and network.", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    print(f"Recording wall power for {args.duration}s...")
    logger.start()
    try:
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    logger.stop()
    logger.save(args.output, fmt=args.format)
    print(f"Saved {logger.sample_count} samples to {args.output}")


if __name__ == "__main__":
    main()
