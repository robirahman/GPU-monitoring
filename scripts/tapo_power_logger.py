#!/usr/bin/env python3
"""SPAR Tier 5 Wall-Power Logger — polls TP-Link Tapo P115 smart plug.

Collects wall-power measurements at ~1 Hz from a Tapo P115 energy-monitoring
smart plug using the `tapo` library (pip install tapo).

Mirrors the ShellyPowerLogger / TelemetryCollector interface (start/stop/save)
so it can run as a sidecar alongside GPU telemetry collection.

Standalone usage:
    python tapo_power_logger.py --tapo-ip 192.168.0.154 --duration 60 --output data/power.parquet

As a library (used by run_workload_with_power.py):
    logger = TapoPowerLogger(tapo_ip="192.168.0.154",
                             tapo_email="user@example.com",
                             tapo_password="secret")
    logger.start()
    # ... run workload ...
    logger.stop()
    logger.save("data/power.parquet")
"""

import argparse
import asyncio
import logging
import os
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

log = logging.getLogger("spar.tapo")

# Default credentials file location (repo root / tplink.env)
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ENV = _REPO_ROOT / "tplink.env"


def _load_env(env_path: Path | str | None = None) -> dict[str, str]:
    """Load key: value pairs from the tplink.env file."""
    path = Path(env_path) if env_path else _DEFAULT_ENV
    if not path.exists():
        return {}
    result = {}
    for line in path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if ":" in line:
            key, _, val = line.partition(":")
            result[key.strip()] = val.strip()
    return result


class TapoPowerLogger:
    """Polls a TP-Link Tapo P115 at ~1 Hz and records wall power data."""

    def __init__(
        self,
        tapo_ip: str,
        tapo_email: str,
        tapo_password: str,
        interval_sec: float = 1.0,
        run_id: str = "",
        workload_label: str = "unknown",
    ):
        self._ip = tapo_ip
        self._email = tapo_email
        self._password = tapo_password
        self._interval = interval_sec
        self._run_id = run_id
        self._workload_label = workload_label
        self._records: list[dict] = []
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._consecutive_errors = 0
        self._device = None

    async def _connect(self):
        """Authenticate with TP-Link cloud and get a local device handle."""
        from tapo import ApiClient

        client = ApiClient(self._email, self._password)
        self._device = await client.p115(self._ip)
        log.info("Connected to Tapo P115 at %s", self._ip)

    async def _async_sample(self) -> dict | None:
        """Poll the Tapo P115 once and return a measurement dict.

        Automatically reconnects on session timeout (KLAP sessions are short-lived).
        """
        now = datetime.now(timezone.utc)
        try:
            usage = await self._device.get_energy_usage()
            return {
                "timestamp_utc": now.isoformat(),
                "timestamp_epoch": now.timestamp(),
                "wall_power_w": usage.current_power / 1000.0,  # mW -> W
                "voltage_v": float("nan"),  # P115 API doesn't expose voltage
                "current_a": float("nan"),  # P115 API doesn't expose current
                "cumulative_wh": float(getattr(usage, "today_energy", 0)),
            }
        except Exception as e:
            err_str = str(e)
            if "SessionTimeout" in err_str or "403" in err_str:
                log.info("Tapo session expired, reconnecting...")
                try:
                    await self._connect()
                    usage = await self._device.get_energy_usage()
                    return {
                        "timestamp_utc": now.isoformat(),
                        "timestamp_epoch": now.timestamp(),
                        "wall_power_w": usage.current_power / 1000.0,
                        "voltage_v": float("nan"),
                        "current_a": float("nan"),
                        "cumulative_wh": float(getattr(usage, "today_energy", 0)),
                    }
                except Exception as e2:
                    log.warning("Tapo reconnect failed: %s", e2)
                    return None
            log.warning("Tapo poll error: %s", e)
            return None

    def start(self):
        """Start background polling thread."""
        self._stop_event.clear()
        self._consecutive_errors = 0
        self._thread = threading.Thread(target=self._poll_loop, daemon=True)
        self._thread.start()
        log.info("Tapo power logger started (ip=%s, interval=%.1fs)", self._ip, self._interval)

    def stop(self):
        """Stop the polling thread and wait for it to finish."""
        if self._thread is None:
            return
        self._stop_event.set()
        self._thread.join(timeout=5.0)
        self._thread = None
        log.info("Tapo power logger stopped. %d samples collected.", len(self._records))

    def _poll_loop(self):
        """Main polling loop running in a background thread with its own event loop."""
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

        try:
            loop.run_until_complete(self._connect())
        except Exception as e:
            log.error("Failed to connect to Tapo P115: %s", e)
            return

        while not self._stop_event.is_set():
            t0 = time.monotonic()
            row = loop.run_until_complete(self._async_sample())
            if row is not None:
                row["workload_label"] = self._workload_label
                row["run_id"] = self._run_id
                self._records.append(row)
                self._consecutive_errors = 0
            else:
                self._consecutive_errors += 1
                if self._consecutive_errors >= 10:
                    log.error("10 consecutive Tapo poll failures; stopping logger.")
                    break

            elapsed = time.monotonic() - t0
            sleep_time = max(0, self._interval - elapsed)
            if sleep_time > 0:
                self._stop_event.wait(sleep_time)

        loop.close()

    def to_dataframe(self) -> pd.DataFrame:
        """Convert collected records to a pandas DataFrame."""
        return pd.DataFrame(self._records)

    def save(self, output_path: str, fmt: str = "parquet"):
        """Save collected data to Parquet or CSV."""
        df = self.to_dataframe()
        if df.empty:
            log.warning("No Tapo data to save (0 records).")
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
        """Test that the Tapo P115 is reachable. Returns True on success."""
        loop = asyncio.new_event_loop()
        try:
            loop.run_until_complete(self._connect())
            row = loop.run_until_complete(self._async_sample())
        except Exception as e:
            log.warning("Tapo connection test failed: %s", e)
            return False
        finally:
            loop.close()

        if row is not None:
            log.info("Tapo connection OK: %.1f W", row["wall_power_w"])
            return True
        return False


def main():
    parser = argparse.ArgumentParser(description="Tapo P115 wall-power logger")
    parser.add_argument("--tapo-ip", type=str, default=None, help="Tapo plug IP (default: from tplink.env)")
    parser.add_argument("--tapo-email", type=str, default=None, help="TP-Link account email (default: from tplink.env)")
    parser.add_argument("--tapo-password", type=str, default=None, help="TP-Link account password (default: from tplink.env)")
    parser.add_argument("--env-file", type=str, default=None, help="Path to tplink.env credentials file")
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

    # Load credentials from env file, CLI args override
    env = _load_env(args.env_file)
    tapo_ip = args.tapo_ip or env.get("local_ip")
    tapo_email = args.tapo_email or env.get("username")
    tapo_password = args.tapo_password or env.get("password")

    if not all([tapo_ip, tapo_email, tapo_password]):
        print("ERROR: Tapo IP, email, and password are required.", file=__import__("sys").stderr)
        print("       Provide via CLI args or tplink.env file.", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    logger = TapoPowerLogger(
        tapo_ip=tapo_ip,
        tapo_email=tapo_email,
        tapo_password=tapo_password,
        interval_sec=args.interval,
        workload_label=args.label,
    )

    print(f"Testing connection to Tapo P115 at {tapo_ip}...")
    if not logger.test_connection():
        print("ERROR: Cannot reach Tapo P115. Check IP, credentials, and network.", file=__import__("sys").stderr)
        __import__("sys").exit(1)

    print(f"Recording wall power for {args.duration}s...")
    logger.start()
    try:
        import time
        time.sleep(args.duration)
    except KeyboardInterrupt:
        print("\nInterrupted.")
    logger.stop()
    logger.save(args.output, fmt=args.format)
    print(f"Saved {logger.sample_count} samples to {args.output}")


if __name__ == "__main__":
    main()
