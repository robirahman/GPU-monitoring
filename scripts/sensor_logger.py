#!/usr/bin/env python3
"""SPAR Physical-Sensor Logger — polls the SAIGE testbed sensor snapshot API.

The testbed (node1, our 3x RTX 3090 box) is instrumented by a DAQ host carrying:
  - UltraMic 200K acoustic microphone (200 kHz)
  - 2x PicoScope 2204A oscilloscopes with current clamps on the GPU/MOBO power rails
    (channel_map: GPU0, GPU1, GPU2, MOBO)
  - NVML power readout for node1's GPUs

A "lazy" sensor-streamer samples these only while a client polls
(GET /sensors/api/snapshot) and stops ~12 s after polling stops. This logger
keeps capture warm by polling at a fixed interval and records a timestamped row
per poll, mirroring the NVML TelemetryCollector / ShellyPowerLogger interface
(start/stop/save) so it can run as a sidecar alongside GPU telemetry collection.

Scalar features per poll (joinable to NVML telemetry by run_id + timestamp):
  - mic_rms, mic_dbfs, mic_peak, mic_dominant_hz
  - <rail>_rms_mv, <rail>_peak_mv for each scope rail (GPU0/GPU1/GPU2/MOBO)
  - power_total_w, gpu<i>_power_w, gpu<i>_util_pct
Optionally archives the full downsampled wave arrays to a sidecar .npz.

Standalone usage:
    python sensor_logger.py --duration 60 --output data/sensors/idle.parquet

As a library:
    lg = SensorLogger(run_id="abc", workload_label="idle")
    lg.start(); ...; lg.stop(); lg.save("data/sensors/out.parquet")
"""

import argparse
import logging
import threading
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import requests

log = logging.getLogger("spar.sensors")

DEFAULT_BASE = "https://gpu.felkru.com"
_REPO_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_CREDS = Path("/root/temp_login.txt")  # "username <u> password <p>"


def _load_creds(path=None):
    """Read 'username <u> password <p>' creds, or fall back to env vars."""
    import os
    p = Path(path) if path else _DEFAULT_CREDS
    if p.exists():
        toks = p.read_text().split()
        d = {}
        for i, t in enumerate(toks):
            if t in ("username", "password") and i + 1 < len(toks):
                d[t] = toks[i + 1]
        if "username" in d and "password" in d:
            return d["username"], d["password"]
    u, pw = os.environ.get("SAIGE_USER"), os.environ.get("SAIGE_PASS")
    if u and pw:
        return u, pw
    raise RuntimeError("No SAIGE credentials (temp_login.txt or SAIGE_USER/SAIGE_PASS)")


class SensorClient:
    """Authenticated client for the SAIGE sensor snapshot API (auto re-login)."""

    def __init__(self, base=DEFAULT_BASE, creds_path=None, timeout=8.0):
        self.base = base.rstrip("/")
        self._user, self._pass = _load_creds(creds_path)
        self.timeout = timeout
        self.s = requests.Session()
        self.s.headers["User-Agent"] = "spar-sensor-logger/1.0"
        self.login()

    def login(self):
        self.s.get(f"{self.base}/login", timeout=self.timeout)
        r = self.s.post(f"{self.base}/login",
                        data={"username": self._user, "password": self._pass},
                        timeout=self.timeout, allow_redirects=True)
        # success => redirected away from /login to the dashboard
        if "/login" in r.url:
            raise RuntimeError("SAIGE login failed (still on /login)")
        log.info("SAIGE login OK as %s", self._user)

    def health(self):
        return self.s.get(f"{self.base}/sensors/api/health", timeout=self.timeout).json()

    def snapshot(self):
        """Return one parsed snapshot dict; re-login once on auth bounce."""
        url = f"{self.base}/sensors/api/snapshot"
        r = self.s.get(url, timeout=self.timeout, allow_redirects=False)
        if r.status_code in (301, 302, 401, 403):
            self.login()
            r = self.s.get(url, timeout=self.timeout, allow_redirects=False)
        r.raise_for_status()
        return r.json()


def flatten_snapshot(snap, capture_epoch):
    """Flatten a snapshot into a single scalar-feature row (no wave arrays)."""
    row = {
        "timestamp_utc": datetime.fromtimestamp(capture_epoch, timezone.utc).isoformat(),
        "timestamp_epoch": capture_epoch,
        "sensor_ts": snap.get("ts"),
    }
    m = snap.get("mic", {}) or {}
    row["mic_available"] = bool(m.get("available", False))
    for k in ("rms", "dbfs", "peak", "dominant_hz", "rate_hz"):
        row[f"mic_{k}"] = m.get(k)
    for s in snap.get("scopes", []) or []:
        lbl = s.get("label", f"{s.get('serial','?')}|{s.get('channel','?')}")
        row[f"{lbl}_rms_mv"] = s.get("rms_mv")
        row[f"{lbl}_peak_mv"] = s.get("peak_mv")
        row[f"{lbl}_range_mv"] = s.get("range_mv")
    p = snap.get("power", {}) or {}
    row["power_available"] = bool(p.get("available", False))
    row["power_total_w"] = p.get("total_w")
    for g in p.get("gpus", []) or []:
        i = g.get("index")
        row[f"gpu{i}_power_w"] = g.get("power_w")
        row[f"gpu{i}_power_limit_w"] = g.get("power_limit_w")
        row[f"gpu{i}_util_pct"] = g.get("util_pct")
    return row


class SensorLogger:
    """Polls the snapshot API at ~`interval_sec` and records timestamped rows."""

    def __init__(self, interval_sec=1.0, run_id="", workload_label="unknown",
                 base=DEFAULT_BASE, creds_path=None, store_waves=False):
        self.interval = interval_sec
        self.run_id = run_id
        self.workload_label = workload_label
        self.store_waves = store_waves
        self.client = SensorClient(base=base, creds_path=creds_path)
        self._records = []
        self._waves = []  # list of (epoch, dict-of-arrays) when store_waves
        self._stop = threading.Event()
        self._thread = None
        self._errors = 0

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()
        log.info("Sensor logger started (interval=%.2fs, run_id=%s)", self.interval, self.run_id[:8])

    def stop(self):
        if self._thread is None:
            return
        self._stop.set()
        self._thread.join(timeout=2 * self.interval + 5)
        self._thread = None
        log.info("Sensor logger stopped. %d samples (%d errors).", len(self._records), self._errors)

    def _loop(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                snap = self.client.snapshot()
                row = flatten_snapshot(snap, time.time())
                row["workload_label"] = self.workload_label
                row["run_id"] = self.run_id
                self._records.append(row)
                if self.store_waves:
                    w = {"epoch": row["timestamp_epoch"],
                         "mic": np.asarray((snap.get("mic", {}) or {}).get("wave", []), dtype=np.float32)}
                    for s in snap.get("scopes", []) or []:
                        w[s.get("label", "?")] = np.asarray(s.get("wave", []), dtype=np.float32)
                    self._waves.append(w)
                self._errors = 0
            except Exception as e:
                self._errors += 1
                if self._errors <= 3 or self._errors % 10 == 0:
                    log.warning("snapshot poll failed (%d): %s", self._errors, e)
            dt = self.interval - (time.monotonic() - t0)
            self._stop.wait(max(0.0, dt))

    def to_dataframe(self):
        return pd.DataFrame(self._records)

    def save(self, output_path, fmt="parquet"):
        df = self.to_dataframe()
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        if fmt == "csv":
            df.to_csv(out, index=False)
        else:
            df.to_parquet(out, index=False)
        log.info("Saved %d sensor rows -> %s", len(df), out)
        if self.store_waves and self._waves:
            wpath = out.with_suffix(".waves.npz")
            np.savez_compressed(wpath, **{f"row{i}_{k}": v for i, w in enumerate(self._waves)
                                          for k, v in w.items()})
            log.info("Saved %d wave snapshots -> %s", len(self._waves), wpath)
        return out


def main():
    ap = argparse.ArgumentParser(description="Log SAIGE testbed physical sensors at ~1 Hz.")
    ap.add_argument("--duration", type=float, default=30.0, help="seconds to log")
    ap.add_argument("--interval", type=float, default=1.0, help="poll interval (s)")
    ap.add_argument("--output", type=str, default="data/sensors/sensors.parquet")
    ap.add_argument("--workload-label", type=str, default="manual")
    ap.add_argument("--run-id", type=str, default="")
    ap.add_argument("--base", type=str, default=DEFAULT_BASE)
    ap.add_argument("--creds", type=str, default=None)
    ap.add_argument("--store-waves", action="store_true", help="also archive wave arrays to .npz")
    ap.add_argument("--health", action="store_true", help="print health and exit")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    if args.health:
        import json
        print(json.dumps(SensorClient(base=args.base, creds_path=args.creds).health(), indent=2))
        return

    lg = SensorLogger(interval_sec=args.interval, run_id=args.run_id,
                      workload_label=args.workload_label, base=args.base,
                      creds_path=args.creds, store_waves=args.store_waves)
    lg.start()
    try:
        time.sleep(args.duration)
    finally:
        lg.stop()
        lg.save(args.output)


if __name__ == "__main__":
    main()
