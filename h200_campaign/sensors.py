#!/usr/bin/env python3
"""Slow-rate sensor loggers for the H200 campaign: NVML, BMC (DCMI + SYS_POWER), DCGM.

Every row carries the host wall/monotonic ns immediately before and after the
query (req_*/resp_*), so the analysis knows both when a value was requested and
when it was received. Rows are flushed to parquet parts every 60 s; a meta JSON
records settings, versions, failure counts and the logger's own CPU/RSS/bytes.

  python -m h200_campaign.sensors nvml --out-dir D --prefix P [--interval 1.0]
  python -m h200_campaign.sensors bmc  --out-dir D --prefix P [--interval 0.5]
  python -m h200_campaign.sensors dcgm --out-dir D --prefix P [--interval-ms 100]
"""
from __future__ import annotations

import argparse
import json
import re
import resource
import signal
import subprocess
import time
from pathlib import Path

import numpy as np
import pandas as pd

STOP = False
FLUSH_SEC = 60.0


def _sig(_n, _f):
    global STOP
    STOP = True


class PartWriter:
    def __init__(self, out_dir: Path, prefix: str, kind: str):
        self.out, self.prefix, self.kind = out_dir, prefix, kind
        self.rows, self.part, self.last = [], 0, time.monotonic()
        self.n = 0

    def add(self, row):
        self.rows.append(row)
        self.n += 1
        if time.monotonic() - self.last > FLUSH_SEC:
            self.flush()

    def flush(self):
        if self.rows:
            pd.DataFrame(self.rows).to_parquet(self.out / f"{self.prefix}_{self.kind}_{self.part:04d}.parquet")
            self.part += 1
            self.rows = []
        self.last = time.monotonic()


def _stamp():
    return time.time_ns(), time.monotonic_ns()


def _finish(out, prefix, kind, writer, meta, t0):
    writer.flush()
    ru = resource.getrusage(resource.RUSAGE_SELF)
    rc = resource.getrusage(resource.RUSAGE_CHILDREN)
    meta.update({"rows": writer.n, "elapsed_s": time.monotonic() - t0,
                 "cpu_user_s": ru.ru_utime + rc.ru_utime, "cpu_sys_s": ru.ru_stime + rc.ru_stime,
                 "max_rss_mb": ru.ru_maxrss / 1024,
                 "bytes_written": sum(p.stat().st_size for p in out.glob(f"{prefix}_{kind}_*.parquet"))})
    (out / f"{prefix}_{kind}_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({"kind": kind, "rows": writer.n, "failures": meta.get("failures")}), flush=True)


def _sleep_until(next_t):
    while not STOP:
        dt = next_t - time.monotonic()
        if dt <= 0:
            return
        time.sleep(min(dt, 0.2))


# ---------------------------------------------------------------- NVML
def run_nvml(args, out):
    import pynvml
    pynvml.nvmlInit()
    h = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
    dec = lambda v: v.decode() if isinstance(v, bytes) else v
    fi_instant = getattr(pynvml, "NVML_FI_DEV_POWER_INSTANT", None)
    meta = {"logger": "h200_campaign/sensors.py nvml", "interval_s": args.interval,
            "gpu_name": dec(pynvml.nvmlDeviceGetName(h)), "gpu_uuid": dec(pynvml.nvmlDeviceGetUUID(h)),
            "driver": dec(pynvml.nvmlSystemGetDriverVersion()), "nvml_py": getattr(pynvml, "__version__", ""),
            "pcie_units": "KB/s / 1024 (matches collect_telemetry.py)", "failures": {}}
    for name, fn in (("power_limit_w", lambda: pynvml.nvmlDeviceGetPowerManagementLimit(h) / 1000),
                     ("enforced_power_limit_w", lambda: pynvml.nvmlDeviceGetEnforcedPowerLimit(h) / 1000),
                     ("persistence_mode", lambda: pynvml.nvmlDeviceGetPersistenceMode(h)),
                     ("app_clock_sm_mhz", lambda: pynvml.nvmlDeviceGetApplicationsClock(h, pynvml.NVML_CLOCK_SM)),
                     ("max_clock_sm_mhz", lambda: pynvml.nvmlDeviceGetMaxClockInfo(h, pynvml.NVML_CLOCK_SM)),
                     ("pcie_gen", lambda: pynvml.nvmlDeviceGetCurrPcieLinkGeneration(h)),
                     ("pcie_width", lambda: pynvml.nvmlDeviceGetCurrPcieLinkWidth(h))):
        try:
            meta[name] = fn()
        except pynvml.NVMLError as e:
            meta[name] = f"ERR {e}"
    w = PartWriter(out, args.prefix, "nvml")
    t0 = next_t = time.monotonic()
    while not STOP and time.monotonic() - t0 < args.seconds:
        rw, rm = _stamp()
        row = {"req_wall_ns": rw, "req_mono_ns": rm}
        def g(col, fn):
            try:
                row[col] = fn()
            except pynvml.NVMLError as e:
                row[col] = np.nan
                meta["failures"][col] = meta["failures"].get(col, 0) + 1
        util = None
        try:
            util = pynvml.nvmlDeviceGetUtilizationRates(h)
        except pynvml.NVMLError:
            meta["failures"]["util"] = meta["failures"].get("util", 0) + 1
        row["gpu_utilization_pct"] = util.gpu if util else np.nan
        row["mem_utilization_pct"] = util.memory if util else np.nan
        g("mem_used_mb", lambda: pynvml.nvmlDeviceGetMemoryInfo(h).used // (1024 * 1024))
        g("power_draw_w", lambda: pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0)
        g("temperature_c", lambda: pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU))
        g("sm_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM))
        g("mem_clock_mhz", lambda: pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM))
        g("pcie_tx_mbps", lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024)
        g("pcie_rx_mbps", lambda: pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024)
        # extras, never detector features
        g("clocks_event_reasons", lambda: pynvml.nvmlDeviceGetCurrentClocksEventReasons(h))
        if fi_instant is not None:
            g("power_instant_w", lambda: pynvml.nvmlDeviceGetFieldValues(h, [fi_instant])[0].value.uiVal / 1000.0)
        rw2, rm2 = _stamp()
        row.update(resp_wall_ns=rw2, resp_mono_ns=rm2)
        w.add(row)
        next_t += args.interval
        if next_t < time.monotonic():           # overrun: record it, don't burst-catch-up
            meta["failures"]["schedule_overrun"] = meta["failures"].get("schedule_overrun", 0) + 1
            next_t = time.monotonic() + args.interval
        _sleep_until(next_t)
    pynvml.nvmlShutdown()
    _finish(out, args.prefix, "nvml", w, meta, t0)


# ---------------------------------------------------------------- BMC
RX = {"bmc_power_w": r"Instantaneous power reading:\s+(\d+)", "bmc_power_min_w": r"Minimum during sampling period:\s+(\d+)",
      "bmc_power_max_w": r"Maximum during sampling period:\s+(\d+)", "bmc_power_avg_w": r"Average power reading over sample period:\s+(\d+)"}
TS_RX = re.compile(r"IPMI timestamp:\s+(.+?UTC)")


def run_bmc(args, out):
    meta = {"logger": "h200_campaign/sensors.py bmc", "interval_s": args.interval,
            "sources": ["ipmitool dcmi power reading (in-band KCS)", "ipmitool sensor reading SYS_POWER"],
            "note": "P = whole-node power; DCMI reports a 5 s sampling period", "failures": {}}
    try:
        meta["ipmitool_mc_info"] = subprocess.run(["sudo", "-n", "ipmitool", "mc", "info"], capture_output=True,
                                                  text=True, timeout=10).stdout
    except Exception as e:
        meta["ipmitool_mc_info"] = f"ERR {e}"
    meta["note_sys_power"] = ("SDR SYS_POWER takes ~1.6 s per query, so it is polled back-to-back in its own "
                              "thread into {prefix}_bmcsys_*.parquet and never delays the DCMI loop")

    def query(cmd):
        rw, rm = _stamp()
        try:
            txt, ok = subprocess.run(cmd, capture_output=True, text=True, timeout=5).stdout, True
        except Exception:
            txt, ok = "", False
        rw2, rm2 = _stamp()
        return txt, ok, {"req_wall_ns": rw, "req_mono_ns": rm, "resp_wall_ns": rw2, "resp_mono_ns": rm2}

    def sys_loop(ws):
        while not STOP and time.monotonic() - t0 < args.seconds:
            txt, ok, row = query(["sudo", "-n", "ipmitool", "sensor", "reading", "SYS_POWER"])
            m = re.search(r"SYS_POWER\s*\|\s*([\d.]+)", txt)
            row["bmc_sys_power_w"] = float(m.group(1)) if m else np.nan
            if not (ok and m):
                meta["failures"]["sys"] = meta["failures"].get("sys", 0) + 1
            ws.add(row)
        ws.flush()

    import threading
    w = PartWriter(out, args.prefix, "bmc")
    ws = PartWriter(out, args.prefix, "bmcsys")
    t0 = next_t = time.monotonic()
    th = threading.Thread(target=sys_loop, args=(ws,), daemon=True)
    th.start()
    while not STOP and time.monotonic() - t0 < args.seconds:
        txt, ok, row = query(["sudo", "-n", "ipmitool", "dcmi", "power", "reading"])
        for col, rx in RX.items():
            m = re.search(rx, txt)
            row[col] = float(m.group(1)) if m else np.nan
        m = TS_RX.search(txt)
        row["dcmi_ipmi_timestamp"] = m.group(1) if m else ""
        if not ok or np.isnan(row["bmc_power_w"]):
            meta["failures"]["dcmi"] = meta["failures"].get("dcmi", 0) + 1
        w.add(row)
        next_t += args.interval
        if next_t < time.monotonic():
            meta["failures"]["schedule_overrun"] = meta["failures"].get("schedule_overrun", 0) + 1
            next_t = time.monotonic() + args.interval
        _sleep_until(next_t)
    th.join(timeout=5)
    meta["sys_rows"] = ws.n
    _finish(out, args.prefix, "bmc", w, meta, t0)


# ---------------------------------------------------------------- DCGM
DCGM_FIELDS = [(1001, "dcgm_gr_engine_active"), (1002, "dcgm_sm_active"), (1003, "dcgm_sm_occupancy"),
               (1004, "dcgm_tensor_active"), (1005, "dcgm_dram_active"), (1006, "dcgm_fp64_pipe_active"),
               (1007, "dcgm_fp32_pipe_active"), (1008, "dcgm_fp16_pipe_active"), (1009, "dcgm_pcie_tx_bytes"),
               (1010, "dcgm_pcie_rx_bytes"), (155, "dcgm_power_usage"), (150, "dcgm_gpu_temp"),
               (203, "dcgm_gpu_util"), (204, "dcgm_mem_copy_util")]


def run_dcgm(args, out):
    ids = ",".join(str(f) for f, _ in DCGM_FIELDS)
    names = [n for _, n in DCGM_FIELDS]
    ver = subprocess.run(["dcgmi", "--version"], capture_output=True, text=True).stdout.strip()
    meta = {"logger": "h200_campaign/sensors.py dcgm", "interval_ms": args.interval_ms, "fields": DCGM_FIELDS,
            "dcgmi_version": ver, "timing": "dmon prints no sample time; rows carry host receipt time only",
            "failures": {}}
    cmd = ["dcgmi", "dmon", "-e", ids, "-d", str(args.interval_ms), "-i", str(args.gpu)]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
    w = PartWriter(out, args.prefix, "dcgm")
    t0 = time.monotonic()
    for line in proc.stdout:
        rw, rm = _stamp()
        if STOP or time.monotonic() - t0 > args.seconds:
            break
        parts = line.split()
        if len(parts) < 2 or parts[0] != "GPU":
            continue
        vals = parts[2:]
        if len(vals) < len(names):
            meta["failures"]["short_line"] = meta["failures"].get("short_line", 0) + 1
            continue
        row = {"resp_wall_ns": rw, "resp_mono_ns": rm}
        for n, v in zip(names, vals):
            try:
                row[n] = float(v)
            except ValueError:
                row[n] = np.nan
        w.add(row)
        if time.monotonic() - w.last > FLUSH_SEC:
            w.flush()
    proc.terminate()
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
    _finish(out, args.prefix, "dcgm", w, meta, t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("kind", choices=["nvml", "bmc", "dcgm"])
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seconds", type=float, default=86400.0)
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--interval", type=float, default=None)
    ap.add_argument("--interval-ms", type=int, default=100)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    if args.interval is None:
        args.interval = {"nvml": 1.0, "bmc": 0.5}.get(args.kind, 1.0)
    {"nvml": run_nvml, "bmc": run_bmc, "dcgm": run_dcgm}[args.kind](args, out)


if __name__ == "__main__":
    main()
