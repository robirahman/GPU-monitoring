#!/usr/bin/env python3
"""
Re-run workloads that complete too fast on B200 for 1Hz NVML sampling.
Target: each workload runs >= 15 seconds so we get >= 15 telemetry samples.
"""
import sys, time, uuid, threading, logging
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

try:
    import pynvml
except ImportError:
    sys.exit("pynvml not found: pip install pynvml")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

DEVICE    = "cuda:0"
DATA_DIR  = Path(__file__).parent / "single_gpu_tests" / "data"
DATA_DIR.mkdir(parents=True, exist_ok=True)
MIN_RUNTIME = 15  # seconds — enough for 15+ NVML samples at 1Hz


class TelemetryCollector:
    def __init__(self, gpu_index=0, interval=1.0, label="unknown"):
        pynvml.nvmlInit()
        self._handle   = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
        self._interval = interval
        self._label    = label
        self._records  = []
        self._stop     = threading.Event()
        self._thread   = None
        name = pynvml.nvmlDeviceGetName(self._handle)
        self.gpu_name  = name.decode() if isinstance(name, bytes) else name

    def start(self):
        self._stop.clear()
        self._thread = threading.Thread(target=self._poll, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _poll(self):
        while not self._stop.is_set():
            t0 = time.monotonic()
            try:
                row = self._sample()
            except Exception:
                row = {}
            row["workload_label"]   = self._label
            row["timestamp_epoch"]  = time.time()
            self._records.append(row)
            self._stop.wait(max(0, self._interval - (time.monotonic() - t0)))

    def _sample(self):
        h = self._handle
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
        row  = {
            "gpu_utilization_pct": util.gpu,
            "mem_utilization_pct": util.memory,
            "mem_used_mb":         mem.used // (1024 * 1024),
            "mem_total_mb":        mem.total // (1024 * 1024),
            "power_draw_w":        pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "temperature_c":       pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz":        pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
            "mem_clock_mhz":       pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_MEM),
        }
        try:
            row["pcie_tx_mbps"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_TX_BYTES) / 1024
            row["pcie_rx_mbps"] = pynvml.nvmlDeviceGetPcieThroughput(h, pynvml.NVML_PCIE_UTIL_RX_BYTES) / 1024
        except Exception:
            row["pcie_tx_mbps"] = row["pcie_rx_mbps"] = -1
        return row

    def save(self, path):
        df = pd.DataFrame(self._records)
        df["gpu_name"] = self.gpu_name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path), index=False)
        return df

    def cleanup(self):
        try: pynvml.nvmlShutdown()
        except: pass


def run_with_telemetry(label, fn, pre_s=2, post_s=2):
    col = TelemetryCollector(gpu_index=0, interval=1.0, label=label)
    col.start()
    time.sleep(pre_s)
    try:
        result = fn()
    finally:
        time.sleep(post_s)
        col.stop()
    for old in DATA_DIR.glob(f"{label}_*.parquet"):
        old.unlink()
        log.info("  Removed old: %s", old.name)
    fname = DATA_DIR / f"{label}_{uuid.uuid4().hex[:6]}.parquet"
    df = col.save(str(fname))
    log.info("  Saved %d samples -> %s  (util_mean=%.0f%%  power_mean=%.0fW)",
             len(df), fname.name,
             df["gpu_utilization_pct"].mean(), df["power_draw_w"].mean())
    col.cleanup()
    return df, result


def make_mlp():
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3*32*32, 2048), nn.ReLU(),
        nn.Linear(2048, 2048),    nn.ReLU(),
        nn.Linear(2048, 10),
    )


def workload_mlp():
    """MLP training — run until MIN_RUNTIME reached."""
    log.info("MLP training (target %ds runtime)...", MIN_RUNTIME)
    dev   = torch.device(DEVICE)
    model = make_mlp().to(dev)
    opt   = optim.Adam(model.parameters(), lr=1e-3)
    crit  = nn.CrossEntropyLoss().to(dev)
    batch_size = 1024
    model.train()
    t0 = time.time()
    total_imgs = 0
    while time.time() - t0 < MIN_RUNTIME:
        for _ in range(50):  # inner loop to reduce time.time() overhead
            x = torch.randn(batch_size, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (batch_size,), device=dev)
            opt.zero_grad()
            crit(model(x), y).backward()
            opt.step()
            total_imgs += batch_size
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    log.info("  MLP: %.0f img/s  elapsed=%.1fs", total_imgs / elapsed, elapsed)
    return {"workload": "mlp_training", "elapsed_s": elapsed, "total_imgs": total_imgs}


def workload_mining_proxy():
    """Mining proxy — run until MIN_RUNTIME reached."""
    log.info("Mining proxy (target %ds runtime)...", MIN_RUNTIME)
    dev = torch.device(DEVICE)
    a = torch.randn(256, 256, device=dev)
    b = torch.randn(256, 256, device=dev)
    t0 = time.time()
    n_iters = 0
    while time.time() - t0 < MIN_RUNTIME:
        for _ in range(500):
            for _ in range(8):
                a = torch.matmul(a, b.T).remainder_(1e6)
                b = torch.matmul(b, a.T).remainder_(1e6)
            _ = a.sum() + b.sum()
            n_iters += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    log.info("  Mining: elapsed=%.1fs  iters=%d", elapsed, n_iters)
    return {"workload": "mining_proxy", "elapsed_s": elapsed, "n_iters": n_iters}


def workload_rendering():
    """Rendering proxy — run until MIN_RUNTIME reached."""
    log.info("Rendering proxy (target %ds runtime)...", MIN_RUNTIME)
    dev = torch.device(DEVICE)
    n_samples = 2_000_000  # larger ray batch
    t0 = time.time()
    n_iters = 0
    while time.time() - t0 < MIN_RUNTIME:
        for _ in range(10):
            origins    = torch.rand(n_samples, 3, device=dev) * 2 - 1
            directions = torch.randn(n_samples, 3, device=dev)
            directions = directions / directions.norm(dim=-1, keepdim=True)
            oc   = origins
            a_   = (directions * directions).sum(-1)
            b_   = 2.0 * (oc * directions).sum(-1)
            c_   = (oc * oc).sum(-1) - 1.0
            disc = b_ * b_ - 4 * a_ * c_
            hit  = disc > 0
            col  = torch.where(hit.unsqueeze(-1),
                               torch.ones(n_samples, 3, device=dev) * 0.8,
                               torch.zeros(n_samples, 3, device=dev))
            _ = col.sum()
            n_iters += 1
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    log.info("  Rendering: elapsed=%.1fs  iters=%d", elapsed, n_iters)
    return {"workload": "rendering", "elapsed_s": elapsed, "n_iters": n_iters}


if __name__ == "__main__":
    if not torch.cuda.is_available():
        sys.exit("No CUDA GPU found")

    log.info("=" * 60)
    log.info("Re-running fast workloads on %s (target >= %ds each)",
             torch.cuda.get_device_name(0), MIN_RUNTIME)
    log.info("=" * 60)

    run_with_telemetry("mlp_training",  workload_mlp)
    run_with_telemetry("mining_proxy",  workload_mining_proxy)
    run_with_telemetry("rendering",     workload_rendering)

    log.info("Done. New parquets in %s", DATA_DIR)
