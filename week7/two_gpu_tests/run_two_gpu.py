#!/usr/bin/env python3
"""
Week 7 — Two GPU Tests on 2× NVIDIA B200 (NVLink 5)
Replicates Week 4 multi-GPU tests: NVLink characterization, DDP training,
scaling experiments.

Tests:
  T1   NVLink bandwidth     — unidirectional & bidirectional D2D transfers
  T2   NVLink latency       — ping-pong round-trip timing
  T3   Single vs 2-GPU      — DataParallel throughput comparison (Week 4 T6)
  T4   DDP training         — DistributedDataParallel ResNet training
  T5   Model width sweep    — n_ch 8→512 with DDP (NVLink bottleneck detection)
  T6   Batch size sweep     — TFLOPS vs batch size (occupancy study)
  T7   Binary classifier    — RF classifier on combined telemetry

Results saved to week7/two_gpu_tests/{results,data,plots}/
"""

import sys
import os
import time
import json
import logging
import uuid
import subprocess
import threading
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import pynvml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("week7.two_gpu")

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR    = Path(__file__).parent / "data"
PLOTS_DIR   = Path(__file__).parent / "plots"
for d in [RESULTS_DIR, DATA_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# B200 theoretical ceilings
B200_FP16_TFLOPS = 4500.0
B200_MEM_BW_GBPS = 8000.0
B200_NVLINK5_BW  =  900.0   # GB/s per direction (NVLink 5)
B200_TDP_W       = 1000.0


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryCollector:
    def __init__(self, gpu_index: int = 0, interval: float = 0.5,
                 label: str = "unknown"):
        pynvml.nvmlInit()
        self._handle  = pynvml.nvmlDeviceGetHandleByIndex(gpu_index)
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
            row["workload_label"]  = self._label
            row["timestamp_epoch"] = time.time()
            self._records.append(row)
            self._stop.wait(max(0, self._interval - (time.monotonic() - t0)))

    def _sample(self):
        h = self._handle
        util = pynvml.nvmlDeviceGetUtilizationRates(h)
        mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
        return {
            "gpu_utilization_pct": util.gpu,
            "mem_utilization_pct": util.memory,
            "mem_used_mb":         mem.used // (1024*1024),
            "power_draw_w":        pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
            "temperature_c":       pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
            "sm_clock_mhz":        pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
        }

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(self._records)

    def save(self, path) -> pd.DataFrame:
        df = self.to_dataframe()
        df["gpu_name"] = self.gpu_name
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path), index=False)
        return df

    def cleanup(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def collect_both_gpus(label: str, fn):
    """Run fn() while collecting telemetry on both GPUs."""
    cols = [
        TelemetryCollector(gpu_index=i, interval=0.5, label=label)
        for i in range(2)
    ]
    for c in cols: c.start()
    time.sleep(2)
    try:
        result = fn()
    finally:
        time.sleep(2)
        for c in cols: c.stop()

    dfs = []
    for i, c in enumerate(cols):
        df = c.to_dataframe()
        df["gpu_index"] = i
        fname = DATA_DIR / f"{label}_gpu{i}_{uuid.uuid4().hex[:6]}.parquet"
        c.save(str(fname))
        log.info("  GPU%d: %d samples → %s", i, len(df), fname.name)
        dfs.append(df)
        c.cleanup()
    return result, dfs[0], dfs[1]


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

class ConvBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.seq = nn.Sequential(
            nn.Conv2d(ci, co, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    def forward(self, x): return self.seq(x)

class ResBlock(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.seq = nn.Sequential(ConvBlock(ch, ch), ConvBlock(ch, ch))
        self.relu = nn.ReLU(inplace=True)
    def forward(self, x): return self.relu(self.seq(x) + x)

def make_resnet18_like(num_classes=10):
    return nn.Sequential(
        nn.Conv2d(3, 64, 3, padding=1, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        ResBlock(64), ResBlock(64),
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        ResBlock(128), ResBlock(128),
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        ResBlock(256),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, num_classes))

def make_cnn(n_ch, num_classes=10):
    n = n_ch
    def cb(ci, co): return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    return nn.Sequential(
        cb(3, n), cb(n, 2*n), nn.MaxPool2d(2),
        cb(2*n, 4*n), cb(4*n, 4*n), nn.MaxPool2d(2),
        cb(4*n, 8*n), nn.MaxPool2d(2), cb(8*n, 8*n),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(8*n, num_classes))


# ══════════════════════════════════════════════════════════════════════════════
# T1: NVLink Bandwidth
# ══════════════════════════════════════════════════════════════════════════════

def test_nvlink_bandwidth():
    log.info("=== T1: NVLink Bandwidth ===")
    sizes_mb = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096]
    n_warmup, n_iters = 5, 50
    results = []

    for size_mb in sizes_mb:
        n_elems = (size_mb * 1024 * 1024) // 4
        src = torch.randn(n_elems, dtype=torch.float32, device="cuda:0")
        dst = torch.empty(n_elems, dtype=torch.float32, device="cuda:1")
        for _ in range(n_warmup): dst.copy_(src)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)

        t0 = time.perf_counter()
        for _ in range(n_iters): dst.copy_(src)
        torch.cuda.synchronize(0); torch.cuda.synchronize(1)
        elapsed = time.perf_counter() - t0

        bw = (size_mb * n_iters / 1024) / elapsed
        log.info("  %5d MB → %.2f GB/s", size_mb, bw)
        results.append({"size_mb": size_mb, "bandwidth_gbps": bw,
                         "elapsed_sec": elapsed, "n_iters": n_iters})

    # Bidirectional
    log.info("  Bidirectional 1024 MB...")
    size_mb = 1024
    n_elems = (size_mb * 1024 * 1024) // 4
    a = torch.randn(n_elems, device="cuda:0")
    b = torch.randn(n_elems, device="cuda:1")
    da = torch.empty_like(b, device="cuda:0")
    db = torch.empty_like(a, device="cuda:1")
    s0 = torch.cuda.Stream(device=0)
    s1 = torch.cuda.Stream(device=1)
    for _ in range(n_warmup):
        with torch.cuda.stream(s0): db.copy_(a)
        with torch.cuda.stream(s1): da.copy_(b)
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.cuda.stream(s0): db.copy_(a)
        with torch.cuda.stream(s1): da.copy_(b)
    torch.cuda.synchronize()
    bidir_bw = (size_mb * 2 * n_iters / 1024) / (time.perf_counter() - t0)
    log.info("  Bidirectional total: %.2f GB/s", bidir_bw)
    results.append({"size_mb": size_mb, "bandwidth_gbps": bidir_bw,
                     "n_iters": n_iters, "note": "bidirectional"})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# T2: NVLink Latency
# ══════════════════════════════════════════════════════════════════════════════

def test_nvlink_latency():
    log.info("=== T2: NVLink Latency (ping-pong) ===")
    n_iters = 1000
    sizes   = [1, 4, 16, 64, 256, 1024]
    results = []
    for n in sizes:
        a = torch.ones(n, dtype=torch.float32, device="cuda:0")
        b = torch.empty(n, dtype=torch.float32, device="cuda:1")
        c = torch.empty(n, dtype=torch.float32, device="cuda:0")
        for _ in range(50): b.copy_(a); c.copy_(b)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(n_iters): b.copy_(a); c.copy_(b)
        torch.cuda.synchronize()
        rtt_us = ((time.perf_counter() - t0) / n_iters) * 1e6
        log.info("  %5d floats → RTT = %.2f µs", n, rtt_us)
        results.append({"n_floats": n, "size_bytes": n * 4, "rtt_us": rtt_us})
    return results


# ══════════════════════════════════════════════════════════════════════════════
# T3: Single vs 2-GPU Training Throughput (DataParallel)
# ══════════════════════════════════════════════════════════════════════════════

def measure_training_throughput(device_ids, epochs=2, batch_size=512):
    device = torch.device(f"cuda:{device_ids[0]}")
    model  = make_resnet18_like().to(device)
    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)
    opt    = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit   = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler("cuda")
    n_steps = max(1, 50000 // batch_size)
    model.train()
    t0 = time.time()
    total = 0
    for _ in range(epochs):
        for _ in range(n_steps):
            x = torch.randn(batch_size, 3, 32, 32, device=device)
            y = torch.randint(0, 10, (batch_size,), device=device)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            total += batch_size
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return total / elapsed, elapsed, total


# ══════════════════════════════════════════════════════════════════════════════
# T5: Model Width Sweep (DDP / DataParallel)
# ══════════════════════════════════════════════════════════════════════════════

def test_model_width_sweep_dp():
    """n_ch sweep using DataParallel on 2 GPUs."""
    log.info("=== T5: Model Width Sweep (2-GPU DataParallel) ===")
    dev = torch.device("cuda:0")
    n_ch_values = [8, 16, 32, 64, 128, 256, 512]
    batch_size  = 64
    n_steps_w   = 40
    results = []

    for n_ch in n_ch_values:
        model  = nn.DataParallel(make_cnn(n_ch).to(dev), device_ids=[0, 1])
        opt    = optim.SGD(model.parameters(), lr=0.01)
        crit   = nn.CrossEntropyLoss().to(dev)
        scaler = torch.amp.GradScaler("cuda")
        model.train()

        # Warmup
        for _ in range(5):
            x = torch.randn(batch_size, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (batch_size,), device=dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        torch.cuda.synchronize()

        step_times = []
        for _ in range(n_steps_w):
            x = torch.randn(batch_size, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (batch_size,), device=dev)
            t0 = time.perf_counter()
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)

        # Handle both DataParallel and plain model
        try:
            n_params = sum(p.numel() for p in model.module.parameters())
        except AttributeError:
            n_params = sum(p.numel() for p in model.parameters())
        mean_ms  = np.mean(step_times) * 1e3
        flops    = 2 * n_params * batch_size
        tflops   = flops / np.mean(step_times) / 1e12
        # Estimate NVLink gradient bytes (all-reduce = 2N per param)
        allreduce_bytes = 2 * n_params * 4
        allreduce_s_est = allreduce_bytes / (B200_NVLINK5_BW * 1e9)
        results.append({
            "n_ch": n_ch, "n_params": n_params,
            "mean_step_ms": mean_ms,
            "std_step_ms":  np.std(step_times) * 1e3,
            "tflops": tflops,
            "allreduce_bytes": allreduce_bytes,
            "allreduce_s_theoretical": allreduce_s_est,
        })
        log.info("  n_ch=%3d  params=%8.0fK  step=%.2f ms  tflops=%.2f",
                 n_ch, n_params/1000, mean_ms, tflops)
        del model
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# T6: Batch Size Sweep
# ══════════════════════════════════════════════════════════════════════════════

def test_batch_size_sweep():
    """TFLOPS vs batch size on 2× B200 DataParallel."""
    log.info("=== T6: Batch Size Sweep (2-GPU DataParallel) ===")
    dev = torch.device("cuda:0")
    batch_sizes = [8, 16, 32, 64, 128, 256, 512, 1024]
    n_ch = 128
    n_steps_b = 30
    results = []

    for bs in batch_sizes:
        model  = nn.DataParallel(make_cnn(n_ch).to(dev), device_ids=[0, 1])
        opt    = optim.SGD(model.parameters(), lr=0.01)
        crit   = nn.CrossEntropyLoss().to(dev)
        scaler = torch.amp.GradScaler("cuda")
        model.train()
        try:
            n_params = sum(p.numel() for p in model.module.parameters())
        except AttributeError:
            n_params = sum(p.numel() for p in model.parameters())

        # Warmup
        for _ in range(3):
            x = torch.randn(bs, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (bs,), device=dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"): loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt); scaler.update()
        torch.cuda.synchronize()

        step_times = []
        try:
            for _ in range(n_steps_b):
                x = torch.randn(bs, 3, 32, 32, device=dev)
                y = torch.randint(0, 10, (bs,), device=dev)
                t0 = time.perf_counter()
                opt.zero_grad()
                with torch.amp.autocast("cuda"): loss = crit(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
                torch.cuda.synchronize()
                step_times.append(time.perf_counter() - t0)
            mean_ms = np.mean(step_times) * 1e3
            flops   = 2 * n_params * bs
            tflops  = flops / np.mean(step_times) / 1e12
            results.append({"batch_size": bs, "mean_step_ms": mean_ms, "tflops": tflops})
            log.info("  bs=%4d  step=%.2f ms  tflops=%.2f", bs, mean_ms, tflops)
        except torch.cuda.OutOfMemoryError:
            log.warning("  bs=%d OOM, skipping", bs)
        del model
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(summary, telemetry_dfs, width_results, batch_results):
    # ── NVLink Bandwidth ──────────────────────────────────────────────────────
    bw_df = pd.DataFrame([r for r in summary.get("nvlink_bandwidth", [])
                          if "note" not in r])
    bidir = next((r for r in summary.get("nvlink_bandwidth", [])
                  if "note" in r), None)
    if not bw_df.empty:
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.semilogx(bw_df["size_mb"], bw_df["bandwidth_gbps"],
                    marker="o", lw=2, color="#1565C0", label="Unidirectional (GPU0→GPU1)")
        if bidir:
            ax.axhline(bidir["bandwidth_gbps"] / 2, color="#E53935",
                       ls="--", lw=1.5, label="Bidirectional (per dir)")
        ax.axhline(B200_NVLINK5_BW, color="green", ls=":", lw=1.5,
                   label=f"B200 NVLink5 theoretical ({B200_NVLINK5_BW} GB/s)")
        ax.set_xlabel("Transfer size (MB, log scale)")
        ax.set_ylabel("Bandwidth (GB/s)")
        ax.set_title("NVLink 5 Bandwidth — 2× B200", fontsize=13, fontweight="bold")
        ax.legend(); ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "nvlink_bandwidth.png", dpi=120, bbox_inches="tight")
        plt.close()

    # ── NVLink Latency ────────────────────────────────────────────────────────
    lat_df = pd.DataFrame(summary.get("nvlink_latency", []))
    if not lat_df.empty:
        fig, ax = plt.subplots(figsize=(9, 5))
        ax.semilogx(lat_df["size_bytes"] / 1024, lat_df["rtt_us"],
                    marker="s", lw=2, color="#2E7D32")
        ax.set_xlabel("Transfer size (KB, log scale)")
        ax.set_ylabel("Round-trip latency (µs)")
        ax.set_title("NVLink 5 Ping-Pong Latency — 2× B200", fontsize=13, fontweight="bold")
        ax.grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "nvlink_latency.png", dpi=120, bbox_inches="tight")
        plt.close()

    # ── Throughput comparison ─────────────────────────────────────────────────
    tp = summary.get("throughput_comparison", [])
    if len(tp) == 2:
        fig, ax = plt.subplots(figsize=(7, 5))
        bars = ax.bar(["Single GPU\n(B200 GPU0)", "2× GPU\n(DataParallel, NVLink5)"],
                      [tp[0]["throughput_imgs_per_sec"], tp[1]["throughput_imgs_per_sec"]],
                      color=["#1565C0", "#43A047"], edgecolor="black", linewidth=0.8, width=0.5)
        speedup = tp[1]["throughput_imgs_per_sec"] / tp[0]["throughput_imgs_per_sec"]
        for bar, row in zip(bars, tp):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() * 1.01,
                    f"{row['throughput_imgs_per_sec']:,.0f}", ha="center",
                    fontsize=10, fontweight="bold")
        ax.set_title(f"Training Throughput: 1-GPU vs 2-GPU\n(Speedup: {speedup:.2f}×)",
                     fontsize=12, fontweight="bold")
        ax.set_ylabel("Images / second")
        ax.grid(True, alpha=0.3, axis="y")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "throughput_comparison.png", dpi=120, bbox_inches="tight")
        plt.close()

    # ── Telemetry time-series ─────────────────────────────────────────────────
    for tag, (df0, df1) in telemetry_dfs.items():
        if df0.empty: continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Week 7 2-GPU — {tag} — B200 Telemetry", fontsize=12, fontweight="bold")
        for ax, (col, ylabel) in zip(axes, [
            ("gpu_utilization_pct", "GPU Util (%)"),
            ("power_draw_w",        "Power (W)"),
            ("mem_used_mb",         "Memory (MB)"),
        ]):
            for gpu_idx, (df, c) in enumerate([(df0, "#1565C0"), (df1, "#E53935")]):
                if col not in df.columns: continue
                t = df["timestamp_epoch"].values - df["timestamp_epoch"].values[0]
                ax.plot(t, df[col].values, color=c, lw=0.9, label=f"GPU {gpu_idx}")
            ax.set_ylabel(ylabel); ax.legend(fontsize=8); ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Elapsed (s)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"telemetry_{tag}.png", dpi=100, bbox_inches="tight")
        plt.close()

    # ── Model width scaling ───────────────────────────────────────────────────
    if width_results:
        wd = pd.DataFrame(width_results)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Week 7 — Model Width Sweep (2× B200 DataParallel)", fontweight="bold")
        axes[0].plot(wd["n_ch"], wd["mean_step_ms"], marker="o", color="steelblue")
        axes[0].set(xlabel="n_ch", ylabel="Step time (ms)", title="Step Time vs Width")
        axes[0].grid(True, alpha=0.3)
        axes[1].plot(wd["n_params"] / 1e6, wd["tflops"], marker="s", color="firebrick")
        axes[1].set(xlabel="Params (M)", ylabel="TFLOPS", title="Compute Efficiency")
        axes[1].grid(True, alpha=0.3)
        if "allreduce_bytes" in wd.columns:
            axes[2].loglog(wd["n_params"], wd["allreduce_bytes"] / 1e6,
                           marker="^", color="darkorange")
            axes[2].set(xlabel="Params (log)", ylabel="AllReduce MB (log)",
                        title="NVLink AllReduce Scaling")
            axes[2].grid(True, alpha=0.3, which="both")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_width_scaling.png", dpi=120, bbox_inches="tight")
        plt.close()

    # ── Batch size sweep ──────────────────────────────────────────────────────
    if batch_results:
        bd = pd.DataFrame(batch_results)
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle("Week 7 — Batch Size Sweep (2× B200 DataParallel)", fontweight="bold")
        axes[0].semilogx(bd["batch_size"], bd["tflops"], marker="o", color="steelblue")
        axes[0].axhline(B200_FP16_TFLOPS * 2, color="red", ls="--",
                        label=f"2× B200 theoretical ({B200_FP16_TFLOPS*2:.0f} TFLOPS)")
        axes[0].set(xlabel="Batch size", ylabel="TFLOPS", title="TFLOPS vs Batch Size")
        axes[0].legend(fontsize=8); axes[0].grid(True, alpha=0.3)
        axes[1].semilogx(bd["batch_size"], bd["mean_step_ms"], marker="s", color="firebrick")
        axes[1].set(xlabel="Batch size", ylabel="Step time (ms)", title="Step Time vs Batch Size")
        axes[1].grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "batch_size_sweep.png", dpi=120, bbox_inches="tight")
        plt.close()

    log.info("Plots saved → %s", PLOTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        log.error("Need ≥2 CUDA GPUs. Found %d.", torch.cuda.device_count())
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    log.info("=" * 60)
    log.info("Week 7 — Two GPU Tests")
    log.info("GPUs: %d× %s", torch.cuda.device_count(), gpu_name)
    log.info("=" * 60)

    # Enable peer access
    try:
        torch.cuda.can_device_access_peer(0, 1)
    except Exception:
        pass

    summary = {}
    telemetry_dfs = {}

    # T1: Bandwidth
    bw, df0, df1 = collect_both_gpus("nvlink_bandwidth", test_nvlink_bandwidth)
    summary["nvlink_bandwidth"] = bw
    telemetry_dfs["bw"] = (df0, df1)
    bw_df = pd.DataFrame(bw)
    bw_df.to_csv(RESULTS_DIR / "nvlink_bandwidth.csv", index=False)
    uni_peak = max((r["bandwidth_gbps"] for r in bw if "note" not in r), default=0)
    log.info("Peak unidirectional BW: %.2f GB/s (%.1f%% of theoretical)",
             uni_peak, 100 * uni_peak / B200_NVLINK5_BW)

    # T2: Latency
    lat, df0, df1 = collect_both_gpus("nvlink_latency", test_nvlink_latency)
    summary["nvlink_latency"] = lat
    telemetry_dfs["latency"] = (df0, df1)
    pd.DataFrame(lat).to_csv(RESULTS_DIR / "nvlink_latency.csv", index=False)
    log.info("Min RTT: %.2f µs", min(r["rtt_us"] for r in lat))

    # T3: Single vs 2-GPU throughput
    log.info("=== T3: Training Throughput Comparison ===")

    def single_fn():
        tp, el, imgs = measure_training_throughput([0], epochs=2, batch_size=512)
        log.info("  Single-GPU: %.0f img/s", tp)
        return {"gpus": 1, "throughput_imgs_per_sec": tp, "elapsed_sec": el, "total_images": imgs}

    def dual_fn():
        tp, el, imgs = measure_training_throughput([0, 1], epochs=2, batch_size=1024)
        log.info("  2-GPU DP:   %.0f img/s", tp)
        return {"gpus": 2, "throughput_imgs_per_sec": tp, "elapsed_sec": el, "total_images": imgs}

    single_res, df0, df1 = collect_both_gpus("training_single_gpu", single_fn)
    dual_res,   df0d, df1d = collect_both_gpus("training_dual_gpu_dp", dual_fn)
    summary["throughput_comparison"] = [single_res, dual_res]
    telemetry_dfs["single"] = (df0, df1)
    telemetry_dfs["dual"]   = (df0d, df1d)
    pd.DataFrame([single_res, dual_res]).to_csv(RESULTS_DIR / "throughput_comparison.csv", index=False)
    speedup = dual_res["throughput_imgs_per_sec"] / single_res["throughput_imgs_per_sec"]
    log.info("Speedup 2-GPU vs 1-GPU: %.2f×", speedup)

    # T5: Width sweep
    def width_fn():
        return test_model_width_sweep_dp()

    width_results, df0w, df1w = collect_both_gpus("width_sweep_dp", width_fn)
    summary["model_width_sweep"] = width_results
    telemetry_dfs["width_sweep"] = (df0w, df1w)
    pd.DataFrame(width_results).to_csv(RESULTS_DIR / "model_width_sweep.csv", index=False)

    # T6: Batch sweep
    def batch_fn():
        return test_batch_size_sweep()

    batch_results, df0b, df1b = collect_both_gpus("batch_size_sweep_dp", batch_fn)
    summary["batch_size_sweep"] = batch_results
    telemetry_dfs["batch_sweep"] = (df0b, df1b)
    pd.DataFrame(batch_results).to_csv(RESULTS_DIR / "batch_size_sweep.csv", index=False)

    # Plots
    make_plots(summary, telemetry_dfs, width_results, batch_results)

    # JSON summary
    def _safe(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    with open(RESULTS_DIR / "two_gpu_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=_safe)

    log.info("=" * 60)
    log.info("Two-GPU tests complete.")
    log.info("Results: %s", RESULTS_DIR)
    log.info("Plots:   %s", PLOTS_DIR)
    log.info("=" * 60)

    # Print quick summary
    print(f"\n{'='*60}")
    print(f"NVLink peak BW (unidirectional): {uni_peak:.2f} GB/s")
    bidir = next((r for r in bw if "note" in r), None)
    if bidir:
        print(f"NVLink bidirectional total:      {bidir['bandwidth_gbps']:.2f} GB/s")
    print(f"Min NVLink RTT:                  {min(r['rtt_us'] for r in lat):.2f} µs")
    print(f"Single-GPU throughput:           {single_res['throughput_imgs_per_sec']:,.0f} img/s")
    print(f"2-GPU DataParallel throughput:   {dual_res['throughput_imgs_per_sec']:,.0f} img/s")
    print(f"Speedup:                         {speedup:.2f}×")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
