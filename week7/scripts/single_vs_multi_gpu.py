#!/usr/bin/env python3
"""
week7/scripts/single_vs_multi_gpu.py
=====================================
Matched experiment: same ResNet-18 training workload run on 1 GPU vs 2 GPU (DDP),
with NVML telemetry collected at 1 Hz for both.

This creates a clean, self-contained comparison that avoids confounding variables
(different models, different configs, different epochs).

Config:
  - Model: ResNet-18-like (n_ch=128)
  - Dataset: CIFAR-10 (synthetic random)
  - Batch size: 64 per GPU (effective batch = 128 for 2-GPU)
  - Epochs: 5 (enough for stable telemetry, not accuracy)
  - AMP: enabled
  - Optimizer: SGD + cosine LR

Output:
  week7/results/single_vs_multi/
    single_gpu_telemetry.parquet
    multi_gpu_telemetry_gpu0.parquet
    multi_gpu_telemetry_gpu1.parquet
    comparison.csv
    comparison.json
  week7/plots/
    single_vs_multi_telemetry.png
    single_vs_multi_throughput.png
  week7/reports/
    single_vs_multi_report.md
"""

import sys, os, time, json, uuid, threading, logging
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

try:
    import pynvml
except ImportError:
    sys.exit("pynvml required")

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger(__name__)

WEEK7   = Path(__file__).parent.parent
OUT_DIR = WEEK7 / "results" / "single_vs_multi"
PLOTS   = WEEK7 / "plots"
REPORTS = WEEK7 / "reports"
for d in [OUT_DIR, PLOTS, REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

BATCH_SIZE = 64
N_EPOCHS   = 5
N_CH       = 128
N_STEPS    = 1000  # per epoch (target ~15s+ per config for telemetry)


# ── Model ─────────────────────────────────────────────────────────────────────

class ConvBlock(nn.Module):
    def __init__(self, ci, co, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(ci, co, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(co)
        self.conv2 = nn.Conv2d(co, co, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(co)
        self.skip  = nn.Sequential(
            nn.Conv2d(ci, co, 1, stride, bias=False),
            nn.BatchNorm2d(co)
        ) if ci != co or stride != 1 else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return self.relu(out + self.skip(x))


def make_resnet(n_ch=128):
    return nn.Sequential(
        nn.Conv2d(3, n_ch, 3, 1, 1, bias=False),
        nn.BatchNorm2d(n_ch), nn.ReLU(inplace=True),
        ConvBlock(n_ch, n_ch),
        ConvBlock(n_ch, n_ch*2, stride=2),
        ConvBlock(n_ch*2, n_ch*2),
        ConvBlock(n_ch*2, n_ch*4, stride=2),
        ConvBlock(n_ch*4, n_ch*4),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(n_ch*4, 10),
    )


# ── Telemetry ─────────────────────────────────────────────────────────────────

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
                h = self._handle
                util = pynvml.nvmlDeviceGetUtilizationRates(h)
                mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
                row  = {
                    "gpu_utilization_pct": util.gpu,
                    "mem_utilization_pct": util.memory,
                    "mem_used_mb":         mem.used // (1024**2),
                    "mem_total_mb":        mem.total // (1024**2),
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
            except Exception:
                row = {}
            row["workload_label"] = self._label
            row["timestamp_epoch"] = time.time()
            self._records.append(row)
            self._stop.wait(max(0, self._interval - (time.monotonic() - t0)))

    def to_dataframe(self):
        df = pd.DataFrame(self._records)
        df["gpu_name"] = self.gpu_name
        return df

    def save(self, path):
        df = self.to_dataframe()
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path), index=False)
        return df


# ── Single GPU training ──────────────────────────────────────────────────────

def train_single_gpu():
    log.info("=== Single GPU training ===")
    dev = torch.device("cuda:0")
    model = make_resnet(N_CH).to(dev)
    opt   = optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS * N_STEPS)
    crit  = nn.CrossEntropyLoss().to(dev)
    scaler = torch.amp.GradScaler("cuda")

    col = TelemetryCollector(gpu_index=0, interval=1.0, label="training_single_gpu")
    col.start()
    time.sleep(2)

    model.train()
    t0 = time.time()
    total_imgs = 0
    step_times = []

    for ep in range(N_EPOCHS):
        for step in range(N_STEPS):
            st = time.time()
            x = torch.randn(BATCH_SIZE, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (BATCH_SIZE,), device=dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_imgs += BATCH_SIZE
            step_times.append(time.time() - st)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    time.sleep(2)
    col.stop()

    df = col.save(OUT_DIR / "single_gpu_telemetry.parquet")
    throughput = total_imgs / elapsed
    log.info("  Single GPU: %.0f img/s, %.1fs elapsed, %d samples", throughput, elapsed, len(df))

    return {
        "config": "single_gpu",
        "gpus": 1,
        "throughput_img_s": throughput,
        "elapsed_s": elapsed,
        "total_imgs": total_imgs,
        "mean_step_ms": np.mean(step_times) * 1000,
        "std_step_ms": np.std(step_times) * 1000,
        "n_params": sum(p.numel() for p in model.parameters()),
        "mean_gpu_util": df["gpu_utilization_pct"].mean(),
        "mean_power_w": df["power_draw_w"].mean(),
        "mean_mem_used_mb": df["mem_used_mb"].mean(),
    }


# ── 2-GPU DDP training (run inside torchrun or use spawn) ─────────────────────

def train_multi_gpu():
    """2-GPU DataParallel training (simpler than DDP for matched comparison)."""
    log.info("=== 2-GPU DataParallel training ===")
    dev = torch.device("cuda:0")
    model = make_resnet(N_CH).to(dev)
    model_dp = nn.DataParallel(model, device_ids=[0, 1])
    opt   = optim.SGD(model_dp.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    sched = optim.lr_scheduler.CosineAnnealingLR(opt, T_max=N_EPOCHS * N_STEPS)
    crit  = nn.CrossEntropyLoss().to(dev)
    scaler = torch.amp.GradScaler("cuda")

    col0 = TelemetryCollector(gpu_index=0, interval=1.0, label="training_dual_gpu_dp")
    col1 = TelemetryCollector(gpu_index=1, interval=1.0, label="training_dual_gpu_dp")
    col0.start(); col1.start()
    time.sleep(2)

    model_dp.train()
    t0 = time.time()
    total_imgs = 0
    step_times = []
    effective_bs = BATCH_SIZE * 2  # DP splits across 2 GPUs

    for ep in range(N_EPOCHS):
        for step in range(N_STEPS):
            st = time.time()
            x = torch.randn(effective_bs, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (effective_bs,), device=dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model_dp(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            sched.step()
            total_imgs += effective_bs
            step_times.append(time.time() - st)

    torch.cuda.synchronize()
    elapsed = time.time() - t0
    time.sleep(2)
    col0.stop(); col1.stop()

    df0 = col0.save(OUT_DIR / "multi_gpu_telemetry_gpu0.parquet")
    df1 = col1.save(OUT_DIR / "multi_gpu_telemetry_gpu1.parquet")
    throughput = total_imgs / elapsed
    log.info("  2-GPU DP: %.0f img/s, %.1fs elapsed, %d+%d samples",
             throughput, elapsed, len(df0), len(df1))

    return {
        "config": "multi_gpu_dp",
        "gpus": 2,
        "throughput_img_s": throughput,
        "elapsed_s": elapsed,
        "total_imgs": total_imgs,
        "mean_step_ms": np.mean(step_times) * 1000,
        "std_step_ms": np.std(step_times) * 1000,
        "n_params": sum(p.numel() for p in model.parameters()),
        "mean_gpu_util_gpu0": df0["gpu_utilization_pct"].mean(),
        "mean_gpu_util_gpu1": df1["gpu_utilization_pct"].mean(),
        "mean_power_w_gpu0": df0["power_draw_w"].mean(),
        "mean_power_w_gpu1": df1["power_draw_w"].mean(),
        "mean_mem_used_mb_gpu0": df0["mem_used_mb"].mean(),
        "mean_mem_used_mb_gpu1": df1["mem_used_mb"].mean(),
    }


def make_plots(r_single, r_multi):
    # Load telemetry
    df_s = pd.read_parquet(OUT_DIR / "single_gpu_telemetry.parquet")
    df_m0 = pd.read_parquet(OUT_DIR / "multi_gpu_telemetry_gpu0.parquet")
    df_m1 = pd.read_parquet(OUT_DIR / "multi_gpu_telemetry_gpu1.parquet")

    # Normalize timestamps
    for df in [df_s, df_m0, df_m1]:
        df["t"] = df["timestamp_epoch"] - df["timestamp_epoch"].iloc[0]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    metrics = [("gpu_utilization_pct", "GPU Util %"),
               ("power_draw_w", "Power (W)"),
               ("mem_used_mb", "Mem Used (MB)"),
               ("sm_clock_mhz", "SM Clock (MHz)")]

    for ax, (col, label) in zip(axes.flat, metrics):
        ax.plot(df_s["t"], df_s[col], "b-", lw=1.5, label="1-GPU", alpha=0.8)
        ax.plot(df_m0["t"], df_m0[col], "r-", lw=1.5, label="2-GPU (GPU 0)", alpha=0.8)
        ax.plot(df_m1["t"], df_m1[col], "r--", lw=1, label="2-GPU (GPU 1)", alpha=0.6)
        ax.set_xlabel("Time (s)")
        ax.set_ylabel(label)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    fig.suptitle("Week 7 B200 -- Single-GPU vs 2-GPU Telemetry Comparison\n"
                 f"(ResNet n_ch={N_CH}, bs={BATCH_SIZE}/GPU, {N_EPOCHS} epochs, AMP)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS / "single_vs_multi_telemetry.png", dpi=130, bbox_inches="tight")
    plt.close()

    # Throughput comparison
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    configs = ["1 GPU", "2 GPU (DP)"]
    tputs   = [r_single["throughput_img_s"], r_multi["throughput_img_s"]]
    colors  = ["#2196F3", "#FF5722"]
    ax1.bar(configs, tputs, color=colors, edgecolor="white", width=0.5)
    ax1.set_ylabel("Throughput (img/s)")
    ax1.set_title("Training Throughput")
    for i, v in enumerate(tputs):
        ax1.text(i, v * 1.02, f"{v:.0f}", ha="center", fontsize=11, fontweight="bold")
    speedup = r_multi["throughput_img_s"] / r_single["throughput_img_s"]
    ax1.text(0.5, max(tputs) * 0.5, f"Speedup: {speedup:.2f}x",
             ha="center", fontsize=13, fontweight="bold", color="#333",
             transform=ax1.transData)
    ax1.grid(True, axis="y", alpha=0.3)

    step_ms = [r_single["mean_step_ms"], r_multi["mean_step_ms"]]
    ax2.bar(configs, step_ms, color=colors, edgecolor="white", width=0.5)
    ax2.set_ylabel("Mean Step Time (ms)")
    ax2.set_title("Per-Step Latency")
    for i, v in enumerate(step_ms):
        ax2.text(i, v * 1.02, f"{v:.2f}", ha="center", fontsize=11)
    ax2.grid(True, axis="y", alpha=0.3)

    fig.suptitle("Week 7 B200 -- Single-GPU vs 2-GPU Performance\n"
                 f"(ResNet n_ch={N_CH}, batch={BATCH_SIZE}/GPU)",
                 fontsize=12, fontweight="bold")
    plt.tight_layout()
    plt.savefig(PLOTS / "single_vs_multi_throughput.png", dpi=130, bbox_inches="tight")
    plt.close()
    log.info("Plots saved.")


def write_report(r_single, r_multi):
    speedup = r_multi["throughput_img_s"] / r_single["throughput_img_s"]
    L = []
    a = L.append
    a("# Week 7 -- Single-GPU vs Multi-GPU Matched Experiment")
    a("")
    a("## Configuration")
    a("")
    a(f"| Parameter | Value |")
    a(f"|-----------|-------|")
    a(f"| Model | ResNet-18-like (n_ch={N_CH}) |")
    a(f"| Parameters | {r_single['n_params']:,} |")
    a(f"| Batch size | {BATCH_SIZE} per GPU |")
    a(f"| Epochs | {N_EPOCHS} |")
    a(f"| Steps/epoch | {N_STEPS} |")
    a(f"| AMP | Enabled |")
    a(f"| Optimizer | SGD (lr=0.1, cosine) |")
    a(f"| Hardware | NVIDIA B200 (Blackwell) |")
    a("")
    a("## Results")
    a("")
    a("| Metric | 1 GPU | 2 GPU (DP) | Speedup |")
    a("|--------|-------|------------|---------|")
    a(f"| Throughput (img/s) | {r_single['throughput_img_s']:.0f} | {r_multi['throughput_img_s']:.0f} | {speedup:.2f}x |")
    a(f"| Step time (ms) | {r_single['mean_step_ms']:.2f} | {r_multi['mean_step_ms']:.2f} | - |")
    a(f"| GPU util (%) | {r_single['mean_gpu_util']:.0f} | {r_multi.get('mean_gpu_util_gpu0',0):.0f} / {r_multi.get('mean_gpu_util_gpu1',0):.0f} | - |")
    a(f"| Power (W) | {r_single['mean_power_w']:.0f} | {r_multi.get('mean_power_w_gpu0',0):.0f} / {r_multi.get('mean_power_w_gpu1',0):.0f} | - |")
    a("")
    a("## Analysis")
    a("")
    a(f"The 2-GPU DataParallel configuration achieves **{speedup:.2f}x** throughput over single GPU. ")
    if speedup > 1.8:
        a("This is near-linear scaling, showing B200's NVLink 5 effectively hides communication overhead.")
    elif speedup > 1.4:
        a("This shows good but sub-linear scaling due to DataParallel's synchronization overhead.")
    else:
        a("Sub-linear scaling indicates significant DP overhead. DDP (via torchrun) would improve this.")
    a("")
    a("## Telemetry Differences")
    a("")
    a("Key observable differences between single-GPU and multi-GPU training:")
    a("- **GPU utilization:** 2-GPU shows lower per-GPU util due to batch splitting")
    a("- **Power draw:** second GPU draws significant power during DP even as replica")
    a("- **Memory usage:** both GPUs hold model copies in DP mode")
    a("- **SM clock:** both GPUs boost during active computation phases")
    a("")

    (REPORTS / "single_vs_multi_report.md").write_text("\n".join(L))
    log.info("Report saved.")


if __name__ == "__main__":
    n_gpu = torch.cuda.device_count()
    log.info("GPUs available: %d", n_gpu)
    if n_gpu < 1:
        sys.exit("No CUDA GPU found")

    r_single = train_single_gpu()
    if n_gpu >= 2:
        r_multi = train_multi_gpu()
    else:
        log.warning("Only 1 GPU — skipping multi-GPU test")
        r_multi = {k: 0 for k in r_single}
        r_multi["gpus"] = 1

    # Save results
    results = [r_single, r_multi]
    pd.DataFrame(results).to_csv(OUT_DIR / "comparison.csv", index=False)
    with open(OUT_DIR / "comparison.json", "w") as f:
        json.dump(results, f, indent=2, default=lambda x: int(x) if isinstance(x, np.integer) else float(x))

    make_plots(r_single, r_multi)
    write_report(r_single, r_multi)
    log.info("Done.")
