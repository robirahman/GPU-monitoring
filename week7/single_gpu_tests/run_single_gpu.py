#!/usr/bin/env python3
"""
Week 7 — Single GPU Tests on NVIDIA B200
Replicates workloads from Weeks 2, 3, 4, 5 on local B200 hardware.

Workloads profiled:
  W1   idle          — GPU idle baseline
  W2   resnet_fp32   — ResNet-18-like training (FP32)
  W3   resnet_amp    — ResNet-18-like training (AMP / FP16)
  W4   mlp_training  — 3-layer MLP training
  W5   inference     — ResNet-50-like inference (large batch)
  W6   cufft_hpc     — FFT HPC benchmark
  W7   nbody_sim     — N-body gravitational simulation
  W8   mining_proxy  — Crypto mining workload proxy
  W9   rendering     — Monte Carlo rendering proxy
  W10  model_scaling — Model width sweep (n_ch 8→512) as in Week 4

Telemetry collected at 1 Hz for each workload.
Results saved to week7/single_gpu_tests/{results,data,plots}/
"""

import sys
import os
import time
import json
import logging
import uuid
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim

import pynvml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("week7.single_gpu")

# ── Paths ──────────────────────────────────────────────────────────────────────
WEEK7       = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR    = Path(__file__).parent / "data"
PLOTS_DIR   = Path(__file__).parent / "plots"
for d in [RESULTS_DIR, DATA_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Hardware constants (B200) ──────────────────────────────────────────────────
B200_FP16_TFLOPS  = 4500.0   # peak FP16 TF
B200_FP32_TFLOPS  =  140.0   # peak FP32 TF
B200_MEM_BW_GBPS  = 8000.0   # HBM3e bandwidth
B200_TDP_W        = 1000.0
DEVICE            = "cuda:0"


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY COLLECTOR (inline, no external dep)
# ══════════════════════════════════════════════════════════════════════════════
import threading

class TelemetryCollector:
    """Lightweight pynvml-based GPU telemetry poller."""
    def __init__(self, gpu_index: int = 0, interval: float = 1.0,
                 label: str = "unknown"):
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
            row["workload_label"] = self._label
            row["timestamp_epoch"] = time.time()
            self._records.append(row)
            elapsed = time.monotonic() - t0
            self._stop.wait(max(0, self._interval - elapsed))

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

    def to_dataframe(self):
        return pd.DataFrame(self._records)

    def save(self, path):
        df = self.to_dataframe()
        df["gpu_name"] = self.gpu_name
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(str(path), index=False)
        return df

    def cleanup(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def run_with_telemetry(label: str, fn, pre_s=3, post_s=3) -> pd.DataFrame:
    """Run fn() while collecting telemetry; return df."""
    col = TelemetryCollector(gpu_index=0, interval=1.0, label=label)
    col.start()
    time.sleep(pre_s)
    try:
        result = fn()
    finally:
        time.sleep(post_s)
        col.stop()
    fname = DATA_DIR / f"{label}_{uuid.uuid4().hex[:6]}.parquet"
    df = col.save(str(fname))
    log.info("  Saved %d samples → %s", len(df), fname.name)
    col.cleanup()
    return df, result


# ══════════════════════════════════════════════════════════════════════════════
# MODEL DEFINITIONS
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

def make_resnet50_like(num_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3, 64, 7, stride=2, padding=3, bias=False),
        nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        nn.MaxPool2d(3, stride=2, padding=1),
        ResBlock(64), ResBlock(64), ResBlock(64),
        nn.Conv2d(64, 128, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(128), nn.ReLU(inplace=True),
        ResBlock(128), ResBlock(128), ResBlock(128),
        nn.Conv2d(128, 256, 3, stride=2, padding=1, bias=False),
        nn.BatchNorm2d(256), nn.ReLU(inplace=True),
        ResBlock(256), ResBlock(256), ResBlock(256),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(256, num_classes))

def make_mlp(hidden=512, num_classes=10):
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3072, hidden), nn.ReLU(),
        nn.Linear(hidden, hidden // 2), nn.ReLU(),
        nn.Linear(hidden // 2, num_classes))

def make_cnn(n_ch, num_classes=10):
    """Variable-width CNN for scaling study."""
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
# WORKLOADS
# ══════════════════════════════════════════════════════════════════════════════

def workload_idle(duration=30):
    """W1: GPU idle baseline — no CUDA ops."""
    log.info("W1: Idle baseline (%ds)...", duration)
    time.sleep(duration)
    return {"workload": "idle", "duration_s": duration}


def workload_resnet_fp32(epochs=3, batch_size=256):
    """W2: ResNet-18-like training, full FP32."""
    log.info("W2: ResNet FP32 training (epochs=%d, bs=%d)...", epochs, batch_size)
    dev = torch.device(DEVICE)
    model = make_resnet18_like().to(dev)
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit  = nn.CrossEntropyLoss().to(dev)
    n_steps = max(1, 50000 // batch_size)
    model.train()
    t0 = time.time()
    total_imgs = 0
    for ep in range(epochs):
        for _ in range(n_steps):
            x = torch.randn(batch_size, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (batch_size,), device=dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_imgs += batch_size
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "resnet_fp32", "throughput_img_s": total_imgs / elapsed,
            "elapsed_s": elapsed, "total_imgs": total_imgs}


def workload_resnet_amp(epochs=3, batch_size=512):
    """W3: ResNet-18-like training with AMP (FP16)."""
    log.info("W3: ResNet AMP training (epochs=%d, bs=%d)...", epochs, batch_size)
    dev = torch.device(DEVICE)
    model = make_resnet18_like().to(dev)
    opt    = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit   = nn.CrossEntropyLoss().to(dev)
    scaler = torch.amp.GradScaler("cuda")
    n_steps = max(1, 50000 // batch_size)
    model.train()
    t0 = time.time()
    total_imgs = 0
    for ep in range(epochs):
        for _ in range(n_steps):
            x = torch.randn(batch_size, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (batch_size,), device=dev)
            opt.zero_grad()
            with torch.amp.autocast("cuda"):
                loss = crit(model(x), y)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            total_imgs += batch_size
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "resnet_amp", "throughput_img_s": total_imgs / elapsed,
            "elapsed_s": elapsed, "total_imgs": total_imgs}


def workload_mlp(epochs=30, batch_size=1024):
    """W4: 3-layer MLP training."""
    log.info("W4: MLP training (epochs=%d, bs=%d)...", epochs, batch_size)
    dev = torch.device(DEVICE)
    model = make_mlp().to(dev)
    opt  = optim.Adam(model.parameters(), lr=1e-3)
    crit = nn.CrossEntropyLoss().to(dev)
    n_steps = max(1, 50000 // batch_size)
    model.train()
    t0 = time.time()
    total_imgs = 0
    for ep in range(epochs):
        for _ in range(n_steps):
            x = torch.randn(batch_size, 3, 32, 32, device=dev)
            y = torch.randint(0, 10, (batch_size,), device=dev)
            opt.zero_grad()
            loss = crit(model(x), y)
            loss.backward()
            opt.step()
            total_imgs += batch_size
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "mlp_training", "throughput_img_s": total_imgs / elapsed,
            "elapsed_s": elapsed, "total_imgs": total_imgs}


def workload_inference(batch_size=512, n_batches=200):
    """W5: ResNet-50-like inference (no gradient)."""
    log.info("W5: Inference (bs=%d, n_batches=%d)...", batch_size, n_batches)
    dev = torch.device(DEVICE)
    model = make_resnet50_like().to(dev)
    model.eval()
    t0 = time.time()
    total_imgs = 0
    with torch.no_grad():
        for _ in range(n_batches):
            x = torch.randn(batch_size, 3, 224, 224, device=dev)
            _ = model(x)
            total_imgs += batch_size
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "inference", "throughput_img_s": total_imgs / elapsed,
            "elapsed_s": elapsed, "total_imgs": total_imgs}


def workload_cufft(n_iters=500, size=4096):
    """W6: FFT HPC benchmark — bandwidth-bound, high memory utilization."""
    log.info("W6: cuFFT HPC benchmark (size=%d, iters=%d)...", size, n_iters)
    dev = torch.device(DEVICE)
    x = torch.randn(64, size, size, dtype=torch.float32, device=dev)
    t0 = time.time()
    for _ in range(n_iters):
        y = torch.fft.fft2(x)
        _ = torch.fft.ifft2(y)
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "cufft_hpc", "elapsed_s": elapsed,
            "n_iters": n_iters, "size": size}


def workload_nbody(n_particles=32768, n_steps=300):
    """W7: N-body gravitational simulation — compute-intensive."""
    log.info("W7: N-body simulation (n=%d, steps=%d)...", n_particles, n_steps)
    dev = torch.device(DEVICE)
    pos = torch.randn(n_particles, 3, device=dev)
    vel = torch.zeros(n_particles, 3, device=dev)
    mass = torch.ones(n_particles, 1, device=dev)
    dt = 0.001
    t0 = time.time()
    for _ in range(n_steps):
        diff = pos.unsqueeze(1) - pos.unsqueeze(0)  # [N, N, 3]
        dist_sq = (diff ** 2).sum(-1) + 1e-6
        force = (mass.unsqueeze(0) * diff / dist_sq.unsqueeze(-1) ** 1.5).sum(1)
        vel += force * dt
        pos += vel * dt
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "nbody_sim", "elapsed_s": elapsed,
            "n_particles": n_particles, "n_steps": n_steps}


def workload_mining_proxy(n_iters=20000):
    """W8: Crypto mining proxy — repetitive hash-like SHA256 computation."""
    log.info("W8: Mining proxy (iters=%d)...", n_iters)
    dev = torch.device(DEVICE)
    # Separate fixed buffers for each half — avoids shape collapse across iters
    a = torch.randn(256, 256, device=dev)
    b = torch.randn(256, 256, device=dev)
    t0 = time.time()
    for _ in range(n_iters):
        for _ in range(8):
            a = torch.matmul(a, b.T).remainder_(1e6)
            b = torch.matmul(b, a.T).remainder_(1e6)
        _ = a.sum() + b.sum()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "mining_proxy", "elapsed_s": elapsed, "n_iters": n_iters}


def workload_rendering(n_samples=500000, n_iters=2000):
    """W9: Monte Carlo path tracing proxy — random ray sampling."""
    log.info("W9: Rendering proxy (samples=%d, iters=%d)...", n_samples, n_iters)
    dev = torch.device(DEVICE)
    t0 = time.time()
    for _ in range(n_iters):
        origins    = torch.rand(n_samples, 3, device=dev) * 2 - 1
        directions = torch.randn(n_samples, 3, device=dev)
        directions = directions / directions.norm(dim=-1, keepdim=True)
        sphere_c   = torch.zeros(3, device=dev)
        sphere_r   = 1.0
        oc = origins - sphere_c
        a  = (directions * directions).sum(-1)
        b  = 2.0 * (oc * directions).sum(-1)
        c  = (oc * oc).sum(-1) - sphere_r ** 2
        disc = b * b - 4 * a * c
        hit  = disc > 0
        col  = torch.where(hit.unsqueeze(-1),
                           torch.ones(n_samples, 3, device=dev) * 0.8,
                           torch.zeros(n_samples, 3, device=dev))
        _ = col.sum()
    torch.cuda.synchronize()
    elapsed = time.time() - t0
    return {"workload": "rendering", "elapsed_s": elapsed,
            "n_samples": n_samples, "n_iters": n_iters}


def workload_model_scaling():
    """W10: Model width sweep n_ch ∈ {8,16,32,64,128,256,512} — Week 4 replication."""
    log.info("W10: Model width scaling sweep...")
    dev = torch.device(DEVICE)
    n_ch_values = [8, 16, 32, 64, 128, 256, 512]
    batch_size  = 64
    n_steps_w   = 40
    results = []

    for n_ch in n_ch_values:
        model = make_cnn(n_ch).to(dev)
        opt   = optim.SGD(model.parameters(), lr=0.01)
        crit  = nn.CrossEntropyLoss().to(dev)
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

        # Measure
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

        n_params = sum(p.numel() for p in model.parameters())
        mean_ms  = np.mean(step_times) * 1e3
        # Roofline: flops ≈ 2 * params per forward
        flops     = 2 * n_params * batch_size
        tflops    = flops / (np.mean(step_times)) / 1e12
        results.append({
            "n_ch": n_ch, "n_params": n_params,
            "mean_step_ms": mean_ms,
            "std_step_ms":  np.std(step_times) * 1e3,
            "tflops":       tflops,
        })
        log.info("  n_ch=%3d  params=%8.0fK  step=%.2f ms  tflops=%.2f",
                 n_ch, n_params/1000, mean_ms, tflops)
        del model
        torch.cuda.empty_cache()

    return results


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER (from Week 4/5)
# ══════════════════════════════════════════════════════════════════════════════

def run_classifier(all_dfs: dict) -> dict:
    """Extract features from collected telemetry and train RF classifier."""
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score
    from sklearn.model_selection import train_test_split

    METRICS = ["gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
               "power_draw_w", "temperature_c", "sm_clock_mhz"]

    rows = []
    for label, df in all_dfs.items():
        if df.empty:
            continue
        sub = df[METRICS].dropna()
        if len(sub) < 5:
            continue
        feat = {}
        for col in METRICS:
            v = sub[col].values.astype(float)
            if len(v) == 0:
                continue
            feat[f"{col}_mean"] = v.mean()
            feat[f"{col}_std"]  = v.std()
            feat[f"{col}_min"]  = v.min()
            feat[f"{col}_max"]  = v.max()
            feat[f"{col}_cv"]   = v.std() / (v.mean() + 1e-9)
        feat["workload"] = label
        # Binary label: is this ML training?
        feat["is_training"] = int(label in {
            "resnet_fp32", "resnet_amp", "mlp_training"
        })
        rows.append(feat)

    if len(rows) < 3:
        log.warning("Not enough data for classifier evaluation")
        return {}

    df_feat = pd.DataFrame(rows)
    feature_cols = [c for c in df_feat.columns
                    if c not in ("workload", "is_training")]
    X = np.array(df_feat[feature_cols].fillna(0).to_numpy(), dtype=float)
    y = np.array(df_feat["is_training"].to_numpy(), dtype=int)
    labels_all = np.array(df_feat["workload"].tolist())

    if len(set(y)) < 2:
        log.warning("Only one class present, skipping classifier")
        return {}

    X_tr, X_te, y_tr, y_te, l_tr, l_te = train_test_split(
        X, y, labels_all, test_size=0.3, random_state=42, stratify=y
    )
    clf = RandomForestClassifier(n_estimators=200, random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    acc    = accuracy_score(y_te, y_pred)
    report = classification_report(y_te, y_pred,
                                   target_names=["non_training", "training"],
                                   output_dict=True)
    importances = dict(zip(feature_cols,
                           clf.feature_importances_.tolist()))
    log.info("  Binary classifier accuracy: %.3f", acc)
    return {"accuracy": acc, "report": report,
            "feature_importances": importances}


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(all_dfs: dict, scaling_results: list):
    colors = plt.cm.tab10.colors

    # ── 1. GPU utilization by workload ────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    labels_order = list(all_dfs.keys())
    means = [all_dfs[l]["gpu_utilization_pct"].mean() for l in labels_order]
    stds  = [all_dfs[l]["gpu_utilization_pct"].std()  for l in labels_order]
    x = range(len(labels_order))
    bars = ax.bar(x, means, yerr=stds, capsize=4,
                  color=[colors[i % 10] for i in range(len(labels_order))],
                  edgecolor="black", linewidth=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("GPU Utilization (%)")
    ax.set_title("Week 7 — Single GPU: GPU Utilization per Workload (B200)", fontweight="bold")
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "gpu_util_by_workload.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 2. Power draw by workload ─────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(14, 5))
    pmeans = [all_dfs[l]["power_draw_w"].mean() for l in labels_order]
    pstds  = [all_dfs[l]["power_draw_w"].std()  for l in labels_order]
    ax.bar(x, pmeans, yerr=pstds, capsize=4,
           color=[colors[i % 10] for i in range(len(labels_order))],
           edgecolor="black", linewidth=0.7)
    ax.set_xticks(list(x))
    ax.set_xticklabels(labels_order, rotation=35, ha="right", fontsize=9)
    ax.set_ylabel("Power Draw (W)")
    ax.set_title("Week 7 — Single GPU: Power Draw per Workload (B200)", fontweight="bold")
    ax.axhline(B200_TDP_W, color="red", linestyle="--", linewidth=1.2, label=f"B200 TDP ({B200_TDP_W}W)")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "power_by_workload.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── 3. Time-series for each workload ──────────────────────────────────────
    for label, df in all_dfs.items():
        if df.empty or "gpu_utilization_pct" not in df.columns:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Week 7 — {label} — B200 Telemetry", fontsize=12, fontweight="bold")
        t = df["timestamp_epoch"].values - df["timestamp_epoch"].values[0]
        metrics = [("gpu_utilization_pct", "GPU Util (%)", "steelblue"),
                   ("power_draw_w",        "Power (W)",    "firebrick"),
                   ("mem_used_mb",         "Memory (MB)",  "darkorange")]
        for ax, (col, ylabel, color) in zip(axes, metrics):
            if col in df.columns:
                ax.plot(t, df[col].values, color=color, linewidth=0.9)
            ax.set_ylabel(ylabel)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Elapsed (s)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"timeseries_{label}.png", dpi=100, bbox_inches="tight")
        plt.close()

    # ── 4. Model scaling ──────────────────────────────────────────────────────
    if scaling_results:
        sc_df = pd.DataFrame(scaling_results)
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        fig.suptitle("Week 7 — Model Width Scaling (B200 Single GPU)", fontweight="bold")

        axes[0].plot(sc_df["n_ch"], sc_df["mean_step_ms"], marker="o", color="steelblue")
        axes[0].set_xlabel("Channel width (n_ch)")
        axes[0].set_ylabel("Mean step time (ms)")
        axes[0].set_title("Step Time vs Model Width")
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(sc_df["n_params"] / 1e6, sc_df["tflops"], marker="s", color="firebrick")
        axes[1].set_xlabel("Parameters (M)")
        axes[1].set_ylabel("TFLOPS")
        axes[1].set_title("Compute Efficiency")
        axes[1].grid(True, alpha=0.3)

        axes[2].loglog(sc_df["n_params"], sc_df["mean_step_ms"], marker="^", color="darkorange")
        axes[2].set_xlabel("Parameters (log)")
        axes[2].set_ylabel("Step time (ms, log)")
        axes[2].set_title("Roofline Proxy")
        axes[2].grid(True, alpha=0.3, which="both")

        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_scaling.png", dpi=120, bbox_inches="tight")
        plt.close()

    log.info("Plots saved to %s", PLOTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    if not torch.cuda.is_available():
        log.error("No CUDA GPU found. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    log.info("=" * 60)
    log.info("Week 7 — Single GPU Tests")
    log.info("GPU: %s", gpu_name)
    log.info("PyTorch: %s", torch.__version__)
    log.info("=" * 60)

    all_dfs = {}
    all_results = {}

    # W1: Idle
    df, r = run_with_telemetry("idle", workload_idle)
    all_dfs["idle"] = df
    all_results["idle"] = r

    # W2: ResNet FP32
    df, r = run_with_telemetry("resnet_fp32", workload_resnet_fp32)
    all_dfs["resnet_fp32"] = df
    all_results["resnet_fp32"] = r
    log.info("  ResNet FP32 throughput: %.0f img/s", r.get("throughput_img_s", 0))

    # W3: ResNet AMP
    df, r = run_with_telemetry("resnet_amp", workload_resnet_amp)
    all_dfs["resnet_amp"] = df
    all_results["resnet_amp"] = r
    log.info("  ResNet AMP throughput: %.0f img/s", r.get("throughput_img_s", 0))

    # W4: MLP
    df, r = run_with_telemetry("mlp_training", workload_mlp)
    all_dfs["mlp_training"] = df
    all_results["mlp_training"] = r
    log.info("  MLP throughput: %.0f img/s", r.get("throughput_img_s", 0))

    # W5: Inference
    df, r = run_with_telemetry("inference", workload_inference)
    all_dfs["inference"] = df
    all_results["inference"] = r
    log.info("  Inference throughput: %.0f img/s", r.get("throughput_img_s", 0))

    # W6: cuFFT HPC
    df, r = run_with_telemetry("cufft_hpc", workload_cufft)
    all_dfs["cufft_hpc"] = df
    all_results["cufft_hpc"] = r

    # W7: N-body
    df, r = run_with_telemetry("nbody_sim", workload_nbody)
    all_dfs["nbody_sim"] = df
    all_results["nbody_sim"] = r

    # W8: Mining proxy
    df, r = run_with_telemetry("mining_proxy", workload_mining_proxy)
    all_dfs["mining_proxy"] = df
    all_results["mining_proxy"] = r

    # W9: Rendering
    df, r = run_with_telemetry("rendering", workload_rendering)
    all_dfs["rendering"] = df
    all_results["rendering"] = r

    # W10: Model scaling (no separate telemetry wrapper — scaling runs its own timing)
    log.info("W10: Running model width scaling sweep...")
    scaling_results = workload_model_scaling()
    all_results["model_scaling"] = scaling_results
    sc_df = pd.DataFrame(scaling_results)
    sc_df.to_csv(RESULTS_DIR / "model_scaling.csv", index=False)

    # ── Summary CSV ────────────────────────────────────────────────────────────
    summary_rows = []
    for label, df in all_dfs.items():
        if df.empty:
            continue
        row = {"workload": label, "gpu": gpu_name, "n_samples": len(df)}
        for col in ["gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
                    "power_draw_w", "temperature_c", "sm_clock_mhz"]:
            if col in df.columns:
                v = df[col].dropna().values
                row[f"{col}_mean"] = float(v.mean()) if len(v) else float("nan")
                row[f"{col}_std"]  = float(v.std())  if len(v) else float("nan")
                row[f"{col}_cv"]   = float(v.std() / (v.mean() + 1e-9)) if len(v) else float("nan")
        summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(RESULTS_DIR / "single_gpu_summary.csv", index=False)
    log.info("Saved summary → single_gpu_summary.csv")

    # ── Binary classifier ──────────────────────────────────────────────────────
    clf_results = run_classifier(all_dfs)
    if clf_results:
        with open(RESULTS_DIR / "classifier_results.json", "w") as fh:
            json.dump(clf_results, fh, indent=2)
        log.info("  Classifier saved → classifier_results.json")

    # ── Plots ──────────────────────────────────────────────────────────────────
    make_plots(all_dfs, scaling_results)

    # ── Full results JSON ──────────────────────────────────────────────────────
    def _safe(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        if isinstance(o, np.ndarray): return o.tolist()
        return str(o)

    with open(RESULTS_DIR / "all_results.json", "w") as fh:
        json.dump(all_results, fh, indent=2, default=_safe)

    log.info("=" * 60)
    log.info("Single GPU tests complete.")
    log.info("Results: %s", RESULTS_DIR)
    log.info("Plots:   %s", PLOTS_DIR)
    log.info("=" * 60)

    # Print summary table
    print("\n" + "="*70)
    print(f"{'Workload':<20} {'GPU Util%':>9} {'Power W':>8} {'Mem MB':>8}")
    print("-"*70)
    for row in summary_rows:
        print(f"{row['workload']:<20} "
              f"{row.get('gpu_utilization_pct_mean', float('nan')):>9.1f} "
              f"{row.get('power_draw_w_mean', float('nan')):>8.1f} "
              f"{row.get('mem_used_mb_mean', float('nan')):>8.0f}")
    print("="*70)


if __name__ == "__main__":
    main()
