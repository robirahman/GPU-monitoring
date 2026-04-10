#!/usr/bin/env python3
"""
Week 7 — Edge Cases (Adversarial Workloads) on 2× B200
Replicates Week 4 edge case study with 6 adversarial strategies
plus 2 new B200-specific edge cases.

Cases:
  BASELINE_TRAIN   Standard DDP training (reference)
  BASELINE_INFER   Standard inference     (reference)
  EC1  Phantom Training   — inference + fake allreduce of training-sized tensor
  EC2  Silent Training    — training with allreduce suppressed (gradient accumulation)
  EC3  Sparse Sync        — gradient accumulation ×16 (allreduce every 16 steps)
  EC4  Mining-Like Infer  — large-batch (bs=512) high-power inference
  EC5  Frozen Backbone    — train only final linear head (~0 NVLink traffic)
  EC6  Low-Intensity      — tiny model (n_ch=8) at inference power level
  EC7  AMP Masking        — FP16 training that looks like inference power profile
  EC8  High-batch Idle    — memory-loaded idle (fake high mem-util signal)

Results: week7/edge_cases/{results,data,plots}/
"""

import sys
import os
import time
import json
import logging
import uuid
import threading
from pathlib import Path
from contextlib import contextmanager

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

import pynvml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger("week7.edge_cases")

RESULTS_DIR = Path(__file__).parent / "results"
DATA_DIR    = Path(__file__).parent / "data"
PLOTS_DIR   = Path(__file__).parent / "plots"
for d in [RESULTS_DIR, DATA_DIR, PLOTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# B200 hardware ceilings
B200_FP16_TFLOPS = 4500.0
B200_FP32_TFLOPS =  140.0
B200_MEM_BW_GBPS = 8000.0
B200_NVLINK5_BW  =  900.0
B200_TDP_W       = 1000.0

N_CH      = 128
BATCH     = 64
N_STEPS   = 80
N_WARMUP  = 5
IMG_SIZE  = 32
N_CLASSES = 10


# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════

class TelemetryCollector:
    def __init__(self, gpu_indices=(0, 1), interval: float = 0.2):
        self.gpu_indices = gpu_indices
        self.interval    = interval
        self._stop       = threading.Event()
        self._records    = {i: [] for i in gpu_indices}
        self._thread     = None
        pynvml.nvmlInit()
        self._handles = {i: pynvml.nvmlDeviceGetHandleByIndex(i)
                         for i in gpu_indices}

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
            ts = time.time()
            for i, h in self._handles.items():
                try:
                    util = pynvml.nvmlDeviceGetUtilizationRates(h)
                    mem  = pynvml.nvmlDeviceGetMemoryInfo(h)
                    self._records[i].append({
                        "timestamp_epoch":    ts,
                        "gpu_utilization_pct": util.gpu,
                        "mem_utilization_pct": util.memory,
                        "mem_used_mb":         mem.used // (1024*1024),
                        "power_draw_w":        pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0,
                        "temperature_c":       pynvml.nvmlDeviceGetTemperature(h, pynvml.NVML_TEMPERATURE_GPU),
                        "sm_clock_mhz":        pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM),
                    })
                except Exception:
                    pass
            self._stop.wait(max(0, self.interval - (time.monotonic() - t0)))

    def to_dataframes(self):
        return {i: pd.DataFrame(recs) for i, recs in self._records.items()}

    def save(self, label):
        dfs = self.to_dataframes()
        for i, df in dfs.items():
            if df.empty: continue
            fname = DATA_DIR / f"{label}_gpu{i}_{uuid.uuid4().hex[:6]}.parquet"
            df["gpu_index"] = i
            df.to_parquet(str(fname), index=False)
        return dfs

    def cleanup(self):
        try:
            pynvml.nvmlShutdown()
        except Exception:
            pass


def run_edge_case(label: str, fn) -> dict:
    """Run edge case workload, collect telemetry, extract stats."""
    tc = TelemetryCollector(gpu_indices=[0, 1], interval=0.2)
    tc.start()
    time.sleep(2)
    t0_wall = time.time()
    try:
        result = fn()
    except Exception as e:
        log.error("  %s failed: %s", label, e)
        result = {"error": str(e)}
    elapsed = time.time() - t0_wall
    time.sleep(2)
    tc.stop()
    dfs = tc.save(label)
    tc.cleanup()

    # Extract aggregate stats for each GPU
    stats = {"label": label, "elapsed_s": elapsed}
    for i, df in dfs.items():
        if df.empty: continue
        for col in ["gpu_utilization_pct", "power_draw_w", "mem_used_mb", "sm_clock_mhz"]:
            if col not in df.columns: continue
            v = df[col].dropna().values
            if len(v) == 0: continue
            stats[f"gpu{i}_{col}_mean"] = float(v.mean())
            stats[f"gpu{i}_{col}_std"]  = float(v.std())
            stats[f"gpu{i}_{col}_cv"]   = float(v.std() / (v.mean() + 1e-9))
    if result:
        stats.update({k: v for k, v in result.items()
                      if not isinstance(v, (torch.Tensor, np.ndarray))})
    log.info("  %-20s  GPU0_util=%.1f%%  GPU0_power=%.0fW  CV=%.2f",
             label,
             stats.get("gpu0_gpu_utilization_pct_mean", 0),
             stats.get("gpu0_power_draw_w_mean", 0),
             stats.get("gpu0_gpu_utilization_pct_cv", 0))
    return stats, dfs


# ══════════════════════════════════════════════════════════════════════════════
# MODEL
# ══════════════════════════════════════════════════════════════════════════════

def make_model(n_ch=N_CH):
    n = n_ch
    def cb(ci, co): return nn.Sequential(
        nn.Conv2d(ci, co, 3, padding=1, bias=False),
        nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    return nn.Sequential(
        cb(3, n), cb(n, 2*n), nn.MaxPool2d(2),
        cb(2*n, 4*n), cb(4*n, 4*n), nn.MaxPool2d(2),
        cb(4*n, 8*n), nn.MaxPool2d(2), cb(8*n, 8*n),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(8*n, N_CLASSES))

def _batch(bs=BATCH, dev=None):
    x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE)
    return x.to(dev) if dev else x


# ══════════════════════════════════════════════════════════════════════════════
# BASELINES
# ══════════════════════════════════════════════════════════════════════════════

def baseline_train():
    """Standard AMP training on GPU0 (single-GPU reference)."""
    dev = torch.device("cuda:0")
    model = make_model().to(dev)
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit  = nn.CrossEntropyLoss().to(dev)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for _ in range(N_WARMUP):
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad()
        with torch.amp.autocast("cuda"): loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    step_times = []
    for _ in range(N_STEPS):
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        t0 = time.perf_counter()
        opt.zero_grad()
        with torch.amp.autocast("cuda"): loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    return {"mean_step_ms": np.mean(step_times) * 1e3}


def baseline_infer():
    """Standard FP32 inference on GPU0 (reference)."""
    dev = torch.device("cuda:0")
    model = make_model().to(dev)
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP + N_STEPS):
            x = _batch(BATCH, dev)
            _ = model(x)
    torch.cuda.synchronize()
    return {"mode": "inference"}


# ══════════════════════════════════════════════════════════════════════════════
# EC1: Phantom Training
# ══════════════════════════════════════════════════════════════════════════════

def ec1_phantom_training():
    """EC1: Inference + fake allreduce of training-sized tensor."""
    dev = torch.device("cuda:0")
    model = make_model().to(dev)
    model.eval()
    n_params = sum(p.numel() for p in model.parameters())
    fake_grad = torch.randn(n_params, device="cuda:1")
    fake_grad_dst = torch.empty_like(fake_grad, device="cuda:0")

    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(_batch(BATCH, dev))
            fake_grad_dst.copy_(fake_grad)  # fake allreduce

    step_times = []
    with torch.no_grad():
        for _ in range(N_STEPS):
            t0 = time.perf_counter()
            _ = model(_batch(BATCH, dev))
            fake_grad_dst.copy_(fake_grad)  # fake NVLink traffic
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
    return {
        "label": "EC1_phantom_train",
        "mean_step_ms": np.mean(step_times) * 1e3,
        "fake_allreduce_bytes": n_params * 4,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EC2: Silent Training
# ══════════════════════════════════════════════════════════════════════════════

def ec2_silent_training():
    """EC2: Training with gradient sync suppressed (no NVLink traffic)."""
    dev = torch.device("cuda:0")
    model = make_model().to(dev)
    opt  = optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss().to(dev)
    model.train()
    # No gradient sync — just compute backward on GPU0, never send
    for _ in range(N_WARMUP):
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    torch.cuda.synchronize()
    step_times = []
    for _ in range(N_STEPS):
        t0 = time.perf_counter()
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    return {"label": "EC2_silent_train", "mean_step_ms": np.mean(step_times) * 1e3}


# ══════════════════════════════════════════════════════════════════════════════
# EC3: Sparse Sync (gradient accumulation ×16)
# ══════════════════════════════════════════════════════════════════════════════

def ec3_sparse_sync():
    """EC3: Allreduce every 16 steps (gradient accumulation)."""
    ACCUM = 16
    dev = torch.device("cuda:0")
    model = make_model().to(dev)
    opt  = optim.SGD(model.parameters(), lr=0.01)
    crit = nn.CrossEntropyLoss().to(dev)
    n_params = sum(p.numel() for p in model.parameters())
    grad_buf = torch.zeros(n_params, device="cuda:1")
    grad_dst = torch.zeros(n_params, device="cuda:0")
    model.train()
    for _ in range(N_WARMUP):
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward()
    torch.cuda.synchronize()
    step_times = []; allreduce_count = 0
    for step in range(N_STEPS):
        t0 = time.perf_counter()
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        loss = crit(model(x), y); loss.backward()
        if (step + 1) % ACCUM == 0:
            # Only sync every ACCUM steps
            grad_dst.copy_(grad_buf)
            opt.step(); opt.zero_grad(); allreduce_count += 1
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    return {
        "label": "EC3_sparse_sync",
        "mean_step_ms": np.mean(step_times) * 1e3,
        "allreduce_count": allreduce_count,
        "accum_factor": ACCUM,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EC4: Mining-Like Inference (high-batch, high-power)
# ══════════════════════════════════════════════════════════════════════════════

def ec4_mining_like_infer():
    """EC4: Large-batch inference mimicking mining power profile."""
    dev   = torch.device("cuda:0")
    BS    = 512
    model = make_model(n_ch=256).to(dev)
    model.eval()
    with torch.no_grad():
        for _ in range(N_WARMUP):
            _ = model(_batch(BS, dev))
    torch.cuda.synchronize()
    step_times = []
    with torch.no_grad():
        for _ in range(N_STEPS):
            t0 = time.perf_counter()
            _ = model(_batch(BS, dev))
            torch.cuda.synchronize()
            step_times.append(time.perf_counter() - t0)
    return {
        "label": "EC4_mining_like_infer",
        "batch_size": BS,
        "n_ch": 256,
        "mean_step_ms": np.mean(step_times) * 1e3,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EC5: Frozen Backbone (train only head)
# ══════════════════════════════════════════════════════════════════════════════

def ec5_frozen_backbone():
    """EC5: Only the final linear layer is trained — ~0 gradient bytes."""
    dev  = torch.device("cuda:0")
    model = make_model().to(dev)
    # Freeze everything except the last linear
    for name, param in model.named_parameters():
        if "Linear" not in type(model[-1]).__name__ or name.split(".")[0] != str(len(model)-1):
            param.requires_grad = False
    # Simple: freeze all, unfreeze last
    for param in model.parameters():
        param.requires_grad = False
    for param in model[-1].parameters():
        param.requires_grad = True

    opt  = optim.SGD(filter(lambda p: p.requires_grad, model.parameters()), lr=0.01)
    crit = nn.CrossEntropyLoss().to(dev)
    grad_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    model.train()
    for _ in range(N_WARMUP):
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    torch.cuda.synchronize()
    step_times = []
    for _ in range(N_STEPS):
        t0 = time.perf_counter()
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    return {
        "label": "EC5_frozen_backbone",
        "trainable_params": grad_params,
        "mean_step_ms": np.mean(step_times) * 1e3,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EC6: Low-Intensity Training
# ══════════════════════════════════════════════════════════════════════════════

def ec6_low_intensity():
    """EC6: Tiny model (n_ch=8, bs=4) at inference power levels."""
    dev   = torch.device("cuda:0")
    model = make_model(n_ch=8).to(dev)
    opt   = optim.SGD(model.parameters(), lr=0.01)
    crit  = nn.CrossEntropyLoss().to(dev)
    BS = 4
    model.train()
    for _ in range(N_WARMUP):
        x = _batch(BS, dev); y = torch.randint(0, N_CLASSES, (BS,), device=dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
    torch.cuda.synchronize()
    step_times = []
    for _ in range(N_STEPS):
        t0 = time.perf_counter()
        x = _batch(BS, dev); y = torch.randint(0, N_CLASSES, (BS,), device=dev)
        opt.zero_grad(); loss = crit(model(x), y); loss.backward(); opt.step()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
    return {
        "label": "EC6_low_intensity",
        "n_ch": 8, "batch_size": BS,
        "mean_step_ms": np.mean(step_times) * 1e3,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EC7: AMP Masking (B200-specific)
# ══════════════════════════════════════════════════════════════════════════════

def ec7_amp_masking():
    """EC7 (B200-new): FP8/FP16 AMP training with power-throttle to look
    like inference — exploits B200's dynamic power capping."""
    dev = torch.device("cuda:0")
    model = make_model(n_ch=32).to(dev)  # small enough to stay low-power
    opt   = optim.Adam(model.parameters(), lr=1e-4)  # Adam vs SGD: smoother
    crit  = nn.CrossEntropyLoss().to(dev)
    scaler = torch.amp.GradScaler("cuda")
    model.train()
    for _ in range(N_WARMUP):
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad()
        with torch.amp.autocast("cuda"): loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
    torch.cuda.synchronize()
    step_times = []
    for _ in range(N_STEPS):
        t0 = time.perf_counter()
        x = _batch(BATCH, dev); y = torch.randint(0, N_CLASSES, (BATCH,), device=dev)
        opt.zero_grad()
        with torch.amp.autocast("cuda"): loss = crit(model(x), y)
        scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        torch.cuda.synchronize()
        step_times.append(time.perf_counter() - t0)
        # Throttle: sleep to simulate inference-like duty cycle
        time.sleep(0.005)
    return {
        "label": "EC7_amp_masking",
        "n_ch": 32,
        "mean_step_ms": np.mean(step_times) * 1e3,
    }


# ══════════════════════════════════════════════════════════════════════════════
# EC8: High-Batch Memory-Loaded Idle
# ══════════════════════════════════════════════════════════════════════════════

def ec8_memory_loaded_idle():
    """EC8 (B200-new): Allocate large buffers (fake high mem-util) but idle compute."""
    dev = torch.device("cuda:0")
    # Allocate ~50GB on B200 (has 183GB)
    n_bytes = 50 * 1024**3 // 4
    big_buf = torch.zeros(n_bytes, dtype=torch.float32, device=dev)
    # Just do trivial ops to keep memory resident
    t0 = time.time()
    for _ in range(N_STEPS):
        _ = big_buf[0]   # no compute
        time.sleep(0.2)
    elapsed = time.time() - t0
    del big_buf
    torch.cuda.empty_cache()
    return {
        "label": "EC8_memory_loaded_idle",
        "allocated_gb": 50.0,
        "elapsed_s": elapsed,
    }


# ══════════════════════════════════════════════════════════════════════════════
# CLASSIFIER ON EDGE CASES
# ══════════════════════════════════════════════════════════════════════════════

def run_classifier(all_stats: list) -> dict:
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.metrics import classification_report, accuracy_score

    rows = []
    for s in all_stats:
        label = s.get("label", "unknown")
        is_training = int(label in {
            "baseline_train", "EC2_silent_train", "EC3_sparse_sync",
            "EC5_frozen_backbone", "EC6_low_intensity", "EC7_amp_masking"
        })
        row = {
            "label": label,
            "is_training": is_training,
        }
        for k in ["gpu0_gpu_utilization_pct_mean", "gpu0_gpu_utilization_pct_cv",
                  "gpu0_power_draw_w_mean", "gpu0_power_draw_w_cv",
                  "gpu0_sm_clock_mhz_mean", "gpu0_mem_used_mb_mean"]:
            row[k] = s.get(k, 0.0)
        rows.append(row)

    if len(rows) < 4:
        return {}

    df = pd.DataFrame(rows)
    feat_cols = [c for c in df.columns if c not in ("label", "is_training")]
    X = df[feat_cols].fillna(0).values
    y = df["is_training"].values

    if len(set(y)) < 2:
        return {"note": "Only one class present"}

    # Leave-one-out style: train on all but test on each
    from sklearn.model_selection import LeaveOneOut
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    preds = []
    for train_idx, test_idx in LeaveOneOut().split(X):
        clf.fit(X[train_idx], y[train_idx])
        preds.append(clf.predict(X[test_idx])[0])

    acc = accuracy_score(y, preds)
    log.info("  Edge case classifier LOO accuracy: %.3f", acc)

    clf.fit(X, y)
    importances = dict(zip(feat_cols, clf.feature_importances_.tolist()))
    return {
        "loo_accuracy": acc,
        "n_cases": len(rows),
        "feature_importances": importances,
        "labels": df["label"].tolist(),
        "true":  y.tolist(),
        "pred":  preds,
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTS
# ══════════════════════════════════════════════════════════════════════════════

def make_plots(all_stats: list):
    df = pd.DataFrame(all_stats)
    if df.empty:
        return

    labels = df["label"].tolist()
    x = range(len(labels))

    # ── Radar chart: 4 metrics ─────────────────────────────────────────────────
    metrics = [
        ("gpu0_gpu_utilization_pct_mean", "GPU Util %"),
        ("gpu0_power_draw_w_mean",        "Power W"),
        ("gpu0_gpu_utilization_pct_cv",   "Util CV"),
        ("gpu0_sm_clock_mhz_mean",        "SM Clock MHz"),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Week 7 — Edge Cases: B200 Telemetry Signatures", fontsize=14, fontweight="bold")
    colors = plt.cm.tab10.colors

    for ax, (col, ylabel) in zip(axes.flat, metrics):
        if col not in df.columns:
            ax.set_visible(False)
            continue
        vals = df[col].fillna(0).values
        bars = ax.bar(x, vals,
                      color=[colors[i % 10] for i in range(len(labels))],
                      edgecolor="black", linewidth=0.6)
        ax.set_xticks(list(x))
        ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontweight="bold")
        ax.grid(True, alpha=0.3, axis="y")

    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "edge_cases_summary.png", dpi=120, bbox_inches="tight")
    plt.close()

    # ── CV scatter: detectability ──────────────────────────────────────────────
    if ("gpu0_gpu_utilization_pct_cv" in df.columns and
            "gpu0_power_draw_w_mean" in df.columns):
        fig, ax = plt.subplots(figsize=(10, 7))
        for i, row in df.iterrows():
            color = "red" if row.get("label", "") in {
                "baseline_train", "EC2_silent_train", "EC3_sparse_sync",
                "EC5_frozen_backbone", "EC6_low_intensity", "EC7_amp_masking"
            } else "blue"
            ax.scatter(row.get("gpu0_gpu_utilization_pct_cv", 0),
                       row.get("gpu0_power_draw_w_mean", 0),
                       c=color, s=120, zorder=5)
            ax.annotate(row.get("label", f"case{i}"),
                        (row.get("gpu0_gpu_utilization_pct_cv", 0),
                         row.get("gpu0_power_draw_w_mean", 0)),
                        fontsize=7, ha="left", va="bottom")
        from matplotlib.lines import Line2D
        legend_elements = [
            Line2D([0], [0], marker="o", color="w", markerfacecolor="red", markersize=10, label="Training"),
            Line2D([0], [0], marker="o", color="w", markerfacecolor="blue", markersize=10, label="Non-training"),
        ]
        ax.legend(handles=legend_elements)
        ax.set_xlabel("GPU Utilization CV (temporal variability)")
        ax.set_ylabel("Mean Power Draw (W)")
        ax.set_title("Edge Case Detectability — Week 7 B200\n"
                     "(Red=training, Blue=non-training)", fontweight="bold")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "edge_case_scatter.png", dpi=120, bbox_inches="tight")
        plt.close()

    log.info("Plots saved → %s", PLOTS_DIR)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

CASES = [
    ("baseline_train",  baseline_train),
    ("baseline_infer",  baseline_infer),
    ("EC1_phantom",     ec1_phantom_training),
    ("EC2_silent",      ec2_silent_training),
    ("EC3_sparse",      ec3_sparse_sync),
    ("EC4_mining",      ec4_mining_like_infer),
    ("EC5_frozen",      ec5_frozen_backbone),
    ("EC6_low_int",     ec6_low_intensity),
    ("EC7_amp_mask",    ec7_amp_masking),
    ("EC8_mem_idle",    ec8_memory_loaded_idle),
]


def main():
    if not torch.cuda.is_available():
        log.error("No CUDA GPU found. Exiting.")
        sys.exit(1)

    gpu_name = torch.cuda.get_device_name(0)
    log.info("=" * 60)
    log.info("Week 7 — Edge Cases")
    log.info("GPUs: %d× %s", torch.cuda.device_count(), gpu_name)
    log.info("Cases: %d", len(CASES))
    log.info("=" * 60)

    all_stats = []

    for label, fn in CASES:
        log.info("Running: %s", label)
        stats, dfs = run_edge_case(label, fn)
        stats["label"] = label
        all_stats.append(stats)
        torch.cuda.empty_cache()
        time.sleep(3)  # cooldown between cases

    # Summary CSV
    summary_df = pd.DataFrame(all_stats)
    summary_df.to_csv(RESULTS_DIR / "edge_cases_summary.csv", index=False)
    log.info("Saved → edge_cases_summary.csv")

    # Classifier
    clf_results = run_classifier(all_stats)
    if clf_results:
        with open(RESULTS_DIR / "classifier_results.json", "w") as fh:
            json.dump(clf_results, fh, indent=2)
        log.info("Edge case classifier LOO accuracy: %.3f",
                 clf_results.get("loo_accuracy", 0))

    # Plots
    make_plots(all_stats)

    # JSON dump
    def _safe(o):
        if isinstance(o, (np.integer,)): return int(o)
        if isinstance(o, (np.floating,)): return float(o)
        return str(o)

    with open(RESULTS_DIR / "edge_cases_full.json", "w") as fh:
        json.dump(all_stats, fh, indent=2, default=_safe)

    log.info("=" * 60)
    log.info("Edge cases complete. Results: %s", RESULTS_DIR)
    log.info("=" * 60)

    # Print summary table
    print("\n" + "="*80)
    print(f"{'Case':<22} {'GPU0 Util%':>10} {'Power W':>8} {'Util CV':>8} {'Training?':>10}")
    print("-"*80)
    TRAIN_LABELS = {
        "baseline_train", "EC2_silent_train", "EC3_sparse_sync",
        "EC5_frozen_backbone", "EC6_low_intensity", "EC7_amp_masking",
    }
    for s in all_stats:
        lbl = s.get("label", "?")
        is_train = lbl in TRAIN_LABELS or "train" in lbl.lower()
        print(f"{lbl:<22} "
              f"{s.get('gpu0_gpu_utilization_pct_mean', 0):>10.1f} "
              f"{s.get('gpu0_power_draw_w_mean', 0):>8.1f} "
              f"{s.get('gpu0_gpu_utilization_pct_cv', 0):>8.3f} "
              f"{'YES' if is_train else 'no':>10}")
    print("="*80)
    if clf_results:
        print(f"\nEdge Case Classifier LOO Accuracy: {clf_results.get('loo_accuracy', 0):.1%}")


if __name__ == "__main__":
    main()
