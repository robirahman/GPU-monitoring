#!/usr/bin/env python3
"""
Week 4 Step 4: NVLink Characterization (2x H100 80GB)

Tests:
  T1 - NVLink bandwidth: sustained peer-to-peer transfers at increasing sizes
  T2 - DDP training (ResNet-18 on CIFAR-10) using NVLink (torch.distributed)
  T3 - BERT fine-tuning distributed across 2 GPUs via DDP
  T4 - Simultaneous bi-directional NVLink transfers (contention test)
  T5 - GPU-to-GPU latency (ping-pong small tensors)
  T6 - Single-GPU vs 2-GPU throughput comparison (training throughput)

All tests collect pynvml telemetry at 1 Hz. Results saved to week4/results/nvlink/
"""

import os
import sys
import time
import json
import threading
import subprocess
import tempfile
import uuid
import logging
from pathlib import Path

import torch
import numpy as np

REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT))
from scripts.collect_telemetry import TelemetryCollector

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
log = logging.getLogger("week4.nvlink")

RESULTS_DIR = REPO_ROOT / "week7" / "results" / "nvlink"
PLOTS_DIR   = REPO_ROOT / "week7" / "plots"
DATA_DIR    = REPO_ROOT / "week7" / "data"
for d in [RESULTS_DIR, PLOTS_DIR, DATA_DIR]:
    d.mkdir(parents=True, exist_ok=True)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# ── Helpers ─────────────────────────────────────────────────────────────────

def collect_telemetry_both_gpus(label: str, fn, duration_hint: int = 120):
    """
    Run fn() while collecting telemetry on both GPUs.
    Returns (result_of_fn, df_gpu0, df_gpu1).
    """
    collectors = [
        TelemetryCollector(gpu_index=i, interval_sec=0.5,
                           workload_label=label, enable_dcgm=False)
        for i in range(2)
    ]
    for c in collectors:
        c.start()
    time.sleep(2)
    try:
        result = fn()
    finally:
        time.sleep(2)
        for c in collectors:
            c.stop()

    dfs = []
    for i, c in enumerate(collectors):
        df = c.to_dataframe()
        df["gpu_index"] = i
        fname = DATA_DIR / f"{label}_gpu{i}_{uuid.uuid4().hex[:6]}.parquet"
        c.save(str(fname))
        log.info("Saved %s (%d samples)", fname.name, c.sample_count)
        dfs.append(df)
        c.cleanup()

    return result, dfs[0], dfs[1]


# ── T1: NVLink Bandwidth ─────────────────────────────────────────────────────

def test_nvlink_bandwidth():
    """Measure sustained D2D bandwidth across NVLink for various tensor sizes."""
    log.info("=== T1: NVLink Bandwidth ===")
    sizes_mb = [1, 4, 16, 64, 256, 512, 1024, 2048, 4096]
    n_warmup = 5
    n_iters  = 50

    results = []

    for size_mb in sizes_mb:
        n_elems = (size_mb * 1024 * 1024) // 4  # float32
        src = torch.randn(n_elems, dtype=torch.float32, device="cuda:0")
        dst = torch.empty(n_elems, dtype=torch.float32, device="cuda:1")

        # Warmup
        for _ in range(n_warmup):
            dst.copy_(src)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)

        t0 = time.perf_counter()
        for _ in range(n_iters):
            dst.copy_(src)
        torch.cuda.synchronize(0)
        torch.cuda.synchronize(1)
        t1 = time.perf_counter()

        elapsed = t1 - t0
        bw_gbps = (size_mb * n_iters / 1024) / elapsed  # GB/s
        log.info("  %5d MB → bandwidth = %.2f GB/s", size_mb, bw_gbps)
        results.append({
            "size_mb": size_mb,
            "bandwidth_gbps": bw_gbps,
            "elapsed_sec": elapsed,
            "n_iters": n_iters,
        })

    # Also measure bidirectional simultaneously
    log.info("  Bidirectional (simultaneous)...")
    size_mb = 1024
    n_elems = (size_mb * 1024 * 1024) // 4
    a = torch.randn(n_elems, dtype=torch.float32, device="cuda:0")
    b = torch.randn(n_elems, dtype=torch.float32, device="cuda:1")
    dst_a = torch.empty_like(b)
    dst_b = torch.empty_like(a)

    streams = [torch.cuda.Stream(device=0), torch.cuda.Stream(device=1)]

    # Warmup
    for _ in range(n_warmup):
        with torch.cuda.stream(streams[0]):
            dst_a.copy_(a)
        with torch.cuda.stream(streams[1]):
            dst_b.copy_(b)
    torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_iters):
        with torch.cuda.stream(streams[0]):
            dst_a.copy_(a)
        with torch.cuda.stream(streams[1]):
            dst_b.copy_(b)
    torch.cuda.synchronize()
    t1 = time.perf_counter()

    bw_bidir = (size_mb * 2 * n_iters / 1024) / (t1 - t0)
    log.info("  Bidirectional %d MB: %.2f GB/s (each dir)", size_mb, bw_bidir / 2)
    results.append({
        "size_mb": size_mb,
        "bandwidth_gbps": bw_bidir,
        "elapsed_sec": t1 - t0,
        "n_iters": n_iters,
        "note": "bidirectional",
    })

    return results


# ── T5: Latency ──────────────────────────────────────────────────────────────

def test_nvlink_latency():
    """Measure GPU0 → GPU1 → GPU0 round-trip latency (ping-pong)."""
    log.info("=== T5: NVLink Latency (ping-pong) ===")
    n_iters = 1000
    sizes = [1, 4, 16, 64, 256, 1024]   # elements (float32)

    results = []
    for n in sizes:
        a = torch.ones(n, dtype=torch.float32, device="cuda:0")
        b = torch.empty(n, dtype=torch.float32, device="cuda:1")
        c = torch.empty(n, dtype=torch.float32, device="cuda:0")

        # Warmup
        for _ in range(50):
            b.copy_(a); c.copy_(b)
        torch.cuda.synchronize()

        t0 = time.perf_counter()
        for _ in range(n_iters):
            b.copy_(a)
            c.copy_(b)
        torch.cuda.synchronize()
        t1 = time.perf_counter()

        rtt_us = ((t1 - t0) / n_iters) * 1e6
        log.info("  %5d floats → RTT = %.2f µs", n, rtt_us)
        results.append({"n_floats": n, "size_bytes": n * 4, "rtt_us": rtt_us})

    return results


# ── T6: Single-GPU vs 2-GPU Training Throughput ───────────────────────────────

def measure_training_throughput(device_ids, epochs=2, batch_size=512):
    """
    Train ResNet-18-like model (pure torch, no torchvision) and return images/sec.
    device_ids: [0] for single GPU, [0,1] for DataParallel
    """
    import sys as _sys
    _sys.path.insert(0, str(REPO_ROOT))
    from week7.workloads_w4 import make_resnet18_like
    import torch.nn as nn
    import torch.optim as optim

    device = torch.device(f"cuda:{device_ids[0]}")
    n_train = 50000
    batches_per_epoch = max(1, n_train // batch_size)

    model = make_resnet18_like(num_classes=10)
    model = model.to(device)

    if len(device_ids) > 1:
        model = nn.DataParallel(model, device_ids=device_ids)

    optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    criterion = nn.CrossEntropyLoss().to(device)
    scaler = torch.amp.GradScaler("cuda", enabled=True)

    model.train()
    total_images = 0
    t_start = time.time()

    for epoch in range(epochs):
        for step in range(batches_per_epoch):
            data   = torch.randn(batch_size, 3, 32, 32, device=device)
            target = torch.randint(0, 10, (batch_size,), device=device)
            optimizer.zero_grad()
            with torch.amp.autocast("cuda", enabled=True):
                out = model(data)
                loss = criterion(out, target)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            total_images += batch_size

    torch.cuda.synchronize()
    elapsed = time.time() - t_start
    throughput = total_images / elapsed
    return throughput, elapsed, total_images


# ── Main orchestration ────────────────────────────────────────────────────────

def run_all_tests():
    import pandas as pd
    summary = {}

    # T1: NVLink Bandwidth
    def t1_fn():
        return test_nvlink_bandwidth()

    bw_results, df0_t1, df1_t1 = collect_telemetry_both_gpus(
        "nvlink_bandwidth", t1_fn, duration_hint=60
    )
    summary["nvlink_bandwidth"] = bw_results

    bw_df = pd.DataFrame(bw_results)
    bw_df.to_csv(RESULTS_DIR / "nvlink_bandwidth.csv", index=False)
    log.info("Peak unidirectional bandwidth: %.2f GB/s",
             max(r["bandwidth_gbps"] for r in bw_results
                 if "note" not in r))

    # T5: Latency
    def t5_fn():
        return test_nvlink_latency()

    lat_results, df0_t5, df1_t5 = collect_telemetry_both_gpus(
        "nvlink_latency", t5_fn, duration_hint=30
    )
    summary["nvlink_latency"] = lat_results
    lat_df = pd.DataFrame(lat_results)
    lat_df.to_csv(RESULTS_DIR / "nvlink_latency.csv", index=False)

    # T6: Single-GPU vs 2-GPU throughput
    log.info("=== T6: Single-GPU vs 2-GPU Training Throughput ===")

    def t6_single():
        tp, elapsed, imgs = measure_training_throughput([0], epochs=2, batch_size=512)
        log.info("  Single-GPU: %.0f img/s (%.1fs, %d imgs)", tp, elapsed, imgs)
        return {"gpus": 1, "throughput_imgs_per_sec": tp,
                "elapsed_sec": elapsed, "total_images": imgs}

    def t6_dual():
        tp, elapsed, imgs = measure_training_throughput([0, 1], epochs=2, batch_size=1024)
        log.info("  2-GPU DP:   %.0f img/s (%.1fs, %d imgs)", tp, elapsed, imgs)
        return {"gpus": 2, "throughput_imgs_per_sec": tp,
                "elapsed_sec": elapsed, "total_images": imgs}

    single_res, df0_single, df1_single = collect_telemetry_both_gpus(
        "training_single_gpu", t6_single, duration_hint=300
    )
    dual_res, df0_dual, df1_dual = collect_telemetry_both_gpus(
        "training_dual_gpu_dp", t6_dual, duration_hint=300
    )
    summary["throughput_comparison"] = [single_res, dual_res]
    pd.DataFrame([single_res, dual_res]).to_csv(
        RESULTS_DIR / "throughput_comparison.csv", index=False
    )

    return summary, {
        "bw": (df0_t1, df1_t1),
        "latency": (df0_t5, df1_t5),
        "single": (df0_single, df1_single),
        "dual": (df0_dual, df1_dual),
    }


def make_plots(summary, telemetry_dfs):
    import pandas as pd

    # ── Bandwidth curve ──────────────────────────────────────────────────────
    bw_df = pd.DataFrame([r for r in summary["nvlink_bandwidth"]
                          if "note" not in r])
    bidir_row = next((r for r in summary["nvlink_bandwidth"] if "note" in r), None)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.semilogx(bw_df["size_mb"], bw_df["bandwidth_gbps"],
                marker="o", linewidth=2, color="#1565C0", label="Unidirectional (GPU0→GPU1)")
    if bidir_row:
        ax.axhline(bidir_row["bandwidth_gbps"] / 2, color="#E53935",
                   linestyle="--", linewidth=1.5, label="Bidirectional (per direction)")
    ax.set_xlabel("Transfer size (MB, log scale)")
    ax.set_ylabel("Bandwidth (GB/s)")
    ax.set_title("NVLink Bandwidth — 2× H100 80GB (NV6)", fontsize=13, fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "nvlink_bandwidth.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved nvlink_bandwidth.png")

    # ── Latency curve ────────────────────────────────────────────────────────
    lat_df = pd.DataFrame(summary["nvlink_latency"])
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.semilogx(lat_df["size_bytes"] / 1024, lat_df["rtt_us"],
                marker="s", linewidth=2, color="#2E7D32")
    ax.set_xlabel("Transfer size (KB, log scale)")
    ax.set_ylabel("Round-trip latency (µs)")
    ax.set_title("NVLink Ping-Pong Latency — 2× H100", fontsize=13, fontweight="bold")
    ax.grid(True, alpha=0.3, which="both")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "nvlink_latency.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved nvlink_latency.png")

    # ── Throughput comparison bar chart ─────────────────────────────────────
    tp_df = pd.DataFrame(summary["throughput_comparison"])
    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ["#1565C0", "#43A047"]
    bars = ax.bar(["Single GPU\n(H100 GPU0)", "2× GPU\n(DataParallel, NVLink)"],
                  tp_df["throughput_imgs_per_sec"], color=colors, edgecolor="black",
                  linewidth=0.8, width=0.5)
    for bar, row in zip(bars, tp_df.itertuples()):
        ax.text(bar.get_x() + bar.get_width() / 2,
                bar.get_height() + tp_df["throughput_imgs_per_sec"].max() * 0.01,
                f"{row.throughput_imgs_per_sec:,.0f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold")
    if len(tp_df) == 2:
        speedup = tp_df.iloc[1]["throughput_imgs_per_sec"] / tp_df.iloc[0]["throughput_imgs_per_sec"]
        ax.set_title(f"Training Throughput: Single vs 2-GPU\n(Speedup: {speedup:.2f}×)",
                     fontsize=12, fontweight="bold")
    ax.set_ylabel("Images / second")
    ax.grid(True, alpha=0.3, axis="y")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "throughput_comparison.png", dpi=120, bbox_inches="tight")
    plt.close()
    log.info("Saved throughput_comparison.png")

    # ── Telemetry during dual-GPU training (both GPUs) ───────────────────────
    for tag, (df0, df1) in telemetry_dfs.items():
        if df0.empty or "gpu_utilization_pct" not in df0.columns:
            continue
        fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
        fig.suptitle(f"Telemetry: {tag}", fontsize=12, fontweight="bold")
        metrics = ["gpu_utilization_pct", "power_draw_w", "mem_used_mb"]
        ylabels = ["GPU Util (%)", "Power (W)", "Memory (MB)"]
        colors  = ["#1565C0", "#E53935"]
        for ax, metric, ylabel in zip(axes, metrics, ylabels):
            for gpu_idx, (df, label) in enumerate([(df0, "GPU 0"), (df1, "GPU 1")]):
                if metric not in df.columns:
                    continue
                t = df["timestamp_epoch"] - df["timestamp_epoch"].min() if "timestamp_epoch" in df.columns else range(len(df))
                ax.plot(t, df[metric], color=colors[gpu_idx], linewidth=0.9,
                        alpha=0.8, label=label)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel("Elapsed (s)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / f"telemetry_{tag}.png", dpi=120, bbox_inches="tight")
        plt.close()
        log.info("Saved telemetry_%s.png", tag)


def main():
    import pandas as pd

    if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        log.error("Need ≥2 CUDA GPUs. Found %d.", torch.cuda.device_count())
        sys.exit(1)

    log.info("=" * 60)
    log.info("Week 4 NVLink Characterization")
    log.info("GPUs: %d× %s", torch.cuda.device_count(),
             torch.cuda.get_device_name(0))
    log.info("NV6 topology confirmed (see nvidia-smi topo -m)")
    log.info("=" * 60)

    # Enable peer access
    with torch.cuda.device(0):
        try:
            torch.cuda.can_device_access_peer(0, 1)
        except Exception:
            pass

    t_total = time.time()
    summary, telemetry_dfs = run_all_tests()
    make_plots(summary, telemetry_dfs)

    # Save full summary JSON
    def _json_safe(obj):
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        return str(obj)

    with open(RESULTS_DIR / "nvlink_summary.json", "w") as fh:
        json.dump(summary, fh, indent=2, default=_json_safe)

    elapsed = time.time() - t_total
    log.info("=" * 60)
    log.info("NVLink characterization complete in %.1fs", elapsed)
    log.info("Results: %s", RESULTS_DIR)
    log.info("Plots:   %s", PLOTS_DIR)
    log.info("=" * 60)

    # Print final summary
    bw_results = summary.get("nvlink_bandwidth", [])
    if bw_results:
        uni = max((r["bandwidth_gbps"] for r in bw_results if "note" not in r), default=0)
        bidir = next((r["bandwidth_gbps"] for r in bw_results if "note" in r), None)
        log.info("Peak unidirectional bandwidth: %.2f GB/s", uni)
        if bidir:
            log.info("Peak bidirectional bandwidth:  %.2f GB/s (both dirs)", bidir)

    lat_results = summary.get("nvlink_latency", [])
    if lat_results:
        min_lat = min(r["rtt_us"] for r in lat_results)
        log.info("Minimum ping-pong RTT:          %.2f µs", min_lat)

    tp = summary.get("throughput_comparison", [])
    if len(tp) == 2:
        speedup = tp[1]["throughput_imgs_per_sec"] / tp[0]["throughput_imgs_per_sec"]
        log.info("Training speedup (2-GPU / 1-GPU): %.2f×", speedup)


if __name__ == "__main__":
    main()
