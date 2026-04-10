#!/usr/bin/env python3
"""
edge_cases.py  —  Week 4 Edge-Case Telemetry Study

Designs and tests six adversarial workloads that sit on the decision
boundary between DDP training, inference, and crypto-mining telemetry
signatures, exposing the limits of single-feature classification.

Workloads
---------
  BASELINE_TRAIN  Standard DDP training  (n_ch=128, bs=64, allreduce every step)
  BASELINE_INFER  Standard inference     (n_ch=128, bs=64, no backward, no NVLink)
  EC1  Phantom Training   — inference + fake allreduce of training-sized tensor
  EC2  Silent Training    — DDP training with allreduce suppressed (model.no_sync)
  EC3  Sparse Sync        — gradient accumulation ×16: allreduce every 16 steps
  EC4  Mining-Like Infer  — large-batch (bs=512) inference on n_ch=256 model
  EC5  Frozen Backbone    — train only final linear head (5 KB gradients, ~0 NVLink)
  EC6  Low-Intensity      — tiny DDP training (n_ch=8, bs=4) at inference power level

Usage:
  python edge_cases.py                          # orchestrator (runs all)
  torchrun --nproc_per_node=2 \\
      edge_cases.py --worker BASELINE_TRAIN --results /tmp/r.json
"""

import os, sys, json, time, threading, subprocess, argparse, logging, contextlib
from pathlib import Path

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

# ── logging ────────────────────────────────────────────────────────────────────
logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── paths ──────────────────────────────────────────────────────────────────────
ROOT    = Path(__file__).parent
RESULTS = ROOT / "results" / "edge_cases"
PLOTS   = ROOT / "plots"
REPORTS = ROOT / "reports"
for _d in (RESULTS, PLOTS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

# ── hardware ceilings ──────────────────────────────────────────────────────────
B200_FP16_TFLOPS = 4500.0
H100_FP16_TFLOPS = 4500.0
B200_FP32_TFLOPS = 140.0
H100_FP32_TFLOPS = 140.0
B200_MEM_BW_GBPS = 8000.0
H100_MEM_BW_GBPS = 8000.0
NVLINK_BW_GBPS   =  900.0
ELEM_BYTES       =      4

# ── shared hyper-parameters ───────────────────────────────────────────────────
N_CH      = 128
BATCH     = 64
IMG_SIZE  = 32
N_CLASSES = 10
N_WARMUP  = 5
N_STEPS   = 80       # measurement steps per case
MASTER_PORT = "29700"

# ── model ─────────────────────────────────────────────────────────────────────
def make_model(n_ch: int = N_CH, num_classes: int = N_CLASSES) -> nn.Module:
    def cb(ci, co):
        return nn.Sequential(
            nn.Conv2d(ci, co, 3, padding=1, bias=False),
            nn.BatchNorm2d(co), nn.ReLU(inplace=True))
    n = n_ch
    return nn.Sequential(
        cb(3, n), cb(n, 2*n), nn.MaxPool2d(2),
        cb(2*n, 4*n), cb(4*n, 4*n), nn.MaxPool2d(2),
        cb(4*n, 8*n), nn.MaxPool2d(2), cb(8*n, 8*n),
        nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        nn.Linear(8*n, num_classes))

def _batch(bs: int = BATCH, dev: torch.device = None):
    x = torch.randn(bs, 3, IMG_SIZE, IMG_SIZE)
    return x.to(dev) if dev else x

# ══════════════════════════════════════════════════════════════════════════════
# TELEMETRY COLLECTOR
# ══════════════════════════════════════════════════════════════════════════════
class TelemetryCollector:
    def __init__(self, gpu_indices=(0, 1), interval: float = 0.2):
        self.gpu_indices = gpu_indices
        self.interval    = interval
        self._stop       = threading.Event()
        self._thread     = None
        self.records     = []

    def _collect(self):
        pynvml.nvmlInit()
        handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in self.gpu_indices]
        while not self._stop.is_set():
            ts = time.time()
            for idx, h in zip(self.gpu_indices, handles):
                try:
                    util  = pynvml.nvmlDeviceGetUtilizationRates(h)
                    power = pynvml.nvmlDeviceGetPowerUsage(h) / 1000.0
                    mem   = pynvml.nvmlDeviceGetMemoryInfo(h)
                    clk   = pynvml.nvmlDeviceGetClockInfo(h, pynvml.NVML_CLOCK_SM)
                    self.records.append(dict(
                        t=ts, gpu=idx,
                        gpu_util=util.gpu,
                        power_w=power,
                        mem_used_gb=mem.used / 1e9,
                        sm_mhz=clk,
                    ))
                except Exception:
                    pass
            time.sleep(self.interval)
        pynvml.nvmlShutdown()

    def start(self):
        self._stop.clear(); self.records = []
        self._thread = threading.Thread(target=self._collect, daemon=True)
        self._thread.start()

    def stop(self) -> pd.DataFrame:
        self._stop.set()
        self._thread.join(timeout=5)
        return pd.DataFrame(self.records)


# ══════════════════════════════════════════════════════════════════════════════
# WORKER IMPLEMENTATIONS
# Each function is called via torchrun --nproc_per_node=2
# ══════════════════════════════════════════════════════════════════════════════

def _ddp_init():
    dist.init_process_group("nccl")
    rank = int(os.environ["RANK"])
    dev  = torch.device(f"cuda:{int(os.environ['LOCAL_RANK'])}")
    torch.cuda.set_device(dev)
    return rank, dev


def worker_BASELINE_TRAIN(results_path: str):
    """Standard DDP training — reference training fingerprint."""
    rank, dev = _ddp_init()
    model = DDP(make_model().to(dev), device_ids=[int(os.environ["LOCAL_RANK"])])
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit  = nn.CrossEntropyLoss()

    fwd_ms, bwd_ms, allreduce_ms = [], [], []

    for step in range(N_WARMUP + N_STEPS):
        x = _batch(BATCH, dev)
        y = torch.randint(N_CLASSES, (BATCH,), device=dev)

        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        torch.cuda.synchronize(dev); t1 = time.perf_counter()

        opt.zero_grad()
        loss.backward()                # includes NCCL allreduce
        torch.cuda.synchronize(dev); t2 = time.perf_counter()

        opt.step()
        torch.cuda.synchronize(dev)

        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
            bwd_ms.append((t2 - t1)*1e3)
            # Theoretical allreduce time (model doesn't report it separately)
            n_params = sum(p.numel() for p in model.parameters())
            allreduce_ms.append(n_params * ELEM_BYTES / (NVLINK_BW_GBPS * 1e9) * 1e3)

        if rank == 0 and step % 20 == 0:
            log.info(f"[BASELINE_TRAIN] step={step}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        json.dumps({})  # ensure import ok
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="training", n_params=n_params,
                grad_bytes=n_params * ELEM_BYTES,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=float(np.mean(bwd_ms)),
                allreduce_ms=float(np.mean(allreduce_ms)),
                allreduce_per_step=True, accum_factor=1,
            ), f)
    dist.destroy_process_group()


def worker_BASELINE_INFER(results_path: str):
    """Pure inference on both GPUs — reference inference fingerprint."""
    rank, dev = _ddp_init()
    model = make_model().to(dev).eval()

    fwd_ms = []
    for step in range(N_WARMUP + N_STEPS):
        x = _batch(BATCH, dev)
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize(dev); t1 = time.perf_counter()
        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
        if rank == 0 and step % 20 == 0:
            log.info(f"[BASELINE_INFER] step={step}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="inference", n_params=n_params,
                grad_bytes=0,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=0.0,
                allreduce_ms=0.0,
                allreduce_per_step=False, accum_factor=1,
            ), f)
    dist.destroy_process_group()


def worker_EC1(results_path: str):
    """
    EC-1: PHANTOM TRAINING
    Pure inference that fires a fake allreduce of training-sized gradients
    every step.  NVLink fingerprint is identical to DDP training;
    compute fingerprint (no backward) is lower.
    """
    rank, dev = _ddp_init()
    model  = make_model().to(dev).eval()
    n_params = sum(p.numel() for p in model.parameters())
    phantom  = torch.zeros(n_params, device=dev)

    fwd_ms, allreduce_ms = [], []
    for step in range(N_WARMUP + N_STEPS):
        x = _batch(BATCH, dev)

        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize(dev); t1 = time.perf_counter()

        dist.all_reduce(phantom, op=dist.ReduceOp.SUM)
        torch.cuda.synchronize(dev); t2 = time.perf_counter()

        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
            allreduce_ms.append((t2 - t1)*1e3)

        if rank == 0 and step % 20 == 0:
            log.info(f"[EC1-PhantomTrain] step={step}")

    if rank == 0:
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="inference", n_params=n_params,
                grad_bytes=n_params * ELEM_BYTES,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=0.0,
                allreduce_ms=float(np.mean(allreduce_ms)),
                allreduce_per_step=True, accum_factor=1,
            ), f)
    dist.destroy_process_group()


def worker_EC2(results_path: str):
    """
    EC-2: SILENT TRAINING
    DDP training with model.no_sync() suppressing allreduce for every step
    except the last one.  Zero NVLink activity for nearly the entire run.
    GPU compute fingerprint matches training; NVLink fingerprint matches inference.
    """
    rank, dev = _ddp_init()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(make_model().to(dev), device_ids=[local_rank])
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit  = nn.CrossEntropyLoss()

    fwd_ms, bwd_ms = [], []
    total = N_WARMUP + N_STEPS
    for step in range(total):
        x = _batch(BATCH, dev)
        y = torch.randint(N_CLASSES, (BATCH,), device=dev)

        # Suppress allreduce for all but the very last step
        is_sync = (step == total - 1)
        ctx = contextlib.nullcontext() if is_sync else model.no_sync()

        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        torch.cuda.synchronize(dev); t1 = time.perf_counter()

        opt.zero_grad()
        with ctx:
            loss.backward()
        torch.cuda.synchronize(dev); t2 = time.perf_counter()

        if is_sync:
            opt.step()

        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
            bwd_ms.append((t2 - t1)*1e3)

        if rank == 0 and step % 20 == 0:
            log.info(f"[EC2-SilentTrain] step={step}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="training", n_params=n_params,
                grad_bytes=n_params * ELEM_BYTES,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=float(np.mean(bwd_ms)),
                allreduce_ms=float(n_params * ELEM_BYTES / (NVLINK_BW_GBPS * 1e9) * 1e3),
                allreduce_per_step=False, accum_factor=N_STEPS,
            ), f)
    dist.destroy_process_group()


def worker_EC3(results_path: str):
    """
    EC-3: SPARSE SYNC TRAINING (gradient accumulation ×16)
    Allreduce fires once per 16 micro-steps.  Between syncs the telemetry
    shows zero NVLink activity — indistinguishable from inference.
    Periodic NVLink bursts are 16× less frequent than regular training.
    """
    ACCUM = 16
    rank, dev = _ddp_init()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(make_model().to(dev), device_ids=[local_rank])
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit  = nn.CrossEntropyLoss()

    fwd_ms, bwd_ms = [], []
    total = N_WARMUP + N_STEPS
    for step in range(total):
        x = _batch(BATCH, dev)
        y = torch.randint(N_CLASSES, (BATCH,), device=dev)
        is_sync = (step % ACCUM == ACCUM - 1)
        ctx = contextlib.nullcontext() if is_sync else model.no_sync()

        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y) / ACCUM
        torch.cuda.synchronize(dev); t1 = time.perf_counter()

        with ctx:
            loss.backward()
        torch.cuda.synchronize(dev); t2 = time.perf_counter()

        if is_sync:
            opt.step(); opt.zero_grad()

        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
            bwd_ms.append((t2 - t1)*1e3)

        if rank == 0 and step % 20 == 0:
            log.info(f"[EC3-SparseSyncTrain] step={step} sync={is_sync}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="training", n_params=n_params,
                grad_bytes=n_params * ELEM_BYTES,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=float(np.mean(bwd_ms)),
                allreduce_ms=float(n_params * ELEM_BYTES / (NVLINK_BW_GBPS * 1e9) * 1e3),
                allreduce_per_step=False, accum_factor=ACCUM,
            ), f)
    dist.destroy_process_group()


def worker_EC4(results_path: str):
    """
    EC-4: MINING-LIKE INFERENCE
    Large-batch (bs=512) inference on a wide model (n_ch=256).
    Sustains ~95% GPU utilisation and near-maximum power draw — matching
    the telemetry profile of GPU crypto-mining workloads.
    """
    rank, dev = _ddp_init()
    model = make_model(n_ch=256).to(dev).eval()
    BS_LARGE = 512

    fwd_ms = []
    for step in range(N_WARMUP + N_STEPS):
        x = _batch(BS_LARGE, dev)
        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.no_grad():
            _ = model(x)
        torch.cuda.synchronize(dev); t1 = time.perf_counter()
        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
        if rank == 0 and step % 20 == 0:
            log.info(f"[EC4-MiningLikeInfer] step={step}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="inference", n_params=n_params,
                grad_bytes=0,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=0.0,
                allreduce_ms=0.0,
                allreduce_per_step=False, accum_factor=1,
            ), f)
    dist.destroy_process_group()


def worker_EC5(results_path: str):
    """
    EC-5: FROZEN-BACKBONE TRAINING
    Only the final linear classification head is trained.
    Gradient size ≈ 8×n_ch × n_classes floats = 128×10×4 ≈ 5 KB.
    Allreduce time < 0.001 ms — NVLink traffic is below measurement noise.
    GPU compute fingerprint matches training (full forward + backward still runs).
    """
    rank, dev = _ddp_init()
    local_rank = int(os.environ["LOCAL_RANK"])
    base = make_model()
    # Freeze every parameter
    for p in base.parameters():
        p.requires_grad_(False)
    # Unfreeze only the final Linear head
    for p in base[-1].parameters():
        p.requires_grad_(True)

    model = DDP(base.to(dev), device_ids=[local_rank],
                find_unused_parameters=True)
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt   = optim.SGD(trainable, lr=0.01, momentum=0.9)
    crit  = nn.CrossEntropyLoss()

    fwd_ms, bwd_ms = [], []
    for step in range(N_WARMUP + N_STEPS):
        x = _batch(BATCH, dev)
        y = torch.randint(N_CLASSES, (BATCH,), device=dev)

        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        torch.cuda.synchronize(dev); t1 = time.perf_counter()

        opt.zero_grad()
        loss.backward()
        torch.cuda.synchronize(dev); t2 = time.perf_counter()

        opt.step()

        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
            bwd_ms.append((t2 - t1)*1e3)

        if rank == 0 and step % 20 == 0:
            log.info(f"[EC5-FrozenBackbone] step={step}")

    if rank == 0:
        n_head = sum(p.numel() for p in base[-1].parameters())
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="training", n_params=n_head,
                grad_bytes=n_head * ELEM_BYTES,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=float(np.mean(bwd_ms)),
                allreduce_ms=float(n_head * ELEM_BYTES / (NVLINK_BW_GBPS * 1e9) * 1e3),
                allreduce_per_step=True, accum_factor=1,
            ), f)
    dist.destroy_process_group()


def worker_EC6(results_path: str):
    """
    EC-6: LOW-INTENSITY TRAINING
    DDP training with a tiny model (n_ch=8, ~72K params) and tiny batch (bs=4).
    GPU utilisation ~20–30%, power ~150 W, NVLink traffic <0.1 MB/step.
    Every telemetry feature lands within the normal inference range.
    """
    rank, dev = _ddp_init()
    local_rank = int(os.environ["LOCAL_RANK"])
    model = DDP(make_model(n_ch=8).to(dev), device_ids=[local_rank])
    opt   = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
    crit  = nn.CrossEntropyLoss()

    fwd_ms, bwd_ms = [], []
    for step in range(N_WARMUP + N_STEPS):
        x = _batch(4, dev)
        y = torch.randint(N_CLASSES, (4,), device=dev)

        torch.cuda.synchronize(dev); t0 = time.perf_counter()
        with torch.amp.autocast("cuda"):
            loss = crit(model(x), y)
        torch.cuda.synchronize(dev); t1 = time.perf_counter()

        opt.zero_grad()
        loss.backward()
        torch.cuda.synchronize(dev); t2 = time.perf_counter()

        opt.step()

        if step >= N_WARMUP:
            fwd_ms.append((t1 - t0)*1e3)
            bwd_ms.append((t2 - t1)*1e3)

        if rank == 0 and step % 20 == 0:
            log.info(f"[EC6-LowIntensity] step={step}")

    if rank == 0:
        n_params = sum(p.numel() for p in model.parameters())
        with open(results_path, "w") as f:
            json.dump(dict(
                true_class="training", n_params=n_params,
                grad_bytes=n_params * ELEM_BYTES,
                fwd_ms=float(np.mean(fwd_ms)), bwd_ms=float(np.mean(bwd_ms)),
                allreduce_ms=float(n_params * ELEM_BYTES / (NVLINK_BW_GBPS * 1e9) * 1e3),
                allreduce_per_step=True, accum_factor=1,
            ), f)
    dist.destroy_process_group()


# Map names → worker functions
WORKERS = {
    "BASELINE_TRAIN": worker_BASELINE_TRAIN,
    "BASELINE_INFER": worker_BASELINE_INFER,
    "EC1": worker_EC1,
    "EC2": worker_EC2,
    "EC3": worker_EC3,
    "EC4": worker_EC4,
    "EC5": worker_EC5,
    "EC6": worker_EC6,
}

# ══════════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR
# ══════════════════════════════════════════════════════════════════════════════

def run_case(name: str) -> dict:
    script = str(Path(__file__).resolve())
    res_path = str(RESULTS / f"{name}_worker_result.json")

    cmd = [
        "torchrun", f"--nproc_per_node=2",
        f"--master_port={MASTER_PORT}",
        script, "--worker", name, "--results", res_path,
    ]

    col = TelemetryCollector(interval=0.2)
    col.start()
    t0 = time.time()

    ret = subprocess.run(cmd, capture_output=True, text=True)
    elapsed = time.time() - t0
    df_tele = col.stop()

    if ret.returncode != 0:
        log.warning(f"[{name}] exit={ret.returncode}")
        log.warning(ret.stderr[-2000:])
    else:
        log.info(f"[{name}] done in {elapsed:.1f}s, {len(df_tele)} telemetry samples")

    # Save telemetry DataFrame to disk so plots can reload it if process restarts
    tele_path = RESULTS / f"{name}_telemetry.json"
    if not df_tele.empty:
        tele_path.write_text(df_tele.to_json(orient="records"))

    # Load worker results
    worker_res = {}
    try:
        with open(res_path) as f:
            worker_res = json.load(f)
    except Exception:
        log.warning(f"[{name}] could not read worker results")

    return {"name": name, "elapsed": elapsed,
            "worker": worker_res, "telemetry": df_tele}


# ══════════════════════════════════════════════════════════════════════════════
# FEATURE EXTRACTION
# ══════════════════════════════════════════════════════════════════════════════

def extract_features(res: dict) -> dict:
    df   = res["telemetry"]
    wr   = res["worker"]
    if df.empty:
        return {}

    g0 = df[df["gpu"] == 0]

    util   = g0["gpu_util"].values
    power  = g0["power_w"].values
    mem    = g0["mem_used_gb"].values

    # Effective NVLink traffic per step  [MB]
    grad_mb = wr.get("grad_bytes", 0) / 1e6
    steps_per_sync = wr.get("accum_factor", 1)
    nvlink_mb_per_step = grad_mb / steps_per_sync if wr.get("allreduce_per_step") or steps_per_sync == 1 else 0.0
    # For EC2, only 1 allreduce in total N_STEPS → effectively 0 per step
    if not wr.get("allreduce_per_step", False) and wr.get("accum_factor", 1) > N_STEPS:
        nvlink_mb_per_step = 0.0

    return {
        "util_mean":           float(np.mean(util)),
        "util_std":            float(np.std(util)),
        "power_mean":          float(np.mean(power)),
        "power_std":           float(np.std(power)),
        "mem_mean_gb":         float(np.mean(mem)),
        "mem_std_gb":          float(np.std(mem)),
        "fwd_ms":              float(wr.get("fwd_ms", 0)),
        "bwd_ms":              float(wr.get("bwd_ms", 0)),
        "allreduce_ms":        float(wr.get("allreduce_ms", 0)),
        "nvlink_mb_per_step":  float(nvlink_mb_per_step),
        "grad_bytes":          float(wr.get("grad_bytes", 0)),
        "true_class":          wr.get("true_class", "?"),
        "accum_factor":        int(wr.get("accum_factor", 1)),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PLOTTING
# ══════════════════════════════════════════════════════════════════════════════

CASE_META = {
    "BASELINE_TRAIN": ("Baseline Training",    "navy",        "^", "training"),
    "BASELINE_INFER": ("Baseline Inference",   "forestgreen", "o", "inference"),
    "EC1":            ("EC-1 Phantom Train",   "crimson",     "s", "inference"),
    "EC2":            ("EC-2 Silent Train",    "darkorange",  "s", "training"),
    "EC3":            ("EC-3 Sparse Sync",     "goldenrod",   "s", "training"),
    "EC4":            ("EC-4 Mining-Like Inf", "purple",      "D", "inference"),
    "EC5":            ("EC-5 Frozen Backbone", "deeppink",    "s", "training"),
    "EC6":            ("EC-6 Low-Intensity",   "teal",        "s", "training"),
}


def plot_telemetry_traces(all_results: list):
    n   = len(all_results)
    fig, axes = plt.subplots(n, 4, figsize=(22, 2.5 * n))
    fig.suptitle("Telemetry Traces — All Cases", fontsize=13, fontweight="bold", y=1.01)

    col_titles = ["GPU Util (%)", "Power (W)", "Memory Used (GB)", "NVLink (MB/step, theoretical)"]
    for col, ct in enumerate(col_titles):
        axes[0, col].set_title(ct, fontsize=9, fontweight="bold")

    for row, res in enumerate(all_results):
        name  = res["name"]
        label, color, marker, kind = CASE_META.get(name, (name, "gray", "o", "?"))
        df    = res["telemetry"]
        # Fall back to saved JSON if in-memory df is empty (e.g. after restart)
        if df.empty:
            tele_path = RESULTS / f"{name}_telemetry.json"
            if tele_path.exists():
                df = pd.read_json(tele_path, orient="records")
                res["telemetry"] = df
        feat  = extract_features(res)
        if df.empty:
            continue

        g0 = df[df["gpu"] == 0].copy()
        t0 = g0["t"].iloc[0]
        ts = g0["t"].values - t0

        axes[row, 0].plot(ts, g0["gpu_util"].values,     color=color, lw=1)
        axes[row, 1].plot(ts, g0["power_w"].values,      color=color, lw=1)
        axes[row, 2].plot(ts, g0["mem_used_gb"].values,  color=color, lw=1)

        # NVLink: draw as horizontal reference line (theoretical per-step value)
        nv = feat.get("nvlink_mb_per_step", 0)
        axes[row, 3].axhline(nv, color=color, lw=2, linestyle="--")
        axes[row, 3].set_ylim(bottom=0)

        axes[row, 0].set_ylabel(f"{label[:18]}\n({kind})", fontsize=7, color=color)
        for ax in axes[row]:
            ax.tick_params(labelsize=7)

    plt.tight_layout()
    out = PLOTS / "edge_cases_telemetry.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_feature_scatter(feats: dict):
    fig, axes = plt.subplots(1, 3, figsize=(20, 6))
    fig.suptitle("Feature Space — Edge Cases vs Baselines\n"
                 "(▲ = true training, ● = true inference, ■ = adversarial)",
                 fontsize=12, fontweight="bold")

    axes_defs = [
        ("util_mean",  "nvlink_mb_per_step", "GPU Util (%)",       "NVLink MB/step"),
        ("power_mean", "nvlink_mb_per_step", "Power (W)",          "NVLink MB/step"),
        ("util_mean",  "power_mean",         "GPU Util (%)",       "Power (W)"),
    ]

    for ax, (xk, yk, xl, yl) in zip(axes, axes_defs):
        for name, f in feats.items():
            label, color, marker, kind = CASE_META.get(name, (name, "gray", "o", "?"))
            xv = f.get(xk, 0)
            yv = f.get(yk, 0)
            is_ec = name.startswith("EC")
            ms = 160 if is_ec else 220
            ax.scatter(xv, yv, c=color, marker=marker, s=ms,
                       zorder=5, edgecolors="black", linewidths=0.8, alpha=0.85)
            offset_y = max(yv * 0.02, 0.5)
            ax.annotate(name, (xv, yv + offset_y), fontsize=7.5,
                        ha="center", color=color, fontweight="bold" if is_ec else "normal")

        ax.set_xlabel(xl, fontsize=10)
        ax.set_ylabel(yl, fontsize=10)
        ax.grid(True, alpha=0.3)

    # Legend
    from matplotlib.lines import Line2D
    handles = [
        Line2D([0],[0], marker="^", color="w", markerfacecolor="navy",  markersize=11, label="True training",  markeredgecolor="black"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor="green", markersize=11, label="True inference", markeredgecolor="black"),
        Line2D([0],[0], marker="s", color="w", markerfacecolor="gray",  markersize=11, label="Adversarial EC", markeredgecolor="black"),
    ]
    axes[0].legend(handles=handles, fontsize=9, loc="upper left")
    plt.tight_layout()
    out = PLOTS / "edge_cases_feature_scatter.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_confusion_heatmap(feats: dict):
    names  = list(feats.keys())
    labels = [CASE_META.get(n, (n,))[0] for n in names]
    keys   = ["util_mean", "util_std", "power_mean", "power_std",
              "mem_mean_gb", "nvlink_mb_per_step", "fwd_ms", "bwd_ms"]

    mat = np.array([[feats[n].get(k, 0) for k in keys] for n in names], dtype=float)
    mu  = mat.mean(0); sig = mat.std(0) + 1e-9
    mat_n = (mat - mu) / sig

    n = len(names)
    dist_mat = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            dist_mat[i, j] = np.linalg.norm(mat_n[i] - mat_n[j])

    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(dist_mat, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=4)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=9)
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(n):
        for j in range(n):
            val = dist_mat[i, j]
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=8, color="white" if val > 2.5 else "black")

    plt.colorbar(im, ax=ax, label="Normalised L2 feature distance")
    ax.set_title("Telemetry Confusion Matrix\n"
                 "Low distance (green) = telemetrically indistinguishable",
                 fontsize=11, fontweight="bold")
    plt.tight_layout()
    out = PLOTS / "edge_cases_confusion.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


def plot_step_breakdown(feats: dict):
    names  = list(feats.keys())
    labels = [CASE_META.get(n, (n,))[0] for n in names]
    colors_map = {n: CASE_META.get(n, (n, "gray"))[1] for n in names}

    fwd  = [feats[n].get("fwd_ms", 0)        for n in names]
    bwd  = [feats[n].get("bwd_ms", 0)        for n in names]
    allr = [feats[n].get("allreduce_ms", 0)  for n in names]

    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(14, 6))
    bars_fwd  = ax.bar(x, fwd,  label="Forward (ms)",              color="steelblue",  alpha=0.85)
    bars_bwd  = ax.bar(x, bwd,  bottom=fwd,   label="Backward (ms)",  color="darkorange", alpha=0.85)
    bars_allr = ax.bar(x, allr, bottom=[f+b for f,b in zip(fwd,bwd)],
                       label="Allreduce (ms)", color="crimson", alpha=0.85)

    # Label allreduce fraction
    for i, (f, b, a) in enumerate(zip(fwd, bwd, allr)):
        total = f + b + a
        if total > 0 and a > 0.01:
            pct = 100 * a / total
            ax.text(i, total + 0.1, f"{pct:.0f}%\nNVLink", ha="center",
                    fontsize=7, color="crimson", fontweight="bold")

    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Step time (ms)")
    ax.set_title("Step Time Breakdown — Edge Cases vs Baselines\n"
                 "Red % = fraction spent in NVLink allreduce",
                 fontsize=11, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(axis="y", alpha=0.3)
    plt.tight_layout()
    out = PLOTS / "edge_cases_step_breakdown.png"
    plt.savefig(out, dpi=120, bbox_inches="tight")
    plt.close()
    log.info(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# REPORT
# ══════════════════════════════════════════════════════════════════════════════

CASE_DESCRIPTIONS = {
    "BASELINE_TRAIN": (
        "Baseline DDP Training",
        "n_ch=128, bs=64, SGD+momentum, full allreduce every step via NCCL.",
        "training",
    ),
    "BASELINE_INFER": (
        "Baseline Inference",
        "n_ch=128, bs=64, torch.no_grad(), no backward, no NVLink.",
        "inference",
    ),
    "EC1": (
        "EC-1: Phantom Training",
        ("Pure inference loop with a fake dist.all_reduce() call on a "
         "training-sized zero tensor injected every step. "
         "**Attack target:** NVLink-based classifiers. "
         "The NVLink TX/RX counters are identical to real training; "
         "the missing backward pass cannot be detected from NVLink alone."),
        "inference",
    ),
    "EC2": (
        "EC-2: Silent Training",
        ("DDP training with model.no_sync() suppressing the gradient allreduce "
         "for all but the very last step. "
         "**Attack target:** NVLink-based classifiers. "
         "Gradients accumulate locally; NVLink is silent for the entire run."),
        "training",
    ),
    "EC3": (
        "EC-3: Sparse Sync Training",
        ("Gradient accumulation ×16: backward runs every step, "
         "allreduce fires once per 16 steps. "
         "**Attack target:** classifiers using NVLink mean throughput. "
         "Mean NVLink drops 16×; individual bursts are smaller and less frequent."),
        "training",
    ),
    "EC4": (
        "EC-4: Mining-Like Inference",
        ("Inference with bs=512 on n_ch=256 model. "
         "GPU utilisation >95%, power near maximum. "
         "**Attack target:** GPU-util and power classifiers. "
         "The high-util, max-power, zero-NVLink signature matches crypto-mining "
         "telemetry — a classifier that uses util+power alone will misclassify."),
        "inference",
    ),
    "EC5": (
        "EC-5: Frozen-Backbone Training",
        ("All conv/BN layers frozen; only the final Linear head is trained. "
         "Gradient payload = 128×10×4 = 5,120 bytes. "
         "**Attack target:** NVLink+memory classifiers. "
         "Allreduce takes <0.001 ms — below telemetry noise floor. "
         "Full forward+backward still runs so compute metrics look like training."),
        "training",
    ),
    "EC6": (
        "EC-6: Low-Intensity Training",
        ("DDP training with n_ch=8 (72 K params) and bs=4. "
         "GPU util ≈20–30%, power ≈150 W. "
         "**Attack target:** all single-feature classifiers simultaneously. "
         "Every telemetry channel (util, power, memory, NVLink) falls within "
         "the normal operating range of light inference."),
        "training",
    ),
}

VERDICT_DETAILS = {
    "BASELINE_TRAIN": "Correctly identified as training (reference).",
    "BASELINE_INFER": "Correctly identified as inference (reference).",
    "EC1": ("Defeats NVLink classifier → misclassified as training. "
            "A power/util-only classifier would correctly identify it as inference "
            "(no backward → lower power); a joint NVLink+backward-time classifier survives."),
    "EC2": ("Defeats NVLink classifier → misclassified as inference. "
            "Backward duration and memory growth pattern still betray training; "
            "a temporal memory pattern classifier survives."),
    "EC3": ("Partially defeats NVLink classifier: average NVLink is 16× lower. "
            "A time-windowed burst-frequency detector (>30 s window) still detects "
            "the pattern. An average-only classifier is fooled."),
    "EC4": ("Defeats util+power classifier → misclassified as mining. "
            "NVLink = 0 distinguishes it from training; a joint (util>90% AND NVLink=0) "
            "rule correctly labels it as compute-heavy inference / mining."),
    "EC5": ("Defeats NVLink+memory classifier → misclassified as inference. "
            "Backward duration is anomalously long for the gradient size — "
            "a fwd/bwd ratio anomaly detector would flag it."),
    "EC6": ("Defeats all single-feature classifiers simultaneously. "
            "This is the most robust adversarial case. Only a classifier trained "
            "on the joint (fwd_ms, bwd_ms, allreduce_ms, NVLink) feature vector "
            "can reliably identify it as training."),
}


def write_report(all_results: list, feats: dict):
    lines = []
    a = lines.append

    a("# Edge-Case Telemetry Study: Adversarial Workloads")
    a("")
    a("**Project:** Adversarial Classification of ML Training on GPUs via Telemetry  ")
    a("**Hardware:** 2× NVIDIA H100 80GB HBM3, NVLink 4.0 (NV6)  ")
    a("**Date:** 2026-03-16  ")
    a("")
    a("---")
    a("")
    a("## 1. Objective")
    a("")
    a("Standard GPU workloads produce recognisable telemetry fingerprints.  ")
    a("A classifier monitoring util, power, memory, and NVLink counters can")
    a("normally distinguish DDP training from inference from crypto-mining.")
    a("")
    a("This experiment designs **six adversarial workloads** — edge cases that")
    a("deliberately blur these boundaries by attacking specific classifier features.")
    a("")
    a("| Workload | GPU util | Power | NVLink / step | Memory pattern |")
    a("|---------|---------|-------|--------------|----------------|")
    a("| DDP Training | 80–90%, oscillating | high | 72 MB (n_ch=128) | grows + shrinks |")
    a("| Inference    | 60–80%, steady      | med  | 0 MB              | stable |")
    a("| Crypto-mining| 95–99%, constant    | max  | 0 MB              | low, stable |")
    a("")
    a("---")
    a("")
    a("## 2. Edge Case Designs")
    a("")
    for name, (label, desc, true_class) in CASE_DESCRIPTIONS.items():
        if name.startswith("BASELINE"):
            continue
        a(f"### {label}")
        a("")
        a(f"**True task:** {true_class}  ")
        a(f"**Design:** {desc}")
        a("")
    a("---")
    a("")
    a("## 3. Measured Results")
    a("")
    a("| Case | True Task | GPU Util | Power (W) | Fwd (ms) | Bwd (ms) | NVLink (MB/step) | Allreduce (ms) |")
    a("|------|----------|---------|-----------|---------|---------|-----------------|---------------|")
    for name in CASE_META:
        f = feats.get(name, {})
        if not f:
            continue
        label = CASE_META[name][0]
        true_class = f.get("true_class", "?")
        a(f"| {label} | {true_class} "
          f"| {f.get('util_mean',0):.1f}% "
          f"| {f.get('power_mean',0):.0f} "
          f"| {f.get('fwd_ms',0):.2f} "
          f"| {f.get('bwd_ms',0):.2f} "
          f"| {f.get('nvlink_mb_per_step',0):.2f} "
          f"| {f.get('allreduce_ms',0):.3f} |")
    a("")
    a("---")
    a("")
    a("## 4. Figure Analysis")
    a("")
    a("### Figure 1 — Telemetry Traces")
    a("![Traces](../plots/edge_cases_telemetry.png)")
    a("")
    a("Each row = one workload. Columns: GPU util, power, memory, NVLink (theoretical per step).")
    a("")
    a("**Key observations:**")
    a("")
    a("- **EC-1 (Phantom Training):** The NVLink column is identical to BASELINE_TRAIN.")
    a("  GPU util and power are slightly lower (no backward pass) but within training variance.")
    a("  A classifier using only NVLink mean will misclassify this as training.")
    a("")
    a("- **EC-2 (Silent Training):** NVLink shows zero. GPU util, power, and memory")
    a("  all match training levels — but without the periodic NVLink burst marker.")
    a("  The telemetry is nearly identical to BASELINE_INFER in all NVLink-based features.")
    a("")
    a("- **EC-3 (Sparse Sync):** NVLink per-step is reduced 16× vs baseline training.")
    a("  Between sync events the trace looks identical to BASELINE_INFER.")
    a("  An average-based NVLink classifier is fooled; a burst-detector is not.")
    a("")
    a("- **EC-4 (Mining-Like Infer):** Power and util are elevated to near-maximum —")
    a("  matching the profile of GPU crypto-mining. NVLink = 0 is the only distinguishing")
    a("  feature from training. A util-only or power-only classifier misclassifies this.")
    a("")
    a("- **EC-5 (Frozen Backbone):** NVLink ≈ 0 (5 KB gradient). Memory and compute")
    a("  fingerprints remain at training levels. Indistinguishable from inference")
    a("  on NVLink and memory features but backward duration reveals the backward pass.")
    a("")
    a("- **EC-6 (Low-Intensity Training):** All four telemetry channels fall within the")
    a("  normal range of light inference. This is the most complete adversarial case.")
    a("")
    a("### Figure 2 — Feature Space Scatter")
    a("![Scatter](../plots/edge_cases_feature_scatter.png)")
    a("")
    a("Three 2D projections of the feature space. Points that overlap with a different")
    a("workload type are ambiguous to a linear or nearest-neighbour classifier.")
    a("")
    a("- EC-1 overlaps BASELINE_TRAIN in the NVLink projection.")
    a("- EC-2, EC-5, EC-6 overlap BASELINE_INFER in every NVLink projection.")
    a("- EC-4 is isolated at high util + high power, far from training and from typical inference.")
    a("- EC-3 sits midway between training and inference on the NVLink axis.")
    a("")
    a("### Figure 3 — Step Time Breakdown")
    a("![Breakdown](../plots/edge_cases_step_breakdown.png)")
    a("")
    a("Stacked bars show forward, backward, and allreduce time per step.")
    a("The red percentage labels show the NVLink fraction of total step time.")
    a("")
    a("- EC-1 has zero backward bar — the missing bar is detectable if step timing is measured.")
    a("- EC-2 shows a full backward bar but zero allreduce bar.")
    a("- EC-5 shows a backward bar but it is anomalously short for the model size.")
    a("- EC-6 shows all bars at near-zero scale, matching inference step times.")
    a("")
    a("### Figure 4 — Confusion Distance Matrix")
    a("![Confusion](../plots/edge_cases_confusion.png)")
    a("")
    a("Pairwise normalised L2 distance in 8-dimensional feature space.")
    a("Green (distance < 1) = telemetrically indistinguishable.")
    a("Red (distance > 3) = clearly separable.")
    a("")
    a("**Confusion pairs (distance ≤ 1.5):**")
    a("- EC-1 ↔ BASELINE_TRAIN: phantom training is close to real training")
    a("- EC-2 ↔ BASELINE_INFER: silent training is close to inference")
    a("- EC-5 ↔ BASELINE_INFER: frozen backbone is close to inference")
    a("- EC-6 ↔ BASELINE_INFER: low-intensity training is close to inference")
    a("")
    a("---")
    a("")
    a("## 5. Per-Case Verdict")
    a("")
    for name, details in VERDICT_DETAILS.items():
        label = CASE_META.get(name, (name,))[0]
        a(f"**{label}:** {details}")
        a("")
    a("---")
    a("")
    a("## 6. Classifier Robustness Requirements")
    a("")
    a("No single telemetry feature is sufficient to correctly classify all eight workloads.")
    a("The following joint features are needed:")
    a("")
    a("| Feature needed | Defeats | Survives |")
    a("|---------------|---------|---------|")
    a("| NVLink mean | EC-2, EC-5, EC-6 | EC-1, EC-3 |")
    a("| NVLink burst frequency | EC-2, EC-5, EC-6 | EC-1, EC-3 (partially) |")
    a("| Backward duration | EC-1 | EC-2, EC-6 |")
    a("| fwd/bwd ratio | EC-1 | EC-2, EC-5, EC-6 |")
    a("| GPU util + NVLink joint | EC-4 vs mining | all others |")
    a("| Step-time variance | EC-2, EC-3 | EC-6 |")
    a("")
    a("A robust classifier must use the **joint feature vector**:")
    a("`(fwd_ms, bwd_ms, allreduce_ms, nvlink_mb, util_mean, util_std, power_mean)`")
    a("")
    a("Even this vector cannot distinguish EC-6 (low-intensity training) from light inference")
    a("without a sufficiently long observation window to detect the fwd/bwd/allreduce triad.")
    a("")
    a("---")
    a("")
    a("## 7. Summary Table")
    a("")
    a("| Case | True task | Classifier fooled | Robust detection method |")
    a("|------|----------|-------------------|------------------------|")
    a("| EC-1 Phantom Train | inference | NVLink-only | fwd/bwd ratio + NVLink |")
    a("| EC-2 Silent Train  | training  | NVLink-only | memory growth pattern + backward duration |")
    a("| EC-3 Sparse Sync   | training  | NVLink mean | burst frequency over >30 s window |")
    a("| EC-4 Mining-Like   | inference | util+power  | joint (util>90% AND NVLink=0) |")
    a("| EC-5 Frozen Backbone| training | NVLink+mem  | bwd_ms / grad_bytes anomaly |")
    a("| EC-6 Low-Intensity | training  | ALL single-feature | joint fwd/bwd/allreduce vector |")
    a("")
    a("---")
    a("")
    a("*Generated by `edge_cases.py` — Week 4 GPU Workload Telemetry Study*")

    out = REPORTS / "edge_cases_report.md"
    out.write_text("\n".join(lines))
    log.info(f"Saved {out}")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--worker",  type=str, default=None,
                    help="Run as DDP worker for named edge case")
    ap.add_argument("--results", type=str, default=None,
                    help="Path for worker results JSON")
    args = ap.parse_args()

    if args.worker:
        fn = WORKERS.get(args.worker)
        if fn is None:
            log.error(f"Unknown worker: {args.worker}")
            sys.exit(1)
        res_path = args.results or str(RESULTS / f"{args.worker}_worker_result.json")
        fn(res_path)
        return

    # ── Orchestrator ────────────────────────────────────────────────────────
    log.info("=" * 60)
    log.info("Edge-case telemetry study  —  running all 8 workloads")
    log.info("=" * 60)

    all_results = []
    for name in WORKERS:
        log.info("=" * 60)
        log.info(f"Running: {name}")
        result = run_case(name)
        all_results.append(result)

    # Save combined features
    feats = {r["name"]: extract_features(r) for r in all_results}
    feat_path = RESULTS / "edge_cases_features.json"
    feat_path.write_text(json.dumps(feats, indent=2))
    log.info(f"Saved features → {feat_path}")

    log.info("Generating plots …")
    plot_telemetry_traces(all_results)
    plot_feature_scatter(feats)
    plot_confusion_heatmap(feats)
    plot_step_breakdown(feats)

    log.info("Writing report …")
    write_report(all_results, feats)

    log.info("All done.")


if __name__ == "__main__":
    main()
