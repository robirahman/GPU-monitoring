#!/venv/main/bin/python
"""
week5/corner_cases/tier1_telemetry.py
=======================================
For each corner-case config that falls within the week-4 DDP training AI
range (45–1483 FLOP/byte), runs a 30-second continuous workload while
sampling pynvml at ~10 Hz.  Computes temporal statistics and compares them
against a training baseline.  Reports which signals trigger false detection
and which discriminate correctly.

Output
------
  results/tier1_traces.json          — raw per-config time-series
  results/tier1_summary.json         — per-config feature deltas vs training
  plots/tier1_*.png                  — temporal trace plots per group
  reports/tier1_analysis.md          — full per-config analysis report
"""

import sys, os, json, time, math, threading, logging, warnings
from pathlib import Path
from collections import defaultdict

sys.path.insert(0, "/tmp/pynvml_pkg")
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pynvml
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

CC_DIR  = Path(__file__).parent
RESULTS = CC_DIR / "results"
PLOTS   = CC_DIR / "plots"
REPORTS = CC_DIR.parent / "reports"
for d in [RESULTS, PLOTS, REPORTS]:
    d.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0")

# ── H100 NVL constants ────────────────────────────────────────────────────────
H100_FP16_TFLOPS = 1979.0
H100_FP32_TFLOPS =   67.0
H100_HBM_GBps    = 3900.0
RIDGE_FP32 = H100_FP32_TFLOPS * 1e12 / (H100_HBM_GBps * 1e9)
RIDGE_FP16 = H100_FP16_TFLOPS * 1e12 / (H100_HBM_GBps * 1e9)
TRAIN_AI_LO, TRAIN_AI_HI = 45.0, 1483.0

# ── pynvml ─────────────────────────────────────────────────────────────────────

def nvml_init():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)


def sample_nvml(handle):
    u  = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_util":    float(u.gpu),
        "mem_util":    float(u.memory),
        "power_w":     float(pw),
        "mem_used_gb": float(mi.used / 1e9),
        "ts":          time.time(),
    }


# ── Continuous sampler ────────────────────────────────────────────────────────

class NvmlSampler:
    """Background thread samples pynvml at ~10 Hz while a workload runs."""

    def __init__(self, handle, interval=0.1):
        self.handle   = handle
        self.interval = interval
        self.samples  = []
        self._stop    = threading.Event()
        self._thread  = threading.Thread(target=self._run, daemon=True)

    def start(self):
        self._stop.clear()
        self.samples = []
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop.set()
        self._thread.join(timeout=2)

    def _run(self):
        while not self._stop.is_set():
            try:
                self.samples.append(sample_nvml(self.handle))
            except Exception:
                pass
            time.sleep(self.interval)


# ── Workload runners (30-second loops) ────────────────────────────────────────

def run_for_duration(fn, duration_s=30, warmup_s=2):
    """Call fn() repeatedly for warmup_s then duration_s, return wall time."""
    t0 = time.time()
    while time.time() - t0 < warmup_s:
        fn()
    torch.cuda.synchronize()
    t1 = time.time()
    while time.time() - t1 < duration_s:
        fn()
    torch.cuda.synchronize()


# ── Model imports (reuse from corner_cases.py) ────────────────────────────────

def make_resnet(arch):
    import torchvision.models as tvm
    return {
        "resnet18":  tvm.resnet18,
        "resnet50":  tvm.resnet50,
        "resnet101": tvm.resnet101,
        "resnet152": tvm.resnet152,
    }[arch](weights=None)


class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, max_len=2048):
        super().__init__()
        self.ln1  = nn.LayerNorm(d_model)
        self.ln2  = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio), nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model))
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())

    def forward(self, x):
        s    = x.size(1)
        mask = self.causal_mask[:s, :s]
        h, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x),
                         attn_mask=mask, need_weights=False)
        x = x + h
        return x + self.mlp(self.ln2(x))


class MiniLLM(nn.Module):
    CONFIGS = {
        "gpt2":   dict(d=768,  n_layer=12, n_head=12),
        "gpt2-m": dict(d=1024, n_layer=24, n_head=16),
        "gpt2-l": dict(d=1280, n_layer=36, n_head=20),
    }
    def __init__(self, name="gpt2", vocab=50257, max_len=2048):
        super().__init__()
        c = self.CONFIGS[name]
        d, n_layer, n_head = c["d"], c["n_layer"], c["n_head"]
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks  = nn.ModuleList([GPT2Block(d, n_head) for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        b, s = ids.shape
        x = self.tok_emb(ids) + self.pos_emb(torch.arange(s, device=ids.device))
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))


class ViT(nn.Module):
    def __init__(self, img_size=224, patch=16, d=768, n_layer=12, n_head=12,
                 n_classes=1000):
        super().__init__()
        n_patches    = (img_size // patch) ** 2
        self.patch_emb = nn.Conv2d(3, d, patch, stride=patch)
        self.cls_tok   = nn.Parameter(torch.randn(1, 1, d))
        self.pos_emb   = nn.Parameter(torch.randn(1, n_patches + 1, d))
        self.blocks    = nn.ModuleList([GPT2Block(d, n_head, max_len=n_patches + 1)
                                        for _ in range(n_layer)])
        self.ln        = nn.LayerNorm(d)
        self.head      = nn.Linear(d, n_classes)

    def forward(self, x):
        b = x.size(0)
        t = self.patch_emb(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls_tok.expand(b, -1, -1), t], 1) + self.pos_emb
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.ln(t[:, 0]))


# ── Tier-1 representative configs ────────────────────────────────────────────
#
# We pick the most dangerous in-range config from each group (the one whose
# snapshot telemetry most resembled training) plus a spread across the AI range.

TIER1_CONFIGS = [
    # ── Training baseline (CC-E full_train) ────────────────────────────────
    {
        "name": "TRAIN/resnet50/bs4/fp16",
        "group": "TRAIN",
        "workload": "training_full",
        "fn_builder": lambda: _build_train_step("resnet50", bs=4, dtype=torch.float16),
    },
    # ── CC-A CNN inference ─────────────────────────────────────────────────
    {
        "name": "CC-A/resnet18/bs1/img224/fp16",
        "group": "CC-A",
        "workload": "cnn_inference",
        "fn_builder": lambda: _build_inference("resnet18", 1, 224, torch.float16),
    },
    {
        "name": "CC-A/resnet50/bs4/img224/fp16",
        "group": "CC-A",
        "workload": "cnn_inference",
        "fn_builder": lambda: _build_inference("resnet50", 4, 224, torch.float16),
    },
    {
        "name": "CC-A/resnet50/bs16/img224/fp16",
        "group": "CC-A",
        "workload": "cnn_inference",
        "fn_builder": lambda: _build_inference("resnet50", 16, 224, torch.float16),
    },
    # ── CC-B LLM prefill ───────────────────────────────────────────────────
    {
        "name": "CC-B/gpt2/bs4/s128/fp16",
        "group": "CC-B",
        "workload": "llm_prefill",
        "fn_builder": lambda: _build_llm_prefill("gpt2", bs=4, seq=128, dtype=torch.float16),
    },
    {
        "name": "CC-B/gpt2-m/bs4/s512/fp32",
        "group": "CC-B",
        "workload": "llm_prefill",
        "fn_builder": lambda: _build_llm_prefill("gpt2-m", bs=4, seq=512, dtype=torch.float32),
    },
    {
        "name": "CC-B/gpt2-l/bs4/s128/fp16",
        "group": "CC-B",
        "workload": "llm_prefill",
        "fn_builder": lambda: _build_llm_prefill("gpt2-l", bs=4, seq=128, dtype=torch.float16),
    },
    # ── CC-C LLM decode (memory-bound, just inside range) ─────────────────
    {
        "name": "CC-C/gpt2-m/bs256/ctx128/fp16",
        "group": "CC-C",
        "workload": "llm_decode",
        "fn_builder": lambda: _build_llm_decode("gpt2-m", bs=256, ctx=128),
    },
    # ── CC-D quantisation ─────────────────────────────────────────────────
    {
        "name": "CC-D/resnet50/bs16/fp32",
        "group": "CC-D",
        "workload": "quant_inference",
        "fn_builder": lambda: _build_inference("resnet50", 16, 224, torch.float32),
    },
    {
        "name": "CC-D/gpt2/bs1/s512/fp32",
        "group": "CC-D",
        "workload": "quant_inference",
        "fn_builder": lambda: _build_llm_prefill("gpt2", bs=1, seq=512, dtype=torch.float32),
    },
    # ── CC-E forward-only (hardest to distinguish) ─────────────────────────
    {
        "name": "CC-E/fwd_only/bs1/fp16",
        "group": "CC-E",
        "workload": "fwd_only",
        "fn_builder": lambda: _build_fwd_only("resnet50", bs=1, dtype=torch.float16),
    },
    {
        "name": "CC-E/fwd_only/bs16/fp16",
        "group": "CC-E",
        "workload": "fwd_only",
        "fn_builder": lambda: _build_fwd_only("resnet50", bs=16, dtype=torch.float16),
    },
    # ── CC-F ViT inference ─────────────────────────────────────────────────
    {
        "name": "CC-F/ViT-B/16/bs1/fp16",
        "group": "CC-F",
        "workload": "vit_inference",
        "fn_builder": lambda: _build_vit("ViT-B/16", bs=1),
    },
    {
        "name": "CC-F/ViT-B/8/bs1/fp16",
        "group": "CC-F",
        "workload": "vit_inference",
        "fn_builder": lambda: _build_vit("ViT-B/8", bs=1),
    },
]


# ── Workload builders ─────────────────────────────────────────────────────────

def _build_inference(arch, bs, img_size, dtype):
    model = make_resnet(arch).to(DEVICE).to(dtype).eval()
    x = torch.randn(bs, 3, img_size, img_size, device=DEVICE, dtype=dtype)
    def fn():
        with torch.no_grad():
            model(x)
    return fn, model


def _build_llm_prefill(name, bs, seq, dtype):
    model = MiniLLM(name).to(DEVICE).to(dtype).eval()
    ids   = torch.randint(0, 50257, (bs, seq), device=DEVICE)
    def fn():
        with torch.no_grad():
            model(ids)
    return fn, model


def _build_llm_decode(name, bs, ctx):
    model = MiniLLM(name).to(DEVICE).to(torch.float16).eval()
    ids   = torch.randint(0, 50257, (bs, 1), device=DEVICE)
    def fn():
        with torch.no_grad():
            model(ids)
    return fn, model


def _build_fwd_only(arch, bs, dtype):
    model = make_resnet(arch).to(DEVICE).to(dtype)
    x = torch.randn(bs, 3, 224, 224, device=DEVICE, dtype=dtype)
    y = torch.randint(0, 1000, (bs,), device=DEVICE)
    def fn():
        with torch.cuda.amp.autocast():
            model(x)   # forward only — no backward
    return fn, model


def _build_train_step(arch, bs, dtype):
    # AMP training: model in fp32, autocast handles fp16 activations
    model = make_resnet(arch).to(DEVICE)   # keep fp32 params for GradScaler
    opt   = torch.optim.SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    x = torch.randn(bs, 3, 224, 224, device=DEVICE, dtype=torch.float32)
    y = torch.randint(0, 1000, (bs,), device=DEVICE)
    def fn():
        opt.zero_grad()
        with torch.cuda.amp.autocast():
            loss = F.cross_entropy(model(x), y)
        scaler.scale(loss).backward()
        scaler.step(opt); scaler.update()
    return fn, model


def _build_vit(variant, bs):
    cfgs = {
        "ViT-S/16": dict(img_size=224, patch=16, d=384,  n_layer=12, n_head=6),
        "ViT-B/16": dict(img_size=224, patch=16, d=768,  n_layer=12, n_head=12),
        "ViT-L/16": dict(img_size=224, patch=16, d=1024, n_layer=24, n_head=16),
        "ViT-B/8":  dict(img_size=224, patch=8,  d=768,  n_layer=12, n_head=12),
    }
    cfg   = cfgs[variant]
    model = ViT(**cfg).to(DEVICE).to(torch.float16).eval()
    x     = torch.randn(bs, 3, cfg["img_size"], cfg["img_size"],
                        device=DEVICE, dtype=torch.float16)
    def fn():
        with torch.no_grad():
            model(x)
    return fn, model


# ── Trace statistics ──────────────────────────────────────────────────────────

def trace_stats(samples):
    if not samples:
        return {}
    keys = ["gpu_util", "mem_util", "power_w", "mem_used_gb"]
    out  = {}
    for k in keys:
        v = np.array([s[k] for s in samples])
        out[f"{k}_mean"] = float(v.mean())
        out[f"{k}_std"]  = float(v.std())
        out[f"{k}_min"]  = float(v.min())
        out[f"{k}_max"]  = float(v.max())
        # Coefficient of variation
        out[f"{k}_cv"]   = float(v.std() / (abs(v.mean()) + 1e-9))
        # Autocorrelation at lag-1
        if len(v) > 3:
            mu  = v.mean(); var = v.var()
            if var > 1e-12:
                out[f"{k}_acf1"] = float(((v[:-1]-mu)*(v[1:]-mu)).mean() / var)
            else:
                out[f"{k}_acf1"] = 0.0
        else:
            out[f"{k}_acf1"] = 0.0
    return out


# ── Main measurement loop ─────────────────────────────────────────────────────

def measure_tier1(handle, cfg, duration_s=30):
    name = cfg["name"]
    log.info(f"  Tier-1: {name} ({duration_s}s) ...")
    try:
        fn, model = cfg["fn_builder"]()
    except Exception as e:
        log.warning(f"  SKIP {name}: {e}")
        return None

    sampler = NvmlSampler(handle)
    sampler.start()
    try:
        run_for_duration(fn, duration_s=duration_s, warmup_s=3)
    except Exception as e:
        log.warning(f"  ERROR during run {name}: {e}")
    finally:
        sampler.stop()

    samples = sampler.samples
    stats   = trace_stats(samples)
    stats["n_samples"] = len(samples)
    stats["name"]      = name
    stats["group"]     = cfg["group"]
    stats["workload"]  = cfg["workload"]

    # Cleanup
    try:
        del model
        torch.cuda.empty_cache()
    except Exception:
        pass

    log.info(f"    util={stats.get('gpu_util_mean',0):.1f}%  "
             f"power={stats.get('power_w_mean',0):.0f}W  "
             f"mem={stats.get('mem_used_gb_mean',0):.1f}GB  "
             f"n={len(samples)}")
    return {"name": name, "group": cfg["group"], "workload": cfg["workload"],
            "stats": stats, "samples": samples}


# ── Plotting ──────────────────────────────────────────────────────────────────

def plot_temporal_traces(all_traces):
    """One figure per group showing temporal traces vs training baseline."""
    train_trace = next((t for t in all_traces if t["group"] == "TRAIN"), None)
    if not train_trace:
        return

    groups = sorted(set(t["group"] for t in all_traces if t["group"] != "TRAIN"))
    metrics = ["gpu_util", "power_w", "mem_used_gb", "mem_util"]
    metric_labels = ["GPU Util (%)", "Power (W)", "Mem Used (GB)", "Mem Util (%)"]

    for grp in groups:
        traces_in_grp = [t for t in all_traces if t["group"] == grp]
        n_traces = len(traces_in_grp)
        if not n_traces:
            continue

        fig, axes = plt.subplots(len(metrics), 1, figsize=(12, 3 * len(metrics)))
        fig.suptitle(f"Tier-1 Temporal Traces: {grp} vs Training Baseline", fontsize=13)

        for ax, metric, mlabel in zip(axes, metrics, metric_labels):
            # Training baseline
            if train_trace["samples"]:
                ts_train = np.array([s["ts"] for s in train_trace["samples"]])
                ts_train -= ts_train[0]
                vals_train = np.array([s[metric] for s in train_trace["samples"]])
                ax.plot(ts_train, vals_train, "k-", lw=2.5, alpha=0.8, label="Training (baseline)")

            colors = plt.cm.tab10(np.linspace(0, 0.7, n_traces))
            for i, t in enumerate(traces_in_grp):
                if not t["samples"]:
                    continue
                ts = np.array([s["ts"] for s in t["samples"]])
                ts -= ts[0]
                vals = np.array([s[metric] for s in t["samples"]])
                short_name = t["name"].replace(f"{grp}/", "")
                ax.plot(ts, vals, "-", color=colors[i], lw=1.5,
                        alpha=0.85, label=short_name)

            ax.set_ylabel(mlabel, fontsize=9)
            ax.set_xlabel("Time (s)" if metric == metrics[-1] else "")
            ax.legend(fontsize=7, loc="upper right", ncol=2)
            ax.grid(True, alpha=0.3)

        plt.tight_layout()
        fname = PLOTS / f"tier1_{grp.replace('/', '_')}.png"
        plt.savefig(fname, dpi=120, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved: {fname.name}")


def plot_signal_comparison(all_traces):
    """Bar chart comparing mean signals across all configs vs training."""
    metrics = ["gpu_util_mean", "power_w_mean", "mem_used_gb_mean",
               "mem_util_mean", "power_w_cv", "gpu_util_acf1"]
    labels  = ["GPU Util Mean (%)", "Power Mean (W)", "Mem Used Mean (GB)",
               "Mem Util Mean (%)", "Power CV (variability)", "GPU Util ACF1"]

    train_rec = next((t for t in all_traces if t["group"] == "TRAIN"), None)
    if not train_rec:
        return

    names    = [t["name"].split("/", 1)[-1] for t in all_traces]
    groups   = [t["group"] for t in all_traces]
    group_colors = {"TRAIN": "#000000", "CC-A": "#2196F3", "CC-B": "#FF9800",
                    "CC-C": "#9C27B0", "CC-D": "#4CAF50",
                    "CC-E": "#F44336", "CC-F": "#00BCD4"}

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for ax, metric, label in zip(axes, metrics, labels):
        vals   = [t["stats"].get(metric, 0) for t in all_traces]
        colors = [group_colors.get(g, "gray") for g in groups]
        y_pos  = range(len(names))
        bars   = ax.barh(y_pos, vals, color=colors, alpha=0.8, edgecolor="white")
        # Highlight training
        ax.barh(0, vals[0], color="black", alpha=1.0)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(names, fontsize=6)
        ax.set_xlabel(label, fontsize=9)
        ax.axvline(vals[0], color="red", ls="--", lw=1.5, alpha=0.7,
                   label=f"Training: {vals[0]:.2f}")
        ax.legend(fontsize=7)
        ax.grid(True, axis="x", alpha=0.3)

    # Legend for groups
    handles = [plt.Rectangle((0, 0), 1, 1, color=c, label=g)
               for g, c in group_colors.items()]
    fig.legend(handles=handles, loc="lower center", ncol=7,
               fontsize=8, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Tier-1 Signal Comparison: In-Range Configs vs Training", fontsize=13)
    plt.tight_layout()
    fname = PLOTS / "tier1_signal_comparison.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved: {fname.name}")


# ── Analysis report ───────────────────────────────────────────────────────────

def write_analysis(all_traces):
    train_rec = next((t for t in all_traces if t["group"] == "TRAIN"), None)
    if not train_rec:
        return
    ts = train_rec["stats"]

    lines = []
    a = lines.append

    a("# Tier-1 Telemetry Analysis: In-Range Corner Cases")
    a("")
    a("For each corner case whose arithmetic intensity (AI) falls within the")
    a("week-4 DDP training range (45–1483 FLOP/byte), a 30-second continuous")
    a("workload was run while sampling pynvml at ~10 Hz.  This report identifies")
    a("which measurements cause false detection and which signals discriminate.")
    a("")
    a("## Training Baseline (resnet50, bs=4, FP16 AMP full backward+step)")
    a("")
    a(f"| Signal | Mean | Std | CV | ACF1 |")
    a(f"|--------|------|-----|----|------|")
    for k in ["gpu_util", "power_w", "mem_used_gb", "mem_util"]:
        a(f"| {k} | {ts.get(f'{k}_mean',0):.2f} | {ts.get(f'{k}_std',0):.2f} | "
          f"{ts.get(f'{k}_cv',0):.3f} | {ts.get(f'{k}_acf1',0):.3f} |")
    a("")

    # Group analyses
    groups_order = ["CC-A", "CC-B", "CC-C", "CC-D", "CC-E", "CC-F"]
    group_desc = {
        "CC-A": "CNN Inference",
        "CC-B": "LLM Prefill",
        "CC-C": "LLM Decode",
        "CC-D": "Quantisation Comparison",
        "CC-E": "Forward-Only Training Variant",
        "CC-F": "Vision Transformer Inference",
    }

    for grp in groups_order:
        grp_traces = [t for t in all_traces if t["group"] == grp]
        if not grp_traces:
            continue

        a(f"---")
        a(f"")
        a(f"## {grp}: {group_desc.get(grp, grp)}")
        a("")

        for t in grp_traces:
            s = t["stats"]
            a(f"### `{t['name']}`")
            a("")
            a(f"| Signal | Value | vs Training | Verdict |")
            a(f"|--------|-------|-------------|---------|")

            # gpu_util
            v = s.get("gpu_util_mean", 0); tr = ts.get("gpu_util_mean", 0)
            diff = v - tr
            verdict = _verdict("gpu_util", v, tr, s, ts)
            a(f"| GPU util mean | {v:.1f}% | {diff:+.1f}% | {verdict} |")

            # power
            v = s.get("power_w_mean", 0); tr = ts.get("power_w_mean", 0)
            diff = v - tr
            verdict = _verdict("power", v, tr, s, ts)
            a(f"| Power mean | {v:.1f} W | {diff:+.1f} W | {verdict} |")

            # power CV (variability)
            v = s.get("power_w_cv", 0); tr = ts.get("power_w_cv", 0)
            a(f"| Power CV | {v:.3f} | tr={tr:.3f} | {'SIMILAR' if abs(v-tr)<0.05 else 'DIFFERENT'} |")

            # power ACF1
            v = s.get("power_w_acf1", 0); tr = ts.get("power_w_acf1", 0)
            a(f"| Power ACF1 | {v:.3f} | tr={tr:.3f} | {'SIMILAR' if abs(v-tr)<0.1 else 'DIFFERENT'} |")

            # mem_used_gb
            v = s.get("mem_used_gb_mean", 0); tr = ts.get("mem_used_gb_mean", 0)
            diff = v - tr
            a(f"| Mem used mean | {v:.2f} GB | {diff:+.2f} GB | {'HIGHER' if v > tr * 1.5 else 'SIMILAR'} |")

            # mem_util
            v = s.get("mem_util_mean", 0); tr = ts.get("mem_util_mean", 0)
            diff = v - tr
            a(f"| Mem util mean | {v:.1f}% | {diff:+.1f}% | {'HIGHER' if v > tr + 5 else 'LOWER' if v < tr - 5 else 'SIMILAR'} |")

            # GPU util ACF1
            v = s.get("gpu_util_acf1", 0); tr = ts.get("gpu_util_acf1", 0)
            a(f"| GPU util ACF1 | {v:.3f} | tr={tr:.3f} | {'SIMILAR' if abs(v-tr)<0.1 else 'DIFFERENT'} |")

            a("")

            # Narrative analysis
            a("**False-detection cause:**")
            causes = _false_detection_causes(t, ts)
            for c in causes:
                a(f"- {c}")
            a("")
            a("**Discriminating signals:**")
            discriminators = _discriminating_signals(t, ts)
            for d in discriminators:
                a(f"- {d}")
            a("")

    # Summary table
    a("---")
    a("")
    a("## Summary: False Detection Risk by Config")
    a("")
    a("| Config | Group | GPU Util | Power | Mem GB | Risk | Key Discriminator |")
    a("|--------|-------|----------|-------|--------|------|-------------------|")
    for t in all_traces:
        if t["group"] == "TRAIN":
            continue
        s = t["stats"]
        risk = _risk_level(t, ts)
        disc = _top_discriminator(t, ts)
        a(f"| {t['name'].split('/',1)[-1]:45s} | {t['group']} | "
          f"{s.get('gpu_util_mean',0):5.1f}% | {s.get('power_w_mean',0):5.0f}W | "
          f"{s.get('mem_used_gb_mean',0):5.1f} | {risk} | {disc} |")
    a("")
    a("## Plot Files")
    a("")
    a("| File | Description |")
    a("|------|-------------|")
    a("| `tier1_CC-A.png` | CNN inference temporal traces vs training |")
    a("| `tier1_CC-B.png` | LLM prefill temporal traces vs training |")
    a("| `tier1_CC-C.png` | LLM decode temporal traces vs training |")
    a("| `tier1_CC-D.png` | Quantisation temporal traces vs training |")
    a("| `tier1_CC-E.png` | Forward-only variant temporal traces vs training |")
    a("| `tier1_CC-F.png` | ViT inference temporal traces vs training |")
    a("| `tier1_signal_comparison.png` | Bar chart comparison across all configs |")

    report_path = REPORTS / "tier1_analysis.md"
    report_path.write_text("\n".join(lines))
    log.info(f"Report saved: {report_path}")


def _verdict(signal, v, tr, s, ts):
    if signal == "gpu_util":
        if abs(v - tr) < 5:  return "SIMILAR — hard to distinguish"
        if v > tr + 10:      return "HIGHER than training"
        return                        "LOWER than training"
    if signal == "power":
        if abs(v - tr) < 30:  return "SIMILAR — hard to distinguish"
        if v > tr + 30:       return "HIGHER than training"
        return                         "LOWER than training"
    return "—"


def _false_detection_causes(trace, ts):
    s = trace["stats"]
    causes = []
    # Similar GPU util
    if abs(s.get("gpu_util_mean", 0) - ts.get("gpu_util_mean", 0)) < 15:
        causes.append(f"GPU util ≈ training ({s.get('gpu_util_mean',0):.0f}% vs "
                      f"{ts.get('gpu_util_mean',0):.0f}%) — SM appears equally busy")
    # Similar power
    if abs(s.get("power_w_mean", 0) - ts.get("power_w_mean", 0)) < 50:
        causes.append(f"Power draw ≈ training ({s.get('power_w_mean',0):.0f}W vs "
                      f"{ts.get('power_w_mean',0):.0f}W) — roofline-based rules pass")
    # Similar power ACF1
    if abs(s.get("power_w_acf1", 0) - ts.get("power_w_acf1", 0)) < 0.15:
        causes.append("Power autocorrelation (ACF1) similar — steady load pattern matches training")
    if not causes:
        causes.append("AI is within training range, causing roofline-only classifiers to misclassify")
    return causes


def _discriminating_signals(trace, ts):
    s = trace["stats"]
    discs = []
    # Mem used
    if s.get("mem_used_gb_mean", 0) > ts.get("mem_used_gb_mean", 0) * 2:
        discs.append(f"mem_used_gb is {s.get('mem_used_gb_mean',0):.1f} GB vs "
                     f"{ts.get('mem_used_gb_mean',0):.1f} GB for training "
                     f"(inference holds activations; training uses AMP lazy alloc)")
    elif s.get("mem_used_gb_mean", 0) < ts.get("mem_used_gb_mean", 0) * 0.5:
        discs.append(f"mem_used_gb is LOW ({s.get('mem_used_gb_mean',0):.1f} GB) — "
                     f"no gradient/optimizer state memory")
    # GPU util lower
    if s.get("gpu_util_mean", 0) < ts.get("gpu_util_mean", 0) - 20:
        discs.append(f"gpu_util_mean = {s.get('gpu_util_mean',0):.0f}% vs "
                     f"{ts.get('gpu_util_mean',0):.0f}% for training — "
                     f"intermittent execution (no continuous backward pass)")
    # Power lower
    if s.get("power_w_mean", 0) < ts.get("power_w_mean", 0) - 50:
        discs.append(f"power_w_mean = {s.get('power_w_mean',0):.0f}W vs "
                     f"{ts.get('power_w_mean',0):.0f}W — no gradient compute raises power")
    # Power CV
    if abs(s.get("power_w_cv", 0) - ts.get("power_w_cv", 0)) > 0.05:
        discs.append(f"power_w_cv = {s.get('power_w_cv',0):.3f} vs "
                     f"{ts.get('power_w_cv',0):.3f} — different variability pattern")
    # PCIe (not measured in tier1 but can be inferred)
    if trace["workload"] in ("llm_decode", "fwd_only"):
        discs.append("No PCIe gradient upload (pcie_tx = 0) — "
                     "training DDP gradient synchronisation absent")
    if not discs:
        discs.append("WARNING: No clear single-signal discriminator found — "
                     "requires multi-signal temporal classifier")
    return discs


def _risk_level(trace, ts):
    s = trace["stats"]
    score = 0
    if abs(s.get("gpu_util_mean", 0) - ts.get("gpu_util_mean", 0)) < 15: score += 2
    if abs(s.get("power_w_mean", 0) - ts.get("power_w_mean", 0)) < 50:  score += 2
    if abs(s.get("power_w_acf1", 0) - ts.get("power_w_acf1", 0)) < 0.15: score += 1
    if abs(s.get("gpu_util_acf1", 0) - ts.get("gpu_util_acf1", 0)) < 0.15: score += 1
    if score >= 5: return "HIGH"
    if score >= 3: return "MEDIUM"
    return "LOW"


def _top_discriminator(trace, ts):
    s = trace["stats"]
    # Check each signal for biggest relative difference
    candidates = []
    for sig in ["mem_used_gb", "mem_util", "gpu_util", "power_w"]:
        v  = s.get(f"{sig}_mean", 0)
        tr = ts.get(f"{sig}_mean", 0)
        if tr > 0.01:
            rel = abs(v - tr) / (abs(tr) + 1e-9)
            candidates.append((rel, sig))
    candidates.sort(reverse=True)
    if candidates:
        rel, sig = candidates[0]
        v  = s.get(f"{sig}_mean", 0)
        tr = ts.get(f"{sig}_mean", 0)
        return f"{sig} ({v:.1f} vs {tr:.1f})"
    return "—"


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Tier-1 Telemetry Test ===")
    handle = nvml_init()
    all_traces = []

    for cfg in TIER1_CONFIGS:
        result = measure_tier1(handle, cfg, duration_s=30)
        if result:
            all_traces.append(result)
        torch.cuda.synchronize()
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    # Save raw traces
    # Serialise samples (ts already float)
    out = []
    for t in all_traces:
        out.append({
            "name":     t["name"],
            "group":    t["group"],
            "workload": t["workload"],
            "stats":    t["stats"],
            # skip samples to keep file small — stats have all we need
        })
    with open(RESULTS / "tier1_summary.json", "w") as f:
        json.dump(out, f, indent=2)

    log.info("Generating plots ...")
    plot_temporal_traces(all_traces)
    plot_signal_comparison(all_traces)

    log.info("Writing analysis report ...")
    write_analysis(all_traces)

    log.info("=== Tier-1 Complete ===")


if __name__ == "__main__":
    main()
