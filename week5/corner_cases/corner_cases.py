#!/venv/main/bin/python
"""
week5/corner_cases/corner_cases.py
====================================
Real-application parameter sweeps designed to find workload configurations
whose GPU telemetry is indistinguishable from DDP training.

Corner Case Groups
------------------
CC-A  CNN Inference – batch/resolution/precision sweep (ResNet-18/50/101/152)
CC-B  LLM Prefill  – GPT-2 family, seq_len & batch sweep (compute-bound)
CC-C  LLM Decode   – GPT-2 autoregressive decode (memory-bound)
CC-D  Quantisation – FP32 / FP16 / BF16 inference comparison
CC-E  Training Edge– bs=1 grad-accum, frozen-backbone, forward-only training
CC-F  ViT Inference – vision transformer patch/head sweep

For each config, measures:
  - Achieved TFLOPS  (FLOP count / elapsed time)
  - HBM bandwidth    (bytes / elapsed time)
  - Arithmetic intensity AI = FLOP / byte
  - pynvml util, power, memory
  - Roofline position and regime

H100 NVL ceilings (measured / spec):
  FP16 Tensor Core : 1979 TFLOPS
  FP32 CUDA        :   67 TFLOPS
  HBM3e bandwidth  : 3900 GB/s  (NVL variant, 2× HBM stacks vs SXM5)
  FP32 ridge       :  ~17 FLOP/byte
  FP16 ridge       : ~507 FLOP/byte
"""

import sys, os, json, time, math, argparse, logging, warnings
from pathlib import Path

sys.path.insert(0, "/tmp/pynvml_pkg")
warnings.filterwarnings("ignore")

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pynvml

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s [%(levelname)s] %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── Hardware constants ─────────────────────────────────────────────────────────
H100_FP16_TFLOPS  = 1979.0      # tensor-core FP16
H100_FP32_TFLOPS  =   67.0      # CUDA-core FP32
H100_HBM_GBps     = 3900.0      # H100 NVL HBM3e
RIDGE_FP32        = H100_FP32_TFLOPS * 1e12 / (H100_HBM_GBps * 1e9)  # ≈17
RIDGE_FP16        = H100_FP16_TFLOPS * 1e12 / (H100_HBM_GBps * 1e9)  # ≈507

OUT_DIR = Path(__file__).parent / "results"
OUT_DIR.mkdir(exist_ok=True)

DEVICE = torch.device("cuda:0")

# ── pynvml helpers ─────────────────────────────────────────────────────────────

def nvml_init():
    pynvml.nvmlInit()
    return pynvml.nvmlDeviceGetHandleByIndex(0)

def nvml_snapshot(handle):
    u  = pynvml.nvmlDeviceGetUtilizationRates(handle)
    pw = pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0
    mi = pynvml.nvmlDeviceGetMemoryInfo(handle)
    return {
        "gpu_util": u.gpu,
        "mem_util": u.memory,
        "power_w":  pw,
        "mem_used_gb": mi.used / 1e9,
    }

# ── Timing helper ──────────────────────────────────────────────────────────────

def cuda_time(fn, warmup=10, repeats=50):
    """Run fn() warmup+repeats times; return median elapsed ms."""
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = torch.cuda.Event(enable_timing=True)
    t1 = torch.cuda.Event(enable_timing=True)
    times = []
    for _ in range(repeats):
        t0.record()
        fn()
        t1.record()
        torch.cuda.synchronize()
        times.append(t0.elapsed_time(t1))
    return float(np.median(times))

# ── FLOP / byte counters ───────────────────────────────────────────────────────

def conv_flops(b, cin, cout, k, h, w):
    """FLOPs for one Conv2d (multiply-add = 2 ops)."""
    return 2 * b * cin * cout * k * k * h * w

def linear_flops(b, in_f, out_f):
    return 2 * b * in_f * out_f

def attn_flops(b, s, d, heads):
    """FLOPs for multi-head self-attention: QKV proj + scores + values + out proj."""
    qkv   = 3 * linear_flops(b * s, d, d)
    scores = 2 * b * heads * s * s * (d // heads)
    vals   = 2 * b * heads * s * s * (d // heads)
    out    = linear_flops(b * s, d, d)
    return qkv + scores + vals + out

def mlp_flops(b, s, d, mlp_ratio=4):
    return 2 * linear_flops(b * s, d, d * mlp_ratio) + 2 * linear_flops(b * s, d * mlp_ratio, d)

def transformer_flops(b, s, d, n_layers, mlp_ratio=4, heads=None):
    heads = heads or max(1, d // 64)
    per_layer = attn_flops(b, s, d, heads) + mlp_flops(b, s, d, mlp_ratio)
    return per_layer * n_layers

def resnet_flops(batch, arch="resnet50", img_size=224):
    """Approximate FLOPs for ResNet forward pass."""
    configs = {
        "resnet18":  [(64,64,3,56,2), (64,128,3,28,2), (128,256,3,14,2), (256,512,3,7,2)],
        "resnet50":  [(64,256,1,56,3), (256,512,1,28,4), (512,1024,1,14,6), (1024,2048,1,7,3)],
        "resnet101": [(64,256,1,56,3), (256,512,1,28,4), (512,1024,1,14,23),(1024,2048,1,7,3)],
        "resnet152": [(64,256,1,56,3), (256,512,1,28,8), (512,1024,1,14,36),(1024,2048,1,7,3)],
    }
    scale = img_size / 224.0
    total = conv_flops(batch, 3, 64, 7, int(112 * scale), int(112 * scale))
    for cin, cout, k, h, n_blocks in configs.get(arch, configs["resnet50"]):
        h_s = int(h * scale)
        for _ in range(n_blocks):
            total += conv_flops(batch, cin, cout, k, h_s, h_s) * 3
    total += linear_flops(batch, 2048 if "resnet50" in arch or "resnet101" in arch
                          or "resnet152" in arch else 512, 1000)
    return total

def model_param_bytes(model, dtype):
    n = sum(p.numel() for p in model.parameters())
    return n * (2 if dtype in (torch.float16, torch.bfloat16) else 4)

# ── Model builders ─────────────────────────────────────────────────────────────

def make_resnet(arch="resnet50"):
    import torchvision.models as tvm
    fn = {"resnet18": tvm.resnet18, "resnet50": tvm.resnet50,
          "resnet101": tvm.resnet101, "resnet152": tvm.resnet152}
    return fn[arch](weights=None)

class GPT2Block(nn.Module):
    def __init__(self, d_model, n_heads, mlp_ratio=4, max_len=2048):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.ln2 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.mlp  = nn.Sequential(
            nn.Linear(d_model, d_model * mlp_ratio),
            nn.GELU(),
            nn.Linear(d_model * mlp_ratio, d_model))
        self.register_buffer("causal_mask",
            torch.triu(torch.ones(max_len, max_len), diagonal=1).bool())

    def forward(self, x):
        s = x.size(1)
        mask = self.causal_mask[:s, :s]
        h = self.ln1(x)
        h, _ = self.attn(h, h, h, attn_mask=mask, need_weights=False)
        x = x + h
        x = x + self.mlp(self.ln2(x))
        return x

class MiniLLM(nn.Module):
    """Minimal causal LM matching GPT-2 family shapes."""
    CONFIGS = {
        "gpt2":    dict(d=768,  n_layer=12, n_head=12),
        "gpt2-m":  dict(d=1024, n_layer=24, n_head=16),
        "gpt2-l":  dict(d=1280, n_layer=36, n_head=20),
        "gpt2-xl": dict(d=1600, n_layer=48, n_head=25),
    }
    def __init__(self, name="gpt2", vocab=50257, max_len=2048):
        super().__init__()
        c = self.CONFIGS[name]
        d, n_layer, n_head = c["d"], c["n_layer"], c["n_head"]
        self.d       = d
        self.n_layer = n_layer
        self.n_head  = n_head
        self.tok_emb = nn.Embedding(vocab, d)
        self.pos_emb = nn.Embedding(max_len, d)
        self.blocks  = nn.ModuleList([GPT2Block(d, n_head, max_len=max_len)
                                      for _ in range(n_layer)])
        self.ln_f    = nn.LayerNorm(d)
        self.head    = nn.Linear(d, vocab, bias=False)

    def forward(self, ids):
        b, s = ids.shape
        pos  = torch.arange(s, device=ids.device).unsqueeze(0)
        x    = self.tok_emb(ids) + self.pos_emb(pos)
        for blk in self.blocks:
            x = blk(x)
        return self.head(self.ln_f(x))

class ViT(nn.Module):
    """Minimal Vision Transformer."""
    def __init__(self, img_size=224, patch=16, d=768, n_layer=12, n_head=12,
                 n_classes=1000):
        super().__init__()
        n_patches = (img_size // patch) ** 2
        self.patch_emb = nn.Conv2d(3, d, patch, stride=patch)
        self.cls_tok   = nn.Parameter(torch.randn(1, 1, d))
        self.pos_emb   = nn.Parameter(torch.randn(1, n_patches + 1, d))
        self.blocks    = nn.ModuleList([GPT2Block(d, n_head, max_len=n_patches + 1)
                                        for _ in range(n_layer)])
        self.ln        = nn.LayerNorm(d)
        self.head      = nn.Linear(d, n_classes)
        self.d = d; self.n_layer = n_layer; self.n_head = n_head
        self.n_patches = n_patches; self.patch = patch

    def forward(self, x):
        b = x.size(0)
        t = self.patch_emb(x).flatten(2).transpose(1, 2)
        t = torch.cat([self.cls_tok.expand(b, -1, -1), t], 1) + self.pos_emb
        for blk in self.blocks:
            t = blk(t)
        return self.head(self.ln(t[:, 0]))

# ── Single measurement ─────────────────────────────────────────────────────────

def measure(model, inputs, flops, case_name, dtype_str, handle,
            warmup=5, repeats=30):
    """Run model, collect timing + telemetry, return result dict."""
    model.eval()
    with torch.no_grad():
        elapsed_ms = cuda_time(lambda: model(*inputs), warmup=warmup, repeats=repeats)

    elapsed_s = elapsed_ms / 1000.0
    achieved_tflops = flops / elapsed_s / 1e12

    # Approximate memory bytes = model params + activations
    param_bytes = model_param_bytes(model, next(model.parameters()).dtype)
    # Activation estimate: roughly same as param size for inference
    mem_bytes = param_bytes * 2
    ai = flops / mem_bytes

    snap = nvml_snapshot(handle)

    return {
        "case":             case_name,
        "dtype":            dtype_str,
        "elapsed_ms":       round(elapsed_ms, 3),
        "flops":            flops,
        "param_bytes":      param_bytes,
        "ai":               round(ai, 1),
        "achieved_tflops":  round(achieved_tflops, 3),
        "ridge_fp32":       round(RIDGE_FP32, 1),
        "ridge_fp16":       round(RIDGE_FP16, 1),
        "regime":           ("memory-bound" if ai < RIDGE_FP32
                             else "compute-fp32" if ai < RIDGE_FP16
                             else "compute-fp16"),
        **snap,
    }

# ── Case groups ────────────────────────────────────────────────────────────────

def run_cc_a_cnn(handle, results):
    """CC-A: CNN inference — batch / resolution / precision sweep."""
    log.info("── CC-A: CNN inference sweep ──")
    archs     = ["resnet18", "resnet50", "resnet101", "resnet152"]
    batches   = [1, 4, 16, 64, 256, 1024]
    img_sizes = [224, 512]
    dtypes    = [("fp16", torch.float16), ("bf16", torch.bfloat16)]

    for arch in archs:
        for img_size in img_sizes:
            try:
                model = make_resnet(arch).to(DEVICE)
                for dtype_str, dtype in dtypes:
                    model = model.to(dtype)
                    for bs in batches:
                        try:
                            x = torch.randn(bs, 3, img_size, img_size,
                                            device=DEVICE, dtype=dtype)
                            flops = resnet_flops(bs, arch, img_size)
                            name  = f"CC-A/{arch}/bs{bs}/img{img_size}/{dtype_str}"
                            r = measure(model, (x,), flops, name, dtype_str, handle)
                            r["group"] = "CC-A"; r["arch"] = arch
                            r["batch"] = bs; r["img_size"] = img_size
                            r["workload"] = "inference"
                            results.append(r)
                            log.info(f"  {name:45s}  AI={r['ai']:6.0f}  "
                                     f"{r['achieved_tflops']:6.1f} TFLOPS  {r['regime']}")
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            log.warning(f"  OOM: {arch} bs={bs} img={img_size}")
                            break
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning(f"  skip {arch}: {e}")


def run_cc_b_llm_prefill(handle, results):
    """CC-B: LLM prefill — compute-bound for large (batch × seq_len)."""
    log.info("── CC-B: LLM prefill sweep ──")
    models_cfg = [
        ("gpt2",    "gpt2"),
        ("gpt2-m",  "gpt2-m"),
        ("gpt2-l",  "gpt2-l"),
    ]
    batches  = [1, 4, 16, 64]
    seq_lens = [32, 128, 512, 1024, 2048]
    dtypes   = [("fp16", torch.float16), ("fp32", torch.float32)]

    for model_name, cfg_key in models_cfg:
        try:
            cfg = MiniLLM.CONFIGS[cfg_key]
            d, n_layer, n_head = cfg["d"], cfg["n_layer"], cfg["n_head"]
            for dtype_str, dtype in dtypes:
                model = MiniLLM(cfg_key).to(DEVICE).to(dtype)
                for bs in batches:
                    for seq_len in seq_lens:
                        try:
                            ids = torch.randint(0, 50257, (bs, seq_len), device=DEVICE)
                            flops = transformer_flops(bs, seq_len, d, n_layer,
                                                      heads=n_head)
                            name  = f"CC-B/{model_name}/bs{bs}/s{seq_len}/{dtype_str}"
                            r = measure(model, (ids,), flops, name, dtype_str, handle,
                                        warmup=3, repeats=20)
                            r["group"] = "CC-B"; r["arch"] = model_name
                            r["batch"] = bs; r["seq_len"] = seq_len
                            r["workload"] = "llm_prefill"
                            results.append(r)
                            log.info(f"  {name:45s}  AI={r['ai']:6.0f}  "
                                     f"{r['achieved_tflops']:6.1f} TFLOPS  {r['regime']}")
                        except torch.cuda.OutOfMemoryError:
                            torch.cuda.empty_cache()
                            log.warning(f"  OOM: {model_name} bs={bs} s={seq_len}")
                del model
                torch.cuda.empty_cache()
        except Exception as e:
            log.warning(f"  skip {model_name}: {e}")


def run_cc_c_llm_decode(handle, results):
    """CC-C: LLM decode — memory-bound token-by-token generation."""
    log.info("── CC-C: LLM decode sweep ──")
    models_cfg = [("gpt2", "gpt2"), ("gpt2-m", "gpt2-m")]
    batches    = [1, 4, 16, 64, 256]
    ctx_lens   = [128, 512, 1024]

    for model_name, cfg_key in models_cfg:
        try:
            cfg = MiniLLM.CONFIGS[cfg_key]
            d, n_layer, n_head = cfg["d"], cfg["n_layer"], cfg["n_head"]
            model = MiniLLM(cfg_key).to(DEVICE).to(torch.float16)
            for bs in batches:
                for ctx in ctx_lens:
                    try:
                        # Decode = process 1 new token given ctx-length KV cache
                        # Approximate: run model on 1 token (decode step)
                        ids = torch.randint(0, 50257, (bs, 1), device=DEVICE)
                        flops = transformer_flops(bs, 1, d, n_layer, heads=n_head)
                        # Memory: load all KV cache + weights
                        kv_bytes = bs * ctx * d * 2 * n_layer * 2  # K+V per layer
                        param_bytes = model_param_bytes(model, torch.float16)
                        mem_bytes   = param_bytes + kv_bytes
                        name = f"CC-C/{model_name}/bs{bs}/ctx{ctx}/fp16"
                        r = measure(model, (ids,), flops, name, "fp16", handle,
                                    warmup=5, repeats=30)
                        r["ai"] = round(flops / mem_bytes, 2)
                        r["regime"] = ("memory-bound" if r["ai"] < RIDGE_FP32
                                       else "compute-fp32")
                        r["group"] = "CC-C"; r["arch"] = model_name
                        r["batch"] = bs; r["ctx_len"] = ctx
                        r["workload"] = "llm_decode"
                        results.append(r)
                        log.info(f"  {name:45s}  AI={r['ai']:6.2f}  "
                                 f"{r['achieved_tflops']:6.3f} TFLOPS  {r['regime']}")
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        log.warning(f"  OOM: {model_name} bs={bs} ctx={ctx}")
            del model
            torch.cuda.empty_cache()
        except Exception as e:
            log.warning(f"  skip {model_name}: {e}")


def run_cc_d_quantisation(handle, results):
    """CC-D: Quantisation effect — FP32/FP16/BF16 for same (model, batch, seq)."""
    log.info("── CC-D: Quantisation sweep ──")
    configs = [
        ("resnet50", "cnn"),
        ("gpt2",     "llm"),
    ]
    batches  = [16, 64, 256]
    dtypes   = [("fp32", torch.float32), ("fp16", torch.float16),
                ("bf16", torch.bfloat16)]

    for arch_name, kind in configs:
        for dtype_str, dtype in dtypes:
            try:
                if kind == "cnn":
                    model = make_resnet(arch_name).to(DEVICE).to(dtype)
                    for bs in batches:
                        x = torch.randn(bs, 3, 224, 224, device=DEVICE, dtype=dtype)
                        flops = resnet_flops(bs, arch_name, 224)
                        name  = f"CC-D/{arch_name}/bs{bs}/{dtype_str}"
                        r = measure(model, (x,), flops, name, dtype_str, handle)
                        r["group"] = "CC-D"; r["arch"] = arch_name
                        r["batch"] = bs; r["workload"] = "inference_quant"
                        results.append(r)
                        log.info(f"  {name:45s}  AI={r['ai']:6.0f}  "
                                 f"{r['achieved_tflops']:6.1f} TFLOPS  {r['regime']}")
                else:
                    cfg = MiniLLM.CONFIGS["gpt2"]
                    d, n_layer, n_head = cfg["d"], cfg["n_layer"], cfg["n_head"]
                    model = MiniLLM("gpt2").to(DEVICE).to(dtype)
                    for bs in [1, 16, 64]:
                        ids = torch.randint(0, 50257, (bs, 512), device=DEVICE)
                        flops = transformer_flops(bs, 512, d, n_layer, heads=n_head)
                        name  = f"CC-D/gpt2/bs{bs}/s512/{dtype_str}"
                        r = measure(model, (ids,), flops, name, dtype_str, handle,
                                    warmup=3, repeats=20)
                        r["group"] = "CC-D"; r["arch"] = "gpt2"
                        r["batch"] = bs; r["workload"] = "llm_quant"
                        results.append(r)
                        log.info(f"  {name:45s}  AI={r['ai']:6.0f}  "
                                 f"{r['achieved_tflops']:6.1f} TFLOPS  {r['regime']}")
                del model
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning(f"  skip {arch_name}/{dtype_str}: {e}")


def run_cc_e_training_edge(handle, results):
    """CC-E: Training variants — small-batch, forward-only, grad-accum."""
    log.info("── CC-E: Training edge cases ──")
    from torch.optim import SGD

    arch = "resnet50"

    # E1: Forward-only training (no backward — looks like inference)
    model = make_resnet(arch).to(DEVICE).to(torch.float16)
    for bs in [1, 4, 16, 64, 256]:
        try:
            x = torch.randn(bs, 3, 224, 224, device=DEVICE, dtype=torch.float16)
            y = torch.randint(0, 1000, (bs,), device=DEVICE)
            flops = resnet_flops(bs, arch, 224)
            def fwd_only():
                with torch.cuda.amp.autocast():
                    loss = F.cross_entropy(model(x), y)
            name = f"CC-E/fwd_only/bs{bs}/fp16"
            r = measure(model, (x,), flops, name, "fp16", handle)
            r["group"] = "CC-E"; r["variant"] = "forward_only"
            r["batch"] = bs; r["workload"] = "training_fwd_only"
            results.append(r)
            log.info(f"  {name:45s}  AI={r['ai']:6.0f}  "
                     f"{r['achieved_tflops']:6.1f} TFLOPS  {r['regime']}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); break
    del model; torch.cuda.empty_cache()

    # E2: Full training with gradient (backward included)
    model = make_resnet(arch).to(DEVICE).to(torch.float16)
    opt   = SGD(model.parameters(), lr=0.01)
    scaler = torch.cuda.amp.GradScaler()
    for bs in [1, 4, 16, 64]:
        try:
            x = torch.randn(bs, 3, 224, 224, device=DEVICE, dtype=torch.float16)
            y = torch.randint(0, 1000, (bs,), device=DEVICE)
            flops = resnet_flops(bs, arch, 224) * 3  # fwd + bwd ≈ 3×
            def train_step():
                opt.zero_grad()
                with torch.cuda.amp.autocast():
                    loss = F.cross_entropy(model(x), y)
                scaler.scale(loss).backward()
                scaler.step(opt); scaler.update()
            name = f"CC-E/full_train/bs{bs}/fp16"
            r = measure(model, (x,), flops, name, "fp16", handle,
                        warmup=3, repeats=20)
            r["group"] = "CC-E"; r["variant"] = "full_training"
            r["batch"] = bs; r["workload"] = "training_full"
            results.append(r)
            log.info(f"  {name:45s}  AI={r['ai']:6.0f}  "
                     f"{r['achieved_tflops']:6.1f} TFLOPS  {r['regime']}")
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache(); break
    del model; torch.cuda.empty_cache()


def run_cc_f_vit(handle, results):
    """CC-F: Vision Transformer — patch/head/depth sweep."""
    log.info("── CC-F: ViT inference sweep ──")
    configs = [
        ("ViT-S/16", dict(img_size=224, patch=16, d=384,  n_layer=12, n_head=6)),
        ("ViT-B/16", dict(img_size=224, patch=16, d=768,  n_layer=12, n_head=12)),
        ("ViT-L/16", dict(img_size=224, patch=16, d=1024, n_layer=24, n_head=16)),
        ("ViT-B/8",  dict(img_size=224, patch=8,  d=768,  n_layer=12, n_head=12)),
        ("ViT-H/14", dict(img_size=224, patch=14, d=1280, n_layer=32, n_head=16)),
    ]
    batches = [1, 8, 32, 128, 512]
    dtype   = torch.float16

    for vit_name, cfg in configs:
        try:
            model = ViT(**cfg).to(DEVICE).to(dtype)
            n_patches = cfg["n_patches"] if "n_patches" in cfg else (cfg["img_size"] // cfg["patch"]) ** 2
            s = n_patches + 1  # +cls token
            d = cfg["d"]; n_layer = cfg["n_layer"]; n_head = cfg["n_head"]
            for bs in batches:
                try:
                    x = torch.randn(bs, 3, cfg["img_size"], cfg["img_size"],
                                    device=DEVICE, dtype=dtype)
                    flops = transformer_flops(bs, s, d, n_layer, heads=n_head)
                    name  = f"CC-F/{vit_name}/bs{bs}/fp16"
                    r = measure(model, (x,), flops, name, "fp16", handle,
                                warmup=3, repeats=20)
                    r["group"] = "CC-F"; r["arch"] = vit_name
                    r["batch"] = bs; r["workload"] = "vit_inference"
                    results.append(r)
                    log.info(f"  {name:45s}  AI={r['ai']:6.0f}  "
                             f"{r['achieved_tflops']:6.1f} TFLOPS  {r['regime']}")
                except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError):
                    try: torch.cuda.empty_cache()
                    except Exception: pass
                    break
            try:
                del model; torch.cuda.empty_cache()
            except Exception: pass
        except Exception as e:
            log.warning(f"  skip {vit_name}: {e}")


# ── Main ───────────────────────────────────────────────────────────────────────

def main():
    handle  = nvml_init()
    results = []

    runners = [
        run_cc_a_cnn,
        run_cc_b_llm_prefill,
        run_cc_c_llm_decode,
        run_cc_d_quantisation,
        run_cc_e_training_edge,
        run_cc_f_vit,
    ]

    for fn in runners:
        try:
            fn(handle, results)
        except Exception as e:
            log.error(f"Group failed: {fn.__name__}: {e}")
        try:
            torch.cuda.empty_cache()
        except Exception:
            pass

    out_path = OUT_DIR / "corner_cases_results.json"
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    log.info(f"\nSaved {len(results)} records → {out_path}")

    # Quick summary
    from collections import Counter
    groups = Counter(r["group"] for r in results)
    regimes = Counter(r["regime"] for r in results)
    log.info(f"Groups: {dict(groups)}")
    log.info(f"Regimes: {dict(regimes)}")

    # Print boundary cases (closest to training regime)
    training_ai_range = (45, 1483)  # week4 training range
    boundary = [r for r in results
                if training_ai_range[0] <= r["ai"] <= training_ai_range[1]
                and r["workload"] != "training_full"]
    log.info(f"\n{'='*60}")
    log.info(f"Configs within training AI range ({training_ai_range[0]}–{training_ai_range[1]} FLOP/B):")
    for r in sorted(boundary, key=lambda x: x["ai"]):
        log.info(f"  {r['case']:50s} AI={r['ai']:6.0f}  {r['achieved_tflops']:6.1f} TFLOPS")


if __name__ == "__main__":
    main()
