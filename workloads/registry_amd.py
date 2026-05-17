"""AMD/ROCm workload registry — analogue of workloads/registry.py.

Differences from the NVIDIA registry:
  - `ffmpeg_nvenc` is replaced by `ffmpeg_vaapi` (VA-API on the AMD VCN engine)
  - Workloads requiring Blender/GROMACS or actual NVENC are removed
  - Multi-GPU DDP / FSDP entries are removed (single consumer GPU only)
  - Frontier-scale LLM entries (>16 GB VRAM) are removed
  - Batch sizes for ML training are sized for 16 GB consumer cards
    (RX 9070 XT, 7900 XTX, etc.); see Week 8 laptop instructions for the
    rationale on the smaller-batch regime.

The PyTorch workloads themselves are unchanged from the NVIDIA registry —
PyTorch ROCm aliases the `torch.cuda` namespace, so files like
`pytorch_training.py`, `cufft_benchmark.py`, `nbody_sim.py`, and
`mining_proxy.py` run as-is on AMD with no source modifications.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOADS_DIR = os.path.join(_REPO_ROOT, "workloads")

# Default batch sizes for 16 GB consumer cards. These mirror the laptop
# RTX 4050 (6 GB) reductions in spirit but use the larger envelope of the
# RX 9070 XT / 7900 XTX class card. Adjust per workload below if needed.
_BATCH_RESNET = "256"
_BATCH_MLP = "512"
_BATCH_GPT2 = "4"
_BATCH_BERT = "16"
_BATCH_RESNET50_INFER = "128"

WORKLOAD_REGISTRY_AMD = {
    # --- Baselines ---
    "idle": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "idle.py")],
        "description": "Idle GPU baseline (120s of sleep)",
        "default_timeout": 130,
    },

    # --- ML Training (PyTorch — runs unchanged on ROCm) ---
    "pytorch_resnet_cifar10": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "resnet18", "--dataset", "cifar10",
            "--epochs", "10", "--batch-size", _BATCH_RESNET,
        ],
        "description": "ResNet-18 / CIFAR-10, 10 epochs, FP32 (16 GB-sized)",
        "default_timeout": 1200,
    },
    "pytorch_resnet_cifar10_amp": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "resnet18", "--dataset", "cifar10",
            "--epochs", "10", "--batch-size", _BATCH_RESNET, "--amp",
        ],
        "description": "ResNet-18 / CIFAR-10, 10 epochs, AMP (BF16 on RDNA)",
        "default_timeout": 1200,
    },
    "pytorch_mlp_cifar10": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "simple_mlp", "--dataset", "cifar10",
            "--epochs", "20", "--batch-size", _BATCH_MLP,
        ],
        "description": "MLP / CIFAR-10, 20 epochs, FP32",
        "default_timeout": 900,
    },
    "gpt2_wikitext2": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "gpt2_finetune.py"),
            "--steps", "400", "--batch-size", _BATCH_GPT2,
        ],
        "description": "GPT-2 124M / WikiText-2, 400 steps, FP32",
        "default_timeout": 1500,
    },
    "gpt2_wikitext2_amp": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "gpt2_finetune.py"),
            "--steps", "400", "--batch-size", _BATCH_GPT2, "--amp",
        ],
        "description": "GPT-2 124M / WikiText-2, 400 steps, AMP",
        "default_timeout": 1500,
    },
    "bert_sst2": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "bert_finetune.py"),
            "--steps", "300", "--batch-size", _BATCH_BERT,
        ],
        "description": "BERT-base / SST-2, 300 steps, FP32",
        "default_timeout": 1200,
    },
    "bert_sst2_amp": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "bert_finetune.py"),
            "--steps", "300", "--batch-size", _BATCH_BERT, "--amp",
        ],
        "description": "BERT-base / SST-2, 300 steps, AMP",
        "default_timeout": 1200,
    },

    # --- ML Inference ---
    "resnet50_inference": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "inference.py"),
            "--model", "resnet50", "--batch-size", _BATCH_RESNET50_INFER,
            "--duration", "600",
        ],
        "description": "ResNet-50 batch inference, 600s, no gradients",
        "default_timeout": 660,
    },

    # --- Scientific HPC (pure torch, runs unchanged) ---
    "cufft_benchmark": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "scientific_hpc.py"),
            "--workload", "cufft", "--duration", "600",
        ],
        "description": "3D FFT benchmark via torch.fft (rocFFT-backed on AMD), 600s",
        "default_timeout": 660,
    },
    "nbody_sim": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "scientific_hpc.py"),
            "--workload", "nbody", "--duration", "600", "--n-bodies", "4096",
        ],
        "description": "Gravitational N-body, compute-bound, 4096 bodies, 600s",
        "default_timeout": 660,
    },

    # --- Crypto Mining (proxy) ---
    "mining_ethash_proxy": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "mining_proxy.py"),
            "--duration", "600", "--dag-size-mb", "1024",
        ],
        "description": "Ethash-like memory-hard hashing proxy, 1 GB DAG, 600s",
        "default_timeout": 660,
    },

    # --- Rendering ---
    "rendering_proxy": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "rendering_proxy.py"),
            "--duration", "600", "--rays", "32768", "--bounces", "4",
        ],
        "description": "Monte Carlo path tracing proxy (irregular compute), 600s",
        "default_timeout": 660,
    },

    # --- Other (AMD-specific replacement for ffmpeg_nvenc) ---
    "ffmpeg_vaapi": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "ffmpeg_vaapi.py"),
            "--duration", "300",
        ],
        "description": "FFmpeg VA-API hardware video encoding (AMD VCN engine)",
        "default_timeout": 360,
    },

    # --- Modern Inference (small enough for 16 GB) ---
    "deepseek_r1_inference": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "modern_inference.py"),
            "--model", "deepseek_r1", "--duration", "300", "--max-tokens", "256",
        ],
        "description": "DeepSeek-R1-Distill-Qwen-1.5B autoregressive generation",
        "default_timeout": 420,
    },
    "qwen25_inference": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "modern_inference.py"),
            "--model", "qwen25", "--duration", "300", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 0.5B autoregressive generation",
        "default_timeout": 420,
    },
    "whisper_inference": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "modern_inference.py"),
            "--model", "whisper", "--duration", "300",
        ],
        "description": "Whisper large-v3-turbo audio transcription",
        "default_timeout": 420,
    },

    # --- Round 5b consumer-tier LLM inference (fits in 16 GB FP16) ---
    "llm_infer_phi3_mini": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "phi3_mini",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "Phi-3-Mini-4K 3.8B FP16 — single-GPU consumer anchor",
        "default_timeout": 420,
    },
    "llm_infer_qwen25_1p5b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "qwen25_1p5b",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 1.5B Instruct, FP16 — mid-size LLM inference anchor",
        "default_timeout": 420,
    },
    "llm_infer_smollm2_1p7b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "smollm2_1p7b",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "SmolLM2 1.7B Instruct (Apache 2.0)",
        "default_timeout": 420,
    },
    "llm_infer_tinyllama_bs8": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "tinyllama",
            "--duration", "300", "--batch-size", "8", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "TinyLlama-1.1B bs=8 — batch sweep",
        "default_timeout": 420,
    },

    # --- Image / vision inference ---
    "llm_infer_clip_embed_bs64": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "clip_l14", "--duration", "300", "--batch-size", "64",
        ],
        "description": "CLIP ViT-L/14 image embedding @ batch 64",
        "default_timeout": 420,
    },
    "llm_infer_vit_l_bs64": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "vit_l", "--duration", "300", "--batch-size", "64",
        ],
        "description": "ViT-L/16 ImageNet classification @ batch 64",
        "default_timeout": 420,
    },

    # --- Adversarial (single-GPU strategies only) ---
    "adversarial_A_util_modulation": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_training.py"),
            "--strategy", "A", "--duration", "300",
            "--sleep-min", "0.1", "--sleep-max", "0.5",
        ],
        "description": "Strategy A: training with sleep gaps",
        "default_timeout": 360,
    },
    "adversarial_B_low_util": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_training.py"),
            "--strategy", "B", "--duration", "300",
        ],
        "description": "Strategy B: tiny-batch CPU-bottlenecked training",
        "default_timeout": 360,
    },
    "adversarial_D_temporal_disruption": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_training.py"),
            "--strategy", "D", "--duration", "300",
        ],
        "description": "Strategy D: random batches + CUDA noise",
        "default_timeout": 360,
    },
    "adversarial_K_online_learning": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
            "--strategy", "K", "--duration", "300", "--batch-size", "128",
        ],
        "description": "Strategy K: online learning (inference + backward per batch)",
        "default_timeout": 360,
    },
    "adversarial_L_diluted_5": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
            "--strategy", "L", "--duration", "300", "--dilution", "5",
            "--batch-size", "128",
        ],
        "description": "Strategy L: diluted training, backward on 1/5 batches",
        "default_timeout": 360,
    },
    "adversarial_L_diluted_10": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
            "--strategy", "L", "--duration", "300", "--dilution", "10",
            "--batch-size", "128",
        ],
        "description": "Strategy L: diluted training, backward on 1/10 batches",
        "default_timeout": 360,
    },

    # --- Tier 2: missing adversarial families ported from NVIDIA registry ---
    # Strategies G (clock_throttled) and J (PID thermal-matched) require
    # NVIDIA-specific tooling (nvidia-smi -pl, pynvml) and are intentionally
    # omitted from this AMD registry. The remaining 8 single-GPU evasion
    # families ought to run as-is on ROCm since they are pure PyTorch +
    # subprocess + sleep mechanics.

    "adversarial_E_memory_minimal": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_advanced.py"),
            "--strategy", "E", "--duration", "300",
        ],
        "description": "Strategy E: gradient checkpointing + aggressive GC",
        "default_timeout": 360,
    },
    "adversarial_F_interleave": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_advanced.py"),
            "--strategy", "F", "--duration", "300",
            "--train-frac", "0.3", "--interleave-sec", "10",
        ],
        "description": "Strategy F: train/inference interleaving (30% train)",
        "default_timeout": 360,
    },
    "adversarial_H_mimicry_mining": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
            "--strategy", "H", "--duration", "300", "--target", "mining",
        ],
        "description": "Strategy H: train while mimicking mining telemetry profile",
        "default_timeout": 360,
    },
    "adversarial_H_mimicry_cufft": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
            "--strategy", "H", "--duration", "300", "--target", "cufft",
        ],
        "description": "Strategy H: train while mimicking cuFFT telemetry profile",
        "default_timeout": 360,
    },
    "adversarial_I_stochastic": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
            "--strategy", "I", "--duration", "300",
        ],
        "description": "Strategy I: stochastic phase switching (kills autocorrelation)",
        "default_timeout": 360,
    },
    "adversarial_J_pid_75": {
        # Now works on AMD: adversarial_round2.py auto-detects pynvml vs amdsmi
        # for the PID feedback loop's util/power reads.
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
            "--strategy", "J", "--duration", "300", "--target-util", "75",
        ],
        "description": "Strategy J: PID thermal-matched training (util target 75%)",
        "default_timeout": 360,
    },
    "adversarial_M_composite_memmin": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
            "--strategy", "M", "--duration", "300",
            "--idle-fraction", "0.2", "--prealloc-mb", "4096",
        ],
        "description": "Strategy M: composite + memory minimal (idle=20%, prealloc=4GB)",
        "default_timeout": 360,
    },
    "adversarial_N_grad_accum": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
            "--strategy", "N", "--duration", "300", "--accum-steps", "8",
        ],
        "description": "Strategy N: grad-accum dilution (step every 8 batches)",
        "default_timeout": 360,
    },

    # composite_idle_pad sweep — 5 idle fractions over the [0, 95] range
    "adversarial_composite_idle_00": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
            "--idle-fraction", "0.00", "--duration", "300",
        ],
        "description": "Composite evasion @ 0% idle (full throughput baseline)",
        "default_timeout": 360,
    },
    "adversarial_composite_idle_20": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
            "--idle-fraction", "0.20", "--duration", "300",
        ],
        "description": "Composite evasion @ 20% idle",
        "default_timeout": 360,
    },
    "adversarial_composite_idle_50": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
            "--idle-fraction", "0.50", "--duration", "300",
        ],
        "description": "Composite evasion @ 50% idle",
        "default_timeout": 360,
    },
    "adversarial_composite_idle_80": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
            "--idle-fraction", "0.80", "--duration", "300",
        ],
        "description": "Composite evasion @ 80% idle",
        "default_timeout": 360,
    },
    "adversarial_composite_idle_95": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
            "--idle-fraction", "0.95", "--duration", "300",
        ],
        "description": "Composite evasion @ 95% idle (near full evasion attempt)",
        "default_timeout": 360,
    },

    # --- Tier 4: broader workload diversity ---
    "sdxl_inference": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "sdxl_inference.py"),
            "--duration", "300", "--steps", "4", "--batch-size", "1",
        ],
        "description": "SDXL Turbo image generation (4-step distilled, fp16)",
        "default_timeout": 420,
    },
    "lora_finetune_qwen25_0p5b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "lora_finetune.py"),
            "--base", "qwen25_0p5b", "--steps", "200",
            "--batch-size", "2", "--seq-len", "256",
        ],
        "description": "LoRA fine-tune Qwen 2.5 0.5B on WikiText-2",
        "default_timeout": 600,
    },
    "lora_finetune_smollm2_135m": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "lora_finetune.py"),
            "--base", "smollm2_135m", "--steps", "300",
            "--batch-size", "4", "--seq-len", "256",
        ],
        "description": "LoRA fine-tune SmolLM2 135M on WikiText-2 (smallest training run)",
        "default_timeout": 600,
    },
    "yolo_train_coco128": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "yolo_train.py"),
            "--duration", "300", "--imgsz", "320", "--batch", "16",
        ],
        "description": "YOLOv8n training on COCO128 — object detection (different gradient pattern than classification)",
        "default_timeout": 420,
    },
    "llm_infer_qwen25_3b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "qwen25_3b",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 3B inference — larger end of the consumer LLM tier",
        "default_timeout": 420,
    },

    # --- Tier 3: whitebox evasion strategies (Round 5 in the paper) ---
    # The paper identifies WB-L (LoRA + heavy dilution) as the single
    # evasion family that drops below 50% detection. We replicate the 4
    # whitebox variants on AMD so we can test cross-vendor classifier
    # robustness to whitebox attack.
    "whitebox_WB-I_inference": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-I", "--duration", "300",
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--batch-size", "1", "--seq-len", "256",
        ],
        "description": "WB-I: pure inference baseline (negative control)",
        "default_timeout": 420,
    },
    "whitebox_WB-D_diluted": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-D", "--duration", "300",
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--batch-size", "1", "--seq-len", "256", "--dilution", "10",
        ],
        "description": "WB-D: diluted training (backward 1/10 batches) with whitebox knowledge",
        "default_timeout": 420,
    },
    "whitebox_WB-L_lora": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-L", "--duration", "300",
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--batch-size", "1", "--seq-len", "256", "--dilution", "10",
        ],
        "description": "WB-L: LoRA adapters + heavy dilution (paper headline hardest strategy)",
        "default_timeout": 420,
    },
    "whitebox_WB-O_oracle": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-O", "--duration", "300",
            "--model", "Qwen/Qwen2.5-0.5B-Instruct",
            "--batch-size", "1", "--seq-len", "256",
        ],
        "description": "WB-O: oracle-rate-limited (memory-level spoofing, paper Stage-2 WB defense target)",
        "default_timeout": 420,
    },

    # --- Tier 3 extended: paper-canonical whitebox sweep (matches NVIDIA registry) ---
    # The existing whitebox_WB-* entries use Qwen 0.5B + dilution=10. The
    # paper's Round-5 sweep uses TinyLlama-1.1B + grad-ckpt + dilution
    # N in {5, 10, 20}. These entries match the paper exactly so we can
    # compute the dilution-vs-detection curve on AMD comparable to NVIDIA
    # Figure 2 / Table 7. Names match TRAINING_LABELS / INFERENCE_LABELS in
    # classifier/threeway_improved.py.
    "whitebox_inference": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-I", "--batch-size", "1", "--seq-len", "256",
            "--duration", "300",
        ],
        "description": "White-box inference baseline: TinyLlama-1.1B forward-only (paper WB-I)",
        "default_timeout": 420,
    },
    "whitebox_diluted_N5": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-D", "--dilution", "5", "--grad-ckpt",
            "--batch-size", "1", "--seq-len", "256", "--duration", "300",
        ],
        "description": "White-box diluted full fine-tuning: TinyLlama-1.1B, N=5 (20%)",
        "default_timeout": 420,
    },
    "whitebox_diluted_N20": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-D", "--dilution", "20", "--grad-ckpt",
            "--batch-size", "1", "--seq-len", "256", "--duration", "300",
        ],
        "description": "White-box diluted full fine-tuning: TinyLlama-1.1B, N=20 (5%)",
        "default_timeout": 420,
    },
    "whitebox_lora_N5": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-L", "--dilution", "5", "--grad-ckpt",
            "--batch-size", "2", "--seq-len", "256", "--duration", "300",
        ],
        "description": "White-box diluted LoRA fine-tuning: TinyLlama-1.1B, N=5 (20%)",
        "default_timeout": 420,
    },
    "whitebox_lora_N20": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-L", "--dilution", "20", "--grad-ckpt",
            "--batch-size", "2", "--seq-len", "256", "--duration", "300",
        ],
        "description": "White-box diluted LoRA fine-tuning: TinyLlama-1.1B, N=20 (5%)",
        "default_timeout": 420,
    },
    "whitebox_oracle": {
        # Requires ClassifierOracle with amdsmi backend (added 2026-05-17).
        # The oracle polls amdsmi at 1 Hz, runs the same classifier the
        # defender uses, and rate-limits training steps when P(training)
        # exceeds --oracle-threshold. Paper's Stage-2 WB defense target.
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
            "--strategy", "WB-O", "--grad-ckpt",
            "--batch-size", "1", "--seq-len", "256", "--duration", "300",
            "--oracle-threshold", "0.3",
        ],
        "description": "White-box oracle-guided fine-tuning: TinyLlama-1.1B, in-loop classifier feedback",
        "default_timeout": 420,
    },
}


def get_workload(label: str) -> dict:
    if label not in WORKLOAD_REGISTRY_AMD:
        available = ", ".join(sorted(WORKLOAD_REGISTRY_AMD.keys()))
        raise ValueError(f"Unknown AMD workload '{label}'. Available: {available}")
    return WORKLOAD_REGISTRY_AMD[label]


def list_workloads() -> list[str]:
    return sorted(WORKLOAD_REGISTRY_AMD.keys())
