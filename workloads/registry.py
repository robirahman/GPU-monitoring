"""Workload registry — maps label strings to commands and metadata.

To add a new workload for Week 3+, add an entry to WORKLOAD_REGISTRY below.
Each entry needs:
    - "command": list of strings (subprocess args)
    - "description": human-readable description
    - "default_timeout": optional timeout in seconds
"""

import os
import sys

# Resolve paths relative to the repository root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOADS_DIR = os.path.join(_REPO_ROOT, "workloads")

WORKLOAD_REGISTRY = {
    # --- Baselines ---
    "idle": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "idle.py")],
        "description": "Idle GPU baseline (120 seconds of sleep)",
        "default_timeout": 130,
    },
    # --- ML Training ---
    "pytorch_resnet_cifar10": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "resnet18",
            "--dataset", "cifar10",
            "--epochs", "10",
            "--batch-size", "512",
        ],
        "description": "ResNet-18 training on CIFAR-10, 10 epochs, FP32",
        "default_timeout": 900,
    },
    "pytorch_resnet_cifar10_amp": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "resnet18",
            "--dataset", "cifar10",
            "--epochs", "10",
            "--batch-size", "512",
            "--amp",
        ],
        "description": "ResNet-18 training on CIFAR-10, 10 epochs, mixed precision (AMP)",
        "default_timeout": 900,
    },
    "pytorch_mlp_cifar10": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "simple_mlp",
            "--dataset", "cifar10",
            "--epochs", "20",
            "--batch-size", "512",
        ],
        "description": "Simple MLP training on CIFAR-10, 20 epochs, FP32",
        "default_timeout": 900,
    },
    "gpt2_wikitext2": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "gpt2_finetune.py"),
            "--steps", "400",
            "--batch-size", "8",
        ],
        "description": "GPT-2 124M fine-tuning on WikiText-2, 400 steps, FP32",
        "default_timeout": 1200,
    },
    "gpt2_wikitext2_amp": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "gpt2_finetune.py"),
            "--steps", "400",
            "--batch-size", "8",
            "--amp",
        ],
        "description": "GPT-2 124M fine-tuning on WikiText-2, 400 steps, AMP",
        "default_timeout": 1200,
    },
    "bert_sst2": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "bert_finetune.py"),
            "--steps", "300",
            "--batch-size", "32",
        ],
        "description": "BERT-base fine-tuning on SST-2, 300 steps, FP32",
        "default_timeout": 900,
    },
    "bert_sst2_amp": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "bert_finetune.py"),
            "--steps", "300",
            "--batch-size", "32",
            "--amp",
        ],
        "description": "BERT-base fine-tuning on SST-2, 300 steps, AMP",
        "default_timeout": 900,
    },
    # --- ML Inference ---
    "resnet50_inference": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "inference.py"),
            "--model", "resnet50",
            "--batch-size", "256",
            "--duration", "600",
        ],
        "description": "ResNet-50 batch inference on CIFAR-10, 600s, no gradients",
        "default_timeout": 660,
    },
    # --- Scientific HPC ---
    "cufft_benchmark": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "scientific_hpc.py"),
            "--workload", "cufft",
            "--duration", "600",
        ],
        "description": "3D cuFFT benchmark (memory + cache bandwidth), 600s",
        "default_timeout": 660,
    },
    "nbody_sim": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "scientific_hpc.py"),
            "--workload", "nbody",
            "--duration", "600",
            "--n-bodies", "4096",
        ],
        "description": "Gravitational N-body simulation (compute-bound), 4096 bodies, 600s",
        "default_timeout": 660,
    },
    "gromacs_adh": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "gromacs_sim.py"),
            "--benchmark", "adh",
            "--duration", "600",
        ],
        "description": "GROMACS ADH benchmark MD simulation, GPU-accelerated (requires gromacs)",
        "default_timeout": 660,
    },
    # --- Crypto Mining ---
    "mining_ethash_proxy": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "mining_proxy.py"),
            "--duration", "600",
            "--dag-size-mb", "1024",
        ],
        "description": "Ethash-like crypto mining proxy (memory-hard), 1GB DAG, 600s",
        "default_timeout": 660,
    },
    # --- Rendering ---
    "rendering_proxy": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "rendering_proxy.py"),
            "--duration", "600",
            "--rays", "32768",
            "--bounces", "4",
        ],
        "description": "Monte Carlo path tracing proxy (irregular compute, no tensor cores), 600s",
        "default_timeout": 660,
    },
    "blender_bmw": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "blender_render.py"),
            "--samples", "1024",
            "--loops", "5",
        ],
        "description": "Blender Cycles BMW benchmark, CUDA rendering (requires blender)",
        "default_timeout": 900,
    },
    # --- Crypto Mining (standalone) ---
    "ethash_cuda": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "ethash_cuda.py"),
            "--duration", "300",
            "--dag-size", "1024",
        ],
        "description": "Ethash-like CUDA kernel (memory-hard hashing), safe for cloud",
        "default_timeout": 360,
    },
    # --- Standalone workload files (collaborator variants) ---
    "cufft_benchmark_standalone": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "cufft_benchmark.py"),
            "--duration", "300",
            "--size", "4096",
        ],
        "description": "cuFFT benchmark: repeated 4096x4096 FFTs on GPU (standalone)",
        "default_timeout": 360,
    },
    "nbody_sim_standalone": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "nbody_sim.py"),
            "--duration", "300",
            "--particles", "16384",
        ],
        "description": "N-body gravitational simulation, 16K particles, FP32 (standalone)",
        "default_timeout": 360,
    },
    "resnet50_inference_standalone": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "resnet50_inference.py"),
            "--batch-size", "256",
            "--loops", "10",
        ],
        "description": "ResNet-50 inference on CIFAR-10 test set (standalone)",
        "default_timeout": 600,
    },
    # --- Adversarial Training (evasion strategies) ---
    "adversarial_A_util_modulation": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "adversarial_training.py"),
            "--strategy", "A",
            "--duration", "300",
            "--sleep-min", "0.1",
            "--sleep-max", "0.5",
        ],
        "description": "Strategy A: Training with sleep gaps to mimic rendering bursts",
        "default_timeout": 360,
    },
    "adversarial_B_low_util": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "adversarial_training.py"),
            "--strategy", "B",
            "--duration", "300",
        ],
        "description": "Strategy B: Tiny-batch CPU-bottlenecked training to mimic inference",
        "default_timeout": 360,
    },
    # Composite adversarial with parameterized idle fraction
    # (registered dynamically by sweep script; these are common presets)
    "adversarial_composite_00": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
                    "--idle-fraction", "0.0", "--duration", "120"],
        "description": "Composite adversarial: 0% idle (baseline, full throughput)",
        "default_timeout": 180,
    },
    "adversarial_composite_30": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
                    "--idle-fraction", "0.3", "--duration", "120"],
        "description": "Composite adversarial: 30% idle",
        "default_timeout": 180,
    },
    "adversarial_composite_50": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
                    "--idle-fraction", "0.5", "--duration", "120"],
        "description": "Composite adversarial: 50% idle",
        "default_timeout": 180,
    },
    "adversarial_composite_70": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
                    "--idle-fraction", "0.7", "--duration", "120"],
        "description": "Composite adversarial: 70% idle",
        "default_timeout": 180,
    },
    "adversarial_composite_90": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
                    "--idle-fraction", "0.9", "--duration", "120"],
        "description": "Composite adversarial: 90% idle",
        "default_timeout": 180,
    },
    "adversarial_composite_95": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_composite.py"),
                    "--idle-fraction", "0.95", "--duration", "120"],
        "description": "Composite adversarial: 95% idle",
        "default_timeout": 240,
    },
    "adversarial_D_temporal_disruption": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "adversarial_training.py"),
            "--strategy", "D",
            "--duration", "300",
        ],
        "description": "Strategy D: Random batch sizes + CUDA noise to break periodicity",
        "default_timeout": 360,
    },
    # --- Modern Inference ---
    "deepseek_r1_inference": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "modern_inference.py"),
            "--model", "deepseek_r1",
            "--duration", "300",
            "--max-tokens", "256",
        ],
        "description": "DeepSeek-R1-Distill-Qwen-1.5B autoregressive text generation",
        "default_timeout": 420,
    },
    "qwen25_inference": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "modern_inference.py"),
            "--model", "qwen25",
            "--duration", "300",
            "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 0.5B autoregressive text generation",
        "default_timeout": 420,
    },
    "whisper_inference": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "modern_inference.py"),
            "--model", "whisper",
            "--duration", "300",
        ],
        "description": "Whisper large-v3-turbo audio transcription (synthetic audio)",
        "default_timeout": 420,
    },
    # --- Round 5b: Large-LLM inference anchors (single-GPU) ---
    "llm_infer_phi3_mini": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "phi3_mini",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "Phi-3-Mini-4K 3.8B, FP16, bs=1, seq=256 — 5080-class single-GPU",
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
    "llm_infer_qwen25_3b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "qwen25_3b",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 3B Instruct, FP16 — open-license 3B inference anchor (replaces Llama 3.2 3B)",
        "default_timeout": 420,
    },
    "llm_infer_smollm2_1p7b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "smollm2_1p7b",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "SmolLM2 1.7B Instruct (HuggingFace, Apache 2.0) — vendor/arch diversity (replaces Gemma 2 2B)",
        "default_timeout": 420,
    },
    "llm_infer_mistral_7b_q4": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "mistral_7b_q4",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "Mistral 7B Instruct GPTQ 4-bit — quantised inference path (gptqmodel backend on Linux)",
        "default_timeout": 480,
    },
    "llm_infer_tinyllama_bs8": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "tinyllama",
            "--duration", "300", "--batch-size", "8", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "TinyLlama-1.1B bs=8 — batch sweep vs Round 5 whitebox_inference (bs=1)",
        "default_timeout": 420,
    },
    "llm_infer_tinyllama_seq1024": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "tinyllama",
            "--duration", "300", "--batch-size", "1", "--seq-len", "1024",
            "--max-tokens", "256",
        ],
        "description": "TinyLlama-1.1B seq=1024 — sequence-length sweep",
        "default_timeout": 420,
    },
    "llm_infer_qwen25_3b_bs4": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "qwen25_3b",
            "--duration", "300", "--batch-size", "4", "--seq-len", "256",
            "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 3B bs=4 — scale x batch cell (replaces Llama 3.2 3B bs=4)",
        "default_timeout": 420,
    },
    "llm_infer_qwen25_1p5b_streaming": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "qwen25_1p5b",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--max-tokens", "1024",
        ],
        "description": "Qwen 2.5 1.5B long-generation — KV cache growth pattern",
        "default_timeout": 480,
    },
    # --- Round 5b (phase 3): Long-context LLM inference (KV-cache growth anchors) ---
    "llm_infer_phi3_mini_seq2048": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "phi3_mini",
            "--duration", "300", "--batch-size", "1", "--seq-len", "2048",
            "--max-tokens", "256",
        ],
        "description": "Phi-3-Mini seq=2048 — long-context prefill + decode",
        "default_timeout": 480,
    },
    "llm_infer_tinyllama_seq2048": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "tinyllama",
            "--duration", "300", "--batch-size", "1", "--seq-len", "2048",
            "--max-tokens", "256",
        ],
        "description": "TinyLlama-1.1B seq=2048 — mid KV-cache sweep cell",
        "default_timeout": 480,
    },
    "llm_infer_tinyllama_seq4096": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "tinyllama",
            "--duration", "300", "--batch-size", "1", "--seq-len", "4096",
            "--max-tokens", "512",
        ],
        "description": "TinyLlama-1.1B seq=4096 — large KV-cache decode (memory-growing inference anchor)",
        "default_timeout": 540,
    },
    "llm_infer_qwen25_1p5b_seq4096": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "qwen25_1p5b",
            "--duration", "300", "--batch-size", "1", "--seq-len", "4096",
            "--max-tokens", "512",
        ],
        "description": "Qwen 2.5 1.5B seq=4096 — GQA KV-cache shape at large context",
        "default_timeout": 540,
    },
    # --- Round 5b (phase 3): Large-batch inference (sustained high-util regime) ---
    "llm_infer_qwen25_3b_bs16": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "qwen25_3b",
            "--duration", "300", "--batch-size", "16", "--seq-len", "256",
            "--max-tokens", "128",
        ],
        "description": "Qwen 2.5 3B bs=16 — sustained high-util inference",
        "default_timeout": 480,
    },
    "llm_infer_clip_embed_bs256": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "clip_l14",
            "--duration", "300", "--batch-size", "256",
        ],
        "description": "CLIP ViT-L/14 image embedding @ batch 256 — max-throughput feature extractor",
        "default_timeout": 420,
    },
    "llm_infer_bge_embed_bs128": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "bge_large",
            "--duration", "300", "--batch-size", "128",
        ],
        "description": "BGE-large-en-v1.5 text embedding @ batch 128 — bidirectional encoder inference",
        "default_timeout": 420,
    },
    # --- Round 5b (phase 3): Vision transformer classification ---
    "llm_infer_vit_l_bs64": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "vit_l",
            "--duration", "300", "--batch-size", "64",
        ],
        "description": "ViT-L/16 ImageNet classification @ batch 64 — non-CLIP encoder-only vision path",
        "default_timeout": 420,
    },
    # --- Round 5b (phase 4): Matched-pair LoRA training (consumer tier) ---
    # Paired with existing llm_infer_* workloads at the same model scale so
    # the classifier can learn train-vs-infer on the same GPU / same model.
    "llm_train_qwen25_1p5b_lora": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "lora_training.py"),
            "--model", "Qwen/Qwen2.5-1.5B-Instruct",
            "--duration", "300", "--batch-size", "2", "--seq-len", "256",
            "--rank", "16",
        ],
        "description": "Qwen 2.5 1.5B LoRA FT — matched-pair train for llm_infer_qwen25_1p5b",
        "default_timeout": 480,
    },
    "llm_train_smollm2_1p7b_lora": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "lora_training.py"),
            "--model", "HuggingFaceTB/SmolLM2-1.7B-Instruct",
            "--duration", "300", "--batch-size", "2", "--seq-len", "256",
            "--rank", "16",
        ],
        "description": "SmolLM2 1.7B LoRA FT — matched-pair train for llm_infer_smollm2_1p7b",
        "default_timeout": 480,
    },
    "llm_train_qwen25_3b_lora": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "lora_training.py"),
            "--model", "Qwen/Qwen2.5-3B-Instruct",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--rank", "16",
        ],
        "description": "Qwen 2.5 3B LoRA FT — matched-pair train for llm_infer_qwen25_3b",
        "default_timeout": 540,
    },
    "llm_train_phi3_mini_lora": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "lora_training.py"),
            "--model", "microsoft/Phi-3-mini-4k-instruct",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--rank", "16", "--grad-ckpt",
            "--target-modules", "qkv_proj,o_proj",
        ],
        "description": "Phi-3-Mini 3.8B LoRA FT (grad-ckpt on, qkv_proj fused) — matched-pair train for llm_infer_phi3_mini",
        "default_timeout": 600,
    },
    "llm_train_mistral_7b_lora": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "lora_training.py"),
            "--model", "mistralai/Mistral-7B-Instruct-v0.2",
            "--duration", "300", "--batch-size", "1", "--seq-len", "256",
            "--rank", "16", "--grad-ckpt",
        ],
        "description": "Mistral 7B LoRA FT (grad-ckpt on) — matched-pair train for llm_infer_mistral_7b_q4. Won't fit 16 GB; run on 24 GB+ hardware.",
        "default_timeout": 720,
    },
    # --- Round 5b: Non-LLM inference (underrepresented modalities) ---
    "llm_infer_sdxl": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "sdxl",
            "--duration", "300", "--sdxl-steps", "30", "--sdxl-size", "1024",
        ],
        "description": "Stable Diffusion XL 1024x1024 / 30 steps — diffusion path",
        "default_timeout": 600,
    },
    "llm_infer_clip_embed_batched": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "clip_l14",
            "--duration", "300", "--batch-size", "64",
        ],
        "description": "CLIP ViT-L/14 image embedding @ batch 64 — feature-extractor path",
        "default_timeout": 420,
    },
    # --- Round 5b (phase 5): Matched-pair dense inference at FSDP sizes ---
    # 7B / 14B / 32B FP16 via vLLM — single-GPU, fits any datacenter GPU
    # (A100-80, H100-80, H200, B200). Paired with existing FSDP-7B/13B/30B
    # training runs so the classifier can learn train-vs-infer at matched
    # scale on datacenter hardware.
    "llm_infer_qwen25_7b_fp16": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_7b_fp16",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 7B Instruct FP16 (~14 GB) via vLLM — matched-pair for fsdp_7b training",
        "default_timeout": 600,
    },
    "llm_infer_qwen25_14b_fp16": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_14b_fp16",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 14B Instruct FP16 (~28 GB) via vLLM — matched-pair for fsdp_13b training",
        "default_timeout": 720,
    },
    "llm_infer_qwen25_32b_fp16": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_32b_fp16",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 32B Instruct FP16 (~64 GB) via vLLM — matched-pair for fsdp_30b training",
        "default_timeout": 900,
    },
    # --- Round 5b: Datacenter-only large-LLM inference (>80 GB memory) ---
    # Must be run on H200 (141 GB) or B200 (192 GB); won't fit on A100-80/5080/5090.
    "llm_infer_qwen25_72b_fp16": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_72b_fp16",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 72B Instruct FP16 (~144 GB) via vLLM — open-license frontier dense inference (replaces Llama 3.3 70B)",
        "default_timeout": 900,
    },
    "llm_infer_mixtral_8x7b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "mixtral_8x7b",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
        ],
        "description": "Mixtral 8x7B MoE (~90 GB FP16) via vLLM — sparse expert activation",
        "default_timeout": 900,
    },
    # --- SPAR M0: prefill / decode length grid (dense, fits H100-80) ---
    "llm_infer_qwen25_7b_prefill_heavy": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_7b_fp16",
            "--duration", "300", "--batch-size", "4",
            "--input-len", "8192", "--max-tokens", "16",
            "--max-model-len", "8704",
        ],
        "description": "SPAR M0: Qwen2.5-7B prefill-heavy (8k in / 16 out)",
        "default_timeout": 600,
    },
    "llm_infer_qwen25_7b_decode_heavy": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_7b_fp16",
            "--duration", "300", "--batch-size", "4",
            "--input-len", "512", "--max-tokens", "1024",
            "--max-model-len", "2048",
        ],
        "description": "SPAR M0: Qwen2.5-7B decode-heavy (512 in / 1k out)",
        "default_timeout": 600,
    },
    "llm_infer_qwen25_7b_balanced": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_7b_fp16",
            "--duration", "300", "--batch-size", "4",
            "--input-len", "2048", "--max-tokens", "256",
            "--max-model-len", "4096",
        ],
        "description": "SPAR M0: Qwen2.5-7B balanced (2k in / 256 out)",
        "default_timeout": 600,
    },
    # --- SPAR M1: midfill (prefix cache + short append) ---
    "llm_midfill_qwen25_7b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_7b_fp16",
            "--mode", "midfill",
            "--duration", "300", "--batch-size", "2",
            "--prefix-len", "8192", "--append-len", "256", "--max-tokens", "128",
            "--max-model-len", "16384",
        ],
        "description": "SPAR M1: Qwen2.5-7B midfill (8k prefix cache + 256 append)",
        "default_timeout": 720,
    },
    # --- SPAR M0/M3: Mixtral MoE length grid on TP=2 (H100-80) ---
    "llm_infer_mixtral_tp2_prefill_heavy": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "mixtral_8x7b_tp2",
            "--tensor-parallel-size", "2",
            "--duration", "300", "--batch-size", "4",
            "--input-len", "4096", "--max-tokens", "16",
            "--max-model-len", "4608",
        ],
        "description": "SPAR M0/M3: Mixtral-8x7B TP2 prefill-heavy (real MoE weights)",
        "default_timeout": 900,
    },
    "llm_infer_mixtral_tp2_decode_heavy": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "mixtral_8x7b_tp2",
            "--tensor-parallel-size", "2",
            "--duration", "300", "--batch-size", "4",
            "--input-len", "512", "--max-tokens", "1024",
            "--max-model-len", "2048",
        ],
        "description": "SPAR M0/M3: Mixtral-8x7B TP2 decode-heavy (real MoE weights)",
        "default_timeout": 900,
    },
    "llm_midfill_mixtral_tp2": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "mixtral_8x7b_tp2",
            "--tensor-parallel-size", "2",
            "--mode", "midfill",
            "--duration", "300", "--batch-size", "2",
            "--prefix-len", "8192", "--append-len", "256", "--max-tokens", "128",
            "--max-model-len", "16384",
        ],
        "description": "SPAR M1/M3: Mixtral-8x7B TP2 midfill (real MoE + prefix cache)",
        "default_timeout": 1200,
    },
    # --- SPAR M2: synthetic MoE EP AllToAll (interconnect FP family) ---
    "moe_ep_synthetic": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "moe_ep_synthetic.py"),
            "--duration", "300",
            "--layers", "32", "--hidden", "4096",
            "--tokens-per-step", "8", "--experts-per-rank", "4",
            "--comm-log", "data/moe_ep_comm.jsonl",
        ],
        "description": "[MICROBENCH ONLY — not SPAR primary corpus] SPAR M2: synthetic MoE EP AllToAll (dispatch+combine per layer)",
        "default_timeout": 420,
    },
    "moe_ep_synthetic_fake_ar": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "moe_ep_synthetic.py"),
            "--duration", "300",
            "--layers", "32", "--hidden", "4096",
            "--tokens-per-step", "8", "--experts-per-rank", "4",
            "--fake-ar-every", "50", "--fake-ar-mb", "64",
            "--comm-log", "data/moe_ep_fake_ar_comm.jsonl",
        ],
        "description": "[MICROBENCH ONLY — not SPAR primary corpus] SPAR M2: MoE EP AllToAll + rare fake AllReduce (hybrid FP)",
        "default_timeout": 420,
    },
    # --- SPAR M4: real MoE serving with --enable-expert-parallel (EP=TP×DP) ---
    # Launch via: python scripts/run_workload_multi_gpu.py --workload <label> --num-gpus 8
    # (vLLM spawns its own workers; do NOT wrap in torchrun.)
    "moe_serve_olmoe_ep8_decode": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "olmoe_1b7b",
            "--tp", "1", "--dp", "8", "--enable-expert-parallel",
            "--traffic", "decode", "--concurrency", "8", "--duration", "300",
            "--port", "8000",
            "--traffic-log", "data/serving_olmoe_ep8_decode.jsonl",
        ],
        "description": "SPAR M4: OLMoE-1B-7B EP=8 decode traffic (AllToAll fingerprint smoke)",
        "default_timeout": 2400,
    },
    "moe_serve_olmoe_ep8_mixed": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "olmoe_1b7b",
            "--tp", "1", "--dp", "8", "--enable-expert-parallel",
            "--enable-prefix-caching",
            "--traffic", "mixed", "--concurrency", "8", "--duration", "300",
            "--port", "8000",
            "--traffic-log", "data/serving_olmoe_ep8_mixed.jsonl",
        ],
        "description": "SPAR M4/M5: OLMoE EP=8 mixed prefill/midfill/decode traffic",
        "default_timeout": 2400,
    },
    "moe_serve_mixtral_ep8_decode": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "mixtral_8x7b",
            "--tp", "1", "--dp", "8", "--enable-expert-parallel",
            "--traffic", "decode", "--concurrency", "8", "--duration", "300",
            "--max-model-len", "8192", "--port", "8000",
            "--traffic-log", "data/serving_mixtral_ep8_decode.jsonl",
        ],
        "description": "SPAR M4: Mixtral-8x7B EP=8 (TP1×DP8) decode — real MoE AllToAll",
        "default_timeout": 3600,
    },
    "moe_serve_mixtral_tp2dp4_ep_decode": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "mixtral_8x7b",
            "--tp", "2", "--dp", "4", "--enable-expert-parallel",
            "--traffic", "decode", "--concurrency", "8", "--duration", "300",
            "--max-model-len", "8192", "--port", "8000",
            "--traffic-log", "data/serving_mixtral_tp2dp4_ep_decode.jsonl",
        ],
        "description": "SPAR M4: Mixtral EP=8 via TP2×DP4 decode (hybrid attention/expert split)",
        "default_timeout": 3600,
    },
    # --- SPAR M5: production-ish mixed traffic (+ dense TP control) ---
    "moe_serve_mixtral_ep8_mixed": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "mixtral_8x7b",
            "--tp", "2", "--dp", "4", "--enable-expert-parallel",
            "--enable-prefix-caching",
            "--traffic", "mixed", "--concurrency", "8", "--duration", "600",
            "--max-model-len", "16384", "--port", "8000",
            "--traffic-log", "data/serving_mixtral_ep_mixed.jsonl",
        ],
        "description": "SPAR M5: Mixtral EP serving + mixed prefill/midfill/decode traffic",
        "default_timeout": 4200,
    },
    "moe_serve_mixtral_ep8_midfill": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "mixtral_8x7b",
            "--tp", "2", "--dp", "4", "--enable-expert-parallel",
            "--enable-prefix-caching",
            "--traffic", "midfill", "--concurrency", "4", "--duration", "300",
            "--max-model-len", "16384", "--port", "8000",
            "--traffic-log", "data/serving_mixtral_ep_midfill.jsonl",
        ],
        "description": "SPAR M5: Mixtral EP midfill-only (long prefix + short append)",
        "default_timeout": 3600,
    },
    # Qwen2.5-7B has 28 attention heads → TP must divide 28 (TP=2/4 OK; TP=8 INVALID in vLLM).
    "moe_serve_qwen7b_tp4dp2_mixed": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "qwen25_7b",
            "--tp", "4", "--dp", "2",
            "--enable-prefix-caching",
            "--traffic", "mixed", "--concurrency", "8", "--duration", "300",
            "--port", "8000",
            "--traffic-log", "data/serving_qwen7b_tp4dp2_mixed.jsonl",
        ],
        "description": "SPAR M5 control: dense Qwen2.5-7B TP4×DP2 mixed traffic (no EP; replaces broken tp8)",
        "default_timeout": 2400,
    },
    # Deprecated alias — old name kept so scripts don't break; same as tp4dp2 (NOT real TP=8).
    "moe_serve_qwen7b_tp8_mixed": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "qwen25_7b",
            "--tp", "4", "--dp", "2",
            "--enable-prefix-caching",
            "--traffic", "mixed", "--concurrency", "8", "--duration", "300",
            "--port", "8000",
            "--traffic-log", "data/serving_qwen7b_tp4dp2_mixed.jsonl",
        ],
        "description": "[ALIAS→tp4dp2] Was TP=8 (invalid: 28 heads %% 8 != 0). Now identical to moe_serve_qwen7b_tp4dp2_mixed",
        "default_timeout": 2400,
    },
    "llm_infer_qwen25_72b_fp8": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_72b_fp8",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 72B AWQ-4bit via vLLM — quantised-mega inference path",
        "default_timeout": 900,
    },
    "llm_infer_qwen25_72b_tp2": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_72b_tp2",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
            "--tensor-parallel-size", "2",
        ],
        "description": "Qwen 2.5 72B FP16 tensor-parallel across 2 GPUs — open-license NVLink all-reduce signature (replaces Llama 3.1 70B TP2)",
        "default_timeout": 900,
    },
    "llm_infer_deepseek_v3_tp8": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "deepseek_v3_tp8",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
            "--tensor-parallel-size", "8",
        ],
        "description": "DeepSeek-V3 671B MoE (~37B active, FP8) tensor-parallel across 8 GPUs — open-license frontier stretch (replaces Llama 3.1 405B)",
        # First rep has to download ~685 GB of weights; subsequent reps reuse
        # the HF cache and finish in ~15 min. 2 h accommodates a 1 Gbps pipe.
        "default_timeout": 7200,
    },
    # --- Advanced Adversarial ---
    "adversarial_E_memory_minimal": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "adversarial_advanced.py"),
            "--strategy", "E",
            "--duration", "300",
        ],
        "description": "Strategy E: Gradient checkpointing + tiny batch to minimize memory",
        "default_timeout": 360,
    },
    "adversarial_F_interleave_30": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "adversarial_advanced.py"),
            "--strategy", "F",
            "--duration", "300",
            "--train-frac", "0.3",
        ],
        "description": "Strategy F: 30% training interleaved with mining/cuFFT",
        "default_timeout": 360,
    },
    "adversarial_F_interleave_50": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "adversarial_advanced.py"),
            "--strategy", "F",
            "--duration", "300",
            "--train-frac", "0.5",
        ],
        "description": "Strategy F: 50% training interleaved with mining/cuFFT",
        "default_timeout": 360,
    },
    "adversarial_G_clock_throttled": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "adversarial_advanced.py"),
            "--strategy", "G",
            "--duration", "300",
            "--power-limit", "100",
        ],
        "description": "Strategy G: Training at 100W power limit with locked low clocks",
        "default_timeout": 360,
    },
    # --- Round 2 Adversarial (counter-hardened classifier) ---
    "adversarial_H_mimicry_mining": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
                    "--strategy", "H", "--duration", "300", "--target", "mining"],
        "description": "Strategy H: Training disguised as mining via feedback-controlled mimicry",
        "default_timeout": 360,
    },
    "adversarial_H_mimicry_cufft": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
                    "--strategy", "H", "--duration", "300", "--target", "cufft"],
        "description": "Strategy H: Training disguised as cuFFT via feedback-controlled mimicry",
        "default_timeout": 360,
    },
    "adversarial_I_stochastic_30": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
                    "--strategy", "I", "--duration", "300", "--train-frac", "0.3"],
        "description": "Strategy I: Stochastic interleaving (30% training, random phases)",
        "default_timeout": 360,
    },
    "adversarial_J_pid_75": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round2.py"),
                    "--strategy", "J", "--duration", "300", "--target-util", "75"],
        "description": "Strategy J: PID-controlled util-matched training (target 75%)",
        "default_timeout": 360,
    },
    # --- Round 3 Adversarial (strongest possible: training-as-inference) ---
    "adversarial_K_online_learning": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "K", "--duration", "300", "--batch-size", "128"],
        "description": "Strategy K: Online learning — inference + backward pass per batch",
        "default_timeout": 360,
    },
    "adversarial_L_diluted_2": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--duration", "300", "--dilution", "2",
                    "--batch-size", "128"],
        "description": "Strategy L: Diluted training — backward pass on 1/2 batches (batch 128)",
        "default_timeout": 360,
    },
    "adversarial_L_diluted_5": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--duration", "300", "--dilution", "5",
                    "--batch-size", "128"],
        "description": "Strategy L: Diluted training — backward pass on 1/5 batches (batch 128)",
        "default_timeout": 360,
    },
    "adversarial_L_diluted_10": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--duration", "300", "--dilution", "10",
                    "--batch-size", "128"],
        "description": "Strategy L: Diluted training — backward pass on 1/10 batches (batch 128)",
        "default_timeout": 360,
    },
    "adversarial_L_diluted_20": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--duration", "300", "--dilution", "20"],
        "description": "Strategy L: Diluted training — backward pass on 1/20 batches",
        "default_timeout": 360,
    },
    # --- Round 4 Adversarial (targeting LOO weaknesses) ---
    "adversarial_M_composite_memmin_10": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
                    "--strategy", "M", "--idle-fraction", "0.1", "--duration", "300"],
        "description": "Strategy M: Composite + memory minimal, 10% idle",
        "default_timeout": 360,
    },
    "adversarial_M_composite_memmin_20": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
                    "--strategy", "M", "--idle-fraction", "0.2", "--duration", "300"],
        "description": "Strategy M: Composite + memory minimal, 20% idle",
        "default_timeout": 360,
    },
    "adversarial_M_composite_memmin_30": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
                    "--strategy", "M", "--idle-fraction", "0.3", "--duration", "300"],
        "description": "Strategy M: Composite + memory minimal, 30% idle",
        "default_timeout": 360,
    },
    "adversarial_M_composite_memmin_40": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
                    "--strategy", "M", "--idle-fraction", "0.4", "--duration", "300"],
        "description": "Strategy M: Composite + memory minimal, 40% idle",
        "default_timeout": 360,
    },
    "adversarial_N_grad_accum_4": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
                    "--strategy", "N", "--accum-steps", "4", "--duration", "300"],
        "description": "Strategy N: Gradient accumulation dilution, step every 4 batches",
        "default_timeout": 360,
    },
    "adversarial_N_grad_accum_8": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
                    "--strategy", "N", "--accum-steps", "8", "--duration", "300"],
        "description": "Strategy N: Gradient accumulation dilution, step every 8 batches",
        "default_timeout": 360,
    },
    "adversarial_N_grad_accum_16": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round4.py"),
                    "--strategy", "N", "--accum-steps", "16", "--duration", "300"],
        "description": "Strategy N: Gradient accumulation dilution, step every 16 batches",
        "default_timeout": 360,
    },
    # --- Controlled Dilution Sweep (consistent batch size = 128, ResNet-50) ---
    # Pure inference baseline: same model/batch/data as Strategy L, no training
    "dilution_sweep_inference": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "inference.py"),
                    "--model", "resnet50", "--batch-size", "128", "--duration", "300"],
        "description": "Dilution sweep: pure inference (ResNet-50, batch 128, 300s)",
        "default_timeout": 360,
    },
    # N=1: every batch trains (= online learning with same code path as diluted)
    "dilution_sweep_N1": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--dilution", "1", "--batch-size", "128", "--duration", "300"],
        "description": "Dilution sweep: N=1, every batch trains (batch 128, 300s)",
        "default_timeout": 360,
    },
    "dilution_sweep_N2": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--dilution", "2", "--batch-size", "128", "--duration", "300"],
        "description": "Dilution sweep: N=2, 50% training fraction (batch 128, 300s)",
        "default_timeout": 360,
    },
    "dilution_sweep_N3": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--dilution", "3", "--batch-size", "128", "--duration", "300"],
        "description": "Dilution sweep: N=3, 33% training fraction (batch 128, 300s)",
        "default_timeout": 360,
    },
    "dilution_sweep_N5": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--dilution", "5", "--batch-size", "128", "--duration", "300"],
        "description": "Dilution sweep: N=5, 20% training fraction (batch 128, 300s)",
        "default_timeout": 360,
    },
    "dilution_sweep_N10": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--dilution", "10", "--batch-size", "128", "--duration", "300"],
        "description": "Dilution sweep: N=10, 10% training fraction (batch 128, 300s)",
        "default_timeout": 360,
    },
    "dilution_sweep_N20": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_round3.py"),
                    "--strategy", "L", "--dilution", "20", "--batch-size", "128", "--duration", "300"],
        "description": "Dilution sweep: N=20, 5% training fraction (batch 128, 300s)",
        "default_timeout": 360,
    },
    # --- Other ---
    "ffmpeg_nvenc": {
        "command": [
            sys.executable,
            os.path.join(_WORKLOADS_DIR, "ffmpeg_nvenc.py"),
            "--duration", "300",
        ],
        "description": "FFmpeg NVENC hardware video encoding (requires ffmpeg + NVENC)",
        "default_timeout": 360,
    },
    # --- DDP (Multi-GPU) Training ---
    # Note: these use torchrun, not sys.executable. Launch via run_workload_multi_gpu.py.
    "ddp_resnet18_cifar10": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "pytorch_training_ddp.py"),
            "--model", "resnet18", "--duration", "300",
        ],
        "description": "DDP ResNet-18 training on 8 GPUs, CIFAR-10, 300s",
        "default_timeout": 360,
    },
    "ddp_resnet50_cifar10": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "pytorch_training_ddp.py"),
            "--model", "resnet50", "--duration", "300",
        ],
        "description": "DDP ResNet-50 training on 8 GPUs, CIFAR-10, 300s",
        "default_timeout": 360,
    },
    "ddp_resnet18_cifar10_amp": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "pytorch_training_ddp.py"),
            "--model", "resnet18", "--duration", "300", "--amp",
        ],
        "description": "DDP ResNet-18 AMP training on 8 GPUs, CIFAR-10, 300s",
        "default_timeout": 360,
    },
    # --- DDP Adversarial ---
    "adversarial_K_ddp": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "K", "--duration", "300",
        ],
        "description": "DDP online learning camouflage on 8 GPUs, 300s",
        "default_timeout": 360,
    },
    "adversarial_L_ddp_5": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "L", "--dilution", "5", "--duration", "300",
        ],
        "description": "DDP diluted training 1/5 on 8 GPUs, 300s",
        "default_timeout": 360,
    },
    "adversarial_L_ddp_10": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "L", "--dilution", "10", "--duration", "300",
        ],
        "description": "DDP diluted training 1/10 on 8 GPUs, 300s",
        "default_timeout": 360,
    },
    "adversarial_L_ddp_stagger_5": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "L-stagger", "--dilution", "5", "--duration", "300",
        ],
        "description": "DDP staggered diluted training 1/5 on 8 GPUs, 300s",
        "default_timeout": 360,
    },
    "adversarial_K_ddp_accum_4": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "K-accum", "--accum", "4", "--duration", "300",
        ],
        "description": "DDP online learning with 4-step gradient accumulation, 300s",
        "default_timeout": 360,
    },
    "adversarial_K_ddp_accum_8": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "K-accum", "--accum", "8", "--duration", "300",
        ],
        "description": "DDP online learning with 8-step gradient accumulation, 300s",
        "default_timeout": 360,
    },
    # --- Round 5: White-Box Evasion (relevant model scale) ---
    # Default model: TinyLlama-1.1B (fills 16 GB for full AdamW training)
    # All WB-D/WB-L entries use --grad-ckpt and --batch-size 1 for 16 GB GPUs
    "whitebox_inference": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-I", "--batch-size", "1", "--seq-len", "256",
                    "--duration", "300"],
        "description": "White-box inference baseline: TinyLlama-1.1B forward-only",
        "default_timeout": 420,
    },
    "whitebox_diluted_N5": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-D", "--dilution", "5", "--grad-ckpt",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted full fine-tuning: TinyLlama-1.1B, N=5 (20%)",
        "default_timeout": 420,
    },
    "whitebox_diluted_N10": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-D", "--dilution", "10", "--grad-ckpt",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted full fine-tuning: TinyLlama-1.1B, N=10 (10%)",
        "default_timeout": 420,
    },
    "whitebox_diluted_N20": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-D", "--dilution", "20", "--grad-ckpt",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted full fine-tuning: TinyLlama-1.1B, N=20 (5%)",
        "default_timeout": 420,
    },
    "whitebox_diluted_N50": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-D", "--dilution", "50", "--grad-ckpt",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted full fine-tuning: TinyLlama-1.1B, N=50 (2%)",
        "default_timeout": 420,
    },
    "whitebox_diluted_N5_ckpt": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-D", "--dilution", "5",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted full fine-tuning WITHOUT grad ckpt: 1.1B, N=5",
        "default_timeout": 420,
    },
    "whitebox_diluted_N10_ckpt": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-D", "--dilution", "10",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted full fine-tuning WITHOUT grad ckpt: 1.1B, N=10",
        "default_timeout": 420,
    },
    "whitebox_diluted_N20_ckpt": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-D", "--dilution", "20",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted full fine-tuning WITHOUT grad ckpt: 1.1B, N=20",
        "default_timeout": 420,
    },
    # LoRA variants — much smaller optimizer overhead, closer to inference
    "whitebox_lora_N5": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-L", "--dilution", "5", "--grad-ckpt",
                    "--batch-size", "2", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted LoRA fine-tuning: TinyLlama-1.1B, N=5 (20%)",
        "default_timeout": 420,
    },
    "whitebox_lora_N10": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-L", "--dilution", "10", "--grad-ckpt",
                    "--batch-size", "2", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted LoRA fine-tuning: TinyLlama-1.1B, N=10 (10%)",
        "default_timeout": 420,
    },
    "whitebox_lora_N20": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-L", "--dilution", "20", "--grad-ckpt",
                    "--batch-size", "2", "--seq-len", "256", "--duration", "300"],
        "description": "White-box diluted LoRA fine-tuning: TinyLlama-1.1B, N=20 (5%)",
        "default_timeout": 420,
    },
    "whitebox_oracle": {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, "adversarial_whitebox.py"),
                    "--strategy", "WB-O", "--dilution", "10", "--grad-ckpt",
                    "--batch-size", "1", "--seq-len", "256", "--duration", "300"],
        "description": "White-box oracle-guided full training: TinyLlama-1.1B, classifier in loop",
        "default_timeout": 600,
    },
    # --- FSDP (NVLink Stress) ---
    # Model-sharded training via FSDP FULL_SHARD. Heavy AllGather/ReduceScatter
    # on every forward/backward step. Launch via run_workload_multi_gpu.py.
    "fsdp_7b_nvlink": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "fsdp_training.py"),
            "--size", "7b", "--duration", "300",
        ],
        "description": "FSDP 6.7B transformer on 8 GPUs, NVLink stress (300s)",
        "default_timeout": 420,
    },
    "fsdp_13b_nvlink": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "fsdp_training.py"),
            "--size", "13b", "--duration", "300",
        ],
        "description": "FSDP 13B transformer on 8 GPUs, NVLink stress (300s)",
        "default_timeout": 420,
    },
    "fsdp_30b_nvlink": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "fsdp_training.py"),
            "--size", "30b", "--duration", "600",
        ],
        "description": "FSDP 30B transformer on 8 GPUs, heavy NVLink stress (600s)",
        "default_timeout": 720,
    },
    "fsdp_30b_adamw_nvlink": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "fsdp_training.py"),
            "--size", "30b", "--duration", "600",
            "--optimizer", "adamw",
        ],
        "description": "FSDP 30B transformer on 8 GPUs, AdamW optimizer, NVLink stress (600s)",
        "default_timeout": 720,
    },
    "fsdp_70b_nvlink": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "fsdp_training.py"),
            "--size", "70b", "--duration", "600",
        ],
        "description": "FSDP 70B transformer on 8 GPUs, maximum NVLink stress (600s)",
        "default_timeout": 780,
    },
    # --- SPAR interconnect workloads (W10, W15-W19, W23) ---
    # Launch via run_workload_multi_gpu.py --num-gpus 8. Every entry writes a rank0 JSONL
    # comm log (one collective per line) + <stem>.meta.json; see docs/interconnect_W_id_file_map.md.
    # W10: disaggregated prefill<->decode KV handoff -- inference that looks loud (FP weapon)
    "disagg_kv_synth_1gb": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "disagg_kv_synth.py"),
            "--split", "half", "--pattern", "pair",
            "--kv-mb", "1024", "--gap-s", "1", "--duration", "300",
            "--comm-log", "data/disagg_kv_1gb_comm.jsonl",
        ],
        "description": "[MICROBENCH ONLY — not SPAR primary corpus] W10: synthetic disagg KV handoff, 1 GiB P2P per prefill->decode pair every ~1s (FP)",
        "default_timeout": 420,
    },
    "disagg_kv_synth_256mb_burst": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "disagg_kv_synth.py"),
            "--split", "half", "--pattern", "pair",
            "--kv-mb", "256", "--burst", "8", "--gap-s", "5", "--duration", "300",
            "--comm-log", "data/disagg_kv_256mb_burst_comm.jsonl",
        ],
        "description": "[MICROBENCH ONLY — not SPAR primary corpus] W10: synthetic disagg KV handoff, bursts of 8 x 256 MiB then 5s idle (FP)",
        "default_timeout": 420,
    },
    # W18: real DDP training whose gradient sync is KV-shaped (FN twin of W10)
    "train_kv_disguise_ddp": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "train_kv_disguise.py"),
            "--mode", "ddp", "--size", "150m", "--block-mb", "256", "--accum-steps", "4",
            "--duration", "300", "--comm-log", "data/kv_disguise_ddp_comm.jsonl",
        ],
        "description": "[HAND SCHEDULE — runnable now] W18: 150M DDP, AllReduce in 256 MiB KV-sized blocks (FN). After W10: prefer train_kv_disguise_ddp_realprof",
        "default_timeout": 420,
    },
    "train_kv_disguise_p2p": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "train_kv_disguise.py"),
            "--mode", "p2p", "--size", "150m", "--block-mb", "256",
            "--duration", "300", "--comm-log", "data/kv_disguise_p2p_comm.jsonl",
        ],
        "description": "[HAND SCHEDULE — runnable now] W18: 150M DDP, pairwise P2P in 256 MiB blocks (FN). After W10: prefer train_kv_disguise_p2p_realprof",
        "default_timeout": 420,
    },
    # W17: PowerSGD -- synced payload << |theta|
    "ddp_powersgd_resnet18": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "powersgd_ddp.py"),
            "--model", "resnet18", "--powersgd-rank", "4", "--start-powerSGD-iter", "10",
            "--duration", "300", "--comm-log", "data/powersgd_resnet18_comm.jsonl",
        ],
        "description": "W17: DDP ResNet-18 with PowerSGD rank-4 gradient compression, synthetic CIFAR-shaped data (FN)",
        "default_timeout": 420,
    },
    "ddp_powersgd_150m": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "powersgd_ddp.py"),
            "--model", "150m", "--powersgd-rank", "4", "--start-powerSGD-iter", "10",
            "--batch-size", "4", "--seq-len", "512",
            "--duration", "300", "--comm-log", "data/powersgd_150m_comm.jsonl",
        ],
        "description": "W17: DDP 150M causal LM with PowerSGD rank-4 gradient compression (FN)",
        "default_timeout": 420,
    },
    "ddp_topk_resnet18": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "powersgd_ddp.py"),
            "--model", "resnet18", "--hook", "topk", "--topk-ratio", "0.01", "--topk-start-iter", "10",
            "--duration", "300", "--comm-log", "data/topk_resnet18_comm.jsonl",
        ],
        "description": "W17: DDP ResNet-18 with 1% Top-k sparsified gradients + error feedback (all_gather, no AllReduce) (FN)",
        "default_timeout": 420,
    },
    "ddp_topk_150m": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "powersgd_ddp.py"),
            "--model", "150m", "--hook", "topk", "--topk-ratio", "0.01", "--topk-start-iter", "10",
            "--batch-size", "4", "--seq-len", "512",
            "--duration", "300", "--comm-log", "data/topk_150m_comm.jsonl",
        ],
        "description": "W17: DDP 150M causal LM with 1% Top-k sparsified gradients + error feedback (FN)",
        "default_timeout": 420,
    },
    # W16: LoRA-only cross-rank sync -- MB-scale adapters, frozen base never synced
    "train_lora_only_ddp_qwen15": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "lora_ddp.py"),
            "--model", "Qwen/Qwen2.5-1.5B-Instruct",
            "--batch-size", "2", "--seq-len", "256", "--duration", "300",
            "--comm-log", "data/lora_ddp_qwen15_comm.jsonl",
        ],
        "description": "W16: Qwen2.5-1.5B LoRA (r=16 q,v) DDP on 8 GPUs, only adapters AllReduced (~9 MB/step) (FN)",
        "default_timeout": 600,
    },
    "train_lora_only_ddp_tinyllama": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "lora_ddp.py"),
            "--model", "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
            "--batch-size", "2", "--seq-len", "256", "--duration", "300",
            "--comm-log", "data/lora_ddp_tinyllama_comm.jsonl",
        ],
        "description": "W16: TinyLlama-1.1B LoRA (r=16 q,v) DDP on 8 GPUs, only adapters AllReduced (FN)",
        "default_timeout": 600,
    },
    # W15: grad-accum / dilution with interconnect-grade logs (wraps adversarial_ddp.py --comm-log)
    "train_ddp_rare_sync_accum8": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "K-accum", "--accum", "8", "--duration", "300",
            "--comm-log", "data/ddp_rare_sync_accum8_comm.jsonl",
        ],
        "description": "W15: DDP ResNet-50, AllReduce every 8th micro-batch, per-bucket comm log + step windows (FN)",
        "default_timeout": 360,
    },
    "train_ddp_rare_sync_accum4": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "K-accum", "--accum", "4", "--duration", "300",
            "--comm-log", "data/ddp_rare_sync_accum4_comm.jsonl",
        ],
        "description": "W15: DDP ResNet-50, AllReduce every 4th micro-batch, per-bucket comm log + step windows (FN)",
        "default_timeout": 360,
    },
    "train_ddp_dilute_1of10": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "L", "--dilution", "10", "--duration", "300",
            "--comm-log", "data/ddp_dilute_1of10_comm.jsonl",
        ],
        "description": "W15: DDP ResNet-50 training on 1/10 batches, forward-only otherwise, comm log + step windows (FN)",
        "default_timeout": 360,
    },
    "train_ddp_dilute_1of5": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "adversarial_ddp.py"),
            "--strategy", "L", "--dilution", "5", "--duration", "300",
            "--comm-log", "data/ddp_dilute_1of5_comm.jsonl",
        ],
        "description": "W15: DDP ResNet-50 training on 1/5 batches, forward-only otherwise, comm log + step windows (FN)",
        "default_timeout": 360,
    },
    # W19: decode-shaped small AllReduces every step, full |theta| sync every H steps
    "train_hybrid_wire_H50": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "hybrid_wire.py"),
            "--size", "1b", "--sync-every", "50", "--batch-size", "4", "--seq-len", "512",
            "--duration", "300", "--comm-log", "data/hybrid_wire_H50_comm.jsonl",
        ],
        "description": "W19: 1.3B training, W08-sized decode-shaped AllReduces every step + full grad sync every 50 steps (FN)",
        "default_timeout": 420,
    },
    "train_hybrid_wire_H500": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "hybrid_wire.py"),
            "--size", "1b", "--sync-every", "500", "--batch-size", "4", "--seq-len", "512",
            "--duration", "600", "--comm-log", "data/hybrid_wire_H500_comm.jsonl",
        ],
        "description": "W19: 1.3B training, W08-sized decode-shaped AllReduces every step + full grad sync every 500 steps (FN)",
        "default_timeout": 720,
    },
    # W25 PROBE (run last): temporal mix = W19 hybrid_wire @ large H.
    # NOT a clean train|infer label — classifier behavior probe on mixed jobs. Primary corpus ends at W24.
    "train_serve_cover_H20": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "hybrid_wire.py"),
            "--size", "1b", "--sync-every", "20", "--batch-size", "4", "--seq-len", "512",
            "--duration", "300", "--comm-log", "data/serve_cover_H20_comm.jsonl",
        ],
        "description": "[PROBE — not primary train|infer label] W25: 1.3B + W08 decode cover every step, grad sync every 20 (study detector on temporal mix)",
        "default_timeout": 420,
    },
    "train_serve_cover_H100": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "hybrid_wire.py"),
            "--size", "1b", "--sync-every", "100", "--batch-size", "4", "--seq-len", "512",
            "--duration", "600", "--comm-log", "data/serve_cover_H100_comm.jsonl",
        ],
        "description": "[PROBE — not primary train|infer label] W25: 1.3B + W08 decode cover every step, grad sync every 100 (study detector on temporal mix)",
        "default_timeout": 720,
    },
    "train_serve_cover_lora_H20": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "lora_ddp.py"),
            "--model", "Qwen/Qwen2.5-1.5B-Instruct", "--batch-size", "2", "--seq-len", "256",
            "--sync-every", "20", "--cover", "--duration", "300",
            "--comm-log", "data/serve_cover_lora_H20_comm.jsonl",
        ],
        "description": "[PROBE — not primary train|infer label] W25: Qwen2.5-1.5B LoRA sync every 20 under W08 cover (study detector on temporal mix)",
        "default_timeout": 600,
    },
    # W26 PROBE (run last): spatial colocation — breaks one-job=one-label; not a clean class.
    "colocate_serve7_train1_qwen": {
        "command": [
            sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_colocate_serve_train.py"),
            "--label", "colocate_serve7_train1_qwen",
            "--serve-workload", "moe_serve_qwen7b_tp4dp2_mixed", "--serve-gpus", "0-6",
            "--serve-args", "--tp 1 --dp 7 --traffic-log data/colocate_serve7_train1_qwen_serve_traffic.jsonl",
            "--train-workload", "train_lora_only_ddp_qwen15", "--train-gpus", "7",
            "--train-args", "--comm-log data/colocate_serve7_train1_qwen_train_comm.jsonl",
            "--duration", "300",
        ],
        "telemetry_owner": "scripts/run_colocate_serve_train.py",
        "description": "[PROBE — not primary train|infer label] W26: serve GPUs 0-6 || LoRA GPU 7 (study node- vs per-GPU detector)",
        "default_timeout": 4200,
    },
    "colocate_serve_tp4_train_ddp2": {
        "command": [
            sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_colocate_serve_train.py"),
            "--label", "colocate_serve_tp4_train_ddp2",
            "--serve-workload", "moe_serve_qwen7b_tp4dp2_mixed", "--serve-gpus", "0-3",
            "--serve-args", "--tp 4 --dp 1 --traffic-log data/colocate_serve_tp4_train_ddp2_serve_traffic.jsonl",
            "--train-workload", "ddp_resnet18_cifar10", "--train-gpus", "4-5",
            "--duration", "300",
        ],
        "telemetry_owner": "scripts/run_colocate_serve_train.py",
        "description": "[PROBE — not primary train|infer label] W26: TP4 serve 0-3 || DDP train 4-5 (study spatial mix with fabric on both sides)",
        "default_timeout": 4200,
    },
    # W23: single-node DDP/FSDP under an explicit IB-quiet harness (NCCL pinned to NVLink,
    # sidecar JSON + IB port-counter deltas). Wrapper, not a fork of the workloads.
    "train_ddp_nvl_only_ibquiet": {
        "command": [
            sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_nvl_only_ibquiet.py"),
            "--sidecar", "data/train_ddp_nvl_only_ibquiet.json", "--nccl-evidence", "--",
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "pytorch_training_ddp.py"),
            "--model", "resnet50", "--duration", "300",
        ],
        "description": "W23: single-node 8-GPU DDP ResNet-50, NCCL pinned to NVLink, IB counters sampled (NVLink busy / IB quiet)",
        "default_timeout": 420,
    },
    "train_fsdp7b_nvl_only_ibquiet": {
        "command": [
            sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_nvl_only_ibquiet.py"),
            "--sidecar", "data/train_fsdp7b_nvl_only_ibquiet.json", "--nccl-evidence", "--",
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "fsdp_training.py"),
            "--size", "7b", "--duration", "300",
        ],
        "description": "W23: single-node 8-GPU FSDP 6.7B, NCCL pinned to NVLink, IB counters sampled (NVLink busy / IB quiet)",
        "default_timeout": 480,
    },
    # W05: MoE expert-parallel TRAINING fingerprint (AllToAll dispatch/combine, forward AND backward).
    # Synthetic (weights shared across layers) -- counterpart of the infer-shaped moe_ep_synthetic.
    "moe_ep_train_synthetic": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "moe_ep_train_synthetic.py"),
            "--duration", "300", "--layers", "16", "--hidden", "4096",
            "--tokens-per-step", "4096", "--experts-per-rank", "2",
            "--comm-log", "data/moe_ep_train_comm.jsonl",
        ],
        "description": "[MICROBENCH ONLY — not SPAR primary corpus] W05: synthetic MoE EP=8 training, A2A dispatch+combine fwd+bwd per layer (4 A2A/layer/step)",
        "default_timeout": 420,
    },
    "moe_ep_train_synthetic_dp": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "moe_ep_train_synthetic.py"),
            "--duration", "300", "--layers", "16", "--hidden", "4096",
            "--tokens-per-step", "4096", "--experts-per-rank", "2",
            "--ep-size", "4", "--dense-sync",
            "--comm-log", "data/moe_ep_train_dp_comm.jsonl",
        ],
        "description": "[MICROBENCH ONLY — not SPAR primary corpus] W05: synthetic MoE training EP=4 x DP=2, A2A fwd+bwd + dense-grad and expert-grad AllReduce",
        "default_timeout": 420,
    },
    # W04: pipeline-parallel 1F1B training -- P2P activations, no gradient AllReduce.
    # world_size = PP stages: launch with --num-gpus 2 / 4 (matches nproc_per_node).
    "train_pp2_1f1b_150m": {
        "command": [
            "torchrun", "--nproc_per_node=2",
            os.path.join(_WORKLOADS_DIR, "pp_1f1b_train.py"),
            "--size", "150m", "--microbatches", "8", "--micro-batch-size", "4", "--seq-len", "512",
            "--duration", "300", "--comm-log", "data/pp2_1f1b_150m_comm.jsonl", "--comm-log-all-ranks",
        ],
        "description": "[SCHEDULE MICROBENCH — hand-rolled 1F1B; primary: train_pp4_1f1b_qwen3_1p7b_titan] W04: 150M causal LM, 2-stage 1F1B pipeline (M=8), P2P activations only -- run with --num-gpus 2",
        "default_timeout": 420,
    },
    "train_pp4_1f1b_1b": {
        "command": [
            "torchrun", "--nproc_per_node=4",
            os.path.join(_WORKLOADS_DIR, "pp_1f1b_train.py"),
            "--size", "1b", "--microbatches", "16", "--micro-batch-size", "2", "--seq-len", "1024",
            "--duration", "300", "--comm-log", "data/pp4_1f1b_1b_comm.jsonl", "--comm-log-all-ranks",
        ],
        "description": "[SCHEDULE MICROBENCH — hand-rolled 1F1B; primary: train_pp4_1f1b_qwen3_1p7b_titan] W04: 1.3B causal LM, 4-stage 1F1B pipeline (M=16), P2P activations only -- run with --num-gpus 4",
        "default_timeout": 420,
    },
    # W03: tensor-parallel + data-parallel training (Megatron-lite): TP AllReduce on activations
    # + DP AllReduce on grads. world = TP x DP.
    "train_tp2_dp4_1b": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "tp_dp_train.py"),
            "--tp", "2", "--size", "1b", "--batch-size", "4", "--seq-len", "1024",
            "--duration", "300", "--comm-log", "data/tp2_dp4_1b_comm.jsonl",
        ],
        "description": "[PARALLELISM UNIT / ABLATION — replicated vocab, random-init; primary: train_tp2_*_titan] W03: 1.3B training, TP=2 x DP=4 on 8 GPUs, per-layer TP AllReduce + DP grad AllReduce",
        "default_timeout": 420,
    },
    "train_tp2_dp2_150m": {
        "command": [
            "torchrun", "--nproc_per_node=4",
            os.path.join(_WORKLOADS_DIR, "tp_dp_train.py"),
            "--tp", "2", "--size", "150m", "--batch-size", "4", "--seq-len", "512",
            "--duration", "300", "--comm-log", "data/tp2_dp2_150m_comm.jsonl",
        ],
        "description": "[PARALLELISM UNIT / ABLATION — replicated vocab, random-init; primary: train_tp2_*_titan] W03: 150M training, TP=2 x DP=2 on 4 GPUs, per-layer TP AllReduce + DP grad AllReduce -- run with --num-gpus 4",
        "default_timeout": 420,
    },
    # W13 / W14: production DiLoCo (single node here; multi-node via scripts/run_diloco_vcf.sh).
    # Defaults mirror run_diloco_vcf.sh: 1B model, fp16 outer AllReduce, synthetic tokens.
    "diloco_1b_H500": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "diloco_production.py"),
            "--model-size", "1b", "--local-steps", "500", "--reduce-dtype", "fp16",
            "--max-steps", "1500", "--batch-size", "1", "--seq-len", "1024", "--precision", "bf16",
            "--step-log", "data/diloco_1b_H500_steps.jsonl",
            "--comm-log", "data/diloco_1b_H500_comm.jsonl",
        ],
        "description": "W13: DiLoCo 1.3B, 8 replicas, H=500 inner steps, fp16 outer AllReduce (3 outer syncs) (FN)",
        "default_timeout": 1800,
    },
    "diloco_streaming_1b_H512": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "diloco_production.py"),
            "--model-size", "1b", "--local-steps", "512", "--reduce-dtype", "fp16",
            "--streaming", "--num-fragments", "8", "--fragment-sync-delay", "4",
            "--max-steps", "1536", "--batch-size", "1", "--seq-len", "1024", "--precision", "bf16",
            "--step-log", "data/diloco_streaming_1b_H512_steps.jsonl",
            "--comm-log", "data/diloco_streaming_1b_H512_comm.jsonl",
        ],
        "description": "W14: Streaming DiLoCo 1.3B, 8 replicas, H=512, 8 fragments, overlapped fp16 outer sync (FN)",
        "default_timeout": 1800,
    },
    # ══ REAL-framework primary corpus (docs/REAL_WORKLOAD_BAR.md) ══════════════════════════════
    # Real vLLM / torchtitan processes only. No collective JSONL is invented: serve/traffic logs +
    # NVML/DCGM/IB counters (telemetry launcher) + optional NCCL_DEBUG COLL parsing.
    # Status for all of these: code (real) -- NOT yet run on GPUs from this repo.
    #
    # W10: REAL disaggregated prefill<->decode (vLLM NixlConnector; 4 prefill + 4 decode engines, TP=1).
    # Needs `pip install vllm nixl`. Handoff log -> scripts/build_kv_profile.py -> W18 *_realprof.
    "disagg_kv_vllm_qwen7b": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "disagg_kv_vllm.py"),
            "--model", "qwen25_7b", "--num-prefill", "4", "--num-decode", "4",
            "--max-model-len", "16384", "--concurrency", "16", "--duration", "300",
            "--prompt-len-mix", "512:0.3,2048:0.5,8192:0.2",
            "--handoff-log", "data/disagg_kv_vllm_qwen7b_handoffs.jsonl",
            "--traffic-log", "data/disagg_kv_vllm_qwen7b_traffic.jsonl",
        ],
        "description": "W10 PRIMARY (real): vLLM NixlConnector disaggregated serving, Qwen2.5-7B, 4 prefill + 4 decode GPUs, mixed prompt lengths (FP)",
        "default_timeout": 3600,
    },
    "disagg_kv_vllm_qwen7b_longctx": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "disagg_kv_vllm.py"),
            "--model", "qwen25_7b", "--num-prefill", "4", "--num-decode", "4",
            "--max-model-len", "32768", "--concurrency", "8", "--duration", "300",
            "--prompt-len-mix", "16384:0.5,24576:0.5",
            "--handoff-log", "data/disagg_kv_vllm_qwen7b_longctx_handoffs.jsonl",
            "--traffic-log", "data/disagg_kv_vllm_qwen7b_longctx_traffic.jsonl",
        ],
        "description": "W10 PRIMARY (real): vLLM NixlConnector disaggregated serving, Qwen2.5-7B, 16k-24k prompts (~1-1.4 GB KV per handoff) (FP)",
        "default_timeout": 3600,
    },
    # W18 calibrated: needs W10 profile. If missing, train_kv_disguise falls back to --block-mb
    # with a loud warning (so smoke isn't blocked before the first disagg run).
    "train_kv_disguise_ddp_realprof": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "train_kv_disguise.py"),
            "--mode", "ddp", "--size", "150m", "--accum-steps", "4",
            "--schedule-from", "data/profiles/kv_handoff_disagg_kv_vllm_qwen7b.json",
            "--block-mb", "256",
            "--duration", "300", "--comm-log", "data/kv_disguise_ddp_realprof_comm.jsonl",
        ],
        "description": "W18: 150M DDP, AllReduce schedule from real W10 KV profile (falls back to --block-mb 256 if profile missing)",
        "default_timeout": 420,
    },
    "train_kv_disguise_p2p_realprof": {
        "command": [
            "torchrun", "--nproc_per_node=8",
            os.path.join(_WORKLOADS_DIR, "train_kv_disguise.py"),
            "--mode", "p2p", "--size", "150m",
            "--schedule-from", "data/profiles/kv_handoff_disagg_kv_vllm_qwen7b.json",
            "--block-mb", "256",
            "--duration", "300", "--comm-log", "data/kv_disguise_p2p_realprof_comm.jsonl",
        ],
        "description": "W18: 150M P2P disguise from real W10 KV profile (falls back to --block-mb 256 if profile missing)",
        "default_timeout": 420,
    },
    # W20: REAL vLLM serve + co-tenant fake AllReduce padding (co-tenant's own collectives logged; vLLM untouched).
    # NOTE: not tp8 -- Qwen2.5-7B has 28 attention heads, so vLLM cannot shard it 8-way (28 % 8 != 0).
    "moe_serve_qwen7b_tp4dp2_fake_ar": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "qwen25_7b", "--tp", "4", "--dp", "2", "--enable-prefix-caching",
            "--gpu-memory-utilization", "0.85",
            "--traffic", "mixed", "--concurrency", "8", "--duration", "300", "--port", "8000",
            "--traffic-log", "data/serving_qwen7b_tp4dp2_fake_ar.jsonl",
            "--cotenant-fake-ar", "--fake-ar-every-s", "2", "--fake-ar-mb", "64",
            "--fake-ar-log", "data/serving_qwen7b_tp4dp2_fake_ar_cotenant.jsonl",
        ],
        "description": "W20 PRIMARY (real serve): vLLM Qwen2.5-7B TP4xDP2 mixed traffic + co-tenant fake AllReduce (64 MiB / 2 s) padding (FP)",
        "default_timeout": 2700,
    },
    "moe_serve_mixtral_ep8_fake_ar": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_moe_serving.py"),
            "--model", "mixtral_8x7b", "--tp", "2", "--dp", "4", "--enable-expert-parallel",
            "--enable-prefix-caching", "--gpu-memory-utilization", "0.85",
            "--traffic", "mixed", "--concurrency", "8", "--duration", "300",
            "--max-model-len", "16384", "--port", "8000",
            "--traffic-log", "data/serving_mixtral_ep8_fake_ar.jsonl",
            "--cotenant-fake-ar", "--fake-ar-every-s", "2", "--fake-ar-mb", "64",
            "--fake-ar-log", "data/serving_mixtral_ep8_fake_ar_cotenant.jsonl",
        ],
        "description": "W20 PRIMARY (real serve): vLLM Mixtral-8x7B EP=8 mixed traffic + co-tenant fake AllReduce (64 MiB / 2 s) padding (FP)",
        "default_timeout": 4200,
    },
    # W05 / W03 / W04: REAL torchtitan training (jobs in scripts/titan_jobs.py; `pip install torchtitan`;
    # weights random-init unless --load-hf-weights; c4 streamed from the HF hub).
    "moe_ep_train_dsv3_16b_ep8": {
        "command": [sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_titan_job.py"),
                    "--job", "moe_ep_train_dsv3_16b_ep8", "--nccl-collective-log"],
        "description": "W05 PRIMARY (real): torchtitan DeepSeek-MoE-16B training, EP=8, framework-issued A2A dispatch/combine fwd+bwd",
        "default_timeout": 3600,
    },
    "moe_ep_train_qwen3_30b_a3b_ep8": {
        "command": [sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_titan_job.py"),
                    "--job", "moe_ep_train_qwen3_30b_a3b_ep8", "--nccl-collective-log"],
        "description": "W05 PRIMARY (real): torchtitan Qwen3-30B-A3B MoE training, EP=8 (memory fit unverified)",
        "default_timeout": 4200,
    },
    "train_tp2_dp4_qwen3_1p7b_titan": {
        "command": [sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_titan_job.py"),
                    "--job", "train_tp2_dp4_qwen3_1p7b_titan", "--nccl-collective-log"],
        "description": "W03 PRIMARY (real): torchtitan Qwen3-1.7B, TP=2 (SP off) x replicated DP=4 -> AllReduce on activations + grads",
        "default_timeout": 2400,
    },
    "train_tp2_fsdp4_qwen3_14b_titan": {
        "command": [sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_titan_job.py"),
                    "--job", "train_tp2_fsdp4_qwen3_14b_titan", "--nccl-collective-log"],
        "description": "W03 PRIMARY (real): torchtitan Qwen3-14B, TP=2 (SP on) x FSDP2=4 -> TP AG/RS + FSDP AG/RS",
        "default_timeout": 3000,
    },
    "train_pp4_1f1b_qwen3_1p7b_titan": {
        "command": [sys.executable, os.path.join(_REPO_ROOT, "scripts", "run_titan_job.py"),
                    "--job", "train_pp4_1f1b_qwen3_1p7b_titan", "--nccl-collective-log"],
        "description": "W04 PRIMARY (real): torchtitan Qwen3-1.7B, 4-stage 1F1B pipeline (torch.distributed.pipelining) -- run with --num-gpus 4",
        "default_timeout": 2400,
    },
}

# `microbench_` aliases: same entries, unambiguous names. The old labels keep working.
for _label in ("moe_ep_synthetic", "moe_ep_synthetic_fake_ar", "moe_ep_train_synthetic",
               "moe_ep_train_synthetic_dp", "disagg_kv_synth_1gb", "disagg_kv_synth_256mb_burst"):
    WORKLOAD_REGISTRY["microbench_" + _label] = WORKLOAD_REGISTRY[_label]


def get_workload(label: str) -> dict:
    """Look up a workload by label. Raises ValueError if not found."""
    if label not in WORKLOAD_REGISTRY:
        available = ", ".join(sorted(WORKLOAD_REGISTRY.keys()))
        raise ValueError(f"Unknown workload '{label}'. Available: {available}")
    return WORKLOAD_REGISTRY[label]


def list_workloads() -> list[str]:
    """Return sorted list of registered workload labels."""
    return sorted(WORKLOAD_REGISTRY.keys())
