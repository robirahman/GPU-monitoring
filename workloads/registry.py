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
    # --- Workload-diversity round (2026-09-23): multi-node REPLICATED negatives ---
    # These are the existing single-GPU non-LLM kernels at --duration 600, driven
    # by mn_replicated_main so eight independent replicas run per node. The
    # existing llm_infer_* entries above are 300 s and the command is baked into
    # the registry, so a duration change needs its own entry rather than a flag.
    #
    # Purpose: the honest not-training class is nominally ten families but really
    # three programs (nonml_distributed.py, vLLM, TPU-JAX), and held-out
    # infershard_qwen is falsely flagged on 71% of its windows. These are six
    # genuinely different inference programs, which is what that number needs.
    "mn_infer_vit_l_d600": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "vit_l", "--duration", "600", "--batch-size", "64",
        ],
        "description": "ViT-L/16 classification, 600 s — encoder-only vision, multi-node replicated",
        "default_timeout": 900,
    },
    "mn_infer_bge_embed_d600": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "bge_large", "--duration", "600", "--batch-size", "128",
        ],
        "description": "BGE-large text embedding, 600 s — bidirectional encoder, multi-node replicated",
        "default_timeout": 900,
    },
    "mn_infer_clip_embed_d600": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "clip_l14", "--duration", "600", "--batch-size", "256",
        ],
        "description": "CLIP ViT-L/14 image embedding, 600 s — dual-encoder, multi-node replicated",
        "default_timeout": 900,
    },
    "mn_infer_sdxl_d600": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference.py"),
            "--model", "sdxl", "--duration", "600",
            "--sdxl-steps", "30", "--sdxl-size", "1024",
        ],
        "description": "SDXL diffusion, 600 s — iterative denoising, the only non-autoregressive generator",
        "default_timeout": 1200,
    },
    "mn_infer_whisper_d600": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "modern_inference.py"),
            "--model", "whisper_hf", "--duration", "600",
        ],
        "description": "Whisper large-v3-turbo ASR via transformers, 600 s — encoder-decoder",
        "default_timeout": 900,
    },
    "mn_infer_rn50_d600": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "inference.py"),
            "--model", "resnet50", "--batch-size", "256", "--duration", "600",
        ],
        "description": "ResNet-50 classification, 600 s — pure convolutional, no attention anywhere",
        "default_timeout": 900,
    },
    # llm_inference.py's MODEL_REGISTRY stops at qwen25_3b, so the 7B HF cell uses
    # llm_inference_hf_sharded.py, which draws its models from VLLM_MODELS and so
    # sits at exactly the scale the vLLM cells do. With one visible GPU per
    # replica, accelerate's device_map simply places the model on it.
    "mn_infer_hf_qwen7b_d600": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_hf_sharded.py"),
            "--model", "qwen25_7b_fp16", "--duration", "600", "--batch-size", "8",
            "--max-new-tokens", "128", "--prompt-tokens", "1024",
        ],
        "description": "Qwen2.5-7B via HF generate, 600 s — a SECOND LLM engine; every other LLM stream in the corpus is vLLM",
        "default_timeout": 900,
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
    "llm_infer_qwen25_1_5b_bs1": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_1_5b_fp16",
            "--duration", "300", "--batch-size", "1", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 1.5B FP16 via vLLM, batch 1 — latency-bound second "
                       "inference negative. Distinct workload from the batch-8 cell, "
                       "so it contributes an INDEPENDENT job rather than a correlated "
                       "repeat: each cell yields only one job per rep, and the ladder "
                       "needs >=8 jobs per class to clear the eval's low_n flag.",
        "default_timeout": 600,
    },
    "llm_infer_qwen25_1_5b_fp16": {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llm_inference_vllm.py"),
            "--model", "qwen25_1_5b_fp16",
            "--duration", "300", "--batch-size", "8", "--max-tokens", "256",
        ],
        "description": "Qwen 2.5 1.5B Instruct FP16 (~3 GB) via vLLM — matched-pair for fsdp_1b training",
        "default_timeout": 600,
    },
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
}


def get_workload(label: str) -> dict:
    """Look up a workload by label. Raises ValueError if not found."""
    if label not in WORKLOAD_REGISTRY:
        available = ", ".join(sorted(WORKLOAD_REGISTRY.keys()))
        raise ValueError(f"Unknown workload '{label}'. Available: {available}")
    return WORKLOAD_REGISTRY[label]


def list_workloads() -> list[str]:
    """Return sorted list of registered workload labels."""
    return sorted(WORKLOAD_REGISTRY.keys())
