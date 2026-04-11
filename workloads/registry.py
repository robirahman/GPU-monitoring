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
