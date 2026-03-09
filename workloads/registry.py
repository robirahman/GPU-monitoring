"""Workload registry — maps label strings to commands and metadata.

To add a new workload for Week 3+, add an entry to WORKLOAD_REGISTRY below.
Each entry needs:
    - "command": list of strings (subprocess args)
    - "description": human-readable description
    - "default_timeout": optional timeout in seconds
"""

import os

# Resolve paths relative to the repository root
_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOADS_DIR = os.path.join(_REPO_ROOT, "workloads")

WORKLOAD_REGISTRY = {
    # --- Baselines ---
    "idle": {
        "command": ["python3", os.path.join(_WORKLOADS_DIR, "idle.py")],
        "description": "Idle GPU baseline (120 seconds of sleep)",
        "default_timeout": 130,
    },
    # --- ML Training ---
    "pytorch_resnet_cifar10": {
        "command": [
            "python3",
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
            "python3",
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
            "python3",
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
            "python3",
            os.path.join(_WORKLOADS_DIR, "gpt2_finetune.py"),
            "--steps", "400",
            "--batch-size", "8",
        ],
        "description": "GPT-2 124M fine-tuning on WikiText-2, 400 steps, FP32",
        "default_timeout": 1200,
    },
    "gpt2_wikitext2_amp": {
        "command": [
            "python3",
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
            "python3",
            os.path.join(_WORKLOADS_DIR, "bert_finetune.py"),
            "--steps", "300",
            "--batch-size", "32",
        ],
        "description": "BERT-base fine-tuning on SST-2, 300 steps, FP32",
        "default_timeout": 900,
    },
    "bert_sst2_amp": {
        "command": [
            "python3",
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
            "python3",
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
            "python3",
            os.path.join(_WORKLOADS_DIR, "scientific_hpc.py"),
            "--workload", "cufft",
            "--duration", "600",
        ],
        "description": "3D cuFFT benchmark (memory + cache bandwidth), 600s",
        "default_timeout": 660,
    },
    "nbody_sim": {
        "command": [
            "python3",
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
            "python3",
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
            "python3",
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
            "python3",
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
            "python3",
            os.path.join(_WORKLOADS_DIR, "blender_render.py"),
            "--samples", "128",
            "--loops", "3",
        ],
        "description": "Blender Cycles BMW benchmark, CUDA rendering (requires blender)",
        "default_timeout": 900,
    },
    # --- Other ---
    "ffmpeg_nvenc": {
        "command": [
            "python3",
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
