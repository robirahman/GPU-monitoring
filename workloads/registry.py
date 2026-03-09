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
            "--epochs", "5",
            "--batch-size", "128",
        ],
        "description": "ResNet-18 training on CIFAR-10, 5 epochs, FP32",
        "default_timeout": 600,
    },
    "pytorch_resnet_cifar10_amp": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "resnet18",
            "--dataset", "cifar10",
            "--epochs", "5",
            "--batch-size", "128",
            "--amp",
        ],
        "description": "ResNet-18 training on CIFAR-10, 5 epochs, mixed precision (AMP)",
        "default_timeout": 600,
    },
    "pytorch_mlp_cifar10": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "pytorch_training.py"),
            "--model", "simple_mlp",
            "--dataset", "cifar10",
            "--epochs", "5",
            "--batch-size", "256",
        ],
        "description": "Simple MLP training on CIFAR-10, 5 epochs, FP32",
        "default_timeout": 300,
    },
    "gpt2_wikitext2": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "gpt2_finetune.py"),
            "--epochs", "3",
            "--batch-size", "4",
        ],
        "description": "GPT-2 124M fine-tuning on WikiText-2, 3 epochs, FP32",
        "default_timeout": 1200,
    },
    "gpt2_wikitext2_amp": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "gpt2_finetune.py"),
            "--epochs", "3",
            "--batch-size", "8",
            "--amp",
        ],
        "description": "GPT-2 124M fine-tuning on WikiText-2, 3 epochs, AMP",
        "default_timeout": 1200,
    },
    "bert_sst2": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "bert_finetune.py"),
            "--epochs", "3",
            "--batch-size", "32",
        ],
        "description": "BERT-base fine-tuning on SST-2, 3 epochs, FP32",
        "default_timeout": 900,
    },
    "bert_sst2_amp": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "bert_finetune.py"),
            "--epochs", "3",
            "--batch-size", "64",
            "--amp",
        ],
        "description": "BERT-base fine-tuning on SST-2, 3 epochs, AMP",
        "default_timeout": 900,
    },
    # --- ML Inference ---
    "resnet50_inference": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "resnet50_inference.py"),
            "--batch-size", "256",
            "--loops", "10",
        ],
        "description": "ResNet-50 inference on CIFAR-10 test set, no gradients",
        "default_timeout": 600,
    },
    # --- Scientific HPC ---
    "gromacs_adh": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "gromacs_sim.py"),
            "--benchmark", "adh",
            "--duration", "600",
        ],
        "description": "GROMACS ADH benchmark MD simulation, GPU-accelerated",
        "default_timeout": 660,
    },
    "cufft_benchmark": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "cufft_benchmark.py"),
            "--duration", "300",
            "--size", "4096",
        ],
        "description": "cuFFT benchmark: repeated 4096x4096 FFTs on GPU",
        "default_timeout": 360,
    },
    "nbody_sim": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "nbody_sim.py"),
            "--duration", "300",
            "--particles", "16384",
        ],
        "description": "N-body gravitational simulation, 16K particles, FP32",
        "default_timeout": 360,
    },
    # --- Crypto Mining ---
    "ethash_cuda": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "ethash_cuda.py"),
            "--duration", "300",
            "--dag-size", "1024",
        ],
        "description": "Ethash-like CUDA kernel (memory-hard hashing), safe for cloud",
        "default_timeout": 360,
    },
    # --- Rendering ---
    "blender_bmw": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "blender_render.py"),
            "--samples", "128",
            "--loops", "3",
        ],
        "description": "Blender Cycles BMW benchmark, CUDA rendering",
        "default_timeout": 900,
    },
    # --- Other ---
    "ffmpeg_nvenc": {
        "command": [
            "python3",
            os.path.join(_WORKLOADS_DIR, "ffmpeg_nvenc.py"),
            "--duration", "300",
        ],
        "description": "FFmpeg NVENC hardware video encoding",
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
