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
