"""
Centralized label manifest for Week 7 SPAR GPU workload classification.

All workload → category mappings live here. Both feature_engineering.py and
train_classifiers.py import from this module, eliminating the duplicated
hard-coded sets and heuristic guessing that existed before.

Categories:
  training  — ML/DL training workloads (gradient updates happening)
  inference — ML/DL inference only (no gradient)
  other     — non-ML GPU workloads (HPC, mining, rendering, idle)
"""

# ── Authoritative label → binary category mapping ──────────────────────────

LABEL_CATEGORY = {
    # Single-GPU ML training
    "pytorch_resnet_cifar10":       "training",
    "pytorch_resnet_cifar10_amp":   "training",
    "pytorch_mlp_cifar10":          "training",
    "bert_sst2":                    "training",
    "gpt2_wikitext2":               "training",
    "resnet_fp32":                  "training",
    "resnet_amp":                   "training",
    "mlp_training":                 "training",
    # Multi-GPU ML training
    "training_single_gpu":          "training",
    "training_dual_gpu_dp":         "training",
    # Edge cases that ARE training
    "BASELINE_TRAIN":               "training",
    "EC2":                          "training",   # silent training (no logging)
    "EC3":                          "training",   # very small model training
    "EC5":                          "training",   # frozen-backbone fine-tuning
    "EC6":                          "training",   # mixed precision + gradient accumulation

    # Inference
    "resnet50_inference":           "inference",
    "inference":                    "inference",
    "BASELINE_INFER":               "inference",
    "EC1":                          "inference",  # NVLink traffic during inference
    "EC4":                          "inference",  # heavy inference (batch inference)

    # Non-ML GPU workloads
    "idle":                         "other",
    "cufft_hpc":                    "other",
    "nbody_sim":                    "other",
    "mining_proxy":                 "other",
    "rendering":                    "other",

    # Dataset-scale DDP runs (labeled by convention)
    # These are generated dynamically as dscale_n{N} — handled by prefix rule below
}

# Prefix-based rules for dynamically-generated labels
PREFIX_RULES = [
    ("dscale_", "training"),        # dataset scaling DDP runs
    ("ddp_telemetry_", "training"), # DDP telemetry traces
]


def get_category(label: str) -> str:
    """Return 'training', 'inference', or 'other' for a workload label.

    Lookup order:
      1. Exact match in LABEL_CATEGORY
      2. Prefix match in PREFIX_RULES
      3. Default: 'other' (never guess from substring)
    """
    if label in LABEL_CATEGORY:
        return LABEL_CATEGORY[label]
    for prefix, cat in PREFIX_RULES:
        if label.startswith(prefix):
            return cat
    return "other"


def is_training(label: str) -> int:
    """Return 1 if training, 0 otherwise."""
    return int(get_category(label) == "training")


# ── Hardware constants for normalized power features ────────────────────────

GPU_TDP = {
    "NVIDIA B200":      1000.0,   # Blackwell TDP (W)
    "NVIDIA H100":       700.0,   # Hopper SXM5 TDP (W)
    "NVIDIA A100-SXM4-80GB": 400.0,
    "NVIDIA A100-SXM4-40GB": 400.0,
}

DEFAULT_TDP = 700.0  # fallback


def get_tdp(gpu_name) -> float:
    """Return TDP in watts for a GPU name string."""
    if not isinstance(gpu_name, str):
        return DEFAULT_TDP
    for key, tdp in GPU_TDP.items():
        if key in gpu_name:
            return tdp
    return DEFAULT_TDP


# ── Window sizes (canonical, used everywhere) ──────────────────────────────

WINDOW_CONFIGS = [
    {"window_sec": 5,  "stride_sec": 2,  "label": "5s"},
    {"window_sec": 15, "stride_sec": 7,  "label": "15s"},
    {"window_sec": 30, "stride_sec": 15, "label": "30s"},
]

WINDOW_SIZES = [c["window_sec"] for c in WINDOW_CONFIGS]

# Minimum samples for a valid window (prevents near-empty statistics)
MIN_SAMPLES_PER_WINDOW = 3
