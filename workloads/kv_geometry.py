"""KV-cache geometry (pure Python, no torch) shared by W10 (synthetic + real) and W18.

``kv_bytes_per_token = 2 (K and V) * layers * kv_heads * head_dim * dtype_bytes``.
"""
from __future__ import annotations

# (layers, kv_heads, head_dim) -- GQA models commonly disaggregated
KV_PRESETS = {
    "llama70b": (80, 8, 128),
    "llama8b": (32, 8, 128),
    "qwen7b": (28, 4, 128),
    "mixtral8x7b": (32, 8, 128),
}

# serve-model short names (workloads/llm_moe_serving.SERVE_MODELS) -> geometry
SERVE_MODEL_KV = {
    "qwen25_7b": KV_PRESETS["qwen7b"],
    "mixtral_8x7b": KV_PRESETS["mixtral8x7b"],
}


def kv_bytes_per_token(layers: int, kv_heads: int, head_dim: int, dtype_bytes: int = 2) -> int:
    """K and V, all layers. (The factor 2 for K+V is easy to forget.)"""
    return 2 * layers * kv_heads * head_dim * dtype_bytes


def kv_handoff_bytes(layers: int, kv_heads: int, head_dim: int, seq: int, batch: int,
                     dtype_bytes: int = 2) -> int:
    return kv_bytes_per_token(layers, kv_heads, head_dim, dtype_bytes) * seq * batch


def split_blocks(total: int, block: int) -> list[int]:
    """Sizes of consecutive blocks covering ``total`` items (last one may be short)."""
    if block <= 0 or total <= 0:
        return [total] if total > 0 else []
    full, rem = divmod(total, block)
    return [block] * full + ([rem] if rem else [])
