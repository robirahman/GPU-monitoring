#!/usr/bin/env python3
"""Pre-download all models + datasets used by the B200 ML queue.

Runs in background so the GPU queue isn't bottlenecked on weight downloads.
Uses huggingface_hub.snapshot_download (CPU/network bound, does not touch GPU).
Skips already-downloaded models (idempotent).
"""

import os
import sys
import time
from huggingface_hub import snapshot_download
from huggingface_hub.utils import HfHubHTTPError, LocalEntryNotFoundError

HF_TOKEN = os.environ.get("HF_TOKEN")

# Model IDs used by the 15 single-GPU inference workloads + extended training.
# (model_id, note, gated)
MODELS = [
    # Consumer subset (llm_inference.py, transformers)
    ("microsoft/Phi-3-mini-4k-instruct", "phi3_mini", False),
    ("Qwen/Qwen2.5-3B-Instruct", "qwen25_3b / bs4 / bs16", False),
    ("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ", "mistral_7b_q4 (GPTQ 4-bit)", False),
    ("TinyLlama/TinyLlama-1.1B-Chat-v1.0", "tinyllama_bs8 / seq4096", False),
    ("stabilityai/stable-diffusion-xl-base-1.0", "sdxl", False),
    ("google/vit-large-patch16-224", "vit_l_bs64", False),
    # Datacenter-large (llm_inference_vllm.py)
    ("Qwen/Qwen2.5-7B-Instruct", "qwen25_7b_fp16 (~15 GB)", False),
    ("Qwen/Qwen2.5-14B-Instruct", "qwen25_14b_fp16 (~28 GB)", False),
    ("Qwen/Qwen2.5-32B-Instruct", "qwen25_32b_fp16 (~64 GB)", False),
    ("Qwen/Qwen2.5-72B-Instruct", "qwen25_72b_fp16 (~144 GB)", False),
    ("Qwen/Qwen2.5-72B-Instruct-AWQ", "qwen25_72b_fp8 (AWQ-4bit ~45 GB)", False),
    ("mistralai/Mixtral-8x7B-Instruct-v0.1", "mixtral_8x7b (~90 GB, GATED)", True),
    # Training models (extended-epoch queue)
    ("gpt2", "gpt2_wikitext2_ext / _amp_ext", False),
    ("bert-base-uncased", "bert_sst2_ext / _amp_ext", False),
]


def fetch(model_id: str, note: str, gated: bool) -> str:
    t0 = time.time()
    auth_kw = {"token": HF_TOKEN} if HF_TOKEN else {}
    try:
        path = snapshot_download(
            repo_id=model_id,
            **auth_kw,
        )
        dt = time.time() - t0
        return f"OK    {model_id:55s} ({note}) in {dt:.0f}s → {path}"
    except HfHubHTTPError as e:
        status = getattr(e.response, "status_code", "?")
        if status in (401, 403):
            return f"AUTH  {model_id:55s} ({note}) — {status}: need valid HF_TOKEN (gated={gated})"
        return f"FAIL  {model_id:55s} ({note}) — HTTP {status}: {e}"
    except Exception as e:
        return f"FAIL  {model_id:55s} ({note}) — {type(e).__name__}: {e}"


def main():
    print(f"HF_HOME={os.environ.get('HF_HOME', '<default>')}")
    print(f"HF_TOKEN={'set' if HF_TOKEN else 'NOT SET (gated models will fail)'}")
    print(f"Downloading {len(MODELS)} model repos. Non-gated first, then gated.\n")
    sys.stdout.flush()

    ordered = [m for m in MODELS if not m[2]] + [m for m in MODELS if m[2]]
    for m in ordered:
        line = fetch(*m)
        print(line, flush=True)

    print("\nDone.")


if __name__ == "__main__":
    main()
