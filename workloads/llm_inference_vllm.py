#!/usr/bin/env python3
"""Datacenter-scale LLM inference via vLLM (Round 5b + SPAR regime grid).

Covers models that exceed single-consumer-GPU memory or require
tensor-parallel inference. SPAR M0/M1 additions:

  --input-len / --max-tokens  →  prefill-heavy vs decode-heavy shapes
  --mode midfill              →  long prefix warm-up + short continuations
                                 (enable_prefix_caching)

Usage:
    python workloads/llm_inference_vllm.py --model qwen25_7b_fp16 --duration 300
    python workloads/llm_inference_vllm.py --model qwen25_7b_fp16 \\
        --input-len 8192 --max-tokens 16 --duration 300
    python workloads/llm_inference_vllm.py --model qwen25_7b_fp16 \\
        --mode midfill --prefix-len 8192 --append-len 256 --max-tokens 128
    python workloads/llm_inference_vllm.py --model mixtral_8x7b_tp2 \\
        --tensor-parallel-size 2 --input-len 4096 --max-tokens 256
"""

from __future__ import annotations

import argparse
import os
import time


# Registry: short-name -> (HF id, dtype, required_min_gpus, default TP).
# All entries are non-gated (Qwen, Mistral, DeepSeek) to avoid HF access gates.
VLLM_MODELS = {
    # Single-GPU dense inference at FSDP-matched sizes (phase 5).
    # 7B/14B/32B fit in 16/32/64 GB FP16 — any datacenter GPU.
    "qwen25_7b_fp16": (
        "Qwen/Qwen2.5-7B-Instruct",
        "float16",
        1, 1,
    ),
    "qwen25_14b_fp16": (
        "Qwen/Qwen2.5-14B-Instruct",
        "float16",
        1, 1,
    ),
    "qwen25_32b_fp16": (
        "Qwen/Qwen2.5-32B-Instruct",
        "float16",
        1, 1,
    ),
    "qwen25_72b_fp16": (
        "Qwen/Qwen2.5-72B-Instruct",
        "float16",
        1, 1,
    ),
    "mixtral_8x7b": (
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "float16",
        1, 1,
    ),
    # H100-80 / A100-80: Mixtral FP16 needs TP≥2 (~90 GB weights).
    "mixtral_8x7b_tp2": (
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "float16",
        2, 2,
    ),
    "mixtral_8x7b_tp4": (
        "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "float16",
        4, 4,
    ),
    "mistral_7b_q4": (
        "TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",  # GPTQ 4-bit, vLLM Marlin kernels
        "auto",
        1, 1,
    ),
    "qwen25_72b_fp8": (
        "Qwen/Qwen2.5-72B-Instruct-AWQ",   # AWQ 4-bit, fits ~45 GB; adjust to FP8 if available
        "auto",
        1, 1,
    ),
    "qwen25_72b_tp2": (
        "Qwen/Qwen2.5-72B-Instruct",
        "float16",
        2, 2,
    ),
    "deepseek_v3_tp8": (
        "deepseek-ai/DeepSeek-V3",        # 671B MoE (~37B active), FP8 native
        "auto",
        8, 8,
    ),
}


# Longer / more varied prompts than the single-GPU harness — the big
# models benefit from real-length context to actually stress attention.
PROMPTS = [
    "Summarise the plot, themes, and literary techniques of Leo Tolstoy's War and Peace, "
    "paying special attention to the characterisation of Pierre Bezukhov and the Napoleonic war sequences.",
    "Explain in detail how a modern transformer-based language model processes a 4096-token context window, "
    "covering tokenization, positional encoding, multi-head attention, feedforward layers, and KV caching.",
    "Analyse the causes, progression, and consequences of the 2008 global financial crisis, "
    "including the role of mortgage-backed securities, credit default swaps, and regulatory failures.",
    "Describe the cell biology of eukaryotic protein synthesis from DNA transcription through "
    "ribosomal translation, post-translational modifications, and trafficking.",
    "Give a thorough technical overview of modern GPU architecture, covering SMs, tensor cores, "
    "HBM memory hierarchy, warp scheduling, and the CUDA execution model.",
    "Write a balanced analysis of the philosophical arguments for and against consequentialism as "
    "an ethical framework, contrasting it with deontology and virtue ethics.",
]

# Deterministic filler (~1 token ≈ 1 word for English BPE undercounts slightly;
# we overshoot then truncate with the tokenizer when available).
_FILLER = (
    "alpha bravo charlie delta echo foxtrot golf hotel india juliet kilo lima "
    "mike november oscar papa quebec romeo sierra tango uniform victor whiskey "
    "xray yankee zulu "
)


def _pad_prompt(base: str, target_tokens: int, tokenizer=None) -> str:
    """Grow `base` until it has ~target_tokens (exact if tokenizer given)."""
    if target_tokens <= 0:
        return base
    if tokenizer is not None:
        ids = tokenizer.encode(base)
        if len(ids) >= target_tokens:
            return tokenizer.decode(ids[:target_tokens])
        # Append filler chunks until long enough, then truncate.
        parts = [base]
        while True:
            parts.append(_FILLER)
            ids = tokenizer.encode(" ".join(parts))
            if len(ids) >= target_tokens:
                return tokenizer.decode(ids[:target_tokens])
    # Fallback: ~0.75 words/token heuristic → use 1.5 words per target token.
    words_needed = max(0, int(target_tokens * 1.5) - len(base.split()))
    extra = []
    filler_words = _FILLER.split()
    while len(extra) < words_needed:
        extra.extend(filler_words)
    return base + " " + " ".join(extra[:words_needed])


def _resolve_max_model_len(args) -> int:
    """Ensure context window fits the longest prompt + generation we will run."""
    if args.mode == "midfill":
        need = args.prefix_len + args.append_len + args.max_tokens + 64
    else:
        need = args.input_len + args.max_tokens + 64
    return max(args.max_model_len, need)


def run_vllm_standard(llm, args, tokenizer) -> None:
    """M0: fixed input_len × max_tokens generate loop (prefill/decode shapes)."""
    from vllm import SamplingParams

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    # Pre-build padded prompts once (same length every iter → stable regime).
    padded = [
        _pad_prompt(PROMPTS[i % len(PROMPTS)], args.input_len, tokenizer)
        for i in range(max(args.batch_size, len(PROMPTS)))
    ]
    print(f"  regime=standard input_len={args.input_len} max_tokens={args.max_tokens} "
          f"batch_size={args.batch_size}")

    start = time.time()
    total_tokens = 0
    iterations = 0
    while time.time() - start < args.duration:
        batch_prompts = [
            padded[(iterations * args.batch_size + b) % len(padded)]
            for b in range(args.batch_size)
        ]
        outputs = llm.generate(batch_prompts, sampling_params)
        for out in outputs:
            for comp in out.outputs:
                total_tokens += len(comp.token_ids)
        iterations += 1
        elapsed = time.time() - start
        print(f"  iter {iterations}: {total_tokens} tok, "
              f"{total_tokens / elapsed:.1f} tok/s, {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"standard done: {iterations} iters, {total_tokens} tok, "
          f"{total_tokens / elapsed:.1f} tok/s, {elapsed:.1f}s wall")


def run_vllm_midfill(llm, args, tokenizer) -> None:
    """M1: warm long prefix (KV), then short appends — midfill + decode.

    Relies on vLLM enable_prefix_caching so turn N+1 reuses the prefix KV
    instead of re-prefilling from scratch.
    """
    from vllm import SamplingParams

    prefixes = [
        _pad_prompt(PROMPTS[i % len(PROMPTS)], args.prefix_len, tokenizer)
        for i in range(args.batch_size)
    ]
    warm_params = SamplingParams(temperature=0.0, max_tokens=1)
    cont_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    print(f"  regime=midfill prefix_len={args.prefix_len} append_len={args.append_len} "
          f"max_tokens={args.max_tokens} batch_size={args.batch_size}")
    print("  warming prefix KV cache …")
    t_warm = time.time()
    llm.generate(prefixes, warm_params)
    print(f"  warm done in {time.time() - t_warm:.1f}s")

    start = time.time()
    total_tokens = 0
    iterations = 0
    append_idx = 0
    while time.time() - start < args.duration:
        # Unique short suffix each turn so we actually extend (not pure cache hit).
        appends = []
        for b in range(args.batch_size):
            seed = PROMPTS[(append_idx + b) % len(PROMPTS)]
            appends.append(
                _pad_prompt(
                    f"\n\n[turn {iterations} req {b}] {seed}\nAnswer briefly:\n",
                    args.append_len,
                    tokenizer,
                )
            )
        batch = [prefixes[b] + appends[b] for b in range(args.batch_size)]
        outputs = llm.generate(batch, cont_params)
        for out in outputs:
            for comp in out.outputs:
                total_tokens += len(comp.token_ids)
        iterations += 1
        append_idx += args.batch_size
        elapsed = time.time() - start
        print(f"  midfill turn {iterations}: {total_tokens} tok, "
              f"{total_tokens / elapsed:.1f} tok/s, {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"midfill done: {iterations} turns, {total_tokens} tok, "
          f"{total_tokens / elapsed:.1f} tok/s, {elapsed:.1f}s wall")


def run_vllm(args) -> None:
    from vllm import LLM

    model_id, dtype, min_gpus, default_tp = VLLM_MODELS[args.model]
    tp = args.tensor_parallel_size or default_tp
    max_model_len = _resolve_max_model_len(args)

    print(f"Loading {args.model} via vLLM")
    print(f"  model_id={model_id} dtype={dtype} tensor_parallel_size={tp}")
    print(f"  max_model_len={max_model_len} mode={args.mode}")

    llm_kwargs = dict(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=tp,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=max_model_len,
        enforce_eager=args.enforce_eager,
    )
    if args.mode == "midfill":
        llm_kwargs["enable_prefix_caching"] = True

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()

    if args.mode == "midfill":
        run_vllm_midfill(llm, args, tokenizer)
    else:
        run_vllm_standard(llm, args, tokenizer)


def main():
    ap = argparse.ArgumentParser(
        description="vLLM inference with SPAR prefill/decode/midfill regimes",
    )
    ap.add_argument("--model", required=True, choices=list(VLLM_MODELS.keys()))
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Concurrent requests submitted per iteration.")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Generation length (decode pressure).")
    ap.add_argument("--input-len", type=int, default=512,
                    help="M0: pad each prompt to this many tokens (prefill pressure).")
    ap.add_argument("--max-model-len", type=int, default=4096,
                    help="Floor for model context; auto-raised to fit input+output.")
    ap.add_argument("--mode", choices=["standard", "midfill"], default="standard",
                    help="standard=fixed prompt length; midfill=prefix cache + append.")
    ap.add_argument("--prefix-len", type=int, default=8192,
                    help="M1 midfill: cached prefix length in tokens.")
    ap.add_argument("--append-len", type=int, default=256,
                    help="M1 midfill: new tokens appended each turn.")
    ap.add_argument("--tensor-parallel-size", type=int, default=0,
                    help="0 = use model's default TP size")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.9)
    ap.add_argument("--enforce-eager", action="store_true",
                    help="Skip CUDA graph capture (useful on short runs).")
    args = ap.parse_args()

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    # vLLM sometimes benefits from this for MoE on Ampere+.
    os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", "spawn")
    run_vllm(args)


if __name__ == "__main__":
    main()
