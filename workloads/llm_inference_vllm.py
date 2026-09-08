#!/usr/bin/env python3
"""Datacenter-scale LLM inference via vLLM (Round 5b, workloads #12-#16).

Covers models that exceed single-consumer-GPU memory or require
tensor-parallel inference:

  - llama33_70b_fp16         : Llama 3.3 70B (~140 GB FP16), single H200 or B200
  - mixtral_8x7b             : Mixtral 8x7B MoE (~90 GB FP16), single H200/B200
  - qwen25_72b_fp8           : Qwen 2.5 72B FP8 (~72 GB), single H200/B200
  - llama31_70b_tp2          : Llama 3.1 70B FP16, TP across 2 GPUs (2x H100-80/H200)
  - llama31_405b_tp8         : Llama 3.1 405B FP8, TP across 8 GPUs (8x H200)

Uses vLLM for paged attention, continuous batching, and tensor parallelism.
Keeps the telemetry shape pure-forward — the scheduler may batch multiple
concurrent requests, but there are no optimizer states or backward passes.

Usage:
    python workloads/llm_inference_vllm.py --model llama33_70b_fp16 --duration 300
    python workloads/llm_inference_vllm.py --model llama31_70b_tp2 --tensor-parallel-size 2
    python workloads/llm_inference_vllm.py --model llama31_405b_tp8 --tensor-parallel-size 8
"""

import argparse
import os
import time


# Registry: short-name -> (HF id, dtype, required_min_gpus, default TP).
# All entries are non-gated (Qwen, Mistral, DeepSeek) to avoid HF access gates.
VLLM_MODELS = {
    # Single-GPU dense inference at FSDP-matched sizes (phase 5).
    # 7B/14B/32B fit in 16/32/64 GB FP16 — any datacenter GPU.
    # Matched-pair partner for fsdp_1b training: the scope ladder must vary
    # SCOPE only, so the inference negative has to sit at the same model scale as
    # the training positive. With training at 1.3B, a 7B negative would let a
    # classifier separate the arms by size rather than by train-vs-inference.
    "qwen25_1_5b_fp16": (
        "Qwen/Qwen2.5-1.5B-Instruct",
        "float16",
        1, 1,
    ),
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


_PAD_CACHE = {}


def _pad_prompt(llm, base, n_tokens):
    """Grow `base` to exactly n_tokens using the model's own tokenizer.

    Encoding, slicing and decoding back (rather than passing token ids) keeps
    this working across vLLM versions, whose TokensPrompt API has moved around.
    Cached because the padding is identical every iteration and tokenising 32k
    tokens per request per step would show up as CPU time in the telemetry.
    """
    key = (base[:32], n_tokens)
    if key in _PAD_CACHE:
        return _PAD_CACHE[key]
    tok = llm.get_tokenizer()
    filler = (" " + base) * (n_tokens // max(len(tok.encode(base)), 1) + 2)
    ids = tok.encode(base + filler)[:n_tokens]
    _PAD_CACHE[key] = tok.decode(ids)
    return _PAD_CACHE[key]


def run_vllm(args) -> None:
    from vllm import LLM, SamplingParams

    model_id, dtype, min_gpus, default_tp = VLLM_MODELS[args.model]
    tp = args.tensor_parallel_size or default_tp
    pp = args.pipeline_parallel_size

    print(f"Loading {args.model} via vLLM")
    print(f"  model_id={model_id} dtype={dtype} "
          f"tensor_parallel_size={tp} pipeline_parallel_size={pp} "
          f"backend={args.distributed_executor_backend}")

    # tensor_parallel_size shards each layer within a node (NVLink); with
    # pipeline_parallel_size>1, vLLM splits the layer stack across pipeline
    # stages, which is how a model spans two servers. TP8 x PP2 => a 16-GPU,
    # 2-node serving job whose only cross-server traffic is pipeline activation
    # hand-off -- the "sharded inference" negative for the S16 comparison. Ray
    # is required for the multi-node case (mp is single-node only).
    llm = LLM(
        model=model_id,
        dtype=dtype,
        tensor_parallel_size=tp,
        pipeline_parallel_size=pp,
        distributed_executor_backend=args.distributed_executor_backend,
        gpu_memory_utilization=args.gpu_memory_utilization,
        trust_remote_code=True,
        max_model_len=args.max_model_len,
        # enforce eager for smaller batches — compiled graphs are marginal here
        enforce_eager=args.enforce_eager,
    )

    sampling_params = SamplingParams(
        temperature=0.7,
        top_p=0.9,
        max_tokens=args.max_tokens,
    )

    start = time.time()
    total_tokens = 0
    iterations = 0
    while time.time() - start < args.duration:
        # Build a batch of `--batch-size` concurrent requests so vLLM's
        # continuous scheduler actually has something to schedule.
        batch_prompts = [PROMPTS[(iterations * args.batch_size + b) % len(PROMPTS)]
                         for b in range(args.batch_size)]
        if args.prompt_tokens > 0:
            batch_prompts = [_pad_prompt(llm, p, args.prompt_tokens)
                             for p in batch_prompts]
        outputs = llm.generate(batch_prompts, sampling_params)
        for out in outputs:
            for comp in out.outputs:
                total_tokens += len(comp.token_ids)
        iterations += 1

        elapsed = time.time() - start
        print(f"  iter {iterations}: {total_tokens} tok, "
              f"{total_tokens/elapsed:.1f} tok/s, {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"{args.model} done: {iterations} iters, {total_tokens} tok, "
          f"{total_tokens/elapsed:.1f} tok/s, {elapsed:.1f}s wall, TP={tp}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, choices=list(VLLM_MODELS.keys()))
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=8,
                    help="Concurrent requests submitted per iteration.")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--max-model-len", type=int, default=4096)
    ap.add_argument("--prompt-tokens", type=int, default=0,
                    help="Pad every prompt to this many tokens. 0 keeps the "
                         "short PROMPTS as-is.\n"
                         "This exists to make a job MULTI-NODE OBLIGATORY. "
                         "Qwen2.5-72B's weights are only 145 GB and fit on one "
                         "8x80GB node, so raising --max-model-len alone proves "
                         "nothing: it caps sequence length without lengthening "
                         "the actual sequences, and the ~40-token PROMPTS leave "
                         "the KV cache nearly empty. Obligation has to come "
                         "from KV: at 0.328 MB/token, serving --batch-size B "
                         "sequences of N tokens needs B*N*0.328 MB, and "
                         "64 x 32768 = 2.1M tokens = 688 GB, which with the "
                         "weights exceeds one node and fits across two.")
    ap.add_argument("--tensor-parallel-size", type=int, default=0,
                    help="0 = use model's default TP size")
    ap.add_argument("--pipeline-parallel-size", type=int, default=1,
                    help="Pipeline stages. >1 splits the layer stack across "
                         "stages so the model spans multiple nodes (needs "
                         "--distributed-executor-backend ray).")
    ap.add_argument("--distributed-executor-backend", type=str, default=None,
                    choices=["mp", "ray"],
                    help="mp = single-node multiprocessing (default when "
                         "unset); ray = multi-node (start the Ray cluster "
                         "first with `ray start`).")
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
