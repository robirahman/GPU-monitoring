#!/usr/bin/env python3
"""Generic single-GPU LLM / vision inference harness (Round 5b).

Covers Round 5b workloads #1-#11 — anything that fits on RTX 5080 / 5090
or a single A100. For >80 GB datacenter models and tensor-parallel variants,
see `workloads/llm_inference_vllm.py`.

Run via:
    python workloads/llm_inference.py --model phi3_mini --duration 300
    python workloads/llm_inference.py --model llama32_3b --batch-size 4
    python workloads/llm_inference.py --model sdxl --duration 300
    python workloads/llm_inference.py --model clip_l14 --batch-size 64

All LLM modes share one generation loop: tokenize a rotating prompt pool,
call `model.generate` with the requested batch/seq/max-tokens, track
tok/s. Kept forward-only (no optimizer states) so the telemetry footprint
is exactly the "pure inference" signal the classifier needs to anchor on
at large memory sizes.
"""

import argparse
import os
import time

import numpy as np
import torch


# Map registry short-name -> HF model id. Keep any model that fits a 5080
# (16 GB) in FP16 here. Bigger than that goes in llm_inference_vllm.py.
MODEL_REGISTRY = {
    "phi3_mini":        ("microsoft/Phi-3-mini-4k-instruct",              "fp16"),
    "qwen25_1p5b":      ("Qwen/Qwen2.5-1.5B-Instruct",                    "fp16"),
    "qwen25_3b":        ("Qwen/Qwen2.5-3B-Instruct",                      "fp16"),
    "smollm2_1p7b":     ("HuggingFaceTB/SmolLM2-1.7B-Instruct",           "fp16"),
    "mistral_7b_q4":    ("TheBloke/Mistral-7B-Instruct-v0.2-GPTQ",        "gptq"),
    "tinyllama":        ("TinyLlama/TinyLlama-1.1B-Chat-v1.0",            "fp16"),
    "qwen25_0p5b":      ("Qwen/Qwen2.5-0.5B-Instruct",                    "fp16"),
}


# Prompts cover a range of topics/lengths so tokenized prompts don't
# pathologically quantize to a single length, which would bias
# memory/latency shape.
PROMPTS = [
    "Explain the theory of general relativity in detail, covering spacetime curvature,",
    "Write a comprehensive analysis of the economic impacts of climate change on",
    "Describe the process of photosynthesis at the molecular level, including",
    "What are the key differences between classical and quantum computing, and how",
    "Analyze the causes and consequences of World War I, focusing on the alliance",
    "Write a detailed tutorial on building a web application with Python Flask,",
    "Explain how neural networks learn through backpropagation, starting from",
    "Describe the history and evolution of the Internet from ARPANET to",
]


def _pad_prompt_to_len(tokenizer, prompt: str, target_len: int) -> str:
    """Repeat / truncate the prompt so its tokenized length is close to target."""
    ids = tokenizer(prompt, return_tensors="pt")["input_ids"][0]
    if len(ids) >= target_len:
        return tokenizer.decode(ids[:target_len], skip_special_tokens=True)
    # Repeat base prompt until we have enough tokens
    repeat = (target_len // len(ids)) + 1
    return (" ".join([prompt] * repeat))


def run_llm_synthetic_forward(args, device: str) -> None:
    """Synthetic-batch forward-only inference loop.

    Mirrors workloads/adversarial_whitebox.py's strategy_wb_inference exactly:
    generates random token batches at fixed (batch_size, seq_len) and runs
    pure forward passes with torch.no_grad(). No autoregressive generation,
    no KV-cache, no sampling. The point is to give the classifier a labeled
    INFERENCE example that has the same telemetry shape as whitebox_inference
    (WB-I), which the autoregressive .generate() path does NOT produce.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, dtype_tag = MODEL_REGISTRY[args.model]
    print(f"Loading {args.model} ({model_id}, {dtype_tag}) — synthetic forward mode")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": device, "trust_remote_code": False}
    if dtype_tag == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif dtype_tag == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()

    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    vocab = model.config.vocab_size
    print(f"  params: {params_b:.2f}B | vocab: {vocab} | "
          f"batch_size={args.batch_size} | seq_len={args.seq_len}")

    start = time.time()
    iters = 0
    while time.time() - start < args.duration:
        input_ids = torch.randint(
            0, vocab,
            (args.batch_size, args.seq_len),
            device=device,
            dtype=torch.long,
        )
        attention_mask = torch.ones_like(input_ids)
        with torch.no_grad():
            _ = model(input_ids=input_ids, attention_mask=attention_mask)
        iters += 1
        if iters % 10 == 0:
            elapsed = time.time() - start
            tok = iters * args.batch_size * args.seq_len
            print(f"  iter {iters}: {tok} tokens forward, "
                  f"{tok/elapsed:.0f} tok/s, {elapsed:.0f}s")
    elapsed = time.time() - start
    tok = iters * args.batch_size * args.seq_len
    print(f"{args.model} synthetic forward done: {iters} iters, "
          f"{tok} tokens, {tok/elapsed:.1f} tok/s, {elapsed:.1f}s")


def run_llm(args, device: str) -> None:
    from transformers import AutoModelForCausalLM, AutoTokenizer

    model_id, dtype_tag = MODEL_REGISTRY[args.model]
    print(f"Loading {args.model} ({model_id}, {dtype_tag})")

    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    load_kwargs = {"device_map": device, "trust_remote_code": False}
    if dtype_tag == "fp16":
        load_kwargs["torch_dtype"] = torch.float16
    elif dtype_tag == "bf16":
        load_kwargs["torch_dtype"] = torch.bfloat16
    # gptq / awq paths: transformers + auto-gptq / autoawq handle dtype internally.

    model = AutoModelForCausalLM.from_pretrained(model_id, **load_kwargs)
    model.eval()
    params_b = sum(p.numel() for p in model.parameters()) / 1e9
    print(f"  params: {params_b:.2f}B | batch_size={args.batch_size} | "
          f"seq_len={args.seq_len} | max_tokens={args.max_tokens}")

    # Pre-tokenize a batch pool matching requested seq length
    padded = [_pad_prompt_to_len(tokenizer, p, args.seq_len) for p in PROMPTS]

    start = time.time()
    total_tokens = 0
    iterations = 0
    while time.time() - start < args.duration:
        # Build a batch of size args.batch_size by rotating through the pool
        batch_prompts = [padded[(iterations * args.batch_size + b) % len(padded)]
                         for b in range(args.batch_size)]
        inputs = tokenizer(
            batch_prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=args.seq_len,
        ).to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        generated = (outputs.shape[1] - inputs["input_ids"].shape[1]) * args.batch_size
        total_tokens += int(generated)
        iterations += 1

        if iterations % 5 == 0:
            elapsed = time.time() - start
            print(f"  iter {iterations}: {total_tokens} tok, "
                  f"{total_tokens/elapsed:.1f} tok/s, {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"{args.model} done: {iterations} iters, {total_tokens} tok, "
          f"{total_tokens/elapsed:.1f} tok/s, {elapsed:.1f}s wall")


def run_sdxl(args, device: str) -> None:
    """Stable Diffusion XL inference — diffusion path, different shape."""
    from diffusers import StableDiffusionXLPipeline

    print("Loading SDXL base 1.0")
    pipe = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16,
        use_safetensors=True,
        variant="fp16",
    ).to(device)

    prompts = [
        "A photorealistic oil painting of a mountain valley at sunrise, dramatic lighting",
        "A cyberpunk city street at night, neon signs, rain reflections, cinematic",
        "A detailed fantasy dragon soaring over a medieval castle, epic scale",
        "A cozy cabin in a snowy forest at dusk, warm light from the windows",
    ]

    start = time.time()
    iterations = 0
    while time.time() - start < args.duration:
        prompt = prompts[iterations % len(prompts)]
        with torch.no_grad():
            _ = pipe(
                prompt=prompt,
                num_inference_steps=args.sdxl_steps,
                height=args.sdxl_size,
                width=args.sdxl_size,
            ).images[0]
        iterations += 1
        elapsed = time.time() - start
        print(f"  image {iterations}: {elapsed:.0f}s elapsed, {iterations/elapsed:.2f} img/s")

    elapsed = time.time() - start
    print(f"SDXL done: {iterations} images in {elapsed:.1f}s")


def run_vit_l(args, device: str) -> None:
    """ViT-L/16 ImageNet classification at batch — encoder-only vision path."""
    from transformers import ViTForImageClassification

    model_id = "google/vit-large-patch16-224"
    print(f"Loading {model_id}")
    model = ViTForImageClassification.from_pretrained(
        model_id, torch_dtype=torch.float16,
    ).to(device).eval()

    def fake_batch(n):
        return torch.randn(n, 3, 224, 224, dtype=torch.float16, device=device)

    start = time.time()
    iterations = 0
    while time.time() - start < args.duration:
        imgs = fake_batch(args.batch_size)
        with torch.no_grad():
            _ = model(pixel_values=imgs).logits
        iterations += 1
        if iterations % 25 == 0:
            elapsed = time.time() - start
            print(f"  batch {iterations}: {elapsed:.0f}s, "
                  f"{iterations*args.batch_size/elapsed:.1f} img/s")

    elapsed = time.time() - start
    print(f"ViT-L done: {iterations} batches, {iterations*args.batch_size} images, "
          f"{iterations*args.batch_size/elapsed:.1f} img/s in {elapsed:.1f}s")


def run_bge_embed(args, device: str) -> None:
    """BGE-large-en-v1.5 batch text embedding — bidirectional encoder path."""
    from transformers import AutoModel, AutoTokenizer

    model_id = "BAAI/bge-large-en-v1.5"
    print(f"Loading {model_id}")
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()

    # Build a stable batch of tokenised text (vary length across the batch
    # so attention masks exercise both padded and full positions).
    texts = [PROMPTS[i % len(PROMPTS)] * ((i % 3) + 1) for i in range(args.batch_size)]
    encoded = tokenizer(
        texts, padding=True, truncation=True, max_length=512, return_tensors="pt",
    ).to(device)

    start = time.time()
    iterations = 0
    while time.time() - start < args.duration:
        with torch.no_grad():
            out = model(**encoded)
            mask = encoded["attention_mask"].unsqueeze(-1).float()
            pooled = (out.last_hidden_state * mask).sum(1) / mask.sum(1)
            _ = torch.nn.functional.normalize(pooled, p=2, dim=-1)
        iterations += 1
        if iterations % 50 == 0:
            elapsed = time.time() - start
            print(f"  batch {iterations}: {iterations*args.batch_size/elapsed:.1f} texts/s, {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"BGE done: {iterations} batches, {iterations*args.batch_size} texts, "
          f"{iterations*args.batch_size/elapsed:.1f} texts/s in {elapsed:.1f}s")


def run_clip_embed(args, device: str) -> None:
    """CLIP ViT-L/14 image embedding at batch."""
    from transformers import CLIPModel, CLIPProcessor

    print("Loading CLIP ViT-L/14")
    model_id = "openai/clip-vit-large-patch14"
    processor = CLIPProcessor.from_pretrained(model_id)
    model = CLIPModel.from_pretrained(model_id, torch_dtype=torch.float16).to(device).eval()

    # Synthetic RGB tensors — avoids disk I/O variability so telemetry
    # reflects GPU work, not dataloader jitter.
    def fake_batch(n):
        return torch.randn(n, 3, 224, 224, dtype=torch.float16, device=device)

    start = time.time()
    iterations = 0
    while time.time() - start < args.duration:
        imgs = fake_batch(args.batch_size)
        with torch.no_grad():
            _ = model.get_image_features(pixel_values=imgs)
        iterations += 1

        if iterations % 25 == 0:
            elapsed = time.time() - start
            print(f"  batch {iterations}: {elapsed:.0f}s, "
                  f"{iterations*args.batch_size/elapsed:.1f} img/s")

    elapsed = time.time() - start
    print(f"CLIP done: {iterations} batches, {iterations*args.batch_size} images, "
          f"{iterations*args.batch_size/elapsed:.1f} img/s in {elapsed:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model", required=True,
        choices=list(MODEL_REGISTRY.keys()) + ["sdxl", "clip_l14", "vit_l", "bge_large"],
    )
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256,
                    help="Prompt token length (padded/truncated to this).")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--sdxl-steps", type=int, default=30)
    ap.add_argument("--sdxl-size", type=int, default=1024)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--synthetic-forward", action="store_true",
                    help="Use synthetic-batch forward-only loop (matches "
                         "whitebox_inference exactly). Implies no autoregressive "
                         "generation, no KV-cache, no sampling.")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    # Ensure HF_HOME is writable on the current host; override in shell if needed.
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    if args.model == "sdxl":
        run_sdxl(args, args.device)
    elif args.model == "clip_l14":
        run_clip_embed(args, args.device)
    elif args.model == "vit_l":
        run_vit_l(args, args.device)
    elif args.model == "bge_large":
        run_bge_embed(args, args.device)
    elif args.synthetic_forward:
        run_llm_synthetic_forward(args, args.device)
    else:
        run_llm(args, args.device)


if __name__ == "__main__":
    main()
