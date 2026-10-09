#!/usr/bin/env python3
"""Modern model inference workloads for telemetry diversity.

Supports:
  - deepseek_r1: DeepSeek-R1-Distill-Qwen-1.5B autoregressive text generation
  - qwen25: Qwen 2.5 0.5B autoregressive text generation
  - whisper: Whisper large-v3-turbo audio transcription (synthetic audio)

Each produces a fundamentally different GPU telemetry signature:
  - LLM decode: memory-bandwidth-bound, growing KV cache, bursty util
  - Whisper: encoder-decoder, static memory, cross-attention, compute-bound encoder

Usage:
    python modern_inference.py --model deepseek_r1 --duration 300
    python modern_inference.py --model qwen25 --duration 300
    python modern_inference.py --model whisper --duration 300
"""

import argparse
import os
import time

import torch
import numpy as np


def run_deepseek_r1(args, device):
    """DeepSeek-R1-Distill-Qwen-1.5B autoregressive generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading DeepSeek-R1-Distill-Qwen-1.5B...")
    model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    prompts = [
        "Explain the theory of general relativity in detail, covering spacetime curvature,",
        "Write a comprehensive analysis of the economic impacts of climate change on",
        "Describe the process of photosynthesis at the molecular level, including",
        "What are the key differences between classical and quantum computing, and how",
        "Analyze the causes and consequences of World War I, focusing on the alliance",
    ]

    start = time.time()
    total_tokens = 0
    iterations = 0
    while time.time() - start < args.duration:
        prompt = prompts[iterations % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += generated
        iterations += 1

        if iterations % 5 == 0:
            elapsed = time.time() - start
            tps = total_tokens / elapsed
            print(f"  Iter {iterations}: {total_tokens} tokens, {tps:.1f} tok/s, {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"DeepSeek-R1 inference complete: {iterations} iterations, "
          f"{total_tokens} tokens, {total_tokens/elapsed:.1f} tok/s, {elapsed:.1f}s")


def run_qwen25(args, device):
    """Qwen 2.5 0.5B autoregressive generation."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading Qwen 2.5 0.5B...")
    model_name = "Qwen/Qwen2.5-0.5B"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=torch.float16, device_map=device,
    )
    model.eval()
    print(f"Model loaded: {sum(p.numel() for p in model.parameters())/1e9:.1f}B params")

    prompts = [
        "Write a detailed tutorial on building a web application with Python Flask,",
        "Explain how neural networks learn through backpropagation, starting from",
        "Describe the history and evolution of the Internet from ARPANET to",
        "What are the fundamental principles of thermodynamics and how do they",
        "Analyze the literary techniques used in Shakespeare's Hamlet, focusing on",
    ]

    start = time.time()
    total_tokens = 0
    iterations = 0
    while time.time() - start < args.duration:
        prompt = prompts[iterations % len(prompts)]
        inputs = tokenizer(prompt, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=args.max_tokens,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )
        generated = outputs.shape[1] - inputs["input_ids"].shape[1]
        total_tokens += generated
        iterations += 1

        if iterations % 10 == 0:
            elapsed = time.time() - start
            tps = total_tokens / elapsed
            print(f"  Iter {iterations}: {total_tokens} tokens, {tps:.1f} tok/s, {elapsed:.0f}s")

    elapsed = time.time() - start
    print(f"Qwen 2.5 inference complete: {iterations} iterations, "
          f"{total_tokens} tokens, {total_tokens/elapsed:.1f} tok/s, {elapsed:.1f}s")


def run_whisper(args, device):
    """Whisper large-v3-turbo audio transcription on synthetic audio."""
    import whisper

    print("Loading Whisper large-v3-turbo...")
    model = whisper.load_model("large-v3-turbo", device=device,
                               download_root=os.path.expanduser("~/.cache/whisper"))
    print(f"Whisper loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    start = time.time()
    iterations = 0
    while time.time() - start < args.duration:
        # Generate synthetic audio: random mel spectrogram (30s segments)
        # This exercises the full encoder-decoder pipeline
        duration_sec = np.random.choice([5, 10, 15, 30])
        n_samples = duration_sec * 16000  # 16kHz
        audio = np.random.randn(n_samples).astype(np.float32) * 0.1

        # Add some structure: sine waves at speech frequencies
        t = np.linspace(0, duration_sec, n_samples)
        for freq in [200, 400, 800, 1200, 2000]:
            audio += 0.05 * np.sin(2 * np.pi * freq * t + np.random.random() * 2 * np.pi).astype(np.float32)

        result = model.transcribe(audio, language="en", fp16=(device != "cpu"))
        iterations += 1

        if iterations % 5 == 0:
            elapsed = time.time() - start
            print(f"  Iter {iterations}: {elapsed:.0f}s elapsed, "
                  f"{iterations/elapsed:.2f} transcriptions/s")

    elapsed = time.time() - start
    print(f"Whisper inference complete: {iterations} transcriptions, {elapsed:.1f}s, "
          f"{iterations/elapsed:.2f} transcriptions/s")


def run_whisper_hf(args, device):
    """Whisper ASR via transformers -- the dependency-free path.

    `run_whisper` above uses openai-whisper, which is pinned in no requirements
    file, pulls ffmpeg for audio decoding, and hardcodes
    download_root=/home/claude/.cache/whisper, a directory that does not exist on
    a collection node. transformers is already installed (requirements_gcp_extra)
    and takes the log-mel features directly, so nothing has to decode audio.

    This is the encoder-DECODER shape, which the corpus otherwise lacks entirely:
    a convolutional front end feeding a cross-attending autoregressive decoder.
    Every other inference cell is either decoder-only (the Qwen family) or
    encoder-only (ViT, BGE, CLIP).
    """
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_id = "openai/whisper-large-v3-turbo"
    print(f"Loading {model_id} (transformers)")
    processor = WhisperProcessor.from_pretrained(model_id)
    model = WhisperForConditionalGeneration.from_pretrained(
        model_id, torch_dtype=torch.float16).to(device).eval()
    print(f"Whisper loaded: {sum(p.numel() for p in model.parameters())/1e6:.0f}M params")

    sr = 16000
    start = time.time()
    iterations = 0
    while time.time() - start < args.duration:
        # Synthetic speech-band audio. Whisper pads/truncates to its own 30 s
        # window, so varying the source length varies the DECODE length, not the
        # encoder cost -- which is the realistic asymmetry.
        secs = int(np.random.choice([5, 10, 15, 30]))
        t = np.linspace(0, secs, secs * sr, dtype=np.float32)
        audio = (np.random.randn(secs * sr).astype(np.float32) * 0.1)
        for freq in (200, 400, 800, 1200, 2000):
            audio += 0.05 * np.sin(2 * np.pi * freq * t
                                   + np.random.random() * 2 * np.pi).astype(np.float32)

        feats = processor(audio, sampling_rate=sr, return_tensors="pt").input_features
        feats = feats.to(device, dtype=torch.float16)
        with torch.inference_mode():
            ids = model.generate(feats, max_new_tokens=args.max_tokens, language="en")
        iterations += 1
        if iterations % 5 == 0:
            elapsed = time.time() - start
            print(f"  Iter {iterations}: {elapsed:.0f}s elapsed, "
                  f"{iterations/elapsed:.2f} transcriptions/s, "
                  f"last {int(ids.shape[1])} tok")

    elapsed = time.time() - start
    print(f"Whisper (hf) inference complete: {iterations} transcriptions, "
          f"{elapsed:.1f}s, {iterations/elapsed:.2f} transcriptions/s")


def main():
    parser = argparse.ArgumentParser(description="Modern model inference workloads")
    parser.add_argument("--model", type=str, required=True,
                        choices=["deepseek_r1", "qwen25", "whisper", "whisper_hf"],
                        help="Model to run inference on")
    parser.add_argument("--duration", type=int, default=300,
                        help="Run duration in seconds (default: 300)")
    parser.add_argument("--max-tokens", type=int, default=256,
                        help="Max tokens per generation for LLMs (default: 256)")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available")
        args.device = "cpu"
    device = args.device
    print(f"Device: {device}")

    # Only if the caller has not already said where the cache lives. This used to
    # be an unconditional assignment to /home/claude/.cache/huggingface, a path
    # that does not exist on a collection node -- it would override the HF_HOME
    # multinode_queue.sh sets (line 34) and throw away everything
    # preload_vllm_models.sh had already downloaded, re-fetching the weights
    # inside the measurement window.
    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    models = {"deepseek_r1": run_deepseek_r1, "qwen25": run_qwen25,
              "whisper": run_whisper, "whisper_hf": run_whisper_hf}
    models[args.model](args, device)


if __name__ == "__main__":
    main()
