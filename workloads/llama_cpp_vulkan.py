"""llama.cpp Vulkan inference workload (Windows-AMD).

Wraps llama-cli or llama-bench from the llama.cpp Vulkan release. The
Vulkan backend is the most stable LLM inference path on RDNA 4 + Windows
without a HIP install — it binds explicitly to the dGPU adapter.

Two modes:
  --mode generate :: run llama-cli for N tokens of autoregressive generation
                     (steady-state inference signature — high SM, modest VRAM,
                     bursty PCIe RX, very low PCIe TX).
  --mode bench    :: run llama-bench's pp/tg sweep for a fixed wall-time
                     budget (more deterministic timing, useful for sweeps).

Model file is required and must be a GGUF. Suggested defaults that fit in
16 GB VRAM:
  - TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf       (~660 MB)
  - Phi-3-mini-4k-instruct.Q4_K_M.gguf         (~2.3 GB)
  - Qwen2.5-1.5B-Instruct.Q4_K_M.gguf          (~1.0 GB)
  - SmolLM2-1.7B-Instruct.Q4_K_M.gguf          (~1.1 GB)

Download once and pass via --model. We don't auto-download to keep the
runner offline-friendly and avoid hammering HuggingFace mirrors.

Usage:
    python llama_cpp_vulkan.py --mode generate --model models/TinyLlama-Q4_K_M.gguf \\
        --max-tokens 256 --device 0
    python llama_cpp_vulkan.py --mode bench --model models/Phi-3-Q4_K_M.gguf \\
        --duration 300
"""

import argparse
import os
import shutil
import signal
import subprocess
import sys
import time
from pathlib import Path


_DEFAULT_LLAMA_DIR = Path(r"C:\Users\robir\Documents\GPU Monitoring\tools\llama-cpp-vulkan")


def find_binary(name: str, llama_dir: Path) -> str:
    candidates = [llama_dir / f"{name}.exe", llama_dir / name]
    for c in candidates:
        if c.is_file():
            return str(c)
    on_path = shutil.which(name) or shutil.which(f"{name}.exe")
    if on_path:
        return on_path
    print(f"ERROR: {name}.exe not found in {llama_dir} or on PATH.", file=sys.stderr)
    sys.exit(1)


def run_generate(args, llama_cli: str):
    cmd = [
        llama_cli,
        "-m", args.model,
        "-n", str(args.max_tokens),
        "-p", args.prompt,
        "--device", f"Vulkan{args.device}",
        "-ngl", "999",                # offload all layers to GPU
        "--ctx-size", str(args.ctx_size),
        "--no-display-prompt",
    ]
    print(f"Running: {' '.join(cmd)}")
    wall_start = time.time()
    iters = 0
    while time.time() - wall_start < args.duration:
        proc = subprocess.run(cmd, capture_output=False)
        iters += 1
        if proc.returncode != 0:
            print(f"  iteration {iters}: llama-cli returned {proc.returncode}", file=sys.stderr)
            break
        if args.duration <= 0:
            break  # run-once mode
    print(f"Generate done: {iters} iterations in {time.time() - wall_start:.1f}s")


def run_bench(args, llama_bench: str):
    cmd = [
        llama_bench,
        "-m", args.model,
        "-p", "128",                  # prompt-processing batch
        "-n", "128",                  # token-generation batch
        "-r", "3",                    # repetitions
        "--device", f"Vulkan{args.device}",
        "-ngl", "999",
    ]
    print(f"Running: {' '.join(cmd)}")
    stopped = False

    def handle_signal(signum, frame):
        nonlocal stopped
        stopped = True

    signal.signal(signal.SIGTERM, handle_signal)
    if hasattr(signal, "SIGBREAK"):
        signal.signal(signal.SIGBREAK, handle_signal)

    wall_start = time.time()
    runs = 0
    while time.time() - wall_start < args.duration and not stopped:
        proc = subprocess.run(cmd, capture_output=False)
        runs += 1
        if proc.returncode != 0:
            print(f"  bench run {runs}: returned {proc.returncode}", file=sys.stderr)
            break
    print(f"Bench done: {runs} runs in {time.time() - wall_start:.1f}s")


def main():
    ap = argparse.ArgumentParser(description="llama.cpp Vulkan inference (Windows)")
    ap.add_argument("--mode", choices=["generate", "bench"], default="generate")
    ap.add_argument("--model", required=True, help="Path to a .gguf model file")
    ap.add_argument("--max-tokens", type=int, default=256,
                    help="Tokens per generation iteration (generate mode)")
    ap.add_argument("--ctx-size", type=int, default=2048)
    ap.add_argument("--prompt", type=str,
                    default="Write a one-paragraph summary of the laws of thermodynamics.")
    ap.add_argument("--device", type=int, default=0,
                    help="Vulkan device index (0 is usually the dGPU)")
    ap.add_argument("--duration", type=int, default=300,
                    help="Wall-clock seconds to keep looping iterations (0=once)")
    ap.add_argument("--llama-dir", type=str, default=str(_DEFAULT_LLAMA_DIR))
    args = ap.parse_args()

    if not os.path.isfile(args.model):
        print(f"ERROR: model file not found: {args.model}\n"
              "Download a GGUF model first, e.g. with:\n"
              "  curl -L -o models/TinyLlama-Q4_K_M.gguf "
              "https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/"
              "resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf",
              file=sys.stderr)
        sys.exit(1)

    llama_dir = Path(args.llama_dir)
    if args.mode == "generate":
        run_generate(args, find_binary("llama-cli", llama_dir))
    else:
        run_bench(args, find_binary("llama-bench", llama_dir))


if __name__ == "__main__":
    main()
