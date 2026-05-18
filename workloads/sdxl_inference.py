#!/usr/bin/env python3
"""SDXL image generation inference workload.

Uses the SDXL Turbo distilled model (4-step generation, fits in ~7 GB fp16
VRAM) so this works on any consumer ROCm card with 8+ GB. The aim is to
add a non-LLM, non-classification inference workload to the corpus —
diffusion is a genuinely different telemetry signature (heavy U-Net forward
passes with attention, repeated VAE decode steps).

Usage:
    python sdxl_inference.py --duration 300
    python sdxl_inference.py --duration 300 --steps 4 --batch-size 1
"""

import argparse
import time

import torch
from diffusers import AutoPipelineForText2Image


PROMPTS = [
    "a serene forest at sunrise, photorealistic",
    "a futuristic city skyline at dusk, cyberpunk",
    "an oil painting of a sailing ship in stormy seas",
    "a macro photograph of a beetle on a leaf",
    "a watercolor portrait of a tabby cat",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--duration", type=int, default=300)
    parser.add_argument("--steps", type=int, default=4,
                        help="Inference steps (SDXL Turbo works at 1-4)")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--model", type=str, default="stabilityai/sdxl-turbo")
    args = parser.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU visible to PyTorch")
        return 1

    device = torch.device("cuda")
    print(f"SDXL Turbo on {torch.cuda.get_device_name(0)}, steps={args.steps}, bs={args.batch_size}")

    pipe = AutoPipelineForText2Image.from_pretrained(
        args.model, torch_dtype=torch.float16, variant="fp16",
    ).to(device)
    pipe.set_progress_bar_config(disable=True)

    end = time.time() + args.duration
    n = 0
    pi = 0
    while time.time() < end:
        prompt = PROMPTS[pi % len(PROMPTS)]
        pi += 1
        _ = pipe(
            prompt=[prompt] * args.batch_size,
            num_inference_steps=args.steps,
            guidance_scale=0.0,
            output_type="latent",
        ).images
        torch.cuda.synchronize()
        n += args.batch_size

    elapsed = (args.duration if time.time() >= end else time.time() - (end - args.duration))
    print(f"Generated {n} images ({n / elapsed:.2f} img/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
