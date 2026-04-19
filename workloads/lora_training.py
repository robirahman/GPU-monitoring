#!/usr/bin/env python3
"""Generic LoRA fine-tuning harness for matched-pair training data.

Closes the Round 5b data gap: we have inference telemetry at multiple model
sizes on consumer GPUs, but training only at TinyLlama 1.1B scale. This
script runs a pure LoRA fine-tuning loop against any HF causal-LM
`--model` id, so we can collect matched train-vs-infer pairs at 1.5B,
1.7B, 3B, 3.8B (and 7B on 24 GB+ hardware).

Forward+backward+step every batch (no dilution, no oracle gating) so the
telemetry is unambiguously training. Uses AdamW on LoRA adapters only,
gradient checkpointing off by default. Synthetic tokens — avoids
dataloader jitter so the GPU telemetry reflects the compute signature.

Usage:
    python workloads/lora_training.py --model Qwen/Qwen2.5-1.5B-Instruct \\
        --duration 300 --batch-size 2 --seq-len 256 --rank 16
    python workloads/lora_training.py --model microsoft/Phi-3-mini-4k-instruct \\
        --duration 300 --batch-size 1 --seq-len 512
"""

import argparse
import os
import time

import torch


def load_model(model_id: str, device: str):
    """Load a causal-LM in FP16 for LoRA fine-tuning."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_id} (fp16)")
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        device_map=device,
        trust_remote_code=False,
    )
    n_params = sum(p.numel() for p in model.parameters())
    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  params: {n_params / 1e9:.2f} B  |  post-load memory: {mem_mb:.0f} MB")
    return model, tokenizer


def apply_lora(model, rank: int, alpha: int, target_modules):
    """Attach LoRA adapters to the given target modules.

    `target_modules` may be a list of module-name substrings
    (e.g. ['q_proj', 'v_proj']) or the string 'all-linear' (PEFT will
    then apply LoRA to every nn.Linear in the model — works for any arch
    but produces a larger adapter).
    """
    from peft import LoraConfig, TaskType, get_peft_model

    config = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=rank,
        lora_alpha=alpha,
        lora_dropout=0.05,
        target_modules=target_modules,
    )
    peft_model = get_peft_model(model, config)
    n_trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in peft_model.parameters())
    print(f"  LoRA r={rank} alpha={alpha}: "
          f"{n_trainable/1e6:.1f} M trainable / {n_total/1e6:.0f} M total "
          f"({n_trainable/n_total:.2%})")
    return peft_model


def make_synthetic_batch(vocab_size: int, batch_size: int, seq_len: int, device: str):
    input_ids = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask, "labels": input_ids}


def train_loop(model, tokenizer, args, device: str) -> None:
    optim = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
    )
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    model.train()

    vocab_size = tokenizer.vocab_size
    start = time.time()
    last_print = start
    steps = 0
    total_tokens = 0
    total_loss = 0.0

    while time.time() - start < args.duration:
        batch = make_synthetic_batch(vocab_size, args.batch_size, args.seq_len, device)
        out = model(**batch)
        loss = out.loss
        loss.backward()
        optim.step()
        optim.zero_grad(set_to_none=True)

        steps += 1
        total_tokens += args.batch_size * args.seq_len
        total_loss += float(loss.detach())

        now = time.time()
        if now - last_print > 10:
            elapsed = now - start
            mem_mb = torch.cuda.memory_allocated() / 1024**2
            print(f"  step {steps:>5d}  loss={total_loss/steps:.3f}  "
                  f"{total_tokens/elapsed:.0f} tok/s  mem={mem_mb:.0f} MB  "
                  f"t={elapsed:.0f}s")
            last_print = now

    elapsed = time.time() - start
    print(f"LoRA train done: {steps} steps, {total_tokens} tokens, "
          f"{total_tokens/elapsed:.0f} tok/s in {elapsed:.1f}s")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True,
                    help="HF model id (e.g. Qwen/Qwen2.5-1.5B-Instruct)")
    ap.add_argument("--duration", type=int, default=300)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--seq-len", type=int, default=256)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--lora-alpha", type=int, default=32)
    ap.add_argument(
        "--target-modules", default="q_proj,v_proj",
        help="Comma-separated list of module-name substrings, or 'all-linear' "
             "for every nn.Linear in the model. Defaults to Llama/Qwen-style "
             "'q_proj,v_proj'; Phi-3 needs 'qkv_proj' (fused QKV).",
    )
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--grad-ckpt", action="store_true",
                    help="Enable gradient checkpointing (fits larger models)")
    ap.add_argument("--device", default="cuda")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"

    os.environ.setdefault("HF_HOME", os.path.expanduser("~/.cache/huggingface"))

    tm = args.target_modules.strip()
    target_modules = tm if tm == "all-linear" else [s.strip() for s in tm.split(",") if s.strip()]

    model, tokenizer = load_model(args.model, args.device)
    model = apply_lora(model, rank=args.rank, alpha=args.lora_alpha,
                       target_modules=target_modules)
    train_loop(model, tokenizer, args, args.device)


if __name__ == "__main__":
    main()
