#!/usr/bin/env python3
"""LoRA fine-tuning workload on a small base model.

Fine-tunes a small base model (TinyLlama-1.1B or Qwen2.5-0.5B) with LoRA
adapters on a tiny WikiText-2 subset. The point isn't model quality — it's
adding a parameter-efficient training workload to the telemetry corpus.
LoRA has a distinct signature from full fine-tuning: small trainable
parameter footprint, backward passes that only update adapter weights.

This is the closest analogue to whitebox_lora in the NVIDIA registry, which
the paper identifies as the single evasion family that drops below 50%
detection.

Usage:
    python lora_finetune.py --steps 200
    python lora_finetune.py --steps 200 --base qwen25_0p5b
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM, default_data_collator
from datasets import load_dataset
from peft import LoraConfig, get_peft_model


BASE_MODELS = {
    "tinyllama": "TinyLlama/TinyLlama-1.1B-Chat-v1.0",
    "qwen25_0p5b": "Qwen/Qwen2.5-0.5B-Instruct",
    "smollm2_135m": "HuggingFaceTB/SmolLM2-135M-Instruct",
}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base", choices=list(BASE_MODELS), default="qwen25_0p5b")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=2)
    p.add_argument("--seq-len", type=int, default=256)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--lora-r", type=int, default=16)
    p.add_argument("--lora-alpha", type=int, default=32)
    args = p.parse_args()

    if not torch.cuda.is_available():
        print("ERROR: no GPU"); return 1
    device = torch.device("cuda")
    name = BASE_MODELS[args.base]

    print(f"LoRA fine-tune: {name} r={args.lora_r} alpha={args.lora_alpha} "
          f"steps={args.steps} bs={args.batch_size} seq_len={args.seq_len}")

    tok = AutoTokenizer.from_pretrained(name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(name, torch_dtype=torch.float16).to(device)
    cfg = LoraConfig(
        r=args.lora_r, lora_alpha=args.lora_alpha, lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"] if "qwen" in args.base or "llama" in args.base else "all-linear",
        bias="none", task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, cfg)
    model.print_trainable_parameters()

    ds = load_dataset("wikitext", "wikitext-2-v1", split="train")
    def tokenize(batch):
        enc = tok(batch["text"], truncation=True, padding="max_length",
                  max_length=args.seq_len, return_tensors="pt")
        enc["labels"] = enc["input_ids"].clone()
        return enc
    ds = ds.filter(lambda x: len(x["text"]) > 32).select(range(min(2000, len(ds)))).map(
        tokenize, batched=True, remove_columns=["text"]
    )
    ds.set_format("torch")
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True,
                        collate_fn=default_data_collator, num_workers=0)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    model.train()
    start = time.time()
    seen = 0
    for step in range(args.steps):
        try:
            batch = next(iter_data)
        except (NameError, StopIteration):
            iter_data = iter(loader)
            batch = next(iter_data)
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        out = model(**batch)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        seen += args.batch_size
        if (step + 1) % 25 == 0:
            print(f"step {step + 1}/{args.steps}  loss={out.loss.item():.4f}  "
                  f"elapsed={time.time() - start:.1f}s")
    elapsed = time.time() - start
    print(f"Done. {args.steps} LoRA steps in {elapsed:.1f}s ({seen / elapsed:.1f} samples/s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
