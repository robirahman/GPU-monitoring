#!/usr/bin/env python3
"""GPT-2 124M fine-tuning workload for SPAR telemetry collection.

Fine-tunes GPT-2 small on WikiText-2 using Hugging Face Transformers.
Runs for a fixed number of steps to produce a ~10-15 minute GPU workload.

Usage:
    python gpt2_finetune.py
    python gpt2_finetune.py --steps 500 --batch-size 8 --amp
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import GPT2LMHeadModel, GPT2Tokenizer


class WikiTextDataset(Dataset):
    """Tokenized WikiText-2 dataset chunked into fixed-length sequences."""

    def __init__(self, tokenizer, seq_len: int = 512, num_samples: int = 2000):
        print("Loading WikiText-2 dataset...")
        try:
            from datasets import load_dataset
            ds = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
            text = "\n".join(ds["text"])
        except Exception as e:
            print(f"  Dataset load failed ({e}), using synthetic text.")
            text = ("The quick brown fox jumps over the lazy dog. " * 500 + "\n") * 200
        tokens = tokenizer.encode(text, add_special_tokens=False)
        # Truncate/pad to get evenly sized chunks
        n_chunks = min(num_samples, len(tokens) // seq_len)
        tokens = tokens[: n_chunks * seq_len]
        self.input_ids = torch.tensor(tokens).view(n_chunks, seq_len)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        x = self.input_ids[idx]
        return x, x  # input = label for LM


def main():
    parser = argparse.ArgumentParser(description="GPT-2 fine-tuning workload")
    parser.add_argument("--steps", type=int, default=400, help="Training steps (default: 400)")
    parser.add_argument("--batch-size", type=int, default=8, help="Batch size (default: 8)")
    parser.add_argument("--seq-len", type=int, default=512, help="Sequence length (default: 512)")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"GPT-2 124M fine-tuning")
    print(f"  Device: {device}, Steps: {args.steps}, Batch: {args.batch_size}, AMP: {args.amp}")

    tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2LMHeadModel.from_pretrained("gpt2").to(device)

    dataset = WikiTextDataset(tokenizer, seq_len=args.seq_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=(args.device == "cuda"), drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    model.train()
    wall_start = time.time()
    step = 0
    loader_iter = iter(loader)

    while step < args.steps:
        try:
            input_ids, labels = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            input_ids, labels = next(loader_iter)

        input_ids = input_ids.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            outputs = model(input_ids, labels=labels)
            loss = outputs.loss

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(optimizer)
        scaler.update()

        step += 1
        if step % 50 == 0:
            elapsed = time.time() - wall_start
            print(f"  Step {step}/{args.steps}: loss={loss.item():.4f}, elapsed={elapsed:.1f}s")

    wall_time = time.time() - wall_start
    print(f"GPT-2 fine-tuning complete. Steps: {step}, Wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
