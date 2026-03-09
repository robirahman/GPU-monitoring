#!/usr/bin/env python3
"""BERT-base fine-tuning workload for SPAR telemetry collection.

Fine-tunes BERT-base-uncased on SST-2 (sentiment classification) using
Hugging Face Transformers. Runs for a fixed number of steps.

Usage:
    python bert_finetune.py
    python bert_finetune.py --steps 300 --batch-size 32 --amp
"""

import argparse
import time

import torch
from torch.utils.data import DataLoader, Dataset
from transformers import BertForSequenceClassification, BertTokenizer


class Sst2Dataset(Dataset):
    """SST-2 sentiment dataset (or synthetic fallback)."""

    def __init__(self, tokenizer, max_len: int = 128, num_samples: int = 4000):
        print("Loading SST-2 dataset...")
        texts, labels = [], []
        try:
            from datasets import load_dataset
            ds = load_dataset("glue", "sst2", split="train")
            for item in ds:
                texts.append(item["sentence"])
                labels.append(item["label"])
                if len(texts) >= num_samples:
                    break
        except Exception as e:
            print(f"  Dataset load failed ({e}), using synthetic data.")
            texts = ["This movie was great and I loved it." * 3,
                     "Terrible film, waste of time and money." * 3] * (num_samples // 2)
            labels = [1, 0] * (num_samples // 2)

        enc = tokenizer(texts, padding="max_length", truncation=True,
                        max_length=max_len, return_tensors="pt")
        self.input_ids = enc["input_ids"]
        self.attention_mask = enc["attention_mask"]
        self.labels = torch.tensor(labels[:len(self.input_ids)], dtype=torch.long)

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return {
            "input_ids": self.input_ids[idx],
            "attention_mask": self.attention_mask[idx],
            "labels": self.labels[idx],
        }


def main():
    parser = argparse.ArgumentParser(description="BERT fine-tuning workload")
    parser.add_argument("--steps", type=int, default=300, help="Training steps (default: 300)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size (default: 32)")
    parser.add_argument("--max-len", type=int, default=128, help="Max token length (default: 128)")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--amp", action="store_true", help="Enable mixed precision")
    parser.add_argument("--device", type=str, default="cuda")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = torch.device(args.device)

    print(f"BERT-base fine-tuning on SST-2")
    print(f"  Device: {device}, Steps: {args.steps}, Batch: {args.batch_size}, AMP: {args.amp}")

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
    model = BertForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    ).to(device)

    dataset = Sst2Dataset(tokenizer, max_len=args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True,
                        num_workers=2, pin_memory=(args.device == "cuda"), drop_last=True)

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.01)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp)

    model.train()
    wall_start = time.time()
    step = 0
    loader_iter = iter(loader)

    while step < args.steps:
        try:
            batch = next(loader_iter)
        except StopIteration:
            loader_iter = iter(loader)
            batch = next(loader_iter)

        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        with torch.amp.autocast("cuda", enabled=args.amp):
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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
    print(f"BERT fine-tuning complete. Steps: {step}, Wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
