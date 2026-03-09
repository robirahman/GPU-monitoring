#!/usr/bin/env python3
"""BERT-base fine-tuning on SST-2 (sentiment classification).

Uses Hugging Face Transformers. Produces a distinct training signature —
encoder-only architecture with shorter sequences than GPT-2.

Usage:
    python bert_finetune.py --epochs 3 --batch-size 32
    python bert_finetune.py --epochs 3 --batch-size 32 --amp
"""

import argparse
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(description="BERT fine-tuning on SST-2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=32, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=2e-5, help="Learning rate")
    parser.add_argument("--amp", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--max-length", type=int, default=128, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default="/tmp/bert_ft", help="Checkpoint dir")
    args = parser.parse_args()

    print(f"BERT-base fine-tuning on SST-2")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  AMP (FP16): {args.amp}, Max length: {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModelForSequenceClassification.from_pretrained(
        "bert-base-uncased", num_labels=2
    )

    dataset = load_dataset("glue", "sst2")

    def tokenize(examples):
        return tokenizer(
            examples["sentence"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True)
    tokenized = tokenized.rename_column("label", "labels")

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.lr,
        fp16=args.amp,
        logging_steps=50,
        save_strategy="no",
        report_to="none",
        dataloader_num_workers=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
    )

    wall_start = time.time()
    trainer.train()
    wall_time = time.time() - wall_start
    print(f"BERT fine-tuning complete. Total wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
