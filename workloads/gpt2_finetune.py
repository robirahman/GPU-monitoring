#!/usr/bin/env python3
"""GPT-2 124M fine-tuning on WikiText-2.

Uses Hugging Face Transformers. This is the primary "large model training"
signature for the classifier — heavy tensor core usage, high memory, periodic
epoch patterns.

Usage:
    python gpt2_finetune.py --epochs 3 --batch-size 4
    python gpt2_finetune.py --epochs 3 --batch-size 4 --amp
"""

import argparse
import time

import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def main():
    parser = argparse.ArgumentParser(description="GPT-2 fine-tuning on WikiText-2")
    parser.add_argument("--epochs", type=int, default=3, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=4, help="Per-device batch size")
    parser.add_argument("--lr", type=float, default=5e-5, help="Learning rate")
    parser.add_argument("--amp", action="store_true", help="Enable FP16 mixed precision")
    parser.add_argument("--max-length", type=int, default=512, help="Max sequence length")
    parser.add_argument("--output-dir", type=str, default="/tmp/gpt2_ft", help="Checkpoint dir")
    args = parser.parse_args()

    print(f"GPT-2 124M fine-tuning on WikiText-2")
    print(f"  Epochs: {args.epochs}, Batch size: {args.batch_size}, LR: {args.lr}")
    print(f"  AMP (FP16): {args.amp}, Max length: {args.max_length}")

    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("gpt2")

    dataset = load_dataset("wikitext", "wikitext-2-raw-v1")

    def tokenize(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=args.max_length,
            padding="max_length",
        )

    tokenized = dataset.map(tokenize, batched=True, remove_columns=["text"])

    # Filter out empty sequences
    tokenized = tokenized.filter(lambda x: sum(x["attention_mask"]) > 1)

    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

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
        remove_unused_columns=False,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized["train"],
        data_collator=data_collator,
    )

    wall_start = time.time()
    trainer.train()
    wall_time = time.time() - wall_start
    print(f"GPT-2 fine-tuning complete. Total wall time: {wall_time:.1f}s")


if __name__ == "__main__":
    main()
