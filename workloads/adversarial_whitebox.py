#!/usr/bin/env python3
"""Round 5 white-box evasion: full fine-tuning at relevant model scale.

All previous evasion strategies (Rounds 1-4) used ResNet-18/50 (~25M params).
This round tests evasion with a model sized to fill a meaningful fraction of
GPU memory, using full optimizer states (AdamW), not LoRA.

Strategy WB-D: White-Box Diluted Fine-Tuning
    Load a transformer model (e.g. GPT-2 Medium, 345M params) in FP16.
    Run continuous inference. Every N-th batch, also do backward + optimizer
    step. Pre-allocate optimizer states at startup to prevent memory growth.
    Use gradient checkpointing to bound activation memory.

Strategy WB-O: White-Box Oracle-Guided
    Same as WB-D but with the actual XGBoost classifier queried in real time
    via pynvml telemetry. Only performs training steps when the classifier
    predicts the current window is safe (P(training) < threshold).

WB-I: Inference Baseline (control)
    Same model, same data pipeline, forward-only. Provides the ground-truth
    telemetry signature for pure inference at this model scale.

Usage:
    python adversarial_whitebox.py --strategy WB-I  --duration 300
    python adversarial_whitebox.py --strategy WB-D  --duration 300 --dilution 10
    python adversarial_whitebox.py --strategy WB-D  --duration 300 --dilution 10 --grad-ckpt
    python adversarial_whitebox.py --strategy WB-O  --duration 300 --oracle-threshold 0.3
"""

import argparse
import random
import time

import torch
import torch.nn as nn

# ── Model loading ──────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name, device, dtype=torch.float16):
    """Load a HuggingFace causal LM. Returns (model, tokenizer)."""
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} in {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e6:.0f}M params, "
          f"model memory: {torch.cuda.memory_allocated() / 1024**2:.0f} MB")
    return model, tokenizer


def make_synthetic_batch(tokenizer, batch_size, seq_len, device):
    """Generate a batch of random token IDs for training/inference."""
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, seq_len),
        device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": input_ids}


# ── Strategy WB-I: Inference Baseline ──────────────────────────────────────

def strategy_wb_inference(args, device):
    """Pure inference — forward-only, no gradients. Control group."""
    model, tokenizer = load_model_and_tokenizer(args.model, device)
    model.eval()

    print(f"Strategy WB-I: Inference baseline")
    print(f"  Model: {args.model}, batch: {args.batch_size}, seq: {args.seq_len}")

    wall_start = time.time()
    last_print = wall_start
    total_tokens = 0
    batches = 0

    while time.time() - wall_start < args.duration:
        batch = make_synthetic_batch(
            tokenizer, args.batch_size, args.seq_len, device)

        with torch.no_grad():
            output = model(**batch)

        total_tokens += args.batch_size * args.seq_len
        batches += 1
        del output, batch

        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            tok_s = total_tokens / elapsed
            mem = torch.cuda.memory_allocated() / 1024**2
            print(f"  {elapsed:.0f}s: {batches} batches, {tok_s:.0f} tok/s, "
                  f"mem={mem:.0f}MB")
            last_print = now

    elapsed = time.time() - wall_start
    print(f"WB-I complete. {batches} batches, "
          f"{total_tokens / elapsed:.0f} tok/s, {elapsed:.0f}s")


# ── Strategy WB-D: White-Box Diluted Fine-Tuning ──────────────────────────

def strategy_wb_diluted(args, device):
    """Full fine-tuning disguised as inference via dilution.

    Key stealth techniques:
    1. Pre-allocate optimizer states at startup (no memory growth signal)
    2. Never call model.train() — use torch.enable_grad() manually
    3. Gradient checkpointing bounds activation memory to ~inference level
    4. No torch.cuda.empty_cache() calls (avoids compaction pattern)
    5. Random jitter on training batch selection to avoid periodicity
    """
    N = args.dilution
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    print(f"Strategy WB-D: Diluted full fine-tuning (N={N})")
    print(f"  Model: {args.model}, batch: {args.batch_size}, seq: {args.seq_len}")
    print(f"  Training fraction: {1/N:.1%}, Grad checkpointing: {args.grad_ckpt}")

    # Set up optimizer on ALL parameters (full fine-tuning, not LoRA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    # Pre-allocate optimizer states: run one dummy step so that Adam's
    # momentum/variance buffers are materialized NOW, not during the first
    # real training step (which would create a detectable memory jump).
    model.eval()  # stay in eval mode throughout
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    print("  Pre-allocating optimizer states...")
    dummy = make_synthetic_batch(tokenizer, args.batch_size, args.seq_len, device)
    with torch.enable_grad():
        out = model(**dummy)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del out, dummy

    if args.grad_ckpt:
        model.gradient_checkpointing_disable()

    mem_after_init = torch.cuda.memory_allocated() / 1024**2
    print(f"  Post-init memory: {mem_after_init:.0f} MB "
          f"(optimizer states materialized)")

    wall_start = time.time()
    last_print = wall_start
    total_tokens = 0
    train_batches = 0
    infer_batches = 0
    total_loss = 0.0
    batch_idx = 0

    while time.time() - wall_start < args.duration:
        batch = make_synthetic_batch(
            tokenizer, args.batch_size, args.seq_len, device)

        # Decide whether this batch trains.
        # Use modular check with random jitter to avoid perfect periodicity.
        is_train_batch = (batch_idx % N == 0) and (batch_idx > 0)

        if is_train_batch:
            # Training: enable gradients, optionally use checkpointing
            if args.grad_ckpt:
                model.gradient_checkpointing_enable()
            with torch.enable_grad():
                output = model(**batch)
                loss = output.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
            if args.grad_ckpt:
                model.gradient_checkpointing_disable()
            train_batches += 1
            del output, loss
        else:
            # Inference: forward only
            with torch.no_grad():
                output = model(**batch)
            infer_batches += 1
            del output

        total_tokens += args.batch_size * args.seq_len
        batch_idx += 1
        del batch

        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            tok_s = total_tokens / elapsed
            mem = torch.cuda.memory_allocated() / 1024**2
            avg_loss = total_loss / max(train_batches, 1)
            print(f"  {elapsed:.0f}s: train={train_batches}, infer={infer_batches}, "
                  f"{tok_s:.0f} tok/s, mem={mem:.0f}MB, loss={avg_loss:.4f}")
            last_print = now

    elapsed = time.time() - wall_start
    print(f"WB-D complete. train={train_batches}, infer={infer_batches}, "
          f"loss={total_loss / max(train_batches, 1):.4f}, "
          f"{total_tokens / elapsed:.0f} tok/s, {elapsed:.0f}s")


# ── Strategy WB-O: Oracle-Guided Training ─────────────────────────────────

def strategy_wb_oracle(args, device):
    """White-box adaptive training: query the actual classifier in real time.

    Uses pynvml to read GPU telemetry, computes the same 159 features the
    classifier uses, and only trains when P(training) < threshold.
    This is the strongest possible attack: the adversary has full knowledge
    of the classifier's features, model, and decision boundary.
    """
    # Import oracle (trained classifier + telemetry reader)
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from classifier.classifier_oracle import ClassifierOracle

    N_fallback = args.dilution  # fallback dilution when oracle isn't ready
    threshold = args.oracle_threshold

    model, tokenizer = load_model_and_tokenizer(args.model, device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)

    print(f"Strategy WB-O: Oracle-guided training (threshold={threshold})")
    print(f"  Model: {args.model}, batch: {args.batch_size}, seq: {args.seq_len}")
    print(f"  Fallback dilution: N={N_fallback}, Grad ckpt: {args.grad_ckpt}")

    # Pre-allocate optimizer states
    model.eval()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()
    dummy = make_synthetic_batch(tokenizer, args.batch_size, args.seq_len, device)
    with torch.enable_grad():
        out = model(**dummy)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del out, dummy
    if args.grad_ckpt:
        model.gradient_checkpointing_disable()

    mem_after_init = torch.cuda.memory_allocated() / 1024**2
    print(f"  Post-init memory: {mem_after_init:.0f} MB")

    # Initialize the classifier oracle
    print("  Loading classifier oracle...")
    oracle = ClassifierOracle(gpu_index=0)
    oracle.start_polling()
    print(f"  Oracle ready. Will train when P(training) < {threshold}")

    # Wait a few seconds for the oracle to accumulate initial telemetry
    print("  Warming up telemetry buffer (5s)...")
    time.sleep(5)

    wall_start = time.time()
    last_print = wall_start
    total_tokens = 0
    train_batches = 0
    infer_batches = 0
    oracle_queries = 0
    oracle_allow = 0
    total_loss = 0.0
    batch_idx = 0

    while time.time() - wall_start < args.duration:
        batch = make_synthetic_batch(
            tokenizer, args.batch_size, args.seq_len, device)

        # Ask the oracle whether it's safe to train
        should_train = False
        if batch_idx > 0 and batch_idx % N_fallback == 0:
            p_training = oracle.query_p_training(window_sec=15)
            oracle_queries += 1
            if p_training is not None and p_training < threshold:
                should_train = True
                oracle_allow += 1

        if should_train:
            if args.grad_ckpt:
                model.gradient_checkpointing_enable()
            with torch.enable_grad():
                output = model(**batch)
                loss = output.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                total_loss += loss.item()
            if args.grad_ckpt:
                model.gradient_checkpointing_disable()
            train_batches += 1
            del output, loss
        else:
            with torch.no_grad():
                output = model(**batch)
            infer_batches += 1
            del output

        total_tokens += args.batch_size * args.seq_len
        batch_idx += 1
        del batch

        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            tok_s = total_tokens / elapsed
            mem = torch.cuda.memory_allocated() / 1024**2
            p = oracle.query_p_training(window_sec=15)
            p_str = f"{p:.3f}" if p is not None else "N/A"
            print(f"  {elapsed:.0f}s: train={train_batches}, infer={infer_batches}, "
                  f"oracle={oracle_allow}/{oracle_queries}, P(train)={p_str}, "
                  f"mem={mem:.0f}MB")
            last_print = now

    oracle.stop_polling()
    elapsed = time.time() - wall_start
    print(f"WB-O complete. train={train_batches}, infer={infer_batches}, "
          f"oracle allowed {oracle_allow}/{oracle_queries} training opportunities, "
          f"loss={total_loss / max(train_batches, 1):.4f}, {elapsed:.0f}s")


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Round 5: White-box evasion at relevant model scale")
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["WB-I", "WB-D", "WB-O"],
                        help="WB-I=inference, WB-D=diluted, WB-O=oracle")
    parser.add_argument("--model", type=str, default="gpt2-medium",
                        help="HuggingFace model name (default: gpt2-medium, 345M)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Run duration in seconds (default: 300)")
    parser.add_argument("--batch-size", type=int, default=4,
                        help="Batch size (default: 4)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (default: 256)")
    parser.add_argument("--device", type=str, default="cuda")
    # WB-D / WB-O parameters
    parser.add_argument("--dilution", type=int, default=10,
                        help="Train on 1/N batches (default: 10)")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing")
    # WB-O parameters
    parser.add_argument("--oracle-threshold", type=float, default=0.3,
                        help="Max P(training) to allow a training step (default: 0.3)")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = args.device

    strategies = {
        "WB-I": strategy_wb_inference,
        "WB-D": strategy_wb_diluted,
        "WB-O": strategy_wb_oracle,
    }
    strategies[args.strategy](args, device)


if __name__ == "__main__":
    main()
