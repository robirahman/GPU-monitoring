#!/usr/bin/env python3
"""Round 5 white-box evasion: full training at relevant model scale.

All previous evasion strategies (Rounds 1-4) used ResNet-18/50 (~25M params).
This round tests evasion with models sized to fill GPU memory, using both
full AdamW fine-tuning and LoRA.

Model sizing (FP16 training with AdamW requires ~16 bytes/param):
  RTX 5080  (16 GB):  ~1B  params  → TinyLlama-1.1B  (batch 1, seq 256, grad ckpt)
  A100/H100 (80 GB):  ~3-5B params → Qwen2.5-3B or Llama-3.2-3B
  H200      (141 GB): ~7-8B params → Llama-3.1-8B
  B200      (192 GB): ~10-12B      → Llama-3.1-8B (larger batch) or 12B custom

Strategies:
  WB-I:  Inference baseline (control) — forward-only, no gradients
  WB-D:  Diluted full fine-tuning — every N-th batch does backward + AdamW step
  WB-L:  Diluted LoRA fine-tuning — same but only LoRA params are trained
  WB-O:  Oracle-guided training — actual classifier queried in real time

Usage:
    # Inference baseline (RTX 5080)
    python adversarial_whitebox.py --strategy WB-I --model TinyLlama/TinyLlama-1.1B-Chat-v1.0

    # Full fine-tuning, dilution N=10, gradient checkpointing (RTX 5080)
    python adversarial_whitebox.py --strategy WB-D --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --batch-size 1 --seq-len 256 --dilution 10 --grad-ckpt

    # LoRA fine-tuning, dilution N=10 (RTX 5080)
    python adversarial_whitebox.py --strategy WB-L --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --batch-size 2 --seq-len 256 --dilution 10

    # Oracle-guided (RTX 5080)
    python adversarial_whitebox.py --strategy WB-O --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \\
        --batch-size 1 --seq-len 256 --dilution 10 --grad-ckpt
"""

import argparse
import time

import torch
import torch.nn as nn


# ── Model loading ──────────────────────────────────────────────────────────

# Recommended models per GPU VRAM (fills memory for training, not just inference)
MODEL_RECOMMENDATIONS = {
    "16GB":  "TinyLlama/TinyLlama-1.1B-Chat-v1.0",  # 1.1B — batch 1, seq 256, grad ckpt
    "80GB":  "Qwen/Qwen2.5-3B",                       # 3B — batch 2, seq 512
    "141GB": "meta-llama/Llama-3.1-8B",                # 8B — batch 2, seq 512
    "192GB": "meta-llama/Llama-3.1-8B",                # 8B — batch 4, seq 1024
}


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
    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  {n_params / 1e9:.2f}B params ({n_params / 1e6:.0f}M), "
          f"model memory: {mem_mb:.0f} MB")
    return model, tokenizer


def apply_lora(model, rank=16, alpha=32, target_modules=None):
    """Apply LoRA adapters to the model. Returns (peft_model, n_trainable)."""
    from peft import LoraConfig, get_peft_model, TaskType

    if target_modules is None:
        # Target attention projections — works for most transformer architectures
        target_modules = ["q_proj", "v_proj"]

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
    print(f"  LoRA applied: {n_trainable / 1e6:.1f}M trainable / "
          f"{n_total / 1e6:.0f}M total ({n_trainable / n_total:.2%})")
    return peft_model, n_trainable


def make_synthetic_batch(tokenizer, batch_size, seq_len, device):
    """Generate a batch of random token IDs for training/inference."""
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, seq_len), device=device,
    )
    attention_mask = torch.ones_like(input_ids)
    return {"input_ids": input_ids, "attention_mask": attention_mask,
            "labels": input_ids}


def _pre_allocate_optimizer(model, optimizer, tokenizer, args, device):
    """Run one dummy backward pass to materialize AdamW momentum/variance buffers.

    This prevents the memory growth signal that normally betrays training
    initialization. After this call, memory is stable — no more jumps during
    actual training steps.
    """
    print("  Pre-allocating optimizer states...")
    dummy = make_synthetic_batch(tokenizer, args.batch_size, args.seq_len, device)
    with torch.enable_grad():
        out = model(**dummy)
        out.loss.backward()
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
    del out, dummy
    mem_mb = torch.cuda.memory_allocated() / 1024**2
    print(f"  Post-init memory: {mem_mb:.0f} MB (optimizer states materialized)")


def _run_diluted_loop(model, optimizer, tokenizer, args, device, oracle=None):
    """Core training loop shared by WB-D, WB-L, and WB-O.

    If oracle is provided, queries it before each potential training step
    and only trains when P(training) < oracle_threshold.
    """
    N = args.dilution
    threshold = args.oracle_threshold if oracle else None

    wall_start = time.time()
    last_print = wall_start
    total_tokens = 0
    train_batches = 0
    infer_batches = 0
    total_loss = 0.0
    batch_idx = 0
    oracle_queries = 0
    oracle_allow = 0

    while time.time() - wall_start < args.duration:
        batch = make_synthetic_batch(
            tokenizer, args.batch_size, args.seq_len, device)

        # Decide whether this batch trains
        is_candidate = (batch_idx % N == 0) and (batch_idx > 0)

        should_train = False
        if is_candidate:
            if oracle is not None:
                p_training = oracle.query_p_training(window_sec=15)
                oracle_queries += 1
                if p_training is not None and p_training < threshold:
                    should_train = True
                    oracle_allow += 1
            else:
                should_train = True

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
            avg_loss = total_loss / max(train_batches, 1)
            extra = ""
            if oracle is not None:
                p = oracle.query_p_training(window_sec=15)
                p_str = f"{p:.3f}" if p is not None else "N/A"
                extra = f", oracle={oracle_allow}/{oracle_queries}, P(train)={p_str}"
            print(f"  {elapsed:.0f}s: train={train_batches}, infer={infer_batches}, "
                  f"{tok_s:.0f} tok/s, mem={mem:.0f}MB, loss={avg_loss:.4f}{extra}")
            last_print = now

    elapsed = time.time() - wall_start
    oracle_str = ""
    if oracle is not None:
        oracle_str = f", oracle allowed {oracle_allow}/{oracle_queries}"
    print(f"  Complete. train={train_batches}, infer={infer_batches}{oracle_str}, "
          f"loss={total_loss / max(train_batches, 1):.4f}, "
          f"{total_tokens / elapsed:.0f} tok/s, {elapsed:.0f}s")
    return train_batches, infer_batches


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


# ── Strategy WB-D: White-Box Diluted Full Fine-Tuning ─────────────────────

def strategy_wb_diluted(args, device):
    """Full fine-tuning disguised as inference via dilution.

    Key stealth techniques:
    1. Pre-allocate optimizer states at startup (no memory growth signal)
    2. Never call model.train() — use torch.enable_grad() manually
    3. Gradient checkpointing bounds activation memory to ~inference level
    4. No torch.cuda.empty_cache() calls (avoids compaction pattern)
    """
    N = args.dilution
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    n_params = sum(p.numel() for p in model.parameters())
    est_train_mem_gb = n_params * 16 / 1e9  # ~16 bytes/param for FP16 AdamW

    print(f"Strategy WB-D: Diluted full fine-tuning (N={N})")
    print(f"  Model: {args.model} ({n_params/1e9:.2f}B params)")
    print(f"  Estimated training memory: {est_train_mem_gb:.1f} GB "
          f"(FP16 weights + FP32 optimizer states)")
    print(f"  Batch: {args.batch_size}, seq: {args.seq_len}, "
          f"training fraction: {1/N:.1%}, grad ckpt: {args.grad_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    model.eval()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    _pre_allocate_optimizer(model, optimizer, tokenizer, args, device)

    if args.grad_ckpt:
        model.gradient_checkpointing_disable()

    _run_diluted_loop(model, optimizer, tokenizer, args, device)


# ── Strategy WB-L: White-Box Diluted LoRA Fine-Tuning ─────────────────────

def strategy_wb_lora(args, device):
    """LoRA fine-tuning disguised as inference via dilution.

    LoRA adds ~0.1-1% trainable parameters, so the optimizer state overhead
    is negligible (~50-200 MB for a 1B model). The memory signature should be
    nearly identical to inference.

    Key differences from WB-D:
    - Optimizer states are tiny (only LoRA params) — much smaller memory gap
    - Backward pass still requires full backprop through frozen layers for
      gradient flow to LoRA injection points
    - Power/compute overhead per training batch is similar to WB-D (backward
      pass is still ~2x forward), but optimizer step is much cheaper
    """
    N = args.dilution
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    # Apply LoRA
    model, n_trainable = apply_lora(
        model, rank=args.lora_rank, alpha=args.lora_rank * 2)

    est_lora_mem_mb = n_trainable * 16 / 1e6  # 16 bytes/param for AdamW on LoRA params
    print(f"Strategy WB-L: Diluted LoRA fine-tuning (N={N})")
    print(f"  LoRA optimizer memory: ~{est_lora_mem_mb:.0f} MB "
          f"(vs {sum(p.numel() for p in model.parameters()) * 2 / 1e6:.0f} MB model weights)")
    print(f"  Batch: {args.batch_size}, seq: {args.seq_len}, "
          f"training fraction: {1/N:.1%}, grad ckpt: {args.grad_ckpt}")

    # Optimizer on trainable (LoRA) params only
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(trainable_params, lr=1e-4, weight_decay=0.01)

    model.eval()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    _pre_allocate_optimizer(model, optimizer, tokenizer, args, device)

    if args.grad_ckpt:
        model.gradient_checkpointing_disable()

    _run_diluted_loop(model, optimizer, tokenizer, args, device)


# ── Strategy WB-O: Oracle-Guided Training ─────────────────────────────────

def strategy_wb_oracle(args, device):
    """White-box adaptive training: query the actual classifier in real time.

    Uses pynvml to read GPU telemetry, computes the same 159 features the
    classifier uses, and only trains when P(training) < threshold.
    """
    import sys, os
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from classifier.classifier_oracle import ClassifierOracle

    N = args.dilution
    model, tokenizer = load_model_and_tokenizer(args.model, device)

    print(f"Strategy WB-O: Oracle-guided full fine-tuning")
    print(f"  Model: {args.model}, batch: {args.batch_size}, seq: {args.seq_len}")
    print(f"  Fallback dilution: N={N}, threshold: {args.oracle_threshold}, "
          f"grad ckpt: {args.grad_ckpt}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    model.eval()
    if args.grad_ckpt:
        model.gradient_checkpointing_enable()

    _pre_allocate_optimizer(model, optimizer, tokenizer, args, device)

    if args.grad_ckpt:
        model.gradient_checkpointing_disable()

    # Initialize the classifier oracle
    print("  Loading classifier oracle...")
    oracle = ClassifierOracle(gpu_index=0)
    oracle.start_polling()
    print(f"  Oracle ready. Will train when P(training) < {args.oracle_threshold}")
    print("  Warming up telemetry buffer (5s)...")
    time.sleep(5)

    _run_diluted_loop(model, optimizer, tokenizer, args, device, oracle=oracle)

    oracle.stop_polling()


# ── Main ─────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Round 5: White-box evasion at relevant model scale",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Model recommendations (fill GPU memory for training):
  RTX 5080  (16 GB):  TinyLlama/TinyLlama-1.1B-Chat-v1.0  --batch-size 1 --seq-len 256 --grad-ckpt
  A100/H100 (80 GB):  Qwen/Qwen2.5-3B                     --batch-size 2 --seq-len 512 --grad-ckpt
  H200      (141 GB): meta-llama/Llama-3.1-8B              --batch-size 2 --seq-len 512 --grad-ckpt
  B200      (192 GB): meta-llama/Llama-3.1-8B              --batch-size 4 --seq-len 1024 --grad-ckpt""",
    )
    parser.add_argument("--strategy", type=str, required=True,
                        choices=["WB-I", "WB-D", "WB-L", "WB-O"],
                        help="WB-I=inference, WB-D=diluted full, WB-L=diluted LoRA, WB-O=oracle")
    parser.add_argument("--model", type=str,
                        default="TinyLlama/TinyLlama-1.1B-Chat-v1.0",
                        help="HuggingFace model (default: TinyLlama 1.1B)")
    parser.add_argument("--duration", type=int, default=300,
                        help="Run duration in seconds (default: 300)")
    parser.add_argument("--batch-size", type=int, default=1,
                        help="Batch size (default: 1)")
    parser.add_argument("--seq-len", type=int, default=256,
                        help="Sequence length (default: 256)")
    parser.add_argument("--device", type=str, default="cuda")
    # Training parameters
    parser.add_argument("--dilution", type=int, default=10,
                        help="Train on 1/N batches (default: 10)")
    parser.add_argument("--grad-ckpt", action="store_true",
                        help="Enable gradient checkpointing (required for large models)")
    # LoRA parameters
    parser.add_argument("--lora-rank", type=int, default=16,
                        help="LoRA rank (default: 16)")
    # Oracle parameters
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
        "WB-L": strategy_wb_lora,
        "WB-O": strategy_wb_oracle,
    }
    strategies[args.strategy](args, device)


if __name__ == "__main__":
    main()
