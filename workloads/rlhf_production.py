#!/usr/bin/env python3
"""Production RLHF-adjacent workloads for the BASELINE corpus.

Round 6 established that the symmetric-GEMM disguise defeats 1 Hz NVML
telemetry, but two of its claims rested on controls that did not exist:

  * `symgemm_ppo_fused` (disguised PPO) had no non-training partner. The only
    PPO-shaped negative available was `symgemm_decode_g256`, which is plain
    decode: not rollout-matched, and missing the scoring phase entirely.
  * `adversarial_N_grad_accum` showed that stretching the optimiser cadence
    hides training -- but only as an *attack*. Nobody had asked whether an
    honest job with the same shape is also missed.

Both cells here are workloads people organically run, which is what makes them
baseline-corpus members rather than probes. Neither is named `symgemm_*`: that
prefix is held out of the clean training pool by classifier/workload_labels.py,
and these belong in the clean pool.

  BON   Best-of-n / rejection sampling against a reward model.
        For each prompt: sample N candidates from the policy, score every
        candidate with a separate reward model, keep the argmax. This is how
        synthetic training data is generated (STaR, RFT) and how inference-time
        scaling is served. NO weight anywhere is ever updated.

        It is the null pair for symgemm_ppo_fused: identical first two phases
        (rollout generation, then a full-sequence scoring forward), missing
        only the gradient step. If the detector separates BON from PPO it is
        reading the update; if it cannot, the disguise has nothing left to do.

  PPOACC  PPO with gradient accumulation across rollout batches.
        R rollouts -> forward/backward into .grad -> repeat A times -> ONE
        optimiser step. Standard practice whenever the effective rollout batch
        exceeds memory, so the optimiser-state traffic lands once per (R*A)
        generations rather than once per R. Honest training wearing the Round 4
        evasion's temporal signature.

Both emit STEP lines so window labels can tell which windows contain a real
optimiser step:

    STEP ts=<epoch> kind=train|infer batch_idx=<n> tokens=<n> passes=<n> phase=<p>

Usage
    python rlhf_production.py --mode BON    --best-of 8  --duration 610
    python rlhf_production.py --mode BON    --best-of 32 --duration 610
    python rlhf_production.py --mode PPOACC --rollout 8  --accum 4 --duration 610
    python rlhf_production.py --mode PPOACC --rollout 32 --accum 8 --duration 610
    python rlhf_production.py --mode BON --cpu-smoke --duration 20
"""
from __future__ import annotations

import argparse
import sys
import time

import torch
import torch.nn as nn

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}
DEFAULT_MODEL = "TinyLlama/TinyLlama-1.1B-Chat-v1.0"


# ── model plumbing ──────────────────────────────────────────────────────────

def load_model_and_tokenizer(model_name, device, dtype):
    """Mirror of adversarial_symgemm.load_model_and_tokenizer.

    Kept byte-compatible in behaviour so a BON run and a symgemm_ppo_fused run
    differ in what they compute, not in how the model was built. A divergence
    here would show up in the telemetry as a program confound and quietly
    invalidate the comparison the cell exists to support.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype)
    return model.to(device), tok


class RewardModel(nn.Module):
    """A scalar-head reward model, initialised from the same base as the policy.

    That is how reward models are actually built, and it keeps the scoring pass
    the same arithmetic shape as a policy prefill -- which is the point: the
    scoring phase must be compute-bound and indistinguishable from the forward
    half of a training step, or BON would be an easy negative for the wrong
    reason.
    """

    def __init__(self, base, hidden_size, dtype):
        super().__init__()
        self.base = base
        self.score = nn.Linear(hidden_size, 1, dtype=dtype)
        for p in self.parameters():
            p.requires_grad_(False)

    @torch.no_grad()
    def forward(self, input_ids):
        out = self.base(input_ids, output_hidden_states=True)
        # Reward is read off the final position, as in a standard preference
        # model: one scalar per candidate sequence.
        return self.score(out.hidden_states[-1][:, -1, :]).squeeze(-1)


def make_prompts(tokenizer, batch_size, prompt_len, device):
    """On-device synthetic prompts.

    Identical to adversarial_symgemm.make_synthetic_batch, for the same reason:
    generating on the GPU holds pcie_rx_mbps at zero across every cell in the
    comparison, so the highest-importance feature in the binary model cannot
    separate these cells as a data-loading artifact.
    """
    return torch.randint(0, tokenizer.vocab_size, (batch_size, prompt_len),
                         device=device)


def freeze_all(model):
    for p in model.parameters():
        p.requires_grad_(False)


def _param_fingerprint(params):
    """Cheap scalar over the trained weights, to prove BON changes nothing."""
    with torch.no_grad():
        return float(sum(p.float().sum() for p in params))


# ── BON: best-of-n / rejection sampling ─────────────────────────────────────

def run_best_of_n(args, policy, reward, tokenizer, device):
    t0 = time.time()
    last = t0
    batch_idx = 0
    n_cycles = 0
    gen_tokens = score_tokens = 0
    best_sum = 0.0

    while time.time() - t0 < args.duration:
        prompt = make_prompts(tokenizer, args.batch_size, args.prompt_len, device)

        # ── sample N candidates per prompt (one batched generate, as served) ──
        with torch.no_grad():
            cand = policy.generate(
                prompt,
                attention_mask=torch.ones_like(prompt),
                max_new_tokens=args.gen_tokens,
                do_sample=True, temperature=0.9, top_p=0.95,
                num_return_sequences=args.best_of,
                pad_token_id=tokenizer.pad_token_id,
            )
        n_new = (cand.shape[1] - prompt.shape[1]) * cand.shape[0]
        gen_tokens += n_new
        print(f"STEP ts={time.time():.3f} kind=infer batch_idx={batch_idx} "
              f"tokens={n_new} passes=1 phase=rollout", flush=True)
        batch_idx += 1

        # ── score every candidate with the reward model ──────────────────────
        # Chunked so n=32 at long prompt_len does not spike memory; a served
        # best-of-n does the same thing for the same reason.
        scores = []
        for i in range(0, cand.shape[0], args.score_chunk):
            scores.append(reward(cand[i:i + args.score_chunk]))
        scores = torch.cat(scores)
        n_scored = cand.shape[0] * cand.shape[1]
        score_tokens += n_scored
        print(f"STEP ts={time.time():.3f} kind=infer batch_idx={batch_idx} "
              f"tokens={n_scored} passes=1 phase=score", flush=True)
        batch_idx += 1

        # ── keep the argmax per prompt; this is the whole "training" there is ─
        per_prompt = scores.view(args.batch_size, args.best_of)
        best_sum += float(per_prompt.max(dim=1).values.float().mean())
        n_cycles += 1

        del cand, prompt, scores, per_prompt

        now = time.time()
        if now - last >= 30:
            el = now - t0
            print(f"  {el:.0f}s: cycles={n_cycles} n={args.best_of}, "
                  f"{gen_tokens / el:.0f} gen-tok/s, "
                  f"{score_tokens / el:.0f} score-tok/s, "
                  f"mem={torch.cuda.memory_allocated() / 1024**2:.0f}MB"
                  if device != "cpu" else f"  {el:.0f}s: cycles={n_cycles}",
                  flush=True)
            last = now

    el = time.time() - t0
    print(f"BON complete. cycles={n_cycles} best-of-{args.best_of}, "
          f"{gen_tokens / el:.0f} gen-tok/s, {score_tokens / el:.0f} score-tok/s, "
          f"{el:.0f}s")
    print(f"  mean selected reward: {best_sum / max(n_cycles, 1):.4f}")
    return n_cycles


# ── PPOACC: PPO with gradient accumulation across rollout batches ───────────

def run_ppo_accum(args, policy, tokenizer, device, optimizer):
    embed = policy.get_input_embeddings()
    t0 = time.time()
    last = t0
    batch_idx = 0
    n_rollout = n_backward = n_update = 0
    gen_tokens = train_tokens = 0
    loss_first = loss_last = None

    while time.time() - t0 < args.duration:
        done = False
        # ── A accumulation micro-cycles, each R rollouts + one backward ──────
        for _ in range(args.accum):
            seq = None
            for _ in range(args.rollout):
                if time.time() - t0 >= args.duration:
                    done = True
                    break
                prompt = make_prompts(tokenizer, args.batch_size,
                                      args.prompt_len, device)
                with torch.no_grad():
                    seq = policy.generate(
                        prompt,
                        attention_mask=torch.ones_like(prompt),
                        max_new_tokens=args.gen_tokens,
                        do_sample=True, temperature=0.7, top_p=0.9,
                        pad_token_id=tokenizer.pad_token_id,
                    )
                n_new = (seq.shape[1] - prompt.shape[1]) * args.batch_size
                gen_tokens += n_new
                print(f"STEP ts={time.time():.3f} kind=infer "
                      f"batch_idx={batch_idx} tokens={n_new} passes=1 "
                      f"phase=rollout", flush=True)
                batch_idx += 1
                n_rollout += 1
            if seq is None:
                break

            inputs_embeds = embed(seq).detach().requires_grad_(True)
            out = policy(inputs_embeds=inputs_embeds, labels=seq)
            # Scale by 1/A so the accumulated gradient equals the mean over the
            # micro-cycles, which is what makes this equivalent to one large
            # rollout batch rather than an A-times-larger learning rate.
            (out.loss / args.accum).backward()
            loss_val = float(out.loss.detach())
            n_tok = seq.shape[0] * seq.shape[1]
            train_tokens += n_tok
            n_backward += 1
            # kind=train but phase=accum: a gradient exists, no weight moved.
            print(f"STEP ts={time.time():.3f} kind=train batch_idx={batch_idx} "
                  f"tokens={n_tok} passes=3 phase=accum loss={loss_val:.4f}",
                  flush=True)
            batch_idx += 1
            if loss_first is None:
                loss_first = loss_val
            loss_last = loss_val
            del out, inputs_embeds, seq, prompt
            if done:
                break

        if n_backward == 0:
            break
        # ── the one optimiser step per (rollout * accum) generations ─────────
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        n_update += 1
        print(f"STEP ts={time.time():.3f} kind=train batch_idx={batch_idx} "
              f"tokens=0 passes=0 phase=update", flush=True)
        batch_idx += 1
        if done:
            break

        now = time.time()
        if now - last >= 30:
            el = now - t0
            print(f"  {el:.0f}s: rollouts={n_rollout} backwards={n_backward} "
                  f"updates={n_update}, {gen_tokens / el:.0f} gen-tok/s, "
                  f"{train_tokens / el:.0f} train-tok/s, loss={loss_last:.4f}",
                  flush=True)
            last = now

    el = time.time() - t0
    print(f"PPOACC complete. rollouts={n_rollout} backwards={n_backward} "
          f"updates={n_update}, {gen_tokens / el:.0f} gen-tok/s, "
          f"{train_tokens / el:.0f} train-tok/s, {el:.0f}s")
    steps = n_rollout + n_backward + n_update
    print(f"  optimiser-step duty cycle: {n_update / max(steps, 1):.2%} of steps "
          f"(rollout:backward:update = {args.rollout}:1:{args.accum} per cycle)")
    if loss_first is not None and loss_last is not None:
        drift = (loss_first - loss_last) / abs(loss_first) * 100.0
        print(f"  loss {loss_first:.4f} -> {loss_last:.4f} ({drift:+.2f}%)")
        if drift < 2.0:
            # Same gate as adversarial_symgemm: a training cell that does not
            # learn is not a training cell, and bf16 update underflow is silent.
            print(f"  WARNING: loss fell only {drift:.2f}% -- verify the "
                  f"update is taking effect (bf16 underflow?) before trusting "
                  f"this run.")
    return n_update


# ── main ────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", required=True, choices=["BON", "PPOACC"])
    ap.add_argument("--duration", type=float, default=610.0)
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--reward-model", default=None,
                    help="BON: base for the scalar-head reward model. Defaults "
                         "to --model, which is how preference models are "
                         "normally initialised.")
    ap.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--prompt-len", type=int, default=128)
    ap.add_argument("--gen-tokens", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    # BON
    ap.add_argument("--best-of", type=int, default=8,
                    help="candidates sampled per prompt before selection")
    ap.add_argument("--score-chunk", type=int, default=8,
                    help="candidates scored per reward-model forward")
    # PPOACC
    ap.add_argument("--rollout", type=int, default=8,
                    help="generations per backward, as in PPO")
    ap.add_argument("--accum", type=int, default=4,
                    help="backwards accumulated per optimiser step")
    ap.add_argument("--cpu-smoke", action="store_true",
                    help="tiny CPU run to verify the loops; telemetry from it "
                         "is NOT representative and must not be collected")
    args = ap.parse_args()

    if args.cpu_smoke:
        args.device = "cpu"
        args.dtype = "float32"
        # Must ship safetensors: transformers 4.57 refuses torch.load on a .bin
        # checkpoint under torch < 2.6 (CVE-2025-32434), and this box is pinned
        # to torch 2.5.1. A Llama also keeps the smoke path architecturally
        # closer to the TinyLlama policy the real cells use.
        if args.model == DEFAULT_MODEL:
            args.model = "hf-internal-testing/tiny-random-LlamaForCausalLM"
        args.batch_size, args.prompt_len, args.gen_tokens = 2, 16, 8
        args.best_of, args.score_chunk, args.rollout, args.accum = 2, 2, 2, 2
    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = args.device
    dtype = DTYPES[args.dtype]

    policy, tok = load_model_and_tokenizer(args.model, device, dtype)
    policy.eval()   # never .train(): avoids dropout/BN mode-switch artifacts

    if args.mode == "BON":
        freeze_all(policy)
        rm_name = args.reward_model or args.model
        base, _ = load_model_and_tokenizer(rm_name, device, dtype)
        base.eval()
        hidden = getattr(base.config, "hidden_size", None) or base.config.n_embd
        reward = RewardModel(base, hidden, dtype).to(device)
        reward.eval()
        print(f"BON: policy={args.model} reward={rm_name} "
              f"best-of-{args.best_of} batch={args.batch_size} "
              f"prompt={args.prompt_len} gen={args.gen_tokens}")
        # Gate: BON must change nothing. Fingerprint before and after; a
        # best-of-n that moved a weight is a training cell mislabelled as a
        # negative, which is the single worst failure this corpus can have.
        watched = [p for p in policy.parameters()][:16]
        before = _param_fingerprint(watched)
        n = run_best_of_n(args, policy, reward, tok, device)
        after = _param_fingerprint(watched)
        print(f"  weight fingerprint {before:.6e} -> {after:.6e}")
        if before != after or n == 0:
            print("  GATE FAILED: BON must complete >0 cycles and change no "
                  "weight; do not collect this run.")
            return 1
        print(f"  GATE OK: {n} cycles, no weight changed, 0 optimiser steps.")
    else:
        # Same trainable set as adversarial_symgemm's P0, so PPOACC and
        # ppo_rlhf differ only in when the optimiser fires.
        for name, p in policy.named_parameters():
            p.requires_grad_(name.endswith("weight")
                             and ("proj" in name or "lm_head" in name))
        trainable = [p for p in policy.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        n_train = sum(p.numel() for p in trainable)
        print(f"PPOACC: AdamW over {n_train / 1e6:.0f}M params "
              f"({n_train * 8 / 1e9:.2f} GB optimizer state), "
              f"rollout={args.rollout} accum={args.accum} "
              f"-> 1 step per {args.rollout * args.accum} generations")
        watched = trainable[:16]
        before = _param_fingerprint(watched)
        n = run_ppo_accum(args, policy, tok, device, optimizer)
        after = _param_fingerprint(watched)
        print(f"  weight fingerprint {before:.6e} -> {after:.6e}")
        if before == after or n == 0:
            print("  GATE FAILED: PPOACC must complete >0 optimiser steps and "
                  "change the weights; do not collect this run.")
            return 1
        print(f"  GATE OK: {n} optimiser steps, weights changed.")

    if device != "cpu":
        print(f"  peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
