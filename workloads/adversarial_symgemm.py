#!/usr/bin/env python3
"""Round 6 kernel-level evasion: symmetric-GEMM backprop disguise.

Reviewer M7wU's outstanding objection is a custom kernel that "rearranges the
matmuls involved in backprop to split them into a pattern of compute identical
to two forward passes". This file implements that attack, and the controls
needed to tell whether it actually defeats 1 Hz NVML telemetry.

The mechanism, stated exactly. For a linear layer with M = batch*seq tokens,
K in-features, N out-features (nn.Linear stores W as (N, K) and computes
Y = X W^T):

    FWD  Y  = X   W^T   (M,K)x(K,N) -> (M,N)  activation-sized out
    dX   dX = dY  W     (M,N)x(N,K) -> (M,K)  activation-sized out
    dW   dW = dY^T X    (N,M)x(M,K) -> (N,K)  WEIGHT-sized out

GEMM cost 2mnk and compulsory DRAM traffic mk+kn+mn are both symmetric under
permutation of {M,K,N}. All three therefore have identical FLOPs, identical
arithmetic intensity and identical tensor-pipe occupancy already, in stock
cuBLAS. The disguise does not need a custom kernel; it needs M >= K,N so that
dW is not a thin-K split-K GEMM, and it needs the optimizer to stop being a
separate memory-bound kernel.

What is irreducible: learning requires every trained weight to be read AND
WRITTEN once per step. Inference only ever reads weights. Mode L4 removes
everything else -- the gradient tensor, the optimizer kernel, the optimizer
state -- by fusing the update into the dW epilogue via a single addmm_, so the
only remaining difference from matched-throughput inference is that one
P-element weight write.

Modes (the ladder; each removes or adds exactly one mechanism):

  L0  1 forward pass/iter                  negative, token-matched
  L1  3 forward passes/iter                negative, FLOP/throughput-matched
  L3  fwd + dX + dW -> scratch, no update  probe: same FLOPs AND same traffic
                                           as L1, but no learning. Null pair.
  L4  fwd + dX + fused-SGD dW              THE ATTACK. Learns.
  L5  fwd + dX + dW -> grad + AdamW        positive control. What people run.

L1 vs L4 is the decisive comparison: same model, same shapes, same program,
same GEMM issue rate, differing only by the weight write.

Usage:
    python adversarial_symgemm.py --self-test
    python adversarial_symgemm.py --mode L4 --model TinyLlama/TinyLlama-1.1B-Chat-v1.0 \
        --batch-size 4 --seq-len 512 --dtype bfloat16 --lr 1e-3 --duration 600
"""

import argparse
import sys
import time

import torch
import torch.nn as nn


# ── The disguised linear ────────────────────────────────────────────────────

_TRITON = None


def _triton():
    """Lazy import: the CPU self-test and the torch backend must not need it.

    This file runs as a standalone script (the collection queue invokes it as
    `python workloads/adversarial_symgemm.py`), so it cannot rely on being
    imported as part of a package.
    """
    global _TRITON
    if _TRITON is None:
        import os
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        import symgemm_triton
        _TRITON = symgemm_triton
    return _TRITON


class SymLinearFn(torch.autograd.Function):
    """Linear whose backward issues dX and dW as forward-shaped GEMMs.

    Three traps, all load-bearing:

    1. ``w`` is stashed on ``ctx`` rather than via ``save_for_backward``. L4
       mutates it in place; a saved tensor would trip autograd's version
       counter and raise on the next backward.
    2. ``dx`` must be computed BEFORE the in-place update, or dX is taken
       against the already-updated weight and the loss will not fall.
    3. The weights carry ``requires_grad=False`` in L3/L4 (the update is
       manual), so the caller must root the graph elsewhere -- see
       ``forward_pass``, which passes ``inputs_embeds`` with requires_grad set.
    """

    @staticmethod
    def forward(ctx, x, w, mode, lr, scratch, backend):
        ctx.w, ctx.mode, ctx.lr = w, mode, lr
        ctx.scratch, ctx.backend = scratch, backend
        ctx.save_for_backward(x)
        if backend == "triton":
            x2 = x.reshape(-1, x.shape[-1])
            y = _triton().gemm(x2, w.t())
            return y.view(*x.shape[:-1], w.shape[0])
        return torch.matmul(x, w.t())

    @staticmethod
    def backward(ctx, g):
        (x,) = ctx.saved_tensors
        w = ctx.w
        g2 = g.reshape(-1, g.shape[-1])
        x2 = x.reshape(-1, x.shape[-1])

        if ctx.backend == "triton":
            tri = _triton()
            # dX first: the fused update below mutates w in place.
            dx = tri.gemm(g2, w).view(*g.shape[:-1], w.shape[1])
            if ctx.mode == "L3":
                tri.gemm(g2.t(), x2, out=ctx.scratch)
            elif ctx.mode == "L4":
                # The optimizer step happens inside the GEMM epilogue.
                tri.gemm(g2.t(), x2, out=w, alpha=-ctx.lr, accumulate=True)
            return dx, None, None, None, None, None

        # (M,N)x(N,K) -> (M,K). Same shape class as a forward GEMM. Must be
        # computed BEFORE the update, or dX is taken against the new weight.
        dx = torch.matmul(g, w)

        # (N,M)x(M,K) -> (N,K). Weight-sized output; this is the one GEMM
        # whose write traffic inference has no counterpart for.
        if ctx.mode == "L3":
            # Same FLOPs and same compulsory traffic as L4, but the result is
            # thrown away, so no learning occurs. Isolates "does the detector
            # see the dW GEMM" from "does it see the weight update".
            torch.matmul(g2.t(), x2, out=ctx.scratch)
        elif ctx.mode == "L4":
            # W <- W - lr * (dY^T X), fused into the GEMM epilogue. No
            # gradient tensor is ever materialised, no optimizer kernel is
            # ever launched, no optimizer state exists.
            w.addmm_(g2.t(), x2, beta=1.0, alpha=-ctx.lr)

        return dx, None, None, None, None, None


class SymLinear(nn.Module):
    """Drop-in for a bias-free nn.Linear that routes through SymLinearFn."""

    def __init__(self, linear: nn.Linear, mode: str, lr: float,
                 backend: str = "torch"):
        super().__init__()
        if linear.bias is not None:
            raise ValueError("SymLinear expects bias-free layers (Llama-style)")
        self.weight = linear.weight
        self.mode = mode
        self.lr = lr
        self.backend = backend
        # L3 writes dW into a persistent buffer so the allocator is not
        # exercised per step -- otherwise the probe differs from L4 in
        # allocation behaviour as well as in the update itself.
        self.scratch = (
            torch.empty_like(self.weight) if mode == "L3" else None
        )

    def forward(self, x):
        return SymLinearFn.apply(x, self.weight, self.mode, self.lr,
                                 self.scratch, self.backend)


# ── Model surgery ───────────────────────────────────────────────────────────

# Every bias-free projection in a Llama-style block, plus the output head.
TARGET_SUFFIXES = (
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj",
)


def swap_linears(model, mode, lr, include_lm_head=True, backend="torch"):
    """Replace targeted nn.Linear modules with SymLinear. Returns the count.

    Tied embeddings are excluded from the lm_head swap: L4 mutates the weight
    in place, so updating a tied lm_head would silently rewrite the input
    embedding table as well. TinyLlama and Llama-2 both set
    tie_word_embeddings=false, but a model that ties them would corrupt
    quietly rather than fail, so it is checked rather than assumed.
    """
    cfg = getattr(model, "config", None)
    tied = bool(getattr(cfg, "tie_word_embeddings", False)) if cfg else False
    if tied and include_lm_head:
        print("  NOTE: tie_word_embeddings=True -- excluding lm_head from the "
              "swap so the in-place update cannot rewrite the embedding table")
        include_lm_head = False

    # Collect first, then mutate: replacing a child mid-traversal of
    # named_modules() can make the walk visit or skip the wrong nodes.
    targets = []
    for name, module in model.named_modules():
        for child_name, child in module.named_children():
            if not isinstance(child, nn.Linear):
                continue
            full = f"{name}.{child_name}" if name else child_name
            is_target = full.endswith(TARGET_SUFFIXES) or (
                include_lm_head and full.endswith("lm_head"))
            if is_target and child.bias is None:
                targets.append((module, child_name, child))

    n_params = 0
    for module, child_name, child in targets:
        setattr(module, child_name, SymLinear(child, mode, lr, backend))
        n_params += child.weight.numel()
    return len(targets), n_params


def freeze_all(model):
    """No parameter carries a grad: L3/L4 update weights manually, and the
    graph is rooted at inputs_embeds instead. Also guarantees that no stray
    optimizer state can be allocated behind our backs."""
    for p in model.parameters():
        p.requires_grad_(False)


# ── Workload plumbing (mirrors adversarial_whitebox.py) ─────────────────────

def load_model_and_tokenizer(model_name, device, dtype):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(f"Loading {model_name} in {dtype}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        model_name, torch_dtype=dtype, device_map=device,
    )
    n_params = sum(p.numel() for p in model.parameters())
    print(f"  {n_params / 1e9:.2f}B params, "
          f"{torch.cuda.memory_allocated() / 1024**2:.0f} MB allocated")
    return model, tokenizer


def make_synthetic_batch(tokenizer, batch_size, seq_len, device):
    """On-device synthetic tokens.

    Deliberately identical to adversarial_whitebox.py:97. Generating batches on
    the GPU means PCIe traffic is zero for every cell in this experiment, which
    neutralises pcie_rx_mbps -- the single highest-importance feature in the
    binary model, and a data-loading artifact rather than training physics.
    """
    input_ids = torch.randint(
        0, tokenizer.vocab_size, (batch_size, seq_len), device=device)
    return input_ids


def forward_pass(model, embed, input_ids, want_grad):
    """One forward. When want_grad, the graph is rooted at inputs_embeds
    because every weight has requires_grad=False (see freeze_all)."""
    inputs_embeds = embed(input_ids)
    if want_grad:
        inputs_embeds = inputs_embeds.detach().requires_grad_(True)
        return model(inputs_embeds=inputs_embeds, labels=input_ids)
    with torch.no_grad():
        return model(inputs_embeds=inputs_embeds, labels=input_ids)


def run_loop(args, model, tokenizer, device, optimizer=None):
    """Duration-bounded loop shared by every mode."""
    mode = args.mode
    is_train = mode in ("L4", "L5")
    kind = "train" if is_train else "infer"
    # GEMM passes per iteration, for the matched-throughput accounting.
    passes = {"L0": 1, "L1": 3, "L3": 3, "L4": 3, "L5": 3}[mode]
    n_fwd = 3 if mode == "L1" else 1
    embed = model.get_input_embeddings()

    wall_start = time.time()
    last_print = wall_start
    total_tokens = 0
    batch_idx = 0
    loss_first = None
    loss_last = None
    loss_sum = 0.0
    loss_n = 0

    while time.time() - wall_start < args.duration:
        input_ids = make_synthetic_batch(
            tokenizer, args.batch_size, args.seq_len, device)

        if mode in ("L0", "L1"):
            for _ in range(n_fwd):
                out = forward_pass(model, embed, input_ids, want_grad=False)
            loss_val = float(out.loss)
        else:
            out = forward_pass(model, embed, input_ids, want_grad=True)
            loss = out.loss
            loss.backward()
            if mode == "L5":
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
            loss_val = float(loss.detach())
            del loss

        loss_sum += loss_val
        loss_n += 1
        if loss_first is None:
            loss_first = loss_val
        loss_last = loss_val

        tokens = args.batch_size * args.seq_len
        total_tokens += tokens
        # Shares the time.time() epoch with the 1 Hz collector, so steps bin
        # directly into telemetry windows. Prefix format matches
        # adversarial_whitebox.py so the existing re-scorer parses it.
        print(f"STEP ts={time.time():.3f} kind={kind} batch_idx={batch_idx} "
              f"tokens={tokens} passes={passes} loss={loss_val:.4f}", flush=True)
        batch_idx += 1
        del out, input_ids

        now = time.time()
        if now - last_print >= 30:
            elapsed = now - wall_start
            print(f"  {elapsed:.0f}s: {batch_idx} iters, "
                  f"{total_tokens / elapsed:.0f} tok/s, "
                  f"{total_tokens * passes / elapsed:.0f} gemm-tok/s, "
                  f"mem={torch.cuda.memory_allocated() / 1024**2:.0f}MB, "
                  f"loss={loss_sum / max(loss_n, 1):.4f}", flush=True)
            last_print = now
            loss_sum, loss_n = 0.0, 0

    elapsed = time.time() - wall_start
    drift = ((loss_first - loss_last) / loss_first * 100.0
             if loss_first else 0.0)
    print(f"{mode} complete. {batch_idx} iters, "
          f"{total_tokens / elapsed:.0f} tok/s, "
          f"{total_tokens * passes / elapsed:.0f} gemm-tok/s, {elapsed:.0f}s")
    print(f"  loss {loss_first:.4f} -> {loss_last:.4f} ({drift:+.2f}%)")
    if is_train and drift < 2.0:
        # Gate G2: a disguised trainer that does not actually learn is not an
        # attack. bf16 update underflow is the likely cause -- lr*grad below
        # the ulp of w. Surfaced loudly rather than silently voiding the run.
        print(f"  WARNING: loss fell only {drift:.2f}% -- verify the update is "
              f"taking effect (bf16 underflow?) before trusting this run.")


# ── Generation-based cells: PPO shape and pure decode ───────────────────────

def run_generate_loop(args, model, tokenizer, device, optimizer=None):
    """Autoregressive-generation cells, for Area Chair 1uob's question.

    The meta-review asks why detection should generalise across regimes when
    PPO differs from vanilla dense SGD and prefill and decode
    differ from each other. Three cells answer it directly:

    DC  pure autoregressive decode. Memory-bandwidth-bound, M = batch per step,
        KV-cache streaming. The contrasting negative to L1's compute-bound
        forward pass, collected inside the same program so the comparison
        carries no program confound.

    P0  *Undisguised* PPO: R rollout generations, then one ordinary
        autograd + AdamW policy update. This is what a real RLHF job runs, so
        it belongs in the BASELINE corpus as legitimate training -- it is a
        training-regime-coverage cell, not an evasion. Its update materialises
        the gradient tensor and touches ~16 bytes/param of optimizer state, so
        it carries the full ordinary-training DRAM signature at whatever duty
        cycle R implies.

    P1  DISGUISED PPO: identical rollout/update structure, but the policy update
        goes through the fused SymLinearFn epilogue (as L4), so no gradient
        tensor is materialised and no optimizer state exists. This stacks two
        evasions -- temporal dilution (rollouts) and write minimisation
        (fusion) -- and is therefore a Round 6 evasion cell, not a baseline one.

    In both P0 and P1 every trained weight still changes once per update, so the
    irreducible weight-write invariant is present; they differ in how much
    additional traffic the update carries with it.
    """
    embed = model.get_input_embeddings()
    is_ppo = args.mode in ("P0", "P1")
    R = args.rollout if is_ppo else 1

    wall_start = time.time()
    last_print = wall_start
    gen_tokens_total = 0
    train_tokens_total = 0
    batch_idx = 0
    n_rollout = 0
    n_update = 0
    loss_first = loss_last = None
    seq = prompt = None

    while time.time() - wall_start < args.duration:
        seq = None
        # ── rollout: R autoregressive generations, no weights change ────────
        for _ in range(R):
            if time.time() - wall_start >= args.duration:
                break
            prompt = make_synthetic_batch(
                tokenizer, args.batch_size, args.prompt_len, device)
            with torch.no_grad():
                seq = model.generate(
                    prompt,
                    attention_mask=torch.ones_like(prompt),
                    max_new_tokens=args.gen_tokens,
                    do_sample=True, temperature=0.7, top_p=0.9,
                    pad_token_id=tokenizer.pad_token_id,
                )
            n_new = (seq.shape[1] - prompt.shape[1]) * args.batch_size
            gen_tokens_total += n_new
            print(f"STEP ts={time.time():.3f} kind=infer batch_idx={batch_idx} "
                  f"tokens={n_new} passes=1 phase=rollout", flush=True)
            batch_idx += 1
            n_rollout += 1

        # The rollout loop breaks on the duration check, so on the final cycle
        # it can exit before generating anything. Without this guard `seq` is
        # undefined and the run dies at the very end, after the telemetry that
        # matters has already been collected -- the worst time to fail.
        if seq is None:
            break
        if not is_ppo:
            del seq, prompt
            seq = None
            continue

        # ── policy update: one gradient step on the sampled trajectory ──────
        # P1 (disguised): the update fires inside SymLinearFn's fused epilogue,
        # so no gradient tensor and no optimizer state ever exist.
        # P0 (undisguised): ordinary autograd populates .grad and AdamW steps,
        # which is what a real RLHF job does.
        inputs_embeds = embed(seq).detach().requires_grad_(True)
        out = model(inputs_embeds=inputs_embeds, labels=seq)
        out.loss.backward()
        if optimizer is not None:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        loss_val = float(out.loss.detach())
        n_tok = seq.shape[0] * seq.shape[1]
        train_tokens_total += n_tok
        print(f"STEP ts={time.time():.3f} kind=train batch_idx={batch_idx} "
              f"tokens={n_tok} passes=3 phase=update loss={loss_val:.4f}",
              flush=True)
        batch_idx += 1
        n_update += 1
        if loss_first is None:
            loss_first = loss_val
        loss_last = loss_val
        del out, inputs_embeds, seq, prompt
        seq = None

        now = time.time()
        if now - last_print >= 30:
            el = now - wall_start
            print(f"  {el:.0f}s: rollouts={n_rollout} updates={n_update}, "
                  f"{gen_tokens_total / el:.0f} gen-tok/s, "
                  f"{train_tokens_total / el:.0f} train-tok/s, "
                  f"mem={torch.cuda.memory_allocated() / 1024**2:.0f}MB, "
                  f"loss={loss_val:.4f}", flush=True)
            last_print = now

    el = time.time() - wall_start
    print(f"{args.mode} complete. rollouts={n_rollout} updates={n_update}, "
          f"{gen_tokens_total / el:.0f} gen-tok/s, "
          f"{train_tokens_total / el:.0f} train-tok/s, {el:.0f}s")
    if is_ppo:
        duty = n_update / max(n_rollout + n_update, 1)
        print(f"  training duty cycle: {duty:.1%} of steps "
              f"(rollout:update = {R}:1)")
        if loss_first is not None:
            drift = (loss_first - loss_last) / loss_first * 100.0
            print(f"  loss {loss_first:.4f} -> {loss_last:.4f} ({drift:+.2f}%)")
            if drift < 2.0:
                print(f"  WARNING: loss fell only {drift:.2f}% -- verify the "
                      f"policy update is taking effect before trusting this run.")


# ── Self-test ───────────────────────────────────────────────────────────────

def self_test(device):
    """Assert SymLinearFn matches nn.Linear autograd before spending GPU time."""
    torch.manual_seed(0)
    M, K, N, lr = 512, 256, 384, 1e-3
    dtype = torch.float32  # exact comparison; bf16 is checked separately

    x = torch.randn(M, K, device=device, dtype=dtype)
    w0 = torch.randn(N, K, device=device, dtype=dtype) * 0.02

    # Reference: ordinary autograd.
    xr = x.clone().requires_grad_(True)
    wr = w0.clone().requires_grad_(True)
    yr = torch.matmul(xr, wr.t())
    gr = torch.randn_like(yr)
    yr.backward(gr)

    # L3: dX must match; weights must be untouched.
    w3 = w0.clone()
    scratch = torch.empty_like(w3)
    x3 = x.clone().requires_grad_(True)
    y3 = SymLinearFn.apply(x3, w3, "L3", lr, scratch, "torch")
    y3.backward(gr)
    assert torch.allclose(y3, yr, atol=1e-4), "L3 forward mismatch"
    dx_err = (x3.grad - xr.grad).norm() / xr.grad.norm()
    assert dx_err < 1e-3, f"L3 dX rel err {dx_err:.2e}"
    assert torch.equal(w3, w0), "L3 must not modify weights"
    dw_err = (scratch - wr.grad).norm() / wr.grad.norm()
    assert dw_err < 1e-3, f"L3 dW rel err {dw_err:.2e}"

    # L4: dX must match, and the weight must land exactly on the SGD update.
    w4 = w0.clone()
    x4 = x.clone().requires_grad_(True)
    y4 = SymLinearFn.apply(x4, w4, "L4", lr, None, "torch")
    y4.backward(gr)
    dx_err4 = (x4.grad - xr.grad).norm() / xr.grad.norm()
    assert dx_err4 < 1e-3, f"L4 dX rel err {dx_err4:.2e}"
    expected = w0 - lr * wr.grad
    upd_err = (w4 - expected).norm() / expected.norm()
    assert upd_err < 1e-5, f"L4 update rel err {upd_err:.2e}"

    print(f"  L3 dX rel err {dx_err:.2e}, dW rel err {dw_err:.2e}")
    print(f"  L4 dX rel err {dx_err4:.2e}, update rel err {upd_err:.2e}")
    print("  L4 weight delta matches -lr * (dY^T X) exactly (fused addmm_)")

    # Gate G2 on CPU: a disguised trainer whose loss does not fall is a bug,
    # not an attack. Exercises the real path -- module surgery, frozen
    # parameters, graph rooted at the input, updates applied only inside
    # SymLinearFn.backward -- on a stack shaped like a transformer block.
    class Toy(nn.Module):
        def __init__(self, d=256, h=512):
            super().__init__()
            self.q_proj = nn.Linear(d, d, bias=False)
            self.o_proj = nn.Linear(d, d, bias=False)
            self.gate_proj = nn.Linear(d, h, bias=False)
            self.down_proj = nn.Linear(h, d, bias=False)

        def forward(self, x):
            x = self.o_proj(torch.relu(self.q_proj(x)))
            return self.down_proj(torch.relu(self.gate_proj(x)))

    torch.manual_seed(1)
    toy = Toy().to(device)
    target = torch.randn(64, 256, device=device)
    inp = torch.randn(64, 256, device=device)

    freeze_all(toy)
    # lr is large because the toy loss is a mean over 16K elements; it tunes
    # the toy's optimization, not the attack. Correctness is carried by the
    # exact-update assertion above and the weight-delta check below.
    n_sw, n_p = swap_linears(toy, "L4", lr=0.5, include_lm_head=False)
    assert n_sw == 4, f"expected 4 swaps, got {n_sw}"
    w0 = toy.q_proj.weight.clone()

    losses = []
    for _ in range(200):
        x = inp.detach().requires_grad_(True)
        loss = ((toy(x) - target) ** 2).mean()
        loss.backward()          # the update happens inside backward
        losses.append(float(loss.detach()))
    drop = (losses[0] - losses[-1]) / losses[0] * 100.0
    dw = (toy.q_proj.weight - w0).abs().max().item()

    assert all(not p.requires_grad for p in toy.parameters()), \
        "no parameter should carry a grad; the update is manual"
    assert all(p.grad is None for p in toy.parameters()), \
        "no .grad should be populated -- that would mean a second, unfused path"
    assert dw > 0, "weights never changed: the fused update did not fire"
    assert drop > 40.0, f"L4 did not learn: loss fell only {drop:.1f}%"

    print(f"  L4 end-to-end: {n_sw} layers swapped ({n_p/1e3:.0f}K params), "
          f"loss {losses[0]:.4f} -> {losses[-1]:.4f} ({drop:.1f}% drop), "
          f"max|dW|={dw:.2e}")
    print("    ...with zero optimizer state and no .grad tensor anywhere")
    print("self-test PASS")
    return 0


# ── Main ────────────────────────────────────────────────────────────────────

DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16,
          "float32": torch.float32}


def main():
    ap = argparse.ArgumentParser(
        description="Round 6: symmetric-GEMM backprop disguise",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""Modes:
  L0  1 forward/iter                       negative, token-matched
  L1  3 forwards/iter                      negative, FLOP-matched  <- key control
  L3  fwd + dX + dW->scratch, no update    probe (same traffic as L1, no learning)
  L4  fwd + dX + fused-SGD dW              THE ATTACK
  L5  fwd + dX + dW->grad + AdamW          positive control
  DC  autoregressive generation only       negative, memory-bandwidth-bound
  P0  R rollouts + AdamW policy update     UNDISGUISED PPO -> baseline corpus
  P1  R rollouts + fused-SGD update        DISGUISED PPO -> Round 6 evasion""")
    ap.add_argument("--mode",
                    choices=["L0", "L1", "L3", "L4", "L5", "P0", "P1", "DC"])
    ap.add_argument("--model", default="TinyLlama/TinyLlama-1.1B-Chat-v1.0")
    ap.add_argument("--duration", type=int, default=600)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--seq-len", type=int, default=512)
    ap.add_argument("--dtype", choices=list(DTYPES), default="bfloat16")
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--backend", choices=["torch", "triton"], default="torch",
                    help="GEMM backend. 'triton' routes all three GEMMs of a "
                         "step through one kernel at a fixed tile/grid config, "
                         "equalising kernel identity and launch count. Neither "
                         "is observable in the 9 NVML signals at 1 Hz, so this "
                         "is a completeness exercise -- and it needs its own "
                         "matched control (L1 --backend triton) because Triton "
                         "is slower than cuBLAS.")
    # Generation cells (P1 / DC)
    ap.add_argument("--rollout", type=int, default=8,
                    help="P1: autoregressive rollouts per policy update. Sets "
                         "the training duty cycle, as the rollout:update ratio "
                         "does in real PPO.")
    ap.add_argument("--prompt-len", type=int, default=128,
                    help="P1/DC: prompt length fed to model.generate()")
    ap.add_argument("--gen-tokens", type=int, default=64,
                    help="P1/DC: new tokens generated per rollout")
    ap.add_argument("--self-test", action="store_true",
                    help="Verify SymLinearFn against nn.Linear autograd, then exit")
    args = ap.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        print("WARNING: CUDA not available, falling back to CPU")
        args.device = "cpu"
    device = args.device

    if args.self_test:
        return self_test(device)
    if args.mode is None:
        ap.error("--mode is required unless --self-test is given")

    dtype = DTYPES[args.dtype]
    is_gen = args.mode in ("P0", "P1", "DC")
    if is_gen:
        print(f"Mode {args.mode}: prompt={args.prompt_len} "
              f"gen={args.gen_tokens} batch={args.batch_size}"
              + (f" rollout:update = {args.rollout}:1" if args.mode == "P1" else ""))
    else:
        M = args.batch_size * args.seq_len
        print(f"Mode {args.mode}: M = batch*seq = {M} tokens/iter")
        if M < 2048:
            print(f"  NOTE: M={M} is small; dW becomes a thin-K GEMM and the "
                  f"disguise degrades. The attack wants M >= max(K,N).")

    model, tokenizer = load_model_and_tokenizer(args.model, device, dtype)
    if args.grad_ckpt:
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False})
    model.eval()  # never call .train(): avoids dropout/BN mode-switch artifacts

    optimizer = None
    if args.mode in ("L5", "P0"):
        # Ordinary full-parameter AdamW. L5 is the dense-training positive
        # control; P0 is *undisguised* PPO -- a legitimate RL training regime
        # for the baseline corpus, not an evasion. Trainable params are the same
        # set the attack updates, so the two are directly comparable.
        for name, p in model.named_parameters():
            p.requires_grad_(name.endswith("weight")
                             and ("proj" in name or "lm_head" in name))
        trainable = [p for p in model.parameters() if p.requires_grad]
        optimizer = torch.optim.AdamW(trainable, lr=args.lr)
        n_train = sum(p.numel() for p in trainable)
        print(f"  {args.mode}: AdamW over {n_train / 1e6:.0f}M params "
              f"({n_train * 8 / 1e9:.2f} GB of optimizer state)")
    elif args.mode in ("L3", "L4", "P1"):
        # P1's policy update is the same fused-SGD epilogue as L4 -- the point
        # is that PPO's update is structurally an ordinary gradient step.
        freeze_all(model)
        swap_mode = "L4" if args.mode == "P1" else args.mode
        n_swapped, n_params = swap_linears(model, swap_mode, args.lr,
                                           backend=args.backend)
        print(f"  {args.mode}: swapped {n_swapped} linears "
              f"({n_params / 1e6:.0f}M params updated in the dW epilogue, "
              f"0 bytes of optimizer state)")
    else:
        freeze_all(model)

    torch.cuda.reset_peak_memory_stats()
    if is_gen:
        run_generate_loop(args, model, tokenizer, device, optimizer)
    else:
        run_loop(args, model, tokenizer, device, optimizer)
    print(f"  peak memory: {torch.cuda.max_memory_allocated() / 1024**2:.0f} MB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
