#!/usr/bin/env python3
"""Analytic DRAM traffic for the Round 6 symmetric-GEMM ladder.

Answers, without spending a GPU-hour: how much telemetry-visible difference is
there *in principle* between the disguised trainer (L4) and inference matched
to it on FLOPs (L1)?

Accounting is compulsory traffic only -- each GEMM reads its two operands once
and writes its output once. Real tiled GEMMs re-read operands and L2 absorbs
some of that, so every cell here is a lower bound and the measured ratio can
shift. That is exactly why the collection runs exist; this table says what
mechanism to attribute the measurement to.

Units are elements, not bytes, except where an fp32 optimizer state is counted
against a bf16 model -- those entries are scaled by 2 and flagged.

Usage:
    python classifier/rebuttal_analysis/symgemm_traffic.py
    python classifier/rebuttal_analysis/symgemm_traffic.py --config /path/to/config.json
    python classifier/rebuttal_analysis/symgemm_traffic.py --hidden 4096 --inter 11008 \
        --layers 32 --kv-dim 4096 --vocab 32000 --name Llama-2-7b
"""

import argparse
import json
from pathlib import Path

OUT = Path(__file__).resolve().parent / "symgemm_traffic.json"


def linears(hidden, inter, layers, kv_dim, vocab):
    """(name, N_out, K_in) for every trained bias-free linear, nn.Linear order."""
    per_layer = [
        ("q_proj", hidden, hidden),
        ("k_proj", kv_dim, hidden),
        ("v_proj", kv_dim, hidden),
        ("o_proj", hidden, hidden),
        ("gate_proj", inter, hidden),
        ("up_proj", inter, hidden),
        ("down_proj", hidden, inter),
    ]
    out = []
    for i in range(layers):
        out += [(f"layer{i}.{n}", N, K) for n, N, K in per_layer]
    out.append(("lm_head", vocab, hidden))
    return out


def sums(layer_list, M):
    """Sigma-M-K (activations in), Sigma-M-N (activations out), Sigma-K-N (params)."""
    s_mk = sum(M * K for _, _, K in layer_list)
    s_mn = sum(M * N for _, N, _ in layer_list)
    s_kn = sum(N * K for _, N, K in layer_list)
    return s_mk, s_mn, s_kn


def ladder(s_mk, s_mn, s_kn):
    """Compulsory DRAM elements per iteration for each mode.

    fwd  reads X (MK) + W (KN), writes Y (MN)
    dX   reads dY (MN) + W (KN), writes dX (MK)      <- same total as fwd
    dW   reads X (MK) + dY (MN), writes dW (KN)      <- same total as fwd

    The three are equal because {MK, MN, KN} is the same multiset each time.
    That symmetry is the whole reason the disguise works at all.
    """
    one_pass = s_mk + s_mn + s_kn
    p = s_kn

    l0 = one_pass                 # 1 forward
    l1 = 3 * one_pass             # 3 forwards, FLOP-matched to a training step
    l3 = 3 * one_pass             # fwd + dX + dW(discard): identical to L1
    # L4 replaces "write dW" with "read W, write W" via addmm_(beta=1): +P.
    l4 = 3 * one_pass + p
    # L5: materialise dW (already in the 3*one_pass term), then AdamW touches
    # per trained param: read g, read w, read m, read v, write m, write v,
    # write w. m/v/master-w are fp32 against a bf16 model, hence the 2x.
    adamw = p * (1 + 2 + 2 + 2 + 2 + 2 + 2)
    l5 = 3 * one_pass + adamw

    return {"L0": l0, "L1": l1, "L3": l3, "L4": l4, "L5": l5}, one_pass, p


DESC = {
    "L0": ("1 forward pass", "negative, token-matched", "no"),
    "L1": ("3 forward passes", "negative, FLOP-matched", "no"),
    "L3": ("fwd + dX + dW->scratch", "probe: no update", "no"),
    "L4": ("fwd + dX + fused-SGD dW", "THE ATTACK", "yes"),
    "L5": ("fwd + dX + dW + AdamW", "positive control", "yes"),
}
PASSES = {"L0": 1, "L1": 3, "L3": 3, "L4": 3, "L5": 3}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", help="HF config.json to read the architecture from")
    ap.add_argument("--name", default="TinyLlama-1.1B")
    ap.add_argument("--hidden", type=int, default=2048)
    ap.add_argument("--inter", type=int, default=5632)
    ap.add_argument("--layers", type=int, default=22)
    ap.add_argument("--kv-dim", type=int, default=256, help="num_kv_heads * head_dim")
    ap.add_argument("--vocab", type=int, default=32000)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--seq", type=int, default=512)
    args = ap.parse_args()

    if args.config:
        cfg = json.loads(Path(args.config).read_text())
        args.hidden = cfg["hidden_size"]
        args.inter = cfg["intermediate_size"]
        args.layers = cfg["num_hidden_layers"]
        head_dim = cfg.get("head_dim", args.hidden // cfg["num_attention_heads"])
        args.kv_dim = cfg.get("num_key_value_heads",
                              cfg["num_attention_heads"]) * head_dim
        args.vocab = cfg["vocab_size"]
        args.name = Path(args.config).parent.name or args.name

    M = args.batch * args.seq
    ll = linears(args.hidden, args.inter, args.layers, args.kv_dim, args.vocab)
    s_mk, s_mn, s_kn = sums(ll, M)
    cells, one_pass, p = ladder(s_mk, s_mn, s_kn)

    print(f"{args.name}: hidden={args.hidden} inter={args.inter} "
          f"layers={args.layers} kv_dim={args.kv_dim} vocab={args.vocab}")
    print(f"M = batch*seq = {args.batch}*{args.seq} = {M} tokens/iter")
    print(f"  trained params P = Sigma-KN = {p/1e6:,.0f}M")
    print(f"  Sigma-MK = {s_mk/1e6:,.0f}M   Sigma-MN = {s_mn/1e6:,.0f}M")
    print(f"  one GEMM pass = {one_pass/1e6:,.0f}M elements")
    if M < max(args.hidden, args.inter):
        print(f"  WARNING: M={M} < max(K,N)={max(args.hidden, args.inter)}; "
              f"dW is a thin-K GEMM and the disguise degrades.")

    base = cells["L1"]
    print(f"\n{'mode':5s} {'what it does':26s} {'role':24s} {'GEMM':>5s} "
          f"{'DRAM 1e6':>10s} {'vs L1':>8s}  learns")
    print("-" * 96)
    rows = []
    for mode in ("L0", "L1", "L3", "L4", "L5"):
        what, role, learns = DESC[mode]
        v = cells[mode]
        delta = (v - base) / base * 100.0
        print(f"{mode:5s} {what:26s} {role:24s} {PASSES[mode]:4d}x "
              f"{v/1e6:10,.0f} {delta:+7.1f}%  {learns}")
        rows.append({"mode": mode, "what": what, "role": role,
                     "gemm_passes": PASSES[mode], "dram_elements": v,
                     "pct_vs_L1": delta, "learns": learns == "yes"})

    print(f"\nL1 vs L3: {(cells['L3']-cells['L1'])/base*100:+.2f}% "
          f"-- exactly matched (null pair; any measured separation is noise "
          f"or leakage, and calibrates the floor)")
    print(f"L1 vs L4: {(cells['L4']-cells['L1'])/base*100:+.2f}% "
          f"-- the entire telemetry-visible difference the attack leaves behind")

    # How much room does the evader have? Delta = P / (3*(MK+MN+P)), and MK,MN
    # scale with M, so a bigger token batch dilutes the weight write.
    print(f"\nM-sensitivity of the L4-vs-L1 gap (the evader's lever):")
    print(f"  {'M':>7s} {'batch@seq512':>13s} {'gap':>8s}")
    sens = []
    for mm in (512, 1024, 2048, 4096, 8192, 16384):
        mk, mn, kn = sums(ll, mm)
        gap = kn / (3 * (mk + mn + kn)) * 100.0
        print(f"  {mm:7d} {mm//512:13d} {gap:7.2f}%")
        sens.append({"M": mm, "gap_pct": gap})
    print("  Larger M helps the evader twice over: it removes dW's thin-K "
          "shape AND dilutes the weight write.")

    OUT.write_text(json.dumps({
        "model": args.name,
        "arch": {"hidden": args.hidden, "inter": args.inter,
                 "layers": args.layers, "kv_dim": args.kv_dim,
                 "vocab": args.vocab},
        "M": M, "batch": args.batch, "seq": args.seq,
        "sigma_mk": s_mk, "sigma_mn": s_mn, "sigma_kn_params": s_kn,
        "one_pass_elements": one_pass,
        "ladder": rows,
        "m_sensitivity": sens,
    }, indent=2))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
