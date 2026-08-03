#!/usr/bin/env python3
"""Round 6 evaluation: does the symmetric-GEMM disguise defeat 1 Hz NVML?

Protocol. Every ``symgemm_*`` run is held out of training, without exception.
This is why the analysis does NOT go through
``classifier/leave_one_strategy_out.py``: that script's ``clean_mask`` is
"not in the adversarial list", which would sweep the matched negatives
(L0/L1/L3) into the *training* set. The classifier would then have been shown
this exact program at these exact settings labelled not-training, and the
attack becomes trivially separable for a reason that has nothing to do with
detecting training. Its ``test_mask`` also mixes positives and negatives, so
its ``det_rate`` is meaningless for a ladder like this one.

Primary metric is the SAME-SCALE pair, L4 vs L1: identical model, shapes,
program and GEMM issue rate, differing only by the weight read-modify-write.
Scoring against the full benign corpus is reported too, clearly marked as
secondary -- ``classifier/rebuttal_analysis/README.md`` note 4 already records
that the full pool is dominated by small workloads and inflates per-family AUC
from ~0.65 to ~0.999.

Comparisons run:
  L5 vs L1   positive control -- if this is not caught, the pipeline is broken
  L4 vs L1   THE RESULT
  L3 vs L1   null pair (identical compulsory traffic) -- calibrates the floor
  L0 vs L1   token-matched vs FLOP-matched inference
  and the same at M=8192 and at 7B.

Each comparison is scored twice: on the paper's feature set, and with four
DRAM-activity ratios added. Those ratios are derived here from window
statistics that already exist, so the corpus and the published numbers are
untouched -- if they help, they earn their way into
``extract_enhanced_features`` for the camera-ready.

Usage:
    python classifier/rebuttal_analysis/symgemm_eval.py
    python classifier/rebuttal_analysis/symgemm_eval.py --tau 0.75 --k 3 --m 5
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from classifier.threeway_improved import (  # noqa: E402
    load_all_data, sliding_windows, get_feature_cols,
)

OUT = Path(__file__).resolve().parent / "symgemm_eval.json"


# ── The monitor-side candidate features ─────────────────────────────────────

# The four DRAM-activity ratios. As of the Round-6 monitor response these are
# computed inside extract_enhanced_features (threeway_improved.py), so they are
# already in get_feature_cols(win). The ablation below drops them for the
# "paper baseline" arm and keeps them for the "+DRAM" arm.
DRAM_RATIO_COLS = ["mem_util_per_gpu_util", "mem_util_minus_gpu_util",
                   "mem_util_per_gpu_util_p95", "power_per_mem_util",
                   # peak/conditional weight-write features (PPO response)
                   "mem_util_at_high_util", "mem_util_spike",
                   "mem_util_burst_ratio", "mem_util_fft_peak_power"]


def add_dram_ratios(win, feats):
    """Four cross-signal ratios the paper's feature set does not contain.

    Retained for callers that window with an older extractor; when the features
    already exist (the current default) this is a no-op that returns the union.

    ``extract_enhanced_features`` defines exactly three cross-signal features
    (power_per_util, pcie_total_mean, util_per_sm_pct) and none of them
    involve mem_utilization_pct -- NVML's fraction of the sample period during
    which device memory was read or written. That is the only channel in which
    a weight write can show up, and it is the one thing a learner cannot skip.

    A tree model already has all 13 marginal statistics of both signals, so a
    ratio only buys something when the discriminative direction is diagonal --
    when both signals move together and their ratio is the invariant. That is
    exactly the disguised-training case. It is an empirical claim, which is
    why it is tested rather than asserted.
    """
    eps = 1.0
    if all(c in win.columns for c in DRAM_RATIO_COLS):
        # Already produced by the extractor — just make sure they're in the list.
        return feats + [c for c in DRAM_RATIO_COLS if c not in feats]
    mu, gu = win["mem_utilization_pct_mean"], win["gpu_utilization_pct_mean"]
    win["mem_util_per_gpu_util"] = mu / (gu + eps)
    win["mem_util_minus_gpu_util"] = mu - gu
    win["mem_util_per_gpu_util_p95"] = (
        win["mem_utilization_pct_p95"] / (win["gpu_utilization_pct_p95"] + eps))
    win["power_per_mem_util"] = win["power_draw_w_mean"] / (mu + eps)
    return feats + list(DRAM_RATIO_COLS)


# ── Metrics ─────────────────────────────────────────────────────────────────

def kofm(flags, k, m):
    """Index of the first window where k of the last m windows alert."""
    for i in range(len(flags)):
        lo = max(0, i - m + 1)
        if flags[lo:i + 1].sum() >= k:
            return i
    return None


def run_level(win_sub, p, tau, k, m):
    """Per-run alert decision under the deployed k-of-m rule."""
    df = pd.DataFrame({
        "run_id": win_sub["run_id"].values,
        "t": win_sub["window_start"].values,
        "p": p,
    }).sort_values(["run_id", "t"])
    n_alert, ttfa = 0, []
    runs = 0
    for _, g in df.groupby("run_id"):
        runs += 1
        flags = (g["p"].values > tau).astype(int)
        i = kofm(flags, k, m)
        if i is not None:
            n_alert += 1
            ttfa.append(g["t"].values[i] - g["t"].values[0])
    return {
        "n_runs": runs,
        "runs_alerting": n_alert,
        "run_alert_rate": n_alert / runs if runs else float("nan"),
        "median_ttfa_s": float(np.median(ttfa)) if ttfa else None,
    }


def score(win, feats, train_mask, pos_mask, neg_mask, tau, k, m, seed=42):
    tr = win[train_mask]
    clf = RandomForestClassifier(
        n_estimators=300, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=seed, n_jobs=-1)
    clf.fit(tr[feats].fillna(0).values.astype(np.float32),
            (tr["threeway_label"].values == "ml_training").astype(int))
    i1 = list(clf.classes_).index(1)

    def pr(mask):
        return clf.predict_proba(
            win[mask][feats].fillna(0).values.astype(np.float32))[:, i1]

    p, n = pr(pos_mask), pr(neg_mask)
    y = np.r_[np.ones(len(p)), np.zeros(len(n))]
    s = np.r_[p, n]
    fpr, tpr, _ = roc_curve(y, s)
    return {
        "n_train_windows": int(train_mask.sum()),
        "n_pos": len(p), "n_neg": len(n),
        "auc": float(roc_auc_score(y, s)),
        "ap": float(average_precision_score(y, s)),
        "tpr_at_1pct_fpr": float(np.interp(0.01, fpr, tpr)),
        "p_train_pos": float(p.mean()), "p_train_neg": float(n.mean()),
        "window_alert_pos": float((p > tau).mean()),
        "window_alert_neg": float((n > tau).mean()),
        "run_pos": run_level(win[pos_mask], p, tau, k, m),
        "run_neg": run_level(win[neg_mask], n, tau, k, m),
    }


CELLS = [
    # (label, positive cell, negative cell, note)
    ("L5 vs L1  (positive control)", "L5_adam", "L1_fwd3",
     "ordinary AdamW; if this is missed the pipeline is broken"),
    ("L4 vs L1  (THE ATTACK)", "L4_sgdfused", "L1_fwd3",
     "same program, differs only by the weight write"),
    ("L3 vs L1  (null pair)", "L3_graddiscard", "L1_fwd3",
     "identical compulsory traffic; separation here is the noise floor"),
    ("L0 vs L1  (token vs FLOP)", "L0_fwd1", "L1_fwd3",
     "how much of any signal is just throughput"),
    ("L4 vs L1  @ M=8192", "L4_sgdfused_m8192", "L1_fwd3_m8192",
     "strongest attack: dW no longer thin-K, weight write diluted to 4.4%"),
    ("L4 vs L1  @ 7B", "L4_sgdfused_7b", "L1_fwd3_7b",
     "does the result hold at governance-relevant scale"),
    # Area Chair 1uob asked how PPO-shaped workloads differ from vanilla
    # dense sgd". PPO is a diluted trainer -- rollout generation interleaved
    # with policy updates. Scored twice: against the FLOP-matched forward
    # control, and against the generative inference it actually contains,
    # which is the negative a real RLHF job would hide inside.
    ("PPO r8 vs L1   (rollout:update 8:1)", "ppo_r8", "L1_fwd3",
     "PPO shape vs compute-bound forward inference"),
    ("PPO r8 vs decode", "ppo_r8", "decode_g256",
     "PPO vs the generative inference it is made of -- the honest hard case"),
    ("PPO r32 vs decode  (8x diluted)", "ppo_r32", "decode_g256",
     "same, at a 4x longer rollout: does the duty cycle break detection?"),
]

# The prefill/decode axis is NOT an AUC question: both are negatives, so the
# separability of p_train between them means nothing. What matters is whether
# the detector stays quiet on every form of inference. Reported as clearance.
NEGATIVE_CELLS = [
    ("L0_fwd1", "1 forward/iter, token-matched"),
    ("L1_fwd3", "3 forwards/iter, compute-bound (prefill-like)"),
    ("L3_graddiscard", "fwd + dX + dW discarded, no learning"),
    ("decode_g256", "autoregressive decode, memory-bandwidth-bound"),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=15.0)
    args = ap.parse_args()

    print("Loading corpus, windowing (causal)...")
    raw = load_all_data()
    win = sliding_windows(raw, args.window, args.stride,
                          run_level_mode="causal").reset_index(drop=True)
    # The DRAM ratios now live in the extractor, so get_feature_cols already
    # returns them. "+DRAM" = the full upgraded set; "paper" = that set minus
    # the four ratios (the pre-Round-6 monitor).
    feats_dram = add_dram_ratios(win, get_feature_cols(win))
    feats_base = [c for c in feats_dram if c not in DRAM_RATIO_COLS]

    wl = win["workload_label"].astype(str)
    is_sym = wl.str.startswith("symgemm_").values
    print(f"  {len(win)} windows, {len(feats_base)} features, "
          f"{is_sym.sum()} symgemm windows across "
          f"{win.loc[is_sym, 'run_id'].nunique()} runs")
    if is_sym.sum() == 0:
        raise SystemExit(
            "No symgemm_* windows found. Run "
            "scripts/rebuttal_collection/round6_symgemm_queue.sh first.")

    # EVERY symgemm run is held out. The corpus the classifier sees is the
    # paper's corpus, exactly as a real deployed monitor would have it.
    train_mask = ~is_sym

    results = []
    for title, pos_cell, neg_cell, note in CELLS:
        pos_mask = wl.str.startswith(f"symgemm_{pos_cell}").values
        neg_mask = wl.str.startswith(f"symgemm_{neg_cell}").values
        # startswith would let L4_sgdfused also match L4_sgdfused_m8192/_7b.
        if pos_cell == "L4_sgdfused":
            pos_mask &= ~wl.str.contains("_m8192|_7b|_triton").values
        if neg_cell == "L1_fwd3":
            neg_mask &= ~wl.str.contains("_m8192|_7b|_triton").values

        if pos_mask.sum() < 10 or neg_mask.sum() < 10:
            print(f"\n### {title}: not collected "
                  f"(pos={pos_mask.sum()} win, neg={neg_mask.sum()}) -- skipped")
            continue

        print(f"\n### {title}")
        print(f"    {note}")
        row = {"comparison": title, "pos_cell": pos_cell,
               "neg_cell": neg_cell, "note": note}
        print(f"  {'features':16s} {'AUC':>6s} {'AP':>6s} {'TPR@1%':>7s} "
              f"{'alert+':>7s} {'alert-':>7s} {'run+':>6s} {'run-':>6s}")
        for fname, fl in ((f"paper ({len(feats_base)})", feats_base),
                          (f"+DRAM ({len(feats_dram)})", feats_dram)):
            r = score(win, fl, train_mask, pos_mask, neg_mask,
                      args.tau, args.k, args.m)
            row[fname] = r
            print(f"  {fname:16s} {r['auc']:>6.3f} {r['ap']:>6.3f} "
                  f"{r['tpr_at_1pct_fpr']:>7.3f} "
                  f"{r['window_alert_pos']:>7.3f} {r['window_alert_neg']:>7.3f} "
                  f"{r['run_pos']['runs_alerting']}/{r['run_pos']['n_runs']:<4d} "
                  f"{r['run_neg']['runs_alerting']}/{r['run_neg']['n_runs']:<4d}")
        results.append(row)

    # Negatives clearance: does the detector stay quiet across the whole
    # prefill-to-decode span? This is the AC's axis, and the right metric is
    # the false-alert rate on each, not separability between them.
    neg_rows = []
    have_neg = [(c, d) for c, d in NEGATIVE_CELLS
                if wl.str.startswith(f"symgemm_{c}").sum() >= 10]
    if have_neg:
        # One classifier, the same one the comparisons use, scored on each.
        pos_probe = (wl.str.startswith("symgemm_L4_sgdfused").values
                     & ~wl.str.contains("_m8192|_7b|_triton").values)
        print(f"\n### NEGATIVES CLEARANCE (tau={args.tau}, k={args.k}/m={args.m})")
        print(f"  {'cell':22s} {'n win':>6s} {'mean p_train':>13s} "
              f"{'window alert':>13s} {'runs alerting':>14s}   description")
        for cell, desc in have_neg:
            m = wl.str.startswith(f"symgemm_{cell}").values
            if cell == "L1_fwd3":
                m &= ~wl.str.contains("_m8192|_7b|_triton").values
            r = score(win, feats_base, train_mask, pos_probe, m,
                      args.tau, args.k, args.m)
            print(f"  {cell:22s} {r['n_neg']:>6d} {r['p_train_neg']:>13.3f} "
                  f"{r['window_alert_neg']:>13.3f} "
                  f"{r['run_neg']['runs_alerting']:>6d}/"
                  f"{r['run_neg']['n_runs']:<7d} {desc}")
            neg_rows.append({"cell": cell, "description": desc,
                             "n_windows": r["n_neg"],
                             "mean_p_train": r["p_train_neg"],
                             "window_alert_rate": r["window_alert_neg"],
                             "run_alert": r["run_neg"]})
        results.append({"comparison": "NEGATIVES CLEARANCE",
                        "negatives": neg_rows})

    # Secondary: the attack against the whole benign corpus, which is the
    # denominator that makes per-family AUC look near-perfect. Reported so the
    # difference between the two denominators is visible rather than implied.
    pos_mask = (wl.str.startswith("symgemm_L4_sgdfused").values
                & ~wl.str.contains("_m8192|_7b|_triton").values)
    if pos_mask.sum() >= 10:
        benign = (~is_sym) & (win["threeway_label"].values != "ml_training")
        r = score(win, feats_base, train_mask, pos_mask, benign,
                  args.tau, args.k, args.m)
        print(f"\n### SECONDARY: L4 vs the full benign corpus "
              f"({int(benign.sum())} windows)")
        print(f"  AUC {r['auc']:.3f} (compare with the same-scale L4-vs-L1 "
              f"figure above -- same attack, different denominator)")
        results.append({"comparison": "L4 vs full benign corpus (SECONDARY)",
                        "paper (166)": r})

    OUT.write_text(json.dumps({
        "tau": args.tau, "k": args.k, "m": args.m,
        "window_sec": args.window, "stride_sec": args.stride,
        "n_windows": len(win), "n_features_base": len(feats_base),
        "results": results,
    }, indent=2, default=float))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
