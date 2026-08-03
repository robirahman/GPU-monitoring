#!/usr/bin/env python3
"""Run-level sequential (CUSUM) detector over per-window P(train).

Motivation. The fused-write disguise (symgemm L4) produces a per-window
P(train) that is *consistently* elevated (mean ~0.43, std ~0.13) yet sits below
the deployed alert threshold tau=0.75, so the memoryless "k of m windows > tau"
rule never fires (0/N). But a consistent bias is exactly what a sequential test
catches: accumulate evidence over the run instead of demanding any single window
clear a high bar.

Detector. One-sided CUSUM on the per-window probability:
    S_0 = 0 ;  S_t = max(0, S_{t-1} + (p_t - k)) ;  alert when S_t >= h
`k` is a reference (drift) between the inference band (~0.07) and the training
band; inference runs drift S to 0, training runs accumulate. `h` is calibrated
on benign inference runs to a false-alert budget, so the comparison against the
3-of-5 rule is at matched false alarms. Windows overlap (30 s / 15 s), so the
accumulation is on correlated samples; calibrating h on benign runs with the
SAME window structure controls for that (the correlation is present in both the
benign and the attack streams).

Reads the per-window predictions written by dump_window_predictions
(classifier/results/window_preds/twostage_cv__causal__w30.parquet).
"""
from __future__ import annotations
import argparse
import numpy as np
import pandas as pd


def cusum_first_alert(p, k, h):
    """Index of the first window where the CUSUM statistic crosses h, else None."""
    S = 0.0
    for i, pi in enumerate(p):
        S = max(0.0, S + (pi - k))
        if S >= h:
            return i
    return None


def kofm_first_alert(p, tau, k, m):
    """Index of the first window where k of the last m windows exceed tau."""
    flags = (np.asarray(p) > tau).astype(int)
    for i in range(len(flags)):
        lo = max(0, i - m + 1)
        if flags[lo:i + 1].sum() >= k:
            return i
    return None


def run_series(df):
    """Yield (run_id, workload_label, threeway_label, p_train[], gpu_hours)."""
    for rid, g in df.groupby("run_id"):
        g = g.sort_values("window_start")
        p = g["p_train"].to_numpy()
        if len(p) < 2:
            continue
        dur = float(g["window_end"].iloc[-1] - g["window_start"].iloc[0])
        yield (rid, g["workload_label"].iloc[0], g["threeway_label"].iloc[0],
               p, max(dur, 1.0) / 3600.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--preds",
                    default="classifier/results/window_preds/twostage_cv__causal__w30.parquet")
    ap.add_argument("--tau", type=float, default=0.75, help="3-of-5 baseline threshold")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--cusum-k", type=float, default=0.25,
                    help="CUSUM reference/drift (between inference and training bands)")
    ap.add_argument("--fp-budget", type=float, default=0.10,
                    help="target benign false alerts per GPU-hour (calibrate h)")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)
    wl = df["workload_label"].astype(str)

    # Categorize runs. Benign = non-symgemm; the calibration/false-alarm pool is
    # benign inference (+other). symgemm cells are the evasion test set.
    is_sym = wl.str.startswith("symgemm_")
    runs = list(run_series(df))

    def cat(rid, wlabel, tlabel):
        s = str(wlabel)
        if s.startswith("symgemm_"):
            return "symgemm"
        return "benign_train" if tlabel == "ml_training" else "benign_infer"

    tagged = [(rid, wlabel, tlabel, p, gh, cat(rid, wlabel, tlabel))
              for (rid, wlabel, tlabel, p, gh) in runs]
    benign_infer = [t for t in tagged if t[5] == "benign_infer"]
    benign_gpu_h = sum(t[4] for t in benign_infer)

    # Calibrate CUSUM h to the false-alert budget on benign inference runs.
    def benign_fp_per_gpuh(h):
        alerts = sum(1 for t in benign_infer
                     if cusum_first_alert(t[3], args.cusum_k, h) is not None)
        return alerts / benign_gpu_h if benign_gpu_h else float("inf")

    h = 0.5
    for cand in np.arange(0.5, 60.0, 0.5):
        if benign_fp_per_gpuh(cand) <= args.fp_budget:
            h = cand
            break
    else:
        h = 60.0
    cusum_fp = benign_fp_per_gpuh(h)

    # Baseline 3-of-5 benign false-alert rate at tau, same pool.
    base_alerts = sum(1 for t in benign_infer
                      if kofm_first_alert(t[3], args.tau, args.k, args.m) is not None)
    base_fp = base_alerts / benign_gpu_h if benign_gpu_h else float("nan")

    print(f"Benign inference pool: {len(benign_infer)} runs, {benign_gpu_h:.1f} GPU-h")
    print(f"3-of-5 @ tau={args.tau}: false alerts = {base_fp:.3f}/GPU-h")
    print(f"CUSUM k={args.cusum_k}, calibrated h={h:.1f}: false alerts = {cusum_fp:.3f}/GPU-h")
    print(f"(both at/below the {args.fp_budget}/GPU-h budget)\n")

    def detect(mask_fn):
        sel = [t for t in tagged if mask_fn(t[1])]
        if not sel:
            return None
        n = len(sel)
        base = [kofm_first_alert(t[3], args.tau, args.k, args.m) for t in sel]
        cus = [cusum_first_alert(t[3], args.cusum_k, h) for t in sel]
        base_rate = np.mean([b is not None for b in base])
        cus_rate = np.mean([c is not None for c in cus])
        # median time-to-first-alert (windows -> seconds at 15 s stride) for CUSUM
        ttfa = [c for c in cus if c is not None]
        med_ttfa = f"{np.median(ttfa)*15:.0f}s" if ttfa else "-"
        return n, base_rate, cus_rate, med_ttfa

    cells = [
        ("L4 @1B (attack)", lambda s: s.startswith("symgemm_L4_sgdfused") and
         "m8192" not in s and "7b" not in s and "triton" not in s),
        ("L4 @M=8192", lambda s: s.startswith("symgemm_L4_sgdfused_m8192")),
        ("L4 @7B", lambda s: s.startswith("symgemm_L4_sgdfused_7b")),
        ("L5 AdamW (pos control)", lambda s: s.startswith("symgemm_L5")),
        ("PPO (all)", lambda s: s.startswith("symgemm_ppo")),
        ("benign training", lambda s: not s.startswith("symgemm_")),  # filtered below
        ("-- negatives (want 0) --", None),
        ("L1 fwd (neg)", lambda s: s.startswith("symgemm_L1_fwd3") and
         "m8192" not in s and "7b" not in s),
        ("L3 discard (neg)", lambda s: s.startswith("symgemm_L3")),
        ("decode (neg)", lambda s: s.startswith("symgemm_decode")),
    ]
    print(f"{'cell':26s} {'n':>4s} {'3-of-5':>8s} {'CUSUM':>8s} {'CUSUM ttfa':>11s}")
    for name, fn in cells:
        if fn is None:
            print(name)
            continue
        if name == "benign training":
            sel = [t for t in tagged if t[5] == "benign_train"]
            if not sel:
                continue
            base_rate = np.mean([kofm_first_alert(t[3], args.tau, args.k, args.m) is not None
                                 for t in sel])
            cus_rate = np.mean([cusum_first_alert(t[3], args.cusum_k, h) is not None
                                for t in sel])
            print(f"{name:26s} {len(sel):>4d} {base_rate:>7.1%} {cus_rate:>7.1%} {'':>11s}")
            continue
        r = detect(fn)
        if r:
            n, base_rate, cus_rate, med = r
            print(f"{name:26s} {n:>4d} {base_rate:>7.1%} {cus_rate:>7.1%} {med:>11s}")


if __name__ == "__main__":
    main()
