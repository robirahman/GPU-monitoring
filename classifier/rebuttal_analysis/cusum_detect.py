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


def hybrid_first_alert(p, tau, k, m, cusum_k, h):
    """First-alert index of the union rule: 3-of-5 OR CUSUM (min of the two)."""
    idx = [i for i in (kofm_first_alert(p, tau, k, m),
                       cusum_first_alert(p, cusum_k, h)) if i is not None]
    return min(idx) if idx else None


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
                    help="target benign false-alert budget (calibrate h to it)")
    ap.add_argument("--fp-unit", choices=["hour", "run"], default="hour",
                    help="'hour' = false alerts per benign run-hour; "
                         "'run' = false-alert rate per benign run (fraction)")
    args = ap.parse_args()

    df = pd.read_parquet(args.preds)

    # Categorize runs. Benign = non-symgemm; the calibration/false-alarm pool is
    # benign inference (+other). symgemm cells are the evasion test set.
    def cat(wlabel, tlabel):
        if str(wlabel).startswith("symgemm_"):
            return "symgemm"
        return "benign_train" if tlabel == "ml_training" else "benign_infer"

    tagged = [(rid, wlabel, tlabel, p, gh, cat(wlabel, tlabel))
              for (rid, wlabel, tlabel, p, gh) in run_series(df)]
    benign_infer = [t for t in tagged if t[5] == "benign_infer"]
    benign_gpu_h = sum(t[4] for t in benign_infer)
    n_benign = len(benign_infer)

    def hyb(p, h):
        return hybrid_first_alert(p, args.tau, args.k, args.m, args.cusum_k, h)

    def benign_fp(h):
        """Benign false-alert rate of the HYBRID rule, in the chosen unit."""
        alerts = sum(1 for t in benign_infer if hyb(t[3], h) is not None)
        denom = benign_gpu_h if args.fp_unit == "hour" else n_benign
        return alerts / denom if denom else float("inf")

    # Calibrate the CUSUM threshold h so the combined (hybrid) benign FP meets budget.
    h = 80.0
    for cand in np.arange(2.0, 80.0, 0.5):
        if benign_fp(cand) <= args.fp_budget:
            h = cand
            break

    def fp_both(h):
        a = sum(1 for t in benign_infer if hyb(t[3], h) is not None)
        return (a / benign_gpu_h if benign_gpu_h else 0.0,
                a / n_benign if n_benign else 0.0)
    fh, fr = fp_both(h)
    unit = "/run-hour" if args.fp_unit == "hour" else "/run"
    print(f"Benign inference pool: {n_benign} runs, {benign_gpu_h:.1f} run-hours")
    print(f"Hybrid 3-of-{args.m} (tau={args.tau}) OR CUSUM (k={args.cusum_k}), "
          f"calibrated to {args.fp_budget}{unit} -> h={h:.1f}")
    print(f"realized benign false alerts: {fh:.3f}/run-hour = {fr:.3f}/run\n")

    STRIDE = 15.0

    def rates(sel):
        base = np.mean([kofm_first_alert(t[3], args.tau, args.k, args.m) is not None for t in sel])
        cus = np.mean([cusum_first_alert(t[3], args.cusum_k, h) is not None for t in sel])
        hy = [hyb(t[3], h) for t in sel]
        hyr = np.mean([x is not None for x in hy])
        tt = [x for x in hy if x is not None]
        med = f"{np.median(tt)*STRIDE/60:.1f}min" if tt else "-"
        return base, cus, hyr, med

    cells = [
        ("L4 @1B (attack)", lambda s: s.startswith("symgemm_L4_sgdfused") and
         not any(x in s for x in ("m8192", "7b", "triton", "long1b", "short7b"))),
        ("L4 @M=8192", lambda s: s.startswith("symgemm_L4_sgdfused_m8192")),
        ("L4 @7B", lambda s: s.startswith("symgemm_L4_sgdfused_7b")),
        ("L4 @1B long", lambda s: s.startswith("symgemm_L4_sgdfused_long1b")),
        ("L4 @7B short", lambda s: s.startswith("symgemm_L4_sgdfused_short7b")),
        ("L5 AdamW (pos ctrl)", lambda s: s.startswith("symgemm_L5")),
        ("PPO (all)", lambda s: s.startswith("symgemm_ppo")),
        ("benign training", None),
        ("-- negatives (want 0) --", None),
        ("L1 fwd (neg)", lambda s: s.startswith("symgemm_L1_fwd3") and
         "m8192" not in s and "7b" not in s),
        ("L3 discard (neg)", lambda s: s.startswith("symgemm_L3")),
        ("decode (neg)", lambda s: s.startswith("symgemm_decode")),
    ]
    print(f"{'cell':22s} {'n':>4s} {'3-of-5':>8s} {'CUSUM':>8s} {'HYBRID':>8s} {'hyb ttfa':>9s}")
    for name, fn in cells:
        if fn is None and name.startswith("--"):
            print(name); continue
        sel = ([t for t in tagged if t[5] == "benign_train"] if name == "benign training"
               else [t for t in tagged if fn(t[1])])
        if not sel:
            continue
        base, cus, hyr, med = rates(sel)
        print(f"{name:22s} {len(sel):>4d} {base:>7.1%} {cus:>7.1%} {hyr:>7.1%} {med:>9s}")


if __name__ == "__main__":
    main()
