#!/usr/bin/env python3
"""Did the disguise evade the detector, and what did it cost to do so?

One row per disguised cell, carrying BOTH halves. No existing script reports them
together: symgemm_eval.py measures detection, symgemm_throughput.py measures
cost, and they are joined by hand. Everywhere else the cost is not measured at
all -- evasion_bound.py:279 uses `1.0 - idle_frac  # Approximate`,
eval_dilution_sweep.py:72 uses `1/N`, and report/week8_results.md publishes a
`Cost` column that is just the dilution factor. This script measures it.

Three design points that follow from what has already gone wrong in this repo:

1. DETECTION IS REPORTED TWICE, k-of-m AND CUSUM, at a matched false-alarm
   budget. rebuttal_analysis/cusum_detect.py exists because the symgemm L4
   disguise produces a p_train that is *consistently elevated* (mean ~0.43) yet
   sits below tau=0.75, so the memoryless 3-of-5 rule never fires (0/N). Report
   k-of-m alone and you claim total evasion where a CUSUM detector catches it.

2. COST IS USEFUL WORK, NOT tok/s. Measured WB-O logs show the oracle veto
   cutting real gradient updates ~5x (992 -> ~180) while *raising* tok/s, because
   vetoed work becomes forward-only. So cost is computed from `kind=train` STEP
   lines: tokens that actually moved weights.

3. A DISGUISE THAT NEVER TRAINS IS NOT AN EVASION. Both multi-node strategies can
   drive detection to zero by simply not training. Any cell whose useful work is
   ~zero is reported as INVALID rather than scored as a win.

The benign false-alarm rate (`alert_neg`) is reported alongside, because
lora_in_training.py showed a case where apparent detection sat *below* the benign
control -- which is worse than no detection at all.

    python -m classifier.rebuttal_analysis.evasion_cost_eval
    python -m classifier.rebuttal_analysis.evasion_cost_eval --synthetic
"""

from __future__ import annotations

import argparse
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"

# The machine-parseable cost line emitted by adversarial_multinode.py (and, in
# reduced form, adversarial_composite.py:182 -- which nothing has ever read).
TAX_RE = re.compile(r"EVASION_TAX_DATA:\s*(?P<body>.*)")
FIELD_RE = re.compile(r"(\w+)=([-\w.]*)")

# Useful work below this is not an evasion, it is an inference job.
MIN_USEFUL_TOKENS = 1000


# ── cost side ────────────────────────────────────────────────────────────────

def parse_tax_lines(log_globs):
    """-> DataFrame of measured cost, one row per disguised run log."""
    if isinstance(log_globs, str):
        log_globs = [log_globs]
    rows = []
    for pattern in log_globs:
        for path in sorted(glob.glob(pattern)):
            body = None
            with open(path, errors="ignore") as fh:
                for line in fh:
                    m = TAX_RE.search(line)
                    if m:
                        body = m.group("body")   # last one wins
            if body is None:
                continue
            f = {k: v for k, v in FIELD_RE.findall(body) if v != ""}

            def num(key, default=0.0):
                try:
                    return float(f.get(key, default))
                except ValueError:
                    return default

            dur = num("duration")
            useful = num("useful_tokens")
            rows.append({
                "log": Path(path).name,
                "strategy": f.get("strategy", "?"),
                "world_size": int(num("world_size", 1)),
                "train_frac_req": num("train_frac", float("nan")),
                "dilution": num("dilution", float("nan")),
                "steps": int(num("steps")),
                "train_steps": int(num("train_steps")),
                "useful_tokens": int(useful),
                "attributed_tokens": int(num("attributed_tokens")),
                "duration_s": dur,
                "oracle_queries": int(num("oracle_queries")),
                "oracle_allowed": int(num("oracle_allowed")),
                # measured, not requested
                "train_frac_real": (num("train_steps") / num("steps")
                                    if num("steps") else float("nan")),
                "useful_tok_s": useful / dur if dur else float("nan"),
                "attributed_tok_s": num("attributed_tokens") / dur if dur else float("nan"),
            })
    return pd.DataFrame(rows)


def honest_baselines(log_globs="logs/*s{16,8,1}_train_fsdp1b*.log"):
    """Useful tok/s of the honest fsdp1b rungs, keyed by world_size.

    Parsed from the workload's own final `Throughput: N tok/s` line
    (fsdp_training.py:301-308) so the denominator is measured on the same
    hardware, model and settings as the disguise.
    """
    pat = re.compile(r"Throughput:\s*([\d.]+)\s*tok/s")
    scope_re = re.compile(r"(s1|s8|s16)_train_fsdp1b")
    by_scope = {}
    for path in sorted(glob.glob("logs/*.log")):
        sm = scope_re.search(Path(path).name)
        if not sm:
            continue
        with open(path, errors="ignore") as fh:
            txt = fh.read()
        vals = [float(v) for v in pat.findall(txt)]
        if vals:
            by_scope.setdefault(sm.group(1), []).append(vals[-1])
    return {k: float(np.mean(v)) for k, v in by_scope.items()}


# ── detection side ───────────────────────────────────────────────────────────

def score_with_honest_detector(model_path=None, window_sec=None, stride_sec=None):
    """-> windows frame with p_train from the persisted honest-only detector.

    Uses the artifact the adversary itself queries, so "the attacker knows the
    detector" is literally true rather than approximately true.
    """
    from classifier.model_store import load
    from classifier.threeway_improved import load_all_data, sliding_windows

    clf, meta = load(model_path) if model_path else load()
    window_sec = window_sec or meta.get("window_sec", 30.0)
    stride_sec = stride_sec or meta.get("stride_sec", 15.0)
    cols = meta["feature_cols"]

    df = load_all_data()
    win = sliding_windows(df, window_sec, stride_sec, run_level_mode="causal")
    X = win.reindex(columns=cols).fillna(0).values.astype(np.float32)
    proba = clf.predict_proba(X)
    classes = [str(c) for c in clf.classes_]
    win = win.copy()
    win["p_train"] = proba[:, classes.index("ml_training")]
    return win, meta


def detection_for(win, mask, tau, k, m, cusum_k, h):
    """Run-level detection under BOTH rules for the runs selected by `mask`."""
    from classifier.rebuttal_analysis.cusum_detect import (
        cusum_first_alert, kofm_first_alert,
    )
    sub = win[mask]
    if sub.empty:
        return None
    n_runs = kofm_hits = cusum_hits = 0
    ttfas = []
    p_means = []
    for _, g in sub.groupby("run_id"):
        g = g.sort_values("window_start")
        p = g["p_train"].to_numpy()
        if len(p) == 0:
            continue
        n_runs += 1
        p_means.append(float(p.mean()))
        i_k = kofm_first_alert(p, tau, k, m)
        i_c = cusum_first_alert(p, cusum_k, h)
        if i_k is not None:
            kofm_hits += 1
            ws = g["window_start"].to_numpy()
            ttfas.append(float(ws[i_k] - ws[0]))
        if i_c is not None:
            cusum_hits += 1
    if n_runs == 0:
        return None
    return {
        "n_runs": n_runs,
        "p_train_mean": float(np.mean(p_means)),
        "detect_kofm": kofm_hits / n_runs,
        "detect_cusum": cusum_hits / n_runs,
        "missed_kofm": 1.0 - kofm_hits / n_runs,
        "missed_cusum": 1.0 - cusum_hits / n_runs,
        "median_ttfa_s": float(np.median(ttfas)) if ttfas else float("nan"),
    }


def calibrate(win, k, m, cusum_k, target_fp_per_run):
    """Pick (tau, h) on BENIGN runs so both rules share a false-alarm budget.

    Comparing k-of-m against CUSUM at different false-alarm rates would be
    meaningless, so both are calibrated to the same budget on the same runs.
    """
    from classifier.rebuttal_analysis.cusum_detect import (
        cusum_first_alert, kofm_first_alert,
    )
    benign = win[win["threeway_label"] != "ml_training"]
    seqs = [g.sort_values("window_start")["p_train"].to_numpy()
            for _, g in benign.groupby("run_id")]
    seqs = [s for s in seqs if len(s)]
    if not seqs:
        return 0.75, 5.0, float("nan")

    tau = 0.95
    for t in np.round(np.arange(0.05, 0.96, 0.05), 2):
        fp = sum(kofm_first_alert(s, t, k, m) is not None for s in seqs) / len(seqs)
        if fp <= target_fp_per_run:
            tau = float(t)
            break
    h = 50.0
    for cand in np.round(np.arange(0.5, 50.1, 0.5), 1):
        fp = sum(cusum_first_alert(s, cusum_k, cand) is not None for s in seqs) / len(seqs)
        if fp <= target_fp_per_run:
            h = float(cand)
            break
    fp_neg = sum(kofm_first_alert(s, tau, k, m) is not None for s in seqs) / len(seqs)
    return tau, h, float(fp_neg)


# ── synthetic self-test ──────────────────────────────────────────────────────

def synthetic_check():
    """Prove the harness can detect a KNOWN effect.

    scope_ladder_eval.py's --synthetic corpus is trivially separable (AUC 1.000
    everywhere), so it cannot distinguish "no effect" from "no headroom" -- which
    is exactly the ambiguity that made the honest ladder uninterpretable. This
    fixture instead plants a known evasion rate and a known cost, and asserts the
    harness recovers both.
    """
    from classifier.rebuttal_analysis.cusum_detect import (
        cusum_first_alert, kofm_first_alert,
    )
    rng = np.random.RandomState(0)
    tau, k, m, cusum_k, h = 0.75, 3, 5, 0.25, 3.0

    # 10 runs whose p_train is consistently elevated but sub-tau: k-of-m should
    # miss all of them, CUSUM should catch them. This is the symgemm-L4 regime.
    sneaky = [np.clip(rng.normal(0.43, 0.13, 40), 0, 1) for _ in range(10)]
    kofm = sum(kofm_first_alert(p, tau, k, m) is not None for p in sneaky) / 10
    cus = sum(cusum_first_alert(p, cusum_k, h) is not None for p in sneaky) / 10
    print(f"  sub-tau-but-elevated runs: k-of-m detects {kofm:.0%}, "
          f"CUSUM detects {cus:.0%}")
    ok_rules = kofm < 0.2 and cus > 0.8

    # cost recovery from a known tax line
    line = ("  EVASION_TAX_DATA: strategy=MN-I world_size=16 train_frac=0.1 "
            "dilution= steps=1000 train_steps=100 useful_tokens=1000000 "
            "attributed_tokens=10000000 duration=100.0 oracle_queries=0 "
            "oracle_allowed=0")
    tmp = Path("/tmp/_evasion_tax_selftest.log")
    tmp.write_text(line + "\n")
    tax = parse_tax_lines(str(tmp))
    tmp.unlink()
    row = tax.iloc[0]
    ok_cost = (row["train_steps"] == 100 and row["useful_tok_s"] == 10000.0
               and abs(row["train_frac_real"] - 0.1) < 1e-9
               and row["attributed_tok_s"] == 100000.0)
    print(f"  cost line: train_frac_real={row['train_frac_real']:.2f} "
          f"useful={row['useful_tok_s']:.0f} tok/s "
          f"attributed={row['attributed_tok_s']:.0f} tok/s")
    print(f"\n  rules recover the known effect : {ok_rules}")
    print(f"  cost parsed exactly           : {ok_cost}")
    print("RESULT:", "PASS" if (ok_rules and ok_cost) else "FAIL")
    return 0 if (ok_rules and ok_cost) else 1


# ── main ─────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--synthetic", action="store_true",
                    help="validate the harness on a known effect, no corpus needed")
    ap.add_argument("--family", default="advtrain",
                    help="token identifying the disguised cells (default: advtrain)")
    ap.add_argument("--logs", nargs="+", default=["logs/*.log"])
    ap.add_argument("--model", default=None, help="persisted detector path")
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--cusum-k", type=float, default=0.25)
    ap.add_argument("--target-fp-per-run", type=float, default=0.10)
    ap.add_argument("--out-prefix", default="evasion_cost")
    args = ap.parse_args()

    if args.synthetic:
        print("Synthetic harness check (known evasion rate + known cost)")
        raise SystemExit(synthetic_check())

    tax = parse_tax_lines(args.logs)
    if tax.empty:
        print(f"No EVASION_TAX_DATA lines found in {args.logs}. Nothing to score.")
        print("Run the adversarial cells first; the disguised workload emits it.")
        return
    base = honest_baselines()
    print(f"honest fsdp1b baselines (tok/s): {base}")

    win, meta = score_with_honest_detector(args.model)
    print(f"detector: {meta.get('n_windows')} honest windows, "
          f"{meta.get('n_features')} features, window={meta.get('window_sec')}s")
    tau, h, fp_neg = calibrate(win, args.k, args.m, args.cusum_k,
                               args.target_fp_per_run)
    print(f"calibrated on benign runs: tau={tau} h={h} "
          f"(benign k-of-m alert rate {fp_neg:.3f})")

    wl = win["workload_label"].astype(str)
    rows = []
    for label in sorted(wl[wl.str.contains(args.family)].unique()):
        det = detection_for(win, (wl == label).values, tau, h=h, k=args.k,
                            m=args.m, cusum_k=args.cusum_k)
        if det is None:
            continue
        scope = label.split("_")[0] if label.startswith("s") else "na"
        # cost rows whose log name mentions this cell
        t = tax[tax["log"].str.contains(label, regex=False)]
        useful_tok_s = float(t["useful_tok_s"].mean()) if not t.empty else float("nan")
        train_steps = float(t["train_steps"].sum()) if not t.empty else 0.0
        denom = base.get(scope)
        row = {
            "label": label, "scope": scope,
            "n_runs": det["n_runs"],
            "p_train_mean": round(det["p_train_mean"], 4),
            "detect_kofm": round(det["detect_kofm"], 3),
            "detect_cusum": round(det["detect_cusum"], 3),
            "missed_kofm": round(det["missed_kofm"], 3),
            "missed_cusum": round(det["missed_cusum"], 3),
            "median_ttfa_s": det["median_ttfa_s"],
            "train_steps": int(train_steps),
            "useful_tok_s": round(useful_tok_s, 1) if useful_tok_s == useful_tok_s else None,
            "train_frac_real": (round(float(t["train_frac_real"].mean()), 4)
                                if not t.empty else None),
            "vs_honest": (round(useful_tok_s / denom, 3)
                          if denom and useful_tok_s == useful_tok_s else None),
            "benign_alert_rate": round(fp_neg, 3),
        }
        # A disguise that never trained evaded nothing.
        row["valid"] = bool(not t.empty
                            and t["useful_tokens"].sum() >= MIN_USEFUL_TOKENS)
        rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        print(f"no cells matching family '{args.family}' in the corpus yet")
        return
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    csv = RESULTS_DIR / f"{args.out_prefix}_summary.csv"
    out.to_csv(csv, index=False)
    print("\n" + out.to_string(index=False))
    print(f"\nSaved: {csv}")
    if (~out["valid"]).any():
        print("\nINVALID cells (never trained -- not evidence of evasion):")
        print(out.loc[~out["valid"], ["label", "train_steps"]].to_string(index=False))
    print("\nRead `missed_*` as the evasion rate. If missed_kofm is high but "
          "missed_cusum is low, the disguise defeats the deployed 3-of-5 rule "
          "only, and the paper should say so.")


if __name__ == "__main__":
    main()
