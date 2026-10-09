#!/usr/bin/env python3
"""Is section 21's v5e-vs-v6e evasion gap explained by TRAINING-SET SIZE?

Section 21 found the same MN-T script, under the same family holdout, evading
completely on v5e (frac>tau = 0.000) while being caught on v6e (0.494-0.875),
and flagged the obvious confound itself:

    "Training-set size differs (5,510 vs 7,692 windows) and is a partial
     confound, but a 40% data difference does not plausibly explain 0.000
     versus 0.5-0.875."

That sentence is an assertion, not a measurement. This script measures it: it
subsamples the v6e training fold down to the v5e size (and below), stratified by
class so only the SIZE changes and not the composition, and reports the ladder at
each size over several seeds. If v6e still detects at 5,510 windows, size is
ruled out and the generation difference stands on its own.

Family holdout, as in section 21: all typeconf_* groups leave the training fold
together, so the detector has never seen the script.

  python classifier/rebuttal_analysis/tpu_mnt_size_control.py --scope tpuTMS
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from classifier.rebuttal_analysis.tpu_rung_eval import (  # noqa: E402
    load_windows, _can_two_stage, _fit_two_stage, _p_train)
from classifier.threeway_improved import get_feature_cols  # noqa: E402


def stratified_subsample(w, n_target, seed):
    """Sample n_target windows, preserving w's class proportions.

    Plain uniform sampling would let a small class vanish at the smallest sizes,
    which would confound "less data" with "a class disappeared" -- and Stage 2
    needs both ML classes present or it is not a detector at all (_can_two_stage).
    """
    rng = np.random.default_rng(seed)
    if n_target >= len(w):
        return w
    take = []
    for cls, sub in w.groupby("cls"):
        k = int(round(n_target * len(sub) / len(w)))
        k = max(1, min(k, len(sub)))
        take.append(rng.choice(sub.index.values, size=k, replace=False))
    idx = np.concatenate(take)
    return w.loc[idx]


def ladder(clf, w_te, cols, tau):
    p, pml, ptg = _p_train(clf, w_te, cols, two_stage=True)
    lad = w_te.assign(_p=p, _pml=pml, _ptg=ptg)
    out = {}
    for grp, sub in lad.groupby("grp"):
        out[grp] = (float(sub._p.median()), float((sub._p > tau).mean()))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/tpu/*.parquet")
    ap.add_argument("--scope", default="tpuTMS")
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--stride-sec", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--sizes", default="",
                    help="comma-separated training-fold window counts; "
                         "default = the other generation's size, plus halves")
    a = ap.parse_args()

    w = load_windows(a.glob, a.window_sec, a.stride_sec)
    w = w[w.scope == a.scope].reset_index(drop=True)
    cols = get_feature_cols(w)
    print(f"scope {a.scope}: {len(w):,} windows, {w.grp.nunique()} groups, "
          f"{len(cols)} features")
    print("by generation:", dict(w.gen.value_counts()))

    gens = [g for g, n in w.gen.value_counts().items() if n > 50]
    folds = {}
    for gen in gens:
        wg = w[w.gen == gen].reset_index(drop=True)
        te = wg[wg.grp.str.contains("typeconf")]
        tr = wg[~wg.grp.str.contains("typeconf")]
        folds[gen] = (tr, te.reset_index(drop=True))
        print(f"  {gen}: train fold {len(tr):,} win ({tr.grp.nunique()} groups), "
              f"held-out typeconf {len(te):,} win ({te.grp.nunique()} groups)")

    # Full-size baseline per generation, then the size sweep.
    for gen in gens:
        tr, te = folds[gen]
        if te.empty:
            print(f"\n=== {gen}: no typeconf cells, skipping ===")
            continue
        ok, n_tr, n_if = _can_two_stage(tr)
        if not ok:
            print(f"\n=== {gen}: Stage 2 single-class ({n_tr} train / {n_if} "
                  f"infer) -- not a detector, skipping ===")
            continue
        print(f"\n=== {gen}: family holdout, FULL train fold ({len(tr):,} win) ===")
        base = ladder(_fit_two_stage(tr, cols), te, cols, a.tau)
        for k in sorted(base):
            print(f"    {k:34s} p_train={base[k][0]:.3f}  >tau={base[k][1]:.3f}")

        if a.sizes:
            sizes = [int(x) for x in a.sizes.split(",")]
        else:
            others = [len(folds[g][0]) for g in gens if g != gen]
            sizes = sorted({s for s in others if s < len(tr)}
                           | {len(tr) // 2, len(tr) // 4})
        for n in sizes:
            if n >= len(tr):
                continue
            runs = []
            for s in range(a.seeds):
                sub = stratified_subsample(tr, n, seed=s)
                ok_s, _, _ = _can_two_stage(sub)
                if not ok_s:
                    continue
                runs.append(ladder(_fit_two_stage(sub, cols, seed=s), te,
                                   cols, a.tau))
            if not runs:
                print(f"\n  train fold subsampled to {n:,}: every seed lost a "
                      f"class in Stage 2 -- cannot evaluate")
                continue
            print(f"\n  train fold subsampled to {n:,} win "
                  f"({len(runs)} seeds, mean +- sd):")
            for k in sorted(base):
                ps = [r[k][0] for r in runs if k in r]
                fs = [r[k][1] for r in runs if k in r]
                print(f"    {k:34s} p_train={np.mean(ps):.3f}+-{np.std(ps):.3f}"
                      f"  >tau={np.mean(fs):.3f}+-{np.std(fs):.3f}")


if __name__ == "__main__":
    main()
