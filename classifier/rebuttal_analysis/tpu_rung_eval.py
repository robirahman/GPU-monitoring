#!/usr/bin/env python3
"""Evaluate the TPU rung: within-scope, and ACROSS TPU generations.

The within-scope numbers cannot settle the question on their own. Every
adversarial cell is the same script with the same model and filler, so holding
one out by workload still leaves its siblings in the training fold and the
classifier can score a held-out cell by recognising its family. With 8 workload
groups on one topology there is no way to separate the two.

Training on one GENERATION and testing on the other does separate them. v6e and
v5e differ in chip (31996 vs 16126 MB HBM), in how the same model shards (2 chips
vs 8), in interconnect for the same collective (DCN across slices vs ICI within
one), and in what the telemetry even contains -- dcn_collective_lat_us_* is live
on the multislice runs and NaN on the single slice. A detector that transfers
across that is not recognising a script.

Reports, for each evaluation:
  * Stage 1 alone (ML vs other) -- what the corpus could support before inference
    cells existed;
  * the full two-stage p_train with the WB-O level drop on Stage 2, which is what
    the paper's operating point is defined on;
  * per-cell rates at tau, so an F rung that fails is visible rather than averaged
    away, and honest cells are named individually.

Usage:
    python3 classifier/rebuttal_analysis/tpu_rung_eval.py --tau 0.8
"""

import argparse
import glob
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from classifier.threeway_improved import (  # noqa: E402
    _normalize_columns, sliding_windows, threeway_label, workload_group,
    get_feature_cols)
from classifier.twostage import TwoStageClassifier, stage2_feature_mask  # noqa: E402

_SCOPE = re.compile(r"^(tpuT\d+|tpuTMS\d*)_")


def load_windows(pattern, window_sec, stride_sec):
    frames = []
    for f in sorted(glob.glob(pattern)):
        stem = os.path.basename(f)[:-len(".parquet")]
        if "_Google_TPU_" not in stem:
            continue
        d = _normalize_columns(pd.read_parquet(f))
        lab = stem.split("_Google_TPU_")[0]
        d["workload_label"] = lab
        d["run_id"] = stem
        d["threeway_label"] = threeway_label(lab)
        frames.append(d)
    if not frames:
        raise SystemExit(f"no parquets matched {pattern}")
    df = pd.concat(frames, ignore_index=True)
    w = sliding_windows(df, window_sec, stride_sec)
    w["grp"] = workload_group(w.workload_label.values)
    w["cls"] = [threeway_label(l) for l in w.workload_label]
    w["scope"] = [(_SCOPE.match(l).group(1) if _SCOPE.match(l) else "?")
                  for l in w.workload_label]
    # Generation from gpu_name, which the collector now derives from the
    # runtime's own accelerator-type rather than a hardcoded default.
    w["gen"] = (df.groupby("run_id").gpu_name.first()
                .reindex(w.run_id).str.replace("Google TPU ", "", regex=False).values)
    return w


def _can_two_stage(w_tr):
    """Stage 2 needs BOTH ml classes in the training fold.

    The v6e multislice corpus has no ml_inference cells at all -- the campaign ran
    with ADVERSARIAL=1 and STRETCH unset until 2026-09-02 -- so a Stage 2 fitted
    on it reports "1371 train / 0 infer", is single-class, and its predict_proba
    returns one column, which used to crash this script with an IndexError. A
    single-class Stage 2 is not a degraded detector, it is no detector: p(train|ml)
    would be a constant. Skip the two-stage evaluation and say so.
    """
    n_tr = int((w_tr.cls == "ml_training").sum())
    n_if = int((w_tr.cls == "ml_inference").sum())
    return n_tr > 0 and n_if > 0, n_tr, n_if


def _fit_two_stage(w_tr, cols, seed=0):
    X = np.nan_to_num(w_tr[cols].to_numpy(dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    y = w_tr.cls.values
    clf = TwoStageClassifier(random_state=seed)
    clf.fit(X, y, groups=w_tr.run_id.values, feature_cols=list(cols))
    return clf


def _p_train(clf, w_te, cols, two_stage=True):
    """(p_train, p_ml, p_train_given_ml).

    The third is what answers a TYPE-confusion question. p_train alone cannot
    distinguish "Stage 1 rejected it as non-ML" from "Stage 2 called it
    inference", and MN-T attacks only the second. Reporting them separately is
    the difference between measuring the attack and measuring its shadow.
    """
    X = np.nan_to_num(w_te[cols].to_numpy(dtype=np.float32),
                      nan=0.0, posinf=0.0, neginf=0.0)
    pml = clf.predict_proba_ml(X)
    if not two_stage:
        nan = np.full(len(X), np.nan)
        return nan, pml, nan
    return (clf.predict_proba_training(X), pml,
            clf.predict_proba_train_vs_infer(X))


def report(name, w_te, p, tau, ml=None):
    print(f"\n--- {name} ---")
    rows = []
    for grp, sub in w_te.groupby("grp"):
        idx = sub.index
        pv = p[w_te.index.get_indexer(idx)]
        rows.append(dict(cell=grp, cls=sub.cls.iloc[0], n=len(sub),
                         p_med=float(np.median(pv)),
                         rate=float((pv > tau).mean())))
    t = pd.DataFrame(rows).sort_values(["cls", "cell"])
    print(t.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    is_train = (w_te.cls == "ml_training").values
    adv = w_te.grp.str.contains("advtrain").values
    honest = (w_te.cls == "other").values
    print(f"  window TPR@{tau} (all ml_training) = {(p[is_train] > tau).mean():.3f}")
    print(f"  window FPR@{tau} (honest 'other')  = {(p[honest] > tau).mean():.3f}")
    if adv.any():
        print(f"  window TPR@{tau} (disguised only)  = {(p[adv] > tau).mean():.3f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--glob", default="data/tpu/*.parquet")
    ap.add_argument("--window-sec", type=float, default=30.0)
    ap.add_argument("--stride-sec", type=float, default=10.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--drop-memory", action="store_true",
                    help="exclude mem_* features (they carry the arena artifact)")
    a = ap.parse_args()

    w = load_windows(a.glob, a.window_sec, a.stride_sec)
    cols = get_feature_cols(w)
    if a.drop_memory:
        cols = [c for c in cols if not c.startswith("mem_")]
    print(f"windows {len(w):,}  groups {w.grp.nunique()}  features {len(cols)}")
    print("by generation:", dict(w.gen.value_counts()))
    print("by scope:     ", dict(w.scope.value_counts()))
    print("by class:     ", dict(w.cls.value_counts()))
    m = stage2_feature_mask(list(cols))
    print(f"Stage-2 keeps {int(m.sum())} of {len(cols)} features (level drop = WB-O)")

    gens = [g for g, n in w.gen.value_counts().items() if n > 50]
    if len(gens) < 2:
        print(f"\nonly one generation present ({gens}); cross-generation test needs both")
    # ---- cross-generation transfer -------------------------------------------
    for tr_gen in gens:
        for te_gen in gens:
            if tr_gen == te_gen:
                continue
            w_tr = w[w.gen == tr_gen]
            w_te = w[w.gen == te_gen].reset_index(drop=True)
            if w_tr.cls.nunique() < 2 or w_te.empty:
                continue
            ok, n_tr, n_if = _can_two_stage(w_tr)
            clf = _fit_two_stage(w_tr, cols)
            p, pml, p_t = _p_train(clf, w_te, cols, two_stage=ok)
            if ok:
                report(f"TRAIN {tr_gen} ({len(w_tr):,} win) -> TEST {te_gen} "
                       f"({len(w_te):,} win)  [two-stage p_train]", w_te, p, a.tau)
            else:
                print(f"\n--- TRAIN {tr_gen} -> TEST {te_gen}: two-stage SKIPPED "
                      f"({n_tr} training / {n_if} inference windows in the training "
                      f"fold; Stage 2 needs both) ---")
            report(f"TRAIN {tr_gen} -> TEST {te_gen}  [Stage 1 only, ML vs other]",
                   w_te, pml, a.tau)

    # ---- within-generation, held out by WORKLOAD -----------------------------
    from sklearn.model_selection import LeaveOneGroupOut
    for gen in gens:
        wg = w[w.gen == gen].reset_index(drop=True)
        ok, n_tr, n_if = _can_two_stage(wg)
        if not ok:
            print(f"\n--- within {gen}: two-stage SKIPPED ({n_tr} training / "
                  f"{n_if} inference windows) ---")
            continue
        p = np.full(len(wg), np.nan)
        pml = np.full(len(wg), np.nan)
        p_tg = np.full(len(wg), np.nan)
        g = wg.grp.values
        for tr, te in LeaveOneGroupOut().split(np.zeros(len(wg)), None, g):
            w_tr, w_te = wg.iloc[tr], wg.iloc[te].reset_index(drop=True)
            ok_f, _, _ = _can_two_stage(w_tr)
            if not ok_f:
                continue
            clf = _fit_two_stage(w_tr, cols)
            pp, pm, pt = _p_train(clf, w_te, cols, two_stage=True)
            p[te], pml[te], p_tg[te] = pp, pm, pt
        keep = ~np.isnan(p)
        report(f"within {gen}, held out by WORKLOAD ({int(keep.sum()):,} of "
               f"{len(wg):,} win scored)  [two-stage p_train]",
               wg[keep].reset_index(drop=True), p[keep], a.tau)
        report(f"within {gen}  [Stage 2 only: p(train | ML)] -- the TYPE decision "
               f"MN-T attacks", wg[keep].reset_index(drop=True), p_tg[keep], a.tau)
        # MN-T ladder: does p(train|ML) fall as the shaping strengthens, on cells
        # that all train on EVERY step? That ordering is the attack working.
        lad = wg[keep].reset_index(drop=True)
        lad = lad.assign(_p=p[keep], _ptg=p_tg[keep], _pml=pml[keep])
        tc = lad[lad.grp.str.contains("typeconf")]
        if len(tc):
            print("\n  MN-T ladder (all cells train on every step):")
            print(f"    {'cell':38s} {'p_ml':>7s} {'p(t|ML)':>9s} {'p_train':>9s} "
                  f"{'>tau':>6s}")
            for grp, sub in tc.groupby("grp"):
                print(f"    {grp:38s} {sub._pml.median():7.3f} "
                      f"{sub._ptg.median():9.3f} {sub._p.median():9.3f} "
                      f"{float((sub._p > a.tau).mean()):6.3f}")


if __name__ == "__main__":
    main()
