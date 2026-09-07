#!/usr/bin/env python3
"""Baseline three-way detection rate on MULTI-NODE data, per platform.

The persisted honest detector (classifier/models/honest_detector.json) is fitted
across the whole corpus, which is mostly single-GPU. This script answers the
narrower question the adversarial work needs a number for: held out by workload,
how well does the two-stage detector do on jobs that span more than one machine?

Reported per platform, because sections 14-18 established that ABSOLUTE
footprint levels do not transfer across hardware -- pooling H100, A100 and TPU
into one fold would let the model score a window by recognising its hardware.

Positives are honest training; negatives are honest inference and honest non-ML.
Adversarial cells are EXCLUDED from fitting and scored separately, so the number
is a baseline against which a disguise can be measured rather than one already
softened by having seen it.

  PYTHONPATH=. python3 classifier/rebuttal_analysis/multinode_baseline.py
"""
import argparse
import collections
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from classifier.threeway_improved import (  # noqa: E402
    _normalize_columns, get_feature_cols, sliding_windows, threeway_label,
    workload_group)
from classifier.twostage import TwoStageClassifier  # noqa: E402
from sklearn.model_selection import LeaveOneGroupOut  # noqa: E402

# Platform = (vendor family, topology). Only genuinely multi-machine rungs.
PLATFORMS = {
    "H100 s16 (2 nodes x 8)":  dict(hw="H100", scope="s16"),
    "H100 s8 (1 node x 8)":    dict(hw="H100", scope="s8"),
    "A100-40GB s16 (2 nodes)": dict(hw="A100_SXM4_40GB", scope="s16"),
    "TPU v6e multislice":      dict(hw="Google_TPU_v6e", scope="tpuTMS"),
    "TPU v5e multislice":      dict(hw="Google_TPU_v5e", scope="tpuTMS"),
    "TPU v5e 16chip/4host":    dict(hw="Google_TPU_v5e", scope="tpuT16"),
}
ADV = re.compile(r"(advtrain|adversarial|whitebox|typeconf|stalled)", re.I)


def cell_of(stem):
    for tok in ("_Google_TPU_", "_NVIDIA_", "_AMD_"):
        if tok in stem:
            return stem.split(tok)[0], stem.split(tok)[1]
    return None, None


def load(paths, window, stride):
    frames = []
    for p in paths:
        stem = p.stem
        cell, hw = cell_of(stem)
        if cell is None:
            continue
        d = _normalize_columns(pd.read_parquet(p))
        lab = re.sub(r"_(node\d+|gpu\d+)", "", cell)
        d["workload_label"] = lab
        d["run_id"] = stem
        d["threeway_label"] = threeway_label(lab)
        d["hw"] = hw
        frames.append(d)
    if not frames:
        return None
    df = pd.concat(frames, ignore_index=True)
    w = sliding_windows(df, window, stride)
    w["grp"] = workload_group(w.workload_label.values)
    w["cls"] = [threeway_label(l) for l in w.workload_label]
    w["is_adv"] = [bool(ADV.search(l)) for l in w.workload_label]
    return w


def evaluate(w, cols, tau, k, m, seed=0):
    """Fit honest-only, held out by workload; score held-out honest + all adv."""
    honest = w[~w.is_adv].reset_index(drop=True)
    if honest.cls.nunique() < 3:
        return None
    p = np.full(len(honest), np.nan)
    groups = honest.grp.values
    for tr, te in LeaveOneGroupOut().split(np.zeros(len(honest)), None, groups):
        w_tr = honest.iloc[tr]
        if w_tr.cls.nunique() < 3:
            continue
        X = np.nan_to_num(w_tr[cols].to_numpy(np.float32), nan=0.0,
                          posinf=0.0, neginf=0.0)
        clf = TwoStageClassifier(random_state=seed)
        clf.fit(X, w_tr.cls.values, groups=w_tr.run_id.values,
                feature_cols=list(cols))
        Xte = np.nan_to_num(honest.iloc[te][cols].to_numpy(np.float32),
                            nan=0.0, posinf=0.0, neginf=0.0)
        p[te] = clf.predict_proba_training(Xte)
    keep = ~np.isnan(p)
    h, ph = honest[keep].reset_index(drop=True), p[keep]
    pos = h.cls == "ml_training"
    out = dict(
        n_windows=int(len(h)),
        n_groups=int(h.grp.nunique()),
        tpr=float((ph[pos.values] > tau).mean()) if pos.any() else float("nan"),
        fpr=float((ph[~pos.values] > tau).mean()) if (~pos).any() else float("nan"),
        fpr_infer=float((ph[(h.cls == "ml_inference").values] > tau).mean())
        if (h.cls == "ml_inference").any() else float("nan"),
        fpr_nonml=float((ph[(h.cls == "other").values] > tau).mean())
        if (h.cls == "other").any() else float("nan"),
    )
    # Run-level k-of-m on consecutive windows of a run.
    def run_rule(sub, pr):
        alerts, n = 0, 0
        for rid, idx in sub.groupby("run_id").indices.items():
            v = pr[idx]
            n += 1
            if len(v) < m:
                alerts += int((v > tau).sum() >= k)
                continue
            hit = any((v[i:i + m] > tau).sum() >= k for i in range(len(v) - m + 1))
            alerts += int(hit)
        return alerts / n if n else float("nan")
    out["run_tpr"] = run_rule(h[pos.values], ph[pos.values]) if pos.any() else float("nan")
    out["run_fpr"] = run_rule(h[~pos.values], ph[~pos.values]) if (~pos).any() else float("nan")
    return out, h, ph


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=15.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--platforms", nargs="*", default=None)
    a = ap.parse_args()

    import pathlib
    allp = [p for p in pathlib.Path("data").rglob("*.parquet")
            if not p.name.startswith("nic_") and "ncu_profiles" not in str(p)]
    print(f"{len(allp)} candidate parquets under data/")

    rows = []
    for name, sel in PLATFORMS.items():
        if a.platforms and name not in a.platforms:
            continue
        paths = [p for p in allp
                 if sel["hw"] in p.stem and re.match(rf"^{sel['scope']}_", p.stem)]
        if not paths:
            print(f"\n=== {name}: no parquets matched ===")
            continue
        w = load(paths, a.window, a.stride)
        if w is None:
            print(f"\n=== {name}: nothing loaded ===")
            continue
        cols = get_feature_cols(w)
        print(f"\n=== {name} ===")
        print(f"  {len(paths)} streams, {len(w):,} windows, {len(cols)} features, "
              f"classes={dict(w[~w.is_adv].cls.value_counts())}")
        res = evaluate(w, cols, a.tau, a.k, a.m)
        if res is None:
            print("  SKIP: honest fold does not contain all three classes")
            continue
        out, h, ph = res
        print(f"  window  TPR@{a.tau}={out['tpr']:.3f}  FPR={out['fpr']:.3f} "
              f"(infer {out['fpr_infer']:.3f} / nonml {out['fpr_nonml']:.3f})")
        print(f"  run {a.k}-of-{a.m}  TPR={out['run_tpr']:.3f}  FPR={out['run_fpr']:.3f}"
              f"   [{out['n_groups']} workload groups]")
        rows.append(dict(platform=name, **out))
        # adversarial cells, scored by the same held-out-fit models is not
        # possible group-wise here; report them under a fit on ALL honest data.
        adv = w[w.is_adv]
        if len(adv):
            honest = w[~w.is_adv]
            X = np.nan_to_num(honest[cols].to_numpy(np.float32), nan=0.0,
                              posinf=0.0, neginf=0.0)
            clf = TwoStageClassifier(random_state=0)
            clf.fit(X, honest.cls.values, groups=honest.run_id.values,
                    feature_cols=list(cols))
            Xa = np.nan_to_num(adv[cols].to_numpy(np.float32), nan=0.0,
                               posinf=0.0, neginf=0.0)
            pa = clf.predict_proba_training(Xa)
            print(f"  adversarial cells vs an honest-only fit (never seen):")
            for grp, idx in adv.groupby("grp").indices.items():
                v = pa[idx]
                print(f"    {grp:44s} p_train={np.median(v):.3f}  "
                      f">tau={float((v > a.tau).mean()):.3f}  n={len(v)}")
    if rows:
        print("\n" + "=" * 78)
        print(pd.DataFrame(rows).to_string(index=False, float_format="%.3f"))


if __name__ == "__main__":
    main()
