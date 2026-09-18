#!/usr/bin/env python3
"""Three-way detection on the MULTI-NODE corpus, held out by workload FAMILY.

Protocol, and why each piece is there:

* **Multi-node only, measured.** A cell is multi-node if its job spans more than
  one HOST, not more than one device. The chip count in a scope name does not
  tell you that: tpuT16 runs are w0..w3 x c0..c3 (four hosts of four chips) while
  tpuT8 runs are w0c0..w0c7 (one host of eight). s1_/s8_ are excluded by label
  regardless of host count, because multinode_queue.sh derives run_id from
  (stamp, cell, rep) with no node component, so both machines run independent
  single-node jobs under one run_id.

* **Split by FAMILY, not by workload.** Holding out a workload still leaks when
  its siblings differ only in a parameter: s16_advtrain_interleave_f02 / _f05 /
  _f10 / _f20 / _f50 are one experiment swept over a dilution fraction, and
  s16_infershard_qwen72b_kv1k / _kv8k / _kv32k / _kv32k_b8 / _b16 / _b32 /
  _out2k are one vLLM cell swept over context and batch. A model that has seen
  f05 has effectively seen f20. Families strip the scope prefix too, so the same
  workload concept never straddles the split by appearing on two platforms.

* **train / val / test = 60 / 20 / 20 of FAMILIES**, stratified by each family's
  dominant class so every split holds all three. Hyperparameters are chosen on
  val and reported on test, and the whole split is repeated over several seeds --
  with only ~30 families the effective sample size is families, not windows, so a
  single split's test accuracy is close to meaningless. The spread across seeds
  IS the result.

  PYTHONPATH=. python3 classifier/rebuttal_analysis/multinode_family_holdout.py
"""
import argparse
import datetime
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from classifier.threeway_improved import get_feature_cols  # noqa: E402
from sklearn.ensemble import (  # noqa: E402
    RandomForestClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier)
from sklearn.metrics import (  # noqa: E402
    accuracy_score, f1_score, confusion_matrix, classification_report)

SCOPE = re.compile(r"^(?:smoke_)?(s16|s8|s1|tpuT4|tpuT8|tpuT16|tpuTMS4|tpuTMS)_")
ADV = re.compile(r"(advtrain|adversarial|whitebox|symgemm|typeconf|stalled)", re.I)

# Variants of one experiment that the token rule below would otherwise split.
MERGE = {
    "ddpkaccum": "ddp", "ddpstagger": "ddp",   # DDP with accumulation / stagger
    "lp": "lowpower",                          # nonml_lp_d50 is nonml_lowpower
    "filler": "satfiller",                     # filler_only / satfiller_only
    "vllm": "qwen",                            # infershard_vllm_tp8pp2 is qwen
    "flatbubble": "flatarena",                 # flat arena + bubble filler
}


def family(grp):
    """Workload family: class + experiment stem, scope and parameters removed."""
    s = SCOPE.sub("", str(grp))
    parts = s.split("_")
    cls = parts[0]
    base = parts[1] if len(parts) > 1 else ""
    base = re.sub(r"\d+(_\d+)?[bkmg]?$", "", base)     # fsdp1b -> fsdp, qwen72b -> qwen
    base = MERGE.get(base, base)
    return f"{cls}_{base}" if base else cls


def load_multinode(win_path, inv_path):
    w = pd.read_parquet(win_path)
    inv = pd.read_parquet(inv_path, columns=["stream", "host"])
    w["scope"] = w.grp.map(lambda l: (SCOPE.match(str(l)).group(1)
                                      if SCOPE.match(str(l)) else "legacy"))
    w = w.merge(inv.drop_duplicates("stream").rename(columns={"stream": "run_id"}),
                on="run_id", how="left")
    hosts = w.groupby("job_id")["host"].nunique()
    w["hosts"] = w.job_id.map(hosts).fillna(1)
    mn = w[np.where(w.scope.isin(["s1", "s8"]), False,
                    (w.hosts > 1) | (w.scope == "s16"))].reset_index(drop=True)
    mn["family"] = mn.grp.map(family)
    mn["kind"] = np.where(mn.workload_label.str.contains(ADV, na=False),
                          "adversarial", "honest")
    return mn


def split_families(fams, dom, rng, frac=(0.6, 0.2, 0.2)):
    """Stratify families by their dominant class so every split holds all three."""
    tr, va, te = [], [], []
    for cls in sorted(set(dom.values())):
        f = sorted([x for x in fams if dom[x] == cls])
        rng.shuffle(f)
        n = len(f)
        n_tr = max(1, int(round(frac[0] * n)))
        n_va = max(1, int(round(frac[1] * n))) if n - n_tr >= 2 else 0
        tr += f[:n_tr]
        va += f[n_tr:n_tr + n_va]
        te += f[n_tr + n_va:]
    return set(tr), set(va), set(te)


def models():
    return {
        "rf200": RandomForestClassifier(n_estimators=200, n_jobs=-1, random_state=0),
        "rf200_leaf5": RandomForestClassifier(n_estimators=200, min_samples_leaf=5,
                                              n_jobs=-1, random_state=0),
        "rf200_d20": RandomForestClassifier(n_estimators=200, max_depth=20,
                                            n_jobs=-1, random_state=0),
        "rf200_bal": RandomForestClassifier(n_estimators=200, class_weight="balanced",
                                            n_jobs=-1, random_state=0),
        "rf300_mf03": RandomForestClassifier(n_estimators=300, max_features=0.3,
                                             min_samples_leaf=2, n_jobs=-1,
                                             random_state=0),
        "et300": ExtraTreesClassifier(n_estimators=300, n_jobs=-1, random_state=0),
        "hgb": HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                              random_state=0),
    }


def job_accuracy(df, pred):
    """Majority vote per job -- what an operator watching one execution sees."""
    t = pd.DataFrame({"job_id": df.job_id.values, "y": df.threeway_label.values,
                      "p": pred})
    v = t.groupby("job_id").agg(y=("y", lambda s: s.mode().iat[0]),
                                p=("p", lambda s: s.mode().iat[0]))
    return accuracy_score(v.y, v.p), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="/var/tmp/spar_analysis/win_stream_30_15.parquet")
    ap.add_argument("--inventory", default="/var/tmp/spar_analysis/inv4_dedup.parquet")
    ap.add_argument("--seeds", type=int, default=8)
    ap.add_argument("--max-train", type=int, default=0,
                    help="subsample training windows (0 = use all)")
    ap.add_argument("--out",
                    default="classifier/rebuttal_analysis/results/multinode_family_holdout.json")
    a = ap.parse_args()

    mn = load_multinode(a.windows, a.inventory)
    cols = get_feature_cols(mn)
    X = np.nan_to_num(mn[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = mn.threeway_label.values
    fam = mn.family.values
    dom = mn.groupby("family").threeway_label.agg(lambda s: s.mode().iat[0]).to_dict()
    fams = sorted(dom)

    print(f"multi-node windows {len(mn):,}  jobs {mn.job_id.nunique()}  "
          f"streams {mn.run_id.nunique()}  workloads {mn.grp.nunique()}  "
          f"families {len(fams)}  features {len(cols)}")
    print(mn.threeway_label.value_counts().to_string())
    print("\nfamilies (dominant class, workloads, windows):")
    for f in fams:
        s = mn[mn.family == f]
        print(f"  {f:26s} {dom[f]:13s} {s.grp.nunique():3d} wkl {len(s):7,d} win")

    rows, per_seed, fam_acc = [], [], []
    for seed in range(a.seeds):
        rng = np.random.default_rng(seed)
        tr_f, va_f, te_f = split_families(fams, dom, rng)
        itr = np.isin(fam, list(tr_f))
        iva = np.isin(fam, list(va_f))
        ite = np.isin(fam, list(te_f))
        if len(set(y[ite])) < 2 or len(set(y[itr])) < 3:
            print(f"seed {seed}: degenerate split, skipped")
            continue
        Xtr, ytr = X[itr], y[itr]
        if a.max_train and len(Xtr) > a.max_train:
            keep = rng.choice(len(Xtr), a.max_train, replace=False)
            Xtr, ytr = Xtr[keep], ytr[keep]
        best = None
        for name, clf in models().items():
            clf.fit(Xtr, ytr)
            va = accuracy_score(y[iva], clf.predict(X[iva]))
            pte = clf.predict(X[ite])
            te = accuracy_score(y[ite], pte)
            jte, njob = job_accuracy(mn[ite], pte)
            rows.append(dict(seed=seed, model=name, val_acc=va, test_acc=te,
                             test_job_acc=jte, n_test_jobs=njob,
                             test_macro_f1=f1_score(y[ite], pte, average="macro")))
            print(f"  seed {seed} {name:12s} val {va:.4f}  test {te:.4f}  "
                  f"test-job {jte:.4f}", flush=True)
            if best is None or va > best["val_acc"]:
                best = dict(rows[-1]); best["pred"] = pte; best["idx"] = ite
        sel = {k: v for k, v in best.items() if k not in ("pred", "idx")}
        sub = mn[best["idx"]]
        for kind in ("honest", "adversarial"):
            m = (sub.kind == kind).values
            sel[f"test_acc_{kind}"] = (accuracy_score(y[best['idx']][m], best["pred"][m])
                                       if m.any() else float("nan"))
        sel["n_train_fams"], sel["n_val_fams"], sel["n_test_fams"] = \
            len(tr_f), len(va_f), len(te_f)
        sel["test_families"] = sorted(te_f)
        # Per-family accuracy, accumulated across seeds. With only ~4 families in
        # any one test split, the split-level number swings hard on WHICH families
        # landed there; this says which ones are actually hard.
        for f in sorted(te_f):
            m = (sub.family == f).values
            if m.any():
                fam_acc.append(dict(family=f, seed=seed, n=int(m.sum()),
                                    cls=dom[f],
                                    acc=accuracy_score(y[best["idx"]][m],
                                                       best["pred"][m])))
        per_seed.append(sel)
        print(f"  seed {seed} SELECTED {sel['model']} -> test {sel['test_acc']:.4f} "
              f"(honest {sel['test_acc_honest']:.4f} / "
              f"adversarial {sel['test_acc_adversarial']:.4f})", flush=True)

    df = pd.DataFrame(rows)
    ps = pd.DataFrame(per_seed)
    print("\n== mean val / test accuracy per model, across seeds ==")
    print(df.groupby("model")[["val_acc", "test_acc", "test_job_acc", "test_macro_f1"]]
          .agg(["mean", "std"]).round(4).to_string())

    print("\n== SELECTED-ON-VAL model, test-set statistics across "
          f"{len(ps)} family splits ==")
    for c in ("test_acc", "test_job_acc", "test_macro_f1",
              "test_acc_honest", "test_acc_adversarial"):
        v = ps[c].dropna().values
        print(f"  {c:22s} mean {v.mean():.4f}  sd {v.std(ddof=1):.4f}  "
              f"min {v.min():.4f}  med {np.median(v):.4f}  max {v.max():.4f}")
    fa = pd.DataFrame(fam_acc)
    print("\n== per-family test accuracy (pooled over the seeds that held it out) ==")
    g = fa.groupby(["cls", "family"]).agg(times_held_out=("seed", "nunique"),
                                          windows=("n", "mean"),
                                          acc_mean=("acc", "mean"),
                                          acc_min=("acc", "min"),
                                          acc_max=("acc", "max"))
    for (c, f), r in g.sort_values("acc_mean").iterrows():
        print(f"  {c:13s} {f:24s} held out {int(r.times_held_out)}x  "
              f"acc {r.acc_mean:.4f} [{r.acc_min:.4f}, {r.acc_max:.4f}]  "
              f"n~{int(r.windows):,}")
    print(f"\n  model chosen: {ps.model.value_counts().to_dict()}")
    print(f"  val-to-test drop: mean {(ps.val_acc - ps.test_acc).mean():+.4f}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump(dict(generated_utc=datetime.datetime.utcnow().isoformat() + "Z",
                       protocol="multi-node only; family-level 60/20/20 holdout, "
                                "stratified by dominant class; tuned on val, "
                                "reported on test; repeated over seeds",
                       n_windows=int(len(mn)), n_jobs=int(mn.job_id.nunique()),
                       n_workloads=int(mn.grp.nunique()), n_families=len(fams),
                       families={f: dom[f] for f in fams},
                       grid=list(models().keys()),
                       per_seed=ps.to_dict("records"),
                       per_family=fa.to_dict("records"),
                       all_runs=df.to_dict("records")),
                  open(a.out, "w"), indent=2, default=str)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
