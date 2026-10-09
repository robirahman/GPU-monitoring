#!/usr/bin/env python3
"""Leave-one-FAMILY-out on honest multi-node data: train vs not-train.

The question is not "what accuracy can we report" but "does the detector work on
a training program it has never seen". Family-grouped holdout is what measures
that, and the number of folds on the TRAIN side is the number of independent
training programs the corpus owns -- which, as of 2026-09-22, is two:
train_fsdp (every FSDP rung on GPU and TPU, one program swept over size, batch,
sequence length and activation checkpointing) and train_ddp (ResNet-50).

Disguise cells are excluded on purpose. They are a separate claim; mixing them in
lets a fold "pass" on adversarial windows while failing on the honest ones.

Two feature sets are scored side by side:

  all    every window feature
  shape  absolute-level features dropped (stage2_mask_multinode), so the model
         cannot win by memorising a workload's power / memory footprint. The
         identity audit showed level features alone fingerprint workloads at
         88.9%, so a fold that only survives with `all` is memorisation, not
         detection.

  PYTHONPATH=. python3 classifier/rebuttal_analysis/mn_binary_lofo.py \
      --windows /var/tmp/spar_analysis/win_full.parquet \
      --inventory /var/tmp/spar_analysis/inv_mn.parquet
"""
import argparse
import functools
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from classifier.threeway_improved import get_feature_cols  # noqa: E402
from classifier.multinode_features import stage2_mask_multinode  # noqa: E402
from multinode_family_holdout import family, SCOPE, ADV  # noqa: E402
from sklearn.ensemble import RandomForestClassifier  # noqa: E402
from sklearn.metrics import accuracy_score  # noqa: E402

print = functools.partial(print, flush=True)


def job_vote(sub, pred):
    """Majority vote per job -- what an operator watching one execution sees."""
    t = pd.DataFrame({"j": sub.job_id.values, "y": sub.y.values, "p": pred})
    v = t.groupby("j").agg(y=("y", "max"), p=("p", "mean"))
    return float(((v.p > 0.5).astype(int) == v.y).mean()), len(v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--windows", default="/var/tmp/spar_analysis/win_full.parquet")
    ap.add_argument("--inventory", default="/var/tmp/spar_analysis/inv_mn.parquet")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--max-reps", type=int, default=0,
                    help="keep only the first N reps of each workload in the "
                         "TRAINING half of every fold (0 = all). The held-out "
                         "family is always evaluated in full, so the comparison "
                         "isolates what extra reps buy the model rather than "
                         "also shrinking the test set. Reps are ordered by first "
                         "window, which is the collection order.")
    ap.add_argument("--out", default="/var/tmp/spar_analysis/mn_binary_lofo.csv")
    ap.add_argument("--taus", default="0.8,0.5,0.3,0.2,0.1,0.05",
                    help="decision thresholds to report per fold. A single "
                         "accuracy at 0.5 hides the whole trade-off: held-out "
                         "train_fsdp recall is 0.005 at 0.5 and 0.44 at 0.05, "
                         "and the question is what the false-alarm rate does "
                         "over the same range.")
    a = ap.parse_args()
    taus = [float(t) for t in a.taus.split(",")]

    w = pd.read_parquet(a.windows)
    inv = pd.read_parquet(a.inventory, columns=["stream", "host"]).drop_duplicates("stream")
    w["scope"] = w.grp.map(lambda l: (SCOPE.match(str(l)).group(1)
                                      if SCOPE.match(str(l)) else "legacy"))
    w = w.merge(inv.rename(columns={"stream": "run_id"}), on="run_id", how="left")
    hosts = w.groupby("job_id")["host"].nunique()
    w["hosts"] = w.job_id.map(hosts).fillna(1)
    # s1_/s8_ are per-node by construction: multinode_queue.sh derives run_id from
    # (stamp, cell, rep) with no node component, so both machines run independent
    # single-node jobs under one run_id and a naive host count calls them
    # multi-node. s16 genuinely spans hosts; for TPU the host count is measured,
    # because chips-per-host differs by generation.
    mn = w[np.where(w.scope.isin(["s1", "s8"]), False,
                    (w.hosts > 1) | (w.scope == "s16"))].copy()
    mn["family"] = mn.grp.map(family)
    mn = mn[~mn.workload_label.str.contains(ADV, na=False)].reset_index(drop=True)

    # Feature columns BEFORE the binary label exists. get_feature_cols() excludes
    # threeway_label by name but nothing else, so creating mn["y"] first puts the
    # TARGET in the matrix as an int64 column -- which is exactly how the first
    # run of this script scored 1.0000 on all twelve folds, shape-only included.
    # `hosts`/`host` come from the inventory merge and are topology metadata, not
    # telemetry, so they go too.
    cols = [c for c in get_feature_cols(mn) if c not in ("y", "hosts", "host")]
    mn["y"] = (mn.threeway_label == "ml_training").astype(int)

    # Rep index within a workload, by collection order. A rep is one execution
    # of one cell: multinode_queue.sh derives its run_id from (stamp, cell, rep),
    # so distinct jobs of the same grp ARE the reps.
    first = mn.groupby(["grp", "job_id"]).window_start.min().reset_index()
    first["rep"] = first.groupby("grp").window_start.rank(method="dense").astype(int)
    mn = mn.merge(first[["grp", "job_id", "rep"]], on=["grp", "job_id"], how="left")
    mn["rep"] = mn["rep"].fillna(1).astype(int)

    keep = stage2_mask_multinode(cols)
    X = np.nan_to_num(mn[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    y = mn.y.values
    fam = mn.family.values
    fams = sorted(set(fam))
    frac_train = mn.groupby("family").y.mean().to_dict()

    if a.max_reps:
        kept = int((mn["rep"] <= a.max_reps).sum())
        print(f"REP ABLATION: training half capped at {a.max_reps} rep(s) per "
              f"workload -> {kept:,} of {len(mn):,} windows available to fit")
    print(f"honest multi-node windows {len(mn):,}  jobs {mn.job_id.nunique()}  "
          f"workloads {mn.grp.nunique()}  families {len(fams)}  "
          f"features {len(cols)} (shape-only {int(keep.sum())})")
    print(f"  TRAIN windows {int(y.sum()):,}   "
          f"NOT-TRAIN windows {int((1 - y).sum()):,}")
    print("\nfamily                      side        wkl    windows   jobs")
    for f in fams:
        s = mn[mn.family == f]
        print(f"  {f:26s} {'TRAIN' if frac_train[f] > .5 else 'not-train':10s} "
              f"{s.grp.nunique():3d} {len(s):9,d} {s.job_id.nunique():6d}")

    print("\n=== leave-one-family-out ===")
    print(f"{'held-out family':26s} {'side':10s} {'win-acc(all)':>12s} "
          f"{'win-acc(shape)':>14s} {'job-acc(all)':>12s} {'train fams left':>16s}")
    res = []
    for f in fams:
        te = fam == f
        tr = ~te
        if a.max_reps:
            tr = tr & (mn["rep"].values <= a.max_reps)
        if len(set(y[tr])) < 2:
            print(f"  {f:26s} skipped: holding it out leaves one class")
            continue
        n_tr_fam = len({g for g in fams if g != f and frac_train[g] > .5})
        row = {"family": f, "side": "TRAIN" if frac_train[f] > .5 else "not-train",
               "n_train_fams_left": n_tr_fam, "n_win": int(te.sum())}
        for tag, m in (("all", np.ones(len(cols), bool)), ("shape", keep)):
            clf = RandomForestClassifier(n_estimators=a.trees, max_features=0.3,
                                         min_samples_leaf=2, n_jobs=-1,
                                         random_state=0)
            clf.fit(X[tr][:, m], y[tr])
            p = clf.predict(X[te][:, m])
            row[f"win_{tag}"] = accuracy_score(y[te], p)
            if tag == "all":
                row["job_all"], row["n_jobs"] = job_vote(mn[te], p)
                # Fraction of held-out windows called "training" at each tau.
                # Every fold is single-class, so on a TRAIN family this is
                # recall and on a not-train family it is the false-alarm rate --
                # read down a tau column to see whether any threshold serves
                # both at once.
                pr = clf.predict_proba(X[te][:, m])[:, list(clf.classes_).index(1)]
                row["p_train_median"] = float(np.median(pr))
                for t in taus:
                    row[f"flag@{t}"] = float((pr > t).mean())
        res.append(row)
        print(f"  {f:26s} {row['side']:10s} {row['win_all']:12.4f} "
              f"{row['win_shape']:14.4f} {row['job_all']:12.4f} {n_tr_fam:16d}")

    r = pd.DataFrame(res)
    print("\n=== fraction of held-out windows flagged as training, by tau ===")
    print("(TRAIN rows are recall -- want high; not-train rows are the "
          "false-alarm rate -- want low)")
    hdr = "".join(f"{('tau=' + str(t)):>10s}" for t in taus)
    print(f"{'held-out family':26s} {'side':10s} {'p_med':>7s}{hdr}")
    for _, x in r.sort_values(["side", "family"]).iterrows():
        cells = "".join(f"{x[f'flag@{t}']:10.4f}" for t in taus)
        print(f"  {x.family:24s} {x.side:10s} {x.p_train_median:7.3f}{cells}")
    print("\n--- summary ---")
    for side in ("TRAIN", "not-train"):
        s = r[r.side == side]
        if not len(s):
            continue
        print(f"  {side:10s} n={len(s)} families   "
              f"win-acc(all) mean {s.win_all.mean():.4f} min {s.win_all.min():.4f}   "
              f"win-acc(shape) mean {s.win_shape.mean():.4f} min {s.win_shape.min():.4f}")
    r.to_csv(a.out, index=False)
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
