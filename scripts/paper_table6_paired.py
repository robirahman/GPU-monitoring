#!/usr/bin/env python3
"""Table 6 again, with the RF-vs-XGBoost comparison done as a paired test.

The published table compares four classifiers on ONE fold assignment. Under
job-level holdout that is not enough to rank them: whole physical executions
move between folds together, so a single hard job landing in one fold moves
that fold's accuracy by several points. The first regeneration showed exactly
that -- RF 97.39% vs XGB 98.70%, with four of five folds within 0.5 pp and the
entire gap in fold 3 (RF 89.84%, XGB 96.39%).

So: repeat the whole thing over several fold-assignment seeds, score every
classifier on the SAME split each time, and compare RF and XGB pairwise rather
than by comparing two means with overlapping spreads.

    PYTHONPATH=. python3 scripts/paper_table6_paired.py --seeds 10
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from classifier.threeway_improved import get_feature_cols  # noqa: E402
CACHE = ROOT / "cache"


def estimators(fast=False):
    out = [("RandomForest", lambda s: RandomForestClassifier(
                n_estimators=400, class_weight="balanced", random_state=42, n_jobs=-1))]
    try:
        from xgboost import XGBClassifier
        out.append(("XGBoost", lambda s: XGBClassifier(
            n_estimators=400, max_depth=6, learning_rate=0.1, n_jobs=-1,
            random_state=42, eval_metric="logloss", tree_method="hist")))
    except Exception as exc:
        print(f"  xgboost unavailable: {exc}", flush=True)
    if not fast:
        out += [
            ("SVM-RBF", lambda s: make_pipeline(StandardScaler(),
                SVC(kernel="rbf", class_weight="balanced", random_state=42, cache_size=1000))),
            ("LogisticRegression", lambda s: make_pipeline(StandardScaler(),
                LogisticRegression(max_iter=2000, class_weight="balanced", random_state=42))),
        ]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--group-by", default="job_id")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--fast", action="store_true", help="RF and XGB only")
    ap.add_argument("--out", default=str(ROOT / "docs/camera_ready_results_20261022/paper_table6_paired.json"))
    a = ap.parse_args()
    t0 = time.time()

    win = pd.read_parquet(CACHE / f"win_{int(a.window)}.parquet")
    feats = [c for c in get_feature_cols(win)]
    X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    y = (win.threeway_label == "ml_training").to_numpy(int)
    g = win[a.group_by].astype(str).to_numpy()
    cell = win.workload_label.astype(str).to_numpy()
    print(f"{len(win):,} windows, {len(feats)} features, "
          f"{len(np.unique(g)):,} {a.group_by} groups, "
          f"{y.mean()*100:.1f}% training", flush=True)

    ests = estimators(a.fast)
    seeds = [42, 7, 1234, 2026, 99, 31337, 8, 555, 101, 2718][:a.seeds]
    rows = []            # one record per (seed, fold, classifier)
    worst = []           # per-fold error attribution for the paired pair
    for sd in seeds:
        cv = StratifiedGroupKFold(n_splits=a.folds, shuffle=True, random_state=sd)
        for k, (tr, te) in enumerate(cv.split(X, y, g)):
            preds = {}
            for name, make in ests:
                m = make(sd).fit(X[tr], y[tr])
                p = m.predict(X[te])
                preds[name] = p
                rows.append({"seed": sd, "fold": k, "classifier": name,
                             "accuracy": float((p == y[te]).mean()),
                             "macro_f1": float(f1_score(y[te], p, average="macro")),
                             "n_test": int(len(te)),
                             "n_groups_test": int(len(np.unique(g[te])))})
            if "RandomForest" in preds and "XGBoost" in preds:
                rf, xg = preds["RandomForest"], preds["XGBoost"]
                only_rf = (rf != y[te]) & (xg == y[te])    # XGB right, RF wrong
                only_xg = (xg != y[te]) & (rf == y[te])    # RF right, XGB wrong
                top = pd.Series(cell[te][only_rf]).value_counts().head(3).to_dict()
                worst.append({"seed": sd, "fold": k,
                              "acc_rf": float((rf == y[te]).mean()),
                              "acc_xgb": float((xg == y[te]).mean()),
                              "xgb_only_right": int(only_rf.sum()),
                              "rf_only_right": int(only_xg.sum()),
                              "top_cells_xgb_fixes": top})
        print(f"  seed {sd} done [{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== Table 6, mean over "
          f"{len(seeds)} fold-assignment seeds x {a.folds} folds ===")
    summ = (df.groupby("classifier")
              .agg(acc_mean=("accuracy", "mean"), acc_sd=("accuracy", "std"),
                   f1_mean=("macro_f1", "mean"), f1_sd=("macro_f1", "std"),
                   acc_min=("accuracy", "min"), acc_max=("accuracy", "max"))
              .sort_values("acc_mean", ascending=False))
    print((summ * 100).round(2).to_string())

    # seed-level means: 10 quasi-independent replicates of the whole experiment
    per_seed = df.groupby(["classifier", "seed"]).accuracy.mean().unstack()
    print(f"\n=== per-seed mean accuracy (each is a full {a.folds}-fold CV) ===")
    print((per_seed * 100).round(2).to_string())

    out = {"table": 6, "window_sec": a.window, "group_by": a.group_by,
           "folds": a.folds, "seeds": seeds, "windows": int(len(win)),
           "n_features": len(feats), "records": rows,
           "summary": {k: {kk: float(vv) for kk, vv in v.items()}
                       for k, v in summ.to_dict("index").items()}}

    if "RandomForest" in per_seed.index and "XGBoost" in per_seed.index:
        rf_s, xg_s = per_seed.loc["RandomForest"], per_seed.loc["XGBoost"]
        d_seed = (xg_s - rf_s).to_numpy()
        pf = df.pivot_table(index=["seed", "fold"], columns="classifier", values="accuracy")
        d_fold = (pf["XGBoost"] - pf["RandomForest"]).to_numpy()
        n = len(d_seed)
        se = d_seed.std(ddof=1) / np.sqrt(n)
        t = d_seed.mean() / se if se > 0 else float("inf")
        try:
            from scipy import stats
            p = float(stats.ttest_rel(xg_s, rf_s).pvalue)
            w = float(stats.wilcoxon(xg_s, rf_s).pvalue) if n >= 6 else float("nan")
        except Exception:
            p = w = float("nan")
        print("\n=== PAIRED XGBoost - RandomForest ===")
        print(f"  per-seed difference  mean {d_seed.mean()*100:+.3f} pp  "
              f"sd {d_seed.std(ddof=1)*100:.3f}  "
              f"95% CI [{(d_seed.mean()-1.96*se)*100:+.3f}, {(d_seed.mean()+1.96*se)*100:+.3f}] pp")
        print(f"  XGB better in {int((d_seed>0).sum())}/{n} seeds, "
              f"{int((d_fold>0).sum())}/{len(d_fold)} individual folds")
        print(f"  paired t = {t:.2f}, p = {p:.4f}   Wilcoxon p = {w:.4f}")
        print(f"  per-fold difference  mean {d_fold.mean()*100:+.3f} pp  "
              f"sd {d_fold.std(ddof=1)*100:.3f}  "
              f"min {d_fold.min()*100:+.2f}  max {d_fold.max()*100:+.2f} pp")
        big = np.abs(d_fold) > 0.01
        print(f"  folds where |difference| > 1 pp: {int(big.sum())}/{len(d_fold)}")
        out["paired_xgb_minus_rf"] = {
            "per_seed_mean_pp": float(d_seed.mean()*100),
            "per_seed_sd_pp": float(d_seed.std(ddof=1)*100),
            "ci95_pp": [float((d_seed.mean()-1.96*se)*100), float((d_seed.mean()+1.96*se)*100)],
            "seeds_xgb_better": int((d_seed>0).sum()), "n_seeds": n,
            "folds_xgb_better": int((d_fold>0).sum()), "n_folds": int(len(d_fold)),
            "t": float(t), "p_ttest": p, "p_wilcoxon": w,
            "per_fold_mean_pp": float(d_fold.mean()*100),
            "per_fold_sd_pp": float(d_fold.std(ddof=1)*100),
            "folds_over_1pp": int(big.sum())}
        wd = pd.DataFrame(worst)
        out["fold_detail"] = worst
        print("\n=== the five worst folds for RandomForest ===")
        ww = wd.nsmallest(5, "acc_rf")
        for _, r in ww.iterrows():
            print(f"  seed {r.seed:5d} fold {r.fold}  RF {r.acc_rf*100:6.2f}%  "
                  f"XGB {r.acc_xgb*100:6.2f}%   XGB fixes {r.xgb_only_right:5d} windows; "
                  f"top cells {list(r.top_cells_xgb_fixes)[:3]}")
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    print(f"\nwrote {a.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
