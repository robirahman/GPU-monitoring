"""Quantify whether multi-GPU sibling streams splitting across CV folds inflates
detection. Re-runs the binary grouped 5-fold CV twice — grouped by run_id (one
GPU's telemetry file, the paper's protocol) and grouped by physical JOB (sibling
per-GPU streams of one execution merged) — and compares overall accuracy and the
multi-GPU-family detection.

A physical job is identified as (base workload label with any _gpuN stripped,
GPU model, run start time bucketed) — multi-GPU siblings share all three; repeated
runs differ in start time.
"""
from __future__ import annotations
import re
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score

from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols

BUCKET = 20.0  # s; siblings start simultaneously, reps are far apart


def cv(win, feats, groups, tag):
    X = win[feats].fillna(0).values.astype(np.float32)
    y = (win["threeway_label"].values == "ml_training").astype(int)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    yp_all = np.empty(len(y), dtype=int)
    for tr, te in skf.split(X, y, groups):
        c = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(X[tr], y[tr])
        p = c.predict(X[te]); yp_all[te] = p
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, zero_division=0))
    print(f"[{tag}] groups={groups.nunique()}  acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
          f"f1={np.mean(f1s):.4f}")
    return yp_all


def main():
    df = load_all_data()
    win = sliding_windows(df, 30, 15, run_level_mode="causal").reset_index(drop=True)
    feats = get_feature_cols(win)

    # physical job id
    run_start = win.groupby("run_id")["window_start"].transform("min")
    base = win["workload_label"].astype(str).str.replace(r"_gpu\d+", "", regex=True)
    job = (base + "|" + win["gpu_name"].astype(str) + "|"
           + (run_start / BUCKET).round().astype(int).astype(str))
    win["run_grp"] = win["run_id"]
    win["job_grp"] = job

    n_run = win["run_grp"].nunique(); n_job = win["job_grp"].nunique()
    print(f"windows={len(win)}  run_id groups={n_run}  job groups={n_job}  "
          f"(merged {n_run-n_job} sibling run_ids into jobs)")

    yp_run = cv(win, feats, win["run_grp"].rename("grp"), "group=run_id (paper)")
    yp_job = cv(win, feats, win["job_grp"].rename("grp"), "group=job    (fixed)")

    # multi-GPU family detection under each grouping
    mg = win["workload_label"].astype(str).str.contains(r"2gpu|tp8|tp2|_ddp|fsdp|gpu[0-7]",
                                                         case=False, regex=True)
    tr = win["threeway_label"].values == "ml_training"
    for name, yp in [("run_id", yp_run), ("job", yp_job)]:
        m = mg.values & tr
        s = (~mg.values) & tr
        print(f"  [{name}] multiGPU-training recall={ (yp[m]==1).mean():.4f} (n={m.sum()})  "
              f"singleGPU-training recall={ (yp[s]==1).mean():.4f} (n={s.sum()})")


if __name__ == "__main__":
    main()
