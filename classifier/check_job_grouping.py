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
    # Coerce group labels to a uniform <U string array. pandas>=3 astype(str)
    # can hand back an arrow string dtype that reintroduces a float-like NA when
    # sklearn calls np.asarray, and numpy>=2 refuses to sort a mixed str/float
    # array inside StratifiedGroupKFold. A plain numpy string array sorts fine.
    n_groups = pd.Series(groups).nunique()
    groups = np.asarray(groups).astype(str)
    skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
    accs, f1s = [], []
    yp_all = np.empty(len(y), dtype=int)
    for tr, te in skf.split(X, y, groups):
        c = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                   max_features="sqrt", class_weight="balanced",
                                   random_state=42, n_jobs=-1).fit(X[tr], y[tr])
        p = c.predict(X[te]); yp_all[te] = p
        accs.append(accuracy_score(y[te], p)); f1s.append(f1_score(y[te], p, zero_division=0))
    print(f"[{tag}] groups={n_groups}  acc={np.mean(accs):.4f}±{np.std(accs):.4f}  "
          f"f1={np.mean(f1s):.4f}")
    return yp_all


def main():
    df = load_all_data()
    win = sliding_windows(df, 30, 15, run_level_mode="causal").reset_index(drop=True)
    feats = get_feature_cols(win)

    # physical job id
    run_start = win.groupby("run_id")["window_start"].transform("min")
    # Strip every per-stream rank token so all ranks of one physical job share a
    # base label: multi-node GPU (_node1_gpu3), single-node GPU (_gpu3), and TPU
    # multi-host (_w2c1) conventions all collapse to the workload's base name.
    base = (win["workload_label"].astype(str)
            .str.replace(r"_node\d+_gpu\d+", "", regex=True)
            .str.replace(r"_gpu\d+", "", regex=True)
            .str.replace(r"_w\d+c\d+", "", regex=True))
    # fillna + final astype(str) keep the key all-string; a stray NaN
    # workload_label/gpu_name would otherwise make the group column mixed
    # str/float, which numpy>=2 refuses to sort inside StratifiedGroupKFold.
    job = (base.fillna("na") + "|" + win["gpu_name"].astype(str) + "|"
           + (run_start.fillna(-1) / BUCKET).round().astype("int64").astype(str))
    win["run_grp"] = win["run_id"].astype(str)
    win["job_grp"] = job.astype(str)

    n_run = win["run_grp"].nunique(); n_job = win["job_grp"].nunique()
    print(f"windows={len(win)}  run_id groups={n_run}  job groups={n_job}  "
          f"(merged {n_run-n_job} sibling run_ids into jobs)")

    yp_run = cv(win, feats, win["run_grp"].rename("grp"), "group=run_id (paper)")
    yp_job = cv(win, feats, win["job_grp"].rename("grp"), "group=job    (fixed)")

    # multi-GPU family detection under each grouping
    mg = win["workload_label"].astype(str).str.contains(
        r"2gpu|tp8|tp2|_ddp|fsdp|gpu[0-7]|_node\d+|_w\d+c\d+",
        case=False, regex=True)
    tr = win["threeway_label"].values == "ml_training"
    for name, yp in [("run_id", yp_run), ("job", yp_job)]:
        m = mg.values & tr
        s = (~mg.values) & tr
        print(f"  [{name}] multiGPU-training recall={ (yp[m]==1).mean():.4f} (n={m.sum()})  "
              f"singleGPU-training recall={ (yp[s]==1).mean():.4f} (n={s.sum()})")


if __name__ == "__main__":
    main()
