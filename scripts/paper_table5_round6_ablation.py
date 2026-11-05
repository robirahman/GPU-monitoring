#!/usr/bin/env python3
"""Why does hardening collapse detection of the Round 6 attack?

The sequential table says the end-of-Round-5 detector -- which has never seen a
symmetric-GEMM run -- catches 81.5% of symgemm_L4 windows, while the detector
hardened on the whole Round 6 campaign except L4 itself catches 9.4%. Adding
Round 6 data to the corpus makes the Round 6 attack HARDER to see. This isolates
which addition does it, by adding one Round 6 group at a time to corpus_5.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, ".")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "scripts")
from paper_table5_armsrace import family_of, FAMILY_SPEC, DROP_CELLS
from classifier.threeway_improved import get_feature_cols
from classifier.workload_labels import canonical_workload

t0 = time.time()
win = pd.read_parquet("cache/win_30_r6.parquet")
win["cell"] = [canonical_workload(l) for l in win.workload_label]
win = win.loc[~win.cell.isin(DROP_CELLS)].reset_index(drop=True)
win["family"] = [family_of(l) for l in win.workload_label]
rno = win.family.map(lambda f: FAMILY_SPEC[f][0] if isinstance(f, str) else 0).astype(int).to_numpy()
fam = win.family.to_numpy(object)
feats = get_feature_cols(win)
X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
y = (win.threeway_label == "ml_training").to_numpy(int)

TARGET = "symgemm_L4_fused_sgd"
sel = (fam == TARGET) & (y == 1)
c5 = rno < 6
print(f"corpus_5={int(c5.sum()):,}  target windows={int(sel.sum())} "
      f"({win.loc[sel,'run_id'].nunique()} runs, {win.loc[sel,'cell'].nunique()} cells)")

def _model_kind():
    import os
    return os.environ.get("PAPER_MODEL", "rf").strip().lower()


def forest():
    """The classifier under test. PAPER_MODEL=xgb swaps in XGBoost.

    Both configurations are the ones Table 6 compared, so a stack run with
    PAPER_MODEL=xgb is comparable with that measurement rather than being a
    third tuning. Note XGBoost has no class_weight; Table 6 ran it without one
    and the corpus is 57% positive, so the imbalance correction is small.
    """
    import os
    nj = int(os.environ.get("PAPER_NJOBS", "0")) or -1
    if _model_kind() in ("xgb", "xgboost"):
        from xgboost import XGBClassifier
        return XGBClassifier(n_estimators=400, max_depth=6, learning_rate=0.1,
                             n_jobs=nj, random_state=42,
                             eval_metric="logloss", tree_method="hist")
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=nj)

R6_GROUPS = ["symgemm_negatives", "symgemm_L5_adam", "symgemm_ppo_fused",
             "ppo_rlhf", "rl_distributed"]
conds = [("corpus_5 only (no Round 6 data at all)", c5)]
for g in R6_GROUPS:
    conds.append((f"corpus_5 + {g}", c5 | (fam == g)))
conds.append(("corpus_5 + everything except symgemm_negatives",
              (rno <= 6) & (fam != TARGET) & (fam != "symgemm_negatives")))
conds.append(("corpus_6 minus L4 (the table's `unseen`)",
              (rno <= 6) & (fam != TARGET)))

out = []
for name, tr in conds:
    clf = forest().fit(X[tr], y[tr])
    p = clf.predict(X[sel])
    det = float((p == 1).mean())
    out.append({"condition": name, "train_windows": int(tr.sum()),
                "L4_detection": det})
    print(f"  {det*100:6.1f}%   {name}   (train {int(tr.sum()):,}w) "
          f"[{time.time()-t0:.0f}s]", flush=True)

Path("docs/camera_ready_results_20261022/paper_table5_round6_ablation.json").write_text(
    json.dumps({"target": TARGET, "window_sec": 30.0, "conditions": out}, indent=2) + "\n")
print("\nwrote docs/camera_ready_results_20261022/paper_table5_round6_ablation.json")
