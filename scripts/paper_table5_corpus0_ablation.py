#!/usr/bin/env python3
"""How much of the 0.7% Round 6 baseline rests on the synthetic GEMM ladder?

The option (b) baseline detector is fit on corpus_0, which now contains the
symmetric-GEMM negatives. If those cells are the reason L4 is invisible, the
headline is contingent on a judgement call about whether a sustained synthetic
GEMM benchmark is an organic workload. This prices that call.
"""
import sys, json, time
from pathlib import Path
sys.path.insert(0, "."); sys.path.insert(0, "scripts")
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from paper_table5_armsrace import family_of, FAMILY_SPEC, DROP_CELLS, out_of_scope_runs
from classifier.threeway_improved import get_feature_cols
from classifier.workload_labels import canonical_workload

t0 = time.time()
win = pd.read_parquet("cache/win_30_r6.parquet")
win["cell"] = [canonical_workload(l) for l in win.workload_label]
oos, _ = out_of_scope_runs(8, True)
win = win[~win.run_id.astype(str).isin(oos) & ~win.cell.isin(DROP_CELLS)].reset_index(drop=True)
win["family"] = [family_of(l) for l in win.workload_label]
for f, (r, e) in list(FAMILY_SPEC.items()):          # option (b) placement
    if not e:
        FAMILY_SPEC[f] = (0, False)
rno = win.family.map(lambda f: FAMILY_SPEC[f][0] if isinstance(f, str) else 0).astype(int).to_numpy()
fam = win.family.to_numpy(object); cell = win.cell.to_numpy(object)
feats = [c for c in get_feature_cols(win) if c not in {"round","cell","family","subgroup"}]
X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
y = (win.threeway_label == "ml_training").to_numpy(int)

c0 = rno == 0
L4 = (fam == "symgemm_L4_fused_sgd") & (y == 1)
L1 = pd.Series(cell).str.startswith("symgemm_L1").to_numpy()
L3 = pd.Series(cell).str.startswith("symgemm_L3").to_numpy()
L0 = pd.Series(cell).str.startswith("symgemm_L0").to_numpy()
NEG = (fam == "symgemm_negatives")
print(f"corpus_0={int(c0.sum()):,}  L4={int(L4.sum())}  L0={int(L0.sum())} "
      f"L1={int(L1.sum())} L3={int(L3.sum())} all-negatives={int(NEG.sum())}")

def forest():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=-1)

conds = [("corpus_0 as run in option (b)", c0),
         ("minus L3_graddiscard", c0 & ~L3),
         ("minus the L0/L1/L3 synthetic ladder", c0 & ~(L0 | L1 | L3)),
         ("minus every symgemm negative", c0 & ~NEG)]
out = []
for name, tr in conds:
    clf = forest().fit(X[tr], y[tr])
    det = float((clf.predict(X[L4]) == 1).mean())
    l1_in = L1 & tr
    fp = float((clf.predict(X[L1 & ~tr]) == 1).mean()) if int((L1 & ~tr).sum()) else \
         float((clf.predict(X[L1]) == 1).mean())
    out.append({"corpus_0": name, "train_windows": int(tr.sum()),
                "L4_baseline_detection": det, "L1_flagged_as_training": fp})
    print(f"  L4 {det*100:5.1f}%   L1-as-training {fp*100:4.1f}%   {name} "
          f"(train {int(tr.sum()):,}w) [{time.time()-t0:.0f}s]", flush=True)
Path("docs/camera_ready_results_20261022/paper_table5_corpus0_ablation.json").write_text(
    json.dumps({"target": "symgemm_L4_fused_sgd", "conditions": out}, indent=2) + "\n")
print("\nwrote docs/camera_ready_results_20261022/paper_table5_corpus0_ablation.json")
