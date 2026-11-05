"""How much of the Round 6 baseline result rests on L3 being in corpus_0?

L3_graddiscard runs the dW GEMM and discards it. Nobody runs that for its own
sake, so under "the baseline corpus is workloads people organically want to
run" it is an instrument, not a corpus member. This prices that distinction.
"""
import sys, json, time; sys.path.insert(0, "."); sys.path.insert(0, "scripts")
from pathlib import Path
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
# option (b): every non-evasion family lives in corpus_0
ev = win.family.map(lambda f: isinstance(f, str) and FAMILY_SPEC[f][1])
feats = [c for c in get_feature_cols(win) if c not in {"round","cell","family","subgroup"}]
X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
y = (win.threeway_label == "ml_training").to_numpy(int)
cell = win.cell.to_numpy(object); fam = win.family.to_numpy(object)
c0 = (~ev).to_numpy()

def forest():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=-1)

L3 = cell == "symgemm_L3_graddiscard"
LADDER = np.isin(cell, ["symgemm_L3_graddiscard", "symgemm_L0_fwd1", "symgemm_L1_fwd3",
                        "symgemm_L1_fwd3_7b", "symgemm_L1_fwd3_m8192"])
NEG = fam == "symgemm_negatives"
CONDS = [("corpus_0 as run in option (b)", c0),
         ("corpus_0 minus L3_graddiscard", c0 & ~L3),
         ("corpus_0 minus the L0/L1/L3 synthetic ladder", c0 & ~LADDER),
         ("corpus_0 minus every symgemm negative", c0 & ~NEG)]
TARGETS = [("symgemm_L4_fused_sgd", 1), ("symgemm_ppo_fused", 1)]
out = []
for name, tr in CONDS:
    clf = forest().fit(X[tr], y[tr])
    row = {"condition": name, "train_windows": int(tr.sum())}
    for f, cls in TARGETS:
        sel = (fam == f) & (y == cls)
        row[f] = float((clf.predict(X[sel]) == 1).mean())
    # and the false-positive rate the detector shows on L1, the FLOP-matched negative
    sel = np.isin(cell, ["symgemm_L1_fwd3", "symgemm_L1_fwd3_7b", "symgemm_L1_fwd3_m8192"])
    row["L1_false_positive"] = float((clf.predict(X[sel]) == 1).mean())
    out.append(row)
    print(f"  {name:46s} train={row['train_windows']:6,}  "
          f"L4 {row['symgemm_L4_fused_sgd']*100:5.1f}%  "
          f"ppo_fused {row['symgemm_ppo_fused']*100:5.1f}%  "
          f"L1-FP {row['L1_false_positive']*100:5.1f}%  [{time.time()-t0:.0f}s]", flush=True)
Path("docs/camera_ready_results_20261022/paper_table5_l3_sensitivity.json").write_text(
    json.dumps({"conditions": out}, indent=2) + "\n")
print("\nwrote docs/camera_ready_results_20261022/paper_table5_l3_sensitivity.json")
