"""Is the 7B failure fundamental, or just missing same-scale training data?

Every 7B result so far came from a classifier trained with ALL whitebox runs
excluded — so it had never seen a 7B workload of any kind, training or
inference. That is the correct unseen-strategy protocol for measuring evasion,
but it conflates two very different failures:

  (i)  the telemetry genuinely carries no signal at 7B, or
  (ii) the classifier has no same-scale examples, so its decision boundary is
       simply not defined in that region.

Ablation: score the same held-out 7B training run against the same held-out
half of the 7B inference control, under two training sets --
  A  no 7B data at all           (current protocol)
  B  plus the OTHER half of the 7B inference control, and the other 7B
     training runs except the target
If B >> A, the failure is coverage (fixable by calibration data), not physics.
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
sys.path.insert(0, "/root/GPU-monitoring")
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols

SP="/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad"
raw = load_all_data()
win = sliding_windows(raw, 30.0, 15.0, run_level_mode="causal")
feats = get_feature_cols(win)
wl = win["workload_label"].astype(str)

INF7 = "whitebox_inference_7b_1h"
inf = win[wl == INF7].sort_values("window_start")
half = len(inf) // 2
inf_train_idx, inf_test_idx = inf.index[:half], inf.index[half:]      # split by time
neg_test = win.loc[inf_test_idx]
print(f"7B inference control: {len(inf)} windows -> {len(inf_train_idx)} for training, "
      f"{len(inf_test_idx)} held out as test negatives")

TARGETS = [t for t in sorted(set(wl[wl.str.startswith("whitebox_lora_7b")]))
           if "pilot" not in t] + ["whitebox_diluted_7b_N10_1h"]

def fit(mask_excl, extra_keep=None):
    keep = ~mask_excl
    if extra_keep is not None: keep = keep | extra_keep
    tr = win[keep]
    c = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                               class_weight="balanced", random_state=42, n_jobs=8)
    c.fit(tr[feats].fillna(0).values.astype(np.float32),
          (tr["threeway_label"].values == "ml_training").astype(int))
    return c, list(c.classes_).index(1)

base_excl = (wl.str.startswith("whitebox") | wl.str.contains("_mw8"))
out=[]
print(f"\n  {'target 7B run':30s} {'AUC no-7B':>10s} {'AUC with-7B':>12s} {'delta':>7s}")
for tgt in TARGETS:
    tgt_mask = (wl == tgt)
    if tgt_mask.sum() < 30: continue
    test_pos = win[tgt_mask]

    # A: no 7B data at all
    cA,iA = fit(base_excl)
    pA = np.r_[cA.predict_proba(test_pos[feats].fillna(0).values.astype(np.float32))[:,iA],
               cA.predict_proba(neg_test[feats].fillna(0).values.astype(np.float32))[:,iA]]
    # B: keep other 7B runs + first half of the 7B inference control
    keep_extra = pd.Series(False, index=win.index)
    keep_extra.loc[inf_train_idx] = True
    keep_extra |= (wl.str.startswith("whitebox_lora_7b") | wl.str.startswith("whitebox_diluted_7b")) & ~tgt_mask
    cB,iB = fit(base_excl, extra_keep=keep_extra)
    pB = np.r_[cB.predict_proba(test_pos[feats].fillna(0).values.astype(np.float32))[:,iB],
               cB.predict_proba(neg_test[feats].fillna(0).values.astype(np.float32))[:,iB]]
    y = np.r_[np.ones(len(test_pos)), np.zeros(len(neg_test))]
    aA, aB = roc_auc_score(y,pA), roc_auc_score(y,pB)
    print(f"  {tgt:30s} {aA:>10.3f} {aB:>12.3f} {aB-aA:>+7.3f}")
    out.append(dict(target=tgt, auc_no7b=aA, auc_with7b=aB,
                    ap_no7b=average_precision_score(y,pA),
                    ap_with7b=average_precision_score(y,pB)))
json.dump(out, open(f"{SP}/scale_ablation.json","w"), indent=2, default=float)
print(f"\nwrote {SP}/scale_ablation.json")
