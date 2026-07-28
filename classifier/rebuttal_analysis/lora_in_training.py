"""Does putting LoRA examples in the training set fix detection?

The negative result so far is measured under the unseen-family protocol: the
classifier has never seen a LoRA run, and misses undiluted LoRA at 1.1B (0%
alerts), 3B (0%) and 7B (14.8%, below the 16.6% benign control). The
constructive question is whether that is fixable with calibration data.

STRICTLY GROUPED. Every condition holds out BOTH test runs entirely — the
positive LoRA run and the negative inference run never appear in training, at
any window. This matters more than usual here: within-run ICC is ~0.98, so
letting even part of a test run into training makes the task trivial. An
earlier version of this ablation split one inference run in half and reported
AUC 1.000; that number was leakage, and this design exists to avoid repeating
it.

Conditions, in increasing order of what the defender is assumed to have:
  A  no whitebox data at all         — the unseen-family protocol (baseline)
  B  + LoRA at the SAME scale        — defender calibrated on this model size
  C  + LoRA at OTHER scales only     — can cheap 1B LoRA runs generalise up?
  D  + all whitebox except the tests — most permissive
"""
import sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, "/root/GPU-monitoring")
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols

SP = "/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad"
TAU = 0.75

def scale_of(l):
    if "_7b" in l: return "7B"
    if "_3b" in l: return "3B"
    return "1B"

raw = load_all_data()
win = sliding_windows(raw, 30.0, 15.0, run_level_mode="causal")
feats = get_feature_cols(win)
wl = win["workload_label"].astype(str)
is_wb = wl.str.startswith("whitebox") | wl.str.contains("_mw8")
is_lora = wl.str.contains("lora")

# (scale, positive LoRA run, negative inference run) — all held out entirely
CASES = [
    ("1B", "whitebox_lora_1b_N1_1h",  "whitebox_inference_1b_matched"),
    ("3B", "whitebox_lora_3b_N1_rep2","whitebox_inference_3b_rep2"),
    ("7B", "whitebox_lora_7b_N10_rep2","whitebox_inference_7b_rep3"),
]

def evaluate(train_mask, pos_mask, neg_mask):
    tr = win[train_mask]
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                 class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(tr[feats].fillna(0).values.astype(np.float32),
            (tr["threeway_label"].values == "ml_training").astype(int))
    i1 = list(clf.classes_).index(1)
    f = lambda m: clf.predict_proba(win[m][feats].fillna(0).values.astype(np.float32))[:, i1]
    p, n = f(pos_mask), f(neg_mask)
    y = np.r_[np.ones(len(p)), np.zeros(len(n))]; s = np.r_[p, n]
    fpr, tpr, _ = roc_curve(y, s)
    return dict(auc=roc_auc_score(y, s), ap=average_precision_score(y, s),
                alert_pos=float((p > TAU).mean()), alert_neg=float((n > TAU).mean()),
                p_pos=float(p.mean()), p_neg=float(n.mean()),
                tpr_at_1pct=float(np.interp(0.01, fpr, tpr)), n_pos=len(p), n_neg=len(n),
                n_train=int(train_mask.sum()))

results = []
for scale, pos_lab, neg_lab in CASES:
    pos_mask = (wl == pos_lab).values
    neg_mask = (wl == neg_lab).values
    if pos_mask.sum() < 20 or neg_mask.sum() < 20:
        print(f"\n### {scale}: missing runs "
              f"(pos={pos_mask.sum()} windows, neg={neg_mask.sum()}) — skipped")
        continue
    held_out = pos_mask | neg_mask
    same = (wl.map(scale_of) == scale).values

    conds = {
        "A no whitebox at all":        (~is_wb).values & ~held_out,
        "B + same-scale LoRA":         ((~is_wb).values | (is_lora.values & same)) & ~held_out,
        "C + other-scale LoRA only":   ((~is_wb).values | (is_lora.values & ~same)) & ~held_out,
        "D + all whitebox":            ~held_out,
    }
    print(f"\n### {scale}:  positives={pos_lab} ({pos_mask.sum()} win)   "
          f"negatives={neg_lab} ({neg_mask.sum()} win)")
    print(f"  {'condition':28s} {'train win':>9s} {'AUC':>6s} {'AP':>6s} "
          f"{'alert+':>7s} {'alert-':>7s} {'TPR@1%FPR':>10s}")
    for name, tm in conds.items():
        r = evaluate(tm, pos_mask, neg_mask)
        print(f"  {name:28s} {r['n_train']:>9d} {r['auc']:>6.3f} {r['ap']:>6.3f} "
              f"{r['alert_pos']:>7.3f} {r['alert_neg']:>7.3f} {r['tpr_at_1pct']:>10.3f}")
        results.append(dict(scale=scale, condition=name, pos=pos_lab, neg=neg_lab, **r))

json.dump(results, open(f"{SP}/lora_in_training.json", "w"), indent=2, default=float)
print(f"\nwrote {SP}/lora_in_training.json")
