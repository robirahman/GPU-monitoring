"""M7wU Q3 (ROC/PR stratified by evasion family, unseen) and the Q4 follow-up
(is the *undiluted* 7B LoRA operating point detected, i.e. does the evader pay
any throughput cost at all?).

Protocol for both: leave-one-family-out. For each family, a binary RF is trained
with that family's runs removed, then scores (a) that family's windows as
positives and (b) all benign windows as negatives. That is the "unseen attack"
setting the reviewer asks about; the dumped loo_twostage predictions cannot be
used for ROC/PR because they contain only held-out-family windows, with no
negatives to form a curve against.
"""
import sys, glob, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve

sys.path.insert(0, "/root/GPU-monitoring")
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols

SP = "/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad"
TAU = 0.75

raw = load_all_data()
win = sliding_windows(raw, 30.0, 15.0, run_level_mode="causal")
feats = get_feature_cols(win)
wl = win["workload_label"].astype(str)
is_train = (win["threeway_label"] == "ml_training").values
benign = win[~is_train]

def fit_excluding(mask_excl):
    tr = win[~mask_excl]
    clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=8)
    clf.fit(tr[feats].fillna(0).values.astype(np.float32),
            (tr["threeway_label"].values == "ml_training").astype(int))
    return clf, list(clf.classes_).index(1)

# ── Q3: ROC / PR per evasion family, family unseen ────────────────────────
FAMILIES = {
    "whitebox_lora (1B)":        wl.str.match(r"whitebox_lora_N\d+$"),
    "whitebox_full/diluted (1B)":wl.str.startswith("whitebox_diluted"),
    "B_low_util":                wl.str.startswith("adversarial_B_low_util"),
    "I_stochastic":              wl.str.startswith("adversarial_I_stochastic"),
    "L_diluted":                 wl.str.startswith("adversarial_L_diluted"),
    "whitebox_oracle (WB-O)":    wl.str.startswith("whitebox_oracle"),
    "whitebox_lora 7B (all)":    wl.str.startswith("whitebox_lora_7b"),
}
print("=== Q3: ROC / PR by evasion family (family held out = UNSEEN) ===")
print(f"  negatives = all benign windows (n={len(benign)})")
print(f"\n  {'family':28s} {'pos':>6s} {'ROC AUC':>8s} {'PR AP':>7s} "
      f"{'alert@0.75':>11s} {'TPR@1%FPR':>10s}")
q3 = []
for name, m in FAMILIES.items():
    m = m.values & is_train
    if m.sum() < 20:
        print(f"  {name:28s} {int(m.sum()):>6d}  (too few windows)"); continue
    # exclude the whole family (all its runs) from training
    clf, i1 = fit_excluding(pd.Series(m, index=win.index))
    p_pos = clf.predict_proba(win[m][feats].fillna(0).values.astype(np.float32))[:, i1]
    p_neg = clf.predict_proba(benign[feats].fillna(0).values.astype(np.float32))[:, i1]
    y = np.r_[np.ones(len(p_pos)), np.zeros(len(p_neg))]
    s = np.r_[p_pos, p_neg]
    auc = roc_auc_score(y, s); ap = average_precision_score(y, s)
    fpr, tpr, _ = roc_curve(y, s)
    tpr1 = float(np.interp(0.01, fpr, tpr))
    print(f"  {name:28s} {int(m.sum()):>6d} {auc:>8.3f} {ap:>7.3f} "
          f"{(p_pos > TAU).mean():>11.3f} {tpr1:>10.3f}")
    q3.append(dict(family=name, n_pos=int(m.sum()), roc_auc=auc, pr_ap=ap,
                   alert_rate=float((p_pos > TAU).mean()), tpr_at_1pct_fpr=tpr1))

# ── Q4 follow-up: are the 7B operating points detected at all? ────────────
print("\n=== Q4: is each 7B operating point DETECTED? (all whitebox unseen) ===")
excl = (wl.str.startswith("whitebox") | wl.str.contains("_mw8")).values
clf, i1 = fit_excluding(pd.Series(excl, index=win.index))
print(f"  {'run':34s} {'windows':>8s} {'p_train mean':>13s} {'max':>7s} {'alert rate':>11s}")
q4 = []
for lab in sorted(set(wl[wl.str.startswith(("whitebox_lora_7b", "whitebox_inference_7b",
                                            "whitebox_diluted_7b", "whitebox_lora_1b"))])):
    g = win[wl == lab]
    if g.empty: continue
    p = clf.predict_proba(g[feats].fillna(0).values.astype(np.float32))[:, i1]
    print(f"  {lab:34s} {len(p):>8d} {p.mean():>13.3f} {p.max():>7.3f} "
          f"{(p > TAU).mean():>11.3f}")
    q4.append(dict(run=lab, n=len(p), p_mean=float(p.mean()),
                   p_max=float(p.max()), alert_rate=float((p > TAU).mean())))

json.dump(dict(q3=q3, q4=q4), open(f"{SP}/q3q4.json", "w"), indent=2)
print(f"\nwrote {SP}/q3q4.json")
