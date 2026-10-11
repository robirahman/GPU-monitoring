#!/usr/bin/env python3
"""Adversarial-training round: harden the physical-sensor classifier against the
ballast-dilution evasion, then check it still works and doesn't over-trigger.

Protocol
--------
- base runs        = original suite (training/inference/HPC/mining/rendering/idle/adversarial K..J)
- attack runs      = real training disguised to evade (stealth/dilution/cofill/combo/dil_v*),
                     FORCE-labeled ml_training (they ARE training).
- Hold out a test set: two fresh dilution attacks + one cufft + one nbody (false-positive check)
                       + one resnet (training control).
- BASELINE classifier : trained on base only (minus held-out base runs)  [the un-hardened model]
- HARDENED classifier : baseline training set + attack runs (minus held-out attacks)
- Score both on every held-out run.

Goal: HARDENED catches the held-out dilution attacks (which BASELINE misses) WITHOUT regressing to
false positives on cufft/nbody.

Usage: python classifier/harden_physical_classifier.py
"""
import sys, glob, os, re, numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physical_sensor_classifier import load_sensor_data, windows, physical_channels
from sklearn.ensemble import RandomForestClassifier

WIN, STR = 30, 15
chans = physical_channels("GPU0"); temporal = ["GPU0_rms_mv", "mic_rms"]

from classifier.workload_labels import workload_kind

df = load_sensor_data(os.path.join(os.path.dirname(__file__), "..", "sensor_logs"))
# Ground truth comes from the shared taxonomy. Attack membership is a separate
# axis; controls remain in the baseline pool, with their correct objective.
df["is_attack"] = df.workload_label.map(workload_kind).eq("adversarial")

_len = df.groupby("run_id").size()
def longest_run(sub):
    """Pick the longest (full-length) run whose workload_label matches `sub`."""
    ids = df[df.workload_label.str.fullmatch(sub) | df.workload_label.str.contains(sub, regex=True)].run_id.unique()
    ids = [r for r in ids if _len.get(r, 0) >= 35]   # need >=35s for a 30s window
    return [max(ids, key=lambda r: _len[r])] if ids else []

# ---- held-out test set: 2 unseen dilution attacks + a full control panel ----
test_attacks = list(df[df.workload_label.str.contains(r"dil_v2|dil_v3")].run_id.unique())[:2]
controls = {}
for name in ["pytorch_resnet_cifar10", "gpt2_wikitext2", "cufft_benchmark",
             "nbody_sim", "mining_ethash_proxy", "rendering_proxy", "idle", "resnet50_inference"]:
    r = longest_run(name)
    if r: controls[name] = r[0]
heldout = set(test_attacks) | set(controls.values())

train_df = df[~df.run_id.isin(heldout)]
base_train = train_df[~train_df.is_attack]          # un-hardened
hard_train = train_df                                # base + remaining attacks

def fit(sub_df):
    dw = windows(sub_df, chans, temporal, WIN, STR)
    fc = [c for c in dw.columns if c not in ("run_id", "workload_label", "threeway_label")]
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
        class_weight="balanced", random_state=42, n_jobs=-1).fit(
        dw[fc].fillna(0).values.astype(np.float32), (dw.threeway_label.values == "ml_training").astype(int))
    return clf, fc, len(dw)

base_clf, base_fc, nb = fit(base_train)
hard_clf, hard_fc, nh = fit(hard_train)
print(f"BASELINE  trained on {nb} windows / {base_train.run_id.nunique()} runs (no attacks)")
print(f"HARDENED  trained on {nh} windows / {hard_train.run_id.nunique()} runs "
      f"(+{train_df.is_attack.groupby(train_df.run_id).first().sum()} attack runs)\n")

def label_for(rid):
    r = df[df.run_id == rid]
    if r.is_attack.any(): return "ATTACK(train)"
    return "TRAIN" if r.threeway_label.iloc[0] == "ml_training" else "OTHER"

def score(clf, fc, rid):
    w = windows(df[df.run_id == rid], chans, temporal, WIN, STR)
    if w.empty: return None
    p = clf.predict(w[fc].fillna(0).values.astype(np.float32)).mean()
    return p

print(f"{'held-out run':28s} {'truth':14s} {'BASE %train':>14s} {'HARD %train':>14s}")
for rid in list(test_attacks) + list(controls.values()):
    wl = df[df.run_id == rid].workload_label.iloc[0]
    b = score(base_clf, base_fc, rid); h = score(hard_clf, hard_fc, rid)
    if b is None: print(f"{wl[:28]:28s} (no 30s windows)"); continue
    bm = "EVADE" if b <= 0.5 else "catch"; hm = "EVADE" if h <= 0.5 else "catch"
    flag = ""
    if label_for(rid) == "OTHER" and h > 0.5: flag = "  <-- FALSE POSITIVE"
    print(f"{wl[:28]:28s} {label_for(rid):14s} {b*100:7.0f}% {bm:>5s} {h*100:7.0f}% {hm:>5s}{flag}")
print("\nWant: attacks BASE=EVADE -> HARD=catch ; OTHER stays low on both (no false positives).")
