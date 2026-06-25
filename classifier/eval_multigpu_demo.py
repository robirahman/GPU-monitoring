#!/usr/bin/env python3
"""Demonstrate the multi-GPU coverage fix for the physical-sensor classifier.

Shows: a GPU0-only classifier is trivially evaded by training on GPU1/GPU2/multi-GPU,
while the all-clamp classifier (every GPU clamp + node aggregates) catches it without
false-positives on non-training that happens to run on other GPUs.

Prereq: run scripts/collect_multigpu.sh first to populate sensor_logs/mg_*.
Usage:  python classifier/eval_multigpu_demo.py
"""
import os, sys, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from physical_sensor_classifier import load_sensor_data, windows, physical_channels, TEMPORAL
from sklearn.ensemble import RandomForestClassifier

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
WIN, STR = 30, 15
OLD_CH = ["GPU0_rms_mv", "GPU0_peak_mv", "MOBO_rms_mv", "mic_rms", "mic_dbfs", "mic_peak", "mic_dominant_hz"]
OLD_TMP = ["GPU0_rms_mv", "mic_rms"]
NEW_CH, NEW_TMP = physical_channels(), TEMPORAL


def main():
    df = load_sensor_data(os.path.join(PROJECT_ROOT, "sensor_logs"))
    df = df[~df.workload_label.str.contains(r"stealth|ev2b_|^s2_|^s3_|dil_v|evasion2|mg_test", regex=True)]
    mg = df.workload_label.str.startswith("mg_")
    if not mg.any():
        print("No mg_* runs found. Run scripts/collect_multigpu.sh first."); return

    def mglabel(w):
        return "ml_training" if any(k in w for k in ["resnet", "gpt2", "multi"]) else "other"
    df.loc[mg, "threeway_label"] = df.loc[mg, "workload_label"].map(mglabel)
    base40, mg_runs = df[~mg], df[mg]

    def fit(d, ch, tmp):
        dw = windows(d, ch, tmp, WIN, STR)
        fc = [c for c in dw.columns if c not in ("run_id", "workload_label", "threeway_label")]
        clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
            class_weight="balanced", random_state=42, n_jobs=-1).fit(
            dw[fc].fillna(0).values.astype(np.float32), (dw.threeway_label.values == "ml_training").astype(int))
        return clf, fc

    def pred(clf, fc, rid, ch, tmp):
        w = windows(df[df.run_id == rid], ch, tmp, WIN, STR)
        return None if w.empty else clf.predict(w[fc].fillna(0).values.astype(np.float32)).mean()

    def rid_of(sub):
        r = mg_runs[mg_runs.workload_label == sub].run_id.unique(); return r[0] if len(r) else None

    test_runs = [("mg_gpu2_resnet", "train"), ("mg_multi12_resnet", "train"), ("mg_gpu1_resnet", "train"),
                 ("mg_gpu2_cufft", "other"), ("mg_gpu1_idle", "other")]
    held = {rid_of(t[0]) for t in test_runs} - {None}

    old_clf, old_fc = fit(base40, OLD_CH, OLD_TMP)              # deployed-today GPU0-only model
    new_clf, new_fc = fit(df[~df.run_id.isin(held)], NEW_CH, NEW_TMP)  # all-clamp, + multi-GPU training

    print(f"{'held-out run':22s} {'truth':7s} {'OLD(GPU0-only)':>16s} {'NEW(all-clamps)':>17s}")
    for sub, truth in test_runs:
        rid = rid_of(sub)
        if rid is None: print(f"{sub:22s}  (not collected)"); continue
        o, n = pred(old_clf, old_fc, rid, OLD_CH, OLD_TMP), pred(new_clf, new_fc, rid, NEW_CH, NEW_TMP)
        if o is None or n is None: print(f"{sub:22s}  (no 30s windows)"); continue
        ov, nv = ("EVADE" if o <= .5 else "catch"), ("EVADE" if n <= .5 else "catch")
        fp = "  <-- FALSE POS" if (truth == "other" and n > .5) else ""
        print(f"{sub:22s} {truth:7s} {o*100:9.0f}% {ov:>5s} {n*100:10.0f}% {nv:>5s}{fp}")
    print("\nExpect: OLD misses GPU1/2/multi training (EVADE) -> NEW catches it; non-training stays low on NEW.")


if __name__ == "__main__":
    main()
