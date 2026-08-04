#!/usr/bin/env python3
"""Is ordinary (undisguised) PPO detected, and how does it compare to disguised PPO?

Two distinct questions, often conflated:

  1. REGIME COVERAGE (the Area Chair's question). Ordinary RLHF -- rollout
     generation plus an autograd+AdamW policy update (`ppo_rlhf_*`, mode P0) --
     is legitimate training that belongs in the baseline corpus. Does the monitor
     detect it (a) with no RL anywhere in its calibration set, and (b) once RL is
     in the calibration set?

  2. EVASION (Round 6). Disguised PPO (`symgemm_ppo_fused_*`, mode P1) has the
     same rollout/update structure but routes the update through the fused
     SymLinearFn epilogue, so no gradient tensor and no optimizer state exist.
     It stacks temporal dilution on write minimisation.

Both are scored per window and per run (hybrid 3-of-5 OR CUSUM), against the
matched generative-inference negative (`symgemm_decode_*`).
"""
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
from classifier.rebuttal_analysis.cusum_detect import hybrid_first_alert

TAU, K, M, CK = 0.75, 3, 5, 0.25

def main():
    print("Loading + windowing (causal)...")
    win = sliding_windows(load_all_data(), 30, 15, run_level_mode="causal").reset_index(drop=True)
    feats = get_feature_cols(win); wl = win["workload_label"].astype(str)
    plain = wl.str.startswith("ppo_rlhf_").values
    fused = wl.str.startswith("symgemm_ppo_fused").values
    dec   = wl.str.startswith("symgemm_decode").values
    sym   = wl.str.startswith("symgemm_").values
    print(f"  plain-PPO windows={plain.sum()} ({win.loc[plain,'run_id'].nunique()} runs) | "
          f"fused-PPO={fused.sum()} | decode={dec.sum()}")

    def fit(train_mask):
        tr = win[train_mask]
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2, max_features="sqrt",
                                     class_weight="balanced", random_state=42, n_jobs=-1)
        clf.fit(tr[feats].fillna(0).values.astype(np.float32),
                (tr["threeway_label"].values == "ml_training").astype(int))
        i1 = list(clf.classes_).index(1)
        return lambda m: clf.predict_proba(win[m][feats].fillna(0).values.astype(np.float32))[:, i1]

    # Calibrate hybrid h on benign inference (non-symgemm, non-PPO) to 0.10/run-h
    def calib(pr):
        ben = (win["threeway_label"].values == "ml_inference") & ~sym & ~plain
        g = win[ben].groupby("run_id")
        series, hours = [], 0.0
        p_all = pr(ben); off = 0
        for _, gg in win[ben].groupby("run_id"):
            n = len(gg); series.append(p_all[off:off+n]); off += n
            hours += max(float(gg["window_end"].iloc[-1]-gg["window_start"].iloc[0]), 1.0)/3600
        for h in np.arange(2.0, 80.0, 0.5):
            fp = sum(1 for s in series if hybrid_first_alert(s, TAU, K, M, CK, h) is not None)/hours
            if fp <= 0.10: return h, fp
        return 80.0, float('nan')

    def report(name, pr, test_mask, h):
        p = pr(test_mask); pn = pr(dec)
        y = np.r_[np.ones(len(p)), np.zeros(len(pn))]
        auc = roc_auc_score(y, np.r_[p, pn])
        sub = win[test_mask]; off=0; alerts=0; runs=0
        for _, gg in sub.groupby("run_id"):
            n=len(gg); s=p[off:off+n]; off+=n; runs+=1
            if hybrid_first_alert(s, TAU, K, M, CK, h) is not None: alerts+=1
        print(f"  {name:38s} AUC(vs decode)={auc:.3f}  window p_train={p.mean():.3f}  "
              f"run-alert={alerts}/{runs}")

    # (a) NO RL in the calibration set: hold out BOTH ppo variants
    pr = fit(~plain & ~fused); h,fp = calib(pr)
    print(f"\n[A] Zero-shot: NO RL of any kind in training (h={h:.1f}, benign FP={fp:.3f}/run-h)")
    report("ordinary PPO (ppo_rlhf_*)", pr, plain, h)
    report("disguised PPO (ppo_fused_*)", pr, fused, h)

    # (b) RL in the calibration set: ordinary PPO present, disguised still held out
    pr2 = fit(~fused); h2,fp2 = calib(pr2)
    print(f"\n[B] Calibrated on ordinary RL (h={h2:.1f}, benign FP={fp2:.3f}/run-h)")
    report("ordinary PPO (in training corpus)", pr2, plain, h2)
    report("disguised PPO (still unseen)", pr2, fused, h2)

if __name__ == "__main__":
    main()
