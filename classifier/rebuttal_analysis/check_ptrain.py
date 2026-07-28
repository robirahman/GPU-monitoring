"""Replicate analyze_long_run.py's scoring, but report the p_train DISTRIBUTION
so we can tell 'confidently not-training' from 'degenerate all-zero scores'."""
import glob, numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols

RUNS = {
 '7B': ('data/whitebox_lora_7b_long_*.parquet',),
 '1B': ('data/whitebox_lora_N10_long_*.parquet',),
}
df = load_all_data()
win = sliding_windows(df, 30.0, 15.0, run_level_mode="causal")
feats = get_feature_cols(win)

labels = {}
for tag,(pat,) in RUNS.items():
    p = sorted(glob.glob(pat))[0]
    labels[tag] = pd.read_parquet(p, columns=["run_id"])["run_id"].iloc[0]

# one classifier, same protocol as the script: exclude whitebox + the run itself
for tag, rl in labels.items():
    is_long = win["run_id"] == rl
    long_win = win[is_long].sort_values("window_start").reset_index(drop=True)
    train = win[~is_long]
    train = train[~train["workload_label"].astype(str).str.startswith("whitebox")]
    Xtr = train[feats].fillna(0).values.astype(np.float32)
    ytr = (train["threeway_label"].values == "ml_training").astype(int)
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
        max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    i1 = list(clf.classes_).index(1)
    p = clf.predict_proba(long_win[feats].fillna(0).values.astype(np.float32))[:, i1]
    print(f"\n=== {tag} run ({rl}) — {len(p)} windows ===")
    print(f"  p_train: min={p.min():.4f} p25={np.percentile(p,25):.4f} median={np.median(p):.4f} "
          f"mean={p.mean():.4f} p75={np.percentile(p,75):.4f} p95={np.percentile(p,95):.4f} max={p.max():.4f}")
    print(f"  exactly 0.0: {(p==0).sum()}/{len(p)}   >0.5: {(p>0.5).sum()}   >0.75: {(p>0.75).sum()}")
    print(f"  train-class prior in training set: {ytr.mean():.3f}  ({ytr.sum()} of {len(ytr)} windows)")
    # sanity: does this same classifier flag genuine clean-training windows?
    ct = win[(~is_long) & (win["threeway_label"]=="ml_training") &
             (~win["workload_label"].astype(str).str.startswith(("whitebox","adversarial")))]
    if len(ct):
        pc = clf.predict_proba(ct[feats].fillna(0).values.astype(np.float32))[:, i1]
        print(f"  CONTROL clean-training windows (in-sample): mean p_train={pc.mean():.3f}, "
              f">0.75: {(pc>0.75).mean()*100:.1f}%")
