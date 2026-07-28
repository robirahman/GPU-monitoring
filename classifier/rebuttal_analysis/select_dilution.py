"""Pick the WB-L dilution whose per-window detection lands mid-band (40-80%).

Scores each pilot run with the same protocol analyze_long_run.py uses for an
unseen strategy: a binary RF trained on the corpus with ALL whitebox families
excluded, so the pilot is genuinely out-of-family. Writes the winning dilution
to $SP/chosen_dilution so the queue can pick it up.
"""
import glob, os, sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "/root/GPU-monitoring")
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols

SP = "/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad"
TAU = 0.75
TARGET = 0.60
LO, HI = 0.40, 0.80

# Sweep both sides of the band. Detection RISES with less dilution (more
# training): the 7B N=10 run sat at p_train max 0.537, entirely below tau=0.75,
# so 0% detected. N=2/3/5 train more and bracket the high end; N=6/7/8 fill the
# gap between N=5 and the known 0% at N=10, where the straddle point must lie.
pilots = {}          # dilution -> run_id
for d in (2, 3, 5, 6, 7, 8):
    fs = sorted(glob.glob(f"data/whitebox_lora_7b_pilot_N{d}_*.parquet"))
    if fs:
        pilots[d] = pd.read_parquet(fs[-1], columns=["run_id"])["run_id"].iloc[0]

if not pilots:
    print("no pilot runs found", file=sys.stderr)
    sys.exit(1)

df = load_all_data()
win = sliding_windows(df, 30.0, 15.0, run_level_mode="causal")
feats = get_feature_cols(win)

pilot_ids = set(pilots.values())
train = win[~win["run_id"].isin(pilot_ids)]
train = train[~train["workload_label"].astype(str).str.startswith("whitebox")]
Xtr = train[feats].fillna(0).values.astype(np.float32)
ytr = (train["threeway_label"].values == "ml_training").astype(int)

clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
    max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(Xtr, ytr)
i1 = list(clf.classes_).index(1)

rows = []
for d, rid in sorted(pilots.items()):
    w = win[win["run_id"] == rid]
    if w.empty:
        print(f"  N={d}: no windows", file=sys.stderr); continue
    p = clf.predict_proba(w[feats].fillna(0).values.astype(np.float32))[:, i1]
    rate = float((p > TAU).mean())
    rows.append(dict(dilution=d, windows=len(p), alert_rate=rate,
                     p_mean=float(p.mean()), p_max=float(p.max())))
    print(f"  N={d:2d}: {len(p):4d} windows  alert_rate={rate:.3f}  "
          f"p_train mean={p.mean():.3f} max={p.max():.3f}")

# ── Bonus, free: the same fitted classifier answers the correlation question
# on the EXISTING 7 B N=10 8 h run using the CONTINUOUS p_train. Its binary flag
# is degenerate (0/1920 alerts, hence the nan autocorrelation), but p_train
# itself varies (0.171-0.537), so the cross-window correlation structure is
# measurable there right now without spending any new GPU time.
def acf(x, lag):
    if lag >= len(x): return float("nan")
    x0, x1 = np.asarray(x[:-lag], float), np.asarray(x[lag:], float)
    if x0.std() == 0 or x1.std() == 0: return float("nan")
    return float(np.corrcoef(x0, x1)[0, 1])

longs = sorted(glob.glob("data/whitebox_lora_7b_long_*.parquet"))
if longs:
    lid = pd.read_parquet(longs[-1], columns=["run_id"])["run_id"].iloc[0]
    lw = win[win["run_id"] == lid].sort_values("window_start")
    if not lw.empty:
        pl = clf.predict_proba(lw[feats].fillna(0).values.astype(np.float32))[:, i1]
        print(f"\n=== existing 7B N=10 run: continuous p_train correlation "
              f"({len(pl)} windows) ===")
        print(f"  p_train mean={pl.mean():.3f} sd={pl.std():.3f} "
              f"min={pl.min():.3f} max={pl.max():.3f}  (alerts at tau={TAU}: {(pl>TAU).sum()})")
        print("  ACF of p_train  (window=30s stride=15s -> every 2nd window is disjoint)")
        for lag in (1, 2, 4, 8, 16, 40):
            print(f"    lag {lag:>3}: overlapping {acf(pl,lag):+.3f}   "
                  f"disjoint {acf(pl[::2],lag):+.3f}")
        np.save(f"{SP}/p_train_7b_N10.npy", pl)

inband = [r for r in rows if LO <= r["alert_rate"] <= HI]
if inband:
    best = min(inband, key=lambda r: abs(r["alert_rate"] - TARGET))
    reason = f"in band, closest to {TARGET}"
    print(f"\nchosen: N={best['dilution']} (alert_rate={best['alert_rate']:.3f}) — {reason}")
    with open(f"{SP}/chosen_dilution", "w") as fh:
        fh.write(str(best["dilution"]))
else:
    # Refuse to burn 4 GPU-hours at an operating point that cannot answer the
    # question. Write NO choice so the queue aborts, and report the bracket so
    # the next sweep can be aimed between the two nearest dilutions.
    best = None
    reason = "NONE in band — not launching the long run"
    lo_side = [r for r in rows if r["alert_rate"] < LO]
    hi_side = [r for r in rows if r["alert_rate"] > HI]
    print(f"\n{reason}. Observed rates: "
          + ", ".join(f"N={r['dilution']}:{r['alert_rate']:.2f}" for r in rows))
    if hi_side and lo_side:
        print(f"  band lies between N={min(hi_side, key=lambda r: r['alert_rate'])['dilution']}"
              f" and N={max(lo_side, key=lambda r: r['alert_rate'])['dilution']}")
    elif hi_side:
        print(f"  all pilots ABOVE band; try MORE dilution than "
              f"N={max(r['dilution'] for r in rows)}")
    else:
        print(f"  all pilots BELOW band; try LESS dilution than "
              f"N={min(r['dilution'] for r in rows)}")

with open(f"{SP}/pilot_scores.json", "w") as fh:
    json.dump(dict(rows=rows, chosen=best, reason=reason, tau=TAU), fh, indent=2)
