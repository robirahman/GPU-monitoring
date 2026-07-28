"""Question 2, at 10 s windows: does the classifier flag training windows as
training and NON-training windows as non-training?

Only runs with a genuine MIX of training-active and idle windows can answer
this; on an all-active run the second half of the question has no test cases.
Ground truth per window comes from the STEP log (a window is training-active
iff a real `kind=train` step landed inside it).

Protocol: for each target family, a binary RF is fit on all windows EXCEPT that
family's, so the family is an unseen strategy — the same leave-one-strategy-out
setup analyze_long_run.py uses. Reports 10 s and 30 s side by side, because
shrinking the window is what creates idle windows in the first place.
"""
import glob, sys, json
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "/root/GPU-monitoring")
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
from classifier.analyze_long_run import parse_train_ts

SP = "/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad"
TAU = 0.75

# run_id -> step log, read from each log's own "Run ID:" line (printed by the
# launcher). Do NOT pair sorted parquet files with rep numbers by index: the
# filenames sort by run-id hash, not collection order, which silently pairs a
# run with another rep's log and yields 0% training-active windows.
import re, os
step_of = {}
for lg in sorted(glob.glob("logs/*.log")):
    # Read the WHOLE file: run_workload.py's stdout is block-buffered when
    # redirected, so its "Run ID:" header is flushed at process exit and lands
    # at the END of the log, after millions of unbuffered STEP lines.
    try:
        head = open(lg, errors="ignore").read()
    except OSError:
        continue
    m = re.search(r"Run ID:\s+([0-9a-f]{6,})", head)
    if m:
        step_of[m.group(1)] = lg

# Families worth testing: those with a mix of active and idle windows at 30 s,
# plus the 7B long run as the all-active reference case.
FAMILIES = ["whitebox_lora_7b_N500_long", "whitebox_oracle_7b_long",
            "adversarial_L_diluted_200_mw8", "adversarial_L_diluted_10_mw8",
            "whitebox_oracle", "adversarial_I_stochastic"]

raw = load_all_data()
results = []

for wsec, stride in ((30.0, 15.0),):
    print(f"\n{'='*74}\nWINDOW = {wsec:.0f}s (stride {stride:.0f}s)\n{'='*74}", flush=True)
    win = sliding_windows(raw, wsec, stride, run_level_mode="causal")
    feats = get_feature_cols(win)

    for fam in FAMILIES:
        tgt = win[win["workload_label"].astype(str).str.startswith(fam)]
        tgt = tgt[tgt["run_id"].isin(step_of)]
        if tgt.empty:
            print(f"  {fam}: no windows with a step log"); continue

        wl = win["workload_label"].astype(str)
        tr = win[~(wl.str.startswith(fam) | wl.str.startswith("whitebox")
                   | wl.str.contains("_mw8"))]
        clf = RandomForestClassifier(n_estimators=300, min_samples_leaf=2,
            max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=8)
        clf.fit(tr[feats].fillna(0).values.astype(np.float32),
                (tr["threeway_label"].values == "ml_training").astype(int))
        i1 = list(clf.classes_).index(1)

        agg = dict(tp=0, fp=0, fn=0, tn=0, pa=[], pi=[])
        for rid, g in tgt.groupby("run_id"):
            g = g.sort_values("window_start")
            p = clf.predict_proba(g[feats].fillna(0).values.astype(np.float32))[:, i1]
            alert = p > TAU
            ts = parse_train_ts(step_of[rid])
            n_steps = (np.searchsorted(ts, g["window_end"].values)
                       - np.searchsorted(ts, g["window_start"].values))
            active = n_steps > 0
            agg["tp"] += int((alert & active).sum()); agg["fp"] += int((alert & ~active).sum())
            agg["fn"] += int((~alert & active).sum()); agg["tn"] += int((~alert & ~active).sum())
            if active.any():   agg["pa"].append(p[active])
            if (~active).any(): agg["pi"].append(p[~active])

        tp, fp, fn, tn = agg["tp"], agg["fp"], agg["fn"], agg["tn"]
        n = tp + fp + fn + tn
        act, idle = tp + fn, fp + tn
        rec = tp / act if act else float("nan")
        far = fp / idle if idle else float("nan")
        pa = np.concatenate(agg["pa"]).mean() if agg["pa"] else float("nan")
        pi = np.concatenate(agg["pi"]).mean() if agg["pi"] else float("nan")

        print(f"\n  --- {fam} ({n} windows, {100*act/n:.1f}% training-active) ---")
        print(f"        {'active':>10} {'idle':>10}")
        print(f"    alert   {tp:>10} {fp:>10}")
        print(f"    clear   {fn:>10} {tn:>10}")
        if act: print(f"    recall on training-active windows : {rec:.3f}")
        if idle:
            print(f"    FALSE-alert rate on idle windows  : {far:.3f}   "
                  f"(correctly clear: {1-far:.3f})")
            print(f"    mean p_train  active={pa:.3f}  idle={pi:.3f}  "
                  f"separation={pa-pi:+.3f}")
        else:
            print("    no idle windows -> question 2's second half is vacuous here")
        results.append(dict(window_sec=wsec, family=fam, n=n, frac_active=act/n,
                            tp=tp, fp=fp, fn=fn, tn=tn, recall=rec, far=far,
                            p_active=float(pa), p_idle=float(pi)))

with open(f"{SP}/q2_confusion.json", "w") as fh:
    json.dump(results, fh, indent=2, default=float)
print(f"\nwrote {SP}/q2_confusion.json")
