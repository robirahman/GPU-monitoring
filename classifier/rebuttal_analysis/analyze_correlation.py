"""Window-correlation structure of a moderately-evading run.

Answers three questions about a run whose per-window detection flag actually
fluctuates:

  Q1/Q3  How correlated are windows across the run? Reports the autocorrelation
         function of both the binary alert flag and the continuous p_train, the
         two-state Markov transition probabilities the Section 2 extrapolation
         assumes, and the effective number of independent windows.

  Q2     Does the classifier flag training windows as training and non-training
         windows as non-training? Reports the full confusion matrix of alert vs
         STEP-log ground truth, so a "60% detection rate" can be checked for
         whether it is concentrated on the windows that actually trained.

IMPORTANT methodological note, applied throughout: with window=30 s and
stride=15 s, consecutive windows SHARE HALF THEIR SAMPLES, which mechanically
inflates lag-1 autocorrelation regardless of any real temporal structure. Every
correlation is therefore reported twice — over all windows (overlapping) and
over every 2nd window (disjoint). The disjoint figure is the one that should
feed a Markov extrapolation; the overlapping one is an upper bound.
"""
import argparse, glob, sys
import numpy as np, pandas as pd
from sklearn.ensemble import RandomForestClassifier

sys.path.insert(0, "/root/GPU-monitoring")
from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols
from classifier.analyze_long_run import parse_train_ts

ap = argparse.ArgumentParser()
ap.add_argument("--run-parquet", default=None)
ap.add_argument("--step-log", default=None)
ap.add_argument("--dilution", default=None)
ap.add_argument("--window", type=float, default=30.0)
ap.add_argument("--stride", type=float, default=15.0)
ap.add_argument("--tau", type=float, default=0.75)
ap.add_argument("--max-lag", type=int, default=40)
a = ap.parse_args()

if a.run_parquet is None:
    pat = f"data/whitebox_lora_7b_corr_N{a.dilution}_long_*.parquet"
    hits = sorted(glob.glob(pat))
    if not hits:
        sys.exit(f"no run matching {pat}")
    a.run_parquet = hits[-1]
if a.step_log is None:
    a.step_log = f"logs/wb_lora_7b_corr_N{a.dilution}_long.log"

run_id = pd.read_parquet(a.run_parquet, columns=["run_id"])["run_id"].iloc[0]
print(f"run: {a.run_parquet}\nstep log: {a.step_log}\n")

df = load_all_data()
win = sliding_windows(df, a.window, a.stride, run_level_mode="causal")
feats = get_feature_cols(win)

is_run = win["run_id"] == run_id
run_win = win[is_run].sort_values("window_start").reset_index(drop=True)
if run_win.empty:
    sys.exit("run produced no windows")

tr = win[~is_run]
tr = tr[~tr["workload_label"].astype(str).str.startswith("whitebox")]   # unseen strategy
clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2, max_features="sqrt",
                             class_weight="balanced", random_state=42, n_jobs=-1)
clf.fit(tr[feats].fillna(0).values.astype(np.float32),
        (tr["threeway_label"].values == "ml_training").astype(int))
i1 = list(clf.classes_).index(1)
p = clf.predict_proba(run_win[feats].fillna(0).values.astype(np.float32))[:, i1]
alert = p > a.tau

ws, we = run_win["window_start"].values, run_win["window_end"].values
train_ts = parse_train_ts(a.step_log)
n_steps = np.searchsorted(train_ts, we) - np.searchsorted(train_ts, ws)
active = n_steps > 0

print(f"=== run summary ({len(p)} windows, {len(p)*a.stride/3600:.2f} h) ===")
print(f"  per-window alert rate: {alert.mean():.3f}   (target band 0.40-0.80)")
print(f"  p_train: mean={p.mean():.3f} median={np.median(p):.3f} "
      f"min={p.min():.3f} max={p.max():.3f}")
print(f"  training-active windows (STEP log): {active.sum()}/{len(active)} "
      f"({100*active.mean():.1f}%)   median steps/window={np.median(n_steps):.0f}")

# ── Q2: is detection concentrated on the windows that actually trained? ────
print("\n=== Q2: alert vs STEP-log ground truth ===")
tp = int((alert & active).sum());  fn = int((~alert & active).sum())
fp = int((alert & ~active).sum()); tn = int((~alert & ~active).sum())
print(f"                 active(train)   inactive(no train)")
print(f"  alert            {tp:6d}          {fp:6d}")
print(f"  no alert         {fn:6d}          {tn:6d}")
if active.sum():
    print(f"  recall on training-active windows: {tp/max(1,tp+fn):.3f}")
if (~active).sum():
    print(f"  correctly-clear rate on non-training windows: {tn/max(1,fp+tn):.3f}")
    print(f"  false-alert rate on non-training windows:      {fp/max(1,fp+tn):.3f}")
else:
    print("  NOTE: every window contains >=1 training step, so there are no")
    print("        non-training windows to get wrong — Q2's second half is")
    print("        vacuous at this dilution/window size, by construction.")
if active.sum() and (~active).sum():
    print(f"  p_train mean on active={p[active].mean():.3f} vs "
          f"inactive={p[~active].mean():.3f}  (separation="
          f"{p[active].mean()-p[~active].mean():+.3f})")

# ── Q1/Q3: correlation structure ──────────────────────────────────────────
def acf(x, lag):
    if lag >= len(x): return float("nan")
    x0, x1 = x[:-lag].astype(float), x[lag:].astype(float)
    if x0.std() == 0 or x1.std() == 0: return float("nan")
    return float(np.corrcoef(x0, x1)[0, 1])

print("\n=== Q1/Q3: correlation across windows ===")
print("  (window=30s stride=15s -> adjacent windows share 50% of samples;")
print("   'disjoint' subsamples every 2nd window so overlap cannot inflate it)")
af, pf = alert.astype(float), p
adis, pdis = alert[::2].astype(float), p[::2]
print(f"\n  {'lag':>4}  {'alert ACF':>10} {'p_train ACF':>12}   |  "
      f"{'alert ACF':>10} {'p_train ACF':>12}   (disjoint)")
for lag in [1, 2, 3, 4, 8, 16, 24, 40]:
    if lag > a.max_lag: break
    print(f"  {lag:>4}  {acf(af,lag):>10.3f} {acf(pf,lag):>12.3f}   |  "
          f"{acf(adis,lag):>10.3f} {acf(pdis,lag):>12.3f}")

for tag, arr in (("overlapping", af), ("disjoint", adis)):
    n = len(arr)
    if arr.std() == 0:
        print(f"\n  {tag}: flag never varies — no Markov structure to estimate")
        continue
    prev, cur = arr[:-1].astype(bool), arr[1:].astype(bool)
    p11 = cur[prev].mean() if prev.any() else float("nan")
    p01 = cur[~prev].mean() if (~prev).any() else float("nan")
    r1 = acf(arr, 1)
    ess = n * (1 - r1) / (1 + r1) if r1 > -1 else float("nan")
    print(f"\n  {tag} ({n} windows):")
    print(f"    P(alert | prev alert) = {p11:.3f}")
    print(f"    P(alert | prev clear) = {p01:.3f}")
    print(f"    lag-1 ACF = {r1:.3f}")
    print(f"    effective independent windows ~ {ess:.0f} of {n} "
          f"({100*ess/n:.0f}%)")
    if not np.isnan(p11) and not np.isnan(p01) and p01 > 0:
        # stationary alert rate implied by the fitted two-state chain
        pi = p01 / (1 - p11 + p01) if (1 - p11 + p01) != 0 else float("nan")
        print(f"    implied stationary alert rate = {pi:.3f} "
              f"(observed {arr.mean():.3f})")

# time to first k-of-m trigger, for reference against the short-run fit
for k, m in ((3, 5), (2, 3)):
    trig = None
    for i in range(m - 1, len(alert)):
        if alert[i - m + 1:i + 1].sum() >= k:
            trig = ws[i] - ws[0]; break
    print(f"\n  time to first {k}-of-{m} alert: "
          f"{'NEVER' if trig is None else f'{trig:.0f}s ({trig/60:.1f} min)'}")
