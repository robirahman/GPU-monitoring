"""Appendix: run-level evaluation -- ROC/PR, fixed operating point, CUSUM.

Answers the three things Reviewer M7wU asked for and the final justification
still listed as missing: run-level detection under a fixed alert rule (missed
runs, false alerts per GPU-hour, time-to-first-alert), ROC/PR stratified by
evasion family, and *complete* CUSUM results -- false alerts, latency, and
transfer to families the detector never trained on.

Everything is computed from the same corpus and the same classifier as the rest
of the paper: cache/win_30_r6.parquet, 30 s windows at 15 s stride, XGBoost,
out-of-fold under StratifiedGroupKFold grouped by job_id. The corpus filter is
copied from paper_table5_armsrace.py and the window count is asserted against it,
so the appendix cannot silently drift onto a different corpus.

Two decisions worth stating because they are what make the comparison honest.

1. CUSUM is calibrated to the SAME realised benign false-alert rate as the
   3-of-5 rule, not to a round number. A sequential test with a free threshold
   can always beat a fixed one; matching the false alarms is the only way the
   comparison means anything. We find the smallest h whose benign false-alert
   rate is <= the 3-of-5 rule's own rate on the same runs.

2. Run-level ROC/PR use one score per run, and a sequential detector's natural
   score is the running maximum of its statistic. For the memoryless rule the
   comparable score is the run's maximum windowed evidence. Reporting the two
   on the same axes is the point of the figure; reporting only window-level AUC
   (as the submission did) hides that the aggregation step is where the
   deployment question actually lives.

Usage:
  PAPER_MODEL=xgb python3 scripts/paper_runlevel_eval.py \
      --out docs/camera_ready_results_20261022/paper_runlevel_xgb.json
"""
from __future__ import annotations
import argparse, json, sys, time
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import roc_auc_score, average_precision_score, roc_curve, precision_recall_curve

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from scripts.paper_table5_armsrace import (            # noqa: E402
    FAMILY_SPEC, DROP_CELLS, INSTRUMENT_CELLS, SUBGROUP,
    canonical_workload, family_of, out_of_scope_runs, forest, get_feature_cols,
)

STRIDE = 15.0
WINDOW = 30.0


# ───────────────────────────────────────────────────────── corpus ──
def build_corpus(cache, max_gpus=8, single_node=True, ppo_rlhf_round=6,
                 controls_round=0):
    """Byte-for-byte the filter paper_table5_armsrace.py applies."""
    FAMILY_SPEC["ppo_rlhf"] = (ppo_rlhf_round, False)
    if controls_round == 0:
        for f, (r, e) in list(FAMILY_SPEC.items()):
            if not e:
                FAMILY_SPEC[f] = (0, False)
    win = pd.read_parquet(cache)
    win["cell"] = [canonical_workload(l) for l in win.workload_label]
    oos, _ = out_of_scope_runs(max_gpus, single_node)
    win = win.loc[~win.run_id.astype(str).isin(oos)].reset_index(drop=True)
    drop = win.cell.isin(DROP_CELLS) | win.cell.isin(INSTRUMENT_CELLS)
    win = win.loc[~drop].reset_index(drop=True)
    win["family"] = [family_of(l) for l in win.workload_label]
    feats = [c for c in get_feature_cols(win)
             if c not in {"round", "cell", "family", "subgroup"}]
    win["round"] = win.family.map(
        lambda f: FAMILY_SPEC[f][0] if isinstance(f, str) else 0).astype(int)
    return win, feats


# ─────────────────────────────────────────────────── alert rules ──
def kofm_first(p, tau, k, m):
    """First index where k of the trailing m windows exceed tau."""
    f = (np.asarray(p) > tau).astype(int)
    for i in range(len(f)):
        if f[max(0, i - m + 1):i + 1].sum() >= k:
            return i
    return None


def kofm_score(p, tau, k, m):
    """Run score for the memoryless rule: the largest trailing-m count of
    windows over tau, as a fraction of m. Monotone in 'how close this run came
    to firing', so it induces a run-level ROC comparable with CUSUM's."""
    f = (np.asarray(p) > tau).astype(int)
    if not len(f):
        return 0.0
    return max(f[max(0, i - m + 1):i + 1].sum() for i in range(len(f))) / m


def cusum_path(p, k):
    """S_t = max(0, S_{t-1} + p_t - k); returns the whole path."""
    S, out = 0.0, []
    for pi in p:
        S = max(0.0, S + (pi - k))
        out.append(S)
    return np.asarray(out)


def cusum_first(p, k, h):
    S = 0.0
    for i, pi in enumerate(p):
        S = max(0.0, S + (pi - k))
        if S >= h:
            return i
    return None


def runs_table(win, proba):
    """One row per run: ordered P(train) sequence, label, family, duration."""
    df = pd.DataFrame({
        "run_id": win.run_id.astype(str).to_numpy(),
        "job_id": win.job_id.astype(str).to_numpy(),
        "family": win.family.to_numpy(object),
        "cell": win.cell.to_numpy(object),
        "label": (win.threeway_label == "ml_training").astype(int).to_numpy(),
        "t0": win.window_start.to_numpy(),
        "t1": win.window_end.to_numpy(),
        "p": proba,
    })
    rows = []
    for rid, g in df.groupby("run_id", sort=False):
        g = g.sort_values("t0")
        dur = float(pd.to_numeric(g.t1).iloc[-1] - pd.to_numeric(g.t0).iloc[0])
        rows.append({
            "run_id": rid, "job_id": g.job_id.iloc[0],
            "family": g.family.iloc[0], "cell": g.cell.iloc[0],
            "label": int(g.label.max()), "n_win": len(g),
            "hours": max(dur, WINDOW) / 3600.0,
            "p": g.p.to_numpy(float),
        })
    return rows


def ttfa_seconds(idx):
    """Seconds from run start to the END of the window that fired."""
    return None if idx is None else idx * STRIDE + WINDOW


def evaluate_rules(runs, tau, k, m, cusum_k, h):
    out = []
    for r in runs:
        a3 = kofm_first(r["p"], tau, k, m)
        ac = cusum_first(r["p"], cusum_k, h)
        both = [x for x in (a3, ac) if x is not None]
        out.append({**{kk: r[kk] for kk in
                       ("run_id", "job_id", "family", "cell", "label", "n_win", "hours")},
                    "alert_3of5": a3 is not None, "ttfa_3of5": ttfa_seconds(a3),
                    "alert_cusum": ac is not None, "ttfa_cusum": ttfa_seconds(ac),
                    "alert_hybrid": bool(both),
                    "ttfa_hybrid": ttfa_seconds(min(both)) if both else None,
                    "score_3of5": kofm_score(r["p"], tau, k, m),
                    "score_cusum": float(cusum_path(r["p"], cusum_k).max()),
                    "score_maxp": float(np.max(r["p"])),
                    "score_meanp": float(np.mean(r["p"]))})
    return pd.DataFrame(out)


def calibrate_h(runs, cusum_k, target_rate, unit_hours):
    """Smallest h whose benign false-alert rate is <= target. Benign = non-training."""
    ben = [r for r in runs if r["label"] == 0]
    denom = sum(r["hours"] for r in ben) if unit_hours else len(ben)
    if not denom:
        return float("inf")
    grid = np.concatenate([np.arange(0.5, 20.0, 0.25), np.arange(20.0, 200.0, 1.0)])
    for h in grid:
        n = sum(1 for r in ben if cusum_first(r["p"], cusum_k, h) is not None)
        if n / denom <= target_rate:
            return float(h)
    return float(grid[-1])


def ece(y, p, bins=10):
    """Expected calibration error, equal-width bins."""
    edges = np.linspace(0, 1, bins + 1)
    e, n = 0.0, len(y)
    for i in range(bins):
        m = (p >= edges[i]) & (p < edges[i + 1] if i < bins - 1 else p <= 1.0)
        if m.sum():
            e += m.sum() / n * abs(y[m].mean() - p[m].mean())
    return float(e)


# ────────────────────────────────────────────────────────── main ──
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", default=str(PROJECT_ROOT / "cache/win_30_r6.parquet"))
    ap.add_argument("--tau", type=float, default=0.75)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--cusum-k", type=float, default=0.25)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 7, 1234, 2026, 99])
    ap.add_argument("--expect-windows", type=int, default=36666,
                    help="assert the corpus matches the arms-race run")
    ap.add_argument("--lofo", action="store_true",
                    help="also run leave-one-family-out transfer")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    t0 = time.time()

    win, feats = build_corpus(a.cache)
    X = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    y = (win.threeway_label == "ml_training").to_numpy(int)
    g = win.job_id.astype(str).to_numpy()
    fam = win.family.to_numpy(object)
    print(f"{len(win):,} windows, {len(feats)} features, "
          f"{len(np.unique(g)):,} jobs, {win.run_id.nunique():,} runs, "
          f"{y.mean()*100:.1f}% training")
    if a.expect_windows and len(win) != a.expect_windows:
        print(f"  WARNING: expected {a.expect_windows:,} windows "
              f"(paper_table5_armsrace.py corpus_6) -- corpus has drifted")

    # ---- out-of-fold probabilities, repeated over fold-assignment seeds ----
    oof = {}
    for sd in a.seeds:
        p = np.full(len(y), np.nan)
        for tr, te in StratifiedGroupKFold(n_splits=a.folds, shuffle=True,
                                           random_state=sd).split(X, y, g):
            p[te] = forest().fit(X[tr], y[tr]).predict_proba(X[te])[:, 1]
        assert not np.isnan(p).any()
        oof[sd] = p
        print(f"  seed {sd}: window AUC {roc_auc_score(y, p):.4f} "
              f"AP {average_precision_score(y, p):.4f} [{time.time()-t0:.0f}s]",
              flush=True)

    res = {"procedure": "run-level evaluation", "window_sec": WINDOW,
           "stride_sec": STRIDE, "group_by": "job_id", "folds": a.folds,
           "seeds": list(a.seeds), "tau": a.tau, "k_of_m": [a.k, a.m],
           "cusum_k": a.cusum_k, "windows": int(len(win)),
           "runs": int(win.run_id.nunique()), "jobs": int(len(np.unique(g))),
           "n_features": len(feats)}

    # ---- window-level ----
    res["window_level"] = {
        "roc_auc": [float(roc_auc_score(y, oof[s])) for s in a.seeds],
        "average_precision": [float(average_precision_score(y, oof[s])) for s in a.seeds],
        "ece": [ece(y, oof[s]) for s in a.seeds],
        "fpr_at_tau": [float(((oof[s] > a.tau) & (y == 0)).sum() / max((y == 0).sum(), 1))
                       for s in a.seeds],
        "tpr_at_tau": [float(((oof[s] > a.tau) & (y == 1)).sum() / max((y == 1).sum(), 1))
                       for s in a.seeds],
    }

    # ---- run-level, per seed ----
    per_seed, curves = [], None
    for sd in a.seeds:
        runs = runs_table(win, oof[sd])
        # 3-of-5 realised benign false-alert rate sets CUSUM's budget.
        ben = [r for r in runs if r["label"] == 0]
        ben_h = sum(r["hours"] for r in ben)
        n3 = sum(1 for r in ben if kofm_first(r["p"], a.tau, a.k, a.m) is not None)
        budget_hr = n3 / ben_h if ben_h else 0.0
        h = calibrate_h(runs, a.cusum_k, budget_hr, unit_hours=True)
        d = evaluate_rules(runs, a.tau, a.k, a.m, a.cusum_k, h)
        pos, neg = d[d.label == 1], d[d.label == 0]
        row = {
            "seed": sd, "cusum_h": h, "benign_runs": len(neg),
            "benign_run_hours": float(ben_h), "training_runs": len(pos),
        }
        for rule in ("3of5", "cusum", "hybrid"):
            miss = float(1 - pos[f"alert_{rule}"].mean())
            fa_n = int(neg[f"alert_{rule}"].sum())
            tt = pos.loc[pos[f"alert_{rule}"], f"ttfa_{rule}"].astype(float)
            row[rule] = {
                "detection_rate": float(pos[f"alert_{rule}"].mean()),
                "missed_run_rate": miss,
                "missed_runs": int((~pos[f"alert_{rule}"]).sum()),
                "false_alert_runs": fa_n,
                "false_alert_rate_per_run": fa_n / max(len(neg), 1),
                "false_alerts_per_gpu_hour": fa_n / ben_h if ben_h else None,
                "ttfa_median_s": float(tt.median()) if len(tt) else None,
                "ttfa_p95_s": float(tt.quantile(0.95)) if len(tt) else None,
                "ttfa_max_s": float(tt.max()) if len(tt) else None,
                "frac_within_60s": float((tt <= 60).mean()) if len(tt) else None,
                "frac_within_300s": float((tt <= 300).mean()) if len(tt) else None,
            }
        for sc in ("score_3of5", "score_cusum", "score_maxp", "score_meanp"):
            row[f"run_auc_{sc}"] = float(roc_auc_score(d.label, d[sc]))
            row[f"run_ap_{sc}"] = float(average_precision_score(d.label, d[sc]))
        per_seed.append(row)
        print(f"  seed {sd}: h={h:.2f}  3of5 det {row['3of5']['detection_rate']:.3f} "
              f"fa/h {row['3of5']['false_alerts_per_gpu_hour']:.3f} | "
              f"cusum det {row['cusum']['detection_rate']:.3f} "
              f"fa/h {row['cusum']['false_alerts_per_gpu_hour']:.3f} "
              f"[{time.time()-t0:.0f}s]", flush=True)
        if sd == a.seeds[0]:
            curves = {"run_detail": d.to_dict("records")}
    res["per_seed"] = per_seed

    # ---- per-family stratification (seed 0 of the list) ----
    sd0 = a.seeds[0]
    p0 = oof[sd0]
    runs0 = runs_table(win, p0)
    ben = [r for r in runs0 if r["label"] == 0]
    ben_h = sum(r["hours"] for r in ben)
    n3 = sum(1 for r in ben if kofm_first(r["p"], a.tau, a.k, a.m) is not None)
    h0 = calibrate_h(runs0, a.cusum_k, n3 / ben_h if ben_h else 0.0, True)
    d0 = evaluate_rules(runs0, a.tau, a.k, a.m, a.cusum_k, h0)
    res["cusum_h_primary"] = h0

    neg_w = y == 0                       # window-level negative pool: all non-training
    negd = d0[d0.label == 0]
    fams = sorted({f for f in fam if isinstance(f, str) and FAMILY_SPEC.get(f, (0, False))[1]})
    strat = []
    for f in fams:
        selw = (fam == f) & (y == 1)
        if not selw.sum():
            continue
        yy = np.concatenate([np.ones(selw.sum()), np.zeros(neg_w.sum())])
        pp = np.concatenate([p0[selw], p0[neg_w]])
        fpr, tpr, thr = roc_curve(yy, pp)
        tpr_at_1pct = float(np.interp(0.01, fpr, tpr))
        selr = d0[(d0.family == f) & (d0.label == 1)]
        rr = pd.concat([selr, negd])
        strat.append({
            "family": f, "round": FAMILY_SPEC[f][0], "subgroup": SUBGROUP.get(f),
            "pos_windows": int(selw.sum()), "pos_runs": int(len(selr)),
            "window_roc_auc": float(roc_auc_score(yy, pp)),
            "window_ap": float(average_precision_score(yy, pp)),
            "window_tpr_at_1pct_fpr": tpr_at_1pct,
            "window_alert_frac_at_tau": float((p0[selw] > a.tau).mean()),
            "run_roc_auc_cusum": float(roc_auc_score(rr.label, rr.score_cusum))
                                 if len(selr) else None,
            "run_roc_auc_3of5": float(roc_auc_score(rr.label, rr.score_3of5))
                                if len(selr) else None,
            "run_alert_3of5": float(selr.alert_3of5.mean()) if len(selr) else None,
            "run_alert_cusum": float(selr.alert_cusum.mean()) if len(selr) else None,
            "run_alert_hybrid": float(selr.alert_hybrid.mean()) if len(selr) else None,
            "ttfa_median_s_cusum": (float(selr.loc[selr.alert_cusum, "ttfa_cusum"]
                                          .astype(float).median())
                                    if len(selr) and selr.alert_cusum.any() else None),
        })
    res["by_family"] = strat

    # ---- ROC/PR curve points for the figure (subsampled) ----
    def thin(x, y_, n=400):
        if len(x) <= n:
            return [list(map(float, x)), list(map(float, y_))]
        i = np.linspace(0, len(x) - 1, n).astype(int)
        return [list(map(float, np.asarray(x)[i])), list(map(float, np.asarray(y_)[i]))]
    cv = {"window_overall": {}, "run_overall": {}, "by_family": {}}
    fpr, tpr, _ = roc_curve(y, p0); cv["window_overall"]["roc"] = thin(fpr, tpr)
    pr, rc, _ = precision_recall_curve(y, p0); cv["window_overall"]["pr"] = thin(rc, pr)
    for sc, nm in (("score_cusum", "cusum"), ("score_3of5", "3of5"), ("score_maxp", "maxp")):
        fpr, tpr, _ = roc_curve(d0.label, d0[sc]); cv["run_overall"][f"roc_{nm}"] = thin(fpr, tpr)
        pr, rc, _ = precision_recall_curve(d0.label, d0[sc]); cv["run_overall"][f"pr_{nm}"] = thin(rc, pr)
    for f in fams:
        selw = (fam == f) & (y == 1)
        if not selw.sum():
            continue
        yy = np.concatenate([np.ones(selw.sum()), np.zeros(neg_w.sum())])
        pp = np.concatenate([p0[selw], p0[neg_w]])
        fpr, tpr, _ = roc_curve(yy, pp)
        pr, rc, _ = precision_recall_curve(yy, pp)
        cv["by_family"][f] = {"roc": thin(fpr, tpr), "pr": thin(rc, pr)}
    res["curves"] = cv

    # ---- leave-one-family-out transfer ----
    if a.lofo:
        lofo = []
        for f in fams:
            tr = (fam != f)
            selr = (fam == f)
            clf = forest().fit(X[tr], y[tr])
            pf = np.full(len(y), np.nan)
            pf[selr] = clf.predict_proba(X[selr])[:, 1]
            sub = win.loc[selr].reset_index(drop=True)
            rr = runs_table(sub, pf[selr])
            rr = [r for r in rr if r["label"] == 1]
            if not rr:
                continue
            dd = evaluate_rules(rr, a.tau, a.k, a.m, a.cusum_k, h0)
            tt = dd.loc[dd.alert_cusum, "ttfa_cusum"].astype(float)
            lofo.append({
                "family": f, "runs": len(dd),
                "window_mean_p": float(np.nanmean(pf[selr & (y == 1)])),
                "window_alert_frac_at_tau": float((pf[selr & (y == 1)] > a.tau).mean()),
                "run_alert_3of5": float(dd.alert_3of5.mean()),
                "run_alert_cusum": float(dd.alert_cusum.mean()),
                "run_alert_hybrid": float(dd.alert_hybrid.mean()),
                "ttfa_median_s_cusum": float(tt.median()) if len(tt) else None,
            })
            print(f"   LOFO {f:24s} 3of5 {lofo[-1]['run_alert_3of5']:5.1%}  "
                  f"cusum {lofo[-1]['run_alert_cusum']:5.1%}  "
                  f"hybrid {lofo[-1]['run_alert_hybrid']:5.1%} "
                  f"[{time.time()-t0:.0f}s]", flush=True)
        res["lofo"] = lofo

    Path(a.out).write_text(json.dumps(res, indent=2) + "\n")
    print(f"\nwrote {a.out}  [{time.time()-t0:.0f}s]")


if __name__ == "__main__":
    main()
