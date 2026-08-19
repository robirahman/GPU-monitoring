#!/usr/bin/env python3
"""Scope-ladder hypothesis test: does train-vs-inference separability increase
with communication scope (S1 < S8 < S16)?

For each cell (scope x inference-negative x feature-arm x CV-grouping x
channel-ablation) it trains a binary train-vs-inference RandomForest on
shape-only features, takes out-of-fold P(train), and reports window AUC / AP /
TPR@1%FPR plus job-level detection rate and median time-to-first-alert. The
headline is a job-cluster bootstrap of DeltaAUC(S8-S1) and DeltaAUC(S16-S8):
the hypothesis PASSES a negative iff both deltas are positive at p<0.05.

Three feature arms make the observer-model assumption explicit (see
classifier/multinode_features.py):
  pergpu       paper-faithful: per-GPU-stream features, no cross-stream join.
  pergpu_plus  adds the comm channels (NVLink/NIC), still no cross-rank join.
  crossrank    adds the cross-rank correlation/cadence/lag features.

Run on the real corpus once S1/S8/S16 cells exist:
    python -m classifier.rebuttal_analysis.scope_ladder_eval
Validate the machinery locally with no cloud data:
    python -m classifier.rebuttal_analysis.scope_ladder_eval --synthetic
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve
from sklearn.model_selection import StratifiedGroupKFold

from classifier.threeway_improved import sliding_windows, get_feature_cols
from classifier import multinode_features as mf
from classifier import run_level_detection as rl

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
RESULTS_DIR = PROJECT_ROOT / "classifier" / "results"
FIG_DIR = PROJECT_ROOT / "report" / "figures"

ARMS = ("pergpu", "pergpu_plus", "crossrank")
# prefix families dropped by each channel ablation
ABLATIONS = {
    "none": (),
    "drop_pcie": ("pcie",),
    "drop_nvlink": ("nvlink",),
    "drop_nic": ("nic",),
    "drop_all_comm": ("pcie", "nvlink", "nic", "hlo", "dcn", "xragg"),
    "xr_only": "XR_ONLY",
}
RF_KW = dict(n_estimators=400, max_depth=None, min_samples_leaf=2,
             max_features="sqrt", class_weight="balanced", random_state=42, n_jobs=-1)


# ── window construction per arm ──────────────────────────────────────────────

def build_windows(df: pd.DataFrame, arm: str, window: float, stride: float) -> pd.DataFrame:
    """Return a window table with columns incl. run_id (=job for grouping),
    job_id, scope, threeway_label, workload_label, window_start, + features."""
    if arm == "pergpu":
        # Per-stream windows via the deployed extractor. Group each GPU stream
        # independently by temporarily using stream_id as run_id, then attach
        # the job id back for grouped CV.
        ided = mf.assign_stream_and_job_ids(df)
        s2j = ided.drop_duplicates("stream_id").set_index("stream_id")["job_id"]
        s2scope = (ided.drop_duplicates("stream_id").set_index("stream_id")["workload_label"]
                   .map(mf._scope_of))
        per = ided.copy()
        per["run_id"] = per["stream_id"]
        win = sliding_windows(per, window, stride, run_level_mode="causal")
        if win.empty:
            return win
        win["job_id"] = win["run_id"].map(s2j)
        win["scope"] = win["run_id"].map(s2scope)
        return win

    # pergpu_plus and crossrank both come from the job-level windower.
    dfc = mf.normalize_comm_columns(df)
    win = mf.sliding_windows_multinode(dfc, window, stride)
    if win.empty:
        return win
    win["job_id"] = win["run_id"]
    win["scope"] = win["meta_scope"]
    if arm == "pergpu_plus":
        # keep single-stream + comm-channel features, drop the cross-rank join
        drop = [c for c in win.columns if c.startswith("xr_")]
        win = win.drop(columns=drop)
    return win


def select_features(win: pd.DataFrame, ablation: str) -> list:
    feats = get_feature_cols(win)
    spec = ABLATIONS[ablation]
    if spec == "XR_ONLY":
        return [c for c in feats if c.startswith("xr_")]
    return [c for c in feats if not any(tok in c for tok in spec)]


# ── metrics ──────────────────────────────────────────────────────────────────

def tpr_at_fpr(y, p, fpr_target=0.01) -> float:
    if len(np.unique(y)) < 2:
        return float("nan")
    fpr, tpr, _ = roc_curve(y, p)
    ok = fpr <= fpr_target
    return float(tpr[ok].max()) if ok.any() else 0.0


def oof_probs(X, y, groups, n_splits):
    """Out-of-fold P(train) via grouped stratified CV. Returns (p, folds_used)."""
    ng = pd.Series(groups).nunique()
    k = max(2, min(n_splits, ng, int(pd.Series(y).value_counts().min())))
    if k < 2:
        return None, 0
    skf = StratifiedGroupKFold(n_splits=k, shuffle=True, random_state=42)
    p = np.full(len(y), np.nan)
    for tr, te in skf.split(X, y, groups):
        clf = RandomForestClassifier(**RF_KW).fit(X[tr], y[tr])
        p[te] = clf.predict_proba(X[te])[:, 1]
    return p, k


def eval_cell(win, feats, scope, neg_type, n_splits=5):
    """Binary train-vs-(neg_type) within one scope. Returns metrics + an OOF
    per-window frame (for run-level detection)."""
    sub = win[win["scope"] == scope].copy()
    is_train = sub["threeway_label"] == "ml_training"
    is_neg = (sub["threeway_label"] == "ml_inference") & \
             sub["workload_label"].str.contains(neg_type, case=False, na=False)
    sub = sub[is_train | is_neg]
    if sub.empty:
        return None
    y = (sub["threeway_label"] == "ml_training").astype(int).values
    groups = sub["job_id"].astype(str).values
    n_pos_jobs = pd.Series(groups[y == 1]).nunique()
    n_neg_jobs = pd.Series(groups[y == 0]).nunique()
    if n_pos_jobs < 2 or n_neg_jobs < 2:
        return None
    X = sub[feats].fillna(0).values.astype(np.float32)
    p, k = oof_probs(X, y, groups, n_splits)
    if p is None or np.isnan(p).all():
        return None
    keep = ~np.isnan(p)
    y, p, groups = y[keep], p[keep], groups[keep]
    metrics = {
        "auc": float(roc_auc_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "ap": float(average_precision_score(y, p)) if len(np.unique(y)) > 1 else float("nan"),
        "tpr_at_1pct_fpr": tpr_at_fpr(y, p),
        "n_windows_pos": int((y == 1).sum()), "n_windows_neg": int((y == 0).sum()),
        "n_jobs_pos": int(n_pos_jobs), "n_jobs_neg": int(n_neg_jobs), "cv_folds": k,
        "low_n": bool(min(n_pos_jobs, n_neg_jobs) < 8),
    }
    oof = sub.loc[keep, ["job_id", "workload_label", "threeway_label", "window_start"]].copy()
    oof["p_train"] = p
    # per-job AUC contribution for the bootstrap: mean p_train per job + label
    oof["y"] = y
    return {"metrics": metrics, "oof": oof}


def runlevel(oof, k=3, m=5, target_fp=0.1):
    """Job-level detection rate + median TTFA at a tau calibrated on the
    benign (inference) jobs. Aggregates stream windows to one sequence per job."""
    seq = (oof.groupby(["job_id", "window_start"])
           .agg(p_train=("p_train", "mean"),
                threeway_label=("threeway_label", "first"),
                workload_label=("workload_label", "first"))
           .reset_index().rename(columns={"job_id": "run_id"}))
    seqs = list(rl.per_run_sequences(seq))
    benign = [s for s in seqs if not s[2]]
    train = [s for s in seqs if s[2]]
    if not benign or not train:
        return {"tau": None, "det_rate": float("nan"), "median_ttfa_s": float("nan")}
    tau, _, _ = rl.calibrate_tau(benign, k, m, target_fp,
                                 taus=np.round(np.arange(0.05, 0.96, 0.05), 2))
    if tau is None:
        tau = 0.75
    det, ttfas = 0, []
    for _, _, _, p, ws, stride in train:
        idx = rl.kofm_trigger(p > tau, k, m)
        if idx is not None:
            det += 1
            s = stride if np.isfinite(stride) else 15.0
            ttfas.append((ws[idx] - ws[0]) if len(ws) else idx * s)
    return {"tau": float(tau), "det_rate": det / len(train),
            "median_ttfa_s": float(np.median(ttfas)) if ttfas else float("nan"),
            "n_train_jobs": len(train), "n_benign_jobs": len(benign)}


def bootstrap_delta(oof_by_scope, scope_hi, scope_lo, B=2000, seed=42):
    """One-sided job-cluster bootstrap of AUC(scope_hi) - AUC(scope_lo)."""
    if scope_hi not in oof_by_scope or scope_lo not in oof_by_scope:
        return None
    rng = np.random.RandomState(seed)

    def resampled_auc(oof):
        jobs = oof["job_id"].unique()
        pick = rng.choice(jobs, size=len(jobs), replace=True)
        parts = [oof[oof["job_id"] == j] for j in pick]
        s = pd.concat(parts, ignore_index=True)
        if s["y"].nunique() < 2:
            return np.nan
        return roc_auc_score(s["y"].values, s["p_train"].values)

    deltas = []
    for _ in range(B):
        a = resampled_auc(oof_by_scope[scope_hi])
        b = resampled_auc(oof_by_scope[scope_lo])
        if np.isfinite(a) and np.isfinite(b):
            deltas.append(a - b)
    if not deltas:
        return None
    deltas = np.array(deltas)
    return {"delta_mean": float(deltas.mean()),
            "ci_lo": float(np.percentile(deltas, 2.5)),
            "ci_hi": float(np.percentile(deltas, 97.5)),
            "p_delta_le_0": float((deltas <= 0).mean())}


# ── synthetic corpus for local validation ───────────────────────────────────

def synthetic_corpus(n_jobs_per_cell=4, seed=0) -> pd.DataFrame:
    """Build s1/s8/s16 x {train, inferrep, infershard} jobs whose cross-rank
    structure follows the hypothesis, so the harness can be validated with no
    cloud data. Separability is engineered to rise with scope."""
    rng = np.random.RandomState(seed)
    from classifier.threeway_improved import threeway_label
    rows = []
    scope_ranks = {"s1": 1, "s8": 8, "s16": 16}
    scope_nodes = {"s1": 1, "s8": 1, "s16": 2}
    T = 120

    def add(job, label, ranks, nodes, coupling, cadence, t0):
        base = 55 + 25 * np.sin(np.linspace(0, cadence, T))
        for r in range(ranks):
            node = f"host{r // (ranks // nodes)}" if nodes > 1 else "host0"
            if coupling == "sync":
                util = base + rng.normal(0, 3, T)
            elif coupling == "indep":
                util = 55 + 25 * np.sin(np.linspace(0, cadence, T) + rng.uniform(0, 6)) + rng.normal(0, 3, T)
            else:  # partial: coupled within node only
                phase = 0.0 if node == "host0" else 2.5
                util = 55 + 25 * np.sin(np.linspace(0, cadence, T) + phase) + rng.normal(0, 4, T)
            power = 320 + 2.2 * util + rng.normal(0, 4, T)
            for t in range(T):
                rows.append(dict(timestamp_epoch=t0 + t, run_id=job,
                                 gpu_uuid=f"{job}-g{r}", hostname=node,
                                 gpu_name="NVIDIA H100 80GB HBM3", workload_label=label,
                                 gpu_utilization_pct=max(0.0, util[t]), power_draw_w=power[t],
                                 mem_used_mb=42000, mem_total_mb=81920,
                                 encoder_util_pct=0.0, decoder_util_pct=0.0))

    # Stagger every job's start time (>> the 20 s job bucket) so that
    # single-stream reps -- which fall on the legacy label|gpu|start job key --
    # separate into distinct jobs, exactly as minutes-apart real reps do.
    t0 = 2000.0
    for scope, ranks in scope_ranks.items():
        nodes = scope_nodes[scope]
        for i in range(n_jobs_per_cell):
            add(f"{scope}_tr_{i}", f"{scope}_train_fsdp7b", ranks, nodes,
                "sync" if ranks > 1 else "indep", 6, t0); t0 += 600
            add(f"{scope}_rep_{i}", f"{scope}_inferrep_qwen7b", ranks, nodes, "indep", 11, t0); t0 += 600
            if ranks > 1:
                add(f"{scope}_sh_{i}", f"{scope}_infershard_vllm", ranks, nodes, "partial", 9, t0); t0 += 600
    df = pd.DataFrame(rows)
    from classifier.threeway_improved import _normalize_columns
    df = _normalize_columns(df)
    df["threeway_label"] = df["workload_label"].apply(threeway_label)
    return df


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=15.0)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--target-fp-per-gpu-hour", type=float, default=0.1)
    ap.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    ap.add_argument("--negatives", nargs="+", default=["inferrep", "infershard"])
    ap.add_argument("--ablations", nargs="+", default=list(ABLATIONS.keys()))
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--synthetic", action="store_true",
                    help="Validate the harness on a built-in synthetic corpus.")
    ap.add_argument("--out-prefix", default="scope_ladder")
    args = ap.parse_args()

    if args.synthetic:
        print("Loading SYNTHETIC corpus (harness validation, not real data)")
        df = synthetic_corpus()
    else:
        from classifier.threeway_improved import load_all_data
        df = mf.attach_nic_sidecar(load_all_data())

    scopes_present = sorted({mf._scope_of(l) for l in df["workload_label"].unique()} - {"na"})
    print(f"Scopes present: {scopes_present}")
    if not scopes_present:
        print("No scope-ladder cells found (labels need an s1_/s8_/s16_/tpuT* prefix). "
              "Nothing to evaluate; run with --synthetic to validate the harness.")
        return

    summary_rows, mono_rows = [], []
    for arm in args.arms:
        win = build_windows(df, arm, args.window, args.stride)
        if win.empty:
            print(f"[{arm}] no windows"); continue
        for neg in args.negatives:
            for ab in args.ablations:
                feats = select_features(win, ab)
                if not feats:
                    continue
                oof_by_scope = {}
                for scope in scopes_present:
                    res = eval_cell(win, feats, scope, neg)
                    if res is None:
                        continue
                    m = res["metrics"]
                    rlm = runlevel(res["oof"], args.k, args.m, args.target_fp_per_gpu_hour)
                    oof_by_scope[scope] = res["oof"]
                    summary_rows.append({
                        "arm": arm, "negative": neg, "ablation": ab, "scope": scope,
                        **m, **{f"rl_{k}": v for k, v in rlm.items()}})
                    tag = f"{arm:11s} {neg:11s} {ab:14s} {scope:5s}"
                    print(f"  {tag}  AUC={m['auc']:.3f} TPR@1%={m['tpr_at_1pct_fpr']:.3f} "
                          f"jobs={m['n_jobs_pos']}/{m['n_jobs_neg']}"
                          f"{'  [LOW-N]' if m['low_n'] else ''}")
                # monotonicity for this (arm, neg, ablation)
                for hi, lo in (("s8", "s1"), ("s16", "s8")):
                    d = bootstrap_delta(oof_by_scope, hi, lo, B=args.bootstrap)
                    if d:
                        mono_rows.append({"arm": arm, "negative": neg, "ablation": ab,
                                          "delta": f"{hi}-{lo}", **d})

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    summ = pd.DataFrame(summary_rows)
    mono = pd.DataFrame(mono_rows)
    sp = RESULTS_DIR / f"{args.out_prefix}_summary.csv"
    mp = RESULTS_DIR / f"{args.out_prefix}_monotonicity.csv"
    summ.to_csv(sp, index=False)
    mono.to_csv(mp, index=False)

    # headline monotonicity verdict (per arm x negative, ablation=none)
    print("\n=== Monotonicity (ablation=none): PASS iff both deltas > 0 at p<0.05 ===")
    if not mono.empty:
        base = mono[mono["ablation"] == "none"]
        for (arm, neg), g in base.groupby(["arm", "negative"]):
            d = {r["delta"]: r for _, r in g.iterrows()}
            ok = all((d.get(x, {}).get("delta_mean", -1) > 0 and d.get(x, {}).get("p_delta_le_0", 1) < 0.05)
                     for x in ("s8-s1", "s16-s8") if x in d)
            deltas_str = "  ".join(f"{x}:{d[x]['delta_mean']:+.3f}(p={d[x]['p_delta_le_0']:.3f})"
                                   for x in ("s8-s1", "s16-s8") if x in d)
            print(f"  {arm:11s} {neg:11s} {'PASS' if ok and deltas_str else 'n/a ':4s}  {deltas_str}")

    result = {"summary_csv": str(sp), "monotonicity_csv": str(mp),
              "scopes": scopes_present, "arms": list(args.arms),
              "config": vars(args)}
    jp = Path(__file__).resolve().parent / f"{args.out_prefix}_eval.json"
    jp.write_text(json.dumps(result, indent=2, default=str))
    print(f"\nSaved: {sp}\n       {mp}\n       {jp}")


if __name__ == "__main__":
    main()
