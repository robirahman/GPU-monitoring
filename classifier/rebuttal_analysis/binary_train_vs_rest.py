#!/usr/bin/env python3
"""Training vs NON-training on the honest corpus, held out by workload.

The deployed detector is three-way (training / inference / other), but the
operational question is binary: is this machine training a model or not. This
scores exactly that -- p_train against everything else -- on non-adversarial
cells only, so it is the BASELINE an evasion attempt has to be measured against.

Held out by WORKLOAD, not by run. classifier/rebuttal_analysis/
workload_grouped_tuning.md established the gap: run-grouped CV scored 0.9628
accuracy against workload-grouped 0.8470 on the same model, so ~11.6 points was
the model recognising which workload it was looking at rather than whether that
workload trains. A held-out attacker is by definition a workload the detector
has not seen, so workload-grouped is the honest protocol.

TWO CORRECTIONS relative to the 2026-09-16 run of this script:

  * WINDOWS ARE BUILT PER STREAM. sliding_windows groups by run_id, and both the
    TPU queue and the multi-node GPU queue deliberately mandate ONE shared run_id
    across every worker of a cell -- all 16 chips of
    tpuT16_advtrain_satfiller_only_rep1 carry run_id=c106c001eec0 with 16
    distinct gpu_uuids. Grouping by run_id interleaves those 16 chips into a
    single time series, so a 30 s window holds ~480 samples from 16 devices
    instead of 30 from one and every acf / fft / transition feature becomes a
    chip-ordering artifact. The 09-16 numbers were computed that way and are
    superseded. Build the window cache with scripts that force run_id=stream_id
    (see --windows).
  * RUN-LEVEL ALERTING IS REPORTED PER STREAM *AND* PER JOB. Section 30 of
    tpu_multislice_findings.md established the k-of-m rule is sensitive to which
    one you use: per-stream inflates the denominator 16x on a 16-chip job, while
    per-job is what an operator watching one execution actually sees.

  PYTHONPATH=. python3 classifier/rebuttal_analysis/binary_train_vs_rest.py \
      --windows /path/to/win_stream_30_15.parquet
"""
import argparse
import datetime
import json
import os
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(
    os.path.abspath(__file__)))))

from classifier.threeway_improved import (  # noqa: E402
    load_all_data, sliding_windows, get_feature_cols, workload_group)
from classifier.multinode_features import assign_stream_and_job_ids  # noqa: E402
from classifier.twostage import TwoStageClassifier  # noqa: E402
from sklearn.model_selection import GroupKFold  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, precision_recall_fscore_support, roc_auc_score,
    confusion_matrix)

ADV = re.compile(r"(advtrain|adversarial|whitebox|symgemm|typeconf|stalled)", re.I)


def platform_of(label, gpu_name):
    """Coarse hardware family, for the per-platform breakdown."""
    g = str(gpu_name)
    if "TPU" in g:
        return "TPU " + (g.split("TPU")[-1].strip().split()[0] if "TPU" in g else "?")
    if "AMD" in g or "Radeon" in g:
        return "AMD"
    for tag in ("H100", "A100", "B200", "L4", "T4", "V100", "5090", "5080", "4090"):
        if tag in g.replace(" ", ""):
            return "NVIDIA " + tag
    return "NVIDIA other" if g else "unknown"


def build_windows(window, stride):
    """Window per STREAM, not per run_id. See the module docstring."""
    df = load_all_data()
    df = assign_stream_and_job_ids(df)
    stream_to_job = (df[["stream_id", "job_id"]].drop_duplicates()
                     .set_index("stream_id")["job_id"])
    df["run_id"] = df["stream_id"]
    w = sliding_windows(df, window, stride)
    del df
    w["job_id"] = w.run_id.map(stream_to_job).fillna(w.run_id)
    w["grp"] = workload_group(w.workload_label.values)
    return w


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=15.0)
    ap.add_argument("--tau", type=float, default=0.8)
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--k", type=int, default=3)
    ap.add_argument("--m", type=int, default=5)
    ap.add_argument("--windows", default=None,
                    help="cached per-stream window parquet (built by this "
                         "script's build_windows if absent)")
    ap.add_argument("--dump", default=None,
                    help="write per-window (stream, job, y, p) so the run-level "
                         "rules can be re-derived without refitting")
    ap.add_argument("--out",
                    default="classifier/rebuttal_analysis/results/binary_train_vs_rest.json")
    a = ap.parse_args()

    if a.windows and os.path.exists(a.windows):
        w = pd.read_parquet(a.windows)
        print(f"windows loaded from cache {a.windows}")
    else:
        w = build_windows(a.window, a.stride)
        if a.windows:
            w.to_parquet(a.windows, index=False)
            print(f"windows cached to {a.windows}")

    if "job_id" not in w.columns:
        raise SystemExit("window cache lacks job_id -- rebuild it per stream; a "
                         "run_id-grouped cache interleaves the chips of a "
                         "multi-worker cell (see module docstring)")
    if "grp" not in w.columns:
        w["grp"] = workload_group(w.workload_label.values)

    n_all = len(w)
    w = w[~w.workload_label.str.contains(ADV, na=False)].reset_index(drop=True)
    y = (w.threeway_label == "ml_training").astype(int).values
    cols = get_feature_cols(w)
    X = np.nan_to_num(w[cols].to_numpy(np.float32), nan=0.0, posinf=0.0, neginf=0.0)
    print(f"honest windows {len(w):,} of {n_all:,}  training {y.sum():,}  "
          f"non-training {(1 - y).sum():,}")
    print(f"  workload groups {w.grp.nunique()}  streams {w.run_id.nunique():,}  "
          f"jobs {w.job_id.nunique():,}  features {len(cols)}")

    p = np.full(len(w), np.nan)
    gkf = GroupKFold(n_splits=a.folds)
    for i, (tr, te) in enumerate(gkf.split(X, y, groups=w.grp.values), 1):
        sub = w.iloc[tr]
        if sub.threeway_label.nunique() < 3:
            print(f"  fold {i}: skipped, training fold lacks a class")
            continue
        clf = TwoStageClassifier(random_state=0)
        clf.fit(X[tr], sub.threeway_label.values, groups=sub.run_id.values,
                feature_cols=list(cols))
        p[te] = clf.predict_proba_training(X[te])
        print(f"  fold {i}: fitted on {len(tr):,}, scored {len(te):,}", flush=True)

    keep = ~np.isnan(p)
    yk, pk, wk = y[keep], p[keep], w[keep].reset_index(drop=True)
    pred = (pk > a.tau).astype(int)
    acc = accuracy_score(yk, pred)
    pr, rc, f1, _ = precision_recall_fscore_support(yk, pred, average="binary",
                                                    zero_division=0)
    auc = roc_auc_score(yk, pk)
    tn, fp, fn, tp = confusion_matrix(yk, pred).ravel()
    print(f"\n== WINDOW level, tau={a.tau}, held out by workload ==")
    print(f"  accuracy {acc:.4f}   precision {pr:.4f}   recall {rc:.4f}   "
          f"F1 {f1:.4f}   ROC-AUC {auc:.4f}")
    print(f"  TP {tp}  FP {fp}  FN {fn}  TN {tn}   (FPR {fp / (fp + tn):.4f})")

    def run_rule(mask, by, tau):
        """k-of-m consecutive windows above tau, grouped by stream or by job."""
        alerts = n = 0
        sub = wk[mask].reset_index(drop=True)
        pp = pk[np.asarray(mask)]
        for _key, idx in sub.groupby(by).indices.items():
            v = pp[idx]
            n += 1
            if len(v) < a.m:
                alerts += int((v > tau).sum() >= a.k)
            else:
                alerts += int(any((v[i:i + a.m] > tau).sum() >= a.k
                                  for i in range(len(v) - a.m + 1)))
        return (alerts / n if n else float("nan")), n

    pos = (yk == 1)
    runlevel = {}
    print(f"\n== RUN level, {a.k}-of-{a.m}, tau={a.tau} ==")
    for by, name in (("run_id", "stream"), ("job_id", "job")):
        tpr, n_tr = run_rule(pos, by, a.tau)
        fpr, n_ot = run_rule(~pos, by, a.tau)
        runlevel[name] = dict(tpr=float(tpr), fpr=float(fpr),
                              n_training=int(n_tr), n_other=int(n_ot))
        print(f"  by {name:6s}  TPR {tpr:.4f} over {n_tr:5d} training   "
              f"FPR {fpr:.4f} over {n_ot:5d} non-training")

    # Operating-point sweep: the headline claim is quoted at a matched FPR, so
    # the threshold that reaches it has to be read off, not assumed.
    print(f"\n== tau sweep ({a.k}-of-{a.m}, by job) ==")
    print(f"  {'tau':>5s} {'win-TPR':>8s} {'win-FPR':>8s} {'job-TPR':>8s} {'job-FPR':>8s}")
    sweep = []
    for tau in (0.50, 0.60, 0.70, 0.80, 0.90, 0.95):
        wt = float((pk[pos] > tau).mean())
        wf = float((pk[~pos] > tau).mean())
        jt, _ = run_rule(pos, "job_id", tau)
        jf, _ = run_rule(~pos, "job_id", tau)
        sweep.append(dict(tau=tau, window_tpr=wt, window_fpr=wf,
                          job_tpr=float(jt), job_fpr=float(jf)))
        print(f"  {tau:5.2f} {wt:8.4f} {wf:8.4f} {jt:8.4f} {jf:8.4f}")

    print("\n== where the false alarms come from ==")
    byclass = {}
    for lab in ("ml_inference", "other"):
        m = (wk.threeway_label == lab).values
        if m.sum():
            frac = float((pk[m] > a.tau).mean())
            byclass[lab] = dict(windows=int(m.sum()), frac_over_tau=frac)
            print(f"  {lab:14s} windows {m.sum():6d}  frac>tau {frac:.4f}")

    print("\n== per platform (window level, tau=%.2f) ==" % a.tau)
    plat = wk.apply(lambda r: platform_of(r.workload_label,
                                          r.get("gpu_name", "")), axis=1)
    byplat = {}
    for name, idx in plat.groupby(plat).indices.items():
        idx = np.asarray(idx)
        yy, pp2 = yk[idx], pk[idx]
        if len(idx) < 50:
            continue
        row = dict(windows=int(len(idx)), train_windows=int(yy.sum()),
                   tpr=float((pp2[yy == 1] > a.tau).mean()) if yy.sum() else float("nan"),
                   fpr=float((pp2[yy == 0] > a.tau).mean()) if (1 - yy).sum() else float("nan"),
                   auc=float(roc_auc_score(yy, pp2)) if 0 < yy.sum() < len(yy) else float("nan"))
        byplat[name] = row
        print(f"  {name:16s} n {row['windows']:6d}  TPR {row['tpr']:.4f}  "
              f"FPR {row['fpr']:.4f}  AUC {row['auc']:.4f}")

    if a.dump:
        pd.DataFrame({"run_id": wk.run_id.values, "job_id": wk.job_id.values,
                      "workload_label": wk.workload_label.values,
                      "gpu_name": wk.get("gpu_name", pd.Series([""] * len(wk))).values,
                      "threeway_label": wk.threeway_label.values,
                      "y": yk, "p": pk}).to_parquet(a.dump, index=False)
        print(f"dumped per-window predictions -> {a.dump}")

    if a.out:
        os.makedirs(os.path.dirname(a.out), exist_ok=True)
        json.dump({"generated_utc": datetime.datetime.utcnow().isoformat() + "Z",
                   "protocol": "workload-grouped GroupKFold, honest cells only, "
                               "windows built PER STREAM",
                   "tau": a.tau, "folds": a.folds, "k": a.k, "m": a.m,
                   "window_sec": a.window, "stride_sec": a.stride,
                   "n_windows": int(len(wk)), "n_groups": int(w.grp.nunique()),
                   "n_streams": int(wk.run_id.nunique()),
                   "n_jobs": int(wk.job_id.nunique()),
                   "accuracy": float(acc), "precision": float(pr),
                   "recall": float(rc), "f1": float(f1), "roc_auc": float(auc),
                   "tp": int(tp), "fp": int(fp), "fn": int(fn), "tn": int(tn),
                   "window_fpr": float(fp / (fp + tn)),
                   "run_level": runlevel, "tau_sweep": sweep,
                   "false_alarm_source": byclass, "per_platform": byplat},
                  open(a.out, "w"), indent=2)
        print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
