#!/usr/bin/env python3
"""Exhaustive family holdouts, with an optional paired training-data ablation.

No model/threshold selection uses outer test data. Both ablation arms use the
same old-job test rows, feature vocabulary, estimator settings, and random seed.
New jobs receive separate full-model LOFO predictions, never enter the paired
effect estimate. Invalid folds are errors rather than silently skipped families.
Targets are per-window work labels (classifier/window_labels.py): what the job
did inside each window, never what the job was named.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.pipeline import make_pipeline

from classifier.data_sources import sha256
from classifier.operational_metrics import operational_metrics
from classifier.threeway_improved import get_feature_cols
from classifier.multinode_features import stage2_mask_multinode
from classifier.workload_labels import LABEL_SCHEMA_VERSION, FAMILY_SCHEMA_VERSION, workload_family
from classifier.rebuttal_analysis.multinode_family_holdout import load_multinode


def validate_window_cache(path):
    from classifier.cache_provenance import provenance, cache_matches
    sidecar = Path(str(path) + ".provenance.json")
    if not sidecar.is_file():
        raise ValueError("Unversioned window cache; rebuild after the label/identity/feature fixes")
    saved = json.loads(sidecar.read_text())
    expected = provenance(**saved["config"])
    if not cache_matches(path, expected):
        raise ValueError("Stale window cache: data, code, labels, configuration, or artifact changed")
    return saved


def row_ids(frame):
    cols = [c for c in ("job_id", "run_id", "window_start", "window_end") if c in frame]
    if not {"job_id", "window_start", "window_end"} <= set(cols):
        raise ValueError("Window identity needs job_id, window_start, window_end")
    ids = frame[cols].astype(str).agg("|".join, axis=1).map(
        lambda x: hashlib.sha256(x.encode()).hexdigest())
    if ids.duplicated().any():
        raise ValueError("Duplicate test-window identities; deduplicate before evaluation")
    return ids


def fold_plan(frame, new_jobs=(), family_level="mechanism"):
    """Return all folds; ablate complete physical jobs only, never partial windows."""
    frame = frame.copy()
    frame["family"] = frame.grp.map(lambda x: workload_family(x, family_level))
    if (frame.groupby("job_id").family.nunique() > 1).any():
        raise ValueError("A job spans multiple families")
    new_jobs = set(new_jobs)
    known = set(frame.job_id.astype(str))
    if new_jobs - known:
        raise ValueError(f"Ablation jobs absent from eligible cache: {sorted(new_jobs - known)[:5]}")
    added = frame.job_id.astype(str).isin(new_jobs).to_numpy()
    folds = []
    for fam in sorted(frame.family.unique()):
        test = frame.family.eq(fam).to_numpy()
        train = ~test
        folds.append(dict(family=fam, train=np.flatnonzero(train),
                          baseline_train=np.flatnonzero(train & ~added),
                          test=np.flatnonzero(test), paired_test=np.flatnonzero(test & ~added)))
    return frame, folds


def train_share_bin(frac):
    """Bin a share of time spent training; NaN means the window has no STEP evidence."""
    frac = np.asarray(frac, float)
    return np.select([np.isnan(frac), frac <= 0, frac <= 0.05, frac <= 0.2, frac <= 0.5],
                     ["unknown", "0%", "0-5%", "5-20%", "20-50%"], "50-100%")


def metrics(y, pred, probability, jobs, threshold, train_frac=None):
    truth = np.asarray(y) == "ml_training"
    flag = probability >= threshold
    pos, neg = int(truth.sum()), int((~truth).sum())
    j = pd.DataFrame(dict(job=jobs, truth=truth, flag=flag)).groupby("job").max()
    # Existential alert rule is explicit: any positive window in a job. Its
    # false-positive rate is reported separately from window-level rates.
    out = dict(windows=len(y), jobs=len(j), accuracy=float(accuracy_score(y, pred)),
               macro_f1=float(f1_score(y, pred, average="macro", zero_division=0)),
               training_recall=float(flag[truth].mean()) if pos else None,
               false_positive_rate=float(flag[~truth].mean()) if neg else None,
               job_any_recall=float(j.loc[j.truth, "flag"].mean()) if j.truth.any() else None,
               job_any_fpr=float(j.loc[~j.truth, "flag"].mean()) if (~j.truth).any() else None,
               confusion=confusion_matrix(y, pred, labels=["ml_training", "ml_inference", "other"]).tolist())
    if train_frac is not None:
        # Mean share of a training window's time spent training; one minus it
        # is the evasion tax those windows paid.
        tf = np.asarray(train_frac, float)[truth]
        tf = tf[np.isfinite(tf)]
        out["train_frac_mean"] = float(tf.mean()) if len(tf) else None
    return out


def evaluate(frame, new_jobs=(), *, trees=300, seed=0, shape=False,
             family_level="mechanism", threshold=0.5, binary=False, max_reps=0,
             stride_sec=15.0):
    frame, folds = fold_plan(frame.reset_index(drop=True), new_jobs, family_level)
    ids = row_ids(frame)
    tf = (frame.label_train_frac.to_numpy(float) if "label_train_frac" in frame
          else np.full(len(frame), np.nan))
    cols = get_feature_cols(frame)
    if shape:
        cols = list(np.asarray(cols)[stage2_mask_multinode(cols)])
    if not cols:
        raise ValueError("No telemetry features")
    X = frame[cols].replace([np.inf, -np.inf], np.nan).to_numpy(np.float32)
    y = frame.threeway_label.astype(str).to_numpy()
    if binary:
        y = np.where(y == "ml_training", "ml_training", "other")
    if len(set(y)) < 2:
        raise ValueError("Need at least two target classes")
    rep = np.ones(len(frame))
    if max_reps:
        first = frame.groupby(["grp", "job_id"]).window_start.min().rename("first").reset_index()
        first["rep"] = first.groupby("grp")["first"].rank(method="dense")
        rep = frame.merge(first[["grp", "job_id", "rep"]], on=["grp", "job_id"], how="left")["rep"].to_numpy()
    predictions, splits, scores = [], [], []
    paired = bool(new_jobs)
    new_jobs = set(new_jobs)
    for fold in folds:
        fam = fold["family"]
        te = fold["test"]
        arm_predictions = {}
        for arm, key in (("full", "train"), ("baseline", "baseline_train")):
            if arm == "baseline" and (not paired or not len(fold["paired_test"])):
                continue
            tr = fold[key]
            if max_reps:
                tr = tr[rep[tr] <= max_reps]
            if set(y[tr]) != set(y):
                raise ValueError(f"{fam}/{arm}: training lacks a target class; cannot claim complete LOFO")
            if set(frame.job_id.iloc[tr]) & set(frame.job_id.iloc[te]):
                raise AssertionError("Job leakage")
            clf = make_pipeline(SimpleImputer(strategy="median", keep_empty_features=True),
                                RandomForestClassifier(n_estimators=trees, max_features=0.3,
                                                       min_samples_leaf=2, n_jobs=-1, random_state=seed))
            clf.fit(X[tr], y[tr])
            eval_idx = te if arm == "full" else fold["paired_test"]
            pred = clf.predict(X[eval_idx])
            probability = clf.predict_proba(X[eval_idx])[:, list(clf.classes_).index("ml_training")]
            arm_predictions[arm] = (eval_idx, pred, probability)
            splits.append(dict(family=fam, arm=arm, train_jobs=sorted(set(frame.job_id.iloc[tr])),
                               train_ids=sorted(ids.iloc[tr]), test_ids=sorted(ids.iloc[eval_idx])))
            for i, p, pr in zip(eval_idx, pred, probability):
                predictions.append(dict(row_id=ids.iloc[i], family=fam, arm=arm,
                                        job_id=frame.job_id.iloc[i], run_id=frame.run_id.iloc[i],
                                        window_start=float(frame.window_start.iloc[i]),
                                        window_end=float(frame.window_end.iloc[i]),
                                        truth=y[i], prediction=p, p_training=float(pr), train_frac=float(tf[i]),
                                        paired_test=frame.job_id.iloc[i] not in new_jobs,
                                        hardware=frame.gpu_name.iloc[i] if "gpu_name" in frame else "",
                                        kind=frame.kind.iloc[i] if "kind" in frame else "",
                                        job_label=frame.job_label.iloc[i] if "job_label" in frame else "",
                                        label_source=frame.label_source.iloc[i] if "label_source" in frame else ""))
            scores.append(dict(family=fam, arm=arm, test_population="all" if arm == "full" else "old_jobs",
                               **metrics(y[eval_idx], pred, probability, frame.job_id.iloc[eval_idx], threshold,
                                         train_frac=tf[eval_idx])))
        if "baseline" in arm_predictions:
            ix, p, pr = arm_predictions["full"]
            mask = np.isin(ix, fold["paired_test"])
            assert set(ids.iloc[ix[mask]]) == set(splits[-1]["test_ids"])
            if set(splits[-2]["train_ids"]) == set(splits[-1]["train_ids"]):
                raise ValueError(f"{fam}: ablation removes no training windows")
            scores.append(dict(family=fam, arm="full", test_population="old_jobs",
                               **metrics(y[ix[mask]], p[mask], pr[mask], frame.job_id.iloc[ix[mask]], threshold,
                                         train_frac=tf[ix[mask]])))
        print(f"{fam}: tested {len(te)} windows", flush=True)
    predictions = pd.DataFrame(predictions)
    full = predictions[predictions.arm == "full"]
    assert set(full.row_id) == set(ids) and len(full) == len(ids)
    result = dict(protocol="exhaustive_leave_one_family_out", family_level=family_level,
                  label_schema=LABEL_SCHEMA_VERSION, family_schema=FAMILY_SCHEMA_VERSION,
                  target="training_vs_rest" if binary else "threeway_window_work",
                  label_rule="window labeled by logged work: training if any gradient step, "
                             "else inference if any forward pass, else other",
                  threshold=threshold, threshold_selection="fixed before outer tests",
                  job_rule="any positive window; not majority rank vote",
                  model=dict(type="RandomForestClassifier", trees=trees, seed=seed,
                             max_features=0.3, min_samples_leaf=2, imputation="training-fold median"),
                  features=cols, families=sorted(frame.family.unique()),
                  n_windows=len(frame), n_jobs=frame.job_id.nunique(),
                  ablated_jobs=sorted(new_jobs), per_fold=scores)
    result['operational'] = operational_metrics(full, threshold, stride_sec)
    result['operational_by_family'] = {
        str(f): operational_metrics(g, threshold, stride_sec) for f, g in full.groupby('family')}
    for population, part in (("all", full), ("paired_old_jobs", full[full.paired_test]),
                             ("new_jobs", full[~full.paired_test])):
        if len(part):
            result[population] = metrics(part.truth, part.prediction, part.p_training.to_numpy(), part.job_id,
                                         threshold, train_frac=part.train_frac.to_numpy())
    for axis in ("hardware", "kind", "label_source"):
        result["by_" + axis] = {
            str(key): metrics(part.truth, part.prediction, part.p_training.to_numpy(), part.job_id, threshold,
                              train_frac=part.train_frac.to_numpy())
            for key, part in full.groupby(axis, dropna=False)
        }
    known = full[full.train_frac.notna()]
    if len(known):
        # The evasion tax: detection against the share of time spent training,
        # per window and per job (a job's share is the mean over its windows).
        result["by_train_fraction"] = {
            str(b): metrics(part.truth, part.prediction, part.p_training.to_numpy(), part.job_id, threshold)
            for b, part in known.groupby(train_share_bin(known.train_frac))}
        per_job = known.groupby("job_id").agg(share=("train_frac", "mean"),
                                              trained=("truth", lambda t: t.eq("ml_training").any()),
                                              flagged=("p_training", lambda p: bool((p >= threshold).any())))
        result["by_job_train_share"] = {
            str(b): dict(jobs=len(g), training_jobs=int(g.trained.sum()),
                         detected=float(g.loc[g.trained, "flagged"].mean()) if g.trained.any() else None,
                         false_alarms=float(g.loc[~g.trained, "flagged"].mean()) if (~g.trained).any() else None)
            for b, g in per_job.groupby(train_share_bin(per_job.share))}
    if paired:
        base = predictions[predictions.arm == "baseline"]
        result['operational_baseline_old_jobs'] = operational_metrics(base, threshold, stride_sec)
        result['operational_full_old_jobs'] = operational_metrics(full[full.paired_test], threshold, stride_sec)
        result["baseline_old_jobs"] = metrics(base.truth, base.prediction, base.p_training.to_numpy(), base.job_id, threshold)
        full_old = [s["accuracy"] for s in scores if s["arm"] == "full" and s["test_population"] == "old_jobs"]
        base_old = [s["accuracy"] for s in scores if s["arm"] == "baseline"]
        result["paired_family_accuracy_delta"] = float(np.mean(np.array(full_old) - np.array(base_old)))
        deltas = np.array(full_old) - np.array(base_old)
        rng = np.random.default_rng(seed)
        boot = rng.choice(deltas, size=(2000, len(deltas)), replace=True).mean(axis=1)
        result["paired_family_accuracy_delta_bootstrap95"] = np.quantile(boot, [0.025, 0.975]).tolist()
        result["paired_family_count"] = len(deltas)
    return result, predictions, splits


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", required=True)
    ap.add_argument("--inventory")
    ap.add_argument("--out", required=True)
    ap.add_argument("--exclude-campaigns", default="", help="Training-only ablation, e.g. 20261001,20261003")
    ap.add_argument("--new-jobs", help="JSON list of job IDs for a paired training-only ablation")
    ap.add_argument("--family-level", choices=["mechanism", "backbone"], default="mechanism")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--shape", action="store_true")
    ap.add_argument("--binary", action="store_true")
    ap.add_argument("--honest-only", action="store_true")
    ap.add_argument("--max-reps", type=int, default=0)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--steps-only", action="store_true",
                    help="Fit and score only windows labeled from STEP evidence")
    a = ap.parse_args()
    if not 0 <= a.threshold <= 1:
        ap.error("threshold must be between 0 and 1")
    cache = validate_window_cache(a.windows)
    w = load_multinode(a.windows, a.inventory)
    unresolved = w.attrs.get("unresolved")
    if a.steps_only:
        w = w[w.label_source.eq("steps")].reset_index(drop=True)
    if a.honest_only:
        w = w[w.kind == "honest"].reset_index(drop=True)
    new_jobs = set(json.loads(Path(a.new_jobs).read_text())) if a.new_jobs else set()
    if a.exclude_campaigns:
        campaigns = set(a.exclude_campaigns.split(","))
        if "meta_campaign" not in w:
            raise ValueError("Campaign ablation needs a rebuilt cache with source provenance")
        hit = w.meta_campaign.fillna("").map(lambda c: bool(set(c.split("|")) & campaigns))
        new_jobs.update(w.loc[hit, "job_id"])
    if (a.new_jobs or a.exclude_campaigns) and not new_jobs:
        raise ValueError("Ablation removes zero jobs")
    result, predictions, splits = evaluate(w, new_jobs, trees=a.trees, seed=a.seed,
                                           shape=a.shape, family_level=a.family_level,
                                           threshold=a.threshold, binary=a.binary, max_reps=a.max_reps,
                                           stride_sec=float(cache['config']['stride_sec']))
    result["unresolved_excluded"] = unresolved
    result["steps_only"] = a.steps_only
    result["windows_sha256"] = sha256(a.windows)
    result["cache_fingerprint"] = cache["fingerprint"]
    result["code"] = cache["code"]
    result["libraries"] = cache["libraries"]
    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_parquet(out.with_suffix(".predictions.parquet"), index=False)
    out.with_suffix(".splits.json").write_text(json.dumps(splits, indent=2))
    out.write_text(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
