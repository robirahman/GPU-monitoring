#!/usr/bin/env python3
"""Persist a fitted detector, and define once what "honest" means.

Why this exists
---------------
Two gaps this closes.

1. THERE IS NO MODEL PERSISTENCE ANYWHERE IN THIS REPO. Every script refits from
   raw telemetry in-process. That is why classifier_oracle.py spends ~182 s
   retraining XGBoost on every single run before the adversary takes its first
   step -- and why the adversary ends up querying a DIFFERENT model from the one
   the evaluation scores against (the oracle trains at 60 s/30 s on the whole
   unfiltered corpus; every evaluator uses 30 s/15 s with families held out).
   For an evasion experiment that inconsistency is not cosmetic: it makes "the
   attacker knows the detector" untrue in exactly the way that matters.

2. "Honest workloads only" is currently re-implemented inline, slightly
   differently, in at least three places:
       check_ptrain.py:25      ~wl.str.startswith("whitebox")
       lora_in_training.py:42  ~(startswith("whitebox") | contains("_mw8"))
       symgemm_eval.py:242     ~wl.str.startswith("symgemm_")
   rebuttal_analysis/README.md notes 3 and 6 record that corpus-hygiene slips of
   exactly this kind have already changed published conclusions once. One
   definition, used everywhere, is worth more than three careful ones.

The detector persisted here is TwoStageClassifier (classifier/twostage.py),
whose Stage 2 deliberately drops level features -- absolute memory, power, clock
magnitudes -- so that an adversary spoofing levels cannot drag the train-vs-infer
decision. Its own docstring calls that "the WB-O defense". That makes it the
right thing to attack: the interleave disguise works precisely by moving levels,
so this is the confrontation worth measuring rather than a straw detector.

Usage
-----
    python -m classifier.model_store --fit          # fit + save honest detector
    python -m classifier.model_store --describe     # what is in the artifact
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = PROJECT_ROOT / "classifier" / "models" / "honest_detector.joblib"

# Workload-label families that are NOT honest: disguised/evasive training, and
# the adversarial families generally. Matched as prefixes or substrings against
# workload_label. Kept as one list so adding a new attack family is a one-line
# change and cannot be forgotten in some scripts but not others.
#
# NOTE on threeway_label: it maps every one of these to ml_training on purpose
# (threeway_improved.py:234-272) -- their GROUND TRUTH is training. This mask is
# about corpus hygiene (what the detector is allowed to learn from), not labels.
DISGUISED_FAMILIES = (
    "whitebox",      # WB-I/WB-D/WB-L/WB-O
    "symgemm",       # the Round-6 kernel-level ladder incl. L4 and disguised PPO
    "advtrain",      # NEW: multi-node evasive cells (adversarial_multinode.py)
    "adversarial",   # single-GPU strategies A/B/D/E/F/G/H/I/J/K/L/M/N + DDP ones
)
# Substring markers for one-off variants that do not share a family prefix.
DISGUISED_MARKERS = ("_mw8",)


def honest_mask(win: pd.DataFrame) -> np.ndarray:
    """True for windows a detector is allowed to train on.

    Excludes every disguised/adversarial family, so a model fitted under this
    mask has never seen an evasion attempt. That is the assumption the evasion
    result rests on: a detector that had already seen the attack would make the
    experiment circular.
    """
    wl = win["workload_label"].astype(str)
    bad = np.zeros(len(win), dtype=bool)
    for fam in DISGUISED_FAMILIES:
        bad |= wl.str.startswith(fam).values
        # the multinode cells are s16_advtrain_..., so the family token appears
        # after the scope prefix rather than at the start
        bad |= wl.str.contains(f"_{fam}", regex=False).values
    for mark in DISGUISED_MARKERS:
        bad |= wl.str.contains(mark, regex=False).values
    return ~bad


def save(clf, path=DEFAULT_MODEL_PATH, meta: dict | None = None) -> Path:
    """Persist a fitted estimator plus the metadata needed to use it correctly."""
    import joblib

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"clf": clf, "meta": meta or {}}, path)
    if meta:
        path.with_suffix(".json").write_text(json.dumps(meta, indent=2, default=str))
    return path


def load(path=DEFAULT_MODEL_PATH):
    """Return (clf, meta). Raises FileNotFoundError if not fitted yet."""
    import joblib

    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"no persisted detector at {path}; run "
            f"`python -m classifier.model_store --fit` first")
    blob = joblib.load(path)
    return blob["clf"], blob.get("meta", {})


def fit_honest_detector(window_sec: float = 30.0, stride_sec: float = 15.0,
                        path=DEFAULT_MODEL_PATH):
    """Fit TwoStageClassifier on honest windows only, at the evaluators' windowing.

    window_sec/stride_sec default to 30/15 -- what every script in
    rebuttal_analysis/ uses -- rather than the 60/30 the oracle used to fit at.
    Matching them is the point: the attacker should query the same model the
    defence is evaluated with.
    """
    from classifier.threeway_improved import (
        load_all_data, sliding_windows, get_feature_cols, workload_group,
    )
    from classifier.twostage import TwoStageClassifier

    df = load_all_data()
    win = sliding_windows(df, window_sec, stride_sec, run_level_mode="causal")
    feats = get_feature_cols(win)

    keep = honest_mask(win)
    honest = win[keep]
    n_excluded_runs = win.loc[~keep, "run_id"].nunique()
    print(f"honest windows: {len(honest):,} of {len(win):,} "
          f"({n_excluded_runs} disguised run(s) excluded)")
    dist = honest["threeway_label"].value_counts().to_dict()
    print(f"class distribution: {dist}")
    if len(dist) < 2:
        raise SystemExit("refusing to fit: honest corpus has fewer than 2 classes")

    X = honest[feats].fillna(0).values.astype(np.float32)
    y = honest["threeway_label"].values
    # Group by WORKLOAD, not by run. These groups drive the class/group sample
    # weighting, so run-grouping let a workload with many runs (mining_ethash_proxy
    # carries 1372 windows) dominate the non-ML class while a workload with one run
    # contributed almost nothing -- the opposite of what generalisation needs.
    groups = workload_group(honest["workload_label"]).astype(str)

    clf = TwoStageClassifier()
    clf.fit(X, y, groups=groups, feature_cols=feats)

    meta = {
        "window_sec": window_sec,
        "stride_sec": stride_sec,
        "run_level_mode": "causal",
        "n_windows": int(len(honest)),
        "n_features": len(feats),
        "feature_cols": feats,
        "class_distribution": {str(k): int(v) for k, v in dist.items()},
        "excluded_families": list(DISGUISED_FAMILIES),
        "excluded_markers": list(DISGUISED_MARKERS),
        "n_excluded_runs": int(n_excluded_runs),
        "group_key": "workload",
        "n_workload_groups": int(pd.Series(groups).nunique()),
    }
    out = save(clf, path, meta)
    print(f"saved -> {out}")
    return clf, meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fit", action="store_true", help="fit and persist")
    ap.add_argument("--describe", action="store_true", help="show artifact metadata")
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--stride", type=float, default=15.0)
    ap.add_argument("--path", default=str(DEFAULT_MODEL_PATH))
    args = ap.parse_args()

    if args.fit:
        fit_honest_detector(args.window, args.stride, args.path)
    elif args.describe:
        _, meta = load(args.path)
        print(json.dumps({k: v for k, v in meta.items() if k != "feature_cols"},
                         indent=2))
        print(f"feature_cols: {len(meta.get('feature_cols', []))}")
    else:
        ap.error("pass --fit or --describe")


if __name__ == "__main__":
    main()
