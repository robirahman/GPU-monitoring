#!/usr/bin/env python3
"""Honest multinode training-vs-rest LOFO, using the shared exhaustive protocol.

Writes a compatibility CSV plus complete predictions, split manifests, and
metrics for all-feature and shape-only arms. Threshold sweeps are descriptive;
they must not be selected on held-out families and then reported as calibrated.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from classifier.rebuttal_analysis.multinode_family_holdout import load_multinode
from classifier.rebuttal_analysis.paired_family_lofo import evaluate, validate_window_cache


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--windows", default="/var/tmp/spar_analysis/win_full.parquet")
    ap.add_argument("--inventory", default="/var/tmp/spar_analysis/inv_mn.parquet")
    ap.add_argument("--trees", type=int, default=300)
    ap.add_argument("--max-reps", type=int, default=0)
    ap.add_argument("--out", default="/var/tmp/spar_analysis/mn_binary_lofo.csv")
    ap.add_argument("--taus", default="0.8,0.5,0.3,0.2,0.1,0.05")
    ap.add_argument("--family-level", choices=["mechanism", "backbone"], default="mechanism")
    a = ap.parse_args()
    cache = validate_window_cache(a.windows)
    w = load_multinode(a.windows, a.inventory)
    w = w[w.kind == "honest"].reset_index(drop=True)
    out = Path(a.out); out.parent.mkdir(parents=True, exist_ok=True)
    arms = {}
    for tag, shape in (("all", False), ("shape", True)):
        result, pred, splits = evaluate(w, trees=a.trees, shape=shape, binary=True,
                                        max_reps=a.max_reps, family_level=a.family_level)
        result["cache_fingerprint"] = cache["fingerprint"]
        out.with_suffix(f".{tag}.json").write_text(json.dumps(result, indent=2))
        out.with_suffix(f".{tag}.splits.json").write_text(json.dumps(splits, indent=2))
        pred.to_parquet(out.with_suffix(f".{tag}.predictions.parquet"), index=False)
        arms[tag] = (result, pred)
    rows = []
    for fam in arms["all"][0]["families"]:
        p = arms["all"][1].query("family == @fam")
        shape = arms["shape"][1].query("family == @fam")
        row = dict(family=fam, side="TRAIN" if p.truth.eq("ml_training").all() else "not-train",
                   n_win=len(p), n_jobs=p.job_id.nunique(), win_all=float((p.truth == p.prediction).mean()),
                   win_shape=float((shape.truth == shape.prediction).mean()), p_train_median=float(p.p_training.median()))
        for tau in map(float, a.taus.split(",")):
            if not 0 <= tau <= 1:
                raise ValueError("Threshold outside [0,1]")
            row[f"flag@{tau}"] = float((p.p_training >= tau).mean())
        rows.append(row)
    pd.DataFrame(rows).to_csv(out, index=False)


if __name__ == "__main__":
    main()
