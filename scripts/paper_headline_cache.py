#!/usr/bin/env python3
"""The headline three ways, from the cached window frame.

paper_numbers.py rebuilds windows from raw telemetry, which is a ~26 minute
serial step and produces exactly what cache/win_30.parquet already holds
(causal, job labels, 30 s / 15 s). This runs the same evaluate() over the cache
so the XGBoost headline lands in seconds; the slow run is left going as a
cross-check.
"""
import json, os, sys, time
from pathlib import Path
import pandas as pd
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from scripts.regen_headline import evaluate  # noqa: E402

t0 = time.time()
win = pd.read_parquet(ROOT / "cache/win_30.parquet")
model = os.environ.get("PAPER_MODEL", "rf")
out = {"model": model, "window_sec": 30.0, "stride_sec": 15.0,
       "source": "cache/win_30.parquet", "results": []}
print(f"{len(win):,} windows | model={model}", flush=True)
for gb in ("job_id", "run_id", "workload"):
    r = evaluate(win, f"causal, job labels | holdout={gb}", group_by=gb)
    out["results"].append(r)
    print(f"  {gb:9s} acc {r['accuracy']*100:6.2f}% +- {r['std']*100:.2f}  "
          f"prec {r['precision']*100:.2f} rec {r['recall']*100:.2f} "
          f"spec {r['specificity']*100:.2f} FPR {r['false_positive_rate']*100:.2f} "
          f"macroF1 {r['macro_f1']*100:.2f}  groups {r['groups']}  "
          f"[{time.time()-t0:.0f}s]", flush=True)
p = ROOT / f"docs/camera_ready_results_20261022/paper_headline_{model}.json"
p.write_text(json.dumps(out, indent=1) + "\n")
print("wrote", p)
