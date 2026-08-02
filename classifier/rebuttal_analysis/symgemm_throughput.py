#!/usr/bin/env python3
"""What the disguise costs the attacker (Area Chair 1uob's question).

The meta-review asks us to quantify what evasion costs the attacker (e.g.
longer training time). This builds that table from the Round 6 STEP logs
plus the matching 1 Hz telemetry, in the same shape as the operating-point
table already sent to M7wU.

Columns:
  tok/s          useful tokens per second (0 for the L0/L1/L3 negatives)
  gemm-tok/s     tokens x GEMM passes -- the throughput an observer who
                 believed the disguise would attribute to the job
  W              mean power draw over the run
  train Mtok/GPU-h   useful training throughput
  J/train tok    energy per token of actual learning
  vs L5          useful throughput relative to ordinary AdamW training

The interesting quantity is the last one. If the disguise is free, the paper's
threat model has a hole. If it costs 2x, the governance question becomes
whether that price is worth paying, which is a very different claim.

Usage:
    python classifier/rebuttal_analysis/symgemm_throughput.py
    python classifier/rebuttal_analysis/symgemm_throughput.py --logs 'logs/symgemm_*.log'
"""

import argparse
import glob
import json
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parent.parent.parent
OUT = Path(__file__).resolve().parent / "symgemm_throughput.json"

STEP_RE = re.compile(
    r"STEP ts=([\d.]+) kind=(\w+) batch_idx=(\d+) tokens=(\d+) passes=(\d+)")
RUNID_RE = re.compile(r"Run ID:\s*(\S+)")
# Modes whose weights actually change; only these produce useful training.
LEARNS = ("L4", "L5")


def cell_of(stage):
    """symgemm_L4_sgdfused_m8192_rep2 -> (L4_sgdfused_m8192, L4)."""
    name = stage.replace("symgemm_", "")
    name = re.sub(r"_rep\d+$", "", name)
    mode = name.split("_")[0]
    return name, mode


def parse_log(path):
    ts, kinds, tokens, passes = [], [], [], []
    run_id = None
    with open(path, errors="replace") as fh:
        for line in fh:
            m = STEP_RE.search(line)
            if m:
                ts.append(float(m.group(1)))
                kinds.append(m.group(2))
                tokens.append(int(m.group(4)))
                passes.append(int(m.group(5)))
            elif run_id is None:
                r = RUNID_RE.search(line)
                if r:
                    run_id = r.group(1)
    if not ts:
        return None
    return {"run_id": run_id, "ts": np.array(ts), "tokens": np.array(tokens),
            "passes": np.array(passes), "kind": kinds[0]}


def mean_power(stage):
    """Mean power over the run, from the telemetry parquet for this stage."""
    hits = glob.glob(str(REPO / "data" / f"{stage}_*.parquet"))
    if not hits:
        return None, None
    df = pd.read_parquet(sorted(hits)[0],
                         columns=["power_draw_w", "gpu_utilization_pct"])
    p = df["power_draw_w"].replace(-1, np.nan).dropna()
    u = df["gpu_utilization_pct"].replace(-1, np.nan).dropna()
    return (float(p.mean()) if len(p) else None,
            float(u.mean()) if len(u) else None)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", default="logs/symgemm_*.log")
    args = ap.parse_args()

    paths = sorted(glob.glob(str(REPO / args.logs)))
    if not paths:
        raise SystemExit(
            f"No logs matched {args.logs}. Run "
            f"scripts/rebuttal_collection/round6_symgemm_queue.sh first.")

    by_cell = defaultdict(list)
    for p in paths:
        stage = Path(p).stem
        parsed = parse_log(p)
        if parsed is None:
            print(f"  (no STEP lines in {stage} -- skipped)")
            continue
        cell, mode = cell_of(stage)
        elapsed = parsed["ts"][-1] - parsed["ts"][0]
        if elapsed <= 0:
            continue
        tok = parsed["tokens"].sum()
        gemm_tok = int((parsed["tokens"] * parsed["passes"]).sum())
        watts, util = mean_power(stage)
        by_cell[cell].append({
            "stage": stage, "mode": mode, "elapsed_s": elapsed,
            "tok_s": tok / elapsed, "gemm_tok_s": gemm_tok / elapsed,
            "train_tok_s": (tok / elapsed) if mode in LEARNS else 0.0,
            "watts": watts, "util": util,
        })

    rows = []
    for cell, reps in sorted(by_cell.items()):
        mode = reps[0]["mode"]
        agg = {k: float(np.mean([r[k] for r in reps if r[k] is not None]))
               if any(r[k] is not None for r in reps) else None
               for k in ("tok_s", "gemm_tok_s", "train_tok_s", "watts", "util")}
        train_mtok_h = agg["train_tok_s"] * 3600 / 1e6
        j_per_train_tok = (agg["watts"] / agg["train_tok_s"]
                           if agg["train_tok_s"] and agg["watts"] else None)
        rows.append({"cell": cell, "mode": mode, "n_reps": len(reps),
                     "train_mtok_per_gpu_h": train_mtok_h,
                     "j_per_train_tok": j_per_train_tok, **agg})

    base = next((r for r in rows if r["mode"] == "L5"), None)
    print(f"{'cell':26s} {'n':>2s} {'tok/s':>8s} {'gemm-tok/s':>11s} {'W':>6s} "
          f"{'util%':>6s} {'train Mtok/GPU-h':>17s} {'J/train tok':>12s} {'vs L5':>7s}")
    print("-" * 108)
    for r in rows:
        rel = ("--" if not base or not base["train_tok_s"] or not r["train_tok_s"]
               else f"{r['train_tok_s'] / base['train_tok_s']:.2f}x")
        r["vs_L5"] = rel
        w = f"{r['watts']:.0f}" if r["watts"] else "n/a"
        u = f"{r['util']:.0f}" if r["util"] else "n/a"
        j = f"{r['j_per_train_tok']:.4f}" if r["j_per_train_tok"] else "--"
        print(f"{r['cell']:26s} {r['n_reps']:>2d} {r['tok_s']:>8.0f} "
              f"{r['gemm_tok_s']:>11.0f} {w:>6s} {u:>6s} "
              f"{r['train_mtok_per_gpu_h']:>17.2f} {j:>12s} {rel:>7s}")

    print("\nReading:")
    print("  gemm-tok/s is what an observer who believed the disguise would")
    print("  infer as the served inference rate. For L4 it is 3x the real")
    print("  token rate -- the attack presents as an inference deployment")
    print("  running at three times its actual throughput.")
    if base:
        print("  'vs L5' is the attacker's price: useful training throughput")
        print("  relative to ordinary AdamW at the same model and shapes.")

    OUT.write_text(json.dumps(rows, indent=2, default=float))
    print(f"\nSaved: {OUT}")


if __name__ == "__main__":
    main()
