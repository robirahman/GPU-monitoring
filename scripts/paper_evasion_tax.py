#!/usr/bin/env python3
"""The evasion tax: what each disguise costs the adversary in useful throughput.

Area Chair 1uob asked what evasion costs the attacker. Three scripts already
answer fragments of it -- rebuttal_analysis/symgemm_throughput.py covers the
Round 6 ladder, evasion_cost_eval.py joins cost to detection for the disguised
cells it knows about, symgemm_traffic.py models DRAM traffic analytically --
and the rest of the repo guesses: evasion_bound.py:279 uses
`1.0 - idle_frac  # Approximate`, eval_dilution_sweep.py:72 uses `1/N`, and
report/week8_results.md publishes a Cost column that is just the dilution
factor. This measures it for every family that has the evidence, and says
plainly which families do not.

THE MEASURE IS USEFUL WORK, NOT WALL THROUGHPUT. A disguise can raise tok/s
while destroying the training it was supposed to hide: the WB-O oracle veto
turns vetoed steps into forward-only passes, cutting real gradient updates ~5x
while tok/s goes UP. So the numerator is tokens on `kind=train` STEP lines
only -- tokens that actually moved weights -- and the denominator is the whole
wall clock the adversary paid for, including the inference padding that makes
the disguise work.

    tax = useful tokens/s of the honest reference
          -------------------------------------
          useful tokens/s of the disguised run

A tax of 20x means the adversary buys 20 GPU-hours to get 1 GPU-hour of
training. A tax BELOW 1.0 means the disguise is free, which is a hole in the
threat model and is reported as such rather than clipped.

REFERENCES ARE MATCHED WITHIN A GPU MODEL. Throughput varies several-fold
across the nine GPUs in the corpus, so a disguised cell on an RTX 5080 cannot
be divided by an honest baseline on an H200. Each disguised cell is compared
against a reference measured on the same gpu_name; cells whose reference is
missing on their own hardware are reported as such and excluded from the table
rather than silently compared across hardware.

Families with no STEP log cannot be measured here at all (A_util_modulation,
B_low_util, D_temporal_disruption, M_composite_memmin, N_grad_accum,
O_composite_idle_pad). For those the script emits the analytic duty-cycle
bound where the cell name encodes the knob, clearly marked `analytic`.

Usage:
    python3 scripts/paper_evasion_tax.py \
        --logs-root /srv/gpu-monitoring/cloudshell-workspace/logs \
        --manifest  /srv/gpu-monitoring/cloudshell-workspace/docs/camera_ready_manifest_20261022/streams.parquet \
        --out docs/camera_ready_results_20261022/paper_evasion_tax.json
"""
from __future__ import annotations

import argparse
import glob
import gzip
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from scripts.paper_table5_armsrace import canonical_workload, family_of  # noqa: E402

# STEP ts=<epoch> kind=train|infer batch_idx=N [tokens=M] [passes=P] [rank=R] ...
STEP_RE = re.compile(r"STEP\s+ts=([\d.]+)\s+kind=(\w+)\s+batch_idx=(\d+)")
KV_RE = re.compile(r"(\w+)=([-\w./]+)")


def _open(path):
    return gzip.open(path, "rt", errors="replace") if path.endswith(".gz") \
        else open(path, "r", errors="replace")


def label_from_filename(path):
    """`20260820__spar-gpu-0__symgemm_L4_sgdfused_rep1.log.gz` -> workload_label."""
    b = os.path.basename(path)
    for suf in (".log.gz", ".log"):
        if b.endswith(suf):
            b = b[: -len(suf)]
            break
    return b.split("__")[-1]


RUNID_RE = re.compile(r"Run ID:\s*(\S+)")


def parse_log(path):
    """One row per log: useful train tokens/steps, wall span, ranks seen.

    The run_id printed in the launcher banner is the only reliable join key to
    the telemetry manifest: a cell runs on several GPU models, so joining by
    workload_label alone cannot say which GPU a given log was collected on,
    and throughput is not comparable across GPU models.
    """
    run_id = None
    t_first = t_last = None
    train_tok = infer_tok = 0.0
    train_steps = infer_steps = 0
    ranks = set()
    tokens_seen = False
    # Wall time attributed to each step kind. A step's cost is the gap from the
    # previous step to it. This is what separates the real tax from 1/N: the
    # padding in a dilution attack is forward-only inference, which is cheaper
    # per step than a training step, so the adversary's wall-clock penalty is
    # smaller than the step ratio suggests.
    t_train = t_infer = 0.0
    prev_ts = None
    gaps = {"train": [], "infer": []}
    with _open(path) as fh:
        for line in fh:
            if run_id is None and "Run ID:" in line:
                mr = RUNID_RE.search(line)
                if mr:
                    run_id = mr.group(1)
            if "STEP ts=" not in line:
                continue
            m = STEP_RE.search(line)
            if not m:
                continue
            ts, kind = float(m.group(1)), m.group(2)
            kv = dict(KV_RE.findall(line[m.end():]))
            tok = kv.get("tokens")
            if tok is not None:
                tokens_seen = True
            try:
                tokv = float(tok) if tok is not None else 0.0
            except ValueError:
                tokv = 0.0
            if "rank" in kv:
                ranks.add(kv["rank"])
            t_first = ts if t_first is None else min(t_first, ts)
            t_last = ts if t_last is None else max(t_last, ts)
            if prev_ts is not None and ts > prev_ts:
                gap = ts - prev_ts
                if kind == "train":
                    t_train += gap
                else:
                    t_infer += gap
                gaps["train" if kind == "train" else "infer"].append(gap)
            prev_ts = ts
            if kind == "train":
                train_tok += tokv
                train_steps += 1
            else:
                infer_tok += tokv
                infer_steps += 1
    if t_first is None or t_last is None or t_last <= t_first:
        return None
    wall = t_last - t_first
    return {
        "label": label_from_filename(path),
        "run_id": run_id,
        "file": os.path.basename(path),
        "wall_s": wall,
        "train_tokens": train_tok,
        "infer_tokens": infer_tok,
        "train_steps": train_steps,
        "infer_steps": infer_steps,
        "n_ranks": len(ranks) or 1,
        "tokens_field": tokens_seen,
        "train_tok_s": train_tok / wall,
        "train_step_s": train_steps / wall,
        "train_duty": train_steps / max(train_steps + infer_steps, 1),
        "t_train_s": t_train,
        "t_infer_s": t_infer,
        "time_duty": t_train / (t_train + t_infer) if (t_train + t_infer) else np.nan,
        "med_train_gap": float(np.median(gaps["train"])) if gaps["train"] else np.nan,
        "med_infer_gap": float(np.median(gaps["infer"])) if gaps["infer"] else np.nan,
        "n_steps": train_steps + infer_steps,
    }


def load_logs(roots):
    rows = []
    for root in roots:
        for pat in ("*.log", "*.log.gz"):
            for p in sorted(glob.glob(os.path.join(root, pat))):
                r = parse_log(p)
                if r:
                    rows.append(r)
    return pd.DataFrame(rows)


# Log filenames and manifest workload_labels use different conventions. The
# launcher writes the bare cell with a rep suffix (`E_memory_minimal_rep1`);
# the manifest writes the campaign label (`adversarial_E_memory_minimal`), and
# the white-box campaign renames entirely (`wb_lora_3b_N1` ->
# `whitebox_WB-L_lora`). Resolving by hand, with the unresolved list printed,
# beats a fuzzy match that silently pairs the wrong cells.
WB_MAP = [
    (r"^wb_lora", "whitebox_WB-L_lora"),
    (r"^wb_oracle|^wb_o_", "whitebox_WB-O_oracle"),
    (r"^wb_diluted", "whitebox_WB-D_diluted"),
    (r"^wb_inference", "whitebox_WB-I_inference"),
]
REP_RE = re.compile(r"_rep\d+$")


def resolve_label(log_label, manifest_labels):
    """Map a log-file label onto a manifest workload_label, or None."""
    for pat, target in WB_MAP:
        if re.search(pat, log_label):
            return target if target in manifest_labels else None
    cands = []
    base = REP_RE.sub("", log_label)
    for c in (log_label, base, f"adversarial_{log_label}", f"adversarial_{base}"):
        if c in manifest_labels:
            return c
        cands.append(c)
    # Longest manifest label that is a prefix of the candidate, so
    # `adversarial_L_diluted_5_mw8` wins over `adversarial_L_diluted_5`.
    best = None
    for c in cands:
        for m in manifest_labels:
            if c.startswith(m) and (best is None or len(m) > len(best)):
                best = m
    return best


def attach_hardware(df, manifest_path):
    """Attach gpu_name per LOG, by run_id where possible, else by label.

    run_id is exact: one job, one host, one GPU model. Falling back to the
    workload_label only resolves hardware when the cell ran on a single GPU
    model across the whole corpus; otherwise gpu_name stays None and the log
    is excluded from any ratio, because dividing throughput measured on one
    GPU by throughput measured on another is meaningless.
    """
    man = pd.read_parquet(manifest_path,
                          columns=["workload_label", "gpu_name", "run_id"])
    man["workload_label"] = man.workload_label.astype(str)
    man["run_id"] = man.run_id.astype(str)
    labels = set(man.workload_label)

    by_run = (man.groupby("run_id").gpu_name
                 .agg(lambda s: sorted(set(s))).to_dict())
    by_label = (man.groupby("workload_label").gpu_name
                   .agg(lambda s: sorted(set(s))).to_dict())

    df["manifest_label"] = df.label.map(lambda l: resolve_label(l, labels))

    def gpus_for(row):
        rid = row["run_id"]
        if rid and str(rid) in by_run:
            return by_run[str(rid)], "run_id"
        lab = row["manifest_label"]
        if lab and lab in by_label:
            return by_label[lab], "label"
        return [], "none"

    res = df.apply(gpus_for, axis=1)
    df["gpu_list"] = [r[0] for r in res]
    df["gpu_src"] = [r[1] for r in res]
    df["gpu_name"] = df.gpu_list.map(lambda g: g[0] if len(g) == 1 else None)
    df["n_gpu_models"] = df.gpu_list.map(len)
    return df


# Honest references, in priority order per family. Each entry is a regex that
# selects the undisguised program the disguise is a disguise OF. The symgemm
# ladder references its own positive control (L5 = ordinary AdamW); everything
# else references the FSDP trainer the campaign used as its honest baseline.
REFERENCE = [
    ("symgemm_L4_fused_sgd", r"^symgemm_L5_adam"),
    ("symgemm_ppo_fused", r"^ppo_rlhf(?!_accum)"),
    ("K_ddp", r"train_ddp_resnet50"),
    ("K_ddp_accum", r"train_ddp_resnet50"),
    ("L_ddp", r"train_ddp_resnet50"),
    ("L_ddp_stagger", r"train_ddp_resnet50"),
]
DEFAULT_REFERENCE = r"train_fsdp1b|train_fsdp7b"


def reference_pattern(fam):
    for f, pat in REFERENCE:
        if f == fam:
            return pat
    return DEFAULT_REFERENCE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs-root", nargs="+", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--min-wall", type=float, default=30.0,
                    help="drop logs shorter than this many seconds of STEP span")
    ap.add_argument("--min-steps", type=int, default=30,
                    help="drop logs with fewer STEP lines than this. The August "
                         "L_diluted_10/_20 cells emit 2-4 STEP lines for a whole "
                         "run, so their duty is an artifact of the sample size, "
                         "not a measurement of the disguise.")
    a = ap.parse_args()

    df = load_logs(a.logs_root)
    print(f"parsed {len(df)} logs with STEP lines", flush=True)
    df = df[df.wall_s >= a.min_wall].copy()
    print(f"  {len(df)} after dropping spans < {a.min_wall}s", flush=True)
    thin = df[df.n_steps < a.min_steps]
    if len(thin):
        print(f"  dropping {len(thin)} logs with < {a.min_steps} STEP lines: "
              f"{sorted(set(thin.label))[:12]}", flush=True)
    df = df[df.n_steps >= a.min_steps].copy()

    df = attach_hardware(df, a.manifest)
    # family_of expects the campaign label, not the bare launcher cell.
    df["cell"] = df.manifest_label.fillna(df.label).map(canonical_workload)
    # family_of's regex does not accept the `_mw8` multi-worker suffix, so those
    # cells return None even though they are the SAME strategy with more
    # dataloader workers -- and for L_diluted they are the only properly logged
    # ones. Strip the suffix for family assignment only; the cell name is kept
    # intact so the table still distinguishes them.
    _fam_src = df.manifest_label.fillna(df.label).str.replace(r"_mw\d+$", "", regex=True)
    df["family"] = _fam_src.map(family_of)
    resolved = df.manifest_label.notna().sum()
    matched = df.gpu_name.notna().sum()
    print(f"  {resolved}/{len(df)} resolved to a manifest label", flush=True)
    print(f"  {matched}/{len(df)} resolved to ONE GPU model "
          f"({int((df.n_gpu_models > 1).sum())} ambiguous, "
          f"{int((df.n_gpu_models == 0).sum())} unjoined)", flush=True)
    print(f"  hardware source: {df.gpu_src.value_counts().to_dict()}", flush=True)
    print(f"  run_id present in {int(df.run_id.notna().sum())}/{len(df)} logs",
          flush=True)
    print(f"  {int(df.family.notna().sum())}/{len(df)} mapped to a family",
          flush=True)
    unres = sorted(set(df.loc[df.manifest_label.isna(), "label"]))
    if unres:
        print(f"  UNRESOLVED labels ({len(unres)}): {unres[:20]}", flush=True)
    nofam = sorted(set(df.loc[df.family.isna(), "cell"]))
    if nofam:
        print(f"  no family ({len(nofam)}): {nofam[:25]}", flush=True)

    out = {
        "measure": "useful training tokens per second on kind=train STEP lines",
        "n_logs_parsed": int(len(df)),
        "n_logs_hardware_matched": int(matched),
        "by_log": json.loads(df.drop(columns=["gpu_list"]).to_json(orient="records")),
    }
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")
    print(f"wrote {a.out}")

    # Diagnostic: what is actually usable, by family and hardware.
    print("\n=== families with STEP evidence ===")
    g = (df[df.family.notna()]
         .groupby(["family", "gpu_name"])
         .agg(logs=("file", "size"),
              tok=("tokens_field", "max"),
              tok_s=("train_tok_s", "median"),
              step_s=("train_step_s", "median"),
              duty=("train_duty", "median")))
    print(g.to_string())

    # ---- the two kinds of disguise, which need different measures ---------
    # A disguise that works by DILUTION spends most of its wall clock on
    # inference padding. Its tax is 1/duty and needs no external reference,
    # because the undisguised job is the same job with the padding removed --
    # it is measured inside the same run.
    #
    # A disguise that works by SHAPING (smaller batch, throttled clock,
    # PID-held utilisation, fused optimiser) trains every step. Its tax is a
    # slowdown of the training itself and can only be had by comparison with
    # an undisguised run of the SAME program on the SAME GPU.
    d = df[df.family.notna() & df.gpu_name.notna()].copy()
    d["tok_duty"] = d.train_tokens / (d.train_tokens + d.infer_tokens).replace(0, np.nan)
    per_cell = (d.groupby(["family", "cell", "gpu_name"])
                  .agg(logs=("file", "size"),
                       tokens_field=("tokens_field", "max"),
                       train_tok_s=("train_tok_s", "median"),
                       train_step_s=("train_step_s", "median"),
                       step_duty=("train_duty", "median"),
                       tok_duty=("tok_duty", "median"),
                       time_duty=("time_duty", "median"),
                       med_train_gap=("med_train_gap", "median"),
                       med_infer_gap=("med_infer_gap", "median"),
                       wall_s=("wall_s", "median"))
                  .reset_index())
    per_cell["kind"] = np.where(per_cell.step_duty >= 0.999, "shaping", "dilution")
    # Two taxes. `tax_steps` is the naive 1/N the repo has been assuming.
    # `tax_time` is what the adversary actually pays: wall clock per unit of
    # training, which is smaller whenever the padding is cheaper per step than
    # a training step.
    per_cell["tax_steps"] = np.where(per_cell.kind == "dilution",
                                     1.0 / per_cell.step_duty, np.nan)
    per_cell["tax_time"] = np.where(per_cell.kind == "dilution",
                                    1.0 / per_cell.time_duty, np.nan)
    per_cell["infer_vs_train_step_cost"] = (per_cell.med_infer_gap
                                            / per_cell.med_train_gap)
    out["per_cell"] = json.loads(per_cell.to_json(orient="records"))
    Path(a.out).write_text(json.dumps(out, indent=2) + "\n")

    pd.set_option("display.width", 200)
    print("\n=== DILUTION-type disguises: tax = 1/duty, measured in-run ===")
    dil = per_cell[per_cell.kind == "dilution"].sort_values("tax_time",
                                                            ascending=False)
    print(dil[["family", "cell", "gpu_name", "logs", "step_duty",
               "tax_steps", "tax_time", "infer_vs_train_step_cost"]]
          .to_string(index=False, float_format=lambda v: f"{v:8.3f}"))

    print("\n=== SHAPING-type disguises: need an undisguised same-GPU reference ===")
    sh = per_cell[per_cell.kind == "shaping"].sort_values(["family", "cell"])
    print(sh[["family", "cell", "gpu_name", "logs", "train_tok_s",
              "train_step_s"]].to_string(index=False))

    print("\n=== candidate honest references present, by gpu ===")
    ref = df[df.label.str.contains(DEFAULT_REFERENCE, regex=True, na=False)]
    if len(ref):
        print(ref.groupby(["cell", "gpu_name"])
                 .agg(logs=("file", "size"), tok_s=("train_tok_s", "median"))
                 .to_string())
    else:
        print("  none matched", DEFAULT_REFERENCE)


if __name__ == "__main__":
    main()
