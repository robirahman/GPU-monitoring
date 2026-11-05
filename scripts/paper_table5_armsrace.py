#!/usr/bin/env python3
"""Table 5, rebuilt as the sequential arms race it describes.

The published Table 5 was produced (see scripts/paper_table57.py) by fitting ONE
clean-only baseline and ONE all-in hardened CV, then slicing both by round. That
is not the procedure the paper's §4.4 narrative describes. This script runs the
narrative literally:

    corpus_0 = the benign corpus (no evasion data at all)
    for N in 1..6:
        baseline(N)    fit on corpus_{N-1}; predict round-N evasion families.
                       The detector meeting attack N for the first time.
        corpus_N       = corpus_{N-1} + ALL round-N collection (the evasion
                         cells AND the negatives/controls collected with them)
        in_dist(N)     grouped 5-fold CV over corpus_N; out-of-fold predictions
                       on round-N families. The hardened detector on attacks it
                       has trained against.
        unseen(N)      for each round-N family f: fit on corpus_N minus f,
                       predict f. The hardened detector on a sibling attack it
                       has never seen.

Only in_dist uses cross-validation, so only that column depends on the holdout
key; baseline and unseen are fit/predict on disjoint data and are identical
under all three protocols. The script therefore computes them once and sweeps
the holdout key over in_dist alone.

Round membership is by COLLECTION CAMPAIGN, not by label. Round 5's inference
control (whitebox_inference) and the whole Round 6 campaign (the symmetric-GEMM
negatives, undisguised PPO-RLHF, 16-rank distributed RL) enter the corpus at
their own round. Putting them in corpus_0 would hand the round-1 defender data
that did not exist until 2026-08 and flatter every baseline above it.

    PYTHONPATH=. python3 scripts/paper_table5_armsrace.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from classifier.threeway_improved import get_feature_cols  # noqa: E402
from classifier.workload_labels import canonical_workload  # noqa: E402

CACHE = PROJECT_ROOT / "cache"
OUTDIR = PROJECT_ROOT / "docs/camera_ready_results_20261022"
FOLDS = 5

# Round 5 defender response (§4.4); no classifier before R5 has these.
PLATEAU_FEATS = {"first_30s_mem_delta_mb", "first_30s_mem_delta_frac",
                 "first_30s_mem_ratio", "time_to_mem_plateau_samples"}
# Round 3 defender response; available from R4 onward.
PHYSICS_FEATS = ["cumulative_energy_wh", "power_mem_correlation",
                 "util_bimodal_score", "power_efficiency_ratio",
                 "sustained_compute_frac"]


# ---------------------------------------------------------------- taxonomy --
# family -> (round, is_evasion). is_evasion False marks a cell that joined the
# corpus in that round as a control or benign anchor: it expands corpus_N but is
# never scored as an attack.
# Round 4 is published as two rows. Keep that split: the multi-GPU variants are
# a different attack surface and the published ranges differ sharply.
SUBGROUP = {
    "M_composite_memmin": "single-GPU", "N_grad_accum": "single-GPU",
    "O_composite_idle_pad": "single-GPU",
    "K_ddp": "multi-GPU", "K_ddp_accum": "multi-GPU",
    "L_ddp": "multi-GPU", "L_ddp_stagger": "multi-GPU",
}

FAMILY_SPEC = {
    "A_util_modulation":   (1, True),
    "B_low_util":          (1, True),
    "D_temporal_disruption": (1, True),

    "E_memory_minimal":    (2, True),
    "F_interleave":        (2, True),
    "G_clock_throttled":   (2, True),
    "H_mimicry":           (2, True),
    "I_stochastic":        (2, True),
    "J_pid":               (2, True),

    "K_online_learning":   (3, True),
    "L_diluted":           (3, True),

    "M_composite_memmin":  (4, True),
    "N_grad_accum":        (4, True),
    "O_composite_idle_pad": (4, True),
    "K_ddp":               (4, True),
    "K_ddp_accum":         (4, True),
    "L_ddp":               (4, True),
    "L_ddp_stagger":       (4, True),

    "whitebox_full":       (5, True),
    "whitebox_lora":       (5, True),
    "whitebox_inference":  (5, False),   # R5 inference control

    "symgemm_L4_fused_sgd": (6, True),   # the attack: write-minimised SGD
    "symgemm_ppo_fused":    (6, True),   # disguised PPO, fused policy update
    "symgemm_L5_adam":      (6, False),  # positive control: undisguised AdamW
    "symgemm_negatives":    (6, False),  # L0/L1/L3/decode/idle
    "ppo_rlhf":             (6, False),  # undisguised PPO, benign anchor
    "rl_distributed":       (6, False),  # 16-rank IMPALA/PPO/Ape-X
}


# Adversarial cells that belong to no round. `adversarial_training` is three
# 20-second B200 streams from 2026-04-13 whose label carries no strategy letter,
# so it cannot be placed in the sequence. It is disguised training, so it cannot
# go in corpus_0 either -- that would seed the pre-race benign corpus with an
# attack. Three windows; dropped, and counted in the output so the drop is not
# silent.
DROP_CELLS = {"adversarial_training"}

# Cells that are INSTRUMENTS, not workloads. The baseline corpus is meant to be
# things people organically run; `symgemm_L3_graddiscard` computes the weight
# gradient and discards it, which nobody runs for its own sake. It stays a
# probe reported beside the Round 6 result and is excluded from the corpus
# entirely, the same way an out-of-scope multi-node job is.
#
# L0 and L1 are kept: both are sustained batched forward passes, which is what
# any batch-inference, scoring, embedding or perplexity-eval job is. L1's
# "three forwards per iteration" is bookkeeping in the harness, not a different
# kind of compute. Measured consequence of this split, from
# scripts/paper_table5_l3_sensitivity.py: dropping L3 alone moves baseline
# detection of symgemm_L4 from 0.7% to 1.2%; dropping L0 and L1 as well would
# move it to 20.7%, so the Round 6 headline rests on L0/L1, not on L3.
INSTRUMENT_CELLS = {"symgemm_L3_graddiscard"}

MANIFEST = PROJECT_ROOT / "docs/camera_ready_manifest_20261022/streams.parquet"


def out_of_scope_runs(max_gpus=8, single_node=True):
    """run_ids of jobs that exceed the paper's single-node scope.

    The paper is about what one node's telemetry shows, so a job spread over
    more than one host, or holding more GPUs than a node has, is out of scope
    however interesting its trace is. Counted per job over distinct gpu_uuid and
    hostname in the manifest, because the window frame carries neither.
    """
    man = pd.read_parquet(MANIFEST)
    per = man.groupby("job_id").agg(gpus=("gpu_uuid", "nunique"),
                                    hosts=("hostname", "nunique"))
    bad = per[(per.gpus > max_gpus) | (single_node & (per.hosts > 1))].index
    sub = man[man.job_id.isin(bad)]
    return set(sub.run_id.astype(str)), sub


def family_of(label):
    """Collection family for a window label, or None if it predates the race.

    Returns a name in FAMILY_SPEC. Anything unmapped returns None and lands in
    corpus_0; main() refuses to run if an adversarial-looking label does.
    """
    s = canonical_workload(str(label))
    lc = s.lower()

    if lc.startswith("symgemm_"):
        if "_ppo" in lc:
            return "symgemm_ppo_fused"
        if "_l4" in lc:
            return "symgemm_L4_fused_sgd"
        if "_l5" in lc:
            return "symgemm_L5_adam"
        return "symgemm_negatives"          # L0, L1, L3, decode, idle
    if lc.startswith("ppo_rlhf"):
        return "ppo_rlhf"
    if re.match(r"^s\d+_train_ppodist", lc):
        return "rl_distributed"

    if lc.startswith("whitebox"):
        if "inference" in lc:
            return "whitebox_inference"
        return "whitebox_lora" if "lora" in lc else "whitebox_full"

    if lc.startswith("adversarial_composite"):
        return "O_composite_idle_pad"
    m = re.match(r"^adversarial_([a-z]_[a-z_]+?)(?:_\d+)*$", lc)
    if m:
        base = m.group(1)
        for known in sorted(FAMILY_SPEC, key=len, reverse=True):
            if base == known.lower() or base.startswith(known.lower()):
                return known
    return None


def forest():
    return RandomForestClassifier(n_estimators=400, class_weight="balanced",
                                  random_state=42, n_jobs=-1)


def features_for_round(n, standard, plateau, physics):
    """Feature set a round-N-era classifier had, per the §4.4 response timeline.

    Physics features joined as the Round 3 defender response (available from the
    classifier that meets Round 4), memory-plateau features as the Round 5
    response (available from the one that meets Round 6).

    The causal window cache carries no physics features at all: they came from
    `add_evasion_resistant_features`, which is the function that caused the
    within-run temporal leak and is not part of the corrected pipeline. So on
    this cache the physics half of the timeline is inert and only the plateau
    half bites. main() says so rather than letting the column look complete.
    """
    cols = list(standard)
    if n >= 4:
        cols += physics
    if n >= 6:
        cols += plateau
    return cols


# ------------------------------------------------------------------- main --
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=float, default=30.0)
    ap.add_argument("--cache", default=str(CACHE / "win_30_r6.parquet"))
    ap.add_argument("--group-by", nargs="+",
                    default=["job_id", "run_id", "workload"])
    ap.add_argument("--max-round", type=int, default=6)
    ap.add_argument("--max-gpus", type=int, default=8,
                    help="jobs above this many GPUs are out of the paper's scope")
    ap.add_argument("--allow-multinode", action="store_true",
                    help="keep jobs spanning more than one host")
    ap.add_argument("--keep-instruments", action="store_true",
                    help="keep INSTRUMENT_CELLS in the corpus (default: drop)")
    ap.add_argument("--controls-round", type=int, default=None, choices=[0],
                    help="0 moves EVERY non-adversarial family (the Round 5 "
                         "inference control, the symmetric-GEMM negatives, the "
                         "undisguised AdamW positive control and undisguised "
                         "PPO-RLHF) into the pre-race benign corpus, so only "
                         "evasion families are round-indexed. This is the rule "
                         "Table 5's own caption states: 'Baseline classifiers "
                         "are trained only on non-adversarial data.'")
    ap.add_argument("--ppo-rlhf-round", type=int, default=6, choices=[0, 6],
                    help="0 puts undisguised PPO-RLHF in the pre-race benign "
                         "corpus; 6 introduces it with the Round 6 campaign "
                         "that collected it")
    ap.add_argument("--out", default=str(OUTDIR / "paper_table5_armsrace.json"))
    ap.add_argument("--tag", default="", help="suffix for the output filename")
    a = ap.parse_args()
    t0 = time.time()
    # Undisguised PPO-RLHF is benign training, not an evasion, but it was
    # collected during the rebuttal (2026-08) alongside Round 6. Placing it at
    # round 0 asks "what would a detector built on today's benign corpus do";
    # placing it at round 6 asks "what did the published detector do". The flag
    # makes the choice explicit instead of implicit in a taxonomy table.
    FAMILY_SPEC["ppo_rlhf"] = (a.ppo_rlhf_round, False)
    if a.controls_round == 0:
        for f, (r, e) in list(FAMILY_SPEC.items()):
            if not e:
                FAMILY_SPEC[f] = (0, False)

    win = pd.read_parquet(a.cache)
    win["cell"] = [canonical_workload(l) for l in win.workload_label]

    oos, oos_streams = out_of_scope_runs(a.max_gpus, not a.allow_multinode)
    m = win.run_id.astype(str).isin(oos)
    if m.any():
        print(f"out of scope (>{a.max_gpus} GPUs or multi-node): "
              f"{int(m.sum()):,} windows, "
              f"{win.loc[m, 'cell'].nunique()} cells "
              f"{sorted(set(win.loc[m, 'cell']))}", flush=True)
        for _, r in (oos_streams.groupby(["base", "hw"])
                     .agg(gpus=("gpu_uuid", "nunique"), hosts=("hostname", "nunique"))
                     .reset_index().iterrows()):
            print(f"    {r.base:28s} {r.hw:16s} gpus={r.gpus} hosts={r.hosts}",
                  flush=True)
        win = win.loc[~m].reset_index(drop=True)
    else:
        print(f"out of scope (>{a.max_gpus} GPUs or multi-node): none", flush=True)
    drop = win.cell.isin(DROP_CELLS)
    if not a.keep_instruments:
        inst = win.cell.isin(INSTRUMENT_CELLS)
        if inst.any():
            print(f"instruments excluded from the corpus: "
                  f"{int(inst.sum()):,} windows, "
                  f"{sorted(set(win.loc[inst, 'cell']))}", flush=True)
        drop = drop | inst
    if drop.any():
        print(f"dropping {int(drop.sum())} windows from unplaceable cells: "
              f"{sorted(set(win.loc[drop, 'cell']))}", flush=True)
        win = win.loc[~drop].reset_index(drop=True)
    win["family"] = [family_of(l) for l in win.workload_label]

    # Refuse to proceed if anything that looks adversarial fell through the
    # taxonomy: an unmapped attack would silently join the benign corpus and
    # every baseline above it would be measured against its own attack.
    advish = win.workload_label.str.contains(
        r"advers|whitebox|symgemm|ppo|ppodist", case=False, regex=True, na=False)
    stray = sorted(set(win.loc[advish & win.family.isna(), "workload_label"]))
    if stray:
        print("UNMAPPED adversarial-looking labels:", flush=True)
        for s in stray:
            print(f"    {s}  n={int((win.workload_label == s).sum())}", flush=True)
        raise SystemExit("taxonomy incomplete; refusing to run")

    rnd = win.family.map(lambda f: FAMILY_SPEC[f][0] if isinstance(f, str) else 0)
    win["round"] = rnd.astype(int)

    # Feature columns come from the cache's own columns, before any bookkeeping
    # column is attached. `round` is numeric and get_feature_cols does not
    # exclude it, so computing the feature list after adding it hands the model
    # the round number itself -- and since corpus_{N-1} contains no row with
    # round N, that one column alone separates every attack from its training
    # set. Caught by scripts/paper_table5_round6_ablation.py, which disagreed
    # with this script by 77 points on symgemm_L4.
    feats = [c for c in get_feature_cols(win)
             if c not in {"round", "cell", "family", "subgroup"}]
    for bad in ("round", "cell", "family", "subgroup"):
        assert bad not in feats, bad
    plateau = sorted(set(feats) & PLATEAU_FEATS)
    physics = [c for c in PHYSICS_FEATS if c in feats]
    standard = [c for c in feats if c not in PLATEAU_FEATS and c not in physics]
    X_all = win[feats].astype(float).fillna(0.0).to_numpy(np.float32)
    col = {c: i for i, c in enumerate(feats)}
    y = (win.threeway_label == "ml_training").to_numpy(int)
    fam = win.family.to_numpy(object)
    rno = win["round"].to_numpy(int)

    print(f"{len(win):,} windows, {len(feats)} features "
          f"({len(standard)} standard, {len(physics)} physics, {len(plateau)} plateau)")
    if not physics:
        print("  NOTE: no Round-3 physics features in this cache (they came from "
              "add_evasion_resistant_features, the leaky path). The era-feature "
              "column therefore models only the Round-5 plateau response.")
    print(f"corpus_0 (benign, pre-race): {int((rno == 0).sum()):,} windows, "
          f"{int(y[rno == 0].sum()):,} training")
    for n in range(1, a.max_round + 1):
        m = rno == n
        ev = [f for f, (r, e) in FAMILY_SPEC.items() if r == n and e]
        print(f"  round {n}: {int(m.sum()):6,} windows  "
              f"{len(ev)} evasion families  "
              f"cells={win.loc[m, 'cell'].nunique():3d}  "
              f"train-labelled={int(y[m].sum()):,}")
    print(flush=True)

    def Xc(cols):
        return X_all[:, [col[c] for c in cols]]

    rows = []            # per family, per round
    round_rows = []      # per round aggregate
    diag = []
    clean_clf = None     # fit on corpus_0; the published "Baseline det." column

    # ---- the published table's own "Baseline" row ---------------------------
    # "the baseline classifier's accuracy at evaluating held-out non-adversarial
    # workloads when trained on all other non-adversarial workloads" -- grouped
    # CV over corpus_0 alone, no evasion data anywhere in it.
    b0 = np.flatnonzero(rno == 0)
    baseline_row = {"round": 0, "label": "Baseline (no evasion)",
                    "windows": int(len(b0))}
    for gb in a.group_by:
        gcol = (win.cell.to_numpy(object) if gb == "workload"
                else win[gb].astype(str).to_numpy())
        g = gcol[b0]
        k = min(FOLDS, len(np.unique(g)))
        pred = np.full(len(b0), -1, dtype=int)
        for tr, te in StratifiedGroupKFold(
                n_splits=k, shuffle=True, random_state=42).split(X_all[b0], y[b0], g):
            pred[te] = forest().fit(X_all[b0][tr], y[b0][tr]).predict(X_all[b0][te])
        yy = y[b0]
        tp = int(((pred == 1) & (yy == 1)).sum()); fp = int(((pred == 1) & (yy == 0)).sum())
        tn = int(((pred == 0) & (yy == 0)).sum()); fn = int(((pred == 0) & (yy == 1)).sum())
        baseline_row[f"accuracy_{gb}"] = float((pred == yy).mean())
        baseline_row[f"recall_{gb}"] = tp / max(tp + fn, 1)
        baseline_row[f"specificity_{gb}"] = tn / max(tn + fp, 1)
        baseline_row[f"groups_{gb}"] = int(len(np.unique(g)))
        print(f"[Baseline row] group_by={gb} k={k} groups={len(np.unique(g))} "
              f"acc={baseline_row[f'accuracy_{gb}']*100:.2f}% "
              f"recall={baseline_row[f'recall_{gb}']*100:.2f}% "
              f"spec={baseline_row[f'specificity_{gb}']*100:.2f}% "
              f"[{time.time()-t0:.0f}s]", flush=True)

    for n in range(1, a.max_round + 1):
        evasions = sorted(f for f, (r, e) in FAMILY_SPEC.items() if r == n and e)
        controls = sorted(f for f, (r, e) in FAMILY_SPEC.items() if r == n and not e)
        if n == a.max_round:   # benign anchors parked in corpus_0 get one row too
            controls += sorted(f for f, (r, e) in FAMILY_SPEC.items()
                               if r == 0 and not e)
        if not evasions:
            continue
        prior = rno < n                       # corpus_{N-1}
        upto = rno <= n                       # corpus_N

        # ---- 1. baseline: the end-of-round-(N-1) detector, first contact ----
        base_clf = forest().fit(X_all[prior], y[prior])
        if clean_clf is None:          # corpus_0 == corpus_{N-1} at N=1
            clean_clf = base_clf
        # Same fit restricted to the feature set that round's defender actually
        # had, as a sensitivity check on the §4.4 response timeline.
        era_cols = features_for_round(n, standard, plateau, physics)
        Xe = Xc(era_cols)
        era_clf = forest().fit(Xe[prior], y[prior])
        era_n = len(era_cols)
        print(f"[R{n}] corpus_{n-1}={int(prior.sum()):,}w  "
              f"corpus_{n}={int(upto.sum()):,}w  "
              f"baseline fitted [{time.time()-t0:.0f}s]", flush=True)

        # ---- 3. unseen: hardened on corpus_N minus this family --------------
        unseen_pred = {}
        for f in evasions:
            tr = upto & (fam != f)
            clf = forest().fit(X_all[tr], y[tr])
            sel = (fam == f) & (y == 1)
            unseen_pred[f] = (clf.predict(X_all[sel]) == 1)
            print(f"   unseen fit done: {f} [{time.time()-t0:.0f}s]", flush=True)

        # ---- 2. in-distribution: grouped CV over corpus_N, per protocol -----
        idx = np.flatnonzero(upto)
        Xn, yn = X_all[idx], y[idx]
        oof = {}
        for gb in a.group_by:
            gcol = (win.cell.to_numpy(object) if gb == "workload"
                    else win[gb].astype(str).to_numpy())
            g = gcol[idx]
            k = min(FOLDS, len(np.unique(g)))
            p = np.full(len(idx), -1, dtype=int)
            for tr, te in StratifiedGroupKFold(
                    n_splits=k, shuffle=True, random_state=42).split(Xn, yn, g):
                p[te] = forest().fit(Xn[tr], yn[tr]).predict(Xn[te])
            full = np.full(len(y), -1, dtype=int)
            full[idx] = p
            oof[gb] = full
            # Benign specificity of the hardened detector on corpus_N.
            ben = upto & (y == 0)
            spec = float((full[ben] == 0).mean())
            diag.append({"round": n, "group_by": gb, "folds": int(k),
                         "groups": int(len(np.unique(g))),
                         "corpus_windows": int(upto.sum()),
                         "benign_windows": int(ben.sum()),
                         "hardened_specificity": spec})
            print(f"   in-dist CV done: group_by={gb} k={k} "
                  f"groups={len(np.unique(g))} spec={spec*100:.2f}% "
                  f"[{time.time()-t0:.0f}s]", flush=True)

        # ---- assemble ------------------------------------------------------
        for f in evasions + controls:
            # Evasion families are scored on their training windows (recall).
            # Some controls -- the symmetric-GEMM negatives, the Round 5
            # inference control -- carry no training windows at all; for those
            # the same number is a FALSE POSITIVE rate, and leaving them out
            # would report only half of the decisive L1-vs-L4 comparison.
            sel = (fam == f) & (y == 1)
            scored = "training"
            if not sel.sum():
                sel = (fam == f) & (y == 0)
                scored = "non-training"
            if not sel.sum():
                continue
            r = {
                "scored_class": scored,
                "round": FAMILY_SPEC[f][0], "family": f,
                "is_evasion": FAMILY_SPEC[f][1],
                "windows": int(sel.sum()),
                "cells": int(win.loc[sel, "cell"].nunique()),
                "runs": int(win.loc[sel, "run_id"].nunique()),
                "subgroup": SUBGROUP.get(f),
                "baseline_clean_only": float((clean_clf.predict(X_all[sel]) == 1).mean()),
                "prev_round": float((base_clf.predict(X_all[sel]) == 1).mean()),
                "prev_round_era_features": float((era_clf.predict(Xe[sel]) == 1).mean()),
                "era_n_features": era_n,
                "unseen": (float(unseen_pred[f].mean())
                           if f in unseen_pred else None),
            }
            for gb in a.group_by:
                r[f"in_dist_{gb}"] = float((oof[gb][sel] == 1).mean())
            rows.append(r)
            tag = ("" if r["is_evasion"] else
                   "  (control)" if r["scored_class"] == "training" else
                   "  (CONTROL, rates are FALSE POSITIVES)")
            cols_s = "  ".join(f"{gb[:4]} {r['in_dist_'+gb]*100:5.1f}%"
                               for gb in a.group_by)
            print(f"   {f:24s} n={r['windows']:6,}  clean {r['baseline_clean_only']*100:5.1f}%"
                  f"  prev {r['prev_round']*100:5.1f}%"
                  f"  unseen {('%5.1f%%' % (r['unseen']*100)) if r['unseen'] is not None else '    -'}"
                  f"  in-dist[{cols_s}]{tag}", flush=True)

        ev_rows = [r for r in rows if r["round"] == n and r["is_evasion"]]
        subs = sorted({r["subgroup"] for r in ev_rows}, key=lambda x: (x is None, x))
        for sg in subs:
            grp = [r for r in ev_rows if r["subgroup"] == sg]
            agg = {"round": n, "subgroup": sg,
                   "label": f"{n}" + (f" ({sg})" if sg else ""),
                   "families": len(grp),
                   "windows": int(sum(r["windows"] for r in grp))}
            for key in ("baseline_clean_only", "prev_round",
                        "prev_round_era_features", "unseen"):
                agg[f"{key}_min"] = min(r[key] for r in grp)
                agg[f"{key}_max"] = max(r[key] for r in grp)
            for gb in a.group_by:
                vs = [r[f"in_dist_{gb}"] for r in grp]
                agg[f"in_dist_{gb}_min"] = min(vs)
                agg[f"in_dist_{gb}_max"] = max(vs)
            round_rows.append(agg)

    out = Path(a.out)
    if a.tag:
        out = out.with_name(out.stem + "_" + a.tag + out.suffix)
    out.write_text(json.dumps({
        "table": 5, "procedure": "sequential arms race",
        "window_sec": a.window, "cache": a.cache, "folds": FOLDS,
        "n_features": len(feats), "group_by": a.group_by,
        "ppo_rlhf_round": a.ppo_rlhf_round, "max_gpus": a.max_gpus,
        "controls_round": a.controls_round,
        "instruments_excluded": (sorted(INSTRUMENT_CELLS)
                                 if not a.keep_instruments else []),
        "family_round": {f: r for f, (r, e) in FAMILY_SPEC.items()},
        "single_node_only": not a.allow_multinode,
        "baseline_row": baseline_row,
        "by_family": rows, "by_round": round_rows, "diagnostics": diag,
    }, indent=2) + "\n")
    print(f"\nwrote {out}  [{time.time()-t0:.0f}s]")

    def rng(lo, hi):
        return f"{lo*100:.1f}%" if abs(hi - lo) < 5e-4 else f"{lo*100:.1f}-{hi*100:.1f}%"

    df = pd.DataFrame(round_rows)
    for gb in a.group_by:
        print(f"\n=== Table 5  (sequential arms race; in-dist holdout = {gb}) ===")
        print(f"  Baseline (no evasion): accuracy {baseline_row[f'accuracy_{gb}']*100:.1f}%  "
              f"recall {baseline_row[f'recall_{gb}']*100:.1f}%  "
              f"specificity {baseline_row[f'specificity_{gb}']*100:.1f}%  "
              f"({baseline_row['windows']:,} windows, "
              f"{baseline_row[f'groups_{gb}']} groups)")
        out_t = pd.DataFrame({
            "round": df.label, "fam": df.families, "windows": df.windows,
            "baseline_det": [rng(l, h) for l, h in
                             zip(df.baseline_clean_only_min, df.baseline_clean_only_max)],
            "prev_round_det": [rng(l, h) for l, h in
                               zip(df.prev_round_min, df.prev_round_max)],
            "hardened_in_dist": [rng(l, h) for l, h in
                                 zip(df[f"in_dist_{gb}_min"], df[f"in_dist_{gb}_max"])],
            "hardened_unseen": [rng(l, h) for l, h in
                                zip(df.unseen_min, df.unseen_max)],
        })
        print(out_t.to_string(index=False))

    print("\n=== per-family detail (in-dist columns, all protocols) ===")
    fd = pd.DataFrame(rows)
    show = ["round", "family", "is_evasion", "windows", "cells", "runs",
            "baseline_clean_only", "prev_round", "unseen"] + \
           [f"in_dist_{gb}" for gb in a.group_by]
    pc = [c for c in show if c not in ("round", "family", "is_evasion",
                                       "windows", "cells", "runs")]
    fdp = fd[show].copy()
    for c in pc:
        fdp[c] = (fdp[c].astype(float) * 100).round(1)
    print(fdp.to_string(index=False))


if __name__ == "__main__":
    main()
