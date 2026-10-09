"""Does llm.c workload-shaping evade the training-vs-rest classifier, and do the
DCGM / oscilloscope tiers catch it?

Two evaluations:

  HEADLINE  — the paper's NVML-only classifier (trained on the full corpus with
              the shaped family held out) scores each shaped cadence. Evasion
              rate = fraction of a cadence's windows predicted "rest". The
              unshaped baseline_train is a sanity control (must stay detected).

  ABLATION  — a node-local balanced set where every run carries NVML + scope +
              DCGM (the reference set + llm.c baseline as positives/negatives,
              shaped held out). Identical runs/windows; we vary only the feature
              tier — NVML, NVML+DCGM, NVML+scope, all — so the drop in evasion
              is attributable to the added sensor, not to different data.

Run from repo root:  python -m classifier.eval_shaping_evasion
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classifier.threeway_improved import (
    _normalize_columns, threeway_label, sliding_windows, get_feature_cols,
    PROJECT_ROOT,
)
from classifier.scope_features import (
    build_scope_path_map, attach_scope_features, SCOPE_FEATURE_NAMES,
)

DATA = PROJECT_ROOT / "data"
SHAPED_LABELS = ["adversarial_llmc_shaped_g20", "adversarial_llmc_shaped_g35",
                 "adversarial_llmc_shaped_g50", "adversarial_llmc_shaped_g70",
                 "adversarial_llmc_shaped_g100"]
# memory-evader families: minimize the resident footprint (b=1 + grad-accum),
# and pad it to mimic an inference server's flat footprint (-Sm). These attack
# the mem_used_mb feature the shaping-only evader could not touch.
MEM_LABELS = ["adversarial_mm_min_noshape", "adversarial_mm_min_shaped",
              "adversarial_mm_spoof_infer_shaped"]
EVADER_LABELS = SHAPED_LABELS + MEM_LABELS
DCGM_COLS = ["dcgm_gr_engine_active", "dcgm_sm_active", "dcgm_sm_occupancy",
             "dcgm_tensor_active", "dcgm_dram_active", "dcgm_fp64_pipe_active",
             "dcgm_fp32_pipe_active", "dcgm_fp16_pipe_active"]


def _is_sensor_sidecar(path: str) -> bool:
    b = os.path.basename(path)
    return ("_scope_" in b) or ("_dcgm_" in b) or b.endswith("_raw.npy")


def _dcgm_map() -> dict:
    """run_id (12-hex from filename) -> dcgm parquet path."""
    m = {}
    for p in glob.glob(str(DATA / "**" / "*_dcgm_*.parquet"), recursive=True):
        rid = os.path.basename(p).split("_dcgm_", 1)[1].split("_")[0]
        m[rid] = p
    return m


def load_runs(only_dirs=None, attach_dcgm=True) -> pd.DataFrame:
    """Load telemetry parquets (excluding sensor sidecars), optionally merging
    DCGM columns per-sample by nearest epoch. Returns a normalized long frame."""
    dcgm_paths = _dcgm_map() if attach_dcgm else {}
    frames = []
    for p in sorted(DATA.rglob("*.parquet")):
        sp = str(p)
        if _is_sensor_sidecar(sp):
            continue
        if only_dirs and not any(d in sp for d in only_dirs):
            continue
        try:
            df = _normalize_columns(pd.read_parquet(p))
        except Exception:
            continue
        if "gpu_utilization_pct" not in df.columns or "ts" not in df.columns:
            continue
        stem = p.stem
        if "workload_label" not in df.columns:
            df["workload_label"] = stem.split("_NVIDIA_")[0] if "_NVIDIA_" in stem else stem.rsplit("_", 3)[0]
        rid = str(df["run_id"].iloc[0]) if "run_id" in df.columns else stem
        df["run_id"] = rid
        # merge DCGM by nearest epoch if a sibling exists for this run_id
        if attach_dcgm:
            # our runs store the 12-hex run_id in the column; sidecar keyed on same
            dpath = dcgm_paths.get(rid) or dcgm_paths.get(rid[:12])
            if dpath:
                g = pd.read_parquet(dpath).sort_values("epoch")
                left = df.sort_values("ts")
                merged = pd.merge_asof(left, g[["epoch"] + [c for c in DCGM_COLS if c in g.columns]],
                                       left_on="ts", right_on="epoch",
                                       direction="nearest", tolerance=2.0)
                for c in DCGM_COLS:
                    if c in merged.columns:
                        df[c] = merged[c].values
        df["threeway_label"] = df["workload_label"].astype(str).map(threeway_label)
        frames.append(df)
    return pd.concat(frames, ignore_index=True)


def windows_with_scope(df: pd.DataFrame, window_sec=30, stride_sec=15) -> pd.DataFrame:
    # 30 s / 15 s matches the paper's binary training-vs-rest headline and yields
    # ~5 windows per ~100 s run (60 s would give only ~2, too few for stable
    # per-cadence rates).
    w = sliding_windows(df, window_sec=window_sec, stride_sec=stride_sec,
                        run_level_mode="causal")
    w = attach_scope_features(w, build_scope_path_map([str(DATA)]))
    return w


def feature_subset(all_cols, tier):
    dcgm = [c for c in all_cols if c.startswith("dcgm_")]
    scope = [c for c in all_cols if c in SCOPE_FEATURE_NAMES]
    base = [c for c in all_cols if c not in dcgm and c not in scope]
    if tier == "nvml":
        return base
    if tier == "nvml+dcgm":
        return base + dcgm
    if tier == "nvml+scope":
        return base + scope
    return base + dcgm + scope  # all


def evasion_by_label(train_df, test_df, feat_cols):
    Xtr = train_df[feat_cols].fillna(0).values.astype(np.float32)
    ytr = (train_df["threeway_label"] == "ml_training").astype(int).values
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    rows = {}
    for lbl, sub in test_df.groupby("workload_label"):
        X = sub[feat_cols].fillna(0).values.astype(np.float32)
        pred_train = clf.predict(X)
        det = float(pred_train.mean())               # fraction detected as training
        rows[lbl] = {"n_windows": len(sub), "detection": det, "evasion": 1.0 - det}
    return rows, clf


def main():
    pd.set_option("display.width", 140)
    print("Loading runs (this scans data/ and merges DCGM)…", flush=True)
    df = load_runs(attach_dcgm=True)
    win = windows_with_scope(df)
    all_cols = get_feature_cols(win)
    print(f"windows={len(win)}  feature_cols={len(all_cols)} "
          f"(dcgm={sum(c.startswith('dcgm_') for c in all_cols)}, "
          f"scope={sum(c in SCOPE_FEATURE_NAMES for c in all_cols)})")

    shaped_mask = win["workload_label"].isin(SHAPED_LABELS)

    # ── HEADLINE: full-corpus NVML classifier, shaped held out ───────────────
    print("\n=== HEADLINE: paper NVML classifier vs shaped (full corpus, shaped held out) ===")
    nvml_cols = feature_subset(all_cols, "nvml")
    train = win[~shaped_mask]
    rows, _ = evasion_by_label(train, win[shaped_mask], nvml_cols)
    # baseline_train detection as the control
    base_rows, _ = evasion_by_label(
        win[~win["workload_label"].isin(SHAPED_LABELS + ["adversarial_llmc_baseline_train"])],
        win[win["workload_label"] == "adversarial_llmc_baseline_train"], nvml_cols)
    ctl = base_rows.get("adversarial_llmc_baseline_train", {})
    print(f"  CONTROL unshaped baseline_train: detection={ctl.get('detection', float('nan')):.3f} "
          f"(should be high)")
    for lbl in SHAPED_LABELS:
        if lbl in rows:
            r = rows[lbl]
            print(f"  {lbl:34s} n={r['n_windows']:4d}  detection={r['detection']:.3f}  "
                  f"EVASION={r['evasion']:.3f}")

    # ── ABLATION: node-local balanced set, sensor tiers ──────────────────────
    print("\n=== ABLATION: node-local set (reference + baseline), sensor tiers, shaped held out ===")
    # Node-local = ONLY our on-node H200 runs, identified by run_id membership in
    # the DCGM sidecar files we collected (shaping + reference + baseline). These
    # are the only runs carrying all three sensors on one GPU. Selecting by run_id
    # (not "has any dcgm column") excludes original-corpus runs that happen to
    # carry dcgm_* columns from the earlier datacenter collection but have no
    # scope and are different GPUs. Same runs/windows across tiers — only the
    # feature columns change, so any drop in evasion is attributable to the sensor.
    onnode_runs = set(_dcgm_map().keys())
    node = win[win["run_id"].astype(str).isin(onnode_runs)]
    node_train = node[~node["workload_label"].isin(SHAPED_LABELS)]
    node_test = node[node["workload_label"].isin(SHAPED_LABELS)]
    print(f"  node-local windows: train={len(node_train)} (pos={int((node_train['threeway_label']=='ml_training').sum())}), "
          f"test(shaped)={len(node_test)}")
    print(f"  {'cadence':34s} {'NVML':>8} {'+DCGM':>8} {'+scope':>8} {'all':>8}  (evasion rate)")
    for lbl in SHAPED_LABELS:
        line = f"  {lbl:34s}"
        for tier in ("nvml", "nvml+dcgm", "nvml+scope", "all"):
            cols = feature_subset(all_cols, tier)
            rows, _ = evasion_by_label(node_train, node_test[node_test["workload_label"] == lbl], cols)
            ev = rows.get(lbl, {}).get("evasion", float("nan"))
            line += f" {ev:8.3f}"
        print(line, flush=True)


if __name__ == "__main__":
    main()
