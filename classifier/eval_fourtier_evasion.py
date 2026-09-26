"""Four-tier evasion analysis: does the BMC node-power tier catch the memory-
minimisation / frequent-update evaders that fool the NVML-only classifier?

Unlike ``eval_shaping_evasion`` (which grew the scope + DCGM tiers), every run
used here — the honest reference corpus AND the evaders — was captured on the
SAME H200 with all four sensors (NVML + DCGM + scope + BMC). So this is a clean
node-local ablation: identical runs and windows across tiers, only the feature
columns change, and the reference is the honest 4-sensor corpus in
``data/h200_nvl_scope/full_sensor`` (frozen locally; the live gap-fill campaign
writes only to the server copy).

Tiers compared, as evasion rate (= fraction of an evader's windows the training-
vs-rest classifier labels "rest") per evader family:

    NVML   |   +DCGM   |   +scope   |   +BMC   |   all

The headline question: mm_min_noshape and mm_freq run at high GPU utilisation
but with a tiny resident footprint (b=1 + grad-accum), so they read like
inference/other on NVML mem features. BMC sees whole-node draw, which a tiny
footprint does not lower — so BMC should still flag them as heavy compute.

Run from repo root:  python -m classifier.eval_fourtier_evasion
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

from classifier.threeway_improved import get_feature_cols, PROJECT_ROOT
from classifier.scope_features import SCOPE_FEATURE_NAMES
from classifier.eval_shaping_evasion import load_runs, windows_with_scope

DATA = PROJECT_ROOT / "data"
REF_DIR = "full_sensor"          # honest 4-sensor reference corpus (positives + negatives)
EVADER_DIR = "evaders_4tier"     # held-out attack set, all 4 sensors

# The evader families re-collected with all four tiers. Ground-truth label for
# every one is ml_training (they ARE training; threeway_label maps adversarial_*
# -> ml_training). An "evasion" is a training window the classifier calls rest.
EVADER_FAMILIES = [
    "adversarial_mm_min_noshape",     # tiny footprint, high util, NO cadence shaping
    "adversarial_mm_min_shaped",      # + inference-like idle-gap shaping
    "adversarial_mm_spoof_infer_shaped",  # + footprint padded to mimic an infer server
    "adversarial_mm_freq",            # grad-accum 1: near-continuous weight-write DRAM traffic
]

BMC_FEATURE_NAMES = [
    "bmc_power_mean", "bmc_power_std", "bmc_power_min", "bmc_power_max",
    "bmc_power_range", "bmc_power_p95",
]


# ── BMC tier: per-window node-power features ──────────────────────────────────
def build_bmc_path_map(data_dirs) -> dict:
    """Map run_id (12-hex from the filename) -> bmc parquet path."""
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]
    m = {}
    for d in data_dirs:
        for path in glob.glob(os.path.join(str(d), "**", "*_bmc_*.parquet"), recursive=True):
            after = os.path.basename(path).split("_bmc_", 1)[1]
            m[after.split("_")[0]] = path
    return m


def _bmc_window_features(bdf: pd.DataFrame, ws: float, we: float) -> dict:
    """DCMI node power (W) over [ws, we], sliced on the sidecar's epoch column."""
    seg = bdf[(bdf["epoch"] >= ws) & (bdf["epoch"] < we)]
    v = seg["bmc_power_w"].dropna().values.astype(float)
    if v.size == 0:
        # fall back to the nearest single reading so a window is never all-NaN
        # just because the ~0.2 Hz DCMI clock didn't tick inside a 30 s slot
        near = bdf.iloc[(bdf["epoch"] - (ws + we) / 2).abs().argsort()[:1]]
        v = near["bmc_power_w"].dropna().values.astype(float)
    if v.size == 0:
        return {k: np.nan for k in BMC_FEATURE_NAMES}
    return {
        "bmc_power_mean": float(v.mean()),
        "bmc_power_std": float(v.std()),
        "bmc_power_min": float(v.min()),
        "bmc_power_max": float(v.max()),
        "bmc_power_range": float(v.max() - v.min()),
        "bmc_power_p95": float(np.percentile(v, 95)),
    }


def attach_bmc_features(windows: pd.DataFrame, bmc_map: dict) -> pd.DataFrame:
    if windows.empty:
        for k in BMC_FEATURE_NAMES:
            windows[k] = np.nan
        return windows
    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, w in windows[["run_id", "window_start", "window_end"]].iterrows():
        path = bmc_map.get(str(w["run_id"]))
        if path is None:
            rows.append({k: np.nan for k in BMC_FEATURE_NAMES})
            continue
        if path not in cache:
            cache[path] = pd.read_parquet(path).sort_values("epoch")
        rows.append(_bmc_window_features(cache[path], w["window_start"], w["window_end"]))
    feat = pd.DataFrame(rows, index=windows.index)
    for k in BMC_FEATURE_NAMES:
        windows[k] = feat[k].values
    return windows


# ── feature tiers ─────────────────────────────────────────────────────────────
def feature_subset(all_cols, tier):
    dcgm = [c for c in all_cols if c.startswith("dcgm_")]
    scope = [c for c in all_cols if c in SCOPE_FEATURE_NAMES]
    bmc = [c for c in all_cols if c.startswith("bmc_")]
    base = [c for c in all_cols if c not in dcgm and c not in scope and c not in bmc]
    return {
        "nvml": base,
        "nvml+dcgm": base + dcgm,
        "nvml+scope": base + scope,
        "nvml+bmc": base + bmc,
        "all": base + dcgm + scope + bmc,
    }[tier]


def train_clf(train_df, feat_cols):
    Xtr = train_df[feat_cols].fillna(0).values.astype(np.float32)
    ytr = (train_df["threeway_label"] == "ml_training").astype(int).values
    clf = RandomForestClassifier(n_estimators=400, min_samples_leaf=2,
                                 max_features="sqrt", class_weight="balanced",
                                 random_state=42, n_jobs=-1)
    clf.fit(Xtr, ytr)
    return clf


def detection_rate(clf, sub, feat_cols):
    if len(sub) == 0:
        return float("nan"), 0
    X = sub[feat_cols].fillna(0).values.astype(np.float32)
    return float(clf.predict(X).mean()), len(sub)


def detection_proba(clf, sub, feat_cols):
    """Mean P(training) — reveals sub-threshold movement the 0/1 rate hides."""
    if len(sub) == 0:
        return float("nan")
    X = sub[feat_cols].fillna(0).values.astype(np.float32)
    return float(clf.predict_proba(X)[:, 1].mean())


TIERS = ["nvml", "nvml+dcgm", "nvml+scope", "nvml+bmc", "all"]


def main():
    pd.set_option("display.width", 160)
    print("Loading full_sensor reference + evaders_4tier (NVML+DCGM merge)…", flush=True)
    df = load_runs(only_dirs=[REF_DIR, EVADER_DIR], attach_dcgm=True)
    win = windows_with_scope(df)
    win = attach_bmc_features(win, build_bmc_path_map([str(DATA)]))
    all_cols = get_feature_cols(win)
    print(f"windows={len(win)}  feature_cols={len(all_cols)} "
          f"(dcgm={sum(c.startswith('dcgm_') for c in all_cols)}, "
          f"scope={sum(c in SCOPE_FEATURE_NAMES for c in all_cols)}, "
          f"bmc={sum(c.startswith('bmc_') for c in all_cols)})", flush=True)

    ev_mask = win["workload_label"].isin(EVADER_FAMILIES)
    ref = win[~ev_mask]     # honest 4-sensor corpus (positives + negatives), evaders held out
    ev = win[ev_mask]
    pos = int((ref["threeway_label"] == "ml_training").sum())
    print(f"reference windows={len(ref)} (train-positive={pos}, negative={len(ref)-pos}); "
          f"evader windows={len(ev)}", flush=True)
    present = [f for f in EVADER_FAMILIES if f in set(ev["workload_label"])]
    for f in EVADER_FAMILIES:
        if f not in present:
            print(f"  WARNING: evader family not found in data: {f}", flush=True)

    # sanity controls from the honest corpus (report per-tier that the classifier
    # still catches real training and still rejects real inference)
    train_ctl = ref[ref["threeway_label"] == "ml_training"]["workload_label"].value_counts()
    infer_ctl = ref[ref["threeway_label"] == "ml_inference"]["workload_label"].value_counts()

    print("\n=== EVASION RATE by tier (1 - detection; higher = better hidden) ===")
    hdr = f"  {'evader family':36s}" + "".join(f"{t:>12}" for t in TIERS)
    print(hdr)
    clfs = {t: train_clf(ref, feature_subset(all_cols, t)) for t in TIERS}
    for fam in present:
        sub = ev[ev["workload_label"] == fam]
        line = f"  {fam:36s}"
        for t in TIERS:
            cols = feature_subset(all_cols, t)
            det, n = detection_rate(clfs[t], sub, cols)
            line += f"{1.0 - det:12.3f}"
        print(line + f"   (n={len(sub)})", flush=True)

    print("\n=== mean P(training) by tier (hard rate saturates; this shows drift) ===")
    print(hdr)
    for fam in present:
        sub = ev[ev["workload_label"] == fam]
        line = f"  {fam:36s}"
        for t in TIERS:
            p = detection_proba(clfs[t], sub, feature_subset(all_cols, t))
            line += f"{p:12.3f}"
        print(line + "   (want HIGH: these ARE training)", flush=True)

    # detection (not evasion) for the controls: should stay HIGH for training,
    # LOW for inference, in every tier — otherwise a tier is just breaking the
    # classifier rather than catching evaders.
    print("\n=== CONTROLS: detection rate (fraction called training) ===")
    print(hdr)
    if len(train_ctl):
        c_lbl = train_ctl.index[0]
        sub = ref[ref["workload_label"] == c_lbl]
        line = f"  {('train:'+c_lbl)[:36]:36s}"
        for t in clfs:  # note: in-sample; sanity only
            det, _ = detection_rate(clfs[t], sub, feature_subset(all_cols, t))
            line += f"{det:12.3f}"
        print(line + "  (want HIGH)")
    if len(infer_ctl):
        c_lbl = infer_ctl.index[0]
        sub = ref[ref["workload_label"] == c_lbl]
        line = f"  {('infer:'+c_lbl)[:36]:36s}"
        for t in clfs:
            det, _ = detection_rate(clfs[t], sub, feature_subset(all_cols, t))
            line += f"{det:12.3f}"
        print(line + "  (want LOW)")

    # descriptive: what BMC actually sees per family vs the honest classes
    print("\n=== BMC node power (W) by class, window means ===")
    for grp, lbls in (("EVADERS", present),
                      ("ref training", list(train_ctl.index[:3])),
                      ("ref inference", list(infer_ctl.index[:3]))):
        for lbl in lbls:
            s = win[win["workload_label"] == lbl]["bmc_power_mean"].dropna()
            if len(s):
                print(f"  [{grp:12s}] {lbl:36s} bmc={s.mean():6.0f} W  "
                      f"(min {s.min():.0f}, max {s.max():.0f}, n={len(s)})")


if __name__ == "__main__":
    main()
