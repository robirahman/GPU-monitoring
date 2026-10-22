"""MI300X follow-up experiments:

  (1) Per-device-normalized cross-vendor transfer (NVIDIA -> MI300X).
      Each feature is z-scored within its own GPU's distribution (computed
      over that device's windows, no labels used) before train/test. Tests
      whether the zero-shot transfer collapse is just absolute-value domain
      shift that calibration can remove.

  (2) Tier-2 (rocprofiler) ablation, within-MI300X grouped CV.
      Compares Tier-1 (NVML-equivalent signals) vs Tier-1+Tier-2 vs
      Tier-2-only for binary training detection, plus RF feature importance.
      Tier-2 = rdc_tensor_active / sm_occupancy / fp32_pipe_active /
      active_waves / elapsed_cycles — counters NVML cannot provide.
"""
from __future__ import annotations
import logging, warnings, sys
from pathlib import Path
import numpy as np, pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__); warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent.parent; sys.path.insert(0, str(ROOT))

from classifier.threeway_improved import load_all_data, sliding_windows, get_feature_cols, RAW_SIGNALS
from classifier.twostage import TwoStageClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GroupKFold
from sklearn.metrics import f1_score

TIER2 = ["rdc_tensor_active", "rdc_sm_occupancy", "rdc_fp32_pipe_active",
         "rdc_active_waves", "rdc_elapsed_cycles"]


def load_harmonized():
    raw = load_all_data()
    for c in RAW_SIGNALS + ["temperature_hotspot_c", "encoder_util_pct", "decoder_util_pct"] + TIER2:
        if c in raw.columns:
            raw[c] = pd.to_numeric(raw[c], errors="coerce")
    name = raw["gpu_name"].astype(str)
    mi = name.str.contains("MI300X", case=False, na=False)
    nv = name.str.contains("nvidia|geforce|tesla|h100|h200|a100|rtx|b200", case=False, na=False)
    sub = mi & (~(raw["temperature_c"] > 0))
    raw.loc[sub, "temperature_c"] = raw.loc[sub, "temperature_hotspot_c"]
    for c in ("temperature_c", "pcie_tx_mbps", "pcie_rx_mbps"):
        raw.loc[mi & (raw[c] < 0), c] = np.nan
    for c in ("pcie_tx_mbps", "pcie_rx_mbps"):
        raw[c] = raw[c].fillna(0.0)
    raw["vendor"] = np.where(mi, "mi300x", np.where(nv, "nvidia", "other"))
    return raw


# Share of the target device's runs reserved for unsupervised calibration and
# kept out of scoring.
CALIBRATION_SHARE = 0.3


def experiment1(raw):
    log.info("\n" + "="*64 + "\n(1) PER-DEVICE-NORMALIZED TRANSFER  NVIDIA -> MI300X\n" + "="*64)
    df = raw[raw["vendor"].isin(["nvidia", "mi300x"])].copy()
    w = sliding_windows(df, 30.0, 15.0)
    run_gpu = df.groupby("run_id")["gpu_name"].first()
    run_ven = df.groupby("run_id")["vendor"].first()
    w["gpu_name"] = w["run_id"].map(run_gpu); w["vendor"] = w["run_id"].map(run_ven)
    feats = get_feature_cols(w)
    X = w[feats].astype(float).fillna(0.0)

    # Per-device statistics are unsupervised, but they must not come from the
    # windows being scored: taken over the whole frame they are fitted on the
    # MI300X evaluation rows themselves, and on every window of a run including
    # the ones that come after. Hold out a calibration slice of the target
    # device's runs instead, which is what a deployer actually has.
    mi_runs = np.sort(w.loc[w["vendor"] == "mi300x", "run_id"].unique())
    n_cal = max(1, int(round(CALIBRATION_SHARE * len(mi_runs))))
    cal_runs = set(np.random.default_rng(0).permutation(mi_runs)[:n_cal])
    calibration = (w["vendor"] == "mi300x") & w["run_id"].isin(cal_runs)
    stats_on = (w["vendor"] == "nvidia") | calibration
    log.info(f"  calibration: {n_cal} of {len(mi_runs)} MI300X runs held out of scoring")

    def run(Xmat, tag):
        tr = w["vendor"] == "nvidia"
        te = (w["vendor"] == "mi300x") & ~calibration
        clf = TwoStageClassifier(backend="random_forest")
        clf.fit(np.nan_to_num(Xmat[tr.values]), w.loc[tr, "threeway_label"].values,
                groups=w.loc[tr, "run_id"].values, feature_cols=feats)
        p = np.array(clf.classes_)[clf.predict_proba(np.nan_to_num(Xmat[te.values])).argmax(1)]
        yt = w.loc[te, "threeway_label"].values
        three = (p == yt).mean(); binary = ((p == "ml_training") == (yt == "ml_training")).mean()
        log.info(f"  {tag:22} three-way={three:.4f}  binary-is-training={binary:.4f}")

    run(X.values, "raw (baseline)")
    # Per-device z-score: each feature standardized within its own gpu_name.
    Xn = X.copy()
    for col in feats:
        seen = X[col].where(stats_on)
        gm = seen.groupby(w["gpu_name"]).transform("mean")
        gs = seen.groupby(w["gpu_name"]).transform("std").replace(0, 1.0)
        Xn[col] = (X[col] - gm) / gs
    run(Xn.values, "per-device z-scored")

    # Idle-baseline calibration: center each device at its own idle floor and
    # scale by its device std. Preserves "how far above idle", which is the
    # cross-device-comparable signal (unlike unconditional z-scoring, which
    # also removes it). Falls back to device mean for devices lacking idle.
    idle = (w["workload_label"].astype(str).str.contains("idle", case=False, na=False)
            & stats_on)
    Xb = X.copy()
    for col in feats:
        seen = X[col].where(stats_on)
        idle_mean = X.loc[idle, col].groupby(w.loc[idle, "gpu_name"]).mean()
        dev_mean = seen.groupby(w["gpu_name"]).mean()
        dev_std = seen.groupby(w["gpu_name"]).std().replace(0, 1.0)
        base = w["gpu_name"].map(idle_mean).fillna(w["gpu_name"].map(dev_mean))
        Xb[col] = (X[col] - base.values) / w["gpu_name"].map(dev_std).values
    n_idle_dev = w.loc[idle, "gpu_name"].nunique()
    log.info(f"  (idle baseline available for {n_idle_dev} GPU models)")
    run(Xb.values, "idle-baseline calibrated")


def _window_tier(df, cols, stats=("mean","std","min","max","p50")):
    """Window MI300X telemetry into per-window stats over the given columns."""
    rows = []
    for rid, rdf in df.groupby("run_id"):
        rdf = rdf.sort_values("ts") if "ts" in rdf else rdf
        n = len(rdf); win = 30; stride = 15
        # approximate 1Hz: use row windows
        i = 0
        while i < n:
            chunk = rdf.iloc[i:i+win]
            if len(chunk) >= 5:
                feat = {"run_id": rid, "label": rdf["threeway_label"].iloc[0],
                        "workload": rdf["workload_label"].iloc[0]}
                for c in cols:
                    s = pd.to_numeric(chunk[c], errors="coerce").fillna(0).values.astype(float)
                    feat[f"{c}_mean"]=s.mean(); feat[f"{c}_std"]=s.std()
                    feat[f"{c}_min"]=s.min(); feat[f"{c}_max"]=s.max(); feat[f"{c}_p50"]=np.median(s)
                rows.append(feat)
            i += stride
    return pd.DataFrame(rows)


def experiment2(raw):
    log.info("\n" + "="*64 + "\n(2) TIER-2 ABLATION  (within-MI300X, binary training detection)\n" + "="*64)
    mi = raw[raw["vendor"] == "mi300x"].copy()
    t1_cols = [c for c in RAW_SIGNALS if c in mi.columns]
    w = _window_tier(mi, t1_cols + TIER2)
    log.info(f"  MI300X windows={len(w)} runs={w['run_id'].nunique()}")
    y = (w["label"] == "ml_training").astype(int).values
    g = w["run_id"].values
    t1f = [f"{c}_{s}" for c in t1_cols for s in ("mean","std","min","max","p50")]
    t2f = [f"{c}_{s}" for c in TIER2 for s in ("mean","std","min","max","p50")]

    def cv(cols, tag):
        X = w[cols].values
        gkf = GroupKFold(5); accs=[]; f1s=[]
        for tr, te in gkf.split(X, y, g):
            rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
            rf.fit(X[tr], y[tr]); p = rf.predict(X[te])
            accs.append((p == y[te]).mean()); f1s.append(f1_score(y[te], p))
        log.info(f"  {tag:16} acc={np.mean(accs):.4f}  F1(train)={np.mean(f1s):.4f}")
        return np.mean(accs)

    cv(t1f, "Tier-1 only"); cv(t2f, "Tier-2 only"); cv(t1f + t2f, "Tier-1 + Tier-2")

    # Feature importance on the combined model
    rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=0, n_jobs=-1)
    rf.fit(w[t1f + t2f].values, y)
    imp = pd.Series(rf.feature_importances_, index=t1f + t2f).sort_values(ascending=False)
    log.info("  Top-12 features (combined):\n" +
             "\n".join(f"    {'[T2]' if any(t in f for t in TIER2) else '[T1]'} {f:28} {v:.4f}"
                       for f, v in imp.head(12).items()))


def main():
    raw = load_harmonized()
    experiment1(raw)
    experiment2(raw)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
