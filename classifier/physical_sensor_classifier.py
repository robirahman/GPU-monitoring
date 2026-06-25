#!/usr/bin/env python3
"""Binary training-vs-other classifier from PHYSICAL sensors only.

Counterpart to retrain_binary.py (which uses NVML telemetry). This uses ONLY the
testbed's physical instruments -- the UltraMic 200K acoustic mic and the
PicoScope current clamps on the GPU/MOBO power rails -- and deliberately EXCLUDES
the snapshot's NVML-derived power fields, so it answers: can training be detected
from physical measurements alone, with no GPU software telemetry?

Reads sensor_logs/*.parquet (produced by run_workload_with_sensors.py), windows
by run_id, extracts per-channel statistical + temporal features, and runs
RandomForest / XGBoost with StratifiedGroupKFold (grouped by run_id), mirroring
the NVML pipeline's protocol.

Usage:
    python classifier/physical_sensor_classifier.py
    python classifier/physical_sensor_classifier.py --sensor-dir sensor_logs --gpu GPU0
"""
import argparse
import glob
import logging
import os
import sys
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from classifier.threeway_improved import threeway_label  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# Physical channels only (NO NVML power). We watch EVERY GPU current clamp plus
# node-level aggregates, so training on any GPU (or several) is visible -- a
# classifier that only watched GPU0 was trivially evaded by training on GPU1/2.
GPU_CLAMPS = ["GPU0", "GPU1", "GPU2"]
TEMPORAL = ["anyGPU_rms_mv", "mic_rms"]   # signals used for acf/fft temporal features


def physical_channels(active_gpu=None):
    """All physical channels: every GPU clamp + node aggregates + MOBO + mic.
    `active_gpu` is accepted for backward-compat but ignored (we watch all GPUs)."""
    ch = []
    for g in GPU_CLAMPS:
        ch += [f"{g}_rms_mv", f"{g}_peak_mv"]
    ch += ["anyGPU_rms_mv", "anyGPU_peak_mv", "sumGPU_rms_mv", "nGPU_active",
           "MOBO_rms_mv", "mic_rms", "mic_dbfs", "mic_peak", "mic_dominant_hz"]
    return ch


def augment_aggregates(df, active_thresh_mv=8.0):
    """Add node-level aggregate clamp signals (permutation-invariant to which GPU):
    anyGPU = max over clamps, sumGPU = total, nGPU_active = count of busy clamps."""
    rms = [f"{g}_rms_mv" for g in GPU_CLAMPS if f"{g}_rms_mv" in df.columns]
    pk = [f"{g}_peak_mv" for g in GPU_CLAMPS if f"{g}_peak_mv" in df.columns]
    if rms:
        df["anyGPU_rms_mv"] = df[rms].max(axis=1)
        df["sumGPU_rms_mv"] = df[rms].sum(axis=1)
        df["nGPU_active"] = (df[rms] > active_thresh_mv).sum(axis=1)
    if pk:
        df["anyGPU_peak_mv"] = df[pk].max(axis=1)
    return df


def _acf(x, lag):
    x = np.asarray(x, float)
    if len(x) <= lag or np.std(x) < 1e-9:
        return 0.0
    x = x - x.mean()
    d = np.dot(x, x)
    return float(np.dot(x[:-lag], x[lag:]) / d) if d > 1e-12 else 0.0


def _fft_peak(x):
    x = np.asarray(x, float)
    if len(x) < 4 or np.std(x) < 1e-9:
        return 0.0, 0.0
    x = x - x.mean()
    p = np.abs(np.fft.rfft(x)) ** 2
    if len(p) < 2:
        return 0.0, 0.0
    k = 1 + int(np.argmax(p[1:]))
    period = len(x) / k if k > 0 else 0.0
    return float(period), float(p[k] / (p[1:].sum() + 1e-12))


def _stats(x, name):
    x = np.asarray(x, float)
    x = x[~np.isnan(x)]
    if len(x) == 0:
        return {f"{name}_{s}": 0.0 for s in
                ["mean", "std", "min", "max", "p25", "p50", "p75", "p95", "iqr", "range", "cv", "skew", "kurt"]}
    mean = x.mean(); std = x.std()
    p25, p50, p75, p95 = np.percentile(x, [25, 50, 75, 95])
    from scipy.stats import skew, kurtosis
    return {
        f"{name}_mean": mean, f"{name}_std": std, f"{name}_min": x.min(), f"{name}_max": x.max(),
        f"{name}_p25": p25, f"{name}_p50": p50, f"{name}_p75": p75, f"{name}_p95": p95,
        f"{name}_iqr": p75 - p25, f"{name}_range": x.max() - x.min(),
        f"{name}_cv": std / mean if abs(mean) > 1e-9 else 0.0,
        f"{name}_skew": float(skew(x)) if std > 1e-9 else 0.0,
        f"{name}_kurt": float(kurtosis(x)) if std > 1e-9 else 0.0,
    }


def load_sensor_data(sensor_dir):
    frames = []
    for p in sorted(glob.glob(os.path.join(sensor_dir, "*_sensors.parquet"))):
        try:
            df = pd.read_parquet(p)
            if "workload_label" not in df.columns:
                df["workload_label"] = Path(p).stem.split("_NVIDIA_")[0]
            if "run_id" not in df.columns:
                df["run_id"] = Path(p).stem
            frames.append(df)
        except Exception as e:
            log.debug(f"skip {p}: {e}")
    if not frames:
        raise RuntimeError(f"No *_sensors.parquet in {sensor_dir}")
    df = pd.concat(frames, ignore_index=True)
    df = augment_aggregates(df)
    df["threeway_label"] = df["workload_label"].apply(threeway_label)
    return df


def windows(df, channels, temporal_channels, window_sec, stride_sec):
    rows = []
    for run_id, run in df.groupby("run_id"):
        run = run.sort_values("timestamp_epoch")
        t = run["timestamp_epoch"].values
        t0, t1 = t[0], t[-1]
        label = run["threeway_label"].iloc[0]
        wl = run["workload_label"].iloc[0]
        start = t0
        while start + window_sec <= t1 + 1e-6:
            m = (t >= start) & (t < start + window_sec)
            if m.sum() >= 3:
                chunk = run[m]
                feats = {}
                for ch in channels:
                    if ch in chunk.columns:
                        feats.update(_stats(chunk[ch].values, ch))
                for ch in temporal_channels:
                    if ch in chunk.columns:
                        x = chunk[ch].values
                        feats[f"{ch}_acf1"] = _acf(x, 1)
                        feats[f"{ch}_acf5"] = _acf(x, 5)
                        per, pk = _fft_peak(x)
                        feats[f"{ch}_fft_period"] = per
                        feats[f"{ch}_fft_peak"] = pk
                feats["run_id"] = run_id
                feats["workload_label"] = wl
                feats["threeway_label"] = label
                rows.append(feats)
            start += stride_sec
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sensor-dir", default=str(PROJECT_ROOT / "sensor_logs"))
    ap.add_argument("--out", default=str(PROJECT_ROOT / "classifier" / "results" / "physical_binary_results.csv"))
    args = ap.parse_args()

    channels = physical_channels()       # all GPU clamps + node aggregates + MOBO + mic
    temporal = TEMPORAL

    df = load_sensor_data(args.sensor_dir)
    log.info("Loaded %d sensor rows across %d runs", len(df), df["run_id"].nunique())
    dist = df.groupby("threeway_label")["run_id"].nunique()
    log.info("Runs per class:\n%s", dist.to_string())

    results = []
    for win, stride in [(15, 7), (30, 15), (60, 30)]:
        dw = windows(df, channels, temporal, win, stride)
        if dw.empty or dw["run_id"].nunique() < 4:
            log.warning("window %ss: too few windows/runs (%d), skipping", win, len(dw))
            continue
        feat_cols = [c for c in dw.columns if c not in ("run_id", "workload_label", "threeway_label")]
        X = dw[feat_cols].fillna(0).values.astype(np.float32)
        y = (dw["threeway_label"].values == "ml_training").astype(int)
        groups = dw["run_id"].values
        run_bin = (dw.groupby("run_id")["threeway_label"].first() == "ml_training").astype(int)
        n_splits = min(5, int(run_bin.value_counts().min()))
        if n_splits < 2:
            log.warning("window %ss: only one class has runs, skipping", win)
            continue
        log.info("=== window %ss: %d windows, %d features, train=%d/other=%d, %d folds ===",
                 win, len(dw), len(feat_cols), int((y == 1).sum()), int((y == 0).sum()), n_splits)
        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)
        clf = RandomForestClassifier(n_estimators=400, max_depth=None, min_samples_leaf=2,
                                     max_features="sqrt", class_weight="balanced",
                                     random_state=42, n_jobs=-1)
        accs, f1s, yt, yp = [], [], [], []
        for tr, te in cv.split(X, y, groups):
            clf.fit(X[tr], y[tr])
            pred = clf.predict(X[te])
            accs.append(accuracy_score(y[te], pred))
            f1s.append(f1_score(y[te], pred, average="macro", zero_division=0))
            yt.extend(y[te]); yp.extend(pred)
        log.info("  RandomForest acc=%.4f+/-%.4f f1=%.4f", np.mean(accs), np.std(accs), np.mean(f1s))
        log.info("\n%s", classification_report(yt, yp, target_names=["non_training", "training"], zero_division=0))
        log.info("confusion:\n%s", pd.DataFrame(confusion_matrix(yt, yp, labels=[0, 1]),
                 index=["non_train", "train"], columns=["non_train", "train"]).to_string())
        results.append({"classifier": "RandomForest", "window_sec": win,
                        "acc_mean": np.mean(accs), "acc_std": np.std(accs), "f1_macro": np.mean(f1s),
                        "n_windows": len(dw), "n_runs": dw["run_id"].nunique(), "n_features": len(feat_cols)})

    if results:
        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(results).to_csv(args.out, index=False)
        log.info("Saved -> %s", args.out)
        for r in sorted(results, key=lambda x: -x["acc_mean"]):
            log.info("  %ss  acc=%.4f+/-%.4f f1=%.4f", r["window_sec"], r["acc_mean"], r["acc_std"], r["f1_macro"])
    else:
        log.warning("No results — need more collected sensor runs across both classes.")


if __name__ == "__main__":
    main()
