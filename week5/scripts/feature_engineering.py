"""
week5/feature_engineering.py
================================
Sliding-window feature extraction from raw GPU telemetry.

Sources loaded:
  1. week4/data/*.parquet          — 26 workload parquets (9 raw channels)
  2. week4/results/ddp/*.parquet   — DDP training traces
  3. week4/results/dataset_scale/*/telemetry.parquet  — dataset-scale traces
  4. week4/results/edge_cases/*_telemetry.json        — edge-case traces (custom schema)

Output:
  week5/results/windows_30s.parquet   — 30s sliding-window feature matrix
  week5/results/windows_60s.parquet   — 60s sliding-window feature matrix
  week5/results/windows_120s.parquet  — 120s sliding-window feature matrix
  week5/results/feature_meta.json     — feature names and descriptions

Feature count: ~122 per window
  9 signals × 13 stats  = 117   (mean, std, min, max, p25, p50, p75, p95, iqr, range, cv, skew, kurt)
  3 cross-signal ratios  =   3   (power_per_util, pcie_total, util_per_sm_clock_pct)
  2 autocorrelation      =   2   (acf_lag1 for util and power)
  ─────────────────────────────
  Total                  = 122
"""

import sys, os, json, glob, logging
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

WEEK4   = Path(__file__).parent.parent.parent / "week4"
WEEK5   = Path(__file__).parent.parent          # week5/
RESULTS = WEEK5 / "results"
RESULTS.mkdir(exist_ok=True)

# ── Signal columns present in week4/data parquets ─────────────────────────────
RAW_SIGNALS = [
    "gpu_utilization_pct",
    "mem_utilization_pct",
    "mem_used_mb",
    "power_draw_w",
    "temperature_c",
    "sm_clock_mhz",
    "mem_clock_mhz",
    "pcie_tx_mbps",
    "pcie_rx_mbps",
]

STAT_NAMES = ["mean", "std", "min", "max", "p25", "p50", "p75", "p95",
              "iqr", "range", "cv", "skew", "kurt"]

# ── Binary and multi-class label maps ─────────────────────────────────────────
TRAINING_LABELS = {
    "bert_sst2", "gpt2_wikitext2",
    "pytorch_mlp_cifar10", "pytorch_resnet_cifar10", "pytorch_resnet_cifar10_amp",
    "training_dual_gpu_dp", "training_single_gpu",
    "BASELINE_TRAIN", "EC2", "EC3", "EC5", "EC6",   # edge-case training variants
}

# EC-1 injects NVLink but is inference; EC-4 is heavy inference
INFERENCE_LABELS = {
    "resnet50_inference", "BASELINE_INFER", "EC1", "EC4",
}

def binary_label(label: str) -> str:
    """training | inference | other"""
    if label in TRAINING_LABELS:  return "training"
    if label in INFERENCE_LABELS: return "inference"
    # Heuristic from workload name
    lc = label.lower()
    if any(k in lc for k in ("train", "bert", "gpt", "resnet_cifar", "mlp_cifar")):
        return "training"
    if any(k in lc for k in ("infer", "inference")):
        return "inference"
    return "other"


# ── Loaders ───────────────────────────────────────────────────────────────────

def load_week4_data_parquets() -> pd.DataFrame:
    """Load week4/data/*.parquet — standard schema with workload_label."""
    paths = glob.glob(str(WEEK4 / "data" / "*.parquet"))
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            # Normalise timestamp to epoch float
            if "timestamp_epoch" in df.columns:
                df = df.rename(columns={"timestamp_epoch": "ts"})
            elif "timestamp_utc" in df.columns:
                df["ts"] = pd.to_datetime(df["timestamp_utc"]).astype("int64") / 1e9
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"week4/data: {len(frames)} parquets → {len(out):,} rows")
    return out


def load_ddp_parquets() -> pd.DataFrame:
    """Load week4/results/ddp/*.parquet — DDP training traces."""
    paths = glob.glob(str(WEEK4 / "results" / "ddp" / "ddp_telemetry_gpu*.parquet"))
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            if "timestamp_epoch" in df.columns:
                df = df.rename(columns={"timestamp_epoch": "ts"})
            if "workload_label" not in df.columns:
                df["workload_label"] = "training_dual_gpu_dp"
            if "run_id" not in df.columns:
                df["run_id"] = Path(p).stem
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"ddp parquets: {len(frames)} → {len(out):,} rows")
    return out


def load_dataset_scale_parquets() -> pd.DataFrame:
    """Load dataset_scale telemetry parquets.
    Schema: timestamp, gpu, gpu_util, mem_util, power_w, mem_used_mb, sm_clock_mhz, mem_clock_mhz, temp_c
    """
    paths = glob.glob(str(WEEK4 / "results" / "dataset_scale" / "*" / "telemetry.parquet"))
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            n = Path(p).parent.name  # e.g. "n65536"
            # Rename dataset_scale-specific columns to standard schema
            rename_map = {
                "timestamp":    "ts",
                "gpu_util":     "gpu_utilization_pct",
                "mem_util":     "mem_utilization_pct",
                "power_w":      "power_draw_w",
                "temp_c":       "temperature_c",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            if "workload_label" not in df.columns:
                df["workload_label"] = "training_dual_gpu_dp"
            if "run_id" not in df.columns:
                df["run_id"] = f"dscale_{n}"
            if "ts" not in df.columns and "timestamp_epoch" in df.columns:
                df = df.rename(columns={"timestamp_epoch": "ts"})
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"dataset_scale: {len(frames)} parquets → {len(out):,} rows")
    return out


def load_edge_case_jsons() -> pd.DataFrame:
    """Load edge case telemetry JSON files.
    Schema: list of {t, gpu, gpu_util, power_w, mem_used_gb, sm_mhz}
    """
    paths = glob.glob(str(WEEK4 / "results" / "edge_cases" / "*_telemetry.json"))
    frames = []
    for p in paths:
        case_name = Path(p).stem.replace("_telemetry", "")
        try:
            with open(p) as f:
                records = json.load(f)
            if not records:
                continue
            df = pd.DataFrame(records)
            # Rename to standard schema
            rename = {
                "t": "ts",
                "gpu_util": "gpu_utilization_pct",
                "power_w": "power_draw_w",
                "sm_mhz": "sm_clock_mhz",
            }
            df = df.rename(columns=rename)
            # mem_used_gb → mem_used_mb
            if "mem_used_gb" in df.columns:
                df["mem_used_mb"] = df["mem_used_gb"] * 1024
            # Fill missing signals with 0
            for sig in RAW_SIGNALS:
                if sig not in df.columns:
                    df[sig] = 0.0
            df["workload_label"] = case_name
            df["run_id"] = case_name
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"edge_cases: {len(frames)} JSON files → {len(out):,} rows")
    return out


def load_all() -> pd.DataFrame:
    """Merge all telemetry sources into a single DataFrame with standardised columns."""
    frames = []
    for loader in [load_week4_data_parquets, load_ddp_parquets,
                   load_dataset_scale_parquets, load_edge_case_jsons]:
        df = loader()
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No telemetry data found")

    combined = pd.concat(frames, ignore_index=True)

    # Ensure required columns exist
    for sig in RAW_SIGNALS:
        if sig not in combined.columns:
            combined[sig] = 0.0

    # Sort by run then time
    if "ts" in combined.columns:
        combined = combined.sort_values(["run_id", "ts"]).reset_index(drop=True)

    # Add labels
    combined["binary_label"]    = combined["workload_label"].apply(binary_label)
    combined["is_training"]     = (combined["binary_label"] == "training").astype(int)

    log.info(f"Total rows: {len(combined):,} | runs: {combined['run_id'].nunique()} | "
             f"labels: {combined['workload_label'].nunique()}")
    log.info(f"Binary distribution:\n{combined['binary_label'].value_counts().to_string()}")
    return combined


# ── Window feature extraction ─────────────────────────────────────────────────

def acf_lag1(x: np.ndarray) -> float:
    """Autocorrelation at lag-1."""
    if len(x) < 3:
        return 0.0
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    if var < 1e-12:
        return 0.0
    return float(((x[:-1] - mu) * (x[1:] - mu)).mean() / var)


def extract_window_features(chunk: pd.DataFrame) -> dict:
    """Extract ~122 features from a single time window."""
    feats = {}

    for sig in RAW_SIGNALS:
        if sig not in chunk.columns:
            vals = np.zeros(max(len(chunk), 1))
        else:
            vals = chunk[sig].fillna(0).values.astype(float)

        n = len(vals)
        mu   = vals.mean()
        sd   = vals.std()
        vmin = vals.min()
        vmax = vals.max()
        p25, p50, p75, p95 = np.percentile(vals, [25, 50, 75, 95]) if n > 1 else (mu,)*4

        feats[f"{sig}_mean"]  = mu
        feats[f"{sig}_std"]   = sd
        feats[f"{sig}_min"]   = vmin
        feats[f"{sig}_max"]   = vmax
        feats[f"{sig}_p25"]   = p25
        feats[f"{sig}_p50"]   = p50
        feats[f"{sig}_p75"]   = p75
        feats[f"{sig}_p95"]   = p95
        feats[f"{sig}_iqr"]   = p75 - p25
        feats[f"{sig}_range"] = vmax - vmin
        feats[f"{sig}_cv"]    = sd / (abs(mu) + 1e-9)
        feats[f"{sig}_skew"]  = float(sp_stats.skew(vals))   if n > 2 else 0.0
        feats[f"{sig}_kurt"]  = float(sp_stats.kurtosis(vals)) if n > 3 else 0.0

    # Cross-signal derived features
    util  = chunk["gpu_utilization_pct"].fillna(0).values.astype(float)
    power = chunk["power_draw_w"].fillna(0).values.astype(float)
    pcie_tx = chunk["pcie_tx_mbps"].fillna(0).values.astype(float)
    pcie_rx = chunk["pcie_rx_mbps"].fillna(0).values.astype(float)
    sm_clk  = chunk["sm_clock_mhz"].fillna(0).values.astype(float)

    feats["power_per_util"]       = power.mean() / (util.mean() + 1.0)
    feats["pcie_total_mean"]      = (pcie_tx + pcie_rx).mean()
    feats["util_per_sm_pct"]      = util.mean() / (sm_clk.mean() + 1.0) * 1000.0

    # Autocorrelation (lag-1) for two key signals
    feats["acf1_gpu_util"]  = acf_lag1(util)
    feats["acf1_power"]     = acf_lag1(power)

    return feats


def sliding_windows(df: pd.DataFrame, window_sec: float, stride_sec: float) -> pd.DataFrame:
    """Apply sliding windows per run_id, return feature DataFrame.

    For runs shorter than window_sec, the full run is treated as one window
    (min 5 samples required).  This preserves short edge-case and dataset-scale
    traces that would otherwise produce zero windows.
    """
    results = []
    for run_id, run_df in df.groupby("run_id"):
        run_df = run_df.sort_values("ts").reset_index(drop=True)
        t_start = run_df["ts"].iloc[0]
        t_end   = run_df["ts"].iloc[-1]
        duration = t_end - t_start
        label   = run_df["workload_label"].iloc[0]
        binary  = run_df["binary_label"].iloc[0]
        is_tr   = run_df["is_training"].iloc[0]

        def emit(chunk):
            if len(chunk) < 5:
                return
            feats = extract_window_features(chunk)
            feats["run_id"]         = run_id
            feats["window_start"]   = chunk["ts"].iloc[0]
            feats["window_sec"]     = window_sec
            feats["n_samples"]      = len(chunk)
            feats["workload_label"] = label
            feats["binary_label"]   = binary
            feats["is_training"]    = is_tr
            results.append(feats)

        if duration < window_sec:
            # Short run: use the entire run as one window
            emit(run_df)
        else:
            wstart = t_start
            while wstart + window_sec <= t_end + 1:
                chunk = run_df[(run_df["ts"] >= wstart) & (run_df["ts"] < wstart + window_sec)]
                emit(chunk)
                wstart += stride_sec

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("=== Week 5 Feature Engineering ===")
    df_raw = load_all()

    feature_cols = (
        [f"{s}_{st}" for s in RAW_SIGNALS for st in STAT_NAMES]
        + ["power_per_util", "pcie_total_mean", "util_per_sm_pct",
           "acf1_gpu_util", "acf1_power"]
    )
    log.info(f"Feature count per window: {len(feature_cols)}")

    for win_sec, stride_sec in [(5, 2), (15, 7), (30, 15)]:
        log.info(f"Extracting windows: {win_sec}s window, {stride_sec}s stride ...")
        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
        if df_win.empty:
            log.warning(f"  No windows generated for {win_sec}s")
            continue
        out_path = RESULTS / f"windows_{win_sec}s.parquet"
        df_win.to_parquet(out_path, index=False)
        n_train  = (df_win["binary_label"] == "training").sum()
        n_other  = (df_win["binary_label"] != "training").sum()
        log.info(f"  {len(df_win):,} windows → {out_path.name} "
                 f"(training={n_train}, non-training={n_other})")

    # Save feature metadata
    meta = {
        "feature_cols":    feature_cols,
        "n_features":      len(feature_cols),
        "signals":         RAW_SIGNALS,
        "stats_per_signal": STAT_NAMES,
        "derived":         ["power_per_util", "pcie_total_mean", "util_per_sm_pct",
                            "acf1_gpu_util", "acf1_power"],
    }
    with open(RESULTS / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Done. Feature meta saved.")


if __name__ == "__main__":
    main()
