"""
week7/scripts/feature_engineering.py
====================================
Sliding-window feature extraction from Week 7 B200 GPU telemetry.

Sources loaded:
  1. week7/data/*.parquet                        — NVLink + single/dual-GPU telemetry
  2. week7/single_gpu_tests/data/*.parquet       — 9 single-GPU workload parquets
  3. week7/results/ddp/*.parquet                 — DDP training traces
  4. week7/results/dataset_scale/*/telemetry.parquet — dataset-scale traces
  5. week7/results/edge_cases/*_telemetry.json   — edge-case traces

Output:
  week7/results/windows_5s.parquet    — 5s sliding-window feature matrix
  week7/results/windows_15s.parquet   — 15s sliding-window feature matrix
  week7/results/windows_30s.parquet   — 30s sliding-window feature matrix
  week7/results/feature_meta.json     — feature names and descriptions

Feature count: 125 per window
  9 signals x 13 stats  = 117  (mean, std, min, max, p25, p50, p75, p95, iqr, range, cv, skew, kurt)
  5 cross-signal derived =   5  (power_per_util, pcie_total, util_per_sm_pct, power_pct_tdp, acf1_gpu_util)
  3 autocorrelation/misc =   3  (acf1_power, mem_clock_dynamic_range, sm_clock_dynamic_range)
  ─────────────────────────────
  Total                  = 125
"""

import sys, os, json, glob, logging, warnings
import numpy as np
import pandas as pd
from scipy import stats as sp_stats
from pathlib import Path

warnings.filterwarnings("ignore", category=RuntimeWarning)

# Import centralized label manifest
sys.path.insert(0, str(Path(__file__).parent))
from label_manifest import (
    get_category, is_training, get_tdp,
    WINDOW_CONFIGS, MIN_SAMPLES_PER_WINDOW,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

WEEK7   = Path(__file__).parent.parent
RESULTS = WEEK7 / "results"
RESULTS.mkdir(exist_ok=True)

# ── Signal columns ────────────────────────────────────────────────────────────
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


# ── Loaders ───────────────────────────────────────────────────────────────────

def _normalise_ts(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure a 'ts' column exists (epoch float)."""
    if "ts" in df.columns:
        return df
    if "timestamp_epoch" in df.columns:
        df = df.rename(columns={"timestamp_epoch": "ts"})
    elif "timestamp_utc" in df.columns:
        df["ts"] = pd.to_datetime(df["timestamp_utc"]).astype("int64") / 1e9
    elif "timestamp" in df.columns:
        # dataset_scale schema
        col = df["timestamp"]
        if pd.api.types.is_numeric_dtype(col):
            df = df.rename(columns={"timestamp": "ts"})
        else:
            df["ts"] = pd.to_datetime(col).astype("int64") / 1e9
    return df


def _ensure_run_id(df: pd.DataFrame, fallback: str) -> pd.DataFrame:
    if "run_id" not in df.columns:
        df["run_id"] = fallback
    return df


def _ensure_label(df: pd.DataFrame, fallback: str) -> pd.DataFrame:
    if "workload_label" not in df.columns:
        df["workload_label"] = fallback
    return df


def load_week7_data_parquets() -> pd.DataFrame:
    """Load week7/data/*.parquet — NVLink + training telemetry."""
    paths = glob.glob(str(WEEK7 / "data" / "*.parquet"))
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            df = _normalise_ts(df)
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"week7/data: {len(frames)} parquets -> {len(out):,} rows")
    return out


def load_single_gpu_parquets() -> pd.DataFrame:
    """Load week7/single_gpu_tests/data/*.parquet — 9 single-GPU workloads."""
    paths = glob.glob(str(WEEK7 / "single_gpu_tests" / "data" / "*.parquet"))
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            df = _normalise_ts(df)
            # run_id = stem without hash suffix
            stem = Path(p).stem
            parts = stem.rsplit("_", 1)
            label = parts[0] if len(parts) == 2 else stem
            df = _ensure_label(df, label)
            df = _ensure_run_id(df, stem)
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"single_gpu_tests: {len(frames)} parquets -> {len(out):,} rows")
    return out


def load_ddp_parquets() -> pd.DataFrame:
    """Load week7/results/ddp/*.parquet — DDP training traces."""
    paths = glob.glob(str(WEEK7 / "results" / "ddp" / "ddp_telemetry_gpu*.parquet"))
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            df = _normalise_ts(df)
            df = _ensure_label(df, "training_dual_gpu_dp")
            df = _ensure_run_id(df, Path(p).stem)
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"ddp parquets: {len(frames)} -> {len(out):,} rows")
    return out


def load_dataset_scale_parquets() -> pd.DataFrame:
    """Load dataset_scale telemetry parquets."""
    paths = glob.glob(str(WEEK7 / "results" / "dataset_scale" / "*" / "telemetry.parquet"))
    frames = []
    for p in paths:
        try:
            df = pd.read_parquet(p)
            n = Path(p).parent.name
            rename_map = {
                "timestamp": "ts", "gpu_util": "gpu_utilization_pct",
                "mem_util": "mem_utilization_pct", "power_w": "power_draw_w",
                "temp_c": "temperature_c",
            }
            df = df.rename(columns={k: v for k, v in rename_map.items() if k in df.columns})
            df = _normalise_ts(df)
            df = _ensure_label(df, "training_dual_gpu_dp")
            df = _ensure_run_id(df, f"dscale_{n}")
            frames.append(df)
        except Exception as e:
            log.warning(f"skip {p}: {e}")
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    log.info(f"dataset_scale: {len(frames)} parquets -> {len(out):,} rows")
    return out


def load_edge_case_jsons() -> pd.DataFrame:
    """Load edge case telemetry JSON files."""
    paths = glob.glob(str(WEEK7 / "results" / "edge_cases" / "*_telemetry.json"))
    frames = []
    for p in paths:
        case_name = Path(p).stem.replace("_telemetry", "")
        try:
            with open(p) as f:
                records = json.load(f)
            if not records:
                continue
            df = pd.DataFrame(records)
            rename = {"t": "ts", "gpu_util": "gpu_utilization_pct",
                      "power_w": "power_draw_w", "sm_mhz": "sm_clock_mhz"}
            df = df.rename(columns=rename)
            if "mem_used_gb" in df.columns:
                df["mem_used_mb"] = df["mem_used_gb"] * 1024
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
    log.info(f"edge_cases: {len(frames)} JSON files -> {len(out):,} rows")
    return out


def load_all() -> pd.DataFrame:
    """Merge all Week 7 telemetry sources."""
    frames = []
    for loader in [load_week7_data_parquets, load_single_gpu_parquets,
                   load_ddp_parquets, load_dataset_scale_parquets,
                   load_edge_case_jsons]:
        df = loader()
        if not df.empty:
            frames.append(df)

    if not frames:
        raise RuntimeError("No telemetry data found in week7/")

    combined = pd.concat(frames, ignore_index=True)

    for sig in RAW_SIGNALS:
        if sig not in combined.columns:
            combined[sig] = 0.0

    if "ts" in combined.columns:
        combined = combined.sort_values(["run_id", "ts"]).reset_index(drop=True)

    # Labels via centralized manifest (no heuristics)
    combined["binary_label"] = combined["workload_label"].apply(get_category)
    combined["is_training"]  = combined["workload_label"].apply(is_training)

    # Detect GPU TDP for normalized power
    if "gpu_name" in combined.columns:
        combined["tdp_w"] = combined["gpu_name"].apply(get_tdp)
    else:
        combined["tdp_w"] = 1000.0  # B200 default

    log.info(f"Total rows: {len(combined):,} | runs: {combined['run_id'].nunique()} | "
             f"labels: {combined['workload_label'].nunique()}")
    log.info(f"Binary distribution:\n{combined['binary_label'].value_counts().to_string()}")
    return combined


# ── Window feature extraction ─────────────────────────────────────────────────

def acf_lag1(x: np.ndarray) -> float:
    if len(x) < 3:
        return 0.0
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    if var < 1e-12:
        return 0.0
    return float(((x[:-1] - mu) * (x[1:] - mu)).mean() / var)


def extract_window_features(chunk: pd.DataFrame) -> dict:
    """Extract 125 features from a single time window."""
    feats = {}

    for sig in RAW_SIGNALS:
        vals = chunk[sig].fillna(0).values.astype(float) if sig in chunk.columns \
            else np.zeros(max(len(chunk), 1))

        n    = len(vals)
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
    util    = chunk["gpu_utilization_pct"].fillna(0).values.astype(float)
    power   = chunk["power_draw_w"].fillna(0).values.astype(float)
    pcie_tx = chunk["pcie_tx_mbps"].fillna(0).values.astype(float)
    pcie_rx = chunk["pcie_rx_mbps"].fillna(0).values.astype(float)
    sm_clk  = chunk["sm_clock_mhz"].fillna(0).values.astype(float)
    mem_clk = chunk["mem_clock_mhz"].fillna(0).values.astype(float)

    feats["power_per_util"]       = power.mean() / (util.mean() + 1.0)
    feats["pcie_total_mean"]      = (pcie_tx + pcie_rx).mean()
    feats["util_per_sm_pct"]      = util.mean() / (sm_clk.mean() + 1.0) * 1000.0

    # NEW: normalized power as % of TDP (enables cross-GPU comparison)
    tdp = chunk["tdp_w"].iloc[0] if "tdp_w" in chunk.columns else 1000.0
    feats["power_pct_tdp"]        = power.mean() / tdp * 100.0

    # Autocorrelation (lag-1) for two key signals
    feats["acf1_gpu_util"]  = acf_lag1(util)
    feats["acf1_power"]     = acf_lag1(power)

    # NEW: clock dynamic range features (B200 has 12.5x mem_clock range vs A100's ~0x)
    feats["mem_clock_dynamic_range"] = (mem_clk.max() - mem_clk.min()) / (mem_clk.mean() + 1.0)
    feats["sm_clock_dynamic_range"]  = (sm_clk.max() - sm_clk.min()) / (sm_clk.mean() + 1.0)

    return feats


DERIVED_FEATURES = [
    "power_per_util", "pcie_total_mean", "util_per_sm_pct", "power_pct_tdp",
    "acf1_gpu_util", "acf1_power",
    "mem_clock_dynamic_range", "sm_clock_dynamic_range",
]


def sliding_windows(df: pd.DataFrame, window_sec: float, stride_sec: float) -> pd.DataFrame:
    """Apply sliding windows per run_id.

    Short-run handling: runs shorter than window_sec are tagged with
    short_run=True. They are included (to preserve edge-case traces) but
    excluded from headline accuracy metrics by downstream code.
    """
    results = []
    for run_id, run_df in df.groupby("run_id"):
        run_df = run_df.sort_values("ts").reset_index(drop=True)
        t_start  = run_df["ts"].iloc[0]
        t_end    = run_df["ts"].iloc[-1]
        duration = t_end - t_start
        label    = run_df["workload_label"].iloc[0]
        binary   = run_df["binary_label"].iloc[0]
        is_tr    = run_df["is_training"].iloc[0]
        is_short = duration < window_sec

        def emit(chunk):
            if len(chunk) < MIN_SAMPLES_PER_WINDOW:
                return
            feats = extract_window_features(chunk)
            feats["run_id"]         = run_id
            feats["window_start"]   = chunk["ts"].iloc[0]
            feats["window_sec"]     = window_sec
            feats["n_samples"]      = len(chunk)
            feats["run_duration_s"] = duration
            feats["short_run"]      = is_short
            feats["workload_label"] = label
            feats["binary_label"]   = binary
            feats["is_training"]    = is_tr
            results.append(feats)

        if is_short:
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
    log.info("=== Week 7 Feature Engineering (B200) ===")
    df_raw = load_all()

    feature_cols = (
        [f"{s}_{st}" for s in RAW_SIGNALS for st in STAT_NAMES]
        + DERIVED_FEATURES
    )
    log.info(f"Feature count per window: {len(feature_cols)}")

    for cfg in WINDOW_CONFIGS:
        win_sec   = cfg["window_sec"]
        stride_sec = cfg["stride_sec"]
        log.info(f"Extracting windows: {win_sec}s window, {stride_sec}s stride ...")
        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec)
        if df_win.empty:
            log.warning(f"  No windows generated for {win_sec}s")
            continue
        out_path = RESULTS / f"windows_{win_sec}s.parquet"
        df_win.to_parquet(out_path, index=False)

        n_total  = len(df_win)
        n_short  = df_win["short_run"].sum()
        n_full   = n_total - n_short
        n_train  = (df_win["binary_label"] == "training").sum()
        n_other  = n_total - n_train
        log.info(f"  {n_total:,} windows -> {out_path.name} "
                 f"(full={n_full}, short_run={n_short}, training={n_train}, non-training={n_other})")

    # Save feature metadata
    meta = {
        "feature_cols":     feature_cols,
        "n_features":       len(feature_cols),
        "signals":          RAW_SIGNALS,
        "stats_per_signal": STAT_NAMES,
        "derived":          DERIVED_FEATURES,
        "window_sizes":     [c["window_sec"] for c in WINDOW_CONFIGS],
        "note": "power_pct_tdp enables cross-GPU comparison (B200 TDP=1000W, H100=700W). "
                "Raw watts are GPU-specific; normalized power is transferable.",
    }
    with open(RESULTS / "feature_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    log.info("Done. Feature meta saved.")


if __name__ == "__main__":
    main()
