"""
Improved Three-Way GPU Workload Classifier
===========================================
Classifies GPU telemetry windows as: ml_training / ml_inference / other

Key improvements over Week 5 baseline (48-61% → target 80%+):
1. Training-specific temporal features (epoch periodicity, memory growth rate)
2. Inference-distinguishing features (no memory growth + high util, batch-pause ratio)
3. Loads ALL data sources (data/, week4/, week5/, data/h100_nvl/)
4. XGBoost with balanced class weights and proper cross-validation
5. Better label mapping for edge-case workloads

Usage:
    python -m classifier.threeway_improved
    python classifier/threeway_improved.py
"""

import json
import logging
import pathlib
import re
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
warnings.filterwarnings("ignore")

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# ── Label taxonomy ───────────────────────────────────────────────────────────

from classifier.workload_labels import (  # shared by software and physical sensors
    TRAINING_LABELS, INFERENCE_LABELS, OTHER_LABELS, threeway_label, canonical_workload,
)
from classifier.data_sources import accelerator_files, read_parquet, resolve_payload, filename_workload_label, pointer, sha256
from classifier.window_labels import resolve_work_events, assign_work_labels, own_intervals


# ── Raw signal columns ───────────────────────────────────────────────────────

RAW_SIGNALS = [
    "gpu_utilization_pct", "mem_utilization_pct", "mem_used_mb",
    "power_draw_w", "temperature_c", "sm_clock_mhz", "mem_clock_mhz",
    "pcie_tx_mbps", "pcie_rx_mbps",
]

STAT_NAMES = ["mean", "std", "min", "max", "p25", "p50", "p75", "p95",
              "iqr", "range", "cv", "skew", "kurt"]


# ── Data loading ─────────────────────────────────────────────────────────────

# GPU name aliases: SKUs NVML reports differently that we want to treat as
# one class. H200 SXM and H200 NVL are the same silicon / 141 GB HBM3e —
# only the form factor differs, so we merge them for classifier training.
# A100-SXM4-40GB vs 80GB stay separate (different memory capacity matters),
# H100-80GB HBM3 vs H100-NVL stay separate (different memory + TDP class).
GPU_NAME_ALIASES: dict[str, str] = {
    "NVIDIA H200 NVL": "NVIDIA H200",
}


def _normalize_gpu_name(name):
    if name is None:
        return None
    s = str(name).strip()
    return GPU_NAME_ALIASES.get(s, s)


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Rename variant column names to the standard schema."""
    rename = {
        "timestamp_epoch": "ts",
        "timestamp":       "ts",
        "t":               "ts",
        "gpu_util":        "gpu_utilization_pct",
        "mem_util":        "mem_utilization_pct",
        "power_w":         "power_draw_w",
        "temp_c":          "temperature_c",
        "sm_mhz":          "sm_clock_mhz",
    }
    for old, new in rename.items():
        if old in df and new not in df:
            df = df.rename(columns={old: new})

    if "ts" not in df.columns and "timestamp_utc" in df.columns:
        parsed = pd.to_datetime(df["timestamp_utc"], utc=True, errors="coerce")
        df["ts"] = (parsed.dt.as_unit("ns").astype("int64") / 1e9).where(parsed.notna())

    if "ts" in df and pd.api.types.is_datetime64_any_dtype(df["ts"]):
        parsed = pd.to_datetime(df["ts"], utc=True)
        df["ts"] = (parsed.dt.as_unit("ns").astype("int64") / 1e9).where(parsed.notna())
    elif "ts" in df and not pd.api.types.is_numeric_dtype(df["ts"]):
        numeric = pd.to_numeric(df["ts"], errors="coerce")
        if numeric.notna().any():
            df["ts"] = numeric
        else:
            parsed = pd.to_datetime(df["ts"], utc=True, errors="coerce")
            df["ts"] = (parsed.dt.as_unit("ns").astype("int64") / 1e9).where(parsed.notna())

    if "mem_used_gb" in df.columns and "mem_used_mb" not in df.columns:
        df["mem_used_mb"] = df["mem_used_gb"] * 1024
    if "gpu_uuid" not in df:
        for device_col in ("gpu", "gpu_index"):
            if device_col in df:
                df["gpu_uuid"] = "gpu" + df[device_col].astype(str)
                break

    for sig in RAW_SIGNALS:
        if sig not in df.columns:
            df[sig] = np.nan

    # Older NVML exports use text such as "N/A" for unavailable readings.
    # Normalize before concatenation so a mixed numeric/text column remains
    # serializable in the pooled raw cache. Missing readings are not zeros.
    for sig in RAW_SIGNALS + ["encoder_util_pct", "decoder_util_pct"]:
        if sig in df and not pd.api.types.is_numeric_dtype(df[sig]):
            df[sig] = pd.to_numeric(df[sig], errors="coerce")
    if "throttle_status" in df:
        # This is categorical status metadata, not a numeric model feature.
        df["throttle_status"] = df["throttle_status"].astype("string").replace(
            {"N/A": pd.NA, "[N/A]": pd.NA})

    if "gpu_name" in df.columns:
        df["gpu_name"] = df["gpu_name"].map(_normalize_gpu_name)

    return df


MANIFEST_STREAMS = PROJECT_ROOT / "docs/camera_ready_manifest_20261022/streams.parquet"


def tier_files(tier, manifest=MANIFEST_STREAMS):
    """The accelerator files one frozen tier of the corpus is made of.

    Regenerating a camera-ready number means loading the streams that back it
    and no others. The tree holds three experimental eras side by side, so an
    unfiltered load silently mixes the submitted corpus with the rebuttal
    additions and the multi-node follow-up.
    """
    import pandas as _pd
    if not pathlib.Path(manifest).is_file():
        raise FileNotFoundError(f"{manifest} is missing; run scripts/camera_ready_manifest.py")
    streams = _pd.read_parquet(manifest, columns=["file", "tier"])
    wanted = {tier} if isinstance(tier, str) else set(tier)
    unknown = wanted - set(streams.tier.unique())
    if unknown:
        raise ValueError(f"unknown tier(s) {sorted(unknown)}; "
                         f"manifest has {sorted(streams.tier.unique())}")
    return {str(PROJECT_ROOT / f) for f in streams.loc[streams.tier.isin(wanted), "file"]}


def load_all_data(tier=None) -> pd.DataFrame:
    """Load telemetry from all data sources, or from one frozen tier of them."""
    frames = []
    keep = tier_files(tier) if tier is not None else None

    for p in accelerator_files(PROJECT_ROOT):
        if keep is not None and str(p) not in keep:
            continue
        # Resolve before reading. Missing objects and bad schemas are fatal,
        # so this function can never silently train on a partial LFS checkout.
        df = _normalize_columns(read_parquet(p))
        rel = str(p.relative_to(PROJECT_ROOT))
        if "ts" not in df:
            raise ValueError(f"No timestamp in accelerator input {rel}")
        label = filename_workload_label(p)
        if "workload_label" not in df:
            df["workload_label"] = label
        else:
            df["workload_label"] = df.workload_label.fillna(label)
        if "run_id" not in df or df.run_id.isna().any():
            ptr = pointer(p)
            fallback_run = "source:" + (ptr[0] if ptr else sha256(p))
            df["run_id"] = df.run_id.fillna(fallback_run) if "run_id" in df else fallback_run
        df["source_path"] = rel
        frames.append(df)

    ec_dir = PROJECT_ROOT / "week4/results/edge_cases"
    for p in sorted(ec_dir.glob("*_telemetry.json")):
        records = json.loads(resolve_payload(p).read_text())
        if records:
            df = _normalize_columns(pd.DataFrame(records))
            name = p.stem.removesuffix("_telemetry")
            df["workload_label"] = name
            df["run_id"] = f"ec_{name}"
            df["source_path"] = str(p.relative_to(PROJECT_ROOT))
            frames.append(df)

    if not frames:
        raise RuntimeError("No telemetry data found")

    combined = pd.concat(frames, ignore_index=True)
    # Concat has materialized the combined frame. Keeping every input frame
    # alive through identity assignment and filtering doubles corpus residency.
    del frames

    # Drop rows with no timestamp. A few parquets carry an all-null padding row
    # (every metric NaN, including ts/gpu_name/hostname) -- most likely a
    # *.parquet.partial checkpoint flushed with a header and no sample. Such a
    # row cannot be windowed, and it propagates a NaN run_start into
    # assign_stream_and_job_ids, which then dies with IntCastingNaNError.
    # NON-FINITE counts as "no timestamp" too. `.isna()` is False for +/-inf, and
    # assign_stream_and_job_ids casts (run_start / BUCKET) to int64 -- so a single
    # infinite ts anywhere in the corpus raises the same IntCastingNaNError this
    # guard exists to prevent, taking every other run down with it. Observed on
    # the full 3.6 M-row corpus, where it aborted the binary train-vs-rest CV.
    if "ts" in combined.columns:
        _ts = pd.to_numeric(combined["ts"], errors="coerce")
        bad = ~np.isfinite(_ts.to_numpy(dtype="float64", na_value=np.nan))
        n_bad = int(bad.sum())
        if n_bad:
            log.warning(f"dropping {n_bad} row(s) with a missing or non-finite "
                        f"timestamp")
            combined = combined[~bad].reset_index(drop=True)

    # Deduplicate: some files may appear in multiple directories.
    # gpu_uuid must be in the key. NVIDIA runs survive on (ts, run_id) only by
    # luck: one collector process per GPU means each has its own datetime.now(),
    # so the float epochs differ. The TPU collector samples every chip in ONE
    # poll and stamps them with a single `now`
    # (scripts/collect_tpu_telemetry.py:428-429), and the queue mandates a shared
    # run_id across workers -- so a 16-chip slice would silently collapse to one
    # chip. Adding the per-chip uuid keeps the original intent (identical rows
    # from duplicated files still dedup, since a copy shares all three keys).
    if "ts" in combined.columns and "run_id" in combined.columns:
        _key = ["ts", "run_id", "workload_label"]
        if "gpu_uuid" in combined.columns:
            _key.append("gpu_uuid")
        _before = len(combined)
        combined = combined.drop_duplicates(subset=_key, keep="first")
        log.info(f"dedup on {_key}: {_before:,} -> {len(combined):,} rows")
        combined = combined.sort_values(["run_id", "ts"])
        combined.reset_index(drop=True, inplace=True)

    # Add three-way labels
    from classifier.workload_labels import validate_labels
    validate_labels(combined.workload_label.unique())
    combined["threeway_label"] = combined["workload_label"].apply(threeway_label)

    label_dist = combined.groupby("threeway_label")["run_id"].nunique()
    log.info(f"Total rows: {len(combined):,} | runs: {combined['run_id'].nunique()}")
    log.info(f"Three-way distribution (runs):\n{label_dist.to_string()}")

    from classifier.data_sources import apply_quality_exclusions
    return apply_quality_exclusions(combined, PROJECT_ROOT)


# ── Enhanced feature extraction ──────────────────────────────────────────────

def acf_lag(x: np.ndarray, lag: int) -> float:
    """Autocorrelation at given lag."""
    if len(x) < lag + 3:
        return 0.0
    mu = x.mean()
    var = ((x - mu) ** 2).mean()
    if var < 1e-12:
        return 0.0
    return float(((x[:-lag] - mu) * (x[lag:] - mu)).mean() / var)


def fft_dominant_period(x: np.ndarray) -> float:
    """Dominant period (in samples) from FFT. Captures epoch periodicity."""
    if len(x) < 10:
        return 0.0
    x_centered = x - x.mean()
    fft_vals = np.abs(np.fft.rfft(x_centered))
    # Skip DC component (index 0) and very low frequencies
    if len(fft_vals) < 3:
        return 0.0
    fft_vals[0] = 0
    peak_idx = np.argmax(fft_vals[1:]) + 1
    if fft_vals[peak_idx] < 1e-6:
        return 0.0
    period = len(x) / peak_idx
    return float(period)


def fft_peak_power(x: np.ndarray) -> float:
    """Ratio of peak FFT magnitude to total — measures periodicity strength."""
    if len(x) < 10:
        return 0.0
    x_centered = x - x.mean()
    fft_vals = np.abs(np.fft.rfft(x_centered))
    if len(fft_vals) < 3:
        return 0.0
    fft_vals[0] = 0
    total = fft_vals.sum()
    if total < 1e-12:
        return 0.0
    return float(fft_vals.max() / total)


def extract_enhanced_features(chunk: pd.DataFrame) -> dict:
    """Preserve unavailable channels as NaN, distinct from measured zero.

    Small within-window holes are interpolated; channels below 80% availability
    are excluded with all their dependent features. Imputation is fit per fold.
    """
    chunk = chunk.copy()
    missing = set()
    signals = list(dict.fromkeys(RAW_SIGNALS + [c for c in chunk if c.startswith(("dcgm_", "encoder_", "decoder_"))]))
    for sig in signals:
        v = pd.to_numeric(chunk[sig], errors="coerce") if sig in chunk else pd.Series(np.nan, index=chunk.index)
        v = v.where(np.isfinite(v) & (v >= 0))
        if v.notna().mean() < 0.8:
            missing.add(sig)
        chunk[sig] = v.interpolate().ffill().bfill().fillna(0)
    feats = _extract_enhanced_features_complete(chunk)
    deps = {
        "gpu_utilization_pct": ("gpu_util", "duty_cycle", "idle_frac", "util_transitions", "util_burstiness", "util_per_sm", "power_per_util", "mem_util_per_gpu", "mem_util_minus_gpu", "mem_util_at_high", "mem_util_burst"),
        "power_draw_w": ("power",),
        "mem_used_mb": ("mem_used", "mem_slope", "mem_second_half"),
        "mem_utilization_pct": ("mem_util", "power_per_mem_util"),
        "sm_clock_mhz": ("sm_clock", "util_per_sm"),
        "pcie_tx_mbps": ("pcie",), "pcie_rx_mbps": ("pcie",),
    }
    for sig in missing:
        needles = deps.get(sig, (sig,))
        for key in feats:
            depends = key.startswith(sig) or any(n in key for n in needles)
            if sig == "power_draw_w":
                # "FFT peak power" is spectral energy, not electrical power.
                depends = key.startswith((sig, "power_")) or bool(re.fullmatch(r"acf\d+_power", key))
            if depends:
                feats[key] = np.nan
    return feats


def _extract_enhanced_features_complete(chunk: pd.DataFrame) -> dict:
    """Extract ~155 features per window, including training-specific temporals."""
    feats = {}

    # ── 1. Base statistics (9 signals × 13 stats = 117 features) ─────────
    # BATCHED ACROSS SIGNALS, not because it is tidier but because the per-call
    # overhead dominated. Profiled on 2026-09-18: windowing the corpus cost
    # 23.4 ms/window, of which scipy.stats' axis_nan_policy wrapper was 56% --
    # 18 calls per window (9 signals x skew+kurtosis), and most of that time was
    # the wrapper re-deriving each function's signature with inspect and
    # re-broadcasting its arguments, not computing moments. scipy guarantees
    # f(M, axis=0)[i] == f(M[:, i]), verified here including NaN placement on
    # constant columns, so batching is numerically identical and turns 18 calls
    # into 2 (and 9 np.percentile calls into 1).
    cols = [chunk[s].fillna(0).values.astype(float) if s in chunk.columns
            else np.zeros(max(len(chunk), 1)) for s in RAW_SIGNALS]
    M = np.column_stack(cols)
    n = M.shape[0]
    mus = M.mean(axis=0)
    sds = M.std(axis=0)
    vmins = M.min(axis=0)
    vmaxs = M.max(axis=0)
    P = (np.percentile(M, [25, 50, 75, 95], axis=0) if n > 1
         else np.tile(mus, (4, 1)))
    # Constant columns (an idle GPU, or a TPU's hard-wired -1 sentinel) make
    # scipy return NaN here -- that is the EXISTING behaviour, preserved: the
    # hot paths nan_to_num it to 0 downstream.
    skews = (sp_stats.skew(M, axis=0) if n > 2 else np.zeros(len(RAW_SIGNALS)))
    kurts = (sp_stats.kurtosis(M, axis=0) if n > 3 else np.zeros(len(RAW_SIGNALS)))

    for i, sig in enumerate(RAW_SIGNALS):
        mu, sd = mus[i], sds[i]
        p25, p50, p75, p95 = P[0, i], P[1, i], P[2, i], P[3, i]
        feats[f"{sig}_mean"] = mu
        feats[f"{sig}_std"] = sd
        feats[f"{sig}_min"] = vmins[i]
        feats[f"{sig}_max"] = vmaxs[i]
        feats[f"{sig}_p25"] = p25
        feats[f"{sig}_p50"] = p50
        feats[f"{sig}_p75"] = p75
        feats[f"{sig}_p95"] = p95
        feats[f"{sig}_iqr"] = p75 - p25
        feats[f"{sig}_range"] = vmaxs[i] - vmins[i]
        feats[f"{sig}_cv"] = sd / (abs(mu) + 1e-9)
        feats[f"{sig}_skew"] = float(skews[i])
        feats[f"{sig}_kurt"] = float(kurts[i])

    # ── 2. Cross-signal derived features (5) ─────────────────────────────
    util = chunk["gpu_utilization_pct"].fillna(0).values.astype(float)
    power = chunk["power_draw_w"].fillna(0).values.astype(float)
    pcie_tx = chunk["pcie_tx_mbps"].fillna(0).values.astype(float)
    pcie_rx = chunk["pcie_rx_mbps"].fillna(0).values.astype(float)
    sm_clk = chunk["sm_clock_mhz"].fillna(0).values.astype(float)
    mem_used = chunk["mem_used_mb"].fillna(0).values.astype(float)

    feats["power_per_util"] = power.mean() / (util.mean() + 1.0)
    feats["pcie_total_mean"] = (pcie_tx + pcie_rx).mean()
    # Clamp the clock mean at >= 0 before the +1 guard. Non-NVIDIA collectors
    # write -1 for channels the hardware never exposes (TPU: all of power,
    # temperature and both clocks -- scripts/collect_tpu_telemetry.py:74-78), and
    # -1 + 1 = 0 made this exactly +inf. Every hot path uses .fillna(0), which
    # does NOT touch infinities, so model_store --fit, twostage --train,
    # evasion_cost_eval and scope_ladder_eval all raised
    # "Input X contains infinity"; worse, the cross-vendor evaluators call
    # np.nan_to_num without posinf= and silently turned it into 3.4e38.
    # Clamping makes a -1 sentinel behave exactly like an absent column, which
    # _normalize_columns already zero-fills.
    _sm_mean = max(float(sm_clk.mean()), 0.0)
    feats["util_per_sm_pct"] = util.mean() / (_sm_mean + 1.0) * 1000.0

    # DRAM-activity cross-signal ratios (Round 6 monitor response). mem_util is
    # NVML's fraction of the sample period spent reading/writing device memory —
    # the one channel a weight write must touch and inference never does. These
    # ratios make the DRAM-per-compute direction explicit, which the marginal
    # stats alone leave diagonal. See classifier/rebuttal_analysis/symgemm_eval.py.
    mem_util = chunk["mem_utilization_pct"].fillna(0).values.astype(float) \
        if "mem_utilization_pct" in chunk.columns else np.zeros(max(len(chunk), 1))
    mu_mean, gu_mean = mem_util.mean(), util.mean()
    mu_p95 = np.percentile(mem_util, 95) if len(mem_util) > 1 else mu_mean
    gu_p95 = np.percentile(util, 95) if len(util) > 1 else gu_mean
    feats["mem_util_per_gpu_util"] = mu_mean / (gu_mean + 1.0)
    feats["mem_util_minus_gpu_util"] = mu_mean - gu_mean
    feats["mem_util_per_gpu_util_p95"] = mu_p95 / (gu_p95 + 1.0)
    feats["power_per_mem_util"] = power.mean() / (mu_mean + 1.0)

    # Peak / conditional DRAM-write features (Round 6, PPO response). A diluted
    # trainer (PPO's sparse policy update, gradient accumulation) writes weights
    # in a brief periodic BURST, so its window MEAN mem_util is indistinguishable
    # from decode inference (both ~6%) while the update SPIKE is not (mem_util_max
    # ~11 vs ~8). Mean-based ratios above miss that; these target the spike and
    # its rhythm — the one channel a weight write must touch.
    if len(util) > 1:
        gu_hi = np.percentile(util, 75)
        hi = util > gu_hi
        if hi.sum() >= 1:
            mem_hi = mem_util[hi].mean()          # DRAM during the compute bursts
            gpu_hi = util[hi].mean()
        else:                                     # flat window: fall back to means
            mem_hi, gpu_hi = mu_mean, gu_mean
    else:
        mem_hi, gpu_hi = mu_mean, gu_mean
    mu_p50 = np.percentile(mem_util, 50) if len(mem_util) > 1 else mu_mean
    feats["mem_util_at_high_util"] = mem_hi
    feats["mem_util_spike"] = mu_p95 - mu_p50
    feats["mem_util_burst_ratio"] = mem_hi / (gpu_hi + 1.0)
    feats["mem_util_fft_peak_power"] = fft_peak_power(mem_util)

    # ── 3. Autocorrelation at multiple lags (key for periodicity) ────────
    for lag in [1, 2, 5, 10, 20]:
        feats[f"acf{lag}_gpu_util"] = acf_lag(util, lag)
        feats[f"acf{lag}_power"] = acf_lag(power, lag)
        feats[f"acf{lag}_mem_used"] = acf_lag(mem_used, lag)

    # ── 4. TRAINING-SPECIFIC: Memory growth features ─────────────────────
    # Training shows memory growth during model/optimizer init, then plateau
    # Inference: flat memory. Other: varies but no ramp-then-plateau.
    if len(mem_used) >= 5:
        # Memory slope (linear fit)
        x_t = np.arange(len(mem_used), dtype=float)
        coeffs = np.polyfit(x_t, mem_used, 1)
        feats["mem_slope"] = float(coeffs[0])

        # Memory slope in first half vs second half (training: positive then flat)
        mid = len(mem_used) // 2
        if mid >= 3:
            slope_first = np.polyfit(np.arange(mid, dtype=float), mem_used[:mid], 1)[0]
            slope_second = np.polyfit(np.arange(len(mem_used) - mid, dtype=float), mem_used[mid:], 1)[0]
            feats["mem_slope_first_half"] = float(slope_first)
            feats["mem_slope_second_half"] = float(slope_second)
            feats["mem_slope_ratio"] = float(abs(slope_first) / (abs(slope_second) + 1e-6))
        else:
            feats["mem_slope_first_half"] = 0.0
            feats["mem_slope_second_half"] = 0.0
            feats["mem_slope_ratio"] = 0.0

        # Memory stability: std of second half (training should be stable after init)
        if mid >= 3:
            feats["mem_second_half_cv"] = float(mem_used[mid:].std() / (mem_used[mid:].mean() + 1e-9))
        else:
            feats["mem_second_half_cv"] = 0.0
    else:
        feats["mem_slope"] = 0.0
        feats["mem_slope_first_half"] = 0.0
        feats["mem_slope_second_half"] = 0.0
        feats["mem_slope_ratio"] = 0.0
        feats["mem_second_half_cv"] = 0.0

    # ── 5. TRAINING-SPECIFIC: Epoch periodicity via FFT ──────────────────
    feats["gpu_util_fft_period"] = fft_dominant_period(util)
    feats["gpu_util_fft_peak_power"] = fft_peak_power(util)
    feats["power_fft_period"] = fft_dominant_period(power)
    feats["power_fft_peak_power"] = fft_peak_power(power)
    feats["mem_used_fft_peak_power"] = fft_peak_power(mem_used)

    # ── 6. INFERENCE-SPECIFIC: Batch-pause pattern ───────────────────────
    # Inference has bursts of high util then drops; training is more sustained
    if len(util) >= 5:
        # Fraction of time at high utilization (>80%)
        feats["duty_cycle_80"] = float((util >= 80).mean())
        # Fraction of time at very low utilization (<10%) — idle gaps
        feats["idle_frac"] = float((util < 10).mean())
        # Number of transitions between high and low util per second
        high = util >= 50
        transitions = np.sum(np.abs(np.diff(high.astype(int))))
        feats["util_transitions_per_sec"] = float(transitions / (len(util) + 1e-9))
        # Burstiness: ratio of max to mean utilization
        feats["util_burstiness"] = float(util.max() / (util.mean() + 1e-9))
    else:
        feats["duty_cycle_80"] = 0.0
        feats["idle_frac"] = 0.0
        feats["util_transitions_per_sec"] = 0.0
        feats["util_burstiness"] = 0.0

    # ── 7. Power stability features ──────────────────────────────────────
    if len(power) >= 5:
        feats["power_cv"] = float(power.std() / (power.mean() + 1e-9))
        # Rolling variance of power (30-sample window or all if shorter)
        w = min(30, len(power))
        rolling_mean = pd.Series(power).rolling(window=w, min_periods=1).mean().values
        feats["power_rolling_var"] = float(rolling_mean.var())
        # Power range as fraction of mean
        feats["power_range_frac"] = float((power.max() - power.min()) / (power.mean() + 1e-9))
    else:
        feats["power_cv"] = 0.0
        feats["power_rolling_var"] = 0.0
        feats["power_range_frac"] = 0.0

    # ── 8. PCIe pattern features (training has periodic data loading) ────
    pcie_total = pcie_tx + pcie_rx
    if len(pcie_total) >= 5:
        feats["pcie_cv"] = float(pcie_total.std() / (pcie_total.mean() + 1e-9))
        feats["pcie_fft_peak_power"] = fft_peak_power(pcie_total)
        feats["pcie_nonzero_frac"] = float((pcie_total > 1.0).mean())
    else:
        feats["pcie_cv"] = 0.0
        feats["pcie_fft_peak_power"] = 0.0
        feats["pcie_nonzero_frac"] = 0.0

    # ── 9. Encoder/decoder (video encoding signature) ────────────────────
    for col in ("encoder_util_pct", "decoder_util_pct"):
        if col in chunk.columns:
            s = chunk[col].fillna(0).values.astype(float)
            feats[f"{col}_mean"] = float(s.mean())
            feats[f"{col}_nonzero_frac"] = float((s > 0).mean())
        else:
            feats[f"{col}_mean"] = 0.0
            feats[f"{col}_nonzero_frac"] = 0.0

    # ── 10. DCGM features if available ───────────────────────────────────
    if "dcgm_tensor_active" in chunk.columns:
        s = chunk["dcgm_tensor_active"].fillna(0).values.astype(float)
        feats["dcgm_tensor_active_mean"] = float(s.mean())
        feats["dcgm_tensor_active_max"] = float(s.max())
    if "dcgm_fp16_pipe_active" in chunk.columns and "dcgm_fp32_pipe_active" in chunk.columns:
        fp16 = chunk["dcgm_fp16_pipe_active"].fillna(0).mean()
        fp32 = chunk["dcgm_fp32_pipe_active"].fillna(0).mean()
        total = fp16 + fp32
        feats["dcgm_fp16_ratio"] = float(fp16 / total) if total > 0 else 0.0
    # dram_active is the compulsory-weight-write signal that cadence shaping
    # cannot remove; sm_active gives an occupancy duty cycle independent of the
    # NVML util heuristic. Both are collected by DcgmCollector but were not
    # previously featurized.
    for col in ("dcgm_dram_active", "dcgm_sm_active"):
        if col in chunk.columns:
            s = chunk[col].fillna(0).values.astype(float)
            feats[f"{col}_mean"] = float(s.mean())
            feats[f"{col}_max"] = float(s.max())
            feats[f"{col}_p95"] = float(np.percentile(s, 95))
            feats[f"{col}_active_frac"] = float((s > 0.05).mean())

    return feats


# ── Run-level features (pre-allocation detection) ───────────────────────────

def run_level_features(run_df: pd.DataFrame) -> dict:
    """Features that need the whole run, not just a window.

    Designed to catch the WB-O startup signature: Round 5 showed the oracle
    runs a dummy backward+step at init to materialize AdamW buffers (~6.3 GB
    of optimizer state) before trying to look like inference. No legitimate
    inference workload allocates optimizer state at startup, so a sharp
    first-30s memory jump is a strong pre-training tell.
    """
    feats = {}
    mem = run_df["mem_used_mb"].fillna(0).values.astype(float)
    mem_total = float(run_df["mem_total_mb"].iloc[0]) if "mem_total_mb" in run_df.columns else 1.0

    # Telemetry is ~1 Hz; first 30 samples ≈ first 30 s of the run.
    first_n = min(30, len(mem))
    if first_n >= 5 and len(mem) >= 5:
        mem_start = float(np.median(mem[:3]))
        mem_peak_first = float(np.max(mem[:first_n]))
        delta = mem_peak_first - mem_start
        feats["first_30s_mem_delta_mb"] = delta
        feats["first_30s_mem_delta_frac"] = delta / max(mem_total, 1.0)
        feats["first_30s_mem_ratio"] = mem_peak_first / max(mem_start, 1.0)
    else:
        feats["first_30s_mem_delta_mb"] = 0.0
        feats["first_30s_mem_delta_frac"] = 0.0
        feats["first_30s_mem_ratio"] = 1.0

    # Time to 90 % of plateau memory — slow ramp = true training init,
    # near-zero samples = pre-allocation trick.
    if len(mem) >= 10:
        plateau = float(np.median(mem[-min(30, len(mem)):]))
        mem0 = float(mem[0])
        span = plateau - mem0
        if abs(span) < 1e-6:
            feats["time_to_mem_plateau_samples"] = 0.0
        else:
            threshold = mem0 + 0.9 * span
            target = mem >= threshold if span > 0 else mem <= threshold
            hits = np.where(target)[0]
            feats["time_to_mem_plateau_samples"] = float(hits[0]) if len(hits) > 0 else float(len(mem))
    else:
        feats["time_to_mem_plateau_samples"] = 0.0

    return feats


# ── Sliding window extraction ────────────────────────────────────────────────

def sliding_windows(
    df: pd.DataFrame,
    window_sec: float,
    stride_sec: float,
    run_level_mode: str = "causal",
    step_log_map: dict | None = None,
    work_events: dict | str | None = "auto",
) -> pd.DataFrame:
    """Extract sliding-window feature vectors per run.

    ``work_events`` (``{job_id: intervals}`` from window_labels) labels every
    window by the work its job did inside it rather than by the job's name;
    ``job_label`` then keeps the objective.

    It defaults to ``"auto"``, which resolves the committed launcher logs for
    whatever is in ``df``. A diluted job alternates training with inference, so
    inheriting the job label marks its inference-only windows as missed
    training -- measured at 71% of windows for ``L_diluted_10`` and 78% for
    ``wb_lora_7b_N500``. Passing ``None`` restores the old job-label behaviour
    and is for reproducing archived numbers only, the same role
    ``run_level_mode="orig"`` plays for the causal fix.

    ``run_level_mode`` controls the whole-run "pre-allocation" features
    (``first_30s_mem_*``, ``time_to_mem_plateau_samples``):

      - ``"causal"`` – (DEFAULT) run-level features recomputed per window from
        run start up to that window's end only, so no window sees its own future
        while still exploiting the genuinely-observable startup signature. This
        is what a real-time monitor computes and is the reported/deployed
        configuration (see report/neurips_rebuttal.md §1).
      - ``"none"``   – run-level features omitted entirely. Leak-free floor.
      - ``"orig"``   – computed once over the entire run and attached to every
        window. The plateau term reads ``mem[-30:]`` (the end of the run), so
        early windows carry information from their own future. LEAKY: retained
        only to reproduce archived pre-fix numbers, never for training/serving.
    """
    if run_level_mode not in ("orig", "none", "causal"):
        raise ValueError(
            f"run_level_mode must be 'orig', 'none', or 'causal', got {run_level_mode!r}")

    work_events = resolve_work_events(work_events, df)

    results = []
    # One series per physical stream. A shared run_id spans every GPU and host of
    # a multi-GPU job, and interleaving their samples corrupts every feature.
    group_cols = [c for c in ("run_id", "gpu_uuid", "hostname", "workload_label") if c in df]
    for identity, run_df in df.groupby(group_cols, dropna=False, observed=True):
        identity = identity if isinstance(identity, tuple) else (identity,)
        run_id = identity[0]
        run_df = run_df.sort_values("ts").reset_index(drop=True)
        if len(run_df) < 5:
            continue
        stream_id = (run_df["stream_id"].iloc[0] if "stream_id" in run_df
                     else "|".join(map(str, identity)))
        job_id = run_df["job_id"].iloc[0] if "job_id" in run_df else None

        t_start = run_df["ts"].iloc[0]
        t_end = run_df["ts"].iloc[-1]
        duration = t_end - t_start
        label = run_df["threeway_label"].iloc[0]
        wl_label = run_df["workload_label"].iloc[0]
        gpu_name = run_df["gpu_name"].iloc[0] if "gpu_name" in run_df.columns else ""
        run_feats_full = run_level_features(run_df) if run_level_mode == "orig" else None
        # Per-window ground truth (train step timestamps), if a STEP log exists
        # for this run. None => fall back to the run-level label downstream.
        run_train_ts = step_log_map.get(run_id, {}).get("train_ts") if step_log_map else None

        def emit(chunk, wstart, wend):
            if len(chunk) < 5:
                return
            feats = extract_enhanced_features(chunk)
            if run_level_mode == "orig":
                feats.update(run_feats_full)
            elif run_level_mode == "causal":
                # Only samples observable by the window's end — no look-ahead.
                feats.update(run_level_features(run_df[run_df["ts"] < wend]))
            # run_id stays the run-level key. Callers group CV folds by it, and
            # the sibling GPUs of one run must never straddle a train/test split.
            feats["run_id"] = run_id
            feats["stream_id"] = stream_id
            if job_id is not None:
                feats["job_id"] = job_id
            feats["threeway_label"] = label
            feats["workload_label"] = wl_label
            feats["gpu_name"] = gpu_name
            feats["window_start"] = wstart
            feats["window_end"] = wend
            if step_log_map is not None:
                # 1.0 = a real training step occurred in this window, 0.0 = none,
                # NaN = no STEP log for this run (use the run-level label instead).
                if run_train_ts is not None:
                    active = bool(((run_train_ts >= wstart) & (run_train_ts < wend)).any())
                    feats["window_train_active"] = 1.0 if active else 0.0
                else:
                    feats["window_train_active"] = np.nan
            results.append(feats)

        first = len(results)
        if duration < window_sec:
            emit(run_df, t_start, t_end)
        else:
            wstart = t_start
            while wstart + window_sec <= t_end + 1:
                chunk = run_df[(run_df["ts"] >= wstart) & (run_df["ts"] < wstart + window_sec)]
                emit(chunk, wstart, wstart + window_sec)
                wstart += stride_sec
        if work_events is not None:
            intervals = work_events.get(str(job_id if job_id is not None else run_id))
            if "source_path" in run_df:
                # A GPU's windows show that GPU's work when its job logs per rank.
                own = own_intervals(intervals, run_df["source_path"].iloc[0])
                intervals = own if own is not None else intervals
            assign_work_labels(results[first:], intervals, wl_label, label)

    if not results:
        return pd.DataFrame()
    return pd.DataFrame(results)


# ── Classification ───────────────────────────────────────────────────────────

def workload_group(labels) -> np.ndarray:
    """Group key for held-out-by-WORKLOAD validation.

    Grouping CV by run_id leaks workload identity. There are ~2100 runs over
    ~170 distinct workloads, so every held-out run has siblings of the SAME
    workload in the training fold, and the score measures "recognise a workload
    I have already seen" rather than "generalise to one I have not". Measured
    consequence: run-grouped CV endorsed tau=0.10, which turns out to fire on
    >50% of windows of honest non-ML workloads held out by workload.

    The deployment question is whether an HPC job the detector has never seen
    trips it, so validation must hold out whole workloads.
    """
    return np.array([canonical_workload(label) for label in labels])


def get_feature_cols(df: pd.DataFrame) -> list:
    """Return all numeric feature columns (exclude metadata)."""
    exclude = {"run_id", "job_id", "stream_id", "threeway_label", "binary_label", "y",
               "workload_label", "gpu_name", "gpu_uuid", "hostname", "node_id", "rank",
               "window_start", "window_end", "window_sec", "n_samples", "window_train_active",
               "host", "hosts", "scope", "family", "kind", "grp", "rep", "source_path",
               "expected_ranks", "world_size", "node_rank", "is_attack", "label", "target",
               "xr_n_ranks", "n_ranks", "ts", "timestamp_epoch", "observed_devices", "hosts"}
    return [c for c in df.columns if c not in exclude
            and not c.startswith(("meta_", "target_", "label_", "ground_truth"))
            and not c.endswith(("_label", "_id"))
            and pd.api.types.is_numeric_dtype(df[c])]


def compute_sample_weights(y, groups=None, beta=None, per_run=None):
    """Compute per-sample weights with class + (optional) per-run reweighting.

    Default behaviour (groups=None, beta=None) matches the original
    sklearn-"balanced" scheme exactly, so existing callers are unaffected.

    When `groups` is provided, two additional corrections kick in:

    1. **Effective-number-of-samples class weighting** (Cui et al. 2019).
       Replaces inverse frequency with `(1 - beta) / (1 - beta ** n_c)`.
       With beta close to 1 (default 0.9999) this flattens the diminishing
       returns of near-duplicate samples — useful when one class has many
       files from a single workload and another has fewer files from
       several workloads.

    2. **Per-run normalisation.** Each sample's weight is scaled by
       `1 / n_windows_in_its_run`, so a 300s run producing ~9 windows does
       not dominate a 30s run producing ~1 window. Activated by
       `per_run=True` (the default when groups is given).

    Final weights are renormalised so their mean is 1.0.

    Parameters
    ----------
    y : array-like — class labels (encoded integers or strings).
    groups : array-like, optional — run_id per sample. Required for per-run
        normalisation.
    beta : float, optional — effective-number-of-samples smoothing (in [0,1)).
        When None, falls back to the classic inverse-frequency formula.
    per_run : bool, optional — if True, divide by windows-per-run. Defaults
        to True when `groups` is given, False otherwise.

    Returns
    -------
    weights : np.ndarray of shape (len(y),), dtype float32, mean ≈ 1.
    """
    y_arr = np.asarray(y)
    n = len(y_arr)

    if per_run is None:
        per_run = groups is not None
    if per_run and groups is None:
        raise ValueError("per_run=True requires groups to be provided")

    # Step 1: per-run weight — each sample gets 1/n_samples_in_run, so every
    # run contributes a total weight of 1 (a 10-window run no longer
    # outweighs a 1-window run).
    if per_run:
        groups_arr = np.asarray(groups)
        if len(groups_arr) != n:
            raise ValueError("groups must have the same length as y")
        _, inverse_g = np.unique(groups_arr, return_inverse=True)
        counts_g = np.bincount(inverse_g)
        w_run = 1.0 / counts_g[inverse_g]
    else:
        w_run = np.ones(n, dtype=np.float64)

    # Step 2: class weights computed on *effective* class counts
    # (i.e., counts after per-run normalisation). This lets a class with
    # many runs-per-class still balance against one with few runs,
    # independent of how long any single run happens to be.
    classes, inverse_c = np.unique(y_arr, return_inverse=True)
    n_classes = len(classes)
    eff_counts = np.bincount(inverse_c, weights=w_run)

    if beta is None:
        # Classic inverse-frequency, on effective counts.
        # (Reduces to sklearn's "balanced" when per_run is False.)
        total_eff = eff_counts.sum()
        class_w = total_eff / (n_classes * eff_counts)
    else:
        if not (0.0 <= beta < 1.0):
            raise ValueError(f"beta must be in [0, 1); got {beta}")
        # Round effective counts for the Cui formula's integer-n assumption;
        # with beta=0.9999 this is not sensitive.
        eff_n_cui = (1.0 - np.power(beta, eff_counts)) / (1.0 - beta)
        raw = 1.0 / eff_n_cui
        class_w = raw / raw.mean()

    w = w_run * class_w[inverse_c]

    mean_w = w.mean()
    if mean_w > 0:
        w = w / mean_w
    return w.astype(np.float32)


# Keep balanced_subsample for backward compatibility but prefer compute_sample_weights
def balanced_subsample(X, y, groups=None, random_state=42):
    """Downsample to equal class counts, sampling by group (run_id).

    DEPRECATED: prefer compute_sample_weights() which retains all data.
    """
    rng = np.random.RandomState(random_state)
    classes = np.unique(y)

    if groups is None:
        class_indices = {cls: np.where(y == cls)[0] for cls in classes}
        target = min(len(idxs) for idxs in class_indices.values())
        selected = []
        for cls in classes:
            idxs = class_indices[cls]
            chosen = rng.choice(idxs, size=target, replace=False)
            selected.extend(chosen)
        selected = sorted(selected)
        return X[selected], y[selected]

    class_groups = {cls: [] for cls in classes}
    group_to_indices = {}
    for i, (g, c) in enumerate(zip(groups, y)):
        key = (g, c)
        if key not in group_to_indices:
            group_to_indices[key] = []
        group_to_indices[key].append(i)

    for (g, c), idxs in group_to_indices.items():
        class_groups[c].append((g, idxs))

    class_counts = {cls: sum(len(idxs) for _, idxs in glist)
                    for cls, glist in class_groups.items()}
    target = min(class_counts.values())

    selected = []
    for cls in classes:
        glist = class_groups[cls].copy()
        rng.shuffle(glist)
        count = 0
        for g, idxs in glist:
            if count >= target:
                break
            selected.extend(idxs)
            count += len(idxs)

    selected = sorted(selected)
    return X[selected], y[selected]


def train_and_evaluate():
    """Main training and evaluation pipeline."""
    from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.metrics import classification_report, accuracy_score, f1_score
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import Pipeline
    import xgboost as xgb

    # Load data
    log.info("Loading all telemetry data...")
    df_raw = load_all_data()

    # Extract sliding windows at multiple sizes
    best_results = {}

    for win_sec, stride_sec in [(15, 7), (30, 15), (60, 30)]:
        log.info(f"\n{'='*60}")
        log.info(f"Window: {win_sec}s, stride: {stride_sec}s")
        log.info(f"{'='*60}")

        df_win = sliding_windows(df_raw, window_sec=win_sec, stride_sec=stride_sec,
                                 run_level_mode="causal")
        if df_win.empty:
            log.warning(f"No windows for {win_sec}s, skipping")
            continue

        feature_cols = get_feature_cols(df_win)
        X = df_win[feature_cols].fillna(0).values.astype(np.float32)
        y = df_win["threeway_label"].values
        # Fold by physical job: every rank of one execution, including legacy
        # per-collector siblings, lands on the same side of each split.
        group_key = "job_id" if "job_id" in df_win else "run_id"
        groups = df_win[group_key].values

        log.info(f"Windows: {len(df_win)} | Features: {len(feature_cols)}")
        log.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")

        # Check if we have enough classes and samples
        unique_classes = np.unique(y)
        if len(unique_classes) < 3:
            log.warning(f"Only {len(unique_classes)} classes found, need 3. Skipping.")
            continue

        # Check minimum samples per class per run for cross-validation
        run_class_counts = df_win.groupby(group_key)["threeway_label"].first().value_counts()
        min_runs = run_class_counts.min()
        n_splits = min(5, min_runs)
        if n_splits < 2:
            log.warning(f"Not enough runs per class for cross-validation (min={min_runs})")
            # Fall back to single train/test split
            n_splits = 2

        log.info(f"Using {n_splits}-fold cross-validation")

        # Define classifiers
        classifiers = {
            "XGBoost": xgb.XGBClassifier(
                n_estimators=500, max_depth=6, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8,
                use_label_encoder=False, eval_metric="mlogloss",
                random_state=42, n_jobs=-1, verbosity=0,
            ),
            "RandomForest": RandomForestClassifier(
                n_estimators=400, max_depth=None, min_samples_leaf=2,
                max_features="sqrt", class_weight="balanced",
                random_state=42, n_jobs=-1,
            ),
            "GradientBoosting": GradientBoostingClassifier(
                n_estimators=300, max_depth=5, learning_rate=0.05,
                subsample=0.8, random_state=42,
            ),
        }

        cv = StratifiedGroupKFold(n_splits=n_splits, shuffle=True, random_state=42)

        for clf_name, clf in classifiers.items():
            accs, f1s = [], []
            all_y_true, all_y_pred = [], []

            for fold_idx, (tr_idx, te_idx) in enumerate(cv.split(X, y, groups)):
                X_tr, X_te = X[tr_idx], X[te_idx]
                y_tr, y_te = y[tr_idx], y[te_idx]

                # Handle XGBoost label encoding + class weights
                if clf_name == "XGBoost":
                    from sklearn.preprocessing import LabelEncoder
                    le = LabelEncoder()
                    y_tr_enc = le.fit_transform(y_tr)
                    w_tr = compute_sample_weights(y_tr_enc)
                    clf.fit(X_tr, y_tr_enc, sample_weight=w_tr)
                    y_pred_enc = clf.predict(X_te)
                    y_pred = le.inverse_transform(y_pred_enc)
                else:
                    clf.fit(X_tr, y_tr)
                    y_pred = clf.predict(X_te)

                acc = accuracy_score(y_te, y_pred)
                f1 = f1_score(y_te, y_pred, average="macro", zero_division=0)
                accs.append(acc)
                f1s.append(f1)
                all_y_true.extend(y_te)
                all_y_pred.extend(y_pred)

            mean_acc = np.mean(accs)
            std_acc = np.std(accs)
            mean_f1 = np.mean(f1s)

            log.info(f"  [{clf_name:20s}] acc={mean_acc:.4f}±{std_acc:.4f}  f1_macro={mean_f1:.4f}")

            # Print per-class report for best window
            if clf_name == "XGBoost":
                report = classification_report(all_y_true, all_y_pred, zero_division=0)
                log.info(f"\n{report}")

            key = f"{clf_name}_{win_sec}s"
            best_results[key] = {
                "classifier": clf_name,
                "window_sec": win_sec,
                "acc_mean": mean_acc,
                "acc_std": std_acc,
                "f1_macro": mean_f1,
            }

        # Feature importance for best classifier (XGBoost)
        if "XGBoost" in classifiers:
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_enc = le.fit_transform(y)
            xgb_clf = classifiers["XGBoost"]
            xgb_clf.fit(X, y_enc)
            importances = xgb_clf.feature_importances_
            top_idx = np.argsort(importances)[::-1][:25]
            log.info(f"\nTop-25 features (XGBoost, {win_sec}s window):")
            for i in top_idx:
                log.info(f"  {feature_cols[i]:45s} {importances[i]:.4f}")

    # Summary
    log.info("\n" + "=" * 60)
    log.info("SUMMARY OF ALL RESULTS")
    log.info("=" * 60)
    for key, res in sorted(best_results.items(), key=lambda x: -x[1]["acc_mean"]):
        log.info(f"  {key:35s}  acc={res['acc_mean']:.4f}±{res['acc_std']:.4f}  f1={res['f1_macro']:.4f}")

    # Save results
    results_dir = PROJECT_ROOT / "classifier" / "results"
    results_dir.mkdir(exist_ok=True)
    pd.DataFrame(best_results.values()).to_csv(results_dir / "threeway_results.csv", index=False)
    log.info(f"\nResults saved to {results_dir / 'threeway_results.csv'}")


if __name__ == "__main__":
    train_and_evaluate()
