"""High-rate raw current-waveform spectral features (Rogowski probe).

The scope logger keeps, per run, a short raw snippet `*_scope_*_raw.npy` — bare
int16 ADC counts at ~1 MS/s (`scripts/pico_scope_logger.py`). The 1 kHz envelope
parquet (used by `scope_features.py`) throws away everything above ~500 Hz. This
module mines the raw snippet for structure the envelope cannot show — above all
the periodic current signature of the optimizer weight-update, which recurs at
the training step frequency and is a candidate "training tell" that survives when
the NVML utilisation/memory signals have been flattened by an evader.

Sensor-extensibility contract (mirrors scope_features.py / dcgm_logger.py): a new
probe type is added as (1) a logger that writes a per-run sidecar file, and (2) a
feature module here exposing `compute_*_features(path, ...) -> dict` plus a
`build_*_map(dirs)` + `attach_*_features(windows, map)` pair. Nothing else in the
classifier needs to change; features left-join onto the window frame by run_id.

Raw → amps: `counts * (RANGE_MV/1000) / ADC_MAX_16BIT / PROBE_V_PER_A`.
Runs were captured at ±5000 mV, 16-bit. fs is recovered from the sibling
`_scope.parquet` `attrs["sample_rate_hz"]` (fallback 1e6).
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd
from scipy import signal

PROBE_V_PER_A = 0.1        # Micsig RCP60XS, 100 mV/A
RANGE_MV = 5000            # capture range used for all runs
ADC_MAX_16BIT = 32767      # signed 16-bit full scale
COUNTS_TO_AMPS = (RANGE_MV / 1000.0) / ADC_MAX_16BIT / PROBE_V_PER_A

RAW_FEATURE_NAMES = [
    "raw_amps_rms", "raw_crest_factor",
    "raw_bp_sub1hz", "raw_bp_1_10hz", "raw_bp_10_100hz",
    "raw_bp_100hz_1khz", "raw_bp_1_10khz", "raw_bp_10_100khz", "raw_bp_100_500khz",
    "raw_peak_freq_hz", "raw_peak_prominence",
    "raw_lf_spectral_flatness",   # 0..1; low => a few dominant lines (periodic)
    "raw_step_line_score",        # strength of the strongest periodic line 0.5–2000 Hz
    "raw_step_line_freq_hz",
]


def _sibling_fs(raw_path: str) -> float:
    par = raw_path.replace("_raw.npy", ".parquet")
    try:
        return float(pd.read_parquet(par).attrs.get("sample_rate_hz", 1e6))
    except Exception:
        return 1e6


def load_amps(raw_path: str):
    """Return (amps_ac, fs): DC-removed current waveform in amps and its rate."""
    raw = np.load(raw_path).astype(np.float64)
    fs = _sibling_fs(raw_path)
    amps = raw * COUNTS_TO_AMPS
    amps -= amps.mean()   # AC only; the probe cannot see DC anyway
    return amps, fs


def compute_raw_features(raw_path: str) -> dict:
    out = {k: np.nan for k in RAW_FEATURE_NAMES}
    try:
        x, fs = load_amps(raw_path)
    except Exception:
        return out
    if x.size < 4096:
        return out

    rms = float(np.sqrt(np.mean(x ** 2)))
    out["raw_amps_rms"] = rms
    out["raw_crest_factor"] = float(np.max(np.abs(x)) / rms) if rms > 0 else np.nan

    # Welch PSD. 1 s segments give ~1 Hz resolution up to fs/2 (~0.5 MHz).
    nperseg = int(min(len(x), fs))  # ~1 s
    f, pxx = signal.welch(x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    tot = pxx.sum()
    if tot <= 0:
        return out

    def band(lo, hi):
        m = (f >= lo) & (f < hi)
        return float(pxx[m].sum() / tot)
    out["raw_bp_sub1hz"] = band(0, 1)
    out["raw_bp_1_10hz"] = band(1, 10)
    out["raw_bp_10_100hz"] = band(10, 100)
    out["raw_bp_100hz_1khz"] = band(100, 1000)
    out["raw_bp_1_10khz"] = band(1000, 10000)
    out["raw_bp_10_100khz"] = band(10000, 100000)
    out["raw_bp_100_500khz"] = band(100000, 500000)

    # global dominant line (exclude the DC bin)
    nz = f > 0.5
    fp, pp = f[nz], pxx[nz]
    k = int(np.argmax(pp))
    out["raw_peak_freq_hz"] = float(fp[k])
    out["raw_peak_prominence"] = float(pp[k] / (np.median(pp) + 1e-30))

    # Optimizer-step line: the strongest narrow line in the plausible training
    # step-rate band (0.5–2000 Hz). Spectral flatness there measures periodicity:
    # a saturated HPC kernel is broadband (flatness ~1); a stepped trainer has a
    # dominant line (flatness << 1).
    lf = (f >= 0.5) & (f <= 2000)
    p = pxx[lf]
    if p.size and p.min() > 0:
        gm = np.exp(np.mean(np.log(p)))
        out["raw_lf_spectral_flatness"] = float(gm / p.mean())
    if p.size:
        j = int(np.argmax(p))
        out["raw_step_line_score"] = float(p[j] / (np.median(p) + 1e-30))
        out["raw_step_line_freq_hz"] = float(f[lf][j])
    return out


def build_raw_map(data_dirs) -> dict:
    """run_id (12-hex from filename) -> raw npy path."""
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]
    m = {}
    for d in data_dirs:
        for p in glob.glob(os.path.join(str(d), "**", "*_scope_*_raw.npy"), recursive=True):
            rid = os.path.basename(p).split("_scope_", 1)[1].split("_")[0]
            m[rid] = p
    return m


def attach_raw_features(windows: pd.DataFrame, raw_map: dict) -> pd.DataFrame:
    """Left-join per-run raw-spectral features onto a window frame. The raw
    snippet is one short capture per run, so every window of a run gets that
    run's raw features (constant within run). Runs with no raw snippet get NaN."""
    cache: dict[str, dict] = {}
    rows = []
    for rid in windows["run_id"].astype(str):
        if rid not in cache:
            path = raw_map.get(rid)
            cache[rid] = compute_raw_features(path) if path else {k: np.nan for k in RAW_FEATURE_NAMES}
        rows.append(cache[rid])
    feat = pd.DataFrame(rows, index=windows.index)
    for k in RAW_FEATURE_NAMES:
        windows[k] = feat[k].values
    return windows
