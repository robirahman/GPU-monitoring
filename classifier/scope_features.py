"""Per-window oscilloscope (Rogowski current-probe) features.

The scope logger (`scripts/pico_scope_logger.py`) writes a ~1 kHz envelope
parquet per run with columns:
    epoch_s, v_mean, v_rms_ac, v_min, v_max, amps_mean, amps_rms_ac

The probe is a Micsig RCP60XS (100 mV/A, AC/dI-dt only, ~40 Hz LF cutoff), so
`v_rms_ac` / `amps_rms_ac` is the switching-ripple envelope — a burstiness
signal NVML cannot see. These features are joined onto the NVML window frame by
`run_id` + window time, so the classifier can be evaluated with and without the
scope tier. Runs with no sibling `_scope.parquet` simply get NaN scope columns.

The workload-shaping evader inserts idle gaps between training micro-steps.
On a dI/dt sensor each gap edge is an event, so `scope_edge_rate` is predicted
to *rise* with more aggressive shaping — the feature that should make a shaped
run more visible on the very channel it targets, not less.
"""
from __future__ import annotations

import glob
import os

import numpy as np
import pandas as pd

PROBE_V_PER_A = 0.1  # Micsig RCP60XS, 100 mV/A

SCOPE_FEATURE_NAMES = [
    "scope_ripple_a_mean", "scope_ripple_a_std", "scope_ripple_a_p95",
    "scope_ripple_a_min", "scope_ripple_a_range",
    "scope_duty_frac", "scope_edge_rate_hz",
    "scope_cadence_band_ratio", "scope_lf_band_ratio",
]


def compute_scope_window_features(scope: pd.DataFrame, wstart: float, wend: float) -> dict:
    """Features over the scope bins whose epoch falls in [wstart, wend)."""
    m = (scope["epoch_s"].values >= wstart) & (scope["epoch_s"].values < wend)
    a = scope["amps_rms_ac"].values[m].astype(float) if "amps_rms_ac" in scope.columns \
        else scope["v_rms_ac"].values[m].astype(float) / PROBE_V_PER_A
    out = {k: np.nan for k in SCOPE_FEATURE_NAMES}
    if a.size < 16:
        return out

    out["scope_ripple_a_mean"] = float(a.mean())
    out["scope_ripple_a_std"] = float(a.std())
    out["scope_ripple_a_p95"] = float(np.percentile(a, 95))
    out["scope_ripple_a_min"] = float(a.min())
    out["scope_ripple_a_range"] = float(a.max() - a.min())

    # Relative on/off threshold: 20% up from the window's own floor. Captures
    # the gap structure without needing an absolute idle calibration.
    lo, hi = np.percentile(a, 5), np.percentile(a, 95)
    thr = lo + 0.2 * (hi - lo)
    active = a > thr
    out["scope_duty_frac"] = float(active.mean())

    # Sample rate from median bin spacing (envelope is ~1 kHz).
    ep = scope["epoch_s"].values[m]
    dt = np.median(np.diff(ep)) if ep.size > 1 else 1e-3
    fs = 1.0 / dt if dt > 0 else 1000.0
    # Edge rate: threshold crossings per second — the dI/dt gap-edge discriminator.
    crossings = int(np.count_nonzero(np.diff(active.astype(np.int8)) != 0))
    span_s = (ep[-1] - ep[0]) if ep.size > 1 else 1.0
    out["scope_edge_rate_hz"] = float(crossings / span_s) if span_s > 0 else 0.0

    # Spectrum of the ripple envelope. Shaping cadence (~gap period) lands a few
    # Hz to tens of Hz; steady training/inference concentrates power below ~1 Hz.
    x = a - a.mean()
    if x.size >= 64 and np.any(x):
        freqs = np.fft.rfftfreq(x.size, dt)
        p = np.abs(np.fft.rfft(x)) ** 2
        tot = p.sum()
        if tot > 0:
            cadence = (freqs >= 2.0) & (freqs <= 40.0)   # gap cadence band
            lf = (freqs > 0) & (freqs < 2.0)             # slow drift / epoch
            out["scope_cadence_band_ratio"] = float(p[cadence].sum() / tot)
            out["scope_lf_band_ratio"] = float(p[lf].sum() / tot)
    return out


def build_scope_path_map(data_dirs) -> dict:
    """Map full run_id (12-hex, from the scope filename) -> scope parquet path.

    Scope files are named ``{label}_scope_{run_id12}_{ts}.parquet``.
    """
    if isinstance(data_dirs, (str, os.PathLike)):
        data_dirs = [data_dirs]
    m = {}
    for d in data_dirs:
        for path in glob.glob(os.path.join(str(d), "**", "*_scope_*.parquet"), recursive=True):
            base = os.path.basename(path)
            if base.endswith("_raw.npy"):
                continue
            # token after '_scope_' up to the next underscore-timestamp
            after = base.split("_scope_", 1)[1]
            run_id = after.split("_")[0]
            m[run_id] = path
    return m


def attach_scope_features(windows: pd.DataFrame, scope_path_map: dict) -> pd.DataFrame:
    """Left-join per-window scope features onto a window feature frame.

    Windows whose run_id has no scope file get NaN scope columns (so a mixed
    corpus — most runs NVML-only, our shaping runs with scope — still trains).
    """
    if windows.empty:
        for k in SCOPE_FEATURE_NAMES:
            windows[k] = np.nan
        return windows

    cache: dict[str, pd.DataFrame] = {}
    rows = []
    for _, w in windows[["run_id", "window_start", "window_end"]].iterrows():
        rid = w["run_id"]
        path = scope_path_map.get(rid)
        if path is None:
            rows.append({k: np.nan for k in SCOPE_FEATURE_NAMES})
            continue
        if rid not in cache:
            cache[rid] = pd.read_parquet(path, columns=None)
        rows.append(compute_scope_window_features(
            cache[rid], w["window_start"], w["window_end"]))
    feat = pd.DataFrame(rows, index=windows.index)
    for k in SCOPE_FEATURE_NAMES:
        windows[k] = feat[k].values
    return windows
