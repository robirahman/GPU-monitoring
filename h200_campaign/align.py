"""Clock alignment from common physical markers.

Reference timeline: host CLOCK_MONOTONIC ns. GPU burst edges come from CUDA
events (events.py 'gpu_time' lines). Each sensor's edges are detected
independently and compared with those GPU edges; the per-edge differences give
the offset, its spread (alignment uncertainty) and drift between the markers
at the start and at the end of an acquisition.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


def load_events(path) -> pd.DataFrame:
    rows = [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
    ev = pd.DataFrame(rows)
    gpu = ev[ev.event == "gpu_time"][["stream", "pid", "gpu_ref", "gpu_mono_ns"]]
    marks = ev[ev.get("gpu_ref").notna() & (ev.event != "gpu_time")].drop(columns=["gpu_mono_ns", "of"], errors="ignore")
    return ev, marks.merge(gpu, on=["stream", "pid", "gpu_ref"], how="left")


def gpu_edges(marks: pd.DataFrame) -> pd.DataFrame:
    """One row per burst edge: tag, rep, idx, kind (rise/fall), gpu_mono_ns, host_mono_ns."""
    out = []
    for _, r in marks.iterrows():
        if not str(r.event).startswith(("marker_burst", "sustain")):
            continue
        x = r.get("extra") or {}
        out.append({"tag": x.get("tag"), "rep": x.get("rep", -1), "idx": x.get("idx", -1),
                    "kind": "rise" if r.event.endswith("_start") else "fall",
                    "burst": r.event.rsplit("_", 1)[0], "gpu_mono_ns": r.gpu_mono_ns, "host_mono_ns": r.mono_ns})
    return pd.DataFrame(out)


def scope_time_map(polls: pd.DataFrame, fs: float):
    """raw sample index -> host monotonic s, via lower envelope of poll receipt latency."""
    end_idx = (polls.raw_index_first + polls.n_samples).to_numpy(float)
    mono = polls.mono_ns.to_numpy(float) / 1e9
    b, a = np.polyfit(end_idx / fs, mono, 1)
    resid = mono - (a + b * end_idx / fs)
    a_low = a + np.quantile(resid, 0.005)       # earliest receipts ~ zero transfer latency
    lat = resid - np.quantile(resid, 0.005)
    return (lambda idx: a_low + b * np.asarray(idx, float) / fs), {
        "slope": b, "receipt_latency_p50_ms": float(np.median(lat) * 1e3),
        "receipt_latency_p99_ms": float(np.quantile(lat, 0.99) * 1e3)}


def detect_edges(t: np.ndarray, x: np.ndarray, approx_t: np.ndarray, kinds, win_s: float, base_s: float = 0.15):
    """For each expected edge, find the threshold crossing nearest approx_t.

    Threshold = midpoint between the median level in [approx-win-base, approx-win]
    before and [approx+win, approx+win+base] after (for rises), inverted for falls.
    """
    res = []
    for at, k in zip(approx_t, kinds):
        m = (t > at - win_s) & (t < at + win_s)
        if m.sum() < 3:
            res.append(np.nan)
            continue
        tt, xx = t[m], x[m]
        pre = np.median(x[(t > at - win_s - base_s) & (t <= at - win_s * 0.5)]) if True else 0
        post = np.median(x[(t >= at + win_s * 0.5) & (t < at + win_s + base_s)])
        thr = (pre + post) / 2
        above = xx > thr if k == "rise" else xx < thr
        idx = np.flatnonzero(above & ~np.r_[False, above[:-1]])
        if not idx.size or abs(post - pre) < 1e-12:
            res.append(np.nan)
            continue
        j = idx[np.argmin(np.abs(tt[idx] - at))]
        # linear interpolation between j-1 and j
        if j > 0 and xx[j] != xx[j - 1]:
            f = (thr - xx[j - 1]) / (xx[j] - xx[j - 1])
            res.append(tt[j - 1] + f * (tt[j] - tt[j - 1]))
        else:
            res.append(tt[j])
    return np.array(res)
