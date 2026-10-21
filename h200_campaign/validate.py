#!/usr/bin/env python3
"""Validate one recording session directory: sensor health and clock alignment.

  python -m h200_campaign.validate SESSION_DIR [--prefix c]
Writes SESSION_DIR/validation.json and prints a summary.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from h200_campaign.align import detect_edges, gpu_edges, load_events, scope_time_map


def _parts(d: Path, prefix: str, kind: str) -> pd.DataFrame:
    files = sorted(d.glob(f"{prefix}_{kind}_[0-9][0-9][0-9][0-9].parquet"))
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True) if files else pd.DataFrame()


def _offsets(name, t, x, edges, win_s, ref_col="gpu_mono_ns"):
    approx = edges[ref_col].to_numpy(float) / 1e9
    found = detect_edges(t, x, approx, edges.kind.tolist(), win_s)
    off = found - approx
    df = edges.assign(sensor=name, found_s=found, offset_ms=off * 1e3)
    ok = np.isfinite(off)
    summ = {"edges": int(len(off)), "detected": int(ok.sum())}
    if ok.any():
        o = off[ok] * 1e3
        summ.update(offset_ms_median=float(np.median(o)), offset_ms_iqr=float(np.subtract(*np.percentile(o, [75, 25]))),
                    offset_ms_min=float(o.min()), offset_ms_max=float(o.max()))
        for tag in df.tag.dropna().unique():
            m = ok & (df.tag == tag).to_numpy()
            if m.any():
                summ[f"offset_ms_median_{tag}"] = float(np.median(off[m] * 1e3))
    return df, summ


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("session")
    ap.add_argument("--prefix", default="c")
    args = ap.parse_args()
    d, p = Path(args.session), args.prefix
    rep = {"session": str(d)}
    ev, marks = load_events(d / "events.jsonl")
    edges = gpu_edges(marks)
    rep["gpu_edges"] = int(len(edges))
    rep["gpu_edge_host_minus_gpu_ms"] = {
        "median": float(np.median((edges.host_mono_ns - edges.gpu_mono_ns) / 1e6)),
        "max_abs": float(np.max(np.abs(edges.host_mono_ns - edges.gpu_mono_ns) / 1e6))}
    marker_edges = edges[edges.burst == "marker_burst"]
    sustain_edges = edges[edges.burst == "sustain"]
    tables = []

    # ---- scope
    meta = json.loads((d / f"{p}_scope_meta.json").read_text())
    polls, env = _parts(d, p, "scope_polls"), _parts(d, p, "scope_env")
    fs = meta["fs_raw_hz"]
    tmap, tinfo = scope_time_map(polls, fs)
    env["t"] = tmap(20 * (env.dec_index_first + 488))
    full = meta["adc_limits"][1]
    rep["scope"] = {"counts": meta["counts"], "time_map": tinfo, "range": meta["range"],
                    "cpu_frac_core": (meta["cpu_user_s"] + meta["cpu_sys_s"]) / meta["elapsed_s"],
                    "bytes_written": meta["bytes_written"], "n_warnings": meta["n_warnings"]}
    phases = {"idle_pre": (env.t.min(), marker_edges.gpu_mono_ns.min() / 1e9 - 1)}
    if len(sustain_edges):
        s0 = sustain_edges[sustain_edges.kind == "rise"].gpu_mono_ns.iloc[0] / 1e9
        s1 = sustain_edges[sustain_edges.kind == "fall"].gpu_mono_ns.iloc[0] / 1e9
        phases["sustain"] = (s0, s1)
    for name, (a, b) in phases.items():
        m = (env.t >= a) & (env.t <= b)
        e = env[m]
        rep["scope"][f"{name}_peak_frac_full_scale"] = float(max(e.raw_count_max.max(), -e.raw_count_min.min()) / full)
        rep["scope"][f"{name}_rail_samples"] = int(e.rail_samples.sum())
        rep["scope"][f"{name}_v_rms_ac_median"] = float(e.v_rms_ac.median())
        rep["scope"][f"{name}_v_mean_median"] = float(e.v_mean.median())
    sig = env.v_rms_ac.to_numpy()
    for nm, ed in (("scope_markers", marker_edges), ("scope_sustain", sustain_edges)):
        if len(ed):
            df, s = _offsets(nm, env.t.to_numpy(), sig, ed, 0.12)
            rep[nm] = s
            tables.append(df)

    # ---- DCGM (100 ms), NVML (1 s), BMC (0.5 s polling of 5 s DCMI)
    for kind, col, tcol, win in (("dcgm", "dcgm_sm_active", "resp_mono_ns", 0.6),
                                 ("nvml", "power_draw_w", "resp_mono_ns", 3.0),
                                 ("nvml", "gpu_utilization_pct", "resp_mono_ns", 3.0),
                                 ("bmc", "bmc_power_w", "resp_mono_ns", 12.0),
                                 ("bmcsys", "bmc_sys_power_w", "resp_mono_ns", 12.0)):
        df = _parts(d, p, kind)
        if df.empty or col not in df:
            rep[f"{kind}_{col}"] = "missing"
            continue
        mf = d / f"{p}_{kind.replace('bmcsys', 'bmc')}_meta.json"
        km = json.loads(mf.read_text()) if mf.exists() else {}
        if tcol not in df and f"dcmi_{tcol}" in df:      # pre-split BMC format
            tcol = f"dcmi_{tcol}"
        t, x = df[tcol].to_numpy(float) / 1e9, df[col].to_numpy(float)
        dt = np.diff(t)
        info = {"rows": int(len(df)), "interval_s_median": float(np.median(dt)), "interval_s_max": float(dt.max()),
                "failures": km.get("failures"), "cpu_s": km.get("cpu_user_s", 0) + km.get("cpu_sys_s", 0)}
        # value-change cadence: how often the reported value actually changes
        ch = np.flatnonzero(np.diff(x) != 0)
        info["value_change_interval_s_median"] = float(np.median(np.diff(t[ch + 1]))) if ch.size > 2 else None
        for nm, ed in (("markers", marker_edges), ("sustain", sustain_edges)):
            if len(ed) and (nm == "sustain" or kind == "dcgm"):
                tdf, s = _offsets(f"{kind}_{col}_{nm}", t, x, ed, win)
                info[nm] = s
                tables.append(tdf)
        rep[f"{kind}_{col}"] = info

    # ---- health: intervals the monitor could not assess, with reasons (never dropped silently)
    bad = []
    span = (env.t.min(), env.t.max())
    binw = 977 / meta["decimation"]["fs_out_hz"]
    for t in env.t[env.rail_samples > 0]:
        bad.append(("scope", t - binw / 2, t + binw / 2, "adc_rail"))
    pt = tmap(polls.raw_index_first.to_numpy(float))
    pe = tmap((polls.raw_index_first + polls.n_samples).to_numpy(float))
    for a, b in zip(pt[polls.overrange.to_numpy() == 1], pe[polls.overrange.to_numpy() == 1]):
        bad.append(("scope", a, b, "driver_overrange_flag"))
    for a, b in zip(pt[polls.contiguity_break.to_numpy() == 1], pe[polls.contiguity_break.to_numpy() == 1]):
        bad.append(("scope", a, b, "contiguity_break"))
    for kind, tcol, nominal in (("nvml", "resp_mono_ns", 1.0), ("bmc", "resp_mono_ns", 0.5),
                                ("dcgm", "resp_mono_ns", 0.1)):
        df = _parts(d, p, kind)
        if df.empty:
            bad.append((kind, span[0], span[1], "sensor_missing"))
            continue
        if tcol not in df and f"dcmi_{tcol}" in df:
            tcol = f"dcmi_{tcol}"
        t = np.sort(df[tcol].to_numpy(float) / 1e9)
        for a, b in zip(t[:-1], t[1:]):
            if b - a > 2 * nominal:
                bad.append((kind, a, b, f"gap>{2 * nominal:g}s"))
        if t[0] - span[0] > 2 * nominal + 3:
            bad.append((kind, span[0], t[0], "late_start"))
        if span[1] - t[-1] > 2 * nominal + 3:
            bad.append((kind, t[-1], span[1], "early_stop"))
    hb = pd.DataFrame(bad, columns=["sensor", "start_mono_s", "end_mono_s", "reason"])
    hb.to_csv(d / "health_intervals.csv", index=False)
    rep["health"] = {"acquisition_s": float(span[1] - span[0]),
                     "unassessable_s_by_sensor": hb.assign(dur=hb.end_mono_s - hb.start_mono_s)
                     .groupby("sensor").dur.sum().round(4).to_dict(),
                     "intervals_by_reason": hb.reason.value_counts().to_dict()}

    if tables:
        pd.concat(tables, ignore_index=True).to_csv(d / "alignment_edges.csv", index=False)
    (d / "validation.json").write_text(json.dumps(rep, indent=1, default=str))
    print(json.dumps(rep, indent=1, default=str))


if __name__ == "__main__":
    main()
