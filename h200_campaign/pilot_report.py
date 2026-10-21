#!/usr/bin/env python3
"""Summarise a pilot session: health, saturation, alignment, timing, attack cost, overhead.

  python -m h200_campaign.pilot_report SESSION_DIR > report.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def events(d: Path) -> pd.DataFrame:
    f = d / "events.jsonl"
    return pd.DataFrame([json.loads(l) for l in f.read_text().splitlines() if l.strip()]) if f.exists() else pd.DataFrame()


def gpu_time(ev: pd.DataFrame, name: str):
    m = ev[(ev.event == name) & ev.get("gpu_ref", pd.Series(dtype=float)).notna()]
    if m.empty:
        return None
    r = m.iloc[0]
    g = ev[(ev.event == "gpu_time") & (ev.pid == r.pid) & (ev.stream == r.stream) & (ev.gpu_ref == r.gpu_ref)]
    return int(g.gpu_mono_ns.iloc[0]) if len(g) else None


def host_time(ev, name):
    m = ev[ev.event == name]
    return int(m.mono_ns.iloc[0]) if len(m) else None


def acc_curve(metrics):
    key = "val_acc" if any("val_acc" in e for e in metrics["evals"]) else "val_loss"
    return key, [(e["t"], e["updates"], e.get(key)) for e in metrics["evals"]]


def time_to(curve, target, higher=True):
    for t, u, v in curve:
        if v is not None and ((v >= target) if higher else (v <= target)):
            return t, u
    return None, None


def main():
    s = Path(sys.argv[1])
    out = {"session": s.name, "runs": []}
    metrics = {}
    for d in sorted(p for p in s.iterdir() if p.is_dir()):
        man = json.loads((d / "manifest.json").read_text()) if (d / "manifest.json").exists() else {"label": d.name}
        r = {k: man.get(k) for k in ("label", "run_id", "class_label", "family", "ok", "workload_rc", "failure",
                                     "workload_wall_s", "sensors_on", "bytes_on_disk")}
        ev = events(d)
        if not ev.empty:
            ls, le = host_time(ev, "load_start"), host_time(ev, "load_end")
            r["load_s"] = (le - ls) / 1e9 if ls and le else None
            ts, fu = gpu_time(ev, "train_start"), gpu_time(ev, "first_update_submitted")
            if ts and fu:
                r["gpu_train_start_to_first_update_end_s"] = (fu - ts) / 1e9
                r["host_train_start_vs_gpu_ms"] = (host_time(ev, "train_start") - ts) / 1e6
        v = d / "validation.json"
        if v.exists():
            vj = json.loads(v.read_text())
            sc = vj.get("scope", {})
            r["scope"] = {k: sc.get(k) for k in ("range", "cpu_frac_core", "bytes_written")} | \
                         {k: sc["counts"].get(k) for k in ("raw_samples", "contiguity_breaks", "overrange_polls", "rail_samples")} | \
                         {"clock_slope_ppm": (sc["time_map"]["slope"] - 1) * 1e6}
            env = pd.concat([pd.read_parquet(f) for f in sorted(d.glob("c_scope_env_0*.parquet"))])
            full = json.loads((d / "c_scope_meta.json").read_text())["adc_limits"][1]
            r["scope"]["peak_frac_full_scale"] = float(max(env.raw_count_max.max(), -env.raw_count_min.min()) / full)
            r["scope"]["peak_frac_q99_99"] = float(np.quantile(np.maximum(env.raw_count_max, -env.raw_count_min), 0.9999) / full)
            r["alignment_ms"] = {k: {kk: vj[k].get(kk) for kk in ("detected", "offset_ms_median", "offset_ms_iqr",
                                                                  "offset_ms_median_start", "offset_ms_median_end")}
                                 for k in ("scope_markers",) if isinstance(vj.get(k), dict)}
            dc = vj.get("dcgm_dcgm_sm_active", {})
            if isinstance(dc, dict) and "markers" in dc:
                r["alignment_ms"]["dcgm_markers"] = dc["markers"]
            r["health"] = vj.get("health")
            cost = {}
            for k in ("nvml", "bmc", "dcgm"):
                mf = d / f"c_{k}_meta.json"
                if mf.exists():
                    mm = json.loads(mf.read_text())
                    cost[k] = {"cpu_frac_core": (mm["cpu_user_s"] + mm["cpu_sys_s"]) / mm["elapsed_s"],
                               "bytes": mm["bytes_written"], "failures": mm.get("failures")}
            cost["scope"] = {"cpu_frac_core": sc.get("cpu_frac_core"), "bytes": sc.get("bytes_written")}
            r["monitor_cost"] = cost
        mf = d / "metrics.json"
        if mf.exists():
            m = json.loads(mf.read_text())
            metrics[man.get("label")] = m
            r["useful"] = {k: m.get(k) for k in ("updates", "useful_units", "forward_only_units", "useful_units_per_s",
                                                 "completed", "planned_requests", "wer", "service_ms_p50")}
            if "evals" in m:
                r["useful"]["final_eval"] = m["evals"][-1]
        out["runs"].append(r)

    # attack vs matched ordinary training
    o, a = metrics.get("resnet18_cifar10_ord"), metrics.get("resnet18_cifar10_dil5")
    if o and a:
        _, co = acc_curve(o)
        _, ca = acc_curve(a)
        cmp = {"ordinary_final": o["evals"][-1], "attack_final": a["evals"][-1],
               "useful_samples_per_s": {"ordinary": o["useful_units_per_s"], "attack": a["useful_units_per_s"]}}
        for tgt in (0.80, 0.85, 0.88, 0.90):
            to, uo = time_to(co, tgt)
            ta, ua = time_to(ca, tgt)
            cmp[f"time_to_acc_{tgt}"] = {"ordinary_s": to, "attack_s": ta,
                                         "slowdown": (ta / to) if (to and ta) else None,
                                         "ordinary_updates": uo, "attack_updates": ua}
        out["attack_vs_ordinary"] = cmp
    # instrumentation overhead
    n = metrics.get("resnet18_cifar10_ord_noinstr")
    if o and n:
        out["instrumentation_overhead"] = {
            "useful_samples_per_s_instrumented": o["useful_units_per_s"],
            "useful_samples_per_s_uninstrumented": n["useful_units_per_s"],
            "slowdown_frac": 1 - o["useful_units_per_s"] / n["useful_units_per_s"],
            "note": "single pair; instrumented = all sensors + per-update CUDA-event marks"}
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
