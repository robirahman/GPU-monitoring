#!/usr/bin/env python3
"""Progress of the main campaign: runs done / failed per partition and type, hours recorded, disk.

  python -m h200_campaign.campaign_status [--plans h200_campaign/plans_main]
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def est_hours(x):
    from h200_campaign.make_campaign import est_hours as e
    return e(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default=str(REPO / "h200_campaign" / "plans_main"))
    ap.add_argument("--sessions", default=str(Path.home() / "campaign" / "sessions"))
    args = ap.parse_args()
    plans, sess = Path(args.plans), Path(args.sessions)
    summ = json.loads((plans / "campaign_summary.json").read_text())
    tot = defaultdict(lambda: defaultdict(lambda: {"planned": 0, "ok": 0, "failed_final": 0, "hours_ok": 0.0}))
    failures = []
    remaining_h, done_planned_h, done_actual_h, first_start = 0.0, 0.0, 0.0, None
    for f in summ["order"]:
        plan = json.loads((plans / f).read_text())
        led = sess / plan["session_id"] / "ledger.jsonl"
        rows = [json.loads(l) for l in led.read_text().splitlines()] if led.exists() else []
        by = defaultdict(list)
        for r in rows:
            by[r["label"]].append(r)
        for run in plan["runs"]:
            t = tot[plan["partition"]][run["collection_type"]]
            t["planned"] += 1
            rs = by.get(run["label"], [])
            for r in rs:
                d = Path(r.get("dir", ""))
                mf = d / "manifest.json"
                if mf.exists():
                    m = json.loads(mf.read_text())
                    st = m.get("started_wall_ns")
                    first_start = st if first_start is None or (st and st < first_start) else first_start
            finished = any(r.get("ok") for r in rs) or len(rs) >= 2
            if finished:
                done_planned_h += est_hours(run)
            else:
                remaining_h += est_hours(run)
            if any(r.get("ok") for r in rs):
                t["ok"] += 1
                t["hours_ok"] += max(r.get("workload_wall_s") or 0 for r in rs if r.get("ok")) / 3600
            elif len(rs) >= 2:
                t["failed_final"] += 1
                failures.append({"session": plan["session_id"], "label": run["label"],
                                 "failure": rs[-1].get("failure"), "rc": rs[-1].get("workload_rc")})
    du = shutil.disk_usage(sess)
    st = Path.home() / "campaign" / "driver_status.json"
    elapsed_h = (time.time_ns() - first_start) / 3.6e12 if first_start else 0.0
    pace = elapsed_h / done_planned_h if done_planned_h > 0.5 else 1.05     # wall hours per planned hour, incl. overhead
    eta_h = remaining_h * pace
    out = {"driver": json.loads(st.read_text()) if st.exists() else None,
           "elapsed_h": round(elapsed_h, 2), "planned_hours_done": round(done_planned_h, 2),
           "planned_hours_remaining": round(remaining_h, 2), "pace_wall_per_planned": round(pace, 3),
           "eta_hours": round(eta_h, 1),
           "eta_utc": time.strftime("%Y-%m-%d %H:%M UTC", time.gmtime(time.time() + eta_h * 3600)),
           "by_partition": {p: {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                for k, v in d.items()} for p, d in tot.items()},
           "failed_after_retry": failures, "disk_free_gb": round(du.free / 2**30, 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
