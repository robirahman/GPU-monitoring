#!/usr/bin/env python3
"""Progress of the main campaign: runs done / failed per partition and type, hours recorded, disk.

  python -m h200_campaign.campaign_status [--plans h200_campaign/plans_main]
"""
from __future__ import annotations

import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plans", default=str(REPO / "h200_campaign" / "plans_main"))
    ap.add_argument("--sessions", default=str(Path.home() / "campaign" / "sessions"))
    args = ap.parse_args()
    plans, sess = Path(args.plans), Path(args.sessions)
    summ = json.loads((plans / "campaign_summary.json").read_text())
    tot = defaultdict(lambda: defaultdict(lambda: {"planned": 0, "ok": 0, "failed_final": 0, "hours_ok": 0.0}))
    failures = []
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
            if any(r.get("ok") for r in rs):
                t["ok"] += 1
                t["hours_ok"] += max(r.get("workload_wall_s") or 0 for r in rs if r.get("ok")) / 3600
            elif len(rs) >= 2:
                t["failed_final"] += 1
                failures.append({"session": plan["session_id"], "label": run["label"],
                                 "failure": rs[-1].get("failure"), "rc": rs[-1].get("workload_rc")})
    du = shutil.disk_usage(sess)
    st = Path.home() / "campaign" / "driver_status.json"
    out = {"driver": json.loads(st.read_text()) if st.exists() else None,
           "by_partition": {p: {k: {kk: round(vv, 2) if isinstance(vv, float) else vv for kk, vv in v.items()}
                                for k, v in d.items()} for p, d in tot.items()},
           "failed_after_retry": failures, "disk_free_gb": round(du.free / 2**30, 1)}
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
