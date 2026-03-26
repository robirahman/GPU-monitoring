#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
import pandas as pd

RAW = Path("data/raw_telemetry")
DATASETS = Path("data/datasets")
REPORTS = Path("data/models/reports")

def sh(cmd: str):
    print(f"$ {cmd}", flush=True)
    subprocess.check_call(cmd, shell=True)

def latest_features():
    feats = sorted(RAW.glob("*_features.parquet"))
    return feats[-1] if feats else None

def collect_metadata(extra: dict):
    meta = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "host": os.uname().nodename,
        "cwd": str(Path.cwd()),
    }
    meta.update(extra)
    return meta

def run_scenario(name: str, workload: str, seconds: int):
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    env = os.environ.copy()
    env["WORKLOAD"] = workload
    env["SECONDS"] = str(seconds)

    # run harness script
    sh(f"WORKLOAD={workload} SECONDS={seconds} ./scripts/run_harness.sh")

    feat = latest_features()
    if feat is None:
        raise RuntimeError("No features file produced")
    return feat, stamp

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", type=int, default=4, help="how many scenarios to run")
    ap.add_argument("--out_dataset", default="data/datasets/master.parquet")
    args = ap.parse_args()

    DATASETS.mkdir(parents=True, exist_ok=True)
    REPORTS.mkdir(parents=True, exist_ok=True)

    # Simple scenario list (expand this as you go)
    scenarios = [
        {"name":"ml_default", "workload":"ml", "seconds":170},
        {"name":"nonml_memcpy", "workload":"nonml", "seconds":110},
    ] * max(1, args.runs // 2)

    rows = []
    for sc in scenarios[:args.runs]:
        feat_path, stamp = run_scenario(sc["name"], sc["workload"], sc["seconds"])
        df = pd.read_parquet(feat_path)
        df["scenario"] = sc["name"]
        df["workload_family"] = sc["workload"]      # ml/nonml
        df["source_file"] = feat_path.name
        rows.append(df)

        meta = collect_metadata({"scenario": sc, "features_file": feat_path.name})
        with open(REPORTS / f"run_{stamp}.json", "w") as f:
            json.dump(meta, f, indent=2)

    master = pd.concat(rows, ignore_index=True)
    master.to_parquet(args.out_dataset, index=False)
    print("Wrote dataset:", args.out_dataset)
    print(master["label"].value_counts().to_string())

if __name__ == "__main__":
    main()
