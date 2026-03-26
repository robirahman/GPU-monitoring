"""
week5/scripts/check_data.py
Quick diagnostic: print all data sources and sample counts before running pipeline.
"""
import sys, glob, json
from pathlib import Path
sys.path.insert(0, "/root/xgb_pkg")
import pandas as pd

WEEK4 = Path(__file__).parent.parent.parent / "week4"   # Telemetry/week4/

print("=" * 60)
print("Week 5 — Data Source Diagnostics")
print("=" * 60)

# 1. week4/data/*.parquet
parquets = glob.glob(str(WEEK4 / "data" / "*.parquet"))
print(f"\n[1] week4/data/*.parquet  →  {len(parquets)} files")
if parquets:
    df = pd.concat([pd.read_parquet(p) for p in parquets], ignore_index=True)
    print(f"    Total rows: {len(df):,}")
    print(f"    Workload labels ({df['workload_label'].nunique()}):")
    for lbl, cnt in df["workload_label"].value_counts().items():
        print(f"      {lbl:45s}  {cnt:5d} rows")

# 2. DDP parquets
ddp = glob.glob(str(WEEK4 / "results" / "ddp" / "ddp_telemetry_gpu*.parquet"))
print(f"\n[2] week4/results/ddp/*.parquet  →  {len(ddp)} files")
if ddp:
    df2 = pd.concat([pd.read_parquet(p) for p in ddp], ignore_index=True)
    print(f"    Total rows: {len(df2):,}")
    print(f"    Columns: {list(df2.columns)}")

# 3. Dataset scale parquets
dscale = glob.glob(str(WEEK4 / "results" / "dataset_scale" / "*" / "telemetry.parquet"))
print(f"\n[3] dataset_scale telemetry  →  {len(dscale)} files")
if dscale:
    df3 = pd.concat([pd.read_parquet(p) for p in dscale], ignore_index=True)
    print(f"    Total rows: {len(df3):,}")
    print(f"    Columns: {list(df3.columns)[:10]}")

# 4. Edge case JSONs
ec_jsons = glob.glob(str(WEEK4 / "results" / "edge_cases" / "*_telemetry.json"))
print(f"\n[4] edge_case telemetry JSONs  →  {len(ec_jsons)} files")
for p in sorted(ec_jsons):
    with open(p) as f:
        d = json.load(f)
    name = Path(p).stem.replace("_telemetry", "")
    keys = list(d[0].keys()) if d else []
    print(f"    {name:30s}  {len(d):5d} records  keys: {keys}")

print("\n" + "=" * 60)
