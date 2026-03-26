import re
import argparse
from datetime import datetime, timezone
import pandas as pd


TS_RE = re.compile(r"^(?P<ts>[\d\-:T\.]+)\+00:00 \| (?P<msg>.*)$")
PHASE_START_RE = re.compile(r"PHASE=(?P<phase>\w+) start")
PHASE_END_RE = re.compile(r"PHASE=(?P<phase>\w+) end")


def parse_ts(s: str) -> pd.Timestamp:
    # input is ISO with +00:00
    return pd.to_datetime(s, utc=True)


def parse_workload_log(path: str):
    phases = []
    current = None

    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            line = line.strip()
            m = TS_RE.match(line)
            if not m:
                continue
            ts = parse_ts(m.group("ts") + "+00:00") if not m.group("ts").endswith("+00:00") else parse_ts(m.group("ts"))
            msg = m.group("msg")

            ms = PHASE_START_RE.search(msg)
            if ms:
                phase = ms.group("phase")
                current = {"phase": phase, "start": ts, "end": None}
                phases.append(current)
                continue

            me = PHASE_END_RE.search(msg)
            if me and current and me.group("phase") == current["phase"] and current["end"] is None:
                current["end"] = ts
                current = None

    # drop any unclosed phases
    phases = [p for p in phases if p["end"] is not None]
    return phases


def label_row(ts: pd.Timestamp, phases):
    # default label
    label = "unknown"
    for p in phases:
        if p["start"] <= ts <= p["end"]:
            label = p["phase"]
            break
    return label


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--telemetry", required=True, help="nvml parquet/csv file")
    ap.add_argument("--workload_log", required=True, help="workload log created via tee")
    ap.add_argument("--out", default="", help="output labeled parquet")
    args = ap.parse_args()

    if args.telemetry.endswith(".csv"):
        df = pd.read_csv(args.telemetry)
    else:
        df = pd.read_parquet(args.telemetry)

    # parse timestamps
    df["ts_utc"] = pd.to_datetime(df["ts_utc"], utc=True)

    phases = parse_workload_log(args.workload_log)
    if not phases:
        raise SystemExit("No closed PHASE windows found in workload log. Did you run with tee?")

    if not args.out:
        args.out = args.telemetry.replace(".parquet", "_labeled.parquet").replace(".csv", "_labeled.parquet")

    df["label"] = df["ts_utc"].apply(lambda t: label_row(t, phases))

    # convenience: numeric “is_busy”
    df["is_busy"] = (df["label"].isin(["compute_bound", "mem_stream"])).astype(int)

    df.to_parquet(args.out, index=False)

    print("Wrote:", args.out)
    print("Phase windows:")
    for p in phases:
        print(f"  {p['phase']}: {p['start'].isoformat()} -> {p['end'].isoformat()}")
    print("\nLabel counts:")
    print(df["label"].value_counts().to_string())
    print("\nTop 5 util rows:")
    print(df.sort_values("gpu_util_pct", ascending=False).head(5)[["ts_utc","gpu_util_pct","power_w","sm_clock_mhz","mem_clock_mhz","label"]].to_string(index=False))


if __name__ == "__main__":
    main()
