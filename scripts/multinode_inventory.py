#!/usr/bin/env python3
"""What MULTI-NODE telemetry do we have, and how many independent workloads?

corpus_inventory.py answers "how much data, on what hardware". This answers the
different question the classifier actually cares about: how many INDEPENDENT
workloads sit on each side of the train / not-train boundary once the disguise
cells are set aside.

The distinction matters because the detector is held out by workload FAMILY, so
the effective sample size is the number of families, not the number of windows.
A sweep of seven context lengths over one vLLM cell is one family; twelve FSDP
rungs across three platforms are also one family. Counting parquets, runs or even
workloads flatters the corpus by an order of magnitude.

Reads parquet footers only (identity + timestamp columns), so it runs in ~1 GB
next to a windowing job.

  PYTHONPATH=. python3 scripts/multinode_inventory.py [--out inv_mn.parquet]

Also writes the stream -> host map that multinode_family_holdout.py and
mn_binary_lofo.py need to decide which runs genuinely spanned two machines.
"""
import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "classifier" / "rebuttal_analysis"))

from classifier.threeway_improved import threeway_label  # noqa: E402
from multinode_family_holdout import family  # noqa: E402

# _rep\d+$ is stripped along with the rank tokens: the TPU queue bakes the rep
# into the workload_label (tpuT16_train_fsdp1b_rep2) while the GPU queue does
# not, so without this the same TPU cell counts as two or three workloads and
# the TPU side of every table is silently inflated.
RANK = re.compile(r"_node\d+_gpu\d+|_gpu\d+|_w\d+c\d+|_rep\d+$")
MN = re.compile(r"^(?:smoke_)?(s16|tpuT16|tpuTMS4|tpuTMS)_")
ADV = re.compile(r"(advtrain|adversarial|whitebox|symgemm|typeconf|stalled|"
                 r"diluted|ddpstagger|ddpkaccum)", re.I)


def scan():
    rows = []
    for p in sorted((PROJECT_ROOT / "data").rglob("*.parquet")):
        if p.name.startswith("nic_"):
            continue                       # host-NIC sidecars, not GPU streams
        if any("_quarantine" in part for part in p.parts):
            continue
        if not MN.match(p.name):
            continue                       # single-node / legacy, not our question
        try:
            names = set(pq.ParquetFile(p).schema.names)
            tcol = ("ts" if "ts" in names else
                    "timestamp_epoch" if "timestamp_epoch" in names else None)
            if tcol is None:
                continue
            want = [c for c in (tcol, "gpu_name", "workload_label", "run_id",
                                "gpu_uuid", "hostname", "gpu_utilization_pct")
                    if c in names]
            d = pq.read_table(p, columns=want).to_pandas()
        except Exception:
            continue
        t = pd.to_numeric(d[tcol], errors="coerce").dropna()
        t = t[(t > 1e8) & (t < 4e9)]       # plausible epoch seconds only
        if len(t) < 2:
            continue
        lab = str(d["workload_label"].iloc[0]) if "workload_label" in d else p.stem
        rid = str(d["run_id"].iloc[0]) if "run_id" in d else p.stem
        uid = str(d["gpu_uuid"].iloc[0]) if "gpu_uuid" in d else p.stem
        gpu = str(d["gpu_name"].iloc[0]) if "gpu_name" in d else "?"
        host = str(d["hostname"].iloc[0]) if "hostname" in d else ""
        if not host or host in ("nan", "None"):
            # TPU parquets name the worker in the stem; NVIDIA multi-node ones
            # carry a real hostname. Either identifies the MACHINE.
            m = re.search(r"_w(\d+)c\d+", p.stem)
            host = f"w{m.group(1)}" if m else ""
        util = (float(d["gpu_utilization_pct"].median())
                if "gpu_utilization_pct" in d else np.nan)
        rows.append(dict(base=RANK.sub("", lab), run_id=rid,
                         stream=f"{rid}|{uid}", gpu=gpu, host=host,
                         rows=len(t), min=(t.max() - t.min()) / 60, util=util))
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/var/tmp/spar_analysis/inv_mn.parquet")
    a = ap.parse_args()

    d = scan()
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    d.to_parquet(a.out, index=False)
    # One stream can be written into more than one directory; the longest copy
    # wins, matching how load_all_data collapses them.
    d = d.sort_values("min", ascending=False).drop_duplicates("stream", keep="first")
    d["cls"] = d.base.map(threeway_label)
    d["kind"] = np.where(d.base.str.contains(ADV, na=False), "disguised", "honest")
    d["hw"] = d.gpu.str.replace("NVIDIA ", "", regex=False)
    d["fam"] = d.base.map(family)
    d["binary"] = np.where(d.cls == "ml_training", "TRAIN", "NOT-TRAIN")

    print(f"multi-node streams {len(d):,}  workloads {d.base.nunique()}  "
          f"runs {d.run_id.nunique():,}  stream-hours {d['min'].sum() / 60:,.0f}  "
          f"-> {a.out}")

    h = d[d.kind == "honest"]
    print("\n=== HONEST multi-node (disguise cells excluded) ===")
    print(h.groupby("cls").agg(workloads=("base", "nunique"),
                               families=("fam", "nunique"),
                               runs=("run_id", "nunique"),
                               streams=("stream", "size"),
                               hours=("min", lambda s: s.sum() / 60)).to_string())

    for side in ("TRAIN", "NOT-TRAIN"):
        s = h[h.binary == side]
        print(f"\n--- {side}: {s.base.nunique()} workloads -> "
              f"{s.fam.nunique()} FAMILIES ---")
        g = s.groupby("fam").agg(
            wl=("base", "nunique"), runs=("run_id", "nunique"),
            streams=("stream", "size"), hours=("min", lambda x: x.sum() / 60),
            hw=("hw", lambda x: "/".join(sorted(set(x)))))
        for k, r in g.sort_values("hours", ascending=False).iterrows():
            print(f"  {k:24s} {int(r.wl):2d} wkl {int(r.runs):4d} runs "
                  f"{int(r.streams):5d} str {r.hours:7.1f} h   {r.hw}")

    print("\n--- honest multi-node stream-hours, hardware x class ---")
    print(h.pivot_table(index="hw", columns="cls", values="min",
                        aggfunc=lambda s: s.sum() / 60).fillna(0).round(1).to_string())


if __name__ == "__main__":
    main()
