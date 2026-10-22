#!/usr/bin/env python3
"""Long normal-operation recording: a timeline of workload segments inside one acquisition.

--timeline FILE.json: {"segments": [{"gap_before_s": 30, "procs": [{"kind": "serve", "argv": [...]}, ...]}, ...]}
Segments run in order. Each segment's processes start together (concurrent GPU
use), the segment ends when all of them exit, then the next gap (GPU idle) begins.
Workload processes get their own --metrics file and the shared events file.
kind: serve | train | nonml | cufft | render | nbody (interpreter chosen per kind).
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from h200_campaign.events import EventLog  # noqa: E402

MAIN_PY = str(Path.home() / "venvs" / "main" / "bin" / "python")
NONML_PY = str(Path.home() / "venvs" / "nonml" / "bin" / "python")
SCRIPTS = {"serve": "h200_campaign/serve.py", "train": "h200_campaign/train.py", "nonml": "h200_campaign/nonml.py",
           "cufft": "workloads/cufft_benchmark.py", "render": "workloads/rendering_proxy.py",
           "nbody": "workloads/nbody_sim.py"}
STOP = False


def _sig(_n, _f):
    global STOP
    STOP = True


def proc_env(kind, argv, env):
    env = dict(env)
    if kind == "nonml" and "qiskit" in argv:
        libs = glob.glob(str(Path(NONML_PY).parent.parent / "lib/python3.12/site-packages/qiskit_aer_gpu.libs/libgomp-*.so*"))
        if libs:
            env["LD_PRELOAD"] = libs[0]
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--timeline", required=True)
    ap.add_argument("--slack-s", type=float, default=300.0, help="per-segment allowance for loading beyond planned duration")
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--events", default=None)
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _sig)
    tl = json.loads(Path(args.timeline).read_text())
    log = EventLog(args.events, stream="timeline")
    mdir = Path(args.metrics).parent
    out = []
    for i, seg in enumerate(tl["segments"]):
        if STOP:
            break
        gap = float(seg.get("gap_before_s", 0))
        if gap:
            log.emit("gap_start", seg=i, planned_s=gap)
            t_end = time.monotonic() + gap
            while time.monotonic() < t_end and not STOP:
                time.sleep(0.2)
            log.emit("gap_end", seg=i)
        procs = []
        for j, p in enumerate(seg["procs"]):
            kind = p["kind"]
            py = NONML_PY if kind == "nonml" else MAIN_PY
            cmd = [py, str(REPO / SCRIPTS[kind])] + list(p["argv"])
            if kind in ("serve", "train", "nonml"):
                cmd += ["--metrics", str(mdir / f"seg{i:02d}_{j}_{kind}_metrics.json")]
                if args.events:
                    cmd += ["--events", args.events]
            e = proc_env(kind, p["argv"], os.environ)
            log.emit("segment_proc_launch", seg=i, proc=j, kind=kind, label=p.get("label"), cmd=cmd)
            procs.append((j, p, subprocess.Popen(cmd, cwd=REPO, env=e, start_new_session=True,
                                                 stdout=open(mdir / f"seg{i:02d}_{j}_{kind}.log", "w"),
                                                 stderr=subprocess.STDOUT)))
        deadline = time.monotonic() + float(seg.get("planned_s", 600)) + args.slack_s
        rcs = {}
        for j, p, pr in procs:
            try:
                rcs[j] = pr.wait(timeout=max(1.0, deadline - time.monotonic()))
            except subprocess.TimeoutExpired:
                os.killpg(pr.pid, signal.SIGTERM)
                try:
                    rcs[j] = pr.wait(timeout=60)
                except subprocess.TimeoutExpired:
                    os.killpg(pr.pid, signal.SIGKILL)
                    rcs[j] = pr.wait()
                log.emit("segment_proc_timeout", seg=i, proc=j, ok=False)
            log.emit("segment_proc_exit", seg=i, proc=j, rc=rcs[j], ok=rcs[j] == 0, label=p.get("label"))
        out.append({"seg": i, "rcs": rcs, "labels": [p.get("label") for _, p, _ in procs]})
    Path(args.metrics).write_text(json.dumps({"segments": out, "stopped_early": STOP,
                                              "all_ok": all(v == 0 for s in out for v in s["rcs"].values())}, indent=1))
    log.close()
    sys.exit(0 if all(v == 0 for s in out for v in s["rcs"].values()) else 3)


if __name__ == "__main__":
    main()
