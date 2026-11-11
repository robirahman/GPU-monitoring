#!/usr/bin/env python3
"""Brief training inside normal operation (recording type 4).

An inference server (serve.py) runs for the whole recording. Training bursts
(train.py, separate processes sharing the GPU) start at planned offsets. The
TRUE burst boundaries are the CUDA-event times of train.py's train_start and
train_end marks, not the launch times; loading happens before train_start and
is not training.

Burst placement: --bursts "DUR@START[,DUR@START...]" in seconds from the grid
origin. The grid origin is this script's 'sequence_grid_origin' event; the
detector's 10 s windows are [origin + 10k, origin + 10k + 10). START values are
the intended training-start times; because loading precedes training, each
burst process is launched --lead seconds early and then holds at train.py's
start gate (--start-at-mono) until the intended time. The achieved start (GPU
time of train_start) and any lateness (start_gate event) are recorded.

--bursts "" gives the inference-only control with the same serving schedule.
Serving continues for --observe seconds after the last planned burst end.
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
sys.path.insert(0, str(REPO))
from h200_campaign.events import EventLog  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--serve-model", default="phi3", choices=["phi3", "whisper", "wav2vec2", "bark", "dit", "effnet", "olmo", "pythia"])
    ap.add_argument("--serve-rate", type=float, default=1.0)
    ap.add_argument("--bursts", default="", help='e.g. "20@60,45@180"; empty = inference-only control')
    ap.add_argument("--burst-family", default="resnet")
    ap.add_argument("--burst-args", default="", help="extra train.py args, space separated")
    ap.add_argument("--lead", type=float, default=40.0,
                    help="launch this many s early; must exceed load + initial eval (the start gate does the timing)")
    ap.add_argument("--observe", type=float, default=180.0)
    ap.add_argument("--pre", type=float, default=60.0, help="min serving before first burst / total if control")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--metrics", required=True)
    ap.add_argument("--events", default=None)
    args = ap.parse_args()
    log = EventLog(args.events, stream="sequence")
    bursts = [(float(d), float(s)) for d, s in (b.split("@") for b in args.bursts.split(",") if b)]
    end_t = max([s + d for d, s in bursts], default=args.pre) + args.observe
    mdir = Path(args.metrics).parent
    env = os.environ.copy()

    # serving schedule: steady rate for the whole recording, plus 30 s of slack for model load
    serve_cmd = [PY, str(REPO / "h200_campaign" / "serve.py"), "--model", args.serve_model,
                 "--schedule", f"{end_t + 30:.0f}:{args.serve_rate}", "--seed", str(args.seed),
                 "--metrics", str(mdir / "serve_metrics.json")] + (["--events", args.events] if args.events else [])
    log.emit("serve_launch", cmd=serve_cmd)
    srv = subprocess.Popen(serve_cmd, cwd=REPO, env=env, stdout=open(mdir / "serve.log", "w"), stderr=subprocess.STDOUT)
    # wait until serving has started (serve_start event) before fixing the grid origin
    origin = None
    for _ in range(600):
        if args.events and Path(args.events).exists() and '"serve_start"' in Path(args.events).read_text():
            break
        if srv.poll() is not None:
            break
        time.sleep(0.5)
    origin = log.emit("sequence_grid_origin", window_s=10.0, bursts=bursts, observe=args.observe)["mono_ns"] / 1e9
    procs = []
    for i, (dur, start) in enumerate(bursts):
        launch_at = origin + start - args.lead
        while time.monotonic() < launch_at:
            time.sleep(0.05)
        cmd = [PY, str(REPO / "h200_campaign" / "train.py"), "--family", args.burst_family,
               "--duration", str(dur), "--seed", str(args.seed + i), "--eval-every-sec", "1e9",
               "--start-at-mono", f"{origin + start:.6f}",
               "--metrics", str(mdir / f"burst{i}_metrics.json")] + (["--events", args.events] if args.events else [])
        cmd += args.burst_args.split() if args.burst_args else []
        e = env | {"SPAR_BURST_INDEX": str(i)}
        log.emit("burst_launch", idx=i, planned_train_start_rel=start, planned_dur=dur, lead=args.lead, cmd=cmd)
        procs.append(subprocess.Popen(cmd, cwd=REPO, env=e, stdout=open(mdir / f"burst{i}.log", "w"),
                                      stderr=subprocess.STDOUT))
    for i, p in enumerate(procs):
        rc = p.wait()
        log.emit("burst_exit", idx=i, rc=rc, ok=rc == 0)
    while time.monotonic() < origin + end_t:
        time.sleep(0.5)
    log.emit("sequence_observation_end")
    srv.terminate()
    try:
        srv.wait(timeout=60)
    except subprocess.TimeoutExpired:
        srv.kill()
    m = {"bursts_planned": bursts, "burst_family": args.burst_family, "serve_model": args.serve_model,
         "serve_rate": args.serve_rate, "lead_s": args.lead, "observe_s": args.observe, "grid_origin_mono_s": origin,
         "window_s": 10.0, "control": not bursts, "seed": args.seed,
         "burst_rcs": [p.returncode for p in procs]}
    Path(args.metrics).write_text(json.dumps(m, indent=1))
    log.close()


if __name__ == "__main__":
    main()
