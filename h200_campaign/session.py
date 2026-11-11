#!/usr/bin/env python3
"""Run a collection session: a JSON plan of runs, each a separately validated acquisition.

Per run: start scope / NVML / BMC / DCGM loggers -> wait for scope ready ->
start markers -> workload -> end markers -> stop loggers -> validate -> manifest.
Nothing is deleted: failed runs keep their files and are marked in the manifest
and in the session ledger (SESSION_DIR/ledger.jsonl).

Plan JSON:
{"session_id": "...", "partition": "pilot|calibration|test", "notes": "...",
 "sensors": {"scope_range": "V10", "scope_raw": "all", "dcgm": true},
 "runs": [{"label": "bert_sst2_ord", "class_label": "ml_training", "family": "bert",
           "kind": "train|serve|nonml|idle", "argv": ["--family", "bert", ...],
           "job_id": "optional, shared by related variants/repetitions", "timeout_s": 900}, ...]}

  python -m h200_campaign.session PLAN.json --root ~/campaign/sessions [--only LABEL] [--no-sensors]
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import shutil
import signal
import subprocess
import sys
import time
import uuid
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
PY = sys.executable
WORKLOADS = {
    "train": [PY, str(REPO / "h200_campaign" / "train.py")],
    "serve": [PY, str(REPO / "h200_campaign" / "serve.py")],
    "sequence": [PY, str(REPO / "h200_campaign" / "sequence.py")],
    "cufft": [PY, str(REPO / "workloads" / "cufft_benchmark.py")],
    "render": [PY, str(REPO / "workloads" / "rendering_proxy.py")],
    "nbody": [PY, str(REPO / "workloads" / "nbody_sim.py")],
    "idle": [PY, str(REPO / "workloads" / "idle.py")],
    "timeline": [PY, str(REPO / "h200_campaign" / "timeline.py")],
    "nonml": [str(Path.home() / "venvs" / "nonml" / "bin" / "python"), str(REPO / "h200_campaign" / "nonml.py")],
}
FROZEN = REPO / "h200_campaign" / "frozen_settings.json"
MIN_FREE_GB = 40
STOP = False


def _sig(_n, _f):
    global STOP
    STOP = True


def git_state():
    def g(*a):
        return subprocess.run(["git", "-C", str(REPO), *a], capture_output=True, text=True).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty_files": [l for l in g("status", "--porcelain", "--", "h200_campaign").splitlines()]}


def start_sensors(d: Path, prefix: str, cfg: dict):
    procs = {}
    env = os.environ.copy()
    procs["scope"] = subprocess.Popen(
        [PY, str(REPO / "h200_campaign" / "scope_logger.py"), "--out-dir", str(d), "--prefix", prefix,
         "--range", cfg.get("scope_range", "V10"), "--raw", cfg.get("scope_raw", "all"),
         "--hires", cfg.get("scope_hires", "none"), "--ready-file", str(d / "scope_ready.json")],
        stdout=open(d / "scope.log", "w"), stderr=subprocess.STDOUT, env=env)
    kinds = ["nvml", "bmc"] + (["dcgm"] if cfg.get("dcgm", True) else [])
    for k in kinds:
        procs[k] = subprocess.Popen([PY, "-m", "h200_campaign.sensors", k, "--out-dir", str(d), "--prefix", prefix],
                                    stdout=open(d / f"{k}.log", "w"), stderr=subprocess.STDOUT, cwd=REPO, env=env)
    for _ in range(120):
        if (d / "scope_ready.json").exists():
            break
        if procs["scope"].poll() is not None:
            break
        time.sleep(0.25)
    return procs


def stop_sensors(procs):
    for p in procs.values():
        if p.poll() is None:
            p.send_signal(signal.SIGTERM)
    rc = {}
    for k, p in procs.items():
        try:
            rc[k] = p.wait(timeout=120)
        except subprocess.TimeoutExpired:
            p.kill()
            rc[k] = "killed"
    return rc


def _call(cmd, env, timeout):
    """subprocess.call with a timeout, so a hung helper cannot stall the campaign."""
    p = subprocess.Popen(cmd, cwd=REPO, env=env, start_new_session=True)
    try:
        return p.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        os.killpg(p.pid, signal.SIGKILL)
        p.wait()
        return "timeout"


def emit(path: Path, run_id, job_id, event, **extra):
    rec = {"schema": 1, "run_id": run_id, "job_id": job_id, "attempt_id": "1", "stream": "orchestrator",
           "pid": os.getpid(), "event": event, "wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns(),
           "source": "session.py", "ok": extra.pop("ok", True)}
    if extra:
        rec["extra"] = extra
    with open(path, "a") as f:
        f.write(json.dumps(rec, default=str) + "\n")


def clear_stray_gpu_processes():
    """Kill leftover campaign processes still holding the GPU (e.g. orphans of a crashed run); report them."""
    out = subprocess.run(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.split()
    killed = []
    for pid in out:
        try:
            cmd = Path(f"/proc/{pid}/cmdline").read_bytes().replace(b"\0", b" ").decode(errors="replace")
        except OSError:
            continue
        if "gpu-monitoring" in cmd or "h200_campaign" in cmd:
            try:
                os.kill(int(pid), signal.SIGKILL)
                killed.append({"pid": int(pid), "cmd": cmd[:300]})
            except OSError:
                pass
    if killed:
        time.sleep(5)
    return killed


def run_one(spec: dict, plan: dict, root: Path, sensors_on: bool):
    run_id = uuid.uuid4().hex[:12]
    job_id = spec.get("job_id") or f"{plan['session_id']}-{spec['label']}"
    d = root / plan["session_id"] / f"{spec['label']}_{run_id}"
    d.mkdir(parents=True, exist_ok=True)
    ev = d / "events.jsonl"
    env = os.environ | {"SPAR_RUN_ID": run_id, "SPAR_JOB_ID": job_id, "SPAR_ATTEMPT_ID": "1",
                        "SPAR_CLASS_LABEL": spec["class_label"], "H200_EVENTS_PATH": str(ev),
                        "PYTHONPATH": str(REPO)}
    if spec["kind"] == "nonml" and "qiskit" in spec.get("argv", []):
        libs = glob.glob(str(Path.home() / "venvs/nonml/lib/python3.12/site-packages/qiskit_aer_gpu.libs/libgomp-*.so*"))
        if libs:
            env["LD_PRELOAD"] = libs[0]
    cfg = dict(plan.get("sensors", {})) | spec.get("sensor_overrides", {})
    man = {"run_id": run_id, "job_id": job_id, "session_id": plan["session_id"], "partition": plan["partition"],
           "label": spec["label"], "class_label": spec["class_label"], "family": spec.get("family"),
           "kind": spec["kind"], "argv": spec.get("argv", []), "sensors_on": sensors_on,
           "sensor_cfg": cfg, "git": git_state(), "host": os.uname().nodename, "attempt": spec.get("_attempt", 1),
           "frozen_settings_sha256": hashlib.sha256(FROZEN.read_bytes()).hexdigest() if FROZEN.exists() else None,
           "block": plan.get("block"), "day": plan.get("day"), "collection_type": spec.get("collection_type"),
           "attack": spec.get("attack"), "extra": spec.get("extra"),
           "started_wall_ns": time.time_ns()}
    stray = clear_stray_gpu_processes()
    if stray:
        man["stray_gpu_processes_killed"] = stray
    emit(ev, run_id, job_id, "run_start", label=spec["label"], stray_killed=len(stray))
    procs = start_sensors(d, "c", cfg) if sensors_on else {}
    if sensors_on and not (d / "scope_ready.json").exists():
        man["failure"] = "scope logger did not start"
    marker = [PY, "-m", "h200_campaign.markers", "--events", str(ev)]
    if sensors_on:
        time.sleep(3)
        man["marker_start_rc"] = _call(marker + ["--tag", "start"], env, 300)
        time.sleep(3)
    cmd = list(WORKLOADS[spec.get("workload", spec["kind"])]) + spec.get("argv", [])
    if spec["kind"] == "timeline":
        (d / "timeline.json").write_text(json.dumps(spec["timeline"], indent=1))
        cmd += ["--timeline", str(d / "timeline.json")]
    if spec["kind"] in ("train", "serve", "sequence", "timeline", "nonml"):
        cmd += ["--metrics", str(d / "metrics.json"), "--events", str(ev)]
    emit(ev, run_id, job_id, "process_launch", cmd=cmd)
    t0 = time.monotonic()
    with open(d / "workload.log", "w") as wl:
        p = subprocess.Popen(cmd, cwd=REPO, env=env, stdout=wl, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            rc = p.wait(timeout=spec.get("timeout_s", 3600))
        except subprocess.TimeoutExpired:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                rc = p.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
                rc = p.wait()
            man["failure"] = "workload timeout"
    emit(ev, run_id, job_id, "process_exit_observed", rc=rc, ok=rc == 0)
    man["workload_rc"] = rc
    man["workload_wall_s"] = time.monotonic() - t0
    if sensors_on:
        time.sleep(3)
        man["marker_end_rc"] = _call(marker + ["--tag", "end"], env, 300)
        time.sleep(3)
        man["sensor_rc"] = stop_sensors(procs)
        try:
            v = subprocess.run([PY, "-m", "h200_campaign.validate", str(d)], cwd=REPO, env=env,
                               capture_output=True, text=True, timeout=3600)
            man["validate_rc"] = v.returncode
            if v.returncode != 0:
                (d / "validate.err").write_text(v.stderr[-20000:])
        except subprocess.TimeoutExpired:
            man["validate_rc"] = "timeout"
    emit(ev, run_id, job_id, "run_end")
    man["ended_wall_ns"] = time.time_ns()
    man["ok"] = (rc == 0 and "failure" not in man and (not sensors_on or man.get("validate_rc") == 0))
    man["bytes_on_disk"] = sum(f.stat().st_size for f in d.iterdir() if f.is_file())
    (d / "manifest.json").write_text(json.dumps(man, indent=1, default=str))
    with open(root / plan["session_id"] / "ledger.jsonl", "a") as f:
        f.write(json.dumps({k: man.get(k) for k in ("run_id", "label", "attempt", "ok", "workload_rc", "failure",
                                                     "workload_wall_s", "bytes_on_disk")} | {"dir": str(d)}) + "\n")
    return man


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("plan")
    ap.add_argument("--root", default=str(Path.home() / "campaign" / "sessions"))
    ap.add_argument("--only", default=None)
    ap.add_argument("--no-sensors", action="store_true")
    args = ap.parse_args()
    signal.signal(signal.SIGTERM, _sig)
    plan = json.loads(Path(args.plan).read_text())
    root = Path(args.root)
    (root / plan["session_id"]).mkdir(parents=True, exist_ok=True)
    (root / plan["session_id"] / "plan.json").write_text(json.dumps(plan, indent=1))
    ledger = root / plan["session_id"] / "ledger.jsonl"
    for spec in plan["runs"]:
        if STOP:
            break
        if args.only and spec["label"] != args.only:
            continue
        # resume: skip labels already done; a failed run is retried once (its folder is kept)
        prior = [json.loads(l) for l in ledger.read_text().splitlines()] if ledger.exists() else []
        prior = [r for r in prior if r["label"] == spec["label"]]
        if any(r.get("ok") for r in prior) or len(prior) >= 2:
            continue
        free_gb = shutil.disk_usage(root).free / 2**30
        if free_gb < MIN_FREE_GB:
            print(json.dumps({"stopped": "disk", "free_gb": free_gb}), flush=True)
            sys.exit(5)
        spec = dict(spec, _attempt=len(prior) + 1)
        m = run_one(spec, plan, root, not args.no_sensors and spec.get("sensors", True))
        print(json.dumps({k: m.get(k) for k in ("label", "run_id", "attempt", "ok", "workload_rc", "failure",
                                                "workload_wall_s")}), flush=True)
        if not m.get("ok") and m["attempt"] < 2 and not STOP:
            m = run_one(dict(spec, _attempt=2), plan, root, not args.no_sensors and spec.get("sensors", True))
            print(json.dumps({k: m.get(k) for k in ("label", "run_id", "attempt", "ok", "workload_rc", "failure",
                                                    "workload_wall_s")}), flush=True)
    print("SESSION_DONE", flush=True)


if __name__ == "__main__":
    main()
