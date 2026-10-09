"""On-node reference set with all three sensor tiers (NVML + scope + DCGM).

The existing corpus has NVML always and scope for the H200 campaign, but no
DCGM. To fairly ablate the sensor tiers against the shaping evader, we need a
node-local set where BOTH classes (training and rest) carry all three sensors.
This collects that reference set via the registry workloads, using the same
triple-sensor wrapper as the shaping sweep (--dcgm), so the sensor-ablation eval
trains and tests on identical runs and just varies the feature columns.

Positives (ml_training): pytorch_resnet_cifar10, gpt2_wikitext2
Negatives (rest): resnet50_inference (inference), cufft_benchmark, nbody_sim
                  (non-ML/HPC), idle
The unshaped llm.c baseline_train (from the shaping sweep) is an extra positive;
the shaped family is the held-out evader.

Usage: ~/pico-venv/bin/python scripts/run_reference_dcgm.py --reps 3
"""
import argparse
import glob
import json
import os
import subprocess
import sys
import time
import uuid

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = "data/h200_nvl_scope/reference"
PROGRESS = os.path.join(_REPO_ROOT, OUT_DIR, "reference_progress.json")
MAIN_PY = os.path.expanduser("~/pico-venv/bin/python")

CLASSES = [
    "pytorch_resnet_cifar10", "gpt2_wikitext2",   # training (positive)
    "resnet50_inference",                          # inference (rest)
    "cufft_benchmark", "nbody_sim",                # non-ML/HPC (rest)
    "idle",                                        # idle (rest)
]


def load_progress():
    return json.load(open(PROGRESS)) if os.path.exists(PROGRESS) else {}


def validate(label, run_id):
    import pandas as pd
    d = os.path.join(_REPO_ROOT, OUT_DIR)
    tel = [f for f in glob.glob(os.path.join(d, f"{label}_*_{run_id[:8]}_*.parquet"))
           if "_scope" not in os.path.basename(f) and "_dcgm" not in os.path.basename(f)]
    sco = glob.glob(os.path.join(d, f"{label}_scope_{run_id}_*.parquet"))
    dcg = glob.glob(os.path.join(d, f"{label}_dcgm_{run_id}_*.parquet"))
    if not (tel and sco and dcg):
        return False, f"missing sensor file(s): tel={bool(tel)} scope={bool(sco)} dcgm={bool(dcg)}"
    t = pd.read_parquet(tel[0])
    if len(t) < 15:
        return False, f"telemetry too short ({len(t)} rows)"
    g = pd.read_parquet(dcg[0])
    dram = float(g["dcgm_dram_active"].fillna(0).mean())
    return True, f"{len(t)} tel rows, dram {dram:.3f}, util {t.gpu_utilization_pct.mean():.0f}%"


def run_rep(label, rep, attempt):
    run_id = uuid.uuid4().hex[:12]
    cmd = [MAIN_PY, os.path.join(_REPO_ROOT, "scripts", "run_workload_with_scope.py"),
           "--workload", label, "--gpu-index", "0", "--output-dir", OUT_DIR,
           "--run-id", run_id, "--no-dcgm", "--dcgm"]
    log_dir = os.path.join(_REPO_ROOT, OUT_DIR, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{label}_rep{rep}_a{attempt}_{run_id[:8]}.log")
    print(f"\n=== {label} rep {rep} attempt {attempt} (run_id {run_id}) ===", flush=True)
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=_REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        tail = subprocess.run(["tail", "-4", log_path], capture_output=True, text=True).stdout
        return False, f"exit {rc}; {tail.strip()[-250:]}"
    return validate(label, run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--reset", action="store_true")
    args = ap.parse_args()

    classes = [c for c in CLASSES if not args.only or c in args.only.split(",")]
    os.makedirs(os.path.join(_REPO_ROOT, OUT_DIR), exist_ok=True)
    if args.reset and os.path.exists(PROGRESS):
        os.remove(PROGRESS)
    prog = load_progress()

    todo = [(c, rep) for c in classes for rep in range(1, args.reps + 1)
            if prog.get(f"{c}:{rep}", {}).get("status") != "done"]
    print(f"reference set: {len(classes)} classes, {len(todo)} reps", flush=True)
    failed = []
    for label, rep in todo:
        key = f"{label}:{rep}"
        ok, msg = False, ""
        for attempt in range(1, args.retries + 2):
            ok, msg = run_rep(label, rep, attempt)
            print(f"[ref] {key} attempt {attempt}: {'OK' if ok else 'FAIL'} — {msg}", flush=True)
            if ok:
                break
            time.sleep(15)
        prog[key] = {"status": "done" if ok else "failed", "detail": msg, "ts": time.time()}
        json.dump(prog, open(PROGRESS, "w"), indent=1)
        if not ok:
            failed.append(key)
        time.sleep(8)

    done = sum(1 for v in prog.values() if v.get("status") == "done")
    print(f"\n===== reference summary =====\ndone: {done}  failed: {len(failed)}", flush=True)
    for k in failed:
        print(f"  FAILED {k}: {prog[k]['detail']}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
