"""Scale-bound sweep: does memory-minimization evasion break at model scale?

Collects memory-minimized training across a model-size ladder (triple-sensor,
25 s raw scope snippet), to find the crossover model size above which a training
run's resident footprint cannot drop below the benign inference/HPC band
(idle 0.6 / cuFFT 2.0 / resnet50-inference 4.8 GB from the reference set).

- Full fine-tuning (optimizer-state-floored: fp16 w + fp32 master + Adam m + v):
  whitebox_diluted_N5 = TinyLlama-1.1B full AdamW, grad-ckpt, batch 1 -> ~13 GB,
  irreducible. The decisive "cannot hide" case.
- LoRA (weights-floored, ~= matched inference footprint): qwen 1.5B (~3 GB, in
  band), qwen 3B (~6 GB), mistral 7B (~14 GB).
- whitebox_inference: TinyLlama forward-only, the same-weight-scale inference
  reference.

Usage: ~/pico-venv/bin/python scripts/run_scale_sweep.py --reps 3
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
OUT_DIR = "data/h200_nvl_scope/scale"
PROGRESS = os.path.join(_REPO_ROOT, OUT_DIR, "scale_progress.json")
MAIN_PY = os.path.expanduser("~/pico-venv/bin/python")

CLASSES = [
    "whitebox_diluted_N5",          # full-FT TinyLlama-1.1B (optimizer-floored)
    "whitebox_inference",           # TinyLlama inference (same weight scale, ref)
    "llm_train_qwen25_1p5b_lora",   # LoRA ~3 GB (in benign band)
    "llm_train_qwen25_3b_lora",     # LoRA ~6 GB
    "llm_train_mistral_7b_lora",    # LoRA ~14 GB (cannot hide)
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
    return True, (f"{len(t)} tel rows, mem_used {t.mem_used_mb.mean():.0f} MB, "
                  f"util {t.gpu_utilization_pct.mean():.0f}%")


def run_rep(label, rep, attempt):
    run_id = uuid.uuid4().hex[:12]
    cmd = [MAIN_PY, os.path.join(_REPO_ROOT, "scripts", "run_workload_with_scope.py"),
           "--workload", label, "--gpu-index", "0", "--output-dir", OUT_DIR,
           "--run-id", run_id, "--no-dcgm", "--dcgm", "--raw-seconds", "25"]
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

    classes = [c for c in CLASSES if not args.only or any(k in c for k in args.only.split(","))]
    os.makedirs(os.path.join(_REPO_ROOT, OUT_DIR), exist_ok=True)
    if args.reset and os.path.exists(PROGRESS):
        os.remove(PROGRESS)
    prog = load_progress()

    todo = [(c, rep) for c in classes for rep in range(1, args.reps + 1)
            if prog.get(f"{c}:{rep}", {}).get("status") != "done"]
    print(f"scale sweep: {len(classes)} classes, {len(todo)} reps", flush=True)
    failed = []
    for label, rep in todo:
        key = f"{label}:{rep}"
        ok, msg = False, ""
        for attempt in range(1, args.retries + 2):
            ok, msg = run_rep(label, rep, attempt)
            print(f"[scale] {key} attempt {attempt}: {'OK' if ok else 'FAIL'} — {msg}", flush=True)
            if ok:
                break
            time.sleep(15)
        prog[key] = {"status": "done" if ok else "failed", "detail": msg, "ts": time.time()}
        json.dump(prog, open(PROGRESS, "w"), indent=1)
        if not ok:
            failed.append(key)
        time.sleep(8)

    done = sum(1 for v in prog.values() if v.get("status") == "done")
    print(f"\n===== scale summary =====\ndone: {done}  failed: {len(failed)}", flush=True)
    for k in failed:
        print(f"  FAILED {k}: {prog[k]['detail']}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
