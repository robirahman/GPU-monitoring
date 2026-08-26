"""Workload-shaping evasion sweep — llm.c GPT-2 training with idle-gap shaping,
captured on all three sensor tiers (NVML + scope + DCGM) for the classifier study.

Each config runs the patched train_gpt2cu via run_workload_with_scope.py in
command-passthrough mode (no registry entry needed). The label sets the
filename and — because it contains 'train'/'gpt' — auto-maps to ml_training in
the classifier taxonomy. Baseline = unshaped (-Sw 0); shaped configs escalate
the gap so the duty cycle falls from ~saturated toward inference-like.

Usage:
  ~/pico-venv/bin/python scripts/run_shaping_sweep.py            # full sweep, 3 reps
  ... --only baseline_train,shaped_g50 --reps 1 --dry-run
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
OUT_DIR = "data/h200_nvl_scope/shaping"
PROGRESS = os.path.join(_REPO_ROOT, OUT_DIR, "shaping_progress.json")
MAIN_PY = os.path.expanduser("~/pico-venv/bin/python")
LLMC = os.path.expanduser("~/llm.c/train_gpt2cu")
WEIGHTS = os.path.expanduser("~/llm.c/gpt2_124M_bf16.bin")

# GPT-2 124M, b=16 t=1024, total batch 131072 => 8 micro-steps/step => 8 gaps/step.
# ~300 s of training per run at -x tuned per cadence (shaped runs are slower).
COMMON = ["-e", WEIGHTS, "-b", "16", "-t", "1024", "-d", "131072",
          "-v", "100000", "-s", "0"]

# (label, extra train_gpt2cu shaping args, steps). jitter ~= 40% of base gap.
# `-b 1` in extra overrides COMMON's -b 16 (last flag wins); with -d 131072 that
# is 128 grad-accum micro-steps, i.e. the same effective batch / same training,
# but ~1/16 the activation memory. -r 2 recomputes activations. -Sm pads the
# resident footprint to a target level.
CONFIGS = [
    ("adversarial_llmc_baseline_train", ["-Sw", "0"], "220"),
    ("adversarial_llmc_shaped_g20", ["-Sw", "1", "-Sg", "20000", "-Sj", "8000"], "160"),
    ("adversarial_llmc_shaped_g35", ["-Sw", "1", "-Sg", "35000", "-Sj", "14000"], "130"),
    ("adversarial_llmc_shaped_g50", ["-Sw", "1", "-Sg", "50000", "-Sj", "20000"], "110"),
    ("adversarial_llmc_shaped_g70", ["-Sw", "1", "-Sg", "70000", "-Sj", "28000"], "90"),
    ("adversarial_llmc_shaped_g100", ["-Sw", "1", "-Sg", "100000", "-Sj", "40000"], "70"),
    # memory-evader families: minimize footprint (~3.1 GB, below inference) and
    # mimic resnet50 inference's flat footprint (~4.8 GB via -Sm 1650), each with
    # inference-like cadence shaping. b=1 shaping puts a gap on every one of the
    # 128 micro-steps/step, so use few outer steps.
    ("adversarial_mm_min_noshape", ["-b", "1", "-r", "2", "-Sw", "0"], "200"),
    ("adversarial_mm_min_shaped", ["-b", "1", "-r", "2", "-Sw", "1", "-Sg", "50000", "-Sj", "20000"], "20"),
    ("adversarial_mm_spoof_infer_shaped", ["-b", "1", "-r", "2", "-Sm", "1650", "-Sw", "1", "-Sg", "50000", "-Sj", "20000"], "20"),
]


def load_progress():
    if os.path.exists(PROGRESS):
        return json.load(open(PROGRESS))
    return {}


def validate(label, run_id):
    import numpy as np
    import pandas as pd
    d = os.path.join(_REPO_ROOT, OUT_DIR)
    tel = [f for f in glob.glob(os.path.join(d, f"{label}_*_{run_id[:8]}_*.parquet"))
           if "_scope" not in os.path.basename(f) and "_dcgm" not in os.path.basename(f)]
    sco = glob.glob(os.path.join(d, f"{label}_scope_{run_id}_*.parquet"))
    dcg = glob.glob(os.path.join(d, f"{label}_dcgm_{run_id}_*.parquet"))
    if not tel:
        return False, "NVML telemetry missing"
    if not sco:
        return False, "scope parquet missing"
    if not dcg:
        return False, "dcgm parquet missing"
    t = pd.read_parquet(tel[0]); s = pd.read_parquet(sco[0]); g = pd.read_parquet(dcg[0])
    if len(t) < 20:
        return False, f"telemetry too short ({len(t)} rows)"
    te = (pd.to_datetime(t["timestamp_utc"]) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()
    if not (s.epoch_s.min() <= te.iloc[2] and s.epoch_s.max() >= te.iloc[-3]):
        return False, "scope does not cover telemetry span"
    if not (g.epoch.min() <= te.iloc[3] and g.epoch.max() >= te.iloc[-3]):
        return False, "dcgm does not cover telemetry span"
    ripple = float((s.v_rms_ac ** 2).mean() ** 0.5 / 0.1)
    dram = float(g["dcgm_dram_active"].fillna(0).mean())
    util = float(t.gpu_utilization_pct.mean())
    return True, (f"{len(t)} tel, {len(s)} scope, {len(g)} dcgm | "
                  f"nvml_util {util:.0f}% ripple {ripple:.2f}A dram {dram:.3f}")


def run_rep(label, extra, steps, rep, attempt):
    run_id = uuid.uuid4().hex[:12]
    # train_gpt2cu resolves gpt2_tokenizer.bin and the tinyshakespeare data via
    # paths relative to the llm.c dir, but run_workload.py launches the workload
    # with cwd=<GPU-monitoring repo>. Run it from ~/llm.c via a shell wrapper.
    llmc_dir = os.path.dirname(LLMC)
    inner = " ".join([LLMC] + COMMON + ["-x", steps] + extra)
    train_cmd = ["bash", "-lc", f"cd {llmc_dir} && exec {inner}"]
    cmd = [MAIN_PY, os.path.join(_REPO_ROOT, "scripts", "run_workload_with_scope.py"),
           "--workload", label, "--gpu-index", "0", "--output-dir", OUT_DIR,
           "--run-id", run_id, "--no-dcgm", "--dcgm", "--timeout", "900",
           "--"] + train_cmd
    log_dir = os.path.join(_REPO_ROOT, OUT_DIR, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{label}_rep{rep}_a{attempt}_{run_id[:8]}.log")
    print(f"\n=== {label} rep {rep} attempt {attempt} (run_id {run_id}) ===", flush=True)
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=_REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        tail = subprocess.run(["tail", "-4", log_path], capture_output=True, text=True).stdout
        return False, f"exit {rc}; {tail.strip()[-300:]}"
    return validate(label, run_id)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--retries", type=int, default=1)
    ap.add_argument("--only", type=str, default="")
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    configs = CONFIGS
    if args.only:
        keep = [x.strip() for x in args.only.split(",")]
        configs = [c for c in CONFIGS
                   if c[0] in keep or any(k in c[0] for k in keep)]

    os.makedirs(os.path.join(_REPO_ROOT, OUT_DIR), exist_ok=True)
    if args.reset and os.path.exists(PROGRESS):
        os.remove(PROGRESS)
    prog = load_progress()

    todo = [(lbl, ex, st, rep) for (lbl, ex, st) in configs for rep in range(1, args.reps + 1)
            if prog.get(f"{lbl}:{rep}", {}).get("status") != "done"]
    print(f"shaping sweep: {len(configs)} configs, {len(todo)} reps to run", flush=True)
    if args.dry_run:
        for lbl, ex, st, rep in todo:
            print(f"  {lbl} rep{rep}: {' '.join(ex)} -x {st}")
        return

    failed = []
    for lbl, ex, st, rep in todo:
        key = f"{lbl}:{rep}"
        ok, msg = False, ""
        for attempt in range(1, args.retries + 2):
            ok, msg = run_rep(lbl, ex, st, rep, attempt)
            print(f"[sweep] {key} attempt {attempt}: {'OK' if ok else 'FAIL'} — {msg}", flush=True)
            if ok:
                break
            time.sleep(15)
        prog[key] = {"status": "done" if ok else "failed", "detail": msg, "ts": time.time()}
        json.dump(prog, open(PROGRESS, "w"), indent=1)
        if not ok:
            failed.append(key)
        time.sleep(8)

    done = sum(1 for v in prog.values() if v.get("status") == "done")
    print(f"\n===== sweep summary =====\ndone: {done}  failed: {len(failed)}", flush=True)
    for k in failed:
        print(f"  FAILED {k}: {prog[k]['detail']}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
