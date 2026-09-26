#!/usr/bin/env python3
"""H200 NVL scope-synced collection campaign (August 2026 Amodo rental).

Sequentially runs each workload class through scripts/run_workload_with_scope.py,
3 reps each, with a JSON progress file for resume, per-rep validation
(telemetry/scope pair exists and overlaps in time), and bounded retries.

Phase A replicates the April 2026 H200 NVL corpus (12 classes).
Phase B is the core extension (training / inference / non-ML / adversarial).

Usage:
    ~/pico-venv/bin/python scripts/run_h200nvl_scope_campaign.py --phase A
    ~/pico-venv/bin/python scripts/run_h200nvl_scope_campaign.py --phase B
    ... --dry-run | --reps N | --reset
"""

import argparse
import glob
import json
import os
import subprocess
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = "data/h200_nvl_scope"
PROGRESS = os.path.join(_REPO_ROOT, OUT_DIR, "campaign_progress.json")
MAIN_PY = os.path.expanduser("~/pico-venv/bin/python")
VLLM_PY = os.path.expanduser("~/vllm-venv/bin/python")
# transformers<5 + auto-gptq: the transformers-5/gptqmodel-7 stack runs GPTQ
# inference CPU-bound (~4.6 tok/s, GPU idle); this venv restores real GPU kernels
Q4_PY = os.path.expanduser("~/q4-venv/bin/python")

# (label, interpreter) — order matters; q4 GPTQ last in phase A (historically fragile)
PHASE_A = [
    ("llm_infer_qwen25_7b_fp16", VLLM_PY),
    ("llm_infer_qwen25_14b_fp16", VLLM_PY),
    ("llm_infer_qwen25_32b_fp16", VLLM_PY),
    ("llm_train_qwen25_1p5b_lora", MAIN_PY),
    ("llm_train_smollm2_1p7b_lora", MAIN_PY),
    ("llm_train_qwen25_3b_lora", MAIN_PY),
    ("llm_train_phi3_mini_lora", MAIN_PY),
    ("llm_train_mistral_7b_lora", MAIN_PY),
    ("idle", MAIN_PY),
    ("cufft_benchmark", MAIN_PY),
    ("mining_ethash_proxy", MAIN_PY),
    ("llm_infer_mistral_7b_q4", MAIN_PY),
]
PHASE_B = [
    ("pytorch_resnet_cifar10", MAIN_PY),
    ("pytorch_resnet_cifar10_amp", MAIN_PY),
    ("pytorch_mlp_cifar10", MAIN_PY),
    ("bert_sst2", MAIN_PY),
    ("gpt2_wikitext2", MAIN_PY),
    ("resnet50_inference", MAIN_PY),
    ("nbody_sim", MAIN_PY),
    ("rendering_proxy", MAIN_PY),
    ("adversarial_E_memory_minimal", MAIN_PY),
    ("adversarial_F_interleave_30", MAIN_PY),
    ("adversarial_F_interleave_50", MAIN_PY),
    ("adversarial_G_clock_throttled", MAIN_PY),
    ("adversarial_H_mimicry_mining", MAIN_PY),
    ("adversarial_H_mimicry_cufft", MAIN_PY),
    ("adversarial_I_stochastic_30", MAIN_PY),
    ("adversarial_J_pid_75", MAIN_PY),
    ("adversarial_K_online_learning", MAIN_PY),
    ("adversarial_L_diluted_2", MAIN_PY),
    ("adversarial_L_diluted_5", MAIN_PY),
    ("adversarial_L_diluted_10", MAIN_PY),
    ("adversarial_L_diluted_20", MAIN_PY),
]


def load_progress():
    if os.path.exists(PROGRESS):
        with open(PROGRESS) as f:
            return json.load(f)
    return {}


def save_progress(prog):
    with open(PROGRESS, "w") as f:
        json.dump(prog, f, indent=1)


def epoch_col(df):
    import pandas as pd
    return (pd.to_datetime(df["timestamp_utc"]) - pd.Timestamp(0, tz="UTC")).dt.total_seconds()


def validate(label, run_id, all_sensors=False):
    """Required sensor files exist and cover the telemetry span. Returns (ok, msg)."""
    import pandas as pd
    def others(f):
        b = os.path.basename(f)
        return all(k not in b for k in ("_scope", "_dcgm", "_bmc"))
    tel = [f for f in glob.glob(os.path.join(_REPO_ROOT, OUT_DIR, f"{label}_*_{run_id[:8]}_*.parquet"))
           if others(f)]
    sco = glob.glob(os.path.join(_REPO_ROOT, OUT_DIR, f"{label}_scope_{run_id}_*.parquet"))
    if not tel:
        return False, "telemetry parquet missing"
    if not sco:
        return False, "scope parquet missing"
    t = pd.read_parquet(tel[0])
    s = pd.read_parquet(sco[0])
    # H200 finishes small training classes in <30 s; 15 rows ≈ minimum real run
    if len(t) < 15:
        return False, f"telemetry too short ({len(t)} rows)"
    te = epoch_col(t)
    cover = (s.epoch_s.min() <= te.iloc[2]) and (s.epoch_s.max() >= te.iloc[-3])
    if not cover:
        return False, "scope capture does not cover telemetry span"
    ripple_a = float((s.v_rms_ac ** 2).mean() ** 0.5 / 0.1)
    nvml_w = float(t.power_draw_w.mean())
    extra = ""
    if all_sensors:
        dcg = glob.glob(os.path.join(_REPO_ROOT, OUT_DIR, f"{label}_dcgm_{run_id}_*.parquet"))
        bmc = glob.glob(os.path.join(_REPO_ROOT, OUT_DIR, f"{label}_bmc_{run_id}_*.parquet"))
        if not dcg:
            return False, "dcgm parquet missing"
        if not bmc:
            return False, "bmc parquet missing"
        g = pd.read_parquet(dcg[0]); b = pd.read_parquet(bmc[0])
        if len(g) < 5 or g["dcgm_dram_active"].notna().mean() < 0.5:
            return False, f"dcgm empty/invalid ({len(g)} rows) — is nv-hostengine up?"
        if len(b) < 5 or b["bmc_power_w"].notna().mean() < 0.5:
            return False, f"bmc empty/invalid ({len(b)} rows)"
        dram = float(g["dcgm_dram_active"].fillna(0).mean())
        bmc_w = float(b["bmc_power_w"].mean())
        extra = f", {len(g)} dcgm (dram {dram:.2f}), {len(b)} bmc (node {bmc_w:.0f} W)"
    return True, (f"{len(t)} tel, {len(s)} scope, ripple {ripple_a:.2f} A, nvml {nvml_w:.0f} W" + extra)


def run_rep(label, py, rep, attempt, args):
    import uuid
    run_id = uuid.uuid4().hex[:12]
    cmd = [py, os.path.join(_REPO_ROOT, "scripts", "run_workload_with_scope.py"),
           "--workload", label, "--gpu-index", "0", "--output-dir", OUT_DIR,
           "--run-id", run_id, "--no-dcgm"]
    if getattr(args, "all_sensors", False):
        # Ensure the DCGM host engine is up (it can die during host downtime);
        # otherwise the dcgm logger records 0 samples and the rep fails validation.
        if subprocess.call("pgrep -f nv-hostengine >/dev/null", shell=True) != 0:
            subprocess.call("sudo nv-hostengine >/dev/null 2>&1", shell=True)
            time.sleep(3)
        cmd += ["--dcgm", "--bmc", "--raw-seconds", "25"]
    if label == "llm_infer_mistral_7b_q4":
        # Aug 2026: transformers-5/gptqmodel-7 runs GPTQ CPU-bound (~4.6 tok/s,
        # GPU idle). Collect via vLLM's Marlin kernels instead — same model,
        # same bs/seq/max-tokens as the registry entry; documented deviation.
        cmd += ["--timeout", "600", "--",
                VLLM_PY, "workloads/llm_inference_vllm.py",
                "--model", "mistral_7b_q4", "--duration", "300",
                "--batch-size", "1", "--max-tokens", "256"]
    log_dir = os.path.join(_REPO_ROOT, OUT_DIR, "run_logs")
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, f"{label}_rep{rep}_a{attempt}_{run_id[:8]}.log")
    print(f"\n=== {label} rep {rep} attempt {attempt} (run_id {run_id}) log={log_path} ===",
          flush=True)
    with open(log_path, "w") as lf:
        rc = subprocess.call(cmd, cwd=_REPO_ROOT, stdout=lf, stderr=subprocess.STDOUT)
    if rc != 0:
        tail = subprocess.run(["tail", "-3", log_path], capture_output=True, text=True).stdout
        return False, f"exit code {rc}; log tail: {tail.strip()[-300:]}"
    return validate(label, run_id, all_sensors=getattr(args, "all_sensors", False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["A", "B", "all"], default="A")
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--reset", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--only", type=str, default="", help="comma-separated label filter")
    ap.add_argument("--all-sensors", action="store_true",
                    help="record all four tiers: NVML + scope + DCGM + BMC (25 s raw)")
    ap.add_argument("--out-dir", type=str, default="",
                    help="override output dir (default data/h200_nvl_scope)")
    args = ap.parse_args()
    if args.out_dir:
        global OUT_DIR, PROGRESS
        OUT_DIR = args.out_dir
        PROGRESS = os.path.join(_REPO_ROOT, OUT_DIR, "campaign_progress.json")

    plan = {"A": PHASE_A, "B": PHASE_B, "all": PHASE_A + PHASE_B}[args.phase]
    if args.only:
        keep = {x.strip() for x in args.only.split(",")}
        plan = [p for p in plan if p[0] in keep]

    os.makedirs(os.path.join(_REPO_ROOT, OUT_DIR), exist_ok=True)
    if args.reset and os.path.exists(PROGRESS):
        os.remove(PROGRESS)
    prog = load_progress()

    todo = [(l, py, rep) for (l, py) in plan for rep in range(1, args.reps + 1)
            if prog.get(f"{l}:{rep}", {}).get("status") != "done"]
    print(f"phase {args.phase}: {len(plan)} classes, {len(todo)} reps to run", flush=True)
    if args.dry_run:
        for l, py, rep in todo:
            print(f"  {l} rep{rep} [{os.path.basename(os.path.dirname(os.path.dirname(py)))}]")
        return

    failed = []
    for label, py, rep in todo:
        key = f"{label}:{rep}"
        ok, msg = False, ""
        for attempt in range(1, args.retries + 2):
            ok, msg = run_rep(label, py, rep, attempt, args)
            print(f"[campaign] {key} attempt {attempt}: {'OK' if ok else 'FAIL'} — {msg}", flush=True)
            if ok:
                break
            time.sleep(20)
        prog[key] = {"status": "done" if ok else "failed", "detail": msg, "ts": time.time()}
        save_progress(prog)
        if not ok:
            failed.append(key)
        time.sleep(10)

    print("\n===== campaign summary =====", flush=True)
    done = sum(1 for v in prog.values() if v.get("status") == "done")
    print(f"done: {done}  failed: {len(failed)}", flush=True)
    for k in failed:
        print(f"  FAILED {k}: {prog[k]['detail']}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
