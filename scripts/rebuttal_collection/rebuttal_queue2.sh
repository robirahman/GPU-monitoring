#!/bin/bash
# Rebuttal GPU collection queue, v2 (reordered 2026-07-25).
#
# Changes from v1:
#   - The 1 B WB-I benign control was CANCELLED mid-run at 19,844 samples
#     (5.51 h, saved as a complete parquet) and is NOT re-queued.
#   - The 7 B confound-free WB-L re-run is now FIRST: the 1 B 8 h result was
#     negative-and-confounded (0/1,919 windows flagged, but TinyLlama LoRA uses
#     ~4.6 GB vs the corpus's 25-35 GB training runs), so this run is what
#     actually decides rebuttal Section 2.
#   - Stages 3 and 4 (white-box re-collection, other evasion families) follow.
#
# Model footprint: NousResearch/Llama-2-7b-hf at batch 6 / seq 512 fp16 LoRA was
# MEASURED at 30.5 GB peak NVML memory, mid-band against the corpus's 25-35 GB.
# Measured, not estimated: batch 8 / seq 1024 came in at 58.6 GB, far above the
# band, which would have reintroduced the confound in the other direction.
#   batch 4 seq 512 -> 24.7 GB | batch 6 -> 30.5 GB | batch 8 -> 35.9 GB
#
# HF cache lives on /workspace (RunPod network volume, 150 TB free): the 13.5 GB
# model does not fit in the 9.2 GB free on the overlay root.

REPO=/root/GPU-monitoring
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

export HF_HOME=/workspace/hf-cache

PY=python3
MODEL_7B=NousResearch/Llama-2-7b-hf
MODEL_1B=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300
SHORT_TIMEOUT=900
ORACLE_TIMEOUT=1800

mkdir -p logs logs/.done data
MASTER=logs/queue_master.log

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" >> "$MASTER"; }

run_stage() {
  local name="$1"; shift
  local logf="$1"; shift
  local marker="logs/.done/${name}.done"

  if [ -f "$marker" ]; then
    say "SKIP  $name (already completed: $(cat "$marker"))"
    return 0
  fi

  say "START $name -> $logf"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?
  local dur=$((SECONDS - t0))

  if [ $rc -eq 0 ]; then
    echo "$(stamp) rc=0 dur=${dur}s" > "$marker"
    say "OK    $name (${dur}s)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s" > "logs/.done/${name}.failed"
    say "FAIL  $name rc=$rc (${dur}s) -- continuing to next stage"
  fi
  return 0
}

say "===== queue v2 starting (pid $$) — 7B run first, then stages 3 and 4 ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$MASTER" 2>&1

# ── Stage 2b (NEW, FIRST): 8 h confound-free 7 B WB-L run ──────────────────
run_stage wb_lora_7b_long logs/wb_lora_7b_long.log \
  $PY scripts/run_workload.py --workload whitebox_lora_7b_long \
    --timeout 29000 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL_7B" \
    --batch-size 6 --seq-len 512 --dilution 10 --duration 28800

# ── Stage 3: white-box re-collection with per-step logging (3 reps) ────────
for rep in 1 2 3; do
  run_stage "wb_inference_rep${rep}" "logs/wb_inference_rep${rep}.log" \
    $PY scripts/run_workload.py --workload whitebox_inference \
      --timeout $SHORT_TIMEOUT --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$MODEL_1B" \
      --batch-size 1 --seq-len 256 --duration 300

  for N in 5 10 20; do
    run_stage "wb_diluted_N${N}_rep${rep}" "logs/wb_diluted_N${N}_rep${rep}.log" \
      $PY scripts/run_workload.py --workload "whitebox_diluted_N${N}" \
        --timeout $SHORT_TIMEOUT --checkpoint-sec $CKPT -- \
        $PY workloads/adversarial_whitebox.py --strategy WB-D --model "$MODEL_1B" \
        --batch-size 1 --seq-len 256 --dilution "$N" --grad-ckpt --duration 300

    run_stage "wb_lora_N${N}_rep${rep}" "logs/wb_lora_N${N}_rep${rep}.log" \
      $PY scripts/run_workload.py --workload "whitebox_lora_N${N}" \
        --timeout $SHORT_TIMEOUT --checkpoint-sec $CKPT -- \
        $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL_1B" \
        --batch-size 2 --seq-len 256 --dilution "$N" --duration 300
  done

  run_stage "wb_oracle_rep${rep}" "logs/wb_oracle_rep${rep}.log" \
    $PY scripts/run_workload.py --workload whitebox_oracle \
      --timeout $ORACLE_TIMEOUT --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-O --model "$MODEL_1B" \
      --batch-size 1 --seq-len 256 --dilution 10 --grad-ckpt --duration 300
done

# ── Stage 4: other affected families, registry mode (3 reps) ───────────────
for rep in 1 2 3; do
  for N in 10 20; do
    run_stage "L_diluted_${N}_rep${rep}" "logs/L_diluted_${N}_rep${rep}.log" \
      $PY scripts/run_workload.py --workload "adversarial_L_diluted_${N}" \
        --checkpoint-sec $CKPT
  done

  run_stage "I_stochastic_rep${rep}" "logs/I_stochastic_rep${rep}.log" \
    $PY scripts/run_workload.py --workload adversarial_I_stochastic_30 \
      --checkpoint-sec $CKPT

  run_stage "F_interleave_rep${rep}" "logs/F_interleave_rep${rep}.log" \
    $PY scripts/run_workload.py --workload adversarial_F_interleave_50 \
      --checkpoint-sec $CKPT
done

say "===== queue complete ====="
say "done:   $(ls logs/.done/*.done 2>/dev/null | wc -l) stages"
say "failed: $(ls logs/.done/*.failed 2>/dev/null | wc -l) stages"
ls logs/.done/*.failed >> "$MASTER" 2>/dev/null
say "Analysis steps (CPU) were deliberately NOT queued -- see plan section 6."
