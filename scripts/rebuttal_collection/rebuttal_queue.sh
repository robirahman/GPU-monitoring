#!/bin/bash
# Sequential rebuttal GPU collection queue (NeurIPS rebuttal, due 2026-07-27).
#
# This box has ONE visible GPU (A100-SXM4-80GB), so every stage runs strictly
# serially: two workloads sharing the GPU would superimpose their power/util
# traces and destroy the very time-series measurement being collected.
#
# Stages, in order (see report/rebuttal_experiment_plan.md §4 and
# report/rebuttal_collection_handoff.md §1, §1b, §2):
#   1. wb_lora_N10_long   8 h  WB-L N10 long run (the essential new datum)
#   2. wb_inference_long  8 h  benign WB-I control (false-alert rate at scale)
#   3. whitebox re-collection with STEP logging, 3 reps (~3.5 h)
#   4. other affected evasion families with STEP logging, 3 reps (~1 h)
#
# Resumable: each completed stage drops logs/.done/<stage>.done and is skipped
# on a re-run, so restarting never repeats a finished 8 h run.
# A failing stage is recorded and the queue CONTINUES to the next one.

REPO=/root/GPU-monitoring
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

PY=python3
MODEL=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300          # telemetry crash-safety snapshot interval (seconds)
SHORT_TIMEOUT=900 # 3x headroom on the 300 s runs, so a hang cannot stall the queue
ORACLE_TIMEOUT=1800  # WB-O also trains its oracle on the full corpus at startup

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

say "===== queue starting (pid $$) ====="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader >> "$MASTER" 2>&1

# ── Stage 1: the essential 8 h WB-L N10 long run ───────────────────────────
run_stage wb_lora_N10_long logs/wb_lora_N10_long.log \
  $PY scripts/run_workload.py --workload whitebox_lora_N10_long \
    --timeout 29000 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL" \
    --batch-size 2 --seq-len 256 --dilution 10 --duration 28800

# ── Stage 2: 8 h benign inference control ──────────────────────────────────
run_stage wb_inference_long logs/wb_inference_long.log \
  $PY scripts/run_workload.py --workload whitebox_inference_long \
    --timeout 29000 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$MODEL" \
    --batch-size 1 --seq-len 256 --duration 28800

# ── Stage 3: white-box re-collection with per-step logging (3 reps) ────────
for rep in 1 2 3; do
  run_stage "wb_inference_rep${rep}" "logs/wb_inference_rep${rep}.log" \
    $PY scripts/run_workload.py --workload whitebox_inference \
      --timeout $SHORT_TIMEOUT --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-I --model "$MODEL" \
      --batch-size 1 --seq-len 256 --duration 300

  for N in 5 10 20; do
    run_stage "wb_diluted_N${N}_rep${rep}" "logs/wb_diluted_N${N}_rep${rep}.log" \
      $PY scripts/run_workload.py --workload "whitebox_diluted_N${N}" \
        --timeout $SHORT_TIMEOUT --checkpoint-sec $CKPT -- \
        $PY workloads/adversarial_whitebox.py --strategy WB-D --model "$MODEL" \
        --batch-size 1 --seq-len 256 --dilution "$N" --grad-ckpt --duration 300

    run_stage "wb_lora_N${N}_rep${rep}" "logs/wb_lora_N${N}_rep${rep}.log" \
      $PY scripts/run_workload.py --workload "whitebox_lora_N${N}" \
        --timeout $SHORT_TIMEOUT --checkpoint-sec $CKPT -- \
        $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL" \
        --batch-size 2 --seq-len 256 --dilution "$N" --duration 300
  done

  # WB-O last in each rep: its oracle is trained from the corpus at startup,
  # so it is the slowest and the most likely to need attention.
  run_stage "wb_oracle_rep${rep}" "logs/wb_oracle_rep${rep}.log" \
    $PY scripts/run_workload.py --workload whitebox_oracle \
      --timeout $ORACLE_TIMEOUT --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-O --model "$MODEL" \
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
say "Analysis steps (CPU) were deliberately NOT queued -- run them after"
say "reviewing the collected data: see rebuttal_experiment_plan.md section 6."
