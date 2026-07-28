#!/bin/bash
# Queue v8 — fills the window to the EOD-27-AOE deadline (11:59Z Jul 28).
# Chains behind v7. Quick wins first, the 8 h workhorse last (overnight).
#
# A  7B WB-L dilution curve at N=20, 50, 100 (1 h each)
#    We have 15-min pilots at N=2..8, an 8 h run at N=10, and a 2 h run at
#    N=500 — but nothing between 10 and 500, and the pilots are too short to
#    separate steady state from startup transients. Three 1 h points complete a
#    detection-vs-dilution curve at consistent duration and corpus-matched
#    30.4 GB memory. With v7's N=1 and N=3 that is N = 1,3,10,20,50,100,500.
#
# B  1B WB-L undiluted (N=1), 1 h
#    Cross-scale control. If undiluted LoRA IS detected at 1.1B but NOT at 7B,
#    the invisibility is a scale effect, not a property of LoRA per se — a
#    sharper claim than either run alone supports.
#
# C  8 h 7B WB-L N=500
#    The Q1/Q2/Q3 workhorse. The 2 h run at this dilution is the only one with
#    genuine idle windows (135.8 s between training steps, 22% of windows
#    active); 8 h gives ~1,900 windows instead of 480, enough for a firm
#    confusion matrix AND an autocorrelation function out to long lags at a
#    genuinely mixed operating point.

REPO=/root/GPU-monitoring
cd "$REPO" || exit 1
export HF_HOME=/workspace/hf-cache
export PYTHONPATH=/root/GPU-monitoring

PY=python3
M7=NousResearch/Llama-2-7b-hf
M1=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300
MASTER=logs/queue_master.log
mkdir -p logs logs/.done data

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" >> "$MASTER"; }

run_stage() {
  local name="$1"; shift; local logf="$1"; shift
  local marker="logs/.done/${name}.done"
  if [ -f "$marker" ]; then say "SKIP  $name"; return 0; fi
  say "START $name -> $logf"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?; local dur=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "$(stamp) rc=0 dur=${dur}s" > "$marker"; say "OK    $name (${dur}s)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s" > "logs/.done/${name}.failed"
    say "FAIL  $name rc=$rc (${dur}s) -- continuing"
  fi
  return 0
}

say "===== queue v8 waiting for v7 ====="
while pgrep -f 'rebuttal_queue7\.sh' >/dev/null 2>&1; do sleep 30; done
say "===== queue v8 starting (pid $$) ====="

# ── A: dilution curve, 1 h per point ──────────────────────────────────────
for N in 20 50 100; do
  run_stage "wb_lora_7b_N${N}_1h" "logs/wb_lora_7b_N${N}_1h.log" \
    $PY scripts/run_workload.py --workload "whitebox_lora_7b_N${N}_1h" \
      --timeout 4500 --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M7" \
      --batch-size 6 --seq-len 512 --dilution "$N" --duration 3600
done

# ── B: 1B undiluted LoRA, cross-scale control ─────────────────────────────
run_stage "wb_lora_1b_N1_1h" "logs/wb_lora_1b_N1_1h.log" \
  $PY scripts/run_workload.py --workload "whitebox_lora_1b_N1_1h" \
    --timeout 4500 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M1" \
    --batch-size 2 --seq-len 256 --dilution 1 --duration 3600

# ── C: 8 h N=500 — mixed active/idle windows for Q1/Q2/Q3 ─────────────────
run_stage "wb_lora_7b_N500_8h" "logs/wb_lora_7b_N500_8h.log" \
  $PY scripts/run_workload.py --workload "whitebox_lora_7b_N500_8h" \
    --timeout 29500 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$M7" \
    --batch-size 6 --seq-len 512 --dilution 500 --duration 28800

say "===== queue v8 complete ====="
