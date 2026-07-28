#!/bin/bash
# Queue v6: fix L_diluted's dataloader starvation, and create the mixed
# active/idle runs that questions 1-2 actually need.
#
# Background. L_diluted with num_workers=0 is CPU-starved by Resize(224):
# ~4 batches per 300 s run, GPU at ~0.5%. Measured with 8 workers: 687 batches
# in 45 s (1,933 img/s), a ~1000x speedup. That fixes the 24-training-window
# limit on the Section 3 corrected figure, but it ALSO means training steps
# arrive every ~0.67 s at dilution 10, so every window becomes training-active
# and question 2 ("are non-training windows correctly cleared?") goes vacuous.
#
# Hence the sweep over dilution BELOW: at ~15 batches/s, a genuine idle window
# needs the training gap to exceed the window length, i.e. dilution >~150 for a
# 10 s window. N=200 gives a ~13 s gap, N=500 a ~33 s gap — a real mix of
# training-active and idle windows at both 10 s and 30 s.
#
# Stages:
#   A  L_diluted_10 mw8   x3 @ 300 s   the requested fix, matched to corpus reps
#   B  L_diluted_200 mw8  x2 @ 1800 s  mixed active/idle, and plausibly mid-band
#   C  WB-O 7B            x1 @ 7200 s  the ONE family that already separates
#                                      training from idle (recall 0.856,
#                                      false-alert 0.059) — now long, and at
#                                      corpus-matched 30 GB memory
#   D  WB-L 7B N=500      x1 @ 7200 s  sparse-training LoRA: does a rare
#                                      training step get caught at all?
#
# Runs carry DISTINCT labels (…_mw8, …_7b_…) so they are never pooled with the
# num_workers=0 corpus runs they are not comparable to.

REPO=/root/GPU-monitoring
cd "$REPO" || exit 1

export HF_HOME=/workspace/hf-cache
export PYTHONPATH=/root/GPU-monitoring

PY=python3
MODEL_7B=NousResearch/Llama-2-7b-hf
MODEL_1B=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300
MASTER=logs/queue_master.log

mkdir -p logs logs/.done data
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" >> "$MASTER"; }

run_stage() {
  local name="$1"; shift; local logf="$1"; shift
  local marker="logs/.done/${name}.done"
  if [ -f "$marker" ]; then say "SKIP  $name (already completed)"; return 0; fi
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

say "===== queue v6 (L_diluted dataloader fix + mixed-class runs) starting (pid $$) ====="

# ── A: the requested fix, 3 reps at 300 s to match the corpus reps ─────────
for rep in 1 2 3; do
  run_stage "L_diluted_10_mw8_rep${rep}" "logs/L_diluted_10_mw8_rep${rep}.log" \
    $PY scripts/run_workload.py --workload "adversarial_L_diluted_10_mw8" \
      --timeout 900 --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_round3.py --strategy L --duration 300 \
      --dilution 10 --batch-size 128 --num-workers 8
done

# ── B: high dilution, long enough for a decent window count ───────────────
for rep in 1 2; do
  run_stage "L_diluted_200_mw8_rep${rep}" "logs/L_diluted_200_mw8_rep${rep}.log" \
    $PY scripts/run_workload.py --workload "adversarial_L_diluted_200_mw8" \
      --timeout 2400 --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_round3.py --strategy L --duration 1800 \
      --dilution 200 --batch-size 128 --num-workers 8
done

# ── C: 2 h WB-O at 7B — the strongest question-2 instrument we have ────────
# WB-O trains its oracle on the whole corpus at startup (~191 s), so the
# timeout allows for that on top of the run itself.
run_stage "wb_oracle_7b_long" "logs/wb_oracle_7b_long.log" \
  $PY scripts/run_workload.py --workload "whitebox_oracle_7b_long" \
    --timeout 8400 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-O --model "$MODEL_7B" \
    --batch-size 6 --seq-len 512 --dilution 10 --duration 7200

# ── D: 2 h sparse-training LoRA at 7B ─────────────────────────────────────
run_stage "wb_lora_7b_N500_long" "logs/wb_lora_7b_N500_long.log" \
  $PY scripts/run_workload.py --workload "whitebox_lora_7b_N500_long" \
    --timeout 8100 --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL_7B" \
    --batch-size 6 --seq-len 512 --dilution 500 --duration 7200

say "===== queue v6 complete ====="
say "then re-run: python3 <scratch>/q2_confusion.py   (add the new families)"
