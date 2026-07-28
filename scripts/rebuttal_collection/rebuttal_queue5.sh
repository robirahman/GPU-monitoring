#!/bin/bash
# Follow-on to queue v4: fill in the pilot sweep between N=5 and N=10.
#
# Why. Detection RISES with LESS dilution. The 7B N=10 run had p_train max 0.537,
# entirely under tau=0.75, so 0% of windows alerted. v4 piloted N=2/3/5, which
# all train MORE than N=10 and therefore bracket the band only from above — if
# they all come back near 100%, the 40-80% straddle point sits unsampled between
# N=5 and N=10. This adds N=6/7/8 to close that gap.
#
# Chains politely: waits for v4 to exit, does nothing if v4 already launched a
# long run at an in-band dilution, and only pilots the dilutions still missing.

REPO=/root/GPU-monitoring
cd "$REPO" || exit 1

export HF_HOME=/workspace/hf-cache
export PYTHONPATH=/root/GPU-monitoring

PY=python3
MODEL_7B=NousResearch/Llama-2-7b-hf
CKPT=300
SP=/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad
PILOT_SEC=900
LONG_SEC=14400

MASTER=logs/queue_master.log
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

say "===== queue v5 (extend pilot sweep to N=6,7,8) waiting for v4 ====="
while pgrep -f 'rebuttal_queue4\.sh' >/dev/null 2>&1; do sleep 30; done
say "v4 exited"

if ls logs/.done/wb_lora_7b_corr_*_long.done >/dev/null 2>&1; then
  say "v4 already completed a long run at an in-band dilution — v5 has nothing to do"
  say "===== queue v5 complete (no-op) ====="
  exit 0
fi

# ── Pilots for the gap between N=5 and the known 0% at N=10 ───────────────
for N in 6 7 8; do
  run_stage "wb_lora_7b_pilot_N${N}" "logs/wb_lora_7b_pilot_N${N}.log" \
    $PY scripts/run_workload.py --workload "whitebox_lora_7b_pilot_N${N}" \
      --timeout $((PILOT_SEC + 600)) --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL_7B" \
      --batch-size 6 --seq-len 512 --dilution "$N" --duration $PILOT_SEC
done

# ── Re-select over the full sweep {2,3,5,6,7,8} ───────────────────────────
say "scoring the full pilot sweep"
rm -f "$SP/chosen_dilution"
$PY "$SP/select_dilution.py" >> "$MASTER" 2>&1
CHOSEN=$(cat "$SP/chosen_dilution" 2>/dev/null)

if [ -z "$CHOSEN" ]; then
  say "ABORT: no dilution landed in the 40-80% band across N=2,3,5,6,7,8"
  say "  see $SP/pilot_scores.json for the measured curve and the suggested bracket"
  say "===== queue v5 aborted (no long run launched) ====="
  exit 1
fi
say "chosen dilution: N=${CHOSEN}"

run_stage "wb_lora_7b_corr_N${CHOSEN}_long" "logs/wb_lora_7b_corr_N${CHOSEN}_long.log" \
  $PY scripts/run_workload.py --workload "whitebox_lora_7b_corr_N${CHOSEN}_long" \
    --timeout $((LONG_SEC + 900)) --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL_7B" \
    --batch-size 6 --seq-len 512 --dilution "$CHOSEN" --duration $LONG_SEC

say "===== queue v5 complete ====="
say "then: python3 $SP/analyze_correlation.py --dilution ${CHOSEN}"
