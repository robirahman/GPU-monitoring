#!/bin/bash
# Experiment 5: window-correlation structure at a MODERATE detection rate.
#
# Motivation. Both 8 h runs so far were flagged in 0/1920 windows, so
# analyze_long_run.py reported "lag-1 alert autocorrelation: nan" — a flag that
# never varies carries no correlation information. Section 2's run-level
# compounding argument rests on exactly that autocorrelation, so it needs a run
# whose per-window flag actually fluctuates (target 40-80% detection).
#
# Why 7 B WB-L at reduced dilution:
#   - WB-L has per-step STEP logging AND genuine non-training windows, which is
#     required to ask whether the classifier flags training vs non-training
#     windows correctly. Families like composite_*/M_*/K/N/A/B/D/J train on
#     every batch, so that question is vacuous for them; L_diluted is CPU-bound
#     (~4 batches per run) and useless for it.
#   - 7 B at batch 6 / seq 512 sits at 30.4 GB, inside the corpus's 25-35 GB
#     band, so detection is not suppressed by the model-size confound that made
#     the 1 B result ambiguous.
#   - N=10 gave p_train mean 0.328 / max 0.537 (below tau=0.75). Training more
#     often should lift it into the fluctuating regime.
#
# Phases: pilot 3 dilutions -> pick the mid-band one from data -> long run.
#
# Waits for the section-6 analysis to finish first: that job fits random forests
# with n_jobs=-1 across 128 cores, and heavy CPU load perturbs GPU telemetry for
# dataloader-bound workloads. Every run so far was collected on an otherwise
# idle box; keep it that way.

REPO=/root/GPU-monitoring
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

export HF_HOME=/workspace/hf-cache
export PYTHONPATH=/root/GPU-monitoring

PY=python3
MODEL_7B=NousResearch/Llama-2-7b-hf
CKPT=300
SP=/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad
PILOT_SEC=900             # 15 min per pilot: ~60 windows, enough for a rate
LONG_SEC=14400            # 4 h: ~960 windows at 15 s stride, ample for an ACF
ANALYSIS_MAX_WAIT=14400

mkdir -p logs logs/.done data
MASTER=logs/queue_master.log

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" >> "$MASTER"; }

run_stage() {
  local name="$1"; shift
  local logf="$1"; shift
  local marker="logs/.done/${name}.done"
  if [ -f "$marker" ]; then say "SKIP  $name (already completed)"; return 0; fi
  say "START $name -> $logf"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?
  local dur=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "$(stamp) rc=0 dur=${dur}s" > "$marker"; say "OK    $name (${dur}s)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s" > "logs/.done/${name}.failed"
    say "FAIL  $name rc=$rc (${dur}s) -- continuing"
  fi
  return 0
}

say "===== queue v4 (window-correlation experiment) starting (pid $$) ====="

# ── Wait for the CPU analysis so the box is idle during collection ─────────
say "waiting for the section-6 analysis to finish before collecting"
waited=0
while ! grep -q 'ANALYSIS COMPLETE' "$SP/analysis.log" 2>/dev/null; do
  sleep 60; waited=$((waited + 60))
  if [ $waited -ge $ANALYSIS_MAX_WAIT ]; then
    say "WARNING: analysis still running after ${waited}s — proceeding anyway"
    break
  fi
done
say "proceeding after ${waited}s wait"

# ── Phase 1: pilots at three dilutions (15 min each) ──────────────────────
for N in 2 3 5; do
  run_stage "wb_lora_7b_pilot_N${N}" "logs/wb_lora_7b_pilot_N${N}.log" \
    $PY scripts/run_workload.py --workload "whitebox_lora_7b_pilot_N${N}" \
      --timeout $((PILOT_SEC + 600)) --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL_7B" \
      --batch-size 6 --seq-len 512 --dilution "$N" --duration $PILOT_SEC
done

# ── Phase 2: choose the mid-band dilution from the pilot scores ───────────
say "scoring pilots to pick a mid-band dilution"
$PY "$SP/select_dilution.py" >> "$MASTER" 2>&1
CHOSEN=$(cat "$SP/chosen_dilution" 2>/dev/null)

if [ -z "$CHOSEN" ]; then
  say "ABORT: pilot scoring produced no dilution choice"
  say "===== queue v4 aborted ====="
  exit 1
fi
say "chosen dilution: N=${CHOSEN}"

# ── Phase 3: the 4 h run at the chosen dilution ───────────────────────────
run_stage "wb_lora_7b_corr_N${CHOSEN}_long" "logs/wb_lora_7b_corr_N${CHOSEN}_long.log" \
  $PY scripts/run_workload.py --workload "whitebox_lora_7b_corr_N${CHOSEN}_long" \
    --timeout $((LONG_SEC + 900)) --checkpoint-sec $CKPT -- \
    $PY workloads/adversarial_whitebox.py --strategy WB-L --model "$MODEL_7B" \
    --batch-size 6 --seq-len 512 --dilution "$CHOSEN" --duration $LONG_SEC

say "===== queue v4 complete ====="
say "run the correlation analysis with:"
say "  python3 $SP/analyze_correlation.py --dilution ${CHOSEN}"
