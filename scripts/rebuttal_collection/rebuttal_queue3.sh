#!/bin/bash
# Retry queue for the 13 stages that failed in queue v2 (2026-07-25).
#
# Two unrelated causes, both now fixed:
#
#  1. WB-O x3 — AttributeError: 'ClassifierOracle' object has no attribute
#     '_clf'. The classifier setup lived at the end of _init_amd(), so on
#     NVIDIA it never ran and the first query_p_training() raised. Moved into
#     __init__ after the backend dispatch. Verified: the oracle now builds
#     (166 features, 17,876 windows, ~191 s) and returns None before its
#     buffer fills instead of crashing. Needs no dataset, so it runs FIRST,
#     overlapping the CIFAR download.
#
#  2. Family runs x10 — killed at the registry's 360 s default_timeout with
#     ZERO STEP lines. torchvision CIFAR10(download=True) was fetching 170 MB
#     at ~60 KB/s (~48 min), so every run died mid-download and never reached
#     the training loop. The dataset is now fetched once, out of band, and
#     these runs get --timeout 600 so a cached-dataset run has real headroom.
#
# Stages that already succeeded carry logs/.done/*.done and are skipped.

REPO=/root/GPU-monitoring
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

export HF_HOME=/workspace/hf-cache

PY=python3
MODEL_1B=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300
ORACLE_TIMEOUT=1800
FAMILY_TIMEOUT=600          # was the registry default of 360 s — too tight
SP=/tmp/claude-0/-/787a5ab8-78a1-41cf-90e4-cc9f630c7038/scratchpad
CIFAR_MAX_WAIT=5400         # 90 min

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
    rm -f "logs/.done/${name}.failed"      # clear the stale v2 failure
    say "OK    $name (${dur}s)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s" > "logs/.done/${name}.failed"
    say "FAIL  $name rc=$rc (${dur}s) -- continuing to next stage"
  fi
  return 0
}

say "===== queue v3 (retry of 13 failed stages) starting (pid $$) ====="

# ── Part 1: WB-O reps (oracle bug fixed; no dataset needed) ────────────────
for rep in 1 2 3; do
  run_stage "wb_oracle_rep${rep}" "logs/wb_oracle_rep${rep}.log" \
    $PY scripts/run_workload.py --workload whitebox_oracle \
      --timeout $ORACLE_TIMEOUT --checkpoint-sec $CKPT -- \
      $PY workloads/adversarial_whitebox.py --strategy WB-O --model "$MODEL_1B" \
      --batch-size 1 --seq-len 256 --dilution 10 --grad-ckpt --duration 300
done

# ── Part 2: wait for the out-of-band CIFAR-10 fetch, then install it ───────
say "waiting for CIFAR-10 (md5-verified) before the family re-runs"
waited=0
while [ ! -f "$SP/cifar_ready" ] && [ $waited -lt $CIFAR_MAX_WAIT ]; do
  sleep 30
  waited=$((waited + 30))
done

if [ ! -f "$SP/cifar_ready" ]; then
  say "ABORT family re-runs: CIFAR-10 not ready after ${CIFAR_MAX_WAIT}s"
  say "  (fetch log: $SP/cifar_fetch.log)"
else
  say "CIFAR-10 ready after ${waited}s — installing into data/cifar10"
  mkdir -p data/cifar10
  cp -f "$SP/cifar-10-python.tar.gz" data/cifar10/
  # Extract via torchvision itself: it checks the md5, sees a valid archive,
  # and extracts without touching the network.
  if $PY -c "
from torchvision.datasets import CIFAR10
CIFAR10(root='data/cifar10', train=True, download=True)
CIFAR10(root='data/cifar10', train=False, download=True)
print('CIFAR-10 extracted and verified')
" >> "$MASTER" 2>&1; then
    say "CIFAR-10 installed — starting family re-runs"

    for rep in 1 2 3; do
      for N in 10 20; do
        run_stage "L_diluted_${N}_rep${rep}" "logs/L_diluted_${N}_rep${rep}.log" \
          $PY scripts/run_workload.py --workload "adversarial_L_diluted_${N}" \
            --timeout $FAMILY_TIMEOUT --checkpoint-sec $CKPT
      done

      run_stage "I_stochastic_rep${rep}" "logs/I_stochastic_rep${rep}.log" \
        $PY scripts/run_workload.py --workload adversarial_I_stochastic_30 \
          --timeout $FAMILY_TIMEOUT --checkpoint-sec $CKPT

      run_stage "F_interleave_rep${rep}" "logs/F_interleave_rep${rep}.log" \
        $PY scripts/run_workload.py --workload adversarial_F_interleave_50 \
          --timeout $FAMILY_TIMEOUT --checkpoint-sec $CKPT
    done
  else
    say "ABORT family re-runs: CIFAR-10 extraction failed"
  fi
fi

say "===== queue v3 complete ====="
say "done:   $(ls logs/.done/*.done 2>/dev/null | wc -l) stages"
say "failed: $(ls logs/.done/*.failed 2>/dev/null | wc -l) stages"
ls logs/.done/*.failed >> "$MASTER" 2>/dev/null
