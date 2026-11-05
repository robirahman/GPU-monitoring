#!/bin/bash
# Round 6 follow-up: widen symgemm_ppo_fused from 2 cells to 5.
#
# WHY. Under the workload-family holdout the CV groups are workload cells, and
# symgemm_ppo_fused had exactly two (r8, r32) against 5 folds. Either both land
# in the same test fold -- the family is wholly held out and reads ~1.3%, its
# leave-one-family-out value -- or they split and one trains while the other
# tests, reading ~98%. Across five fold-assignment seeds the draws were
# 2.1, 99.6, 0.4, 97.0, 99.1: bimodal, mean 59.7%, sd 53.3, and no seed ever
# produced anything near the mean. The estimator is not noisy, it is measuring
# two different things alternately, and no number of extra seeds repairs that.
# Five cells make every fold split partial, so the quantity becomes defined.
#
# r4, r16, r64 complete a geometric sweep 4-8-16-32-64 around the existing
# pair. A tighter cluster would be more homogeneous and so better behaved, but
# a real spread across rollout counts is a property of the family worth seeing
# rather than averaging away -- so the analysis should report per-cell rates,
# not just the family mean.
#
# Same program, model, shapes, duration and gap as the cells it extends
# (cell_gen in round6_symgemm_queue.sh), so the new cells are poolable with
# r8/r32 rather than being a second experiment.
#
#   9 stages (3 cells x 3 reps) x (600 s + 60 s gap) ~= 1.9 h, ~$7 on a2-ultragpu-1g
#
# Resumable: each stage drops logs/.done/<stage>.done and is skipped on re-run.

set -u
REPO=${REPO:-/opt/spar/repo}
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export PYTHONPATH=$REPO
PY=${PY:-/opt/spar/repo/venv-main/bin/python}
WL=workloads/adversarial_symgemm.py
M1=TinyLlama/TinyLlama-1.1B-Chat-v1.0
CKPT=300
DUR=600             # 600 s -> 39 windows at 30 s/15 s, matching r8/r32
TIMEOUT=900
GAP=60
REPS=${REPS:-3}     # matches the 3 reps r8 and r32 already have
ROLLOUTS=${ROLLOUTS:-"4 16 64"}
GCS=${GCS:-gs://focus-surfer-494820-g0-telemetry/round6_ppo_fused_sweep_20261105}

mkdir -p logs logs/.done data
MASTER=logs/round6_ppo_sweep_master.log
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" | tee -a "$MASTER"; }

# ── Guard: never share the GPU ─────────────────────────────────────────────
# Two workloads on one card superimpose their power and utilisation traces and
# destroy the measurement. Wait for the controls queue to exit and the card to
# clear before touching it.
say "waiting for the GPU to be free (controls queue + compute processes)"
for i in $(seq 1 480); do
  q=$(ps -eo cmd | grep -c "[c]ontrols_queue" || true)
  n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$q" -eq 0 ] && [ "$n" -eq 0 ] && break
  sleep 15
done
q=$(ps -eo cmd | grep -c "[c]ontrols_queue" || true)
n=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
if [ "$q" -ne 0 ] || [ "$n" -ne 0 ]; then
  say "ABORT: GPU still busy after 2 h (queue=$q procs=$n). Nothing collected."
  exit 1
fi
say "GPU clear. $(nvidia-smi --query-gpu=name,memory.used --format=csv,noheader)"

run_stage() {
  local name="$1"; shift
  local logf="logs/${name}.log"
  local marker="logs/.done/${name}.done"
  if [ -f "$marker" ]; then say "SKIP  $name ($(cat "$marker"))"; return 0; fi
  local nproc; nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$nproc" -ne 0 ] && say "WARN  $name: $nproc compute process(es) on the GPU"
  say "START $name -> $logf"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?
  local dur=$((SECONDS - t0))
  if [ $rc -eq 0 ]; then
    echo "$(stamp) rc=0 dur=${dur}s" > "$marker"
    # A PPO cell that took no optimiser step is an inference run wearing a
    # training label, which is worse than a crash because it gets collected.
    local steps
    steps=$(grep -c "kind=train" "$logf" 2>/dev/null || echo 0)
    if [ "$steps" -eq 0 ]; then
      say "GATE FAILED $name: zero train STEP lines -- treating as a failure"
      mv "$marker" "logs/.done/${name}.failed"
    else
      say "OK    $name (${dur}s, ${steps} optimiser steps)"
    fi
  else
    echo "$(stamp) rc=$rc dur=${dur}s" > "logs/.done/${name}.failed"
    say "FAIL  $name rc=$rc (${dur}s) -- continuing"
    tail -5 "$logf" | sed 's/^/        /' | tee -a "$MASTER"
  fi
  sleep $GAP
  return 0
}

cell_gen() {  # cell_gen <stage-name> <mode> [extra args...]
  local name="$1" mode="$2"; shift 2
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" \
      --timeout $TIMEOUT --checkpoint-sec $CKPT -- \
      $PY $WL --mode "$mode" --model "$M1" \
      --batch-size 4 --prompt-len 128 --gen-tokens 64 \
      --dtype bfloat16 --lr 1e-3 --duration $DUR "$@"
}

say "===== symgemm_ppo_fused sweep: rollouts [$ROLLOUTS] x $REPS reps ====="
# Rep-major, as in the queue this extends: every cell appears once per rep so
# thermal drift over the run spreads across cells instead of aliasing onto one.
for rep in $(seq 1 "$REPS"); do
  say "----- rep $rep / $REPS -----"
  for r in $ROLLOUTS; do
    cell_gen "symgemm_ppo_fused_r${r}_rep${rep}" P1 --rollout "$r"
  done
done

say "===== sweep complete ====="
say "done:   $(ls logs/.done/symgemm_ppo_fused_r4_*.done logs/.done/symgemm_ppo_fused_r16_*.done logs/.done/symgemm_ppo_fused_r64_*.done 2>/dev/null | wc -l) / 9 stages"
ls logs/.done/symgemm_ppo_fused_r{4,16,64}_*.failed 2>/dev/null | tee -a "$MASTER"

say "optimiser steps per cell (the window-labelling check):"
for r in $ROLLOUTS; do
  for rep in $(seq 1 "$REPS"); do
    f="logs/symgemm_ppo_fused_r${r}_rep${rep}.log"
    [ -f "$f" ] && say "  r${r} rep${rep}: $(grep -c 'kind=train' "$f") train steps, \
$(grep -c 'kind=infer' "$f") infer"
  done
done

say "uploading to $GCS"
gcloud storage rsync -r data "$GCS/data" 2>&1 | tail -3 | tee -a "$MASTER"
gcloud storage rsync -r logs "$GCS/logs" 2>&1 | tail -3 | tee -a "$MASTER"
say "upload done. The node is still RUNNING -- tear it down when the data is verified."
