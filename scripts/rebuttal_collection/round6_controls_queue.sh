#!/bin/bash
# Round 6, completing it: the organic controls the first pass was missing.
#
# These are NOT a new round. A round in Table 3 is a new ATTACKER capability;
# nothing here attacks anything. Every cell is a control for the Round 6
# symmetric-GEMM result, so they belong to Round 6's collection and the paper
# gains no Table 3 row for them.
#
# Round 6 showed the symmetric-GEMM disguise defeats 1 Hz NVML telemetry, but
# two of its claims had no matched control:
#
#   * symgemm_ppo_fused (disguised PPO) had no non-training partner. The only
#     PPO-shaped negative was symgemm_decode_g256 -- plain decode, not
#     rollout-matched, and with no scoring phase.
#   * adversarial_N_grad_accum showed a stretched optimiser cadence hides
#     training, but only as an attack. Nobody asked whether an HONEST job with
#     the same cadence is also missed.
#
#   * three of the five symgemm_L4_* scale/duration cells had no
#     duration-matched L1. _7b runs 1449 s against a 632 s L1 (2.3x), and
#     _short7b (273 s) and _long1b (3609 s) had none at all, so run-length
#     features could separate attack from control for the wrong reason.
#
# The two new cells are workloads people organically run, which is what makes
# them BASELINE-corpus members and not probes (see the discussion in
# docs/camera_ready_results_20261022/TABLE5_ARMSRACE.md): a
# "PPO-with-the-update-deleted" ablation would be a better null pair on paper
# and a worse corpus member, because nobody runs one.
#
#   bestofn_rm_n{8,32}       best-of-n / rejection sampling vs a reward model.
#                            Generation + full-sequence scoring, no optimiser.
#                            NULL PAIR for symgemm_ppo_fused.
#   ppo_rlhf_accum{4,8}_r{8,32}
#                            PPO with gradient accumulation across rollout
#                            batches. Honest training wearing Round 4's
#                            temporal signature.
#   symgemm_L1_fwd3_{7bmatch,short7b,long1b}
#                            duration-matched negatives for the L4 scale sweep.
#
# Deliberately NOT collected: a "rollouts with the policy update discarded"
# cell. It would be the tightest null pair and it is an instrument, not a
# workload; by the same argument symgemm_L3_graddiscard is an instrument. Both
# belong beside the result, not inside the benign corpus.
#
# ORDERING IS REP-MAJOR, as in round6_symgemm_queue.sh, so thermal drift is
# spread across cells rather than aliasing onto one. 60 s idle gap between
# stages. Serial by necessity: one visible GPU.
#
# RUN THIS FIRST:
#   SMOKE=1 bash scripts/rebuttal_collection/round6_controls_queue.sh
# One 60 s pass over every distinct cell including 7B, ~20 min. Smoke output
# goes to logs/smoke_data/ and is NOT data -- a "smoke_bestofn_*" label does
# not match the bestofn_ rule in threeway_label and would land in "other".
#
# Then:
#   nohup bash scripts/rebuttal_collection/round6_controls_queue.sh &
#
# TIME AND COST (a2-ultragpu-1g, ~$3.7/hr on-demand)
#   12 generation stages (3 reps x 4 cells)  x 610 s   = 2.0 h
#    3 duration-matched L1 stages (1449+273+3609 s)    = 1.5 h
#   + model loads and 60 s gaps                        ~ 0.6 h
#   -------------------------------------------------------------
#   REPS=3   15 stages   ~4.1 h wall   ~$15
#   REPS=1    7 stages   ~2.0 h wall   ~$8    <- already a 1-rep design
# Rep-major ordering means you can stop after any rep and keep a balanced set.

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export PYTHONPATH=$REPO

PY=${PY:-python3}
WL=workloads/rlhf_production.py
SG=workloads/adversarial_symgemm.py
M1=TinyLlama/TinyLlama-1.1B-Chat-v1.0
M7=NousResearch/Llama-2-7b-hf
CKPT=300
DUR=610             # matches symgemm_ppo_fused_r{8,32} (610 s) exactly
TIMEOUT=1200
GAP=60
REPS=${REPS:-3}     # symgemm_ppo_fused has 3 reps; match it
SMOKE=${SMOKE:-0}

# Durations that match the L4 cells they control for, read off the manifest.
DUR_7BMATCH=1449    # symgemm_L4_sgdfused_7b
DUR_SHORT7B=273     # symgemm_L4_sgdfused_short7b
DUR_LONG1B=3609     # symgemm_L4_sgdfused_long1b

PREFIX=""
OUTARG=()
if [ "$SMOKE" = "1" ]; then
  DUR=60; TIMEOUT=600; GAP=10; REPS=1; PREFIX="smoke_"
  DUR_7BMATCH=60; DUR_SHORT7B=60; DUR_LONG1B=60
  OUTARG=(--output-dir logs/smoke_data)
  mkdir -p logs/smoke_data
  echo "SMOKE MODE: 60 s per cell, 1 rep. Output -> logs/smoke_data/, NOT data."
fi

mkdir -p logs logs/.done data
MASTER=logs/round6_controls_master.log
stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" | tee -a "$MASTER"; }

run_stage() {
  local name="$1"; shift
  local logf="logs/${name}.log"
  local marker="logs/.done/${name}.done"
  if [ -f "$marker" ]; then say "SKIP  $name (done: $(cat "$marker"))"; return 0; fi
  local nproc
  nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  [ "$nproc" -ne 0 ] && say "WARN  $name: $nproc compute process(es) already on the GPU"
  say "START $name -> $logf"
  local t0=$SECONDS
  "$@" > "$logf" 2>&1
  local rc=$?
  local dur=$((SECONDS - t0))
  # The workload's own gate is the authority on whether the cell is valid:
  # BON must change no weight, PPOACC must change weights. A rc=0 run whose
  # gate failed is worse than a crash, because it would be collected.
  if [ $rc -eq 0 ] && ! grep -q "GATE FAILED" "$logf"; then
    echo "$(stamp) rc=0 dur=${dur}s" > "$marker"; say "OK    $name (${dur}s)"
  else
    echo "$(stamp) rc=$rc dur=${dur}s" > "logs/.done/${name}.failed"
    say "FAIL  $name rc=$rc (${dur}s) -- continuing"
    tail -6 "$logf" | sed 's/^/        /' | tee -a "$MASTER"
  fi
  sleep $GAP
  return 0
}

# Best-of-n. batch/prompt/gen match cell_gen in round6_symgemm_queue.sh so the
# generation phase is identical to symgemm_ppo_fused's rollout phase.
cell_bon() {  # cell_bon <stage> <n>
  local name="$1" n="$2"; shift 2
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $TIMEOUT --checkpoint-sec $CKPT -- \
      $PY $WL --mode BON --model "$M1" --best-of "$n" --score-chunk 8 \
      --batch-size 4 --prompt-len 128 --gen-tokens 64 \
      --dtype bfloat16 --duration $DUR "$@"
}

# PPO + gradient accumulation. lr 1e-5 matches ppo_rlhf_r{8,32}.
cell_ppoacc() {  # cell_ppoacc <stage> <rollout> <accum>
  local name="$1" r="$2" a="$3"; shift 3
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $TIMEOUT --checkpoint-sec $CKPT -- \
      $PY $WL --mode PPOACC --model "$M1" --rollout "$r" --accum "$a" \
      --batch-size 4 --prompt-len 128 --gen-tokens 64 \
      --dtype bfloat16 --lr 1e-5 --duration $DUR "$@"
}

# Duration-matched L1 negatives. Shapes copied verbatim from cell_1b / cell_7b
# in round6_symgemm_queue.sh -- only --duration differs, which is the whole
# point: same cell, matched run length.
cell_l1_1b() {  # cell_l1_1b <stage> <duration>
  local name="$1" d="$2"
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $((d + 600)) --checkpoint-sec $CKPT -- \
      $PY $SG --mode L1 --model "$M1" \
      --batch-size 4 --seq-len 512 --dtype bfloat16 --lr 1e-3 --duration "$d"
}
cell_l1_7b() {  # cell_l1_7b <stage> <duration>
  local name="$1" d="$2"
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $((d + 900)) --checkpoint-sec $CKPT -- \
      $PY $SG --mode L1 --model "$M7" \
      --batch-size 8 --seq-len 512 --dtype bfloat16 --lr 1e-3 \
      --grad-ckpt --duration "$d"
}

say "===== Round 6 controls queue starting (pid $$) ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$MASTER"

# ── Gate G1: the label rules must place every new cell before any GPU time ──
say "G1: verifying threeway_label on the new cells"
if ! $PY - <<'PYG' 2>&1 | tee -a "$MASTER" | grep -q "LABEL_GATE_PASS"; then
import sys
sys.path.insert(0, ".")
from classifier.workload_labels import threeway_label
want = {"bestofn_rm_n8": "ml_inference", "bestofn_rm_n32": "ml_inference",
        "ppo_rlhf_accum4_r8": "ml_training", "ppo_rlhf_accum8_r32": "ml_training",
        "symgemm_L1_fwd3_7bmatch": "ml_inference",
        "symgemm_L1_fwd3_short7b": "ml_inference",
        "symgemm_L1_fwd3_long1b": "ml_inference"}
bad = {c: threeway_label(c) for c, w in want.items() if threeway_label(c) != w}
print("LABEL_GATE_FAIL", bad) if bad else print("LABEL_GATE_PASS")
PYG
  say "ABORT: a new cell does not label correctly. Collecting it would file"
  say "       honest RL training as 'other' and poison the baseline corpus."
  exit 1
fi

# ── Gate G2: the workload's own CPU self-check ─────────────────────────────
say "G2: CPU smoke of both modes"
for m in BON PPOACC; do
  if ! $PY $WL --mode $m --cpu-smoke --duration 15 2>&1 | tail -3 \
       | tee -a "$MASTER" | grep -q "GATE OK"; then
    say "ABORT: $m CPU smoke did not pass its gate"
    exit 1
  fi
done

for rep in $(seq 1 "$REPS"); do
  say "----- rep $rep / $REPS -----"
  # Null pair first: it is the cell Round 6 was missing, so it is the one
  # worth having if the queue is cut short.
  cell_bon    "${PREFIX}bestofn_rm_n8_rep${rep}"        8
  cell_bon    "${PREFIX}bestofn_rm_n32_rep${rep}"       32
  cell_ppoacc "${PREFIX}ppo_rlhf_accum4_r8_rep${rep}"   8  4
  cell_ppoacc "${PREFIX}ppo_rlhf_accum8_r32_rep${rep}"  32 8
done

# Duration-matched L1 negatives: one each, matching their L4 partners, which
# are also one run each. Shortest first so a truncated queue still closes a gap.
say "----- duration-matched L1 negatives -----"
cell_l1_7b "${PREFIX}symgemm_L1_fwd3_short7b_rep1"  $DUR_SHORT7B
cell_l1_7b "${PREFIX}symgemm_L1_fwd3_7bmatch_rep1"  $DUR_7BMATCH
cell_l1_1b "${PREFIX}symgemm_L1_fwd3_long1b_rep1"   $DUR_LONG1B

say "===== Round 6 controls queue complete ====="
say "done:   $(ls logs/.done/${PREFIX}bestofn_* logs/.done/${PREFIX}ppo_rlhf_accum* logs/.done/${PREFIX}symgemm_L1_fwd3_{short7b,7bmatch,long1b}* 2>/dev/null | grep -c done)"
ls logs/.done/${PREFIX}*.failed 2>/dev/null | tee -a "$MASTER"
if [ "$SMOKE" = "1" ]; then
  say "SMOKE COMPLETE. If every stage says OK:"
  say "  rm -f logs/.done/smoke_* && nohup bash $0 &"
fi
