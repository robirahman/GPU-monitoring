#!/bin/bash
# Round 6 collection: symmetric-GEMM backprop disguise (NeurIPS discussion,
# due 2026-08-03 AoE = 2026-08-04 12:00 UTC).
#
# Answers reviewer M7wU's outstanding objection: does a kernel-level adversary
# who reshapes backprop into forward-shaped GEMMs defeat 1 Hz NVML telemetry?
#
# The ladder (see workloads/adversarial_symgemm.py for the mechanism):
#   L0  1 forward/iter                      negative, token-matched
#   L1  3 forwards/iter                     negative, FLOP-matched  <- KEY CONTROL
#   L3  fwd + dX + dW->scratch, no update   probe: same traffic as L1, no learning
#   L4  fwd + dX + fused-SGD dW             THE ATTACK
#   L5  fwd + dX + dW + AdamW               positive control
#
# L1 vs L4 is the decisive pair: same program, same model, same shapes, same
# GEMM issue rate, differing only by the weight read-modify-write. The analytic
# gap is +12.7% of compulsory DRAM traffic at M=2048 and +4.4% at M=8192
# (classifier/rebuttal_analysis/symgemm_traffic.py).
#
# ORDERING IS REP-MAJOR, deliberately. Prior queues ran all reps of a cell
# contiguously, which confounds cell identity with GPU temperature over a
# multi-hour serial queue. Here every cell appears once per rep, so thermal
# drift is spread across cells instead of aliasing onto one. A 60 s idle gap
# between stages also lets the card settle.
#
# Serial by necessity: one visible GPU. Two workloads sharing it would
# superimpose their power/util traces and destroy the measurement.
#
# Resumable: each completed stage drops logs/.done/<stage>.done and is skipped
# on re-run. A failing stage is recorded and the queue CONTINUES.
#
# Also collected, for Area Chair 1uob's training-heterogeneity question:
#   P1  R rollout generations + 1 policy update   PPO shape (R = 8 and 32)
#   DC  autoregressive decode only                memory-bandwidth-bound negative
#
# ── RUN THIS FIRST ──────────────────────────────────────────────────────────
#   SMOKE=1 bash scripts/rebuttal_collection/round6_symgemm_queue.sh
# One 60 s pass over every distinct cell including 7B, ~35 min. It exercises
# the paths the CPU self-test cannot: model.generate(), the 7B load, and peak
# memory. Cheap insurance against discovering an OOM four hours in. Smoke
# stages write their own .done markers under a smoke_ prefix, so they do not
# collide with the real run.
#
# Then the real run:
#   nohup bash scripts/rebuttal_collection/round6_symgemm_queue.sh &
#
# ── TIME AND COST (A100-80GB at ~$1.2-1.7/hr) ───────────────────────────────
#   rep 1   10 stages   core ladder + M=8192 pair + 3 generation cells
#   rep 2   12 stages   the above + the 7B pair
#   rep 3    8 stages   core ladder + generation cells
#   rep 4    5 stages   core ladder only
#   -------------------------------------------------------------------------
#   REPS=4  35 stages   ~7.2 h wall  ~$9-12   (workload 5.8 h + gaps + loads)
#   REPS=3  30 stages   ~6.2 h wall  ~$8-11
#   REPS=2  22 stages   ~4.5 h wall  ~$6-8
#   rep 1 alone         ~1.7 h wall  ~$3     <- already a complete 1-rep design
#
# Rep-major ordering means you can stop after any rep and still have a balanced
# design. Set REPS to taste, or just kill the queue when the clock says so.
#
# Resumable: each completed stage drops logs/.done/<stage>.done and is skipped
# on re-run. A failing stage is recorded and the queue CONTINUES.

REPO=${REPO:-$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)}
cd "$REPO" || { echo "cannot cd $REPO"; exit 1; }

export HF_HOME=${HF_HOME:-/workspace/hf-cache}
export PYTHONPATH=$REPO

PY=${PY:-python3}   # overridable: this box's deps live in a venv (export PY=/root/rbenv/bin/python)
WL=workloads/adversarial_symgemm.py
M1=TinyLlama/TinyLlama-1.1B-Chat-v1.0
M7=NousResearch/Llama-2-7b-hf
CKPT=300            # telemetry crash-safety snapshot interval (seconds)
DUR=600             # 600 s -> 39 windows at 30 s/15 s
TIMEOUT=900         # headroom over DUR so a hang cannot stall the queue
TIMEOUT7B=1500      # 7B needs longer to load
GAP=60              # idle settle between stages
REPS=${REPS:-4}
RUN_TRITON=${RUN_TRITON:-0}   # gated; see G3 in the plan
SMOKE=${SMOKE:-0}

# Smoke mode: one 60 s pass over every distinct cell, 7B included. Stages are
# prefixed so their .done markers cannot satisfy the real run, and the short
# duration is the whole point -- this is a shakedown, not data.
PREFIX=""
OUTARG=()
if [ "$SMOKE" = "1" ]; then
  DUR=60; TIMEOUT=240; TIMEOUT7B=900; GAP=10; REPS=1; PREFIX="smoke_"
  # Critically, smoke telemetry must NOT land in data/. load_all_data() rglobs
  # data/**, and a "smoke_symgemm_L4_*" label does not match the symgemm_ rule
  # in threeway_label, so it would fall through to the "other" class -- real
  # training telemetry filed as a negative. Send it somewhere the loader
  # never looks.
  OUTARG=(--output-dir logs/smoke_data)
  mkdir -p logs/smoke_data
  echo "SMOKE MODE: 60 s per cell, 1 rep, ~35 min. Output goes to"
  echo "logs/smoke_data/ and is NOT usable data -- this is a shakedown."
fi

mkdir -p logs logs/.done data
MASTER=logs/round6_master.log

stamp() { date -u '+%Y-%m-%dT%H:%M:%SZ'; }
say()   { echo "[$(stamp)] $*" | tee -a "$MASTER"; }

run_stage() {
  local name="$1"; shift
  local logf="logs/${name}.log"
  local marker="logs/.done/${name}.done"

  if [ -f "$marker" ]; then
    say "SKIP  $name (already completed: $(cat "$marker"))"
    return 0
  fi

  # Nothing else may be resident on the GPU, or the traces superimpose.
  local nproc
  nproc=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
  if [ "$nproc" -ne 0 ]; then
    say "WARN  $name: $nproc compute process(es) already on the GPU"
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
    tail -5 "$logf" | sed 's/^/        /' | tee -a "$MASTER"
  fi
  sleep $GAP
  return 0
}

# 1.1B cell at the default M = 4*512 = 2048.
cell_1b() {  # cell_1b <stage-name> <mode> [extra args...]
  local name="$1" mode="$2"; shift 2
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $TIMEOUT --checkpoint-sec $CKPT -- \
      $PY $WL --mode "$mode" --model "$M1" \
      --batch-size 4 --seq-len 512 --dtype bfloat16 --lr 1e-3 \
      --duration $DUR "$@"
}

# M = 16*512 = 8192, the strongest form of the attack: M >= max(K,N)=5632 so
# dW is no longer thin-K, AND the weight write is diluted to 4.4% of traffic.
cell_1b_m8192() {
  local name="$1" mode="$2"; shift 2
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $TIMEOUT --checkpoint-sec $CKPT -- \
      $PY $WL --mode "$mode" --model "$M1" \
      --batch-size 16 --seq-len 512 --dtype bfloat16 --lr 1e-3 \
      --duration $DUR "$@"
}

# Generation cells for the Area Chair's training-heterogeneity question.
# DC is pure autoregressive decode (memory-bandwidth-bound negative, the
# contrasting partner to L1's compute-bound forward pass). P1 is the PPO shape:
# R rollout generations then one policy update, so the training duty cycle is
# 1/(R+1) of steps and far less in wall-clock, as in real RLHF.
cell_gen() {  # cell_gen <stage-name> <mode> [extra args...]
  local name="$1" mode="$2"; shift 2
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $TIMEOUT --checkpoint-sec $CKPT -- \
      $PY $WL --mode "$mode" --model "$M1" \
      --batch-size 4 --prompt-len 128 --gen-tokens 64 \
      --dtype bfloat16 --lr 1e-3 --duration $DUR "$@"
}

# 7B: K=4096, inter=11008, so M=8*512=4096 still leaves the MLP dW thin-K.
# Grad checkpointing keeps activation memory bounded on the 80 GB card.
cell_7b() {
  local name="$1" mode="$2"; shift 2
  run_stage "$name" \
    $PY scripts/run_workload.py --workload "$name" "${OUTARG[@]}" \
      --timeout $TIMEOUT7B --checkpoint-sec $CKPT -- \
      $PY $WL --mode "$mode" --model "$M7" \
      --batch-size 8 --seq-len 512 --dtype bfloat16 --lr 1e-3 \
      --grad-ckpt --duration $DUR "$@"
}

say "===== Round 6 symgemm queue starting (pid $$) ====="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader \
  | tee -a "$MASTER"

# ── Gate G1: the autograd math must be right before any GPU time is spent ──
say "G1: verifying SymLinearFn against nn.Linear autograd"
if ! $PY $WL --self-test 2>&1 | tee -a "$MASTER" | grep -q "self-test PASS"; then
  say "ABORT: self-test failed -- the disguised backward is wrong, "
  say "       every collected trace would be meaningless"
  exit 1
fi

# ── Stage 0: idle thermal baseline + queue smoke test ──────────────────────
run_stage "${PREFIX}symgemm_idle" \
  $PY scripts/run_workload.py --workload "${PREFIX}symgemm_idle" "${OUTARG[@]}" \
    --timeout 400 --checkpoint-sec $CKPT -- \
    $PY workloads/idle.py --duration 300

# ── Reps, cell-interleaved ─────────────────────────────────────────────────
for rep in $(seq 1 "$REPS"); do
  say "----- rep $rep / $REPS -----"

  # Most-decisive first within the rep: attack, its matched control, then the
  # positive control that proves the pipeline can see ordinary training.
  cell_1b "${PREFIX}symgemm_L4_sgdfused_rep${rep}"     L4
  cell_1b "${PREFIX}symgemm_L1_fwd3_rep${rep}"         L1
  cell_1b "${PREFIX}symgemm_L5_adam_rep${rep}"         L5 --lr 1e-4
  cell_1b "${PREFIX}symgemm_L0_fwd1_rep${rep}"         L0
  cell_1b "${PREFIX}symgemm_L3_graddiscard_rep${rep}"  L3

  # The evader's main lever, collected early enough to survive a truncated run.
  if [ "$rep" -le 2 ]; then
    cell_1b_m8192 "${PREFIX}symgemm_L4_sgdfused_m8192_rep${rep}" L4
    cell_1b_m8192 "${PREFIX}symgemm_L1_fwd3_m8192_rep${rep}"     L1
  fi

  # Training-heterogeneity cells (Area Chair 1uob). PPO is the one regime the
  # corpus genuinely lacks and the one the meta-review names; decode is its
  # in-program inference partner. 3 reps is enough for a duty-cycle claim.
  if [ "$rep" -le 3 ]; then
    # DISGUISED PPO (P1: fused-SGD policy update) -- a Round 6 evasion cell,
    # because it stacks temporal dilution on write minimisation. The label says
    # "fused" so it is never mistaken for the legitimate RL regime below.
    cell_gen "${PREFIX}symgemm_ppo_fused_r8_rep${rep}"   P1 --rollout 8
    cell_gen "${PREFIX}symgemm_ppo_fused_r32_rep${rep}"  P1 --rollout 32
    cell_gen "${PREFIX}symgemm_decode_g256_rep${rep}"    DC --gen-tokens 256
    # UNDISGUISED PPO (P0: ordinary autograd + AdamW) -- this is what a real
    # RLHF job runs, so it belongs in the BASELINE corpus as legitimate
    # training, NOT as an evasion. Deliberately not symgemm_*-prefixed: that
    # prefix is excluded from the clean training pool by the analysis scripts.
    cell_gen "${PREFIX}ppo_rlhf_r8_rep${rep}"            P0 --rollout 8 --lr 1e-5
    cell_gen "${PREFIX}ppo_rlhf_r32_rep${rep}"           P0 --rollout 32 --lr 1e-5
  fi

  # Scale check: does the result hold where it matters for governance?
  # In smoke mode run it in rep 1 -- 7B is the most likely thing to OOM,
  # so it is exactly what a shakedown needs to touch.
  if [ "$rep" -eq 2 ] || { [ "$SMOKE" = "1" ] && [ "$rep" -eq 1 ]; }; then
    cell_7b "${PREFIX}symgemm_L4_sgdfused_7b_rep1" L4
    cell_7b "${PREFIX}symgemm_L1_fwd3_7b_rep1"     L1
  fi
done

# ── Gated: Triton backend ──────────────────────────────────────────────────
# Only equalises kernel identity and launch count, neither of which NVML
# exposes at 1 Hz -- so this is a completeness exercise, not the experiment.
# If it runs, L1_triton is MANDATORY: without a kernel-matched control the
# comparison measures "is this Triton?" rather than "is this training?".
if [ "$RUN_TRITON" = "1" ]; then
  for rep in 1 2 3; do
    cell_1b "${PREFIX}symgemm_L4_triton_rep${rep}" L4 --backend triton
    cell_1b "${PREFIX}symgemm_L1_triton_rep${rep}" L1 --backend triton
  done
else
  say "SKIP  Triton cells (RUN_TRITON=0)"
fi

say "===== Round 6 queue complete ====="
say "done:   $(ls logs/.done/symgemm_*.done 2>/dev/null | wc -l) stages"
say "failed: $(ls logs/.done/symgemm_*.failed 2>/dev/null | wc -l) stages"
ls logs/.done/symgemm_*.failed 2>/dev/null | tee -a "$MASTER"
if [ "$SMOKE" = "1" ]; then
  say "SMOKE COMPLETE. If every stage says OK, clear the markers and run"
  say "the real queue:  rm -f logs/.done/smoke_* && nohup bash $0 &"
  exit 0
fi
say "Analysis (CPU) is deliberately NOT queued. Run:"
say "  python classifier/rebuttal_analysis/symgemm_eval.py"
