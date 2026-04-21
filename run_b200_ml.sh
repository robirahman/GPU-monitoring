#!/usr/bin/env bash
# B200 ML data collection: extended small-model training + 15 single-GPU
# inference workloads (defender_inference datacenter set minus TP2/TP8).
# Waits until run_b200_nonml.sh completes before launching.

set -u
cd /workspace/SPAR-GPU-monitoring
export PYTHONPATH=/usr/share/datacenter-gpu-manager-4/bindings/python3:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/workspace/.cache/huggingface}
mkdir -p "$HF_HOME"

LOG=/workspace/SPAR-GPU-monitoring/logs/b200_ml_$(date +%Y%m%d_%H%M%S).log
echo "==== B200 ML collection started $(date -u) ====" | tee -a "$LOG"

# Wait for non-ML queue to finish
while pgrep -f "run_b200_nonml.sh" >/dev/null 2>&1; do
  echo "[$(date -u)] waiting for non-ML queue..." | tee -a "$LOG"
  sleep 60
done
echo "[$(date -u)] non-ML queue finished; starting ML" | tee -a "$LOG"

run_wl() {
  local LABEL="$1"; shift
  echo "---- $(date -u) $LABEL ----" | tee -a "$LOG"
  python scripts/run_workload.py --workload "$LABEL" --output-dir data/ "$@" >>"$LOG" 2>&1
  echo "[$(date -u)] $LABEL exit=$?" | tee -a "$LOG"
}

# Command-mode run (for extended-epoch training overrides)
run_cmd() {
  local LABEL="$1"; shift
  echo "---- $(date -u) $LABEL (extended) ----" | tee -a "$LOG"
  python scripts/run_workload.py --workload "$LABEL" --output-dir data/ -- "$@" >>"$LOG" 2>&1
  echo "[$(date -u)] $LABEL exit=$?" | tee -a "$LOG"
}

# ---------- Extended-epoch training (3 reps each) ----------
# Small-model epochs tripled so telemetry windows have ≥5 epoch cycles.
# Label suffix "_ext" to distinguish from the existing default-param runs in data/.
TRAIN_EXT=(
  "pytorch_resnet_cifar10_ext|python workloads/pytorch_training.py --model resnet18 --dataset cifar10 --epochs 30 --batch-size 512"
  "pytorch_resnet_cifar10_amp_ext|python workloads/pytorch_training.py --model resnet18 --dataset cifar10 --epochs 30 --batch-size 512 --amp"
  "pytorch_mlp_cifar10_ext|python workloads/pytorch_training.py --model simple_mlp --dataset cifar10 --epochs 60 --batch-size 512"
  "gpt2_wikitext2_ext|python workloads/gpt2_finetune.py --steps 1200 --batch-size 8"
  "gpt2_wikitext2_amp_ext|python workloads/gpt2_finetune.py --steps 1200 --batch-size 8 --amp"
  "bert_sst2_ext|python workloads/bert_finetune.py --steps 900 --batch-size 32"
  "bert_sst2_amp_ext|python workloads/bert_finetune.py --steps 900 --batch-size 32 --amp"
)

for REP in 1 2 3; do
  for ENTRY in "${TRAIN_EXT[@]}"; do
    LABEL="${ENTRY%%|*}"
    CMD="${ENTRY#*|}"
    run_cmd "$LABEL" $CMD
  done
done

# Phase B is handled by run_b200_phase_b.py with JIT download orchestration.
# Exit after training to hand off cleanly.
echo "[$(date -u)] Phase A done; exiting. Phase B orchestrator takes over." | tee -a "$LOG"
exit 0

# ---------- 15 single-GPU inference workloads (3 reps each) ----------
# From scripts/run_round5_defender_inference.py DATACENTER_SUBSET_CONSUMER +
# DATACENTER_LARGE_WORKLOADS, minus llm_infer_qwen25_72b_tp2 (2-GPU).
INFERENCE_15=(
  # consumer subset (9, matched to gpt2/bert/pytorch_* B200 training)
  llm_infer_phi3_mini
  llm_infer_qwen25_3b
  llm_infer_mistral_7b_q4
  llm_infer_tinyllama_bs8
  llm_infer_qwen25_3b_bs4
  llm_infer_sdxl
  llm_infer_tinyllama_seq4096
  llm_infer_qwen25_3b_bs16
  llm_infer_vit_l_bs64
  # datacenter large, single-GPU only (6, matched to fsdp_7b/13b/30b/70b B200 training)
  llm_infer_qwen25_7b_fp16
  llm_infer_qwen25_14b_fp16
  llm_infer_qwen25_32b_fp16
  llm_infer_qwen25_72b_fp16
  llm_infer_mixtral_8x7b
  llm_infer_qwen25_72b_fp8
)

for REP in 1 2 3; do
  for WL in "${INFERENCE_15[@]}"; do
    run_wl "$WL"
  done
done

echo "==== B200 ML collection finished $(date -u) ====" | tee -a "$LOG"
