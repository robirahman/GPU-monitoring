#!/usr/bin/env bash
# Re-run workloads lost to the disk-full incident + mistral 3 reps after
# patching the optimum/GPTQModel BACKEND.EXLLAMA_V1 incompatibility.

set -u
cd /workspace/SPAR-GPU-monitoring
export PYTHONPATH=/usr/share/datacenter-gpu-manager-4/bindings/python3:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/workspace/.hf_home}
source /workspace/.hf_auth 2>/dev/null || true

LOG=/workspace/SPAR-GPU-monitoring/logs/b200_reruns_$(date +%Y%m%d_%H%M%S).log
echo "==== B200 reruns started $(date -u) ====" | tee -a "$LOG"

run_wl() {
  local LABEL="$1"
  echo "---- $(date -u) $LABEL ----" | tee -a "$LOG"
  python scripts/run_workload.py --workload "$LABEL" --output-dir data/ >>"$LOG" 2>&1
  echo "[$(date -u)] $LABEL exit=$?" | tee -a "$LOG"
}

run_cmd() {
  local LABEL="$1"; shift
  echo "---- $(date -u) $LABEL (extended) ----" | tee -a "$LOG"
  python scripts/run_workload.py --workload "$LABEL" --output-dir data/ -- "$@" >>"$LOG" 2>&1
  echo "[$(date -u)] $LABEL exit=$?" | tee -a "$LOG"
}

# 1) Short training re-runs first (1-3 min each)
run_cmd pytorch_resnet_cifar10_ext \
  python workloads/pytorch_training.py --model resnet18 --dataset cifar10 --epochs 30 --batch-size 512
run_cmd pytorch_resnet_cifar10_amp_ext \
  python workloads/pytorch_training.py --model resnet18 --dataset cifar10 --epochs 30 --batch-size 512 --amp
run_cmd pytorch_mlp_cifar10_ext \
  python workloads/pytorch_training.py --model simple_mlp --dataset cifar10 --epochs 60 --batch-size 512
run_cmd gpt2_wikitext2_ext \
  python workloads/gpt2_finetune.py --steps 1200 --batch-size 8

# 2) Mistral 3 reps (post-patch)
for REP in 1 2 3; do
  run_wl llm_infer_mistral_7b_q4
done

# 3) Non-ML lost reps: 1× idle, 2× cufft, 2× mining
run_wl idle
run_wl cufft_benchmark
run_wl cufft_benchmark
run_wl mining_ethash_proxy
run_wl mining_ethash_proxy

echo "==== B200 reruns finished $(date -u) ====" | tee -a "$LOG"
