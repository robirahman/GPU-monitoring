#!/usr/bin/env bash
#SBATCH --job-name=diloco-prod
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=8
#SBATCH --cpus-per-task=64
#SBATCH --time=04:00:00
#SBATCH --output=runs/diloco_%j.out
#SBATCH --error=runs/diloco_%j.err
#
# Production DiLoCo on VCF / Slurm. Each GPU rank = one DiLoCo replica.
#
#   MODEL_SIZE=1b LOCAL_STEPS=500 STREAMING=1 sbatch scripts/run_diloco_vcf.sh
#   NNODES=1 NPROC_PER_NODE=8 bash scripts/run_diloco_vcf.sh
set -euo pipefail

# Inner entry used by srun on each node
if [[ "${1:-}" == "--inner" ]]; then
  NODE_RANK="${2:-${SLURM_NODEID:-0}}"
  EXTRA=()
  if [[ "${STREAMING}" == "1" ]]; then
    EXTRA+=(--streaming --num-fragments "$NUM_FRAGMENTS" --fragment-sync-delay "$FRAG_DELAY")
  fi
  cd "$REPO"
  exec torchrun \
    --nnodes="$NNODES" \
    --nproc_per_node="$NPROC" \
    --node_rank="$NODE_RANK" \
    --master_addr="$MASTER_ADDR" \
    --master_port="$MASTER_PORT" \
    workloads/diloco_production.py \
    --model-size "$MODEL_SIZE" \
    --local-steps "$LOCAL_STEPS" \
    --reduce-dtype "$REDUCE_DTYPE" \
    --max-steps "$MAX_STEPS" \
    --batch-size "$BATCH" \
    --seq-len "$SEQ" \
    --precision "$PRECISION" \
    --dataset "$DATASET" \
    --step-log "runs/diloco_steps_${JOBTAG}.jsonl" \
    --comm-log "runs/diloco_comm_${JOBTAG}.jsonl" \
    "${EXTRA[@]}"
fi

REPO="${REPO:-$(cd "$(dirname "$0")/.." && pwd)}"
cd "$REPO"
mkdir -p runs
SCRIPT="$(cd "$(dirname "$0")" && pwd)/$(basename "$0")"

export MODEL_SIZE="${MODEL_SIZE:-1b}"
export LOCAL_STEPS="${LOCAL_STEPS:-500}"
export REDUCE_DTYPE="${REDUCE_DTYPE:-fp16}"
export MAX_STEPS="${MAX_STEPS:-2000}"
export STREAMING="${STREAMING:-0}"
export NUM_FRAGMENTS="${NUM_FRAGMENTS:-8}"
export FRAG_DELAY="${FRAG_DELAY:-4}"
export BATCH="${BATCH:-1}"
export SEQ="${SEQ:-1024}"
export PRECISION="${PRECISION:-bf16}"
export DATASET="${DATASET:-synthetic}"
export MASTER_PORT="${MASTER_PORT:-29500}"
export NCCL_DEBUG="${NCCL_DEBUG:-WARN}"
export CUDA_DEVICE_MAX_CONNECTIONS="${CUDA_DEVICE_MAX_CONNECTIONS:-1}"
export PYTHONUNBUFFERED=1
export REPO

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  export NNODES="${SLURM_NNODES}"
  export NPROC="${SLURM_GPUS_ON_NODE:-${SLURM_GPUS_PER_NODE:-8}}"
  MASTER_ADDR="$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n1)"
  export MASTER_ADDR
  export JOBTAG="${SLURM_JOB_ID}"
else
  export NNODES="${NNODES:-1}"
  export NPROC="${NPROC_PER_NODE:-8}"
  export NODE_RANK="${NODE_RANK:-0}"
  export MASTER_ADDR="${MASTER_ADDR:-$(hostname -I | awk '{print $1}')}"
  export JOBTAG="local"
fi

if [[ "$STREAMING" == "1" ]]; then
  LOCAL_STEPS=$(( (LOCAL_STEPS / NUM_FRAGMENTS) * NUM_FRAGMENTS ))
  if [[ "$LOCAL_STEPS" -lt $((NUM_FRAGMENTS * 2)) ]]; then
    LOCAL_STEPS=$((NUM_FRAGMENTS * 64))
  fi
  export LOCAL_STEPS
fi

echo "master=${MASTER_ADDR}:${MASTER_PORT} nnodes=${NNODES} nproc=${NPROC}"
echo "H=${LOCAL_STEPS} reduce=${REDUCE_DTYPE} streaming=${STREAMING} model=${MODEL_SIZE}"

if [[ -n "${SLURM_JOB_ID:-}" ]]; then
  # Do not expand SLURM_NODEID in the parent — each srun task reads its own.
  srun --ntasks-per-node=1 --cpus-per-task="${SLURM_CPUS_PER_TASK:-64}" \
    bash "$SCRIPT" --inner
else
  bash "$SCRIPT" --inner "$NODE_RANK"
fi
