#!/bin/bash
# MPS co-execution evader: run a steady DRAM-bandwidth cover kernel concurrently
# with tiny-batch training under CUDA MPS, so their kernels share the GPU instead
# of time-slicing on separate contexts (the serialization the plain covered_train.sh
# hit). Goal: a combined NVML signature (util / mem_utilization / dram_active /
# occupancy) pinned flat-high like HPC, with mem_used padded to a benign target,
# while training actually progresses.
#
# Usage: covered_train_mps.sh <cover_gb> <cover_intensity> -- <train_gpt2cu args...>
set -e
COVER_GB="$1"; COVER_INT="$2"; shift 2
[ "$1" = "--" ] && shift
PICO_PY="$HOME/pico-venv/bin/python"
LLMC_DIR="$HOME/llm.c"

# Per-user MPS daemon (idempotent): all clients sharing these dirs run concurrently.
export CUDA_MPS_PIPE_DIRECTORY="$HOME/mps_pipe"
export CUDA_MPS_LOG_DIRECTORY="$HOME/mps_log"
mkdir -p "$CUDA_MPS_PIPE_DIRECTORY" "$CUDA_MPS_LOG_DIRECTORY"
if ! pgrep -f "nvidia-cuda-mps-control -d" >/dev/null; then
    nvidia-cuda-mps-control -d
    sleep 2
fi

# cover as an MPS client
"$PICO_PY" "$HOME/GPU-monitoring/scripts/cover_kernel.py" \
    --duration 9999 --gb "$COVER_GB" --gpu 0 --intensity "$COVER_INT" &
COVER_PID=$!
sleep 4

cleanup() { kill "$COVER_PID" 2>/dev/null; wait "$COVER_PID" 2>/dev/null; }
trap cleanup EXIT

cd "$LLMC_DIR"
# train_gpt2cu inherits the CUDA_MPS_PIPE_DIRECTORY env -> also an MPS client
./train_gpt2cu "$@"
