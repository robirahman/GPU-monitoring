#!/bin/bash
# MPS co-execution evader: run a steady DRAM-bandwidth cover kernel concurrently
# with tiny-batch training under CUDA MPS, so their kernels share the GPU instead
# of time-slicing on separate contexts (the serialization the plain covered_train.sh
# hit). Goal: a combined NVML signature (util / mem_utilization / dram_active /
# occupancy) pinned flat-high like HPC, with mem_used padded to a benign target,
# while training actually progresses.
#
# Tuning note (2026-09-27): cover intensity 1.0 STARVES training (0 steps in
# 900 s); intensity ~0.3 pins NVML flat-high (util/mem_util 100 %) while training
# still progresses (~60 s/step). Use ~0.3.
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
# train_gpt2cu inherits the CUDA_MPS_PIPE_DIRECTORY env -> also an MPS client.
# Preserve the TRAINING exit code: the EXIT trap kills the cover and `wait`s on
# it, and wait on a SIGTERM'd process returns 143 — without capturing the real
# training rc, the script would exit 143 even on a clean run and fail validation.
set +e
./train_gpt2cu "$@"
TRAIN_RC=$?
trap - EXIT
kill "$COVER_PID" 2>/dev/null
wait "$COVER_PID" 2>/dev/null
exit "$TRAIN_RC"
