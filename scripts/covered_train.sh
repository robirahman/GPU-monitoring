#!/bin/bash
# Co-execution evader: run a steady DRAM-bandwidth cover kernel concurrently with
# tiny-batch training on the same GPU, so the combined NVML/DCGM/scope signature
# is dominated by the cover's flat HPC-like envelope and the training's bursty
# compute-bound signature is buried.
#
# Usage: covered_train.sh <cover_gb> -- <train_gpt2cu args...>
set -e
COVER_GB="$1"; shift
[ "$1" = "--" ] && shift
PICO_PY="$HOME/pico-venv/bin/python"
LLMC_DIR="$HOME/llm.c"

# start the cover (long duration; killed when training finishes)
"$PICO_PY" "$HOME/GPU-monitoring/scripts/cover_kernel.py" \
    --duration 9999 --gb "$COVER_GB" --gpu 0 &
COVER_PID=$!
sleep 4  # let the cover reach steady state first

cleanup() { kill "$COVER_PID" 2>/dev/null; wait "$COVER_PID" 2>/dev/null; }
trap cleanup EXIT

cd "$LLMC_DIR"
./train_gpt2cu "$@"
