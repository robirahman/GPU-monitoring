#!/usr/bin/env bash
# Week 3 GPU 1 parallel collection — run alongside run_collection.sh on GPU 0
set -euo pipefail
cd "$(dirname "$0")/.."

RUN() {
    local LABEL=$1
    local RUN_N=$2
    echo "========================================"
    echo "  [GPU1] $LABEL  run $RUN_N / 3"
    echo "========================================"
    python3 scripts/run_workload.py --workload "$LABEL" --gpu-index 1
    echo "--- [GPU1] $LABEL run $RUN_N complete ---"
    sleep 10
}

echo "=== GPU 1 collection started at $(date) ==="

for run in 1 2 3; do RUN gpt2_wikitext2 $run; done
for run in 1 2 3; do RUN gpt2_wikitext2_amp $run; done
for run in 1 2 3; do RUN bert_sst2 $run; done
for run in 1 2 3; do RUN bert_sst2_amp $run; done
for run in 1 2 3; do RUN resnet50_inference $run; done
for run in 1 2 3; do RUN cufft_benchmark $run; done
for run in 1 2 3; do RUN nbody_sim $run; done
for run in 1 2 3; do RUN mining_ethash_proxy $run; done
for run in 1 2 3; do RUN rendering_proxy $run; done

echo "=== GPU 1 collection COMPLETE at $(date) ==="
