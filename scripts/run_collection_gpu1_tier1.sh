#!/usr/bin/env bash
# Week 3 GPU 1 parallel collection — Tier 1 only (pynvml, no DCGM)
set -euo pipefail
cd "$(dirname "$0")/.."

export PATH=/venv/main/bin:$PATH
export HF_HOME=/root/.hf_home
PYTHON=/venv/main/bin/python3

RUN() {
    local LABEL=$1
    local RUN_N=$2
    echo "========================================"
    echo "  [GPU1] $LABEL  run $RUN_N / 3"
    echo "========================================"
    $PYTHON scripts/run_workload.py --workload "$LABEL" --gpu-index 1 --no-dcgm
    echo "--- [GPU1] $LABEL run $RUN_N complete ---"
    sleep 10
}

echo "=== GPU 1 Tier-1 collection started at $(date) ==="

for run in 1 2 3; do RUN gpt2_wikitext2 $run; done
for run in 1 2 3; do RUN gpt2_wikitext2_amp $run; done
for run in 1 2 3; do RUN bert_sst2 $run; done
for run in 1 2 3; do RUN bert_sst2_amp $run; done
for run in 1 2 3; do RUN resnet50_inference $run; done
for run in 1 2 3; do RUN cufft_benchmark $run; done
for run in 1 2 3; do RUN nbody_sim $run; done
for run in 1 2 3; do RUN mining_ethash_proxy $run; done
for run in 1 2 3; do RUN rendering_proxy $run; done

echo "=== GPU 1 Tier-1 collection COMPLETE at $(date) ==="
ls data/*.parquet 2>/dev/null | wc -l
