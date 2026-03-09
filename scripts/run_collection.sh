#!/usr/bin/env bash
# Week 3 full data collection — GPU 0, all workloads, 3 runs each
set -euo pipefail
cd "$(dirname "$0")/.."

RUN() {
    local LABEL=$1
    local RUN_N=$2
    echo "========================================"
    echo "  $LABEL  run $RUN_N / 3"
    echo "========================================"
    python3 scripts/run_workload.py --workload "$LABEL"
    echo "--- $LABEL run $RUN_N complete ---"
    sleep 10
}

echo "=== Week 3 collection started at $(date) ==="

# --- Baseline ---
for run in 1 2 3; do RUN idle $run; done

# --- ML Training ---
for run in 1 2 3; do RUN pytorch_resnet_cifar10 $run; done
for run in 1 2 3; do RUN pytorch_resnet_cifar10_amp $run; done
for run in 1 2 3; do RUN pytorch_mlp_cifar10 $run; done
for run in 1 2 3; do RUN gpt2_wikitext2 $run; done
for run in 1 2 3; do RUN gpt2_wikitext2_amp $run; done
for run in 1 2 3; do RUN bert_sst2 $run; done
for run in 1 2 3; do RUN bert_sst2_amp $run; done

# --- ML Inference ---
for run in 1 2 3; do RUN resnet50_inference $run; done

# --- Scientific HPC ---
for run in 1 2 3; do RUN cufft_benchmark $run; done
for run in 1 2 3; do RUN nbody_sim $run; done

# --- Crypto Mining ---
for run in 1 2 3; do RUN mining_ethash_proxy $run; done

# --- Rendering ---
for run in 1 2 3; do RUN rendering_proxy $run; done

# --- Other ---
for run in 1 2 3; do RUN idle $run; done
for run in 1 2 3; do RUN ffmpeg_nvenc $run; done

echo ""
echo "=== Collection COMPLETE at $(date) ==="
echo "Files:"
ls data/*.parquet 2>/dev/null | wc -l
