#!/usr/bin/env bash
# Run the full Week 3 data collection campaign.
# Usage: bash scripts/run_collection.sh
set -euo pipefail

cd "$(dirname "$0")/.."

RUN_CMD="python3 scripts/run_workload.py"
COOLDOWN=10

echo "=== SPAR Week 3 Data Collection ==="
echo "Started at $(date)"
echo ""

for run in 1 2 3; do
    echo "========== RUN $run / 3 =========="

    echo "--- ML Training ---"
    $RUN_CMD --workload pytorch_resnet_cifar10
    sleep $COOLDOWN
    $RUN_CMD --workload pytorch_resnet_cifar10_amp
    sleep $COOLDOWN
    $RUN_CMD --workload pytorch_mlp_cifar10
    sleep $COOLDOWN
    $RUN_CMD --workload gpt2_wikitext2
    sleep $COOLDOWN
    $RUN_CMD --workload gpt2_wikitext2_amp
    sleep $COOLDOWN
    $RUN_CMD --workload bert_sst2
    sleep $COOLDOWN
    $RUN_CMD --workload bert_sst2_amp
    sleep $COOLDOWN

    echo "--- ML Inference ---"
    $RUN_CMD --workload resnet50_inference
    sleep $COOLDOWN

    echo "--- Scientific HPC ---"
    $RUN_CMD --workload cufft_benchmark
    sleep $COOLDOWN
    $RUN_CMD --workload nbody_sim
    sleep $COOLDOWN
    $RUN_CMD --workload gromacs_adh
    sleep $COOLDOWN

    echo "--- Crypto Mining ---"
    $RUN_CMD --workload ethash_cuda
    sleep $COOLDOWN

    echo "--- Rendering ---"
    $RUN_CMD --workload blender_bmw
    sleep $COOLDOWN

    echo "--- Other ---"
    $RUN_CMD --workload idle
    sleep $COOLDOWN
    $RUN_CMD --workload ffmpeg_nvenc
    sleep $COOLDOWN

    echo "RUN $run COMPLETE at $(date)"
    echo ""
done

echo "=== Collection complete at $(date) ==="
echo ""
echo "Collected files:"
ls -lh data/*.parquet | wc -l
echo "files total"
