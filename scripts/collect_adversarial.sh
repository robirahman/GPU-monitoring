#!/bin/bash
# Collect adversarial training telemetry on H100 NVL
# GPU 0: Strategies A and B (3 runs each)
# Runs sequentially on one GPU; the other GPU runs different strategies
set -e
cd /home/claude/SPAR-GPU-monitoring
OUTDIR="data/h100_nvl"
GPU=$1  # Pass GPU index as argument

run_workload() {
    local workload=$1
    local gpu=$2
    local run_num=$3
    echo "[$(date +%H:%M:%S)] Starting $workload (run $run_num) on GPU $gpu..."
    python3 scripts/run_workload.py \
        --workload "$workload" \
        --gpu-index "$gpu" \
        --output-dir "$OUTDIR" \
        --no-dcgm \
        2>&1 | tail -5
    echo "[$(date +%H:%M:%S)] Done: $workload (run $run_num)"
    echo ""
}

echo "=== Adversarial Collection on GPU $GPU ==="

if [ "$GPU" = "0" ]; then
    # GPU 0: Strategy A (3 runs) + Strategy B (3 runs)
    for run in 1 2 3; do
        run_workload adversarial_A_util_modulation 0 $run
    done
    for run in 1 2 3; do
        run_workload adversarial_B_low_util 0 $run
    done
    echo "=== GPU 0 adversarial collection complete ==="
else
    # GPU 1: Strategy D (3 runs) + bonus: A and B with different params
    for run in 1 2 3; do
        run_workload adversarial_D_temporal_disruption 1 $run
    done
    # Also collect normal training baseline for throughput comparison
    for run in 1 2; do
        run_workload pytorch_resnet_cifar10 1 $run
    done
    echo "=== GPU 1 adversarial collection complete ==="
fi
