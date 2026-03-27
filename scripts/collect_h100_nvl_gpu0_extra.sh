#!/bin/bash
# Extra workloads for GPU 0 after training/inference complete
# Picks up mining and rendering to rebalance with GPU 1
set -e
cd /home/claude/SPAR-GPU-monitoring
OUTDIR="data/h100_nvl"

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
        2>&1 | tail -3
    echo "[$(date +%H:%M:%S)] Done: $workload (run $run_num)"
    echo ""
}

echo "=== GPU 0 extra workloads (rebalanced from GPU 1) ==="

# Mining - all 3 runs on GPU 0
for run in 1 2 3; do
    run_workload mining_ethash_proxy 0 $run
done

# Rendering - all 3 runs on GPU 0
for run in 1 2 3; do
    run_workload rendering_proxy 0 $run
done

echo "=== GPU 0 extra collection complete ==="
