#!/bin/bash
# Collect non-training workloads on GPU 1
set -e
cd /home/claude/SPAR-GPU-monitoring
OUTDIR="data/h100_nvl"
mkdir -p "$OUTDIR"

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

echo "=== Non-training workloads on GPU 1 ==="

# Scientific HPC (3 runs each, 600s each → reduced to 300s for speed)
for run in 1 2 3; do
    run_workload cufft_benchmark 1 $run
    run_workload nbody_sim 1 $run
done

# Crypto mining (3 runs)
for run in 1 2 3; do
    run_workload mining_ethash_proxy 1 $run
done

# Rendering (3 runs)
for run in 1 2 3; do
    run_workload rendering_proxy 1 $run
done

echo "=== GPU 1 collection complete ==="
