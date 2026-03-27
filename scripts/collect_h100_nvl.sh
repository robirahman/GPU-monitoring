#!/bin/bash
# Collect H100 NVL telemetry for all core workloads
# GPU 0: training + inference workloads
# GPU 1: non-training workloads (HPC, mining, rendering)
# Each workload runs 3 times for statistical robustness

set -e
cd /home/claude/SPAR-GPU-monitoring
OUTDIR="data/h100_nvl"
mkdir -p "$OUTDIR"

echo "=== H100 NVL Telemetry Collection ==="
echo "Output: $OUTDIR"
echo ""

# Function to run a workload with retries
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

# === GPU 0: ML Training workloads (3 runs each) ===
echo "=== ML Training workloads on GPU 0 ==="
for run in 1 2 3; do
    run_workload pytorch_resnet_cifar10 0 $run
    run_workload pytorch_resnet_cifar10_amp 0 $run
    run_workload pytorch_mlp_cifar10 0 $run
    run_workload bert_sst2 0 $run
    run_workload bert_sst2_amp 0 $run
    run_workload gpt2_wikitext2 0 $run
    run_workload gpt2_wikitext2_amp 0 $run
done

echo "=== ML Inference on GPU 0 ==="
for run in 1 2 3; do
    run_workload resnet50_inference 0 $run
done

echo "=== GPU 0 collection complete ==="
