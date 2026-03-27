#!/bin/bash
# Collect modern inference workloads on H100 NVL
set -e
cd /home/claude/SPAR-GPU-monitoring
export HF_HOME=/home/claude/.cache/huggingface
OUTDIR="data/h100_nvl"
GPU=$1

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

if [ "$GPU" = "0" ]; then
    echo "=== GPU 0: DeepSeek-R1 + Whisper ==="
    for run in 1 2 3; do
        run_workload deepseek_r1_inference 0 $run
    done
    for run in 1 2 3; do
        run_workload whisper_inference 0 $run
    done
    echo "=== GPU 0 inference complete ==="
else
    echo "=== GPU 1: Qwen 2.5 ==="
    for run in 1 2 3; do
        run_workload qwen25_inference 1 $run
    done
    echo "=== GPU 1 inference complete ==="
fi
