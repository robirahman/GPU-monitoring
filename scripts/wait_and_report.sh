#!/usr/bin/env bash
# Waits for GPU 0 and GPU 1 collection to finish, then runs post_collection.py
set -euo pipefail

GPU0_LOG=/workspace/SPAR-GPU-monitoring/data/collection_log.txt
GPU1_LOG=/workspace/SPAR-GPU-monitoring/data/collection_gpu1_log.txt

echo "=== Watcher started at $(date) ==="
echo "Waiting for both GPU collections to complete..."

# Poll until both logs show COMPLETE
while true; do
    gpu0_done=0
    gpu1_done=0
    [[ -f "$GPU0_LOG" ]] && grep -q "Collection COMPLETE" "$GPU0_LOG" && gpu0_done=1
    [[ -f "$GPU1_LOG" ]] && grep -q "collection COMPLETE" "$GPU1_LOG" && gpu1_done=1

    files=$(ls /workspace/SPAR-GPU-monitoring/data/*.parquet 2>/dev/null | wc -l)
    echo "[$(date +%H:%M:%S)] GPU0 done=$gpu0_done  GPU1 done=$gpu1_done  parquets=$files"

    if [[ $gpu0_done -eq 1 && $gpu1_done -eq 1 ]]; then
        echo "Both collections complete!"
        break
    fi
    sleep 120  # check every 2 minutes
done

echo ""
echo "=== Running post_collection.py at $(date) ==="
export PATH=/venv/main/bin:$PATH
export HF_HOME=/root/.hf_home
cd /workspace/SPAR-GPU-monitoring
python3 scripts/post_collection.py 2>&1 | tee /workspace/SPAR-GPU-monitoring/data/post_collection_log.txt
echo "=== Pipeline complete at $(date) ==="
