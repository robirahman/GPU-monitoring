#!/usr/bin/env bash
set -u
cd /workspace/SPAR-GPU-monitoring
export PYTHONPATH=/usr/share/datacenter-gpu-manager-4/bindings/python3:${PYTHONPATH:-}

LOG=/workspace/SPAR-GPU-monitoring/logs/b200_nonml_$(date +%Y%m%d_%H%M%S).log

echo "==== B200 non-ML collection started $(date -u) ====" | tee -a "$LOG"

for REP in 1 2 3; do
  for WL in idle cufft_benchmark mining_ethash_proxy; do
    echo "---- rep=$REP workload=$WL $(date -u) ----" | tee -a "$LOG"
    python3 scripts/run_workload.py --workload "$WL" --output-dir data/ >>"$LOG" 2>&1
    echo "exit=$?" | tee -a "$LOG"
  done
done

echo "==== B200 non-ML collection finished $(date -u) ====" | tee -a "$LOG"
