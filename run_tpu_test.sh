#!/bin/bash
cd ~/SPAR-GPU-monitoring

# 1. Idle run
echo "Starting idle telemetry (30s)..."
python3 scripts/collect_tpu_telemetry.py --duration 30 --label idle --output data/tpu_idle.parquet &
IDLE_PID=$!
sleep 35
wait $IDLE_PID

# 2. Training run
echo "Starting training telemetry..."
# Start telemetry in background manually so we can wrap the training
python3 scripts/collect_tpu_telemetry.py --duration 120 --label tpu_training --output data/tpu_training.parquet &
TEL_PID=$!

echo "Launching TPU training..."
export PJRT_DEVICE=TPU
python3 workloads/tpu_pytorch_training.py --epochs 2 --batch-size 128

wait $TEL_PID
echo "Test complete."
