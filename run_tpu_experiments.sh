#!/bin/bash
cd ~/SPAR-GPU-monitoring

# 1. Training Trace
echo "Starting TPU Training Trace (300s)..."
python3 scripts/collect_tpu_telemetry.py --duration 330 --label tpu_jax_training --output data/tpu_jax_training.parquet &
TEL_PID=$!

python3 workloads/tpu_jax_training.py
wait $TEL_PID

# Delay between tests
echo "Cooling down for 10s..."
sleep 10

# 2. Inference Trace
echo "Starting TPU Inference Trace (300s)..."
python3 scripts/collect_tpu_telemetry.py --duration 330 --label tpu_jax_inference --output data/tpu_jax_inference.parquet &
TEL_PID=$!

python3 workloads/tpu_jax_inference.py
wait $TEL_PID

echo "All TPU traces complete."