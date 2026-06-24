#!/usr/bin/env bash
# Collect the SPAR workload suite on the local testbed (node1, 3x RTX 3090),
# logging BOTH NVML telemetry AND physical sensors (mic + GPU/MOBO current
# clamps) under a shared run_id per run.
#
# Runs ONE workload at a time on GPU 0 (GPUs 1,2 left idle) because the mic and
# MOBO current rail are shared across the whole node -- concurrent workloads
# would contaminate the physical-sensor labels. Mirrors the paper's wall-power
# protocol (single workload at a time). Each workload is run TWICE for grouped-CV
# run diversity. NVML DCGM (Tier 2) disabled (not installed in this container).
set -u
cd "$(dirname "$0")/.."
PY=/opt/conda/bin/python
OUT=data/
export CIFAR_ROOT=/workspace/.cache/cifar
LOG=/tmp/claude-0/-root/60905851-471d-4cc2-9b02-818b69352841/scratchpad/collect_logs/suite.log
mkdir -p "$(dirname "$LOG")"
GPU=0
REPEATS=2

# workload : timeout_seconds  (training / inference / HPC / mining / rendering / adversarial)
read -r -d '' SUITE <<'EOF'
idle 120
pytorch_resnet_cifar10 200
pytorch_resnet_cifar10_amp 200
pytorch_mlp_cifar10 180
gpt2_wikitext2 220
gpt2_wikitext2_amp 220
bert_sst2 220
bert_sst2_amp 220
resnet50_inference 200
cufft_benchmark 180
nbody_sim 180
mining_ethash_proxy 180
rendering_proxy 180
adversarial_K_online_learning 200
adversarial_L_diluted_5 200
adversarial_L_diluted_10 200
adversarial_composite_50 200
adversarial_H_mimicry_cufft 200
adversarial_J_pid_75 200
EOF

echo "[suite] START $(date -u +%H:%M:%S)  gpu=$GPU repeats=$REPEATS" > "$LOG"
for rep in $(seq 1 "$REPEATS"); do
  while read -r wl to; do
    [ -z "$wl" ] && continue
    echo "[suite] rep$rep >>> $wl (timeout ${to}s) $(date -u +%H:%M:%S)" >> "$LOG"
    $PY scripts/run_workload_with_sensors.py --workload "$wl" --gpu-index "$GPU" \
        --no-dcgm --timeout "$to" --output-dir "$OUT" >> "$LOG" 2>&1
    echo "[suite] rep$rep <<< $wl rc=$? $(date -u +%H:%M:%S)" >> "$LOG"
  done <<< "$SUITE"
done
echo "[suite] COMPLETE $(date -u +%H:%M:%S)" >> "$LOG"
