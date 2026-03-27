#!/bin/bash
# Run evasion tax sweep on one GPU
# Usage: bash run_evasion_sweep.sh <GPU_INDEX> <FRACTIONS...>
# Example: bash run_evasion_sweep.sh 0 0.0 0.1 0.2 0.3 0.4 0.5
set -e
cd /home/claude/SPAR-GPU-monitoring
OUTDIR="data/h100_nvl/evasion_sweep"
mkdir -p "$OUTDIR"
GPU=$1
shift
FRACTIONS="$@"

echo "=== Evasion Sweep on GPU $GPU ==="
echo "Fractions: $FRACTIONS"

for frac in $FRACTIONS; do
    frac_int=$(echo "$frac * 100" | bc | cut -d. -f1)
    label="adversarial_composite_$(printf '%02d' $frac_int)"
    for run in 1 2; do
        echo "[$(date +%H:%M:%S)] $label run $run on GPU $GPU..."
        python3 scripts/run_workload.py \
            --workload "$label" \
            --gpu-index "$GPU" \
            --output-dir "$OUTDIR" \
            --no-dcgm \
            -- python3 workloads/adversarial_composite.py \
                --idle-fraction "$frac" --duration 120 \
            2>&1 | tail -3
        echo "[$(date +%H:%M:%S)] Done: $label run $run"
    done
done

echo "=== GPU $GPU sweep complete ==="
