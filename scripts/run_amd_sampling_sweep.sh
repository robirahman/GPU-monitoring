#!/usr/bin/env bash
# run_amd_sampling_sweep.sh -- Collect one ML training, one ML inference,
# one adversarial, and idle at varying telemetry sample rates: 0.5 / 1 / 2 / 5 / 10 Hz.
#
# Output files have the sample rate encoded in run_id so downstream
# feature extraction can stratify.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

WORKLOADS=(
    idle
    pytorch_resnet_cifar10_amp     # training (AMP variant runs fast vs FP32 1200s timeout)
    llm_infer_phi3_mini             # inference
    adversarial_K_online_learning   # adversarial
)
RATES=(0.5 1.0 2.0 5.0 10.0)
REPS=2

LOG_DIR="data/collection_logs/amd_sampling"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/sweep_${STAMP}.summary"
{
    echo "AMD sampling-rate sweep starting at $(date -Iseconds)"
    echo "Workloads: ${WORKLOADS[*]}"
    echo "Rates (Hz): ${RATES[*]}"
    echo "Reps: ${REPS}"
} > "${SUMMARY}"
cat "${SUMMARY}"

RUN_IDX=0
TOTAL=$(( ${#WORKLOADS[@]} * ${#RATES[@]} * REPS ))
START=$(date +%s)
for wl in "${WORKLOADS[@]}"; do
    for rate in "${RATES[@]}"; do
        # amdsmi --interval expects seconds, so convert Hz to interval
        interval=$(python3 -c "print(1.0/${rate})")
        for rep in $(seq 1 ${REPS}); do
            RUN_IDX=$(( RUN_IDX + 1 ))
            ELAPSED=$(( $(date +%s) - START ))
            tag=$(echo "${rate}hz_rep${rep}" | tr '.' 'p')
            log="${LOG_DIR}/${STAMP}_${wl}_${tag}.log"
            printf "[%d/%d  t+%ds] %s @ %.1f Hz (interval=%.4fs) rep %d\n" \
                "${RUN_IDX}" "${TOTAL}" "${ELAPSED}" "${wl}" "${rate}" "${interval}" "${rep}" \
                | tee -a "${SUMMARY}"
            set +e
            python scripts/run_workload_amd.py \
                --workload "${wl}" \
                --gpu-index 0 \
                --interval "${interval}" \
                --no-rdc \
                --run-id "rate${tag}" > "${log}" 2>&1
            ec=$?
            set -e
            echo "    exit=${ec}" | tee -a "${SUMMARY}"
        done
    done
done

echo "Done at $(date -Iseconds)" >> "${SUMMARY}"
echo "Summary: ${SUMMARY}"
