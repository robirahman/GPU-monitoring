#!/usr/bin/env bash
# run_amd_tier2.sh -- Tier 2 adversarial coverage: 12 missing evasion families,
# 3 reps each. Skips strategies G (clock_throttled) and J (PID thermal-matched)
# which need NVIDIA-only tooling.
#
# Waits for the sampling-rate sweep to finish first if --wait-sampling is set.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

REPS=3
WAIT_SAMPLING=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2 ;;
        --no-wait-sampling) WAIT_SAMPLING=0; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

WORKLOADS=(
    adversarial_E_memory_minimal
    adversarial_F_interleave
    adversarial_H_mimicry_mining
    adversarial_H_mimicry_cufft
    adversarial_I_stochastic
    adversarial_M_composite_memmin
    adversarial_N_grad_accum
    adversarial_composite_idle_00
    adversarial_composite_idle_20
    adversarial_composite_idle_50
    adversarial_composite_idle_80
    adversarial_composite_idle_95
)

if [[ "${WAIT_SAMPLING}" == "1" ]]; then
    echo "=== Waiting for sampling-rate sweep to finish ==="
    while pgrep -f 'run_amd_sampling_sweep' >/dev/null 2>&1; do
        sleep 30
    done
    echo "Sampling sweep done at $(date -Iseconds)"
fi

LOG_DIR="data/collection_logs/amd_corpus"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/tier2_${STAMP}.summary"
{
    echo "AMD Tier 2 adversarial sweep starting at $(date -Iseconds)"
    echo "Workloads (${#WORKLOADS[@]}): ${WORKLOADS[*]}"
    echo "Reps per workload: ${REPS}"
} > "${SUMMARY}"
cat "${SUMMARY}"

RUN_IDX=0
TOTAL=$(( ${#WORKLOADS[@]} * REPS ))
START=$(date +%s)
for wl in "${WORKLOADS[@]}"; do
    for rep in $(seq 1 ${REPS}); do
        RUN_IDX=$(( RUN_IDX + 1 ))
        ELAPSED=$(( $(date +%s) - START ))
        log="${LOG_DIR}/${STAMP}_tier2_${wl}_rep${rep}.log"
        printf "[%d/%d  t+%ds] %s rep %d -> %s\n" \
            "${RUN_IDX}" "${TOTAL}" "${ELAPSED}" "${wl}" "${rep}" "${log}" \
            | tee -a "${SUMMARY}"
        set +e
        python scripts/run_workload_amd.py \
            --workload "${wl}" \
            --gpu-index 0 --interval 1.0 --no-rdc > "${log}" 2>&1
        ec=$?
        set -e
        echo "    exit=${ec}" | tee -a "${SUMMARY}"
    done
done

echo "Done at $(date -Iseconds)" >> "${SUMMARY}"
echo "Summary: ${SUMMARY}"
