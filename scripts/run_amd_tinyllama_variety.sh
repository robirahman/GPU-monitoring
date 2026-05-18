#!/usr/bin/env bash
# run_amd_tinyllama_variety.sh -- Add TinyLlama inference variety to the
# corpus to fix the WB-I corpus-coverage gap.
#
# 5 new variants x 3 reps = 15 runs.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

REPS=3
while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2 ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

WORKLOADS=(
    llm_infer_tinyllama_synthetic_bs1
    llm_infer_tinyllama_bs2
    llm_infer_tinyllama_bs4
    llm_infer_tinyllama_seq1024
    llm_infer_tinyllama_seq2048
)

LOG_DIR="data/collection_logs/amd_corpus"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/tinyllama_variety_${STAMP}.summary"
{
    echo "TinyLlama variety pack starting at $(date -Iseconds)"
    echo "Workloads: ${WORKLOADS[*]}"
    echo "Reps: ${REPS}"
} > "${SUMMARY}"
cat "${SUMMARY}"

RUN_IDX=0
TOTAL=$(( ${#WORKLOADS[@]} * REPS ))
START=$(date +%s)
for wl in "${WORKLOADS[@]}"; do
    for rep in $(seq 1 ${REPS}); do
        RUN_IDX=$(( RUN_IDX + 1 ))
        ELAPSED=$(( $(date +%s) - START ))
        log="${LOG_DIR}/${STAMP}_tlv_${wl}_rep${rep}.log"
        printf "[%d/%d  t+%ds] %s rep %d\n" \
            "${RUN_IDX}" "${TOTAL}" "${ELAPSED}" "${wl}" "${rep}" \
            | tee -a "${SUMMARY}"
        set +e
        python scripts/run_workload_amd.py --workload "${wl}" \
            --gpu-index 0 --interval 1.0 --no-rdc > "${log}" 2>&1
        ec=$?
        set -e
        echo "    exit=${ec}" | tee -a "${SUMMARY}"
    done
done
echo "Done at $(date -Iseconds)" >> "${SUMMARY}"
echo "Summary: ${SUMMARY}"
