#!/usr/bin/env bash
# run_amd_tier3_canonical.sh -- Paper-canonical whitebox sweep on AMD.
#
# Replicates the NVIDIA registry's Round-5 whitebox set (TinyLlama-1.1B
# + grad-ckpt + dilution N in {5, 10, 20}). Adds the missing N=5 and N=20
# variants that the earlier Tier-3 collection didn't cover.
#
# 5 workloads (WB-I, WB-D N5, WB-D N20, WB-L N5, WB-L N20) × 3 reps = 15 runs.
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
    whitebox_inference
    whitebox_diluted_N5
    whitebox_diluted_N20
    whitebox_lora_N5
    whitebox_lora_N20
)

LOG_DIR="data/collection_logs/amd_corpus"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/tier3_canonical_${STAMP}.summary"
{
    echo "AMD Tier 3 canonical whitebox sweep starting at $(date -Iseconds)"
    echo "Workloads (${#WORKLOADS[@]}): ${WORKLOADS[*]}"
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
        log="${LOG_DIR}/${STAMP}_tier3c_${wl}_rep${rep}.log"
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
