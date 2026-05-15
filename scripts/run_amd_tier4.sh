#!/usr/bin/env bash
# run_amd_tier4.sh -- Tier 4 broader workload diversity.
#
# Workloads: SDXL inference, 2 LoRA fine-tunes, YOLO training, Qwen 2.5 3B
# inference. 3 reps each = 15 runs.
#
# Waits for Tier 2 to finish first (the GPU can only run one at a time).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

REPS=3
WAIT_PRIOR=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2 ;;
        --no-wait-prior) WAIT_PRIOR=0; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

WORKLOADS=(
    llm_infer_qwen25_3b
    sdxl_inference
    lora_finetune_smollm2_135m
    lora_finetune_qwen25_0p5b
    yolo_train_coco128
)

if [[ "${WAIT_PRIOR}" == "1" ]]; then
    echo "=== Waiting for prior sweeps (sampling + tier2) to finish ==="
    while pgrep -f 'run_amd_sampling_sweep\|run_amd_tier2' >/dev/null 2>&1; do
        sleep 30
    done
    echo "Prior sweeps done at $(date -Iseconds)"
fi

LOG_DIR="data/collection_logs/amd_corpus"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/tier4_${STAMP}.summary"
{
    echo "AMD Tier 4 (broader workloads) starting at $(date -Iseconds)"
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
        log="${LOG_DIR}/${STAMP}_tier4_${wl}_rep${rep}.log"
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
