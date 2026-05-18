#!/usr/bin/env bash
# run_amd_rerun_failures.sh -- Re-run the 9 LLM-family workloads that were
# blocked by the joblib pyc-cache bug in Pass-1.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

WORKLOADS=(
    deepseek_r1_inference
    qwen25_inference
    whisper_inference
    llm_infer_phi3_mini
    llm_infer_qwen25_1p5b
    llm_infer_smollm2_1p7b
    llm_infer_tinyllama_bs8
    llm_infer_clip_embed_bs64
    llm_infer_vit_l_bs64
)

LOG_DIR="data/collection_logs/amd_corpus"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/run_${STAMP}_rerun.summary"
{
    echo "AMD rerun (LLM failures) starting at $(date -Iseconds)"
    echo "Workloads: ${WORKLOADS[*]}"
} > "${SUMMARY}"

cat "${SUMMARY}"

START_EPOCH=$(date +%s)
RUN_IDX=0
for wl in "${WORKLOADS[@]}"; do
    RUN_IDX=$(( RUN_IDX + 1 ))
    ELAPSED=$(( $(date +%s) - START_EPOCH ))
    log="${LOG_DIR}/${STAMP}_${wl}_rerun.log"
    printf "[%d/%d  t+%ds] %s -> %s\n" \
        "${RUN_IDX}" "${#WORKLOADS[@]}" "${ELAPSED}" "${wl}" "${log}" \
        | tee -a "${SUMMARY}"
    set +e
    python scripts/run_workload_amd.py \
        --workload "${wl}" \
        --gpu-index 0 \
        --interval 1.0 \
        --no-rdc > "${log}" 2>&1
    ec=$?
    set -e
    echo "    exit=${ec}" | tee -a "${SUMMARY}"
done

echo "Done at $(date -Iseconds)" >> "${SUMMARY}"
echo "Summary: ${SUMMARY}"
