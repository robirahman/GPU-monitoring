#!/usr/bin/env bash
# run_amd_corpus.sh -- Sweep every workload in workloads/registry_amd.py
# N times each on the local AMD GPU.
#
# Usage:
#   scripts/run_amd_corpus.sh [--reps N] [--clean-only|--adv-only|--all] \
#                             [--gpu-index 0] [--dry-run]
#
# Output: parquets under data/, per-run stderr/stdout under data/collection_logs/amd_corpus/
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
cd "${REPO_ROOT}"

# Make sure the venv is active.
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

# ROCm runtime env (LD_LIBRARY_PATH for RDC + PyTorch, RDC_HOME).
# shellcheck disable=SC1091
source "${REPO_ROOT}/scripts/rocm_env.sh"

REPS=1
SELECT="all"
GPU_INDEX=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2 ;;
        --clean-only) SELECT="clean"; shift ;;
        --adv-only) SELECT="adv"; shift ;;
        --all) SELECT="all"; shift ;;
        --gpu-index) GPU_INDEX="$2"; shift 2 ;;
        --dry-run) DRY_RUN=1; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

CLEAN_WORKLOADS=(
    idle
    pytorch_resnet_cifar10
    pytorch_resnet_cifar10_amp
    pytorch_mlp_cifar10
    gpt2_wikitext2
    gpt2_wikitext2_amp
    bert_sst2
    bert_sst2_amp
    resnet50_inference
    cufft_benchmark
    nbody_sim
    mining_ethash_proxy
    rendering_proxy
    ffmpeg_vaapi
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

ADV_WORKLOADS=(
    adversarial_A_util_modulation
    adversarial_B_low_util
    adversarial_D_temporal_disruption
    adversarial_K_online_learning
    adversarial_L_diluted_5
    adversarial_L_diluted_10
)

case "${SELECT}" in
    clean) WORKLOADS=("${CLEAN_WORKLOADS[@]}") ;;
    adv)   WORKLOADS=("${ADV_WORKLOADS[@]}") ;;
    all)   WORKLOADS=("${CLEAN_WORKLOADS[@]}" "${ADV_WORKLOADS[@]}") ;;
esac

LOG_DIR="data/collection_logs/amd_corpus"
mkdir -p "${LOG_DIR}"

STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/run_${STAMP}.summary"
{
    echo "AMD corpus sweep starting at $(date -Iseconds)"
    echo "Selection: ${SELECT} (${#WORKLOADS[@]} workloads × ${REPS} reps = $(( ${#WORKLOADS[@]} * REPS )) runs)"
    echo "GPU index: ${GPU_INDEX}"
    echo "Workloads:"
    printf '  %s\n' "${WORKLOADS[@]}"
    echo ""
} > "${SUMMARY}"

cat "${SUMMARY}"
if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[dry-run] exiting before launching any workloads."
    exit 0
fi

RUN_IDX=0
TOTAL=$(( ${#WORKLOADS[@]} * REPS ))
START_EPOCH=$(date +%s)

for wl in "${WORKLOADS[@]}"; do
    for rep in $(seq 1 "${REPS}"); do
        RUN_IDX=$(( RUN_IDX + 1 ))
        ELAPSED=$(( $(date +%s) - START_EPOCH ))
        log="${LOG_DIR}/${STAMP}_${wl}_rep${rep}.log"
        printf "[%d/%d  t+%ds] %s rep %d -> %s\n" \
            "${RUN_IDX}" "${TOTAL}" "${ELAPSED}" "${wl}" "${rep}" "${log}" \
            | tee -a "${SUMMARY}"
        set +e
        python scripts/run_workload_amd.py \
            --workload "${wl}" \
            --gpu-index "${GPU_INDEX}" \
            --interval 1.0 \
            --no-rdc \
            > "${log}" 2>&1
        ec=$?
        set -e
        echo "    exit=${ec}" | tee -a "${SUMMARY}"
        if [[ ${ec} -ne 0 ]]; then
            echo "    (continuing despite non-zero exit)" | tee -a "${SUMMARY}"
        fi
    done
done

echo "" >> "${SUMMARY}"
echo "Done at $(date -Iseconds)" >> "${SUMMARY}"
echo "" | tee -a "${SUMMARY}"
echo "Summary: ${SUMMARY}"
