#!/usr/bin/env bash
# run_amd_datacenter_corpus.sh -- Collection sweep for datacenter AMD nodes
# (MI300X / MI325X / MI355). Runs on a freshly-provisioned cloud instance
# after setup_amd_datacenter.sh has installed ROCm + venv.
#
# Two phases:
#   1. Single-GPU sweep: same as run_amd_corpus.sh on the local card, using
#      gpu-index 0 so we capture the per-card signature comparable to consumer.
#   2. Multi-GPU DDP adversarial: K_ddp, L_ddp_{5,10}, L_ddp_stagger_5,
#      K_ddp_accum_{4,8} — only runs on >=8-GPU nodes.
#
# Usage:
#   scripts/run_amd_datacenter_corpus.sh --reps 3 [--skip-single-gpu] [--skip-multi-gpu]
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"
if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

REPS=3
RUN_SINGLE=1
RUN_MULTI=1

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reps) REPS="$2"; shift 2 ;;
        --skip-single-gpu) RUN_SINGLE=0; shift ;;
        --skip-multi-gpu) RUN_MULTI=0; shift ;;
        *) echo "Unknown arg: $1"; exit 2 ;;
    esac
done

# Detect number of GPUs visible to amdsmi
N_GPUS=$(python3 - <<'PY'
import sys
sys.path.insert(0, "/opt/rocm/share/amd_smi")
import amdsmi
amdsmi.amdsmi_init()
n = 0
for h in amdsmi.amdsmi_get_processor_handles():
    try:
        amdsmi.amdsmi_get_gpu_activity(h)
        n += 1
    except Exception:
        pass
print(n)
amdsmi.amdsmi_shut_down()
PY
)
echo "Detected ${N_GPUS} AMD GPU(s)."

if [[ "${RUN_SINGLE}" == "1" ]]; then
    echo "=== Phase 1: single-GPU sweep (gpu 0, ${REPS} reps) ==="
    scripts/run_amd_corpus.sh --reps "${REPS}" --all --gpu-index 0
fi

if [[ "${RUN_MULTI}" == "1" && "${N_GPUS}" -ge 2 ]]; then
    echo "=== Phase 2: multi-GPU DDP adversarial sweep (${REPS} reps each) ==="
    DDP_WORKLOADS=(
        adversarial_K_ddp
        adversarial_L_ddp_5
        adversarial_L_ddp_10
        adversarial_L_ddp_stagger_5
        adversarial_K_ddp_accum_4
        adversarial_K_ddp_accum_8
    )
    LOG_DIR="data/collection_logs/amd_corpus"
    mkdir -p "${LOG_DIR}"
    STAMP=$(date +%Y%m%d_%H%M%S)
    SUMMARY="${LOG_DIR}/ddp_dc_${STAMP}.summary"
    {
        echo "DDP adversarial sweep on ${N_GPUS}-GPU node, starting $(date -Iseconds)"
        echo "Workloads: ${DDP_WORKLOADS[*]}"
        echo "Reps: ${REPS}"
    } > "${SUMMARY}"
    cat "${SUMMARY}"
    for wl in "${DDP_WORKLOADS[@]}"; do
        for rep in $(seq 1 "${REPS}"); do
            log="${LOG_DIR}/${STAMP}_ddp_${wl}_rep${rep}.log"
            echo "[${wl} rep ${rep}] -> ${log}" | tee -a "${SUMMARY}"
            set +e
            python scripts/run_workload_amd.py \
                --workload "${wl}" --gpu-index 0 --interval 1.0 --no-rdc \
                > "${log}" 2>&1
            ec=$?
            set -e
            echo "    exit=${ec}" | tee -a "${SUMMARY}"
        done
    done
    echo "DDP sweep done at $(date -Iseconds)" >> "${SUMMARY}"
elif [[ "${RUN_MULTI}" == "1" ]]; then
    echo "=== Phase 2 skipped — only 1 GPU visible ==="
fi

echo "=== Datacenter corpus collection done ==="
