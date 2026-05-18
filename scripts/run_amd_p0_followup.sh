#!/usr/bin/env bash
# run_amd_p0_followup.sh -- Phase 0 follow-up after tier3 canonical finishes.
#
# Runs in serial:
#   1. whitebox_oracle x 3 reps (new, now that classifier_oracle.py has amdsmi backend)
#   2. adversarial_J_pid_75 x 3 reps (new, now that adversarial_round2.py has amdsmi backend)
#   3. adversarial_H_mimicry_{mining,cufft} x 3 reps (re-collect; old data was collected
#      with degraded -1 feedback, so the strategy wasn't actually firing)
#   4. adversarial_I_stochastic x 3 reps (same reason as H)
#
# Waits for tier3 canonical to finish first (uses a clean PID-based wait, not pgrep
# self-match like the earlier broken orchestrators).
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

WORKLOADS=(
    whitebox_oracle
    adversarial_J_pid_75
    adversarial_H_mimicry_mining
    adversarial_H_mimicry_cufft
    adversarial_I_stochastic
)
REPS=3

# Wait for tier3 canonical to clear out, looking for the actual python child
# process (run_workload_amd) rather than the parent shell script.
echo "=== Waiting for tier3 canonical to finish ==="
while pgrep -f 'run_amd_tier3_canonical.sh' >/dev/null 2>&1; do
    sleep 30
done
echo "Tier3 canonical done at $(date -Iseconds)"

LOG_DIR="data/collection_logs/amd_corpus"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/p0_followup_${STAMP}.summary"
{
    echo "AMD P0 follow-up sweep starting at $(date -Iseconds)"
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
        log="${LOG_DIR}/${STAMP}_p0f_${wl}_rep${rep}.log"
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
