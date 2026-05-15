#!/usr/bin/env bash
# run_amd_recovery.sh -- Serial recovery after the pgrep contamination bug.
#
# Runs (no waits, no pgrep): Tier 2 → Tier 4 → Tier 3 → contaminated sampling-rate redo.
# Sequential: each sub-script runs to completion before the next starts,
# because they're invoked in serial from this single parent shell.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if [[ -z "${VIRTUAL_ENV:-}" ]]; then
    # shellcheck disable=SC1091
    source "${REPO_ROOT}/.venv/bin/activate"
fi

START=$(date +%s)
echo "=== AMD recovery sweep starting at $(date -Iseconds) ==="

echo ""; echo "=== Tier 2 (12 missing adversarial families × 3 reps = 36 runs) ==="
scripts/run_amd_tier2.sh --reps 3 --no-wait-sampling

echo ""; echo "=== Tier 4 (5 broader workloads × 3 reps = 15 runs) ==="
scripts/run_amd_tier4.sh --reps 3 --no-wait-prior

echo ""; echo "=== Tier 3 (4 whitebox variants × 3 reps = 12 runs) ==="
scripts/run_amd_tier3_whitebox.sh --reps 3 --no-wait-prior

# --- Re-collect the 9 contaminated sampling-rate runs ---
echo ""; echo "=== Sampling-rate redo (idle ×8 + resnet AMP @ 0.5 Hz × 1) ==="

LOG_DIR="data/collection_logs/amd_sampling"
mkdir -p "${LOG_DIR}"
STAMP=$(date +%Y%m%d_%H%M%S)
SUMMARY="${LOG_DIR}/sweep_recover_${STAMP}.summary"
echo "Sampling rate redo starting at $(date -Iseconds)" > "${SUMMARY}"

run_one() {
    local wl="$1" rate="$2" rep="$3"
    local interval
    interval=$(python3 -c "print(1.0/${rate})")
    local tag
    tag=$(echo "${rate}hz_rep${rep}" | tr '.' 'p')
    local log="${LOG_DIR}/${STAMP}_${wl}_${tag}.log"
    printf "[%s @ %.1f Hz rep %d] -> %s\n" "${wl}" "${rate}" "${rep}" "${log}" | tee -a "${SUMMARY}"
    set +e
    python scripts/run_workload_amd.py \
        --workload "${wl}" \
        --gpu-index 0 --interval "${interval}" \
        --run-id "rate${tag}" --no-rdc > "${log}" 2>&1
    local ec=$?
    set -e
    echo "    exit=${ec}" | tee -a "${SUMMARY}"
}

for rate in 1.0 2.0 5.0 10.0; do
    for rep in 1 2; do
        run_one "idle" "${rate}" "${rep}"
    done
done
# Single contaminated pytorch_resnet AMP @ 0.5 Hz (rep 1 was the contaminated one,
# rep 2 ran clean at 02:14+ when Tier 4 was finishing — but to be safe redo rep 1).
run_one "pytorch_resnet_cifar10_amp" "0.5" "1"

ELAPSED=$(( $(date +%s) - START ))
echo "" >> "${SUMMARY}"
echo "Recovery sweep done at $(date -Iseconds) (total ${ELAPSED}s)" >> "${SUMMARY}"
echo ""
echo "=== Recovery done. Total elapsed: ${ELAPSED}s ==="
