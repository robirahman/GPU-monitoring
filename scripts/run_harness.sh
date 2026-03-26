#!/usr/bin/env bash
set -euo pipefail

GPU="${GPU:-0}"
HZ="${HZ:-1}"
SECONDS="${SECONDS:-160}"
WORKLOAD="${WORKLOAD:-ml}"   # ml | nonml

STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TEL="data/raw_telemetry/nvml_gpu${GPU}_${STAMP}.parquet"
LOG="data/raw_telemetry/workload_${STAMP}.log"

mkdir -p data/raw_telemetry

echo "[1] telemetry -> ${TEL}"
python scripts/collect_nvml_telemetry.py --gpu "${GPU}" --seconds "${SECONDS}" --hz "${HZ}" --out "${TEL}" &
TEL_PID=$!

sleep 2
echo "[2] workload (${WORKLOAD}) -> ${LOG}"
if [[ "${WORKLOAD}" == "ml" ]]; then
  python scripts/gpu_workloads.py | tee "${LOG}"
elif [[ "${WORKLOAD}" == "nonml" ]]; then
  python scripts/non_ml_gpu_workload.py | tee "${LOG}"
else
  echo "Unknown WORKLOAD=${WORKLOAD} (use ml|nonml)" >&2
  kill "${TEL_PID}" || true
  exit 1
fi

echo "[3] wait telemetry"
wait "${TEL_PID}"

echo "[4] label"
python scripts/label_telemetry_from_workload_log.py --telemetry "${TEL}" --workload_log "${LOG}"

LAB="${TEL%.parquet}_labeled.parquet"
echo "[5] features"
python scripts/build_features.py --infile "${LAB}"

FEAT="${TEL%.parquet}_features.parquet"
echo "[6] plots"
python scripts/plot_week2_summary.py --infile "${FEAT}"

echo "DONE:"
echo "  ${TEL}"
echo "  ${LOG}"
echo "  ${LAB}"
echo "  ${FEAT}"
