#!/usr/bin/env bash
# W05 (primary) -- REAL MoE expert-parallel TRAINING via torchtitan on VCF h100-sxm-8-0.
#
# Usage (from repo root, inside an interactive allocation with 8 GPUs)::
#
#   bash scripts/run_moe_ep_train_vcf.sh check     # list jobs, print the launch command, validate flags
#   bash scripts/run_moe_ep_train_vcf.sh dsv3      # DeepSeek-MoE-16B, EP=8 (torchtitan deepseek_v3_16b)
#   bash scripts/run_moe_ep_train_vcf.sh qwen3     # Qwen3-30B-A3B, EP=8 (memory fit unverified)
#   STEPS=100 bash scripts/run_moe_ep_train_vcf.sh dsv3
#
# Each run goes through the telemetry launcher (NVML + DCGM NVLink counters for the same window) and
# writes NCCL's own per-collective debug log; convert with:
#   python scripts/parse_nccl_coll_log.py data/titan/<job>/nccl.*.log --cuda-dev 0 --out data/<job>_nccl.jsonl
#
# Requirements: Python >= 3.11, a recent PyTorch, `pip install torchtitan` (0.3.0 CLI). c4 is streamed
# from the HF hub. See docs/interconnect_W_id_file_map.md (W05). Not yet run on the cluster from this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-check}"
STEPS_ARG=()
[[ -n "${STEPS:-}" ]] && STEPS_ARG=(-- --training.steps "$STEPS")

run_one() {
  local label="$1"
  echo "=== $label ==="
  python scripts/run_workload_multi_gpu.py --workload "$label" --num-gpus 8 --timeout 5400
}

case "$PROFILE" in
  check)
    python scripts/run_titan_job.py --list
    python scripts/run_titan_job.py --job moe_ep_train_dsv3_16b_ep8 --dry-run
    ;;
  dsv3)  run_one moe_ep_train_dsv3_16b_ep8 ;;
  qwen3) run_one moe_ep_train_qwen3_30b_a3b_ep8 ;;
  *)
    echo "Unknown profile: $PROFILE (use: check | dsv3 | qwen3)" >&2
    exit 1
    ;;
esac
