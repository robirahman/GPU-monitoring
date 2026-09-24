#!/usr/bin/env bash
# SPAR M4/M5 — MoE EP serving on VCF h100-sxm-8-0
#
# Usage (from repo root, inside an interactive allocation with 8 GPUs)::
#
#   bash scripts/run_moe_serving_vcf.sh m4_olmoe      # small MoE EP smoke
#   bash scripts/run_moe_serving_vcf.sh m4_mixtral    # Mixtral EP=8
#   bash scripts/run_moe_serving_vcf.sh m5_mixed      # Mixtral EP + mixed traffic
#   bash scripts/run_moe_serving_vcf.sh m5_dense_ctl  # dense TP control (no EP)
#
# Or via multi-gpu telemetry launcher::
#
#   python scripts/run_workload_multi_gpu.py --workload moe_serve_mixtral_ep8_mixed --num-gpus 8

set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-m4_olmoe}"
DUR="${DURATION:-300}"
PORT="${PORT:-8000}"

run_one() {
  local label="$1"; shift
  echo "=== $label ==="
  python scripts/run_workload_multi_gpu.py \
    --workload "$label" \
    --num-gpus 8 \
    --timeout $((DUR + 2400)) \
    -- "$@"
}

case "$PROFILE" in
  m4_olmoe)
    # Small MoE, EP=8 (TP=1 DP=8) — fastest EP fingerprint check
    run_one moe_serve_olmoe_ep8_decode \
      python workloads/llm_moe_serving.py \
        --model olmoe_1b7b --tp 1 --dp 8 --enable-expert-parallel \
        --traffic decode --concurrency 8 --duration "$DUR" --port "$PORT" \
        --traffic-log "data/serving_olmoe_ep8_decode.jsonl"
    ;;
  m4_mixtral)
    run_one moe_serve_mixtral_ep8_decode \
      python workloads/llm_moe_serving.py \
        --model mixtral_8x7b --tp 1 --dp 8 --enable-expert-parallel \
        --traffic decode --concurrency 8 --duration "$DUR" --port "$PORT" \
        --max-model-len 8192 \
        --traffic-log "data/serving_mixtral_ep8_decode.jsonl"
    ;;
  m5_mixed)
    run_one moe_serve_mixtral_ep8_mixed \
      python workloads/llm_moe_serving.py \
        --model mixtral_8x7b --tp 2 --dp 4 --enable-expert-parallel \
        --enable-prefix-caching \
        --traffic mixed --concurrency 8 --duration "$DUR" --port "$PORT" \
        --max-model-len 16384 \
        --traffic-log "data/serving_mixtral_ep_mixed.jsonl"
    ;;
  m5_dense_ctl)
    # Dense control: TP×DP, no EP — for op-mix contrast.
    # Qwen2.5-7B has 28 heads → TP must divide 28 (TP=8 is invalid in vLLM).
    run_one moe_serve_qwen7b_tp4dp2_mixed \
      python workloads/llm_moe_serving.py \
        --model qwen25_7b --tp 4 --dp 2 \
        --enable-prefix-caching \
        --traffic mixed --concurrency 8 --duration "$DUR" --port "$PORT" \
        --traffic-log "data/serving_qwen7b_tp4dp2_mixed.jsonl"
    ;;
  *)
    echo "Unknown profile: $PROFILE" >&2
    echo "Use: m4_olmoe | m4_mixtral | m5_mixed | m5_dense_ctl" >&2
    exit 1
    ;;
esac
