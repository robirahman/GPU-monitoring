#!/usr/bin/env bash
# W10 (primary) -- REAL disaggregated prefill/decode (vLLM NixlConnector) on VCF h100-sxm-8-0.
#
# Usage (from repo root, inside an interactive allocation with 8 GPUs)::
#
#   bash scripts/run_disagg_kv_vcf.sh check          # print engine commands, verify vllm/nixl installed
#   bash scripts/run_disagg_kv_vcf.sh qwen7b         # 4 prefill + 4 decode, Qwen2.5-7B, mixed prompt lengths
#   bash scripts/run_disagg_kv_vcf.sh qwen7b_longctx # 16k-24k prompts (GB-scale KV handoffs)
#
# Each profile runs through the telemetry launcher (NVML + DCGM NVLink counters for the same
# window) and afterwards builds the empirical KV-handoff profile that W18 mimics:
#   data/profiles/kv_handoff_<label>.json
#
# Requirements: pip install vllm nixl   (vLLM's NixlConnector is documented as experimental).
# See docs/REAL_WORKLOAD_BAR.md. Not yet run on the cluster from this repo.
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

PROFILE="${1:-check}"
DUR="${DURATION:-300}"

run_one() {
  local label="$1"
  echo "=== $label ==="
  python scripts/run_workload_multi_gpu.py --workload "$label" --num-gpus 8 --timeout $((DUR + 3000))
  local log="data/${label}_handoffs.jsonl"
  if [[ -s "$log" ]]; then
    python scripts/build_kv_profile.py --handoffs "$log" --out "data/profiles/kv_handoff_${label}.json"
  else
    echo "WARNING: no handoff log at $log -- was the run aborted?" >&2
  fi
}

case "$PROFILE" in
  check)
    python workloads/disagg_kv_vllm.py --dry-run --num-prefill 4 --num-decode 4
    python - <<'PY'
from importlib.util import find_spec
from importlib import metadata
for m in ("vllm", "nixl"):
    if find_spec(m) is None:
        print(f"MISSING: {m}  (pip install {m})")
    else:
        try:
            print(f"ok: {m} {metadata.version(m)}")
        except metadata.PackageNotFoundError:
            print(f"ok: {m} (version unknown)")
PY
    ;;
  qwen7b)          run_one disagg_kv_vllm_qwen7b ;;
  qwen7b_longctx)  run_one disagg_kv_vllm_qwen7b_longctx ;;
  *)
    echo "Unknown profile: $PROFILE (use: check | qwen7b | qwen7b_longctx)" >&2
    exit 1
    ;;
esac
