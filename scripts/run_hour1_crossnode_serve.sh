#!/usr/bin/env bash
# Hour-1 cross-node serve helpers for 2×1 Instant Cluster (ens1 fabric).
#
# C — OLMoE DP=2 + EP across nodes (vLLM 0.10.x: n0 API + n1 --headless)
# D — Qwen TP=2 via Ray across nodes
#
# Usage:
#   export MASTER_ADDR=10.65.0.2   # ens1 of NODE0
#   export NODE_RANK=0|1
#   bash scripts/run_hour1_crossnode_serve.sh c-engine
#   bash scripts/run_hour1_crossnode_serve.sh c-traffic   # NODE0 or laptop→n0
#   bash scripts/run_hour1_crossnode_serve.sh d-engine    # Ray TP2
#   bash scripts/run_hour1_crossnode_serve.sh d-traffic   # traffic against TP2 server
#
# HTTP single-GPU fallbacks remain as d-server-single / d-traffic-fallback for debug
# only — do NOT use them as SPAR interconnect evidence.
set -euo pipefail
REPO="${REPO:-/workspace/GPU-monitoring}"
cd "$REPO"
PY="${PY:-/usr/local/bin/python}"
MASTER_ADDR="${MASTER_ADDR:-${PRIMARY_ADDR:-10.65.0.2}}"
NODE_RANK="${NODE_RANK:-0}"
RPC_PORT="${RPC_PORT:-13345}"
API_PORT="${API_PORT:-8000}"
NCCL_ENV=(
  "NCCL_SOCKET_IFNAME=${NCCL_SOCKET_IFNAME:-ens1}"
  "NCCL_IB_DISABLE=${NCCL_IB_DISABLE:-1}"   # RoCE WC 129 on this cluster; Socket/ens1
  "GLOO_SOCKET_IFNAME=${GLOO_SOCKET_IFNAME:-ens1}"
  "VLLM_HOST_IP=${VLLM_HOST_IP:-$(ip -4 -o addr show ens1 2>/dev/null | awk '{print $4}' | cut -d/ -f1 || hostname -I | awk '{print $1}')}"
)

role="${1:-}"
shift || true

run_env() { env "${NCCL_ENV[@]}" "$@"; }

c_engine() {
  # vLLM 0.10.2 multi-node DP+EP via Ray backend.
  # mp/headless ZMQ handshake is broken on this stack (EngineCore HELLO never
  # answered → 5min timeout) even with correct `vllm serve --headless`.
  # Ray path is the same one that made D (TP2) work: place DP engines as Ray
  # actors across nodes; EP AllToAll still goes through NCCL on ens1.
  local model="allenai/OLMoE-1B-7B-0924"
  local VLLM_BIN="${VLLM_BIN:-$(command -v vllm || echo vllm)}"
  if ! "$PY" -c "import ray" 2>/dev/null; then
    echo "[c-engine] installing ray …"
    "$PY" -m pip install -q --break-system-packages "ray[default]==2.48.0"
  fi
  # Ray must advertise ens1 IPs — vLLM DP placement looks for node:{dp_master_ip}.
  local NODE_IP="${VLLM_HOST_IP:-$(ip -4 -o addr show ens1 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
  if [[ "$NODE_RANK" == "1" ]]; then
    echo "[c-engine] ray worker → ${MASTER_ADDR}:6379 (node-ip=${NODE_IP})"
    run_env ray stop >/dev/null 2>&1 || true
    exec env "${NCCL_ENV[@]}" ray start --address="${MASTER_ADDR}:6379" \
      --node-ip-address="$NODE_IP" --num-gpus=1 --block
  fi
  echo "[c-engine] ray start --head on NODE0 (node-ip=${NODE_IP})"
  run_env ray stop >/dev/null 2>&1 || true
  run_env ray start --head --port=6379 --node-ip-address="$NODE_IP" --num-gpus=1
  echo "[c-engine] waiting for 2 Ray nodes …"
  for i in $(seq 1 60); do
    n=$("$PY" -c "import ray; ray.init(address='auto'); print(len([n for n in ray.nodes() if n.get('Alive')])); ray.shutdown()" 2>/dev/null || echo 0)
    echo "  ray alive nodes=$n (t=$i)"
    [[ "$n" -ge 2 ]] && break
    sleep 5
  done
  echo "[c-engine] vllm serve DP2+EP ray backend :${API_PORT}"
  run_env "$VLLM_BIN" serve "$model" \
    --dtype bfloat16 \
    --host 0.0.0.0 \
    --port "$API_PORT" \
    --tensor-parallel-size 1 \
    --data-parallel-size 2 \
    --data-parallel-size-local 1 \
    --data-parallel-backend ray \
    --data-parallel-address "$MASTER_ADDR" \
    --enable-expert-parallel \
    --max-model-len "${MAX_MODEL_LEN:-4096}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --trust-remote-code
}

c_traffic() {
  # Hit NODE0 API only (headless n1 has no HTTP). Prefer completions for OLMoE (no chat template).
  local base="http://${MASTER_ADDR}:${API_PORT}"
  echo "[c-traffic] mixed → $base  (DP2+EP server must already be up)"
  if [[ -f scripts/hour1_completions_traffic.py ]]; then
    run_env "$PY" scripts/hour1_completions_traffic.py \
      --base-url "$base" \
      --model "${MODEL:-allenai/OLMoE-1B-7B-0924}" \
      --duration "${DURATION:-90}" \
      --concurrency "${CONCURRENCY:-8}" \
      --traffic-log "${TRAFFIC_LOG:-data/hour1_olmoe_dp2ep_mixed.jsonl}" \
      "$@"
  else
    run_env "$PY" workloads/llm_moe_serving.py \
      --model olmoe_1b7b --tp 1 --dp 2 --enable-expert-parallel \
      --no-server --base-url "$base" \
      --traffic mixed --concurrency "${CONCURRENCY:-8}" --duration "${DURATION:-90}" \
      --max-model-len "${MAX_MODEL_LEN:-4096}" \
      --traffic-log "${TRAFFIC_LOG:-data/hour1_olmoe_dp2ep_mixed.jsonl}" \
      "$@"
  fi
}

d_engine_ray() {
  # TP=2 across 2×1 via Ray. NODE1 must join before NODE0 starts vLLM.
  local model="Qwen/Qwen2.5-7B-Instruct"
  if ! "$PY" -c "import ray" 2>/dev/null; then
    echo "[d-engine] installing ray …"
    "$PY" -m pip install -q --break-system-packages "ray[default]==2.48.0"
  fi
  local NODE_IP="${VLLM_HOST_IP:-$(ip -4 -o addr show ens1 2>/dev/null | awk '{print $4}' | cut -d/ -f1)}"
  if [[ "$NODE_RANK" == "1" ]]; then
    echo "[d-engine] ray worker → ${MASTER_ADDR}:6379 (node-ip=${NODE_IP})"
    run_env ray stop >/dev/null 2>&1 || true
    exec env "${NCCL_ENV[@]}" ray start --address="${MASTER_ADDR}:6379" \
      --node-ip-address="$NODE_IP" --num-gpus=1 --block
  fi
  echo "[d-engine] ray start --head on NODE0 (node-ip=${NODE_IP})"
  run_env ray stop >/dev/null 2>&1 || true
  run_env ray start --head --port=6379 --node-ip-address="$NODE_IP" --num-gpus=1
  echo "[d-engine] waiting for 2 Ray nodes …"
  for i in $(seq 1 60); do
    n=$("$PY" -c "import ray; ray.init(address='auto'); print(len([n for n in ray.nodes() if n.get('Alive')])); ray.shutdown()" 2>/dev/null || echo 0)
    echo "  ray alive nodes=$n (t=$i)"
    [[ "$n" -ge 2 ]] && break
    sleep 5
  done
  echo "[d-engine] vLLM TP=2 ray backend :${API_PORT}"
  run_env "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$model" --dtype bfloat16 \
    --host 0.0.0.0 --port "$API_PORT" \
    --tensor-parallel-size 2 \
    --distributed-executor-backend ray \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --enable-prefix-caching \
    --trust-remote-code
}

d_traffic() {
  local base="http://${MASTER_ADDR}:${API_PORT}"
  echo "[d-traffic] mixed → $base  (TP2 Ray server)"
  run_env "$PY" workloads/llm_moe_serving.py \
    --model qwen25_7b --tp 2 --dp 1 \
    --no-server --base-url "$base" \
    --traffic mixed --concurrency "${CONCURRENCY:-8}" --duration "${DURATION:-60}" \
    --enable-prefix-caching \
    --traffic-log "${TRAFFIC_LOG:-data/hour1_qwen_tp2_mixed.jsonl}" \
    "$@"
}

d_traffic_fallback() {
  # HTTP client on peer → single-GPU server on NODE0 (weak interconnect: request bytes only)
  local base="http://${MASTER_ADDR}:${API_PORT}"
  echo "[d-traffic-fallback] mixed → $base  (NOT TP AllReduce — document as HTTP fabric)"
  run_env "$PY" workloads/llm_moe_serving.py \
    --model qwen25_7b --tp 1 --dp 1 \
    --no-server --base-url "$base" \
    --traffic mixed --concurrency "${CONCURRENCY:-8}" --duration "${DURATION:-60}" \
    --enable-prefix-caching \
    --traffic-log "${TRAFFIC_LOG:-data/hour1_qwen_http_crossnode_mixed.jsonl}" \
    "$@"
}

d_server_single() {
  # NODE0 only: single-GPU Qwen server for HTTP-fallback path
  local model="Qwen/Qwen2.5-7B-Instruct"
  echo "[d-server-single] Qwen tp1 on NODE0 :${API_PORT}"
  run_env "$PY" -m vllm.entrypoints.openai.api_server \
    --model "$model" --dtype bfloat16 \
    --host 0.0.0.0 --port "$API_PORT" \
    --tensor-parallel-size 1 \
    --max-model-len "${MAX_MODEL_LEN:-8192}" \
    --gpu-memory-utilization "${GPU_MEM_UTIL:-0.90}" \
    --enable-prefix-caching \
    --trust-remote-code
}

case "$role" in
  c-engine) c_engine "$@" ;;
  c-traffic) c_traffic "$@" ;;
  d-engine) d_engine_ray "$@" ;;
  d-traffic) d_traffic "$@" ;;
  d-server-single) d_server_single "$@" ;;
  d-traffic-fallback) d_traffic_fallback "$@" ;;
  *)
    echo "usage: $0 {c-engine|c-traffic|d-engine|d-traffic|d-server-single|d-traffic-fallback}"
    exit 2
    ;;
esac
