#!/usr/bin/env bash
# Pre-fetch the vLLM inference weights BEFORE the collection queue runs.
#
# Why this is a separate step and not left to the cell itself: the sharded and
# replicated inference cells are timed, and their telemetry collector starts
# before the workload. If vLLM has to pull ~145 GB of Qwen2.5-72B first, the
# download lands INSIDE the measurement window (idle GPUs recorded as if they
# were serving) and can outlast TIMEOUT_BIG, failing the headline cell outright.
#
# Run detached on every node, then wait for "ALL PRELOADS COMPLETE":
#   sudo bash scripts/gcp/preload_vllm_models.sh
set -uo pipefail

VENV_VLLM="${VENV_VLLM:-/opt/spar/repo/venv-vllm}"
# Must match the queue's HF_HOME. The queue runs as root, so $HOME is /root.
export HF_HOME="${HF_HOME:-/root/.cache/huggingface}"
MODELS=("${@:-}")
if [[ -z "${MODELS[0]:-}" ]]; then
  # Smallest first, so the replicated-inference cells are unblocked earliest.
  MODELS=("Qwen/Qwen2.5-7B-Instruct" "Qwen/Qwen2.5-72B-Instruct")
fi

echo "[preload] HF_HOME=$HF_HOME"
df -h "$(dirname "$HF_HOME")" | tail -1

for repo in "${MODELS[@]}"; do
  echo "[preload] === $repo ==="
  "${VENV_VLLM}/bin/python" - "$repo" <<'PY'
import sys, time
from huggingface_hub import snapshot_download
repo = sys.argv[1]
t0 = time.time()
# allow_patterns: weights + tokenizer/config only. Skips the duplicate
# consolidated *.pth / original/ copies some repos ship, which would otherwise
# double the transfer for nothing.
path = snapshot_download(
    repo,
    max_workers=16,
    allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model"],
)
print(f"[preload] {repo} -> {path} in {time.time()-t0:.0f}s", flush=True)
PY
  rc=$?
  if [[ $rc -ne 0 ]]; then
    echo "[preload] FAILED $repo (rc=$rc)"
    exit $rc
  fi
  du -sh "$HF_HOME" 2>/dev/null
done

# CIFAR-10 for the DDP/adversarial cells. The workloads pass
# download=(rank == 0), and rank 0 lives on node0 -- with no shared filesystem
# node1's ranks find nothing and the cell dies in ~15 s. Stage it on EVERY node.
# The upstream mirror throttles to ~100 kB/s, so this takes ~25 min: start it
# early, not when the cell is already running.
echo "[preload] === CIFAR-10 -> ${REPO_DIR:-/opt/spar/repo}/data/cifar10 ==="
"${VENV_MAIN:-/opt/spar/repo/venv-main}/bin/python" - <<'PY'
import torchvision
for train in (True, False):
    d = torchvision.datasets.CIFAR10(
        root="/opt/spar/repo/data/cifar10", train=train, download=True)
    print(f"[preload] cifar10 train={train}: {len(d)} samples", flush=True)
PY

echo "[preload] ALL PRELOADS COMPLETE"
