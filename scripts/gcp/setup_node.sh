#!/usr/bin/env bash
# Bring a freshly created a2-ultragpu-8g up to the point where a camera-ready
# cell can run: pinned torch, CIFAR-10 staged, telemetry importable.
#
# The torch pin matters. requirements_v2.txt holds torch 2.5.1+cu121, which is
# what the corpus was collected under; the DLVM's own torch is 2.9 and would
# measure a different batch rate. A reference rate taken under a different
# stack is not a reference for the archived run.
#
# Idempotent: markers under /var/lib/spar let a re-run resume.
set -euo pipefail
REPO="${REPO_DIR:-/opt/spar/repo}"
BUCKET="${BUCKET:-gs://focus-surfer-494820-g0-telemetry}"
STATE=/var/lib/spar
install -d "$STATE"
say() { echo "[$(date -u '+%H:%M:%SZ')] [setup] $*"; }
done_phase() { [ -f "$STATE/phase.$1.done" ]; }
mark() { date -u +%FT%TZ > "$STATE/phase.$1.done"; }

export DEBIAN_FRONTEND=noninteractive
if ! done_phase apt; then
  say "apt: python venv toolchain"
  apt-get -o DPkg::Lock::Timeout=900 -y update >/dev/null
  apt-get -o DPkg::Lock::Timeout=900 -y install python3-venv python3-pip >/dev/null
  mark apt
fi

say "GPUs visible: $(nvidia-smi -L | wc -l)"
nvidia-smi -L | sed 's/^/[setup]   /'

if ! done_phase venv; then
  say "venv-main from requirements_v2.txt (pinned torch 2.5.1+cu121)"
  python3 -m venv "${REPO}/venv-main"
  "${REPO}/venv-main/bin/pip" install -q --upgrade pip wheel
  "${REPO}/venv-main/bin/pip" install -q -r "${REPO}/requirements_v2.txt"
  mark venv
fi

if ! done_phase cifar; then
  say "staging CIFAR-10 (workloads open it with download=True but the node has no egress budget for it)"
  install -d "${REPO}/data"
  gcloud storage cp "${BUCKET}/staging/cifar10.tgz" /tmp/cifar10.tgz
  tar xzf /tmp/cifar10.tgz -C "${REPO}/data"
  rm -f /tmp/cifar10.tgz
  mark cifar
fi

say "verifying the stack the campaign needs"
"${REPO}/venv-main/bin/python" - <<'PYEOF'
import torch, torchvision, pynvml, pandas, pyarrow, os
pynvml.nvmlInit()
print(f"[setup]   torch {torch.__version__} torchvision {torchvision.__version__}")
print(f"[setup]   cuda {torch.cuda.is_available()} devices {torch.cuda.device_count()}")
print(f"[setup]   nvml devices {pynvml.nvmlDeviceGetCount()}")
print(f"[setup]   cifar present: {os.path.isdir('data/cifar10/cifar-10-batches-py')}")
PYEOF
say "node ready"
