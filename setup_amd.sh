#!/usr/bin/env bash
# setup_amd.sh -- Prepares an Ubuntu 24.04+ machine with an RDNA 4 AMD GPU
# (RX 9070 XT etc.) for SPAR GPU telemetry collection. Linux/ROCm analogue
# of setup.sh.
#
# Tested target: Ubuntu 26.04 (resolute), kernel 7.0, AMD Radeon RX 9070 XT
# (gfx1201). ROCm 7.1 is in the universe repo so no third-party AMD repo is
# needed.
#
# Usage: chmod +x setup_amd.sh && sudo ./setup_amd.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="${SUDO_USER:-$USER}"

echo "=== SPAR GPU Telemetry Pipeline Setup (AMD/ROCm) ==="

# --- Section 1: Verify amdgpu kernel module is loaded ---
echo ""
echo "[1/6] Verifying amdgpu kernel module..."
if ! grep -q '^amdgpu ' /proc/modules; then
    echo "ERROR: amdgpu kernel module not loaded. Check 'dmesg | grep amdgpu'."
    exit 1
fi
echo "amdgpu OK."

# --- Section 2: Install ROCm from the Ubuntu universe repo ---
echo ""
echo "[2/6] Installing ROCm (rocm-dev, rocm-smi, rocminfo)..."
apt-get update -qq
# rocm-dev pulls in HIP runtime, all ROCm libs, rocm-smi-lib, amd-smi-dev,
# rocminfo, hipcc, miopen, rccl, rocblas, etc.
apt-get install -y \
    rocm-dev \
    rocm-smi \
    rocminfo

# --- Section 3: Add invoking user to render and video groups ---
echo ""
echo "[3/6] Granting GPU access to user '${REAL_USER}'..."
usermod -a -G render,video "${REAL_USER}"
echo "Note: user '${REAL_USER}' must log out and back in (or reboot) for group membership to take effect."

# --- Section 4: System workload dependencies ---
echo ""
echo "[4/6] Installing workload system dependencies..."
apt-get install -y \
    python3 python3-pip python3-venv \
    ffmpeg \
    mesa-va-drivers vainfo \
    libva-dev \
    unzip

# Verify VA-API works against the AMD VCN engine
if vainfo --display drm --device /dev/dri/renderD128 2>/dev/null | grep -q 'VAEntrypointEncSlice\|VAProfileH264'; then
    echo "FFmpeg VA-API: OK"
else
    echo "WARNING: VA-API encode entry points not detected. ffmpeg_vaapi workload may not work."
fi

# --- Section 5: Python environment + PyTorch ROCm ---
echo ""
echo "[5/6] Setting up Python venv (run subsequent commands as ${REAL_USER})..."

VENV_DIR="${SCRIPT_DIR}/.venv"
sudo -u "${REAL_USER}" python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"

pip install --upgrade -q pip

# PyTorch with ROCm 7.x wheel. Pin the index to the closest official ROCm
# wheel available — AMD publishes rocm6.4 wheels; PyTorch nightlies have
# rocm7. Use the rocm6.4 stable for now (gfx1201 is supported from 6.4 on).
pip install --index-url https://download.pytorch.org/whl/rocm6.4 \
    torch torchvision torchaudio

# Project requirements (excluding torch — installed above)
pip install -r "${SCRIPT_DIR}/requirements_amd.txt"

# Make the amdsmi Python bindings (shipped with ROCm at
# /opt/rocm/share/amd_smi) importable from the venv.
AMDSMI_DIR="/opt/rocm/share/amd_smi"
if [ -d "${AMDSMI_DIR}" ]; then
    VENV_SITE=$("${VENV_DIR}/bin/python3" -c "import site; print(site.getsitepackages()[0])")
    echo "${AMDSMI_DIR}" > "${VENV_SITE}/amdsmi.pth"
    echo "Linked amdsmi from ${AMDSMI_DIR} into venv (via .pth)."
else
    echo "WARNING: ${AMDSMI_DIR} not found. amdsmi Python bindings may be missing."
    echo "  Check: dpkg -L libamd-smi-dev | grep -i python"
fi

# --- Section 6: Verify ---
echo ""
echo "[6/6] Verifying installation..."
sudo -u "${REAL_USER}" bash -lc "
    source '${VENV_DIR}/bin/activate'
    /opt/rocm/bin/rocm-smi --showproductname --showmeminfo vram 2>&1 | head -20
    echo ''
    python3 -c 'import sys; sys.path.insert(0, \"/opt/rocm/share/amd_smi\"); import amdsmi; amdsmi.amdsmi_init(); print(\"  amdsmi: OK\"); amdsmi.amdsmi_shut_down()'
    python3 -c 'import torch; print(f\"  PyTorch: {torch.__version__} (CUDA/HIP available: {torch.cuda.is_available()})\")'
    python3 -c 'import pandas, pyarrow, sklearn; print(\"  pandas + pyarrow + sklearn: OK\")'
    python3 -c 'import transformers, datasets, accelerate; print(\"  transformers + datasets + accelerate: OK\")'
"

echo ""
echo "=== Setup complete ==="
echo "User '${REAL_USER}' must log out/in (or run 'newgrp render') for GPU group membership."
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
echo "Smoke test: python scripts/amdsmi_smoke_test.py"
echo "First workload run: python scripts/run_workload_amd.py --workload idle"
