#!/usr/bin/env bash
# setup_amd_datacenter.sh -- Prepares a cloud datacenter AMD GPU instance
# (MI300X, MI325X, MI355) for SPAR GPU telemetry collection.
#
# Differences from setup_amd.sh (consumer / RDNA 4):
#   - Target OS: Ubuntu 22.04 or 24.04 (typical cloud image — RunPod, TensorWave)
#   - ROCm 6.4+ from AMD's official repo (Ubuntu universe doesn't have it on
#     22.04/24.04 the way 26.04 does for 7.1)
#   - gfx942 (MI300X) / gfx950 (MI355) — well-supported from ROCm 6.0+
#   - Multi-GPU node support: RDC (ROCm Data Center tool) daemon for Tier 2
#     telemetry; works on datacenter cards but is disabled by default on consumer.
#   - PyTorch ROCm 6.4 wheel — explicit because cloud images don't always
#     pre-install it
#   - No VA-API setup (ffmpeg_vaapi workload uses VCN engines that don't
#     exist on datacenter cards)
#
# Tested targets (planned): RunPod MI300X 1x and 8x nodes, TensorWave MI325X.
#
# Usage: chmod +x setup_amd_datacenter.sh && sudo ./setup_amd_datacenter.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REAL_USER="${SUDO_USER:-$USER}"

echo "=== SPAR Datacenter AMD Setup (MI300X/MI325X/MI355) ==="

# --- Section 1: Detect OS + accept Ubuntu 22.04 / 24.04 / 26.04 ---
. /etc/os-release
CODENAME="${UBUNTU_CODENAME:-noble}"
echo "[1/7] OS detected: Ubuntu ${VERSION_ID} (${CODENAME})"
case "${CODENAME}" in
    jammy|noble|oracular|plucky|resolute) ;;
    *) echo "ERROR: Untested Ubuntu codename: ${CODENAME}. Bailing."; exit 1 ;;
esac

# --- Section 2: Verify amdgpu kernel module ---
echo ""
echo "[2/7] Verifying amdgpu kernel module..."
if ! grep -q '^amdgpu ' /proc/modules; then
    echo "ERROR: amdgpu not loaded. On RunPod/cloud images this is usually preloaded."
    echo "If running bare-metal, install AMDGPU DKMS from amdgpu-install."
    exit 1
fi
echo "  amdgpu OK"

# --- Section 3: Install ROCm ---
echo ""
echo "[3/7] Installing ROCm 6.4..."
ROCM_VER="${ROCM_VER:-6.4.1}"

# On 26.04 (resolute), use the universe-packaged ROCm 7.1 (preferred).
# On 22.04/24.04, set up AMD's official repo for ROCm 6.4.
if [[ "${CODENAME}" == "resolute" ]]; then
    apt-get update -qq
    apt-get install -y rocm-dev rocm-smi rocminfo
else
    # AMD's official repo
    mkdir -p --mode=0755 /etc/apt/keyrings
    wget -qO- https://repo.radeon.com/rocm/rocm.gpg.key \
        | gpg --dearmor -o /etc/apt/keyrings/rocm.gpg
    echo "deb [arch=amd64 signed-by=/etc/apt/keyrings/rocm.gpg] https://repo.radeon.com/rocm/apt/${ROCM_VER} ${CODENAME} main" \
        > /etc/apt/sources.list.d/rocm.list
    printf 'Package: *\nPin: release o=repo.radeon.com\nPin-Priority: 600\n' \
        > /etc/apt/preferences.d/rocm-pin-600
    apt-get update -qq
    apt-get install -y rocm-dev rocm-libs rocm-utils rdc
fi

# --- Section 4: RDC daemon for Tier 2 / multi-GPU ---
echo ""
echo "[4/7] Enabling RDC daemon (datacenter Tier-2 telemetry)..."
if command -v rdcd &>/dev/null; then
    systemctl enable rdc 2>/dev/null || true
    systemctl start rdc 2>/dev/null || true
    echo "  rdc service started (if available)"
else
    echo "  WARNING: rdcd not installed; Tier-2 metrics will be skipped."
fi

# --- Section 5: GPU access groups ---
echo ""
echo "[5/7] Adding ${REAL_USER} to render and video groups..."
usermod -a -G render,video "${REAL_USER}"

# --- Section 6: Python venv + PyTorch ROCm ---
echo ""
echo "[6/7] Setting up Python venv + PyTorch ROCm..."
apt-get install -y python3 python3-pip python3-venv git git-lfs

VENV_DIR="${SCRIPT_DIR}/.venv"
sudo -u "${REAL_USER}" python3 -m venv "${VENV_DIR}"
# shellcheck disable=SC1091
source "${VENV_DIR}/bin/activate"
pip install --upgrade -q pip

# PyTorch ROCm wheel. The rocm6.4 index works for gfx942/gfx950 (MI300X/MI355).
pip install --index-url https://download.pytorch.org/whl/rocm6.4 \
    torch torchvision torchaudio

pip install -r "${SCRIPT_DIR}/requirements_amd.txt"

# Install amdsmi Python bindings to match the installed library version.
# Cloud images often ship libamd-smi with a different minor version than the
# pip-published amdsmi wheel, so build from the matching AMD source tag.
LIBAMDSMI_VER=$(dpkg -s libamd-smi-dev 2>/dev/null | awk '/^Version/{print $2}' | cut -d- -f1 | head -c 5)
if [[ -n "${LIBAMDSMI_VER}" ]]; then
    echo "  libamd-smi-dev installed at version ${LIBAMDSMI_VER}; building matching amdsmi-py..."
    TMP=$(mktemp -d)
    git clone --depth 1 --branch "rocm-${LIBAMDSMI_VER}" \
        https://github.com/ROCm/amdsmi.git "${TMP}/amdsmi" || \
        git clone --depth 1 https://github.com/ROCm/amdsmi.git "${TMP}/amdsmi"
    VENV_SITE=$("${VENV_DIR}/bin/python3" -c "import site; print(site.getsitepackages()[0])")
    mkdir -p "${VENV_SITE}/amdsmi"
    cp "${TMP}/amdsmi/py-interface/"*.py "${VENV_SITE}/amdsmi/"
    echo "__version__ = '${LIBAMDSMI_VER}'" > "${VENV_SITE}/amdsmi/_version.py"
fi

# --- Section 7: Verify ---
echo ""
echo "[7/7] Verifying installation..."
sudo -u "${REAL_USER}" bash -lc "
    source '${VENV_DIR}/bin/activate'
    source '${SCRIPT_DIR}/scripts/rocm_env.sh'
    rocminfo | grep 'Marketing Name' | head
    rocm-smi --showproductname --showmeminfo vram 2>&1 | grep -v WARNING | head -20
    python3 -c 'import amdsmi; amdsmi.amdsmi_init(); print(\"  amdsmi:\", amdsmi.__version__ if hasattr(amdsmi, \"__version__\") else \"OK\"); amdsmi.amdsmi_shut_down()'
    python3 -c 'import torch; print(f\"  PyTorch: {torch.__version__} (HIP avail: {torch.cuda.is_available()}, device count: {torch.cuda.device_count()})\")'
    if command -v rdcd &>/dev/null; then
        python3 -c 'import sys; sys.path[:0]=__import__(\"glob\").glob(\"/opt/rocm*/libexec/rdc/python_binding\")+[\"/opt/rocm/share/rdc/python\"]; import RdcReader, rdc_bootstrap; print(\"  RDC bindings: OK\")' \
            || echo '  RDC bindings: NOT importable (Tier-2 will be skipped)'
    fi
"

echo ""
echo "=== Done ==="
echo "Activate the env with:"
echo "  source ${VENV_DIR}/bin/activate && source ${SCRIPT_DIR}/scripts/rocm_env.sh"
echo "(rocm_env.sh sets LD_LIBRARY_PATH for RDC + PyTorch and RDC_HOME; the"
echo " run_amd_*.sh scripts source it automatically.)"
echo "Smoke test: python scripts/amdsmi_smoke_test.py"
echo "Single-GPU sweep: scripts/run_amd_corpus.sh --reps 3"
echo ""
echo "For multi-GPU (8x MI300X) nodes, run the full datacenter corpus"
echo "(Phase 1 single-GPU + Phase 2 DDP adversarial):"
echo "  scripts/run_amd_datacenter_corpus.sh --reps 3"
