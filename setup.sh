#!/usr/bin/env bash
# setup.sh -- Prepares a bare-metal cloud instance for SPAR GPU telemetry collection.
# Target: Ubuntu 22.04+ on Vast.ai / RunPod with NVIDIA drivers pre-installed.
# Usage: chmod +x setup.sh && sudo ./setup.sh
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

echo "=== SPAR GPU Telemetry Pipeline Setup ==="

# --- Section 1: Verify GPU and driver ---
echo ""
echo "[1/6] Verifying GPU and driver..."
if ! command -v nvidia-smi &>/dev/null; then
    echo "ERROR: nvidia-smi not found. NVIDIA driver must be installed first."
    exit 1
fi
nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
echo "GPU and driver OK."

# --- Section 2: Install CUDA toolkit if missing ---
echo ""
echo "[2/6] Checking CUDA toolkit..."
if command -v nvcc &>/dev/null; then
    echo "CUDA toolkit already installed: $(nvcc --version | grep release)"
else
    echo "Installing CUDA toolkit 12.4..."
    apt-get update -qq
    apt-get install -y -qq wget
    wget -q https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.1-1_all.deb
    dpkg -i cuda-keyring_1.1-1_all.deb
    rm cuda-keyring_1.1-1_all.deb
    apt-get update -qq
    apt-get install -y -qq cuda-toolkit-12-4
    echo "CUDA toolkit installed."
fi

# --- Section 3: Install DCGM (Tier 2 metrics) ---
echo ""
echo "[3/6] Checking DCGM..."
if command -v dcgmi &>/dev/null; then
    echo "DCGM already installed."
else
    echo "Installing DCGM..."
    apt-get update -qq
    apt-get install -y -qq datacenter-gpu-manager || {
        echo "WARNING: DCGM installation failed. Tier 2 metrics will be unavailable."
        echo "You can install manually later: apt-get install datacenter-gpu-manager"
    }
fi

# Start DCGM host engine if available
if command -v nv-hostengine &>/dev/null; then
    if ! pgrep -x nv-hostengine &>/dev/null; then
        echo "Starting DCGM host engine..."
        nv-hostengine -d || echo "WARNING: Failed to start nv-hostengine."
    else
        echo "DCGM host engine already running."
    fi
fi

# --- Section 4: Install Nsight Compute (Tier 3 metrics) ---
echo ""
echo "[4/6] Checking Nsight Compute..."
if command -v ncu &>/dev/null; then
    echo "Nsight Compute already installed: $(ncu --version 2>/dev/null | head -1)"
else
    echo "Attempting to install Nsight Compute..."
    apt-get install -y -qq nsight-compute 2>/dev/null || {
        echo "WARNING: Nsight Compute installation failed."
        echo "Tier 3 per-kernel profiling will be unavailable."
        echo "Nsight Compute requires bare-metal access and may not be available on all hosts."
    }
fi

# --- Section 5: Workload system dependencies ---
echo ""
echo "[5/7] Installing workload system dependencies..."
apt-get install -y -qq python3 python3-pip python3-venv unzip

# GROMACS (scientific HPC workload)
if command -v gmx &>/dev/null; then
    echo "GROMACS already installed."
else
    echo "Installing GROMACS..."
    apt-get install -y -qq gromacs || echo "WARNING: GROMACS installation failed. gromacs_adh workload will be unavailable."
fi

# Blender (rendering workload)
if command -v blender &>/dev/null; then
    echo "Blender already installed."
else
    echo "Installing Blender..."
    apt-get install -y -qq blender || echo "WARNING: Blender installation failed. blender_bmw workload will be unavailable."
fi

# FFmpeg should already be present on most images; verify NVENC support
if command -v ffmpeg &>/dev/null; then
    if ffmpeg -encoders 2>/dev/null | grep -q h264_nvenc; then
        echo "FFmpeg with NVENC: OK"
    else
        echo "WARNING: FFmpeg found but h264_nvenc not available. ffmpeg_nvenc workload will be unavailable."
    fi
else
    echo "Installing FFmpeg..."
    apt-get install -y -qq ffmpeg || echo "WARNING: FFmpeg installation failed."
fi

# --- Section 6: Python environment ---
echo ""
echo "[6/7] Setting up Python environment..."

VENV_DIR="${SCRIPT_DIR}/.venv"
if [ ! -d "$VENV_DIR" ]; then
    python3 -m venv "$VENV_DIR"
fi
source "$VENV_DIR/bin/activate"

pip install --upgrade -q pip
pip install -q -r "$SCRIPT_DIR/requirements.txt"

# Make DCGM Python bindings available in the venv if installed system-wide
DCGM_PY_DIR="/usr/lib/python3/dist-packages"
if [ -d "$DCGM_PY_DIR" ] && [ -d "$VENV_DIR/lib" ]; then
    VENV_SITE=$(python3 -c "import site; print(site.getsitepackages()[0])")
    if [ ! -f "$VENV_SITE/dcgm.pth" ]; then
        echo "$DCGM_PY_DIR" > "$VENV_SITE/dcgm.pth"
        echo "Added DCGM Python bindings to venv path."
    fi
fi

# --- Section 7: Verify installation ---
echo ""
echo "[7/7] Verifying installation..."
python3 -c "import pynvml; pynvml.nvmlInit(); print('  pynvml: OK'); pynvml.nvmlShutdown()"
python3 -c "import pandas; import pyarrow; print('  pandas + pyarrow: OK')"
python3 -c "import torch; print(f'  PyTorch: OK (CUDA available: {torch.cuda.is_available()})')"
python3 -c "import transformers; import datasets; import accelerate; print('  transformers + datasets + accelerate: OK')"

# Optional checks
python3 -c "
try:
    import pydcgm
    print('  DCGM Python bindings: OK')
except ImportError:
    print('  DCGM Python bindings: NOT AVAILABLE (Tier 2 will be skipped)')
" 2>/dev/null

# Workload tools
command -v gmx &>/dev/null && echo "  GROMACS: OK" || echo "  GROMACS: NOT AVAILABLE"
command -v blender &>/dev/null && echo "  Blender: OK" || echo "  Blender: NOT AVAILABLE"
if command -v ffmpeg &>/dev/null && ffmpeg -encoders 2>/dev/null | grep -q h264_nvenc; then
    echo "  FFmpeg NVENC: OK"
else
    echo "  FFmpeg NVENC: NOT AVAILABLE"
fi

echo ""
echo "=== Setup complete ==="
echo "Activate the environment with: source ${VENV_DIR}/bin/activate"
echo "Run a test: python scripts/run_workload.py --workload idle --timeout 10"
