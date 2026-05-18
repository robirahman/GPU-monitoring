"""Windows-AMD workload registry — analogue of workloads/registry_amd.py.

Diverges from the Linux/ROCm registry in three places:

  1. Compute stack:    torch-directml replaces PyTorch ROCm. Most
     PyTorch workloads needed a small port (`*_directml.py`) because
     torch.cuda is not aliased on torch-directml the way it is on ROCm.
  2. Video encode:     ffmpeg_amf.py (h264_amf / hevc_amf / av1_amf)
     replaces ffmpeg_vaapi.py and ffmpeg_nvenc.py.
  3. LLM inference:    llama.cpp Vulkan binary (`tools/llama-cpp-vulkan/`)
     replaces transformers-on-torch where Vulkan is faster and more
     stable on RDNA 4. The transformers path also works for non-LLM
     inference (CLIP, ViT, ResNet) via torch-directml.

Telemetry: all entries assume scripts/collect_pdh_windows.py is the
collector (PDH per-engine util + LibreHardwareMonitor HTTP sensors).
The amd-smi-based collector at scripts/collect_amdsmi_windows.py is
optional and requires the AMD HIP SDK to be installed.

Batch sizes target the RX 9070 XT (16 GB VRAM); shrink for the iGPU.
"""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_WORKLOADS_DIR = os.path.join(_REPO_ROOT, "workloads")
_TOOLS_DIR = os.path.join(os.path.dirname(_REPO_ROOT), "tools")  # one level above SPAR-GPU-monitoring

_MODELS_DIR = os.path.join(os.path.dirname(_REPO_ROOT), "models")
_LLAMA_VULKAN_DIR = os.path.join(_TOOLS_DIR, "llama-cpp-vulkan")

# Default batch sizes for the RX 9070 XT (16 GB). Tune downward for iGPU runs.
_BATCH_RESNET = "256"
_BATCH_MLP = "512"
_BATCH_GPT2 = "4"
_BATCH_BERT = "16"
_BATCH_RESNET50_INFER = "128"


def _wl(script: str, *args, description: str, default_timeout: int):
    return {
        "command": [sys.executable, os.path.join(_WORKLOADS_DIR, script), *args],
        "description": description,
        "default_timeout": default_timeout,
    }


def _llama(model_filename: str, *args, description: str, default_timeout: int):
    """Build a llama.cpp Vulkan inference entry. The model file must exist
    under _MODELS_DIR before the entry can run; we don't auto-download."""
    return {
        "command": [
            sys.executable, os.path.join(_WORKLOADS_DIR, "llama_cpp_vulkan.py"),
            "--llama-dir", _LLAMA_VULKAN_DIR,
            "--model", os.path.join(_MODELS_DIR, model_filename),
            *args,
        ],
        "description": description,
        "default_timeout": default_timeout,
        "requires_model": model_filename,
    }


WORKLOAD_REGISTRY_WINDOWS_AMD = {
    # ----------------------------------------------------------------- baseline
    "idle": _wl("idle.py", "--duration", "120",
                description="Idle GPU baseline (120s sleep)",
                default_timeout=130),

    # ---------------------------------------------------------------- ML training
    # All training entries use torch-directml. Requires `pip install torch-directml`.
    "pytorch_resnet_cifar10": _wl(
        "pytorch_training_directml.py",
        "--model", "resnet18", "--dataset", "cifar10",
        "--epochs", "10", "--batch-size", _BATCH_RESNET,
        description="ResNet-18 / CIFAR-10, 10 epochs, FP32, DirectML",
        default_timeout=1800),
    "pytorch_resnet_cifar10_bf16": _wl(
        "pytorch_training_directml.py",
        "--model", "resnet18", "--dataset", "cifar10",
        "--epochs", "10", "--batch-size", _BATCH_RESNET, "--bf16",
        description="ResNet-18 / CIFAR-10, 10 epochs, BF16 manual cast",
        default_timeout=1800),
    "pytorch_mlp_cifar10": _wl(
        "pytorch_training_directml.py",
        "--model", "simple_mlp", "--dataset", "cifar10",
        "--epochs", "20", "--batch-size", _BATCH_MLP,
        description="MLP / CIFAR-10, 20 epochs, FP32",
        default_timeout=1200),
    # TODO: gpt2_finetune_directml.py, bert_finetune_directml.py — port from
    # workloads/{gpt2,bert}_finetune.py. Both use HuggingFace transformers,
    # which works on torch-directml but needs the same device + sync wrapping
    # as pytorch_training_directml.py.

    # --------------------------------------------------------------- ML inference
    # llama.cpp Vulkan path — requires GGUF models at _MODELS_DIR.
    "llm_infer_tinyllama": _llama(
        "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
        "--mode", "bench", "--duration", "300",
        description="TinyLlama-1.1B Q4_K_M, llama-bench, Vulkan",
        default_timeout=420),
    "llm_infer_phi3_mini": _llama(
        "Phi-3-mini-4k-instruct.Q4_K_M.gguf",
        "--mode", "bench", "--duration", "300",
        description="Phi-3-Mini 3.8B Q4_K_M, llama-bench, Vulkan",
        default_timeout=420),
    "llm_infer_qwen25_1p5b": _llama(
        "Qwen2.5-1.5B-Instruct.Q4_K_M.gguf",
        "--mode", "bench", "--duration", "300",
        description="Qwen 2.5 1.5B Q4_K_M, llama-bench, Vulkan",
        default_timeout=420),
    "llm_generate_tinyllama": _llama(
        "TinyLlama-1.1B-Chat-v1.0.Q4_K_M.gguf",
        "--mode", "generate", "--max-tokens", "256", "--duration", "300",
        description="TinyLlama-1.1B autoregressive generate loop, Vulkan",
        default_timeout=420),
    # TODO: ONNX Runtime DirectML EP entries for ResNet-50 / ViT-L / Whisper.
    # These need an onnxruntime_dml_inference.py wrapper plus ONNX model exports.

    # -------------------------------------------------------- Scientific HPC (non-ML)
    "scientific_fft": _wl(
        "scientific_hpc_directml.py",
        "--workload", "cufft", "--duration", "600",
        description="3D FFT benchmark (rocFFT / DirectML), 600s, memory-bound",
        default_timeout=660),
    "scientific_nbody": _wl(
        "scientific_hpc_directml.py",
        "--workload", "nbody", "--duration", "600", "--n-bodies", "4096",
        description="Gravitational N-body (compute-bound, O(N^2)), 4096 bodies",
        default_timeout=660),

    # -------------------------------------------------------- Video encode (non-ML)
    # AMF on this dual-GPU box auto-picks the iGPU; pass --adapter to force the
    # dGPU once you've identified its DXGI index.
    "ffmpeg_amf_h264": _wl(
        "ffmpeg_amf.py", "--duration", "300", "--codec", "h264_amf",
        description="FFmpeg h264_amf encode loop (AMD VCN, Windows)",
        default_timeout=360),
    "ffmpeg_amf_hevc_4k": _wl(
        "ffmpeg_amf.py", "--duration", "300", "--codec", "hevc_amf",
        "--resolution", "3840x2160", "--bitrate", "40M",
        description="FFmpeg hevc_amf 4K encode loop",
        default_timeout=360),
    "ffmpeg_amf_av1": _wl(
        "ffmpeg_amf.py", "--duration", "300", "--codec", "av1_amf", "--bitrate", "8M",
        description="FFmpeg av1_amf encode loop",
        default_timeout=360),

    # ----------------------------------------------------- Crypto mining (non-ML)
    # TODO: mining_proxy_directml.py — port the Ethash-style memory-hard hashing
    # from workloads/mining_proxy.py. The CUDA original uses torch.randint /
    # gather / xor — all supported by torch-directml.
    # Alternative: invoke a real AMD-Windows miner (lolMiner, T-Rex AMD, BzMiner)
    # against a benchmark stratum URL. Note: real mining produces network
    # traffic; only enable in environments where that's acceptable.

    # ---------------------------------------------------------- Rendering (non-ML)
    # TODO: blender_render_hip.py — Blender 4.x with the HIP backend. Requires
    # the AMD HIP SDK install and Blender's "AMD HIP" device option enabled.
    # Renders one frame of a fixed test scene per iteration (similar to
    # blender_render.py on Linux but with `--cycles-device HIP` instead of
    # `OPTIX` / `CUDA`).

    # --------------------------------------------------------- Adversarial (later)
    # TODO: port adversarial_training.py / adversarial_round{2..5}.py to use
    # torch-directml. Strategies A-D (utilization manipulation, temporal
    # disruption) translate cleanly. Strategies G (clock throttling) require
    # AMD-Windows-specific tooling — possibly amd-smi `set --core-clock` or
    # MSI Afterburner-style adjustment. Defer until non-adversarial corpus
    # is collected on Windows.
}


def get_workload(label: str) -> dict:
    if label not in WORKLOAD_REGISTRY_WINDOWS_AMD:
        available = ", ".join(sorted(WORKLOAD_REGISTRY_WINDOWS_AMD))
        raise ValueError(f"Unknown workload {label!r}. Available: {available}")
    return WORKLOAD_REGISTRY_WINDOWS_AMD[label]


def list_workloads() -> list[str]:
    return sorted(WORKLOAD_REGISTRY_WINDOWS_AMD)
