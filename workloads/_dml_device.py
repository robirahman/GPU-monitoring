"""Tiny helper for DirectML device selection on Windows-AMD.

The CUDA / ROCm versions of our PyTorch workloads use ``device="cuda"`` and
``torch.cuda.synchronize()``. Those calls don't work under torch-directml on
Windows. This module provides:

  - ``get_device(prefer="dml")`` — returns a ``torch.device`` pointing at the
    DirectML adapter (defaulting to adapter index 0 = the discrete dGPU on
    most dual-GPU AMD systems; configurable via ``DML_DEVICE_INDEX`` env var).
  - ``sync(device)`` — a no-op-or-cuda-sync helper safe for either backend.

Pin to a specific DirectML adapter when iGPU + dGPU are both visible:
  set DML_DEVICE_INDEX=1   :: if the dGPU is at adapter 1
  set DML_DEVICE_INDEX=0   :: default
"""

import os
import sys

import torch


def list_devices() -> list[str]:
    try:
        import torch_directml
    except ImportError:
        return []
    n = torch_directml.device_count()
    return [torch_directml.device_name(i) for i in range(n)]


def get_device(prefer: str = "dml") -> torch.device:
    """Return a torch.device for the requested backend.

    prefer="dml"  -> torch_directml.device(<DML_DEVICE_INDEX or 0>)
    prefer="cuda" -> torch.device("cuda") if available, else "cpu"
    prefer="cpu"  -> torch.device("cpu")
    """
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if prefer == "dml":
        try:
            import torch_directml
        except ImportError:
            print("ERROR: torch_directml not installed. "
                  "Install with: pip install torch-directml", file=sys.stderr)
            sys.exit(1)
        idx = int(os.environ.get("DML_DEVICE_INDEX", "0"))
        if idx >= torch_directml.device_count():
            print(f"ERROR: DML_DEVICE_INDEX={idx} but only "
                  f"{torch_directml.device_count()} DirectML adapter(s) visible: "
                  f"{list_devices()}", file=sys.stderr)
            sys.exit(1)
        return torch_directml.device(idx)
    raise ValueError(f"unknown prefer={prefer!r}")


def sync(device: torch.device) -> None:
    """Cross-backend synchronize. DirectML auto-syncs at queue boundaries
    (no explicit sync API), so this is effectively a no-op on DML and a
    real ``torch.cuda.synchronize()`` on CUDA/ROCm."""
    if device.type == "cuda":
        torch.cuda.synchronize()
    # DirectML and CPU need no explicit sync


def describe(device: torch.device) -> str:
    if device.type == "privateuseone":  # torch_directml uses this device type
        try:
            import torch_directml
            return f"DirectML[{torch_directml.device_name(device.index or 0)}]"
        except Exception:
            return f"DirectML[{device}]"
    if device.type == "cuda":
        return f"CUDA[{torch.cuda.get_device_name(device.index or 0)}]"
    return str(device)
