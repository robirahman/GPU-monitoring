"""amdsmi smoke test — analogue of nvml_smoke_test.py.

Initializes amdsmi, prints one line of telemetry per detected GPU, and
exits. If this works, the Tier 1 collector should work.
"""

import os
import sys

_AMDSMI_PYTHON = "/opt/rocm/share/amd_smi"
if os.path.isdir(_AMDSMI_PYTHON) and _AMDSMI_PYTHON not in sys.path:
    sys.path.insert(0, _AMDSMI_PYTHON)

import amdsmi


def _enum(cls_name, member):
    cls = getattr(amdsmi, cls_name, None)
    if cls is not None and hasattr(cls, member):
        return getattr(cls, member)
    return None


amdsmi.amdsmi_init()
try:
    handles = amdsmi.amdsmi_get_processor_handles()
    if not handles:
        print("ERROR: amdsmi reports no GPUs. Check ROCm install and `rocm-smi` output.")
        sys.exit(1)

    # amdsmi enumerates AMD CPU sockets alongside GPUs and reports the CPU as
    # processor_type AMD_GPU as well (library quirk). The CPU socket fails
    # amdsmi_get_gpu_activity with UNEXPECTED_DATA — use that as the filter.
    for i, h in enumerate(handles):
        asic = amdsmi.amdsmi_get_gpu_asic_info(h) or {}
        name = asic.get("market_name") or asic.get("vendor_name") or "unknown"

        try:
            activity = amdsmi.amdsmi_get_gpu_activity(h) or {}
        except amdsmi.AmdSmiLibraryException:
            print(f"Processor {i}: {name}  (skipping — not a graphics device)")
            continue

        mem_type = _enum("AmdSmiMemoryType", "VRAM")
        used = amdsmi.amdsmi_get_gpu_memory_usage(h, mem_type)
        total = amdsmi.amdsmi_get_gpu_memory_total(h, mem_type)
        power = (amdsmi.amdsmi_get_power_info(h) or {}).get("current_socket_power")

        edge = _enum("AmdSmiTemperatureType", "EDGE")
        cur = _enum("AmdSmiTemperatureMetric", "CURRENT")
        temp = amdsmi.amdsmi_get_temp_metric(h, edge, cur) if edge and cur else None

        print(f"GPU {i}: {name}")
        print(f"  Activity: gfx={activity.get('gfx_activity', '?')}%  "
              f"umc={activity.get('umc_activity', '?')}%  "
              f"mm={activity.get('mm_activity', '?')}%")
        print(f"  Memory:   used={int(used) // (1024 ** 2)} MB  "
              f"total={int(total) // (1024 ** 2)} MB")
        print(f"  Power:    {power} W")
        print(f"  Temp:     {temp} C (edge)")
finally:
    amdsmi.amdsmi_shut_down()
