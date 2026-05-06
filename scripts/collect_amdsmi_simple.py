"""Simple AMD telemetry collector — analogue of scripts/collect_nvml_telemetry.py.

Lightweight single-file collector with no Tier 2 / RDC, no DataFrame
metadata, and no schema cross-walk. Useful for smoke tests and ad-hoc runs.

Usage:
    python collect_amdsmi_simple.py --gpu 0 --seconds 60 --hz 1 \
        --out data/raw_telemetry/amdsmi_gpu0.parquet
"""

import argparse
import os
import sys
import time
from datetime import datetime, timezone

import pandas as pd

_AMDSMI_PYTHON = "/opt/rocm/share/amd_smi"
if os.path.isdir(_AMDSMI_PYTHON) and _AMDSMI_PYTHON not in sys.path:
    sys.path.insert(0, _AMDSMI_PYTHON)

import amdsmi


def _enum(cls_name, member):
    cls = getattr(amdsmi, cls_name, None)
    if cls is not None and hasattr(cls, member):
        return getattr(cls, member)
    return None


def sample_gpu(handle, gpu_index: int):
    activity = {}
    try:
        activity = amdsmi.amdsmi_get_gpu_activity(handle) or {}
    except Exception:
        pass

    mem_used = -1
    mem_total = -1
    try:
        mem_type = _enum("AmdSmiMemoryType", "VRAM")
        mem_used = amdsmi.amdsmi_get_gpu_memory_usage(handle, mem_type)
        mem_total = amdsmi.amdsmi_get_gpu_memory_total(handle, mem_type)
    except Exception:
        pass

    power_w = None
    try:
        info = amdsmi.amdsmi_get_power_info(handle) or {}
        v = info.get("current_socket_power") or info.get("average_socket_power")
        power_w = float(v) if v is not None else None
    except Exception:
        pass

    temp_c = None
    try:
        edge = _enum("AmdSmiTemperatureType", "EDGE")
        cur = _enum("AmdSmiTemperatureMetric", "CURRENT")
        if edge is not None and cur is not None:
            temp_c = amdsmi.amdsmi_get_temp_metric(handle, edge, cur)
    except Exception:
        pass

    gfx_clock = None
    mem_clock = None
    try:
        gfx_t = _enum("AmdSmiClkType", "GFX")
        mem_t = _enum("AmdSmiClkType", "MEM")
        if gfx_t is not None:
            info = amdsmi.amdsmi_get_clock_info(handle, gfx_t) or {}
            gfx_clock = info.get("clk")
        if mem_t is not None:
            info = amdsmi.amdsmi_get_clock_info(handle, mem_t) or {}
            mem_clock = info.get("clk")
    except Exception:
        pass

    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_index": gpu_index,
        "gpu_util_pct": int(activity.get("gfx_activity", -1) or -1),
        "mem_util_pct": int(activity.get("umc_activity", -1) or -1),
        "mm_util_pct": int(activity.get("mm_activity", -1) or -1),
        "mem_used_mb": int(mem_used) // (1024**2) if mem_used and mem_used > 0 else -1,
        "mem_total_mb": int(mem_total) // (1024**2) if mem_total and mem_total > 0 else -1,
        "power_w": power_w,
        "temp_c": temp_c,
        "gfx_clock_mhz": gfx_clock,
        "mem_clock_mhz": mem_clock,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", type=int, default=0)
    ap.add_argument("--seconds", type=int, default=60, help="duration to run")
    ap.add_argument("--hz", type=float, default=1.0, help="samples per second")
    ap.add_argument("--out", type=str, default="", help="output path (.parquet or .csv)")
    args = ap.parse_args()

    period = 1.0 / args.hz
    if not args.out:
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        args.out = f"data/raw_telemetry/amdsmi_gpu{args.gpu}_{stamp}.parquet"

    amdsmi.amdsmi_init()
    try:
        handles = amdsmi.amdsmi_get_processor_handles()
        if args.gpu >= len(handles):
            print(f"ERROR: gpu={args.gpu} but amdsmi reports {len(handles)} GPU(s).", file=sys.stderr)
            sys.exit(1)
        handle = handles[args.gpu]

        asic = {}
        try:
            asic = amdsmi.amdsmi_get_gpu_asic_info(handle) or {}
        except Exception:
            pass
        name = asic.get("market_name") or asic.get("vendor_name") or "AMD GPU"

        print(f"Logging amdsmi telemetry for GPU {args.gpu}: {name}")
        print(f"Writing -> {args.out}")

        rows = []
        t0 = time.time()
        while True:
            now = time.time()
            if now - t0 >= args.seconds:
                break
            rows.append(sample_gpu(handle, args.gpu))
            elapsed = time.time() - now
            time.sleep(max(0.0, period - elapsed))

        df = pd.DataFrame(rows)
        os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
        if args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        else:
            df.to_parquet(args.out, index=False)

        print(f"Done. Collected {len(df)} samples at ~{args.hz} Hz.")
        print(df.tail(3).to_string(index=False))
    finally:
        amdsmi.amdsmi_shut_down()


if __name__ == "__main__":
    main()
