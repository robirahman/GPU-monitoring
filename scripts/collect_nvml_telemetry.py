import time
import argparse
from datetime import datetime, timezone

import pandas as pd
import pynvml


def sample_gpu(handle, gpu_index: int):
    util = pynvml.nvmlDeviceGetUtilizationRates(handle)
    mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

    # Some calls may fail depending on permissions/MIG/etc; handle robustly.
    try:
        power_mw = pynvml.nvmlDeviceGetPowerUsage(handle)
        power_w = power_mw / 1000.0
    except Exception:
        power_w = None

    try:
        temp_c = pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU)
    except Exception:
        temp_c = None

    try:
        sm_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM)
    except Exception:
        sm_clock = None

    try:
        mem_clock = pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM)
    except Exception:
        mem_clock = None

    return {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_index": gpu_index,
        "gpu_util_pct": int(util.gpu),
        "mem_util_pct": int(util.memory),
        "mem_used_mb": int(mem.used // (1024**2)),
        "mem_total_mb": int(mem.total // (1024**2)),
        "power_w": power_w,
        "temp_c": temp_c,
        "sm_clock_mhz": sm_clock,
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
        args.out = f"data/raw_telemetry/nvml_gpu{args.gpu}_{stamp}.parquet"

    pynvml.nvmlInit()
    try:
        handle = pynvml.nvmlDeviceGetHandleByIndex(args.gpu)
        name = pynvml.nvmlDeviceGetName(handle)
        if isinstance(name, bytes):
            name = name.decode("utf-8", errors="ignore")
        print(f"Logging NVML telemetry for GPU {args.gpu}: {name}")
        print(f"Writing -> {args.out}")

        rows = []
        t0 = time.time()
        i = 0
        while True:
            now = time.time()
            if now - t0 >= args.seconds:
                break

            rows.append(sample_gpu(handle, args.gpu))
            i += 1

            # Sleep to maintain target rate
            elapsed = time.time() - now
            to_sleep = max(0.0, period - elapsed)
            time.sleep(to_sleep)

        df = pd.DataFrame(rows)

        if args.out.endswith(".csv"):
            df.to_csv(args.out, index=False)
        else:
            # default parquet
            df.to_parquet(args.out, index=False)

        print(f"Done. Collected {len(df)} samples at ~{args.hz} Hz.")
        print(df.tail(3).to_string(index=False))

    finally:
        pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
