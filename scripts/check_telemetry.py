"""
Script to write NVML telemetry values to validation folder /check_telemetry/{datetime}_{gpu_model}.csv.
When this script is run from the terminal, it logs telemetry readings until the script is stopped.
"""

import csv
import os
import time
from datetime import datetime, timezone

import pynvml


def main():
    pynvml.nvmlInit()
    handle = pynvml.nvmlDeviceGetHandleByIndex(0)

    gpu_name = pynvml.nvmlDeviceGetName(handle)
    if isinstance(gpu_name, bytes):
        gpu_name = gpu_name.decode()
    gpu_model = gpu_name.replace(" ", "_")

    stamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    out_dir = os.path.join(os.path.dirname(__file__), "..", "check_telemetry")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"{stamp}_{gpu_model}.csv")

    columns = [
        "timestamp",
        "gpu_utilization_pct",
        "mem_utilization_pct",
        "mem_used_mb",
        "mem_total_mb",
        "power_draw_w",
        "temperature_c",
        "sm_clock_mhz",
        "mem_clock_mhz",
        "fan_speed_pct",
    ]

    print(f"Logging to {out_path}  (Ctrl+C to stop)")

    with open(out_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(columns)

        try:
            while True:
                util = pynvml.nvmlDeviceGetUtilizationRates(handle)
                mem = pynvml.nvmlDeviceGetMemoryInfo(handle)

                try:
                    fan = pynvml.nvmlDeviceGetFanSpeed(handle)
                except pynvml.NVMLError:
                    fan = -1

                row = [
                    datetime.now(timezone.utc).isoformat(),
                    util.gpu,
                    util.memory,
                    mem.used // (1024 * 1024),
                    mem.total // (1024 * 1024),
                    pynvml.nvmlDeviceGetPowerUsage(handle) / 1000.0,
                    pynvml.nvmlDeviceGetTemperature(handle, pynvml.NVML_TEMPERATURE_GPU),
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_SM),
                    pynvml.nvmlDeviceGetClockInfo(handle, pynvml.NVML_CLOCK_MEM),
                    fan,
                ]

                writer.writerow(row)
                f.flush()
                print(f"  gpu={row[1]}%  mem={row[2]}%  power={row[5]}W  temp={row[6]}C  sm={row[7]}MHz", end="\r")
                time.sleep(1)
        except KeyboardInterrupt:
            print(f"\nStopped. Wrote {out_path}")
        finally:
            pynvml.nvmlShutdown()


if __name__ == "__main__":
    main()
