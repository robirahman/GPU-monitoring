"""BMC out-of-band power logger — the counterpart to pico_scope_logger.py and
dcgm_logger.py, adding a fourth sensor tier.

Polls the baseboard management controller in-band via `ipmitool` over the local
KCS interface (/dev/ipmi0), so it needs neither the BMC network path nor BMC
credentials — just passwordless sudo on the host. It records the DCMI **node**
power reading (total server draw: CPU + GPU + everything), which is independent
of NVML (GPU-only) and of the Rogowski scope (GPU aux rail) — a third, node-level
view of power. Each sample is wall-clock stamped and the parquet shares the run's
run_id so it joins the NVML / scope / DCGM streams on epoch / run_id.

Sensor-extensibility contract (mirrors dcgm_logger.py / scope_features.py): a new
probe = a logger writing a per-run sidecar + a per-window feature module.

Usage:
  python scripts/bmc_power_logger.py --seconds 320 --out run_bmc.parquet [--interval 1.0]

Note: the DCMI reading's own sampling period is ~5 s (the BMC averages internally),
so sub-5 s polling oversamples a slowly-updating value — fine for alignment, but
the effective bandwidth is ~0.2 Hz. KCS calls take ~0.3-0.8 s each.
"""
import argparse
import re
import signal
import subprocess
import time

import numpy as np
import pandas as pd

INSTANT = re.compile(r"Instantaneous power reading:\s+(\d+)")
PMIN = re.compile(r"Minimum during sampling period:\s+(\d+)")
PMAX = re.compile(r"Maximum during sampling period:\s+(\d+)")
PAVG = re.compile(r"Average power reading over sample period:\s+(\d+)")

STOP = False
def _sig(_n, _f):
    global STOP
    STOP = True


def read_dcmi():
    """Return (instant, min, max, avg) watts, or Nones on failure."""
    try:
        out = subprocess.run(["sudo", "ipmitool", "dcmi", "power", "reading"],
                             capture_output=True, text=True, timeout=10).stdout
    except Exception:
        return (np.nan, np.nan, np.nan, np.nan)
    def g(rx):
        m = rx.search(out)
        return float(m.group(1)) if m else np.nan
    return (g(INSTANT), g(PMIN), g(PMAX), g(PAVG))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=320.0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--run-id", type=str, default="")
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    rows = []
    t0 = time.time()
    while not STOP and (time.time() - t0) < args.seconds:
        tick = time.time()
        inst, pmin, pmax, pavg = read_dcmi()
        rows.append({"epoch": time.time(), "bmc_power_w": inst,
                     "bmc_power_min_w": pmin, "bmc_power_max_w": pmax,
                     "bmc_power_avg_w": pavg})
        dt = args.interval - (time.time() - tick)
        if dt > 0:
            time.sleep(dt)

    df = pd.DataFrame(rows)
    if args.run_id:
        df["run_id"] = args.run_id
    df.to_parquet(args.out)
    n = len(df)
    good = df["bmc_power_w"].notna().mean() if n else 0
    mean = df["bmc_power_w"].mean() if n else float("nan")
    print(f"wrote {args.out}: {n} samples, {good:.0%} valid, mean {mean:.0f} W", flush=True)


if __name__ == "__main__":
    main()
