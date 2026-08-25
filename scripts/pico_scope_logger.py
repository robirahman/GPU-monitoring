"""Continuous PicoScope 5464E+ (psospa) streaming logger for the Rogowski current probe.

Streams channel A, reduces to per-bin envelope stats (mean/rms/min/max), writes parquet.
The Micsig RCP60XS outputs 100 mV/A (AC-coupled by physics: Rogowski coils see dI/dt only).

Usage:
  pico-venv/bin/python pico_stream_logger.py --seconds 130 --out scope_idle.parquet \
      [--rate-hz 1e6] [--bin-ms 1] [--range-mv 2000] [--save-raw raw.npy --raw-seconds 5]
"""
import argparse
import ctypes
import signal
import sys
import time

import numpy as np
import pandas as pd
from picosdk.psospa import psospa as ps
from picosdk.PicoDeviceEnums import picoEnum as enums
from picosdk.PicoDeviceStructs import picoStruct as structs
from picosdk.constants import PICO_STATUS_LOOKUP

PROBE_V_PER_A = 0.100  # Micsig RCP60XS: 100 mV/A

STOP = False
def _sig(_n, _f):
    global STOP
    STOP = True

def status_name(code):
    return PICO_STATUS_LOOKUP.get(code, f"UNKNOWN_{code}")

def check(code, what):
    if code != 0:
        raise RuntimeError(f"{what} failed: {status_name(code)} ({code})")

def open_scope(range_mv):
    handle = ctypes.c_int16()
    check(ps.psospaOpenUnit(ctypes.byref(handle), None,
                            enums.PICO_DEVICE_RESOLUTION["PICO_DR_8BIT"], None), "OpenUnit")
    chA = enums.PICO_CHANNEL["PICO_CHANNEL_A"]
    for ch in ("B", "C", "D"):
        ps.psospaSetChannelOff(handle, enums.PICO_CHANNEL[f"PICO_CHANNEL_{ch}"])
    range_nv = int(range_mv) * 1_000_000
    check(ps.psospaSetChannelOn(handle, chA, enums.PICO_COUPLING["PICO_DC"],
                                -range_nv, range_nv, 0, 0.0,
                                enums.PICO_BANDWIDTH_LIMITER["PICO_BW_FULL"]),
          "SetChannelOn")
    # raise to 16-bit now that only one channel is enabled
    check(ps.psospaSetDeviceResolution(
        handle, enums.PICO_DEVICE_RESOLUTION["PICO_DR_16BIT"]), "SetDeviceResolution(16)")
    min_adc, max_adc = ctypes.c_int16(), ctypes.c_int16()
    ps.psospaGetAdcLimits(handle, enums.PICO_DEVICE_RESOLUTION["PICO_DR_16BIT"],
                          ctypes.byref(min_adc), ctypes.byref(max_adc))
    return handle, chA, max_adc.value

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seconds", type=float, default=130.0)
    ap.add_argument("--out", type=str, required=True)
    ap.add_argument("--rate-hz", type=float, default=1e6)
    ap.add_argument("--bin-ms", type=float, default=1.0)
    ap.add_argument("--range-mv", type=int, default=2000)
    ap.add_argument("--save-raw", type=str, default="")
    ap.add_argument("--raw-seconds", type=float, default=5.0)
    args = ap.parse_args()

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)

    handle, chA, max_adc = open_scope(args.range_mv)
    volts_per_count = (args.range_mv / 1000.0) / max_adc

    ratio_raw = enums.PICO_RATIO_MODE["PICO_RATIO_MODE_RAW"]
    int16_t = enums.PICO_DATA_TYPE["PICO_INT16_T"]

    # double buffers, each ~2 s of data
    buf_len = int(args.rate_hz * 2)
    bufs = [np.zeros(buf_len, dtype=np.int16) for _ in range(2)]
    def register(i, action):
        check(ps.psospaSetDataBuffer(
            handle, chA, bufs[i].ctypes.data_as(ctypes.c_void_p), buf_len,
            int16_t, 0, ratio_raw, action), f"SetDataBuffer[{i}]")
    register(0, 1 | 2)  # PICO_CLEAR_ALL | PICO_ADD

    interval = ctypes.c_double(1e6 / args.rate_hz)  # microseconds
    check(ps.psospaRunStreaming(handle, ctypes.byref(interval),
                                enums.PICO_TIME_UNITS["PICO_US"],
                                0, buf_len, 0, 1, ratio_raw), "RunStreaming")
    actual_rate = 1e6 / interval.value
    t_start = time.time()
    print(f"streaming: {actual_rate:.0f} S/s, range ±{args.range_mv} mV, 16-bit, "
          f"{volts_per_count*1e6:.2f} µV/LSB", flush=True)

    bin_n = max(1, int(round(actual_rate * args.bin_ms / 1000.0)))
    leftover = np.empty(0, dtype=np.int16)
    rows = []
    total = 0
    cur = 0            # buffer currently registered
    consumed = 0       # samples consumed from current buffer
    raw_chunks, raw_budget = [], int(args.raw_seconds * actual_rate) if args.save_raw else 0

    info = structs.PICO_STREAMING_DATA_INFO()
    info.channel, info.mode, info.type = chA, ratio_raw, int16_t
    trig = structs.PICO_STREAMING_DATA_TRIGGER_INFO()

    def flush_bins(arr):
        nonlocal leftover, total, rows
        if leftover.size:
            arr = np.concatenate([leftover, arr])
        nbins = arr.size // bin_n
        if nbins == 0:
            leftover = arr
            return
        chunk, leftover = arr[:nbins*bin_n], arr[nbins*bin_n:]
        v = chunk.astype(np.float32).reshape(nbins, bin_n) * volts_per_count
        t0 = t_start + total / actual_rate
        ts = t0 + (np.arange(nbins) + 0.5) * bin_n / actual_rate
        mean = v.mean(axis=1)
        rows.append(pd.DataFrame({
            "epoch_s": ts,
            "v_mean": mean,
            "v_rms_ac": v.std(axis=1),
            "v_min": v.min(axis=1),
            "v_max": v.max(axis=1),
        }))
        total += nbins * bin_n

    while not STOP and (time.time() - t_start) < args.seconds:
        info.noOfSamples, info.startIndex, info.bufferIndex, info.overflow = 0, 0, 0, 0
        s = ps.psospaGetStreamingLatestValues(handle, ctypes.byref(info), 1, ctypes.byref(trig))
        name = status_name(s)
        if name == "PICO_WAITING_FOR_DATA_BUFFERS":
            # current buffer exhausted: hand the driver the other one
            nxt = 1 - cur
            register(nxt, 2)  # PICO_ADD
            cur, consumed = nxt, 0
            continue
        if s != 0 and name != "PICO_OK":
            print(f"GetStreamingLatestValues: {name}", flush=True)
            if name in ("PICO_BUFFERS_NOT_SET", "PICO_INVALID_BUFFER"):
                break
        end = int(info.startIndex) + int(info.noOfSamples)
        if end > consumed:
            new = bufs[cur][consumed:end].copy()
            consumed = end
            if raw_budget > 0:
                take = min(raw_budget, new.size)
                raw_chunks.append(new[:take].copy())
                raw_budget -= take
            flush_bins(new)
        else:
            time.sleep(0.02)

    ps.psospaStop(handle)
    ps.psospaCloseUnit(handle)

    df = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()
    df["amps_mean"] = df["v_mean"] / PROBE_V_PER_A
    df["amps_rms_ac"] = df["v_rms_ac"] / PROBE_V_PER_A
    df.attrs["sample_rate_hz"] = actual_rate
    df.to_parquet(args.out)
    dur = df["epoch_s"].iloc[-1] - df["epoch_s"].iloc[0] if len(df) else 0
    print(f"wrote {args.out}: {len(df)} bins, {dur:.1f} s span, {total} samples", flush=True)
    if args.save_raw and raw_chunks:
        raw = np.concatenate(raw_chunks)
        np.save(args.save_raw, raw)
        print(f"wrote {args.save_raw}: {raw.size} raw samples "
              f"({raw.size/actual_rate:.1f} s)", flush=True)

if __name__ == "__main__":
    main()
