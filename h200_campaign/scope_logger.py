#!/usr/bin/env python3
"""PicoScope 5464E+ streaming logger with explicit sample accounting.

Replaces scripts/pico_scope_logger.py for this campaign. That logger dropped the
samples delivered together with PICO_WAITING_FOR_DATA_BUFFERS, ignored the
driver's buffer identity and never stored over-range. Here:

* pypicosdk.StreamingSession owns double-buffer rotation and returns the data
  delivered with status 407.
* Every poll that delivers data is recorded (host wall/monotonic ns, driver
  buffer index, start index, sample count, status, over-range flag, cumulative
  raw index) and checked for contiguity: a poll must start where the previous
  one ended in the same buffer, or at 0 in the next buffer.
* ADC over-range (driver flag, per poll) and rail-clipped samples (|count| at the
  ADC limit, per envelope bin) are stored separately. Neither is a lost-sample
  count.
* The stream is acquired at 19.53 MS/s and FIR-decimated x20 (decimate.py), so
  the stored ~977 kS/s waveform is alias-protected below 400 kHz.

Outputs in --out-dir (prefix = --prefix):
  {prefix}_scope_env_NNNN.parquet   envelope bins (977 decimated samples ~ 1.0005 ms)
  {prefix}_scope_polls_NNNN.parquet one row per data-delivering driver poll
  {prefix}_scope_raw_dec_*.npy      decimated waveform segments (int16 counts)
  {prefix}_scope_raw_hires_*.npy    optional undecimated 19.53 MS/s segments
  {prefix}_scope_segments.parquet   index of every saved raw segment
  {prefix}_scope_meta.json          settings, versions, counts, health summary
"""
from __future__ import annotations

import argparse
import json
import os
import resource
import signal
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from h200_campaign.decimate import Decimator  # noqa: E402

PROBE_V_PER_A_CLAIMED = 0.100   # RCP60XS setting recorded by the earlier campaign; not a calibration
ENV_BIN = 977                   # decimated samples per envelope bin
FLUSH_SEC = 60.0

STOP = False


def _sig(_n, _f):
    global STOP
    STOP = True


def parse_schedule(spec: str):
    """'all' | 'none' | 'every:PERIOD:LEN[:OFFSET]' (seconds from stream start)."""
    if spec in ("all", "none"):
        return spec, None
    kind, *vals = spec.split(":")
    assert kind == "every" and len(vals) in (2, 3), spec
    period, length = float(vals[0]), float(vals[1])
    offset = float(vals[2]) if len(vals) == 3 else 0.0
    return "every", (period, length, offset)


def in_schedule(sched, t_rel: float) -> bool:
    kind, p = sched
    if kind == "all":
        return True
    if kind == "none":
        return False
    period, length, offset = p
    if t_rel < offset:
        return False
    return ((t_rel - offset) % period) < length


class SegmentWriter:
    """Accumulates contiguous runs of samples that fall inside a schedule."""

    def __init__(self, out_dir, prefix, tag, fs, sched, index_rows, max_seg_sec=60.0):
        self.out_dir, self.prefix, self.tag, self.fs = out_dir, prefix, tag, fs
        self.sched, self.rows = sched, index_rows
        self.max_n = int(max_seg_sec * fs)
        self.cur, self.cur_start, self.n = [], None, 0

    def feed(self, x: np.ndarray, first_index: int):
        if self.sched[0] == "none" or x.size == 0:
            return
        # schedule resolution: one decision per 10 ms block keeps this cheap
        blk = max(1, int(self.fs * 0.01))
        for s in range(0, x.size, blk):
            idx = first_index + s
            part = x[s:s + blk]
            if in_schedule(self.sched, idx / self.fs):
                if self.cur_start is None:
                    self.cur_start = idx
                self.cur.append(part)
                self.n += part.size
                if self.n >= self.max_n:
                    self.close()
            elif self.cur_start is not None:
                self.close()

    def close(self):
        if self.cur_start is None:
            return
        arr = np.concatenate(self.cur)
        name = f"{self.prefix}_scope_raw_{self.tag}_{self.cur_start:013d}.npy"
        np.save(os.path.join(self.out_dir, name), arr)
        self.rows.append({"kind": self.tag, "file": name, "first_index": int(self.cur_start),
                          "n_samples": int(arr.size), "fs_hz": self.fs,
                          "t_rel_start_s": self.cur_start / self.fs})
        self.cur, self.cur_start, self.n = [], None, 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--seconds", type=float, default=86400.0, help="hard cap; normally stopped by SIGTERM")
    ap.add_argument("--range", default="V5", help="pypicosdk RANGE name, e.g. V1, V2, V5")
    ap.add_argument("--channel", default="A")
    ap.add_argument("--interval-ns", type=float, default=51.2)
    ap.add_argument("--bandwidth", default="BW_20MHZ")
    ap.add_argument("--raw", default="all", help="decimated raw schedule: all | none | every:P:L[:O]")
    ap.add_argument("--hires", default="none", help="undecimated raw schedule (validation only)")
    ap.add_argument("--ready-file", default="", help="touched once streaming has started")
    args = ap.parse_args()

    import pypicosdk as psdk
    from pypicosdk import BANDWIDTH_CH, CHANNEL, COUPLING, RANGE, RESOLUTION, TIME_UNIT, UNIT_INFO

    signal.signal(signal.SIGTERM, _sig)
    signal.signal(signal.SIGINT, _sig)
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    caught_warnings = []
    warnings.showwarning = lambda msg, cat, *a, **k: caught_warnings.append(f"{cat.__name__}: {msg}")

    scope = psdk.psospa()
    scope.open_unit(resolution=RESOLUTION._16BIT)
    info = {}
    for k in ("PICO_DRIVER_VERSION", "PICO_USB_VERSION", "PICO_HARDWARE_VERSION", "PICO_VARIANT_INFO",
              "PICO_BATCH_AND_SERIAL", "PICO_CAL_DATE", "PICO_DIGITAL_HARDWARE_VERSION",
              "PICO_ANALOGUE_HARDWARE_VERSION", "PICO_FIRMWARE_VERSION_1", "PICO_FIRMWARE_VERSION_2"):
        info[k] = scope.get_unit_info(getattr(UNIT_INFO, k))
    ch = getattr(CHANNEL, args.channel)
    rng = getattr(RANGE, args.range)
    scope.set_channel(ch, range=rng, coupling=COUPLING.DC, bandwidth=getattr(BANDWIDTH_CH, args.bandwidth))
    adc_min, adc_max = scope.get_adc_limits()
    range_v = {"mV200": 0.2, "mV500": 0.5, "V1": 1.0, "V2": 2.0, "V5": 5.0, "V10": 10.0, "V20": 20.0}[args.range]
    v_per_count = range_v / adc_max

    raw_sched, hires_sched = parse_schedule(args.raw), parse_schedule(args.hires)
    env_rows, poll_rows, seg_rows = [], [], []
    part = {"env": 0, "polls": 0}

    def flush(kind, rows):
        if rows:
            pd.DataFrame(rows).to_parquet(out / f"{args.prefix}_scope_{kind}_{part[kind]:04d}.parquet")
            part[kind] += 1
            rows.clear()

    counts = dict(raw_samples=0, dec_samples=0, polls=0, contiguity_breaks=0, overrange_polls=0,
                  rail_samples=0, env_bins=0, status={})
    prev = None
    raw_total = 0
    dec_total = 0
    dec_pending = np.empty(0)
    dec_pending_start = 0
    raw_pending = np.empty(0, dtype=np.int16)    # raw samples for the current (partial) env bin
    ru0 = resource.getrusage(resource.RUSAGE_SELF)
    t_wall0 = time.time_ns()
    last_flush = time.monotonic()
    first_poll = None

    with psdk.StreamingSession(scope, sample_interval=args.interval_ns, time_units=TIME_UNIT.NS,
                               samples_per_buffer=4_000_000) as st:
        actual_ns = float(st.actual_sample_interval)
        fs = 1e9 / actual_ns
        dec = Decimator(fs)
        fs_dec = dec.fs_out
        raw_bin = ENV_BIN * dec.factor
        dec_writer = SegmentWriter(out, args.prefix, "dec", fs_dec, raw_sched, seg_rows)
        hires_writer = SegmentWriter(out, args.prefix, "hires", fs, hires_sched, seg_rows, max_seg_sec=5.0)
        if args.ready_file:
            Path(args.ready_file).write_text(json.dumps({"wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns()}))
        for chunk in st:
            wall_ns, mono_ns = time.time_ns(), time.monotonic_ns()
            li = st.last_info["channels"][0]
            status = int(st.last_info["status"])
            b, s0, n = int(li["Buffer index"]), int(li["start index"]), int(li["no of samples"])
            brk = 0
            if prev is not None:
                pb, pend = prev
                if (b == pb and s0 != pend) or (b != pb and s0 != 0) or b < pb or b > pb + 1:
                    brk = 1
            prev = (b, s0 + n)
            ovr = int(bool(li["overflowed?"]))
            x = chunk.data[ch]
            poll_rows.append({"wall_ns": wall_ns, "mono_ns": mono_ns, "buffer_index": b, "start_index": s0,
                              "n_samples": n, "status": status, "overrange": ovr,
                              "raw_index_first": raw_total, "contiguity_break": brk})
            if first_poll is None:
                first_poll = (wall_ns, mono_ns, raw_total + n)
            counts["polls"] += 1
            counts["contiguity_breaks"] += brk
            counts["overrange_polls"] += ovr
            counts["status"][str(status)] = counts["status"].get(str(status), 0) + 1

            hires_writer.feed(x, raw_total)
            raw_total += x.size
            y = dec.feed(x)
            dec_writer.feed(np.clip(np.rint(y), -32768, 32767).astype(np.int16), dec_total)
            dec_total += y.size

            # envelope bins: aligned on decimated index multiples of ENV_BIN;
            # rail counts use the matching undecimated samples.
            dec_pending = np.concatenate([dec_pending, y])
            raw_pending = np.concatenate([raw_pending, x])
            nb = dec_pending.size // ENV_BIN
            nb = min(nb, raw_pending.size // raw_bin)
            if nb:
                v = dec_pending[:nb * ENV_BIN].reshape(nb, ENV_BIN) * v_per_count
                r = raw_pending[:nb * raw_bin].reshape(nb, raw_bin)
                rail = ((r >= adc_max) | (r <= adc_min)).sum(axis=1)
                first = dec_pending_start // ENV_BIN
                for i in range(nb):
                    env_rows.append({"bin": first + i, "dec_index_first": dec_pending_start + i * ENV_BIN,
                                     "v_mean": float(v[i].mean()), "v_rms_ac": float(v[i].std()),
                                     "v_min": float(v[i].min()), "v_max": float(v[i].max()),
                                     "raw_count_min": int(r[i].min()), "raw_count_max": int(r[i].max()),
                                     "rail_samples": int(rail[i])})
                counts["rail_samples"] += int(rail.sum())
                counts["env_bins"] += nb
                dec_pending = dec_pending[nb * ENV_BIN:]
                raw_pending = raw_pending[nb * raw_bin:]
                dec_pending_start += nb * ENV_BIN
            if time.monotonic() - last_flush > FLUSH_SEC:
                flush("env", env_rows)
                flush("polls", poll_rows)
                last_flush = time.monotonic()
            if STOP or (time.monotonic_ns() - first_poll[1]) / 1e9 > args.seconds:
                break

    scope.close_unit()
    dec_writer.close()
    hires_writer.close()
    flush("env", env_rows)
    flush("polls", poll_rows)
    if seg_rows:
        pd.DataFrame(seg_rows).to_parquet(out / f"{args.prefix}_scope_segments.parquet")

    # sample-clock vs host-clock: lower envelope of (host_mono - nominal sample time)
    polls = pd.concat([pd.read_parquet(p) for p in sorted(out.glob(f"{args.prefix}_scope_polls_*.parquet"))])
    end_idx = polls.raw_index_first + polls.n_samples
    lat = polls.mono_ns / 1e9 - end_idx / fs
    lat = lat - lat.min()
    slope = np.polyfit(end_idx / fs, polls.mono_ns / 1e9, 1)[0] if len(polls) > 2 else float("nan")
    ru1 = resource.getrusage(resource.RUSAGE_SELF)
    elapsed = (time.time_ns() - t_wall0) / 1e9
    meta = {
        "logger": "h200_campaign/scope_logger.py", "unit_info": info, "pypicosdk": psdk.__version__ if hasattr(psdk, "__version__") else "1.7.5",
        "channel": args.channel, "coupling": "DC", "range": args.range, "range_v": range_v,
        "bandwidth_limit": args.bandwidth, "resolution_bits": 16, "adc_limits": [adc_min, adc_max],
        "volts_per_count": v_per_count, "probe_v_per_a_claimed": PROBE_V_PER_A_CLAIMED,
        "requested_interval_ns": args.interval_ns, "actual_interval_ns": actual_ns, "fs_raw_hz": fs,
        "decimation": dec.describe(), "env_bin_decimated_samples": ENV_BIN,
        "raw_schedule": args.raw, "hires_schedule": args.hires,
        "counts": counts | {"raw_samples": raw_total, "dec_samples": dec_total},
        "first_poll": {"wall_ns": first_poll[0], "mono_ns": first_poll[1], "raw_index_end": first_poll[2]},
        "host_vs_sample_clock": {"mono_s_per_sample_s": slope,
                                 "poll_latency_spread_s_p50": float(np.median(lat)),
                                 "poll_latency_spread_s_p99": float(np.quantile(lat, 0.99))},
        "expected_raw_samples_from_host_elapsed": float((polls.mono_ns.iloc[-1] - polls.mono_ns.iloc[0]) / 1e9 * fs),
        "cpu_user_s": ru1.ru_utime - ru0.ru_utime, "cpu_sys_s": ru1.ru_stime - ru0.ru_stime,
        "max_rss_mb": ru1.ru_maxrss / 1024, "elapsed_s": elapsed,
        "bytes_written": sum(p.stat().st_size for p in out.glob(f"{args.prefix}_scope_*")),
        "warnings": caught_warnings[:50], "n_warnings": len(caught_warnings),
    }
    (out / f"{args.prefix}_scope_meta.json").write_text(json.dumps(meta, indent=1, default=str))
    print(json.dumps({k: meta[k] for k in ("counts", "host_vs_sample_clock", "cpu_user_s", "elapsed_s")}, default=str), flush=True)


if __name__ == "__main__":
    main()
