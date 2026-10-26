"""Decide whether a finished run actually ran.

Every one of the 34 unusable runs in the camera-ready corpus was detectable the
moment it finished: it either stopped a long way short of its `--duration`, or
ran to length with the GPU idle in 99% of its samples. Nothing checked, so they
were found six months later by comparing against a re-run.

This lives apart from collect_telemetry so it can be imported without an NVIDIA
driver: the check is pure pandas, and it is the part worth testing.
"""
from __future__ import annotations

import json
import logging
import re
from pathlib import Path

log = logging.getLogger(__name__)

HEALTH_BUSY_FLOOR = 0.05      # below this the GPU effectively never ran
HEALTH_STARVED_BUSY = 0.10    # ran to length, but hardly used the device
HEALTH_SPAN_FRAC = 0.5        # stopped before half its requested duration


def requested_duration_of(command):
    """`--duration N` out of a workload argv, or None."""
    argv = [str(x) for x in (command or [])]
    for i, tok in enumerate(argv):
        if tok == "--duration" and i + 1 < len(argv):
            try:
                return float(argv[i + 1])
            except ValueError:
                return None
        m = re.match(r"--duration=(\d+(?:\.\d+)?)$", tok)
        if m:
            return float(m.group(1))
    return None


def assess_run_health(df, requested_duration_s=None):
    """Classify a finished run: (verdict, detail).

    verdict is one of 'ok', 'no_samples', 'aborted', 'truncated', 'starved'.
    'aborted' and 'truncated' both mean the run stopped early -- they differ
    on whether the GPU was working while it lasted, which is the difference
    between a workload that never started and one that died partway.
    """
    if df is None or not len(df):
        return "no_samples", "the collector recorded nothing"
    util = df["gpu_utilization_pct"].astype(float) if "gpu_utilization_pct" in df else None
    if util is None or not len(util):
        return "no_samples", "no utilisation column in the frame"
    busy = float((util > 0).mean())
    span = 0.0
    if "timestamp_epoch" in df and len(df) > 1:
        span = float(df["timestamp_epoch"].max() - df["timestamp_epoch"].min())
    want = requested_duration_s
    short = bool(want and span < HEALTH_SPAN_FRAC * want)
    detail = (f"span {span:.0f}s"
              + (f" of {want:.0f}s requested" if want else "")
              + f", GPU busy in {busy:.1%} of {len(df)} samples")
    if short and busy < HEALTH_BUSY_FLOOR:
        return "aborted", detail
    if short:
        return "truncated", detail
    if busy < HEALTH_STARVED_BUSY:
        return "starved", detail
    return "ok", detail


def report_run_health(df, requested_duration_s=None, output_path=None):
    """Assess, log, and drop a .health.json beside the telemetry. Returns verdict."""
    verdict, detail = assess_run_health(df, requested_duration_s)
    if verdict == "ok":
        log.info("Run health: ok (%s)", detail)
    else:
        log.warning("RUN HEALTH: %s -- %s", verdict.upper(), detail)
    if output_path:
        try:
            side = Path(str(output_path)).with_suffix(".health.json")
            side.write_text(json.dumps(
                {"verdict": verdict, "detail": detail}, indent=2) + "\n")
        except OSError as e:
            log.warning("Could not write %s: %s", output_path, e)
    return verdict


