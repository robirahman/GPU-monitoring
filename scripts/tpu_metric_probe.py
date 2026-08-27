#!/usr/bin/env python3
"""Exhaustively probe every way to read TPU accelerator metrics, and report which
ones actually work on this runtime + libtpu combination.

WHY THIS EXISTS. The first real TPU collection produced 245 rows per chip in
which every accelerator metric was zero or -1: `tpu_metrics_live=0` throughout,
and not just the TPU-native comm channels but `gpu_utilization_pct`,
`mem_used_mb` and `mem_utilization_pct` too. Only host CPU/memory were real. So
the corpus gained no TPU telemetry at all, and the Phase 0 hope that
`dcn_collective_lat_us_*` might recover what power provides on GPU could not even
be tested.

The collector tries libtpu.sdk.tpumonitoring -> tpu_info (python) -> tpu-info
(CLI) -> psutil, with every call guarded, so a total failure is silent by design.
That guarding is right for collection and useless for diagnosis: this script is
the opposite, printing exactly what each path raised.

Run it WHILE A JAX PROCESS HOLDS THE TPU where possible (`--hold`), because
several sources only exist once the TPU runtime is initialised -- the earlier
probe ran against an idle chip, which may be part of why it saw nothing.

Usage on a TPU worker:
    python3 scripts/tpu_metric_probe.py --hold 45 --json /tmp/probe.json
"""

import argparse
import importlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import traceback

RESULT: dict = {"paths": {}, "env": {}, "notes": []}


def rec(name, ok, detail, data=None):
    RESULT["paths"][name] = {"ok": bool(ok), "detail": str(detail)[:2000],
                             "data": data}
    flag = "OK  " if ok else "FAIL"
    print(f"[{flag}] {name}: {str(detail)[:220]}", flush=True)


def probe_env():
    keys = ["TPU_WORKER_ID", "TPU_WORKER_HOSTNAMES", "TPU_ACCELERATOR_TYPE",
            "TPU_CHIPS_PER_HOST_BOUNDS", "TPU_SKIP_MDS_QUERY", "PJRT_DEVICE",
            "JAX_PLATFORMS", "LIBTPU_INIT_ARGS", "TPU_STDERR_LOG_LEVEL",
            "MEGASCALE_NUM_SLICES", "TPU_TOPOLOGY", "TPU_RUNTIME_METRICS_PORTS"]
    RESULT["env"] = {k: os.environ.get(k) for k in keys}
    print("env:", {k: v for k, v in RESULT["env"].items() if v}, flush=True)
    for p in ("/dev/accel0", "/dev/vfio", "/dev/vfio/vfio"):
        RESULT["paths"][f"exists:{p}"] = {"ok": os.path.exists(p), "detail": ""}
        print(f"  exists {p}: {os.path.exists(p)}", flush=True)


def probe_libtpu():
    """libtpu.sdk.tpumonitoring is the collector's richest intended source."""
    try:
        libtpu = importlib.import_module("libtpu")
        rec("import libtpu", True,
            f"version={getattr(libtpu,'__version__','?')} "
            f"file={getattr(libtpu,'__file__','?')} "
            f"attrs={[a for a in dir(libtpu) if not a.startswith('_')][:20]}")
    except Exception as e:
        rec("import libtpu", False, f"{type(e).__name__}: {e}")
        return
    # sdk submodule -- path has moved across releases, so try several
    for modpath in ("libtpu.sdk", "libtpu.sdk.tpumonitoring",
                    "libtpu.tpumonitoring", "libtpu_monitoring"):
        try:
            m = importlib.import_module(modpath)
            rec(f"import {modpath}", True,
                f"attrs={[a for a in dir(m) if not a.startswith('_')][:25]}")
        except Exception as e:
            rec(f"import {modpath}", False, f"{type(e).__name__}: {e}")
    # the actual metric API
    try:
        from libtpu.sdk import tpumonitoring as tm
        try:
            names = list(tm.list_supported_metrics())
            rec("tpumonitoring.list_supported_metrics", True,
                f"{len(names)} metrics", data=[str(n) for n in names][:60])
        except Exception as e:
            rec("tpumonitoring.list_supported_metrics", False,
                f"{type(e).__name__}: {e}")
            names = []
        for n in list(names)[:12]:
            try:
                v = tm.get_metric(n)
                rec(f"get_metric({n})", True, repr(v)[:200])
            except Exception as e:
                rec(f"get_metric({n})", False, f"{type(e).__name__}: {e}")
    except Exception as e:
        rec("from libtpu.sdk import tpumonitoring", False,
            f"{type(e).__name__}: {e}")


def probe_tpu_info():
    """tpu_info python API + CLI. Reads the runtime metrics gRPC endpoint."""
    for modpath in ("tpu_info", "tpu_info.device", "tpu_info.metrics",
                    "tpu_info.cli"):
        try:
            m = importlib.import_module(modpath)
            rec(f"import {modpath}", True,
                f"attrs={[a for a in dir(m) if not a.startswith('_')][:25]}")
        except Exception as e:
            rec(f"import {modpath}", False, f"{type(e).__name__}: {e}")
    try:
        from tpu_info import device
        for fn in ("get_local_chips", "get_chip_type", "get_device_type"):
            if hasattr(device, fn):
                try:
                    rec(f"device.{fn}()", True, repr(getattr(device, fn)())[:200])
                except Exception as e:
                    rec(f"device.{fn}()", False, f"{type(e).__name__}: {e}")
    except Exception as e:
        rec("tpu_info.device usage", False, f"{type(e).__name__}: {e}")
    try:
        from tpu_info import metrics as tmetrics
        for fn in [a for a in dir(tmetrics) if not a.startswith("_")][:12]:
            obj = getattr(tmetrics, fn)
            if callable(obj):
                try:
                    rec(f"metrics.{fn}()", True, repr(obj())[:300])
                except Exception as e:
                    rec(f"metrics.{fn}()", False, f"{type(e).__name__}: {e}")
    except Exception as e:
        rec("tpu_info.metrics usage", False, f"{type(e).__name__}: {e}")
    exe = shutil.which("tpu-info")
    if exe:
        for args in (["--metric", "all"], ["--metrics"], []):
            try:
                out = subprocess.run([exe] + args, capture_output=True,
                                     text=True, timeout=60)
                rec(f"cli tpu-info {' '.join(args) or '(no args)'}",
                    out.returncode == 0,
                    (out.stdout or out.stderr)[:1200])
            except Exception as e:
                rec(f"cli tpu-info {args}", False, f"{type(e).__name__}: {e}")
    else:
        rec("which tpu-info", False, "not on PATH")


def probe_jax_memstats():
    """jax device memory_stats: the one HBM source that needs no libtpu monitoring.

    If this works it is enough to populate mem_used_mb / mem_total_mb, which are
    the two features the Phase 0 floor actually leans on.
    """
    try:
        import jax
        devs = jax.local_devices()
        rec("jax.local_devices", True,
            f"{len(devs)} devices kind={devs[0].device_kind if devs else '?'} "
            f"platform={devs[0].platform if devs else '?'}")
        for i, d in enumerate(devs[:4]):
            try:
                ms = d.memory_stats()
                rec(f"device[{i}].memory_stats", ms is not None,
                    json.dumps({k: ms[k] for k in list(ms)[:8]}) if ms else "None",
                    data=ms if isinstance(ms, dict) else None)
            except Exception as e:
                rec(f"device[{i}].memory_stats", False, f"{type(e).__name__}: {e}")
    except Exception as e:
        rec("jax memory_stats", False, f"{type(e).__name__}: {e}")


def probe_grpc_endpoint():
    """The runtime metrics server tpu-info talks to (default localhost:8431)."""
    import socket
    for port in (8431, 8432, 8471):
        s = socket.socket()
        s.settimeout(3)
        try:
            s.connect(("localhost", port))
            rec(f"tcp localhost:{port}", True, "open")
        except Exception as e:
            rec(f"tcp localhost:{port}", False, f"{type(e).__name__}: {e}")
        finally:
            s.close()


def hold_tpu(seconds):
    """Keep a JAX computation resident so runtime-dependent sources exist."""
    def _work():
        try:
            import jax, jax.numpy as jnp
            x = jnp.ones((2048, 2048))
            end = time.time() + seconds
            acc = x
            while time.time() < end:
                acc = jnp.tanh(acc @ x) * 1.0001
                jax.block_until_ready(acc)
        except Exception:
            RESULT["notes"].append("hold thread: " + traceback.format_exc()[:500])
    t = threading.Thread(target=_work, daemon=True)
    t.start()
    time.sleep(min(12, seconds / 3 + 3))   # let the runtime come up
    return t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--hold", type=int, default=45,
                    help="seconds to keep a JAX workload on the TPU while probing")
    ap.add_argument("--json", type=str, default=None)
    a = ap.parse_args()

    print("=" * 72)
    print("TPU METRIC PROBE")
    print("=" * 72)
    probe_env()
    if a.hold > 0:
        print(f"\n-- starting {a.hold}s JAX hold so the TPU runtime is live --",
              flush=True)
        hold_tpu(a.hold)
    print("\n-- libtpu --", flush=True);        probe_libtpu()
    print("\n-- tpu_info --", flush=True);      probe_tpu_info()
    print("\n-- jax memory_stats --", flush=True); probe_jax_memstats()
    print("\n-- runtime metrics endpoint --", flush=True); probe_grpc_endpoint()
    print("\n-- ListSupportedMetrics (authoritative) --", flush=True)
    probe_list_supported(os.environ.get("SPAR_TPU_METRICS_ADDR", "localhost:8431"))

    ok = sorted(k for k, v in RESULT["paths"].items() if v["ok"])
    print("\n" + "=" * 72)
    print(f"WORKING PATHS ({len(ok)}):")
    for k in ok:
        print(f"  + {k}")
    print("=" * 72)
    if a.json:
        with open(a.json, "w") as f:
            json.dump(RESULT, f, indent=1, default=str)
        print(f"wrote {a.json}")


if __name__ == "__main__":
    main()


def probe_list_supported(addr="localhost:8431"):
    """Ask the server what it ACTUALLY serves, then read each one.

    This is the decisive call. With TPU_RUNTIME_METRICS_PORTS set, the server
    answers -- but tpu_info 0.14.2's MetricName list does not match libtpu
    0.0.17's actual metric set, and the server says so explicitly:
      NOT_FOUND "The requested metric: tpu.runtime.hbm.utilization.percent does
      not exist.; Please call `ListSupportedMetrics` RPC to get all the supported
      metics."
    So stop guessing names from the client library and enumerate them.

    Note megascale.* answered UNAVAILABLE "has no accumulated data yet" rather
    than NOT_FOUND -- it exists but only accumulates on a MULTI-SLICE job, which
    is why a single 16-chip slice cannot produce the DCN channel.
    """
    try:
        import grpc
        from tpu_info.proto import tpu_metric_service_pb2 as pb
        from tpu_info.proto import tpu_metric_service_pb2_grpc as pb_grpc
    except Exception as e:
        rec("grpc stub import", False, f"{type(e).__name__}: {e}")
        return
    try:
        ch = grpc.insecure_channel(addr)
        stub = pb_grpc.RuntimeMetricServiceStub(ch)
        resp = stub.ListSupportedMetrics(
            pb.ListSupportedMetricsRequest(), timeout=20)
        names = [m.metric_name for m in resp.supported_metric]
        rec("ListSupportedMetrics", True, f"{len(names)} metrics served",
            data=names)
        print(f"\n  --- {len(names)} SUPPORTED METRICS ---", flush=True)
        for n in names:
            print(f"    {n}", flush=True)
        print("\n  --- reading each ---", flush=True)
        for n in names:
            try:
                r = stub.GetRuntimeMetric(
                    pb.MetricRequest(metric_name=n), timeout=15)
                got = r.metric
                sample = ""
                for m in list(got.metrics)[:1]:
                    if m.HasField("gauge"):
                        g = m.gauge
                        sample = (f"gauge={g.as_double or g.as_int or g.as_string}")
                    elif m.HasField("counter"):
                        sample = f"counter={m.counter.as_double or m.counter.as_int}"
                    elif m.HasField("distribution"):
                        d = m.distribution
                        sample = (f"dist count={d.count} mean={d.mean:.3f} "
                                  f"min={d.min:.3f} max={d.max:.3f}")
                rec(f"metric:{n}", True,
                    f"n={len(got.metrics)} {sample}"[:300])
            except Exception as e:
                rec(f"metric:{n}", False,
                    f"{type(e).__name__}: {str(e).splitlines()[0][:160]}")
    except Exception as e:
        rec("ListSupportedMetrics", False, f"{type(e).__name__}: {e}")
