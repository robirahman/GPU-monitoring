#!/usr/bin/env python3
"""Enumerate EVERY metric the libtpu metrics server serves, and try to read each.

Run while a workload holds the TPU: the server lives inside the workload process
(TPU_RUNTIME_METRICS_PORTS), so this must be a separate process talking gRPC to
localhost:8431, not something that grabs the chip.
"""
import json, os, sys

ADDR = os.environ.get("SPAR_TPU_METRICS_ADDR", "localhost:8431")
MAPPED = {
    "tpu.runtime.tensorcore.dutycycle.percent",
    "tpu.runtime.hbm.memory.usage.bytes",
    "tpu.runtime.hbm.memory.total.bytes",
    "hlo.execution.timing.distribution.microseconds",
    "hlo.queue.size.gauge",
    "megascale.collective_end_to_end_latencies.microsecond.cumulative.distribution",
    "megascale.dcn_transfer_latencies.microsecond.cumulative.distribution",
}

try:
    import grpc
    from tpu_info.proto import tpu_metric_service_pb2 as pb
    from tpu_info.proto import tpu_metric_service_pb2_grpc as pbg
except Exception as e:
    print(f"IMPORT_FAIL {type(e).__name__}: {e}"); sys.exit(1)

ch = grpc.insecure_channel(ADDR)
stub = pbg.RuntimeMetricServiceStub(ch)
try:
    resp = stub.ListSupportedMetrics(pb.ListSupportedMetricsRequest(), timeout=20)
except Exception as e:
    print(f"LIST_FAIL {type(e).__name__}: {e}"); sys.exit(2)

names = []
for m in resp.supported_metric:
    n = getattr(m, "metric_name", "")   # NOT m.metric.name -- reading the wrong
    # field is why an earlier attempt reported "SERVED 0" while the collector was
    # happily pulling values from the very same server.
    if n:
        names.append(n)
print(f"SERVED {len(names)} metric(s)")

def describe(name):
    try:
        r = stub.GetRuntimeMetric(pb.MetricRequest(metric_name=name), timeout=15)
    except Exception as e:
        return f"ERR {type(e).__name__}: {str(e)[:90]}"
    out = []
    for d in getattr(getattr(r, "metric", r), "metrics", []) or []:
        which = d.WhichOneof("value") if hasattr(d, "WhichOneof") else None
        if which == "gauge":
            g = d.gauge
            w2 = g.WhichOneof("value") if hasattr(g, "WhichOneof") else None
            out.append(f"gauge/{w2}={getattr(g, w2, None) if w2 else None}")
        elif which == "distribution":
            dd = d.distribution
            out.append(f"dist(count={dd.count} mean={dd.mean:.3f} "
                       f"min={dd.min} max={dd.max})")
        else:
            out.append(f"{which}")
    return "; ".join(out[:3]) if out else "EMPTY"

rows = []
for n in sorted(names):
    tag = "MAPPED  " if n in MAPPED else "UNMAPPED"
    val = describe(n)
    rows.append({"name": n, "mapped": n in MAPPED, "value": val})
    print(f"[{tag}] {n}\n            {val}")

missing = [n for n in MAPPED if n not in names]
print(f"\nMAPPED-BUT-NOT-SERVED ({len(missing)}): {missing}")
print(f"UNMAPPED-BUT-SERVED  ({sum(1 for r in rows if not r['mapped'])}): "
      f"{[r['name'] for r in rows if not r['mapped']]}")
with open("/tmp/metric_enum.json", "w") as f:
    json.dump(rows, f, indent=1)
