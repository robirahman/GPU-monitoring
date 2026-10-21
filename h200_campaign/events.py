"""Structured phase events (JSONL) with optional CUDA-event GPU timestamps.

Every event line carries run/job/attempt/stream identity, the event type, host
wall-clock UTC ns and monotonic ns, the emitting source location, success flag
and, when a GPU mark was requested, a reference to the CUDA event that resolves
it. Resolution lines ("gpu_time") arrive later, once the GPU has actually
executed the work: a host call returning does not mean the GPU finished it.

GPU clock anchoring: at start-up one reference CUDA event is recorded and
synchronised, and host monotonic time is read immediately after. Later anchors
are recorded every ANCHOR_SEC without synchronising and chained to the previous
anchor, so every elapsed_time() call spans < ANCHOR_SEC (float32 ms precision
stays ~µs). Marks never force synchronisation of the workload's stream.
"""
from __future__ import annotations

import inspect
import json
import os
import time
from pathlib import Path

ANCHOR_SEC = 60.0
SCHEMA = 1


class EventLog:
    def __init__(self, path: str | None = None, *, stream: str = "workload", gpu: bool = False):
        path = path or os.environ.get("H200_EVENTS_PATH")
        self.path = Path(path) if path else None
        self.f = open(self.path, "a", buffering=1) if self.path else None
        self.ident = {"run_id": os.environ.get("SPAR_RUN_ID", ""), "job_id": os.environ.get("SPAR_JOB_ID", ""),
                      "attempt_id": os.environ.get("SPAR_ATTEMPT_ID", "1"), "stream": stream, "pid": os.getpid()}
        self._gpu = None
        if gpu:
            self._gpu = _GpuClock(self)

    def emit(self, event: str, ok: bool = True, gpu_mark: bool = False, _depth: int = 1, **extra):
        fr = inspect.stack()[_depth]
        rec = {"schema": SCHEMA, **self.ident, "event": event, "wall_ns": time.time_ns(),
               "mono_ns": time.monotonic_ns(), "source": f"{Path(fr.filename).name}:{fr.lineno}", "ok": ok}
        if gpu_mark and self._gpu is not None:
            rec["gpu_ref"] = self._gpu.mark(event)
        if extra:
            rec["extra"] = extra
        self._write(rec)
        return rec

    def _write(self, rec):
        line = json.dumps(rec, default=str)
        if self.f:
            self.f.write(line + "\n")
        else:
            print("EVENT " + line, flush=True)

    def poll_gpu(self):
        """Resolve GPU marks that have completed; cheap and non-blocking."""
        if self._gpu is not None:
            self._gpu.resolve(block=False)

    def close(self):
        if self._gpu is not None:
            self._gpu.resolve(block=True)
        if self.f:
            self.f.flush()
            os.fsync(self.f.fileno())
            self.f.close()
            self.f = None


class _GpuClock:
    def __init__(self, log: EventLog):
        import torch
        self.torch, self.log = torch, log
        torch.cuda.synchronize()
        ref = torch.cuda.Event(enable_timing=True)
        ref.record()
        ref.synchronize()
        self.ref_wall_ns, self.ref_mono_ns = time.time_ns(), time.monotonic_ns()
        # anchors: [event, gpu_ns_since_ref or None, host mono at record]
        self.anchors = [[ref, 0.0, self.ref_mono_ns]]
        self.pending = []      # (mark_id, name, event, anchor_idx)
        self.next_id = 0
        log._write({"schema": SCHEMA, **log.ident, "event": "gpu_clock_ref", "wall_ns": self.ref_wall_ns,
                    "mono_ns": self.ref_mono_ns, "source": "events.py", "ok": True,
                    "extra": {"note": "GPU t=0 is this host instant, after ref.synchronize()"}})

    def mark(self, name):
        now = time.monotonic_ns()
        if (now - self.anchors[-1][2]) / 1e9 > ANCHOR_SEC:
            a = self.torch.cuda.Event(enable_timing=True)
            a.record()
            self.anchors.append([a, None, now])
        ev = self.torch.cuda.Event(enable_timing=True)
        ev.record()
        mid = self.next_id
        self.next_id += 1
        self.pending.append((mid, name, ev, len(self.anchors) - 1))
        if len(self.pending) > 256:
            self.resolve(block=False)
        return mid

    def _anchor_ns(self, i, block):
        a = self.anchors[i]
        if a[1] is None:
            prev = self._anchor_ns(i - 1, block)
            if prev is None or not (block or a[0].query()):
                return None
            if block:
                a[0].synchronize()
            a[1] = prev + self.anchors[i - 1][0].elapsed_time(a[0]) * 1e6
        return a[1]

    def resolve(self, block: bool):
        keep = []
        for mid, name, ev, ai in self.pending:
            if not block and not ev.query():
                keep.append((mid, name, ev, ai))
                continue
            if block:
                ev.synchronize()
            base = self._anchor_ns(ai, block)
            if base is None:
                keep.append((mid, name, ev, ai))
                continue
            gpu_ns = base + self.anchors[ai][0].elapsed_time(ev) * 1e6
            self.log._write({"schema": SCHEMA, **self.log.ident, "event": "gpu_time", "gpu_ref": mid,
                             "of": name, "gpu_mono_ns": int(self.ref_mono_ns + gpu_ns),
                             "wall_ns": time.time_ns(), "mono_ns": time.monotonic_ns(), "source": "events.py", "ok": True})
        self.pending = keep
