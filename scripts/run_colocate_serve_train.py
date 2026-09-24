#!/usr/bin/env python3
"""W25 -- spatial colocation: most GPUs serve (real vLLM), a few train (real DDP/LoRA/DiLoCo).

Space-multiplex FN / cross-job-structuring attack: one node, disjoint GPU sets. Node-aggregate
fabric counters look serve-dominated; the training job is a small rank set (or a single GPU with
no fabric traffic at all). W24 time-slices the *same* ranks; W25 *partitions* GPUs/jobs and breaks
the "one job = one label" assumption.

This orchestrator runs BOTH real workloads concurrently and owns telemetry for all GPUs:

  * one shared ``run_id``; a ``TelemetryCollector`` per GPU in serve U train, one shared
    ``SharedFabricCache`` (NVLink is per GPU; IB/eth counters are node-level)
  * every telemetry row carries ``workload_role = serve | train`` and ``role_label`` (that role's own
    workload label) next to ``gpu_index`` / ``workload_label`` / ``run_id``
  * children get ``SPAR_WORKLOAD_ROLE`` / ``SPAR_RUN_ID`` / ``SPAR_ROLE_LABEL`` in their env, and
    the interconnect workloads stamp them on every comm/steps JSONL event
  * ``<label>_<run>_roles.json`` manifest: per-role GPUs, resolved commands, window timestamps
    (``t_start`` / ``t_ready`` / ``t_end``), return codes, output files -- the window truth
  * the train role starts once the serve role reports healthy (default marker ``Server healthy``),
    so both run concurrently for ``--duration``

It refuses to mix roles silently: GPU sets must be disjoint and non-empty; a collector that does
not carry the role tag is an error; and it will not run under another telemetry launcher
(``run_workload*.py`` marks children with ``SPAR_TELEMETRY_OWNER``; registry labels here carry
``telemetry_owner`` so those launchers refuse them) -- an outer untagged collector on the same GPUs
is exactly the silent mixing this exists to prevent. ``--no-telemetry`` is allowed for dry
plumbing checks but the manifest then says the run is NOT usable as W25 evidence.

Scoring note (W25): evaluate (a) a node-aggregate detector (fabric summed over the node) AND (b)
per-GPU / per-role detectors. IB/eth rows are node-level and cannot be attributed to a role; only
per-GPU NVLink/NVML rows can.

Examples::

    python scripts/run_colocate_serve_train.py --label colocate_serve7_train1_qwen \\
        --serve-workload moe_serve_qwen7b_tp4dp2_mixed --serve-gpus 0-6 --serve-args "--tp 1 --dp 7" \\
        --train-workload train_lora_only_ddp_qwen15 --train-gpus 7 --duration 300
    python scripts/run_colocate_serve_train.py --label t --serve-cmd "python my_serve.py" \\
        --train-cmd "python my_train.py" --serve-gpus 0-3 --train-gpus 4-5 --dry-run

Registry: ``colocate_serve7_train1_qwen``, ``colocate_serve_tp4_train_ddp2`` (run the script, not
``run_workload_multi_gpu.py``). Status: code (real); orchestration tested against mock servers and
fake collectors, not yet run on GPUs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import shlex
import signal
import subprocess
import sys
import threading
import time
import uuid
from collections import deque
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

log = logging.getLogger("spar.colocate")

OWNER = "run_colocate_serve_train"
NPROC_RE = re.compile(r"^(--nproc[_-]per[_-]node)(=(\d+))?$")


# ── planning helpers (pure; unit-tested) ─────────────────────────────────────

def parse_gpu_list(spec: str) -> list[int]:
    """'0-6' | '0,1,2' | '0-3,6' -> sorted unique ints."""
    out: set[int] = set()
    for part in str(spec).split(","):
        part = part.strip()
        if not part:
            continue
        a, sep, b = part.partition("-")
        try:
            lo, hi = int(a), int(b) if sep else int(a)
        except ValueError:
            raise SystemExit(f"bad GPU list {spec!r}")
        if lo < 0 or hi < lo:
            raise SystemExit(f"bad GPU range {part!r}")
        out.update(range(lo, hi + 1))
    if not out:
        raise SystemExit(f"empty GPU list {spec!r}")
    return sorted(out)


def rewrite_nproc(cmd: list[str], n: int) -> tuple[list[str], bool]:
    """Set torchrun's ``--nproc_per_node`` to ``n`` (registry commands hard-code 8)."""
    out, changed, i = list(cmd), False, 0
    while i < len(out):
        m = NPROC_RE.match(out[i])
        if m:
            if m.group(2):                                   # --nproc_per_node=8
                if m.group(3) != str(n):
                    out[i], changed = f"{m.group(1)}={n}", True
            elif i + 1 < len(out) and out[i + 1] != str(n):  # --nproc_per_node 8
                out[i + 1], changed = str(n), True
        i += 1
    return out, changed


def last_flag(cmd: list[str], flag: str, default: int) -> int:
    val = default
    for i, tok in enumerate(cmd):
        if tok == flag and i + 1 < len(cmd):
            try:
                val = int(cmd[i + 1])
            except ValueError:
                pass
    return val


def resolve_role_cmd(role: str, workload: str | None, cmd_str: str | None, gpus: list[int],
                     extra: list[str], duration: int | None) -> tuple[str, list[str]]:
    """(role workload label, final command) for one role."""
    if bool(workload) == bool(cmd_str):
        raise SystemExit(f"--{role}: give exactly one of --{role}-workload / --{role}-cmd")
    if workload:
        from workloads.registry import get_workload
        try:
            cmd = list(get_workload(workload)["command"])
        except ValueError as e:
            raise SystemExit(str(e))
        label = workload
    else:
        cmd, label = shlex.split(cmd_str), f"custom_{role}"
    cmd, changed = rewrite_nproc(cmd, len(gpus))
    if changed:
        log.info("  [%s] torchrun --nproc_per_node set to %d (one rank per assigned GPU)", role, len(gpus))
    cmd += list(extra)
    if duration is not None:
        if "--duration" in cmd:
            cmd += ["--duration", str(duration)]              # argparse: last wins
        else:
            log.warning("  [%s] command has no --duration; not applying --duration %s", role, duration)
    if any(os.path.basename(t) == "llm_moe_serving.py" for t in cmd):
        tp, dp = last_flag(cmd, "--tp", 1), last_flag(cmd, "--dp", 1)
        if tp * dp != len(gpus):
            raise SystemExit(
                f"serve layout TP={tp} x DP={dp} = {tp * dp} GPUs but --serve-gpus has {len(gpus)}: pass e.g. "
                f"--serve-args \"--tp <tp> --dp <dp>\" (Qwen2.5-7B: TP must be 1, 2 or 4; it has 28 heads)")
    return label, cmd


def role_env(role: str, label: str, run_id: str, coloc_label: str, gpus: list[int]) -> dict[str, str]:
    return {"CUDA_VISIBLE_DEVICES": ",".join(map(str, gpus)), "SPAR_WORKLOAD_ROLE": role,
            "SPAR_ROLE_LABEL": label, "SPAR_RUN_ID": run_id, "SPAR_COLOCATE_LABEL": coloc_label,
            "SPAR_TELEMETRY_OWNER": OWNER}


# ── process handling ─────────────────────────────────────────────────────────

class Role:
    def __init__(self, name: str, label: str, gpus: list[int], cmd: list[str], env: dict[str, str]) -> None:
        self.name, self.label, self.gpus, self.cmd, self.env = name, label, gpus, cmd, env
        self.proc: subprocess.Popen | None = None
        self.t_start = self.t_ready = self.t_end = None
        self.rc: int | None = None
        self.tail: deque = deque(maxlen=40)
        self.ready = threading.Event()

    def start(self, ready_re: re.Pattern | None) -> None:
        self.t_start = time.time()
        self.proc = subprocess.Popen(self.cmd, env={**os.environ, **self.env}, cwd=_REPO_ROOT,
                                     stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        if ready_re is None:
            self.t_ready = self.t_start
            self.ready.set()

        def pump() -> None:
            for line in self.proc.stdout:                    # type: ignore[union-attr]
                self.tail.append(line.rstrip())
                sys.stdout.write(f"[{self.name}] {line}")
                if ready_re is not None and not self.ready.is_set() and ready_re.search(line):
                    self.t_ready = time.time()
                    self.ready.set()
        threading.Thread(target=pump, daemon=True).start()

    def poll(self) -> int | None:
        rc = self.proc.poll() if self.proc else None
        if rc is not None and self.t_end is None:
            self.t_end, self.rc = time.time(), rc
        return rc

    def stop(self, grace: float = 30.0) -> None:
        if self.proc is None or self.proc.poll() is not None:
            return
        self.proc.send_signal(signal.SIGTERM)
        try:
            self.proc.wait(timeout=grace)
        except subprocess.TimeoutExpired:
            self.proc.kill()


def _default_collector_factory(**kw):
    from scripts.collect_telemetry import TelemetryCollector
    return TelemetryCollector(**kw)


def _default_fabric(interval: float, n_gpus: int):
    try:
        from scripts.interconnect_collect import SharedFabricCache, infer_topology
    except ImportError:
        return None, None
    return SharedFabricCache(interval_sec=interval), infer_topology(nproc_per_node=n_gpus)


# ── the run ──────────────────────────────────────────────────────────────────

def run(args: argparse.Namespace, collector_factory=None, fabric_factory=None) -> tuple[dict, int]:
    serve_gpus, train_gpus = parse_gpu_list(args.serve_gpus), parse_gpu_list(args.train_gpus)
    overlap = sorted(set(serve_gpus) & set(train_gpus))
    if overlap:
        raise SystemExit(f"serve and train GPU sets overlap on {overlap}: colocation here is SPATIAL; "
                         "sharing GPUs would mix the roles (that is W24's time-slice, not W25)")
    s_label, s_cmd = resolve_role_cmd("serve", args.serve_workload, args.serve_cmd, serve_gpus,
                                      [t for x in args.serve_args for t in shlex.split(x)], args.duration)
    t_label, t_cmd = resolve_role_cmd("train", args.train_workload, args.train_cmd, train_gpus,
                                      [t for x in args.train_args for t in shlex.split(x)], args.duration)
    run_id = uuid.uuid4().hex
    roles = {"serve": Role("serve", s_label, serve_gpus, s_cmd, role_env("serve", s_label, run_id, args.label, serve_gpus)),
             "train": Role("train", t_label, train_gpus, t_cmd, role_env("train", t_label, run_id, args.label, train_gpus))}

    log.info("W25 colocation %s run_id=%s", args.label, run_id[:12])
    for r in roles.values():
        log.info("  %-5s %-32s GPUs %s", r.name, r.label, r.gpus)
        log.info("        cmd: %s", " ".join(r.cmd))
    if args.dry_run:
        return {"dry_run": True, "run_id": run_id, "roles": {k: {"label": v.label, "gpus": v.gpus, "cmd": v.cmd,
                                                                   "env": v.env} for k, v in roles.items()}}, 0
    if os.environ.get("SPAR_TELEMETRY_OWNER"):
        raise SystemExit(f"refusing to run under another telemetry launcher "
                         f"({os.environ['SPAR_TELEMETRY_OWNER']}): its untagged collectors would mix the roles "
                         f"on the same GPUs. Run this script directly.")

    events: list[dict] = []

    def event(name: str, **kw) -> None:
        events.append({"t": time.time(), "event": name, **kw})

    # -- telemetry: every GPU tagged with its role ---------------------------------
    collectors: list[tuple[str, int, object]] = []
    fabric_cache = None
    if args.no_telemetry:
        log.warning("--no-telemetry: NO fabric/NVML rows; this run is NOT usable as W25 evidence")
    else:
        make = collector_factory or _default_collector_factory
        topology = None
        if not args.no_fabric:
            fabric_cache, topology = (fabric_factory or _default_fabric)(args.interval, len(serve_gpus) + len(train_gpus))
        for role in roles.values():
            for g in role.gpus:
                c = make(gpu_index=g, interval_sec=args.interval, workload_label=args.label, run_id=run_id,
                         enable_dcgm=not args.no_dcgm, enable_fabric=not args.no_fabric,
                         fabric_cache=fabric_cache, topology=topology,
                         workload_role=role.name, role_label=role.label)
                if getattr(c, "_workload_role", None) != role.name:
                    raise SystemExit("telemetry collector does not carry the workload_role tag "
                                     "(old scripts/collect_telemetry.py?) -- refusing to mix roles untagged")
                collectors.append((role.name, g, c))
        for _, _, c in collectors:
            c.start()
        event("telemetry_started", gpus=sorted(serve_gpus + train_gpus))
        time.sleep(args.baseline_s)

    ready_re = re.compile(args.serve_ready_regex) if args.serve_ready_regex else None
    serve, train = roles["serve"], roles["train"]
    rc = 0
    stop_all = threading.Event()

    def shutdown(signum=None, frame=None) -> None:
        stop_all.set()
    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    t_begin = time.time()
    deadline = t_begin + args.timeout if args.timeout else None
    try:
        serve.start(ready_re)
        event("serve_started")
        log.info("waiting for serve to become ready (%s) ...", args.serve_ready_regex or "no marker")
        wait_until = time.time() + args.serve_ready_timeout
        while not serve.ready.is_set():
            if serve.poll() is not None:
                raise RuntimeError(f"serve exited rc={serve.rc} before becoming ready:\n" + "\n".join(serve.tail))
            if stop_all.is_set() or time.time() > wait_until:
                raise RuntimeError("serve did not become ready in time" if not stop_all.is_set() else "interrupted")
            time.sleep(0.5)
        event("serve_ready")
        time.sleep(args.train_delay_s)
        train.start(None)
        event("train_started")

        while True:
            s_rc, t_rc = serve.poll(), train.poll()
            if s_rc is not None and t_rc is not None:
                break
            if s_rc not in (None, 0) and t_rc is None:      # serve died mid-run: invalid colocation
                log.error("serve exited rc=%s while train still running: stopping train", s_rc)
                train.stop()
                break
            if stop_all.is_set() or (deadline and time.time() > deadline):
                log.warning("stopping both roles (%s)", "interrupt" if stop_all.is_set() else "timeout")
                rc = 130 if stop_all.is_set() else 124
                break
            time.sleep(0.5)
    except RuntimeError as e:
        log.error("%s", e)
        rc = 1
    finally:
        for r in (train, serve):
            r.stop()
            r.poll()
        event("roles_finished")
        if collectors:
            time.sleep(args.baseline_s)
            for _, _, c in collectors:
                c.stop()

    rc = rc or serve.rc or train.rc or 0

    # -- outputs ---------------------------------------------------------------
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    files: list[dict] = []
    for role_name, g, c in collectors:
        gpu_short = str(getattr(c.metadata, "gpu_name", "gpu")).replace(" ", "_").replace("-", "_")
        path = out_dir / f"{args.label}_{role_name}_gpu{g}_{gpu_short}_{run_id[:12]}_{ts}.parquet"
        c.save(str(path))
        files.append({"role": role_name, "gpu_index": g, "path": str(path)})
    manifest = {
        "wid": "W25", "label": args.label, "run_id": run_id, "rc": rc,
        "t_run_start": t_begin, "t_run_end": time.time(),
        "roles": {name: {"role_label": r.label, "gpus": r.gpus, "cmd": r.cmd, "env": r.env,
                         "t_start": r.t_start, "t_ready": r.t_ready, "t_end": r.t_end, "returncode": r.rc}
                  for name, r in roles.items()},
        "overlap_window": ([train.t_start, min(x for x in (serve.t_end, train.t_end) if x)]
                           if train.t_start and (serve.t_end or train.t_end) else None),
        "events": events,
        "telemetry": None if args.no_telemetry else {"files": files, "interval_s": args.interval,
                                                      "fabric": (fabric_cache.summary() if fabric_cache else None)},
        "usable_as_w25_evidence": not args.no_telemetry and rc == 0,
        "notes": ["workload_role / role_label are stamped on every telemetry row",
                  "IB/eth counters are node-level (not attributable to a role); NVLink/NVML rows are per GPU",
                  "GPU indices are physical NVML indices; check they match CUDA_VISIBLE_DEVICES under your allocation"],
    }
    mpath = out_dir / f"{args.label}_{run_id[:12]}_roles.json"
    mpath.write_text(json.dumps(manifest, indent=2, default=str))
    log.info("manifest -> %s (rc=%s)", mpath, rc)
    return manifest, rc


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="W25 spatial colocation: serve on some GPUs, train on others",
                                 formatter_class=argparse.RawDescriptionHelpFormatter, epilog=__doc__)
    ap.add_argument("--label", default="colocate", help="output/run label (registry label when registered)")
    ap.add_argument("--serve-workload", help="registry label of the serve role (e.g. moe_serve_qwen7b_tp4dp2_mixed)")
    ap.add_argument("--serve-cmd", help="custom serve command (shell-quoted)")
    ap.add_argument("--train-workload", help="registry label of the train role")
    ap.add_argument("--train-cmd", help="custom train command (shell-quoted)")
    ap.add_argument("--num-gpus", type=int, default=8, help="node GPU count for the default partition")
    ap.add_argument("--serve-gpus", default=None, help="default: all but the last --train-count GPUs (e.g. 0-6)")
    ap.add_argument("--train-gpus", default=None, help="default: the last --train-count GPUs (e.g. 7)")
    ap.add_argument("--train-count", type=int, default=1)
    ap.add_argument("--duration", type=int, default=None, help="applied to both roles' --duration")
    ap.add_argument("--serve-args", action="append", default=[], metavar="'--tp 1 --dp 7'",
                    help="extra args appended to the serve command (shell-quoted string, repeatable; "
                         "later flags win). A single bare flag needs the = form: --serve-args=--enforce-eager")
    ap.add_argument("--train-args", action="append", default=[], metavar="'--comm-log x.jsonl'",
                    help="extra args appended to the train command (same rules)")
    ap.add_argument("--serve-ready-regex", default="Server healthy",
                    help="serve stdout marker that starts the train role ('' = start both immediately)")
    ap.add_argument("--serve-ready-timeout", type=int, default=3600)
    ap.add_argument("--train-delay-s", type=float, default=0.0)
    ap.add_argument("--timeout", type=int, default=None, help="kill both roles after N seconds")
    ap.add_argument("--output-dir", default="data/")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--baseline-s", type=float, default=2.0)
    ap.add_argument("--no-dcgm", action="store_true")
    ap.add_argument("--no-fabric", action="store_true")
    ap.add_argument("--no-telemetry", action="store_true",
                    help="plumbing check only; manifest marks the run unusable as W25 evidence")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")
    args = build_parser().parse_args()
    if args.serve_gpus is None or args.train_gpus is None:
        allg = list(range(args.num_gpus))
        args.train_gpus = args.train_gpus or ",".join(map(str, allg[-args.train_count:]))
        args.serve_gpus = args.serve_gpus or ",".join(map(str, allg[:-args.train_count]))
    _, rc = run(args)
    sys.exit(rc)


if __name__ == "__main__":
    main()
