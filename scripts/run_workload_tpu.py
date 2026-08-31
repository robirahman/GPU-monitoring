#!/usr/bin/env python3
"""Per-host TPU workload launcher -- mirrors scripts/run_workload.py.

Starts the host's TPU telemetry collector for its local chips, launches the
workload subprocess, tees its STEP lines to logs/{label}_w{W}.log (with the
`Run ID:` line so window_labels can join telemetry to ground truth), saves one
parquet per chip, writes a GCS-mirrorable `.done` marker, and optionally copies
everything to gs://.../runs/<run_id>/.

The shared --run-id is REQUIRED on a multi-host slice: every worker of one job
must pass the SAME run_id so all 16 per-chip streams group as a single job.
The driver-side queue (scripts/tpu_collection/tpu_queue.sh) derives it once and
fans it out to all workers via `gcloud ... ssh --worker=all`.

Usage (registry label):
    python scripts/run_workload_tpu.py --workload tpuT16_train_fsdp7b \
        --run-id abc123def456 --worker-id 0

Usage (custom command):
    python scripts/run_workload_tpu.py --workload tpuT16_train_fsdp7b \
        --run-id abc123def456 --worker-id 0 -- \
        python workloads/tpu_fsdp_training.py --size 7b --duration 600
"""

import argparse
import logging
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _REPO_ROOT)

from scripts.collect_tpu_telemetry import TpuTelemetryCollector
from workloads.registry_tpu import get_tpu_workload

log = logging.getLogger("spar.tpu_launcher")


def _tee(proc, log_path, run_id):
    """Pump subprocess output to both the per-worker log and our stdout.

    The log begins with `Run ID: <id>` so window_labels.build_step_log_map
    associates this log's STEP lines with the job's telemetry run_id.
    """
    with open(log_path, "w") as lf:
        lf.write(f"Run ID: {run_id}\n")
        lf.flush()
        for raw in iter(proc.stdout.readline, b""):
            line = raw.decode("utf-8", "replace")
            lf.write(line)
            lf.flush()
            sys.stdout.write(line)
            sys.stdout.flush()


def _gcs_sync(gcs_prefix, run_id, worker_id, paths, log_path, done_path):
    """Copy this worker's outputs to the per-run GCS prefix, LOUDLY.

    Was "best effort" via `gsutil -q cp` with check=False and only a debug-level
    warning on exception -- so an upload that never happened looked identical to
    one that succeeded. Consequences observed: one session uploaded only worker
    0's parquets (4 of 16 chips) and the next uploaded nothing at all, while both
    logged "Saved chip 0..3" and exited 0.

    Two changes: prefer `gcloud storage cp` (gsutil is deprecated and is not
    present on every TPU runtime image, which is the likely cause), and report a
    per-worker tally on stdout so the caller can see it and grep for it.
    """
    base = f"{gcs_prefix.rstrip('/')}/runs/{run_id}/w{worker_id}"
    tool = None
    for cand in (["gcloud", "storage", "cp"], ["gsutil", "-q", "cp"]):
        if shutil.which(cand[0]):
            tool = cand
            break
    if tool is None:
        print(f"GCS_SYNC w{worker_id}: FAILED - neither gcloud nor gsutil on PATH",
              flush=True)
        return 0
    wanted = [x for x in list(paths) + [log_path, done_path]
              if x and os.path.exists(x)]
    ok = 0
    for local in wanted:
        try:
            r = subprocess.run(tool + [str(local), f"{base}/"],
                               capture_output=True, text=True, timeout=300)
            if r.returncode == 0:
                ok += 1
            else:
                print(f"GCS_SYNC w{worker_id}: {os.path.basename(local)} rc="
                      f"{r.returncode} {(r.stderr or '').strip()[:200]}", flush=True)
        except Exception as e:
            print(f"GCS_SYNC w{worker_id}: {os.path.basename(local)} "
                  f"{type(e).__name__}: {e}", flush=True)
    print(f"GCS_SYNC w{worker_id}: uploaded {ok}/{len(wanted)} -> {base} "
          f"(tool={tool[0]})", flush=True)
    return ok


def main():
    ap = argparse.ArgumentParser(description="Per-host TPU workload launcher")
    ap.add_argument("--workload", type=str, required=True,
                    help="TPU registry label (also the parquet/label prefix)")
    ap.add_argument("--run-id", type=str, default=None,
                    help="Shared run_id for the job (REQUIRED multi-host). "
                         "Pass the SAME value on every worker.")
    # GLOBALLY unique worker id. TPU_WORKER_ID is per-SLICE, so on a
    # --node-count multislice group slice-0-worker-0 and slice-1-worker-0 are both
    # "0": they would write the same <cell>.w0.done marker and the queue's
    # `got >= NUM_WORKERS` check would undercount, marking a healthy cell FAILED.
    # Fold the slice index in. `slice*1000 + worker` keeps it readable (s0w0->0,
    # s1w0->1000) and still matches the _w\d+c\d+ rank-token regex that
    # multinode_features strips.
    # MEASURED on a live 2x v6e-1 group: MEGASCALE_SLICE_ID, MEGASCALE_NUM_SLICES
    # and MEGASCALE_COORDINATOR_ADDRESS are ALL UNSET, yet jax.distributed still
    # forms the cross-slice mesh (device_count 2, process 0/1 of 2). So slice
    # identity cannot be read from the megascale env, and the previous default
    # would have resolved BOTH slices to 0 -- reintroducing the very marker
    # collision it was meant to prevent.
    #
    # jax.process_index() is the correct answer but is only available inside the
    # workload after initialize(), not here in the launcher. So the ORCHESTRATION
    # must say which slice this is: SPAR_TPU_SLICE_ID, set per-slice by whatever
    # fans the command out. Falls back to the megascale vars if a future runtime
    # does populate them, then to 0.
    _slice = int(os.environ.get("SPAR_TPU_SLICE_ID",
                 os.environ.get("MEGASCALE_SLICE_ID",
                 os.environ.get("MEGASCALE_SLICE_INDEX", "0"))) or 0)
    _wid = int(os.environ.get("TPU_WORKER_ID", 0) or 0)
    # Loudly report the resolved id: a silent collision is exactly the failure
    # mode this guards against, and on a multislice run two identical ids means
    # one slice's parquet and .done marker overwrite the other's.
    print(f"WORKER_ID_RESOLVED: slice={_slice} tpu_worker={_wid} "
          f"-> worker_id={_slice * 1000 + _wid} "
          f"(SPAR_TPU_SLICE_ID={os.environ.get('SPAR_TPU_SLICE_ID', 'unset')})",
          flush=True)
    ap.add_argument("--worker-id", type=int, default=_slice * 1000 + _wid,
                    help="Globally unique worker index (slice*1000 + "
                         "TPU_WORKER_ID; override only if you know why).")
    ap.add_argument("--output-dir", type=str, default="data")
    ap.add_argument("--log-dir", type=str, default="logs")
    ap.add_argument("--interval", type=float, default=1.0)
    ap.add_argument("--checkpoint-sec", type=float, default=60.0)
    ap.add_argument("--timeout", type=int, default=None)
    ap.add_argument("--gcs-prefix", type=str, default=None,
                    help="e.g. gs://bucket/session -> copies to <prefix>/runs/<run_id>/w<W>/")
    ap.add_argument("--multihost-required", action="store_true",
                    help="Fail if --run-id is not supplied (queue sets this).")
    ap.add_argument("--smoke", action="store_true",
                    help="Shakedown: prefix label smoke_, force --duration 60, "
                         "and write to logs/smoke_data/ (never data/).")
    args, extra = ap.parse_known_args()
    if extra and extra[0] == "--":
        extra = extra[1:]

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s")

    if args.multihost_required and not args.run_id:
        ap.error("--run-id is required in multi-host mode")

    # Resolve command: custom (after --) or registry lookup.
    if extra:
        workload_cmd = extra
        timeout = args.timeout
    else:
        # The queue appends _rep{N} to make each rep a distinct run/label, but
        # the registry key is the base cell -- strip the rep suffix for lookup
        # while keeping the full label for output naming (run_id already
        # separates reps, and threeway_label tolerates the suffix).
        import re as _re
        base = _re.sub(r"_rep\d+$", "", args.workload)
        try:
            wl = get_tpu_workload(base)
        except ValueError as e:
            print(f"ERROR: {e}", file=sys.stderr)
            sys.exit(1)
        workload_cmd = list(wl["command"])
        timeout = args.timeout or wl.get("default_timeout")

    # Enable the libtpu metrics gRPC server BEFORE the workload starts. Without
    # this the server never binds, tpu_info's accessors get CONNECTION REFUSED on
    # localhost:8431, and the collector records zeros for every accelerator
    # metric -- which is exactly what happened to the first real TPU collection
    # (245 rows/chip, all 0 or -1, tpu_metrics_live=0). tpu_info's own error
    # message names this variable as the fix. One port per local chip.
    if not os.environ.get("TPU_RUNTIME_METRICS_PORTS"):
        _base = int(os.environ.get("SPAR_TPU_METRICS_BASE_PORT", "8431"))
        _n = int(os.environ.get("SPAR_TPU_CHIPS_PER_HOST", "4"))
        os.environ["TPU_RUNTIME_METRICS_PORTS"] = ",".join(
            str(_base + i) for i in range(_n))
        print(f"TPU_RUNTIME_METRICS_PORTS={os.environ['TPU_RUNTIME_METRICS_PORTS']}",
              flush=True)

    run_id = args.run_id or uuid.uuid4().hex[:12]
    worker_id = args.worker_id
    label = args.workload
    out_subdir = args.output_dir
    if args.smoke:
        # Shakedown, not data: 60 s, smoke_ label, and OUT OF data/ so
        # load_all_data() never rglobs it (matches the Round 6 convention).
        label = "smoke_" + label
        workload_cmd = workload_cmd + ["--duration", "60"]
        out_subdir = os.path.join("logs", "smoke_data")
        timeout = min(timeout or 240, 240)
        args.output_dir = out_subdir
    log_dir = os.path.join(_REPO_ROOT, args.log_dir)
    done_dir = os.path.join(log_dir, ".done")
    Path(log_dir).mkdir(parents=True, exist_ok=True)
    Path(done_dir).mkdir(parents=True, exist_ok=True)
    output_dir = os.path.join(_REPO_ROOT, args.output_dir)
    log_path = os.path.join(log_dir, f"{label}_w{worker_id}.log")

    # Emitted on stdout so an aggregating log carries the join prefix.
    print(f"Run ID: {run_id}", flush=True)

    collector = TpuTelemetryCollector(
        worker_id=worker_id, interval_sec=args.interval,
        workload_label=label, run_id=run_id, output_dir=output_dir,
        checkpoint_sec=args.checkpoint_sec)

    print("=" * 60)
    print("SPAR TPU Workload Run")
    print("=" * 60)
    print(f"  Workload:  {label}")
    print(f"  Command:   {' '.join(workload_cmd)}")
    print(f"  Worker:    {worker_id}   chips={collector.chip_indices}")
    print(f"  Run ID:    {run_id}")
    print(f"  Sources:   {[k for k,v in collector.source.availability.items() if v]}")
    print(f"  Log:       {log_path}")
    print(f"  Timeout:   {timeout or 'none'}s")
    print("=" * 60)

    collector.start()
    time.sleep(2)  # baseline

    env = os.environ.copy()
    env["PYTHONPATH"] = _REPO_ROOT + os.pathsep + env.get("PYTHONPATH", "")

    workload_rc = 0
    proc = None
    try:
        proc = subprocess.Popen(workload_cmd, cwd=_REPO_ROOT, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        tee_thread = threading.Thread(target=_tee, args=(proc, log_path, run_id),
                                      daemon=True)
        tee_thread.start()
        try:
            workload_rc = proc.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            print(f"\nWARNING: workload timed out after {timeout}s; killing.")
            proc.kill()
            proc.wait()
            workload_rc = 124
        tee_thread.join(timeout=5)
    except KeyboardInterrupt:
        print("\nInterrupted; saving what we have.")
        if proc and proc.poll() is None:
            proc.terminate()
            try:
                proc.wait(timeout=5)
            except subprocess.TimeoutExpired:
                proc.kill()
        workload_rc = -2

    time.sleep(2)  # cooldown
    collector.stop()
    paths = collector.save()

    # Resumable per-worker done marker (mirrored to GCS by the queue).
    done_path = os.path.join(done_dir, f"{label}.w{worker_id}.done")
    stamp = datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ")
    with open(done_path, "w") as f:
        f.write(f"{stamp} rc={workload_rc} run_id={run_id} "
                f"chips={len(paths)} worker={worker_id}\n")

    print("\n" + "=" * 60)
    print(f"  Workload exit code: {workload_rc}")
    print(f"  Parquets: {len(paths)} (one per chip)")
    print(f"  Done marker: {done_path}")
    print("=" * 60)

    # Sources are printed at STARTUP too, but at that moment the workload has not
    # launched, so the libtpu metrics server is not listening and grpc_metrics is
    # always absent. Print the set again HERE: this is the only place that shows
    # whether the lazy reconnect actually landed during the run.
    try:
        av = getattr(collector.source, "availability", {})
        print(f"SOURCES_FINAL: {sorted(k for k, v in av.items() if v)}", flush=True)
        srv = getattr(collector.source, "_served", None)
        if srv is not None:
            print(f"GRPC_SERVED: {sorted(srv.values())}", flush=True)
        print(f"GRPC_TRIES: {getattr(collector.source, '_grpc_tries', 'n/a')}",
              flush=True)
    except Exception as e:
        print(f"SOURCES_FINAL: unavailable ({e})", flush=True)

    if args.gcs_prefix:
        _gcs_sync(args.gcs_prefix, run_id, worker_id, paths, log_path, done_path)

    # Propagate the workload rc so the queue records failures honestly.
    sys.exit(workload_rc if workload_rc is not None else 0)


if __name__ == "__main__":
    main()
