"""Collection validity is separate from workload objective and utilization."""
import re

# Every run_workload_multi_gpu.py mode names each parquet it saves: multi-GPU and
# telemetry-only log "GPU i: saved <path>", --parallel logs each replica as
# "GPU i: <workload> <status> -> <file>". That filename is the only link a
# --parallel replica has to its stage log: its recorded run_id is a random
# per-collector id that appears in neither the log name nor the log text.
SAVED_FILE = re.compile(r"[\w.+-]+\.parquet")
REPLICA = re.compile(r"GPU \d+: \S+ (.+?) -> ([\w.+-]+\.parquet)")
# A telemetry-only collector exits 0 whether or not the remote workload ran, so
# its "Telemetry-only run complete" line is deliberately not completion evidence.
# Callers pass the job head's launcher log for those streams instead.
TELEMETRY_ONLY = "Telemetry-only:"
# run_stage names every launcher log <stage>_node<N>[_<rid>].log. Nothing else on
# the host speaks for one file: the queue master log names files too, and it
# carries other cells' failure tails; .nic/.ray logs are sidecar output.
STAGE_LOG = re.compile(r"_node\d+(?:_[0-9a-f]{12})?\.log$")
WORKLOAD_EXIT = re.compile(r"Multi-GPU run complete:.*?\(rc=(-?\d+)\)")
WORKLOAD_FAILED = re.compile(r"Workload finished: (?:FAIL\(|TIMEOUT)")


def assess_collection(label, median_util, log_text="", threshold=1.0, filename=None):
    """Return disposition and evidence; low duty cycle alone never proves failure.

    ``log_text`` is launcher evidence for the workload that produced the file.
    With ``filename``, a --parallel replica is judged by its own status line, not
    by its siblings: one failed replica does not invalidate the others.
    """
    replica = [status for status, name in REPLICA.findall(log_text) if name == filename]
    if replica:
        if any(status != "OK" for status in replica):
            return "failed", "nonzero_replica_exit"
        return "accepted", "replica_completed"
    exits = [int(x) for x in WORKLOAD_EXIT.findall(log_text)]
    failed = any(rc != 0 for rc in exits) or bool(WORKLOAD_FAILED.search(log_text))
    completed = bool(exits) and all(rc == 0 for rc in exits)
    progress = bool(re.search(r"STEP\s+ts=[\d.]+\s+kind=(?:train|infer)\b", log_text))
    control = "stalled" in label.lower() or "idle" in label.lower()
    if failed:
        return "failed", "nonzero_workload_exit"
    if completed and (progress or control):
        return "accepted", "completed_with_progress_or_control"
    if completed:
        # Actor ranks and non-root workers may never emit local STEP lines.
        # Completion is not proof of an optimizer update, but low activity on
        # such a rank is not grounds for discarding a successful collection.
        return "accepted", "completed_no_local_progress"
    if median_util is not None and median_util <= threshold and not control and threshold > 0:
        return "uncertain", "low_activity_without_completion_and_progress"
    return "accepted", "completion_unverified"


def launcher_logs(session_root):
    """Index staged launcher logs once: ``<session>/<host>/logs/*.log``."""
    logs = []
    for path in sorted(session_root.glob("*/logs/*.log")):
        if not STAGE_LOG.search(path.name):
            continue
        text = path.read_text(errors="replace")
        logs.append(dict(path=path, host=path.parent.parent, text=text,
                         files=set(SAVED_FILE.findall(text)),
                         telemetry_only=TELEMETRY_ONLY in text))
    return logs


def evidence_logs(logs, host, filename, run_ids):
    """Launcher logs that speak for one staged parquet.

    The file's own launcher log is found on its host by the saved filename, or by
    a run_id in the log name. A telemetry-only stream is vouched for by the job's
    head instead: the launcher log for the same run_id on another host.
    """
    run_ids = [r for r in run_ids if len(r) >= 8]

    def named(log):
        return any(r in log["path"].name for r in run_ids)

    own = [log for log in logs if log["host"] == host and (filename in log["files"] or named(log))]
    if own and all(log["telemetry_only"] for log in own):
        return [log for log in logs if log["host"] != host and not log["telemetry_only"] and named(log)]
    return own
