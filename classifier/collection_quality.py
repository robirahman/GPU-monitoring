"""Collection validity is separate from workload objective and utilization."""
import re


def assess_collection(label, median_util, log_text="", threshold=1.0):
    """Return disposition and evidence; low duty cycle alone never proves failure."""
    exits = [int(x) for x in re.findall(r"Multi-GPU run complete:.*?\(rc=(-?\d+)\)", log_text)]
    failed = any(rc != 0 for rc in exits) or bool(re.search(r"Workload finished: FAIL\(", log_text))
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
