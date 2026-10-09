"""Temporal alerts per telemetry stream, aggregated to physical jobs."""
import numpy as np


def run_alert_rate(windows, scores, tau, k, m, by="run_id"):
    """Return (alert fraction, count), with jobs alerting if any stream alerts.

    Evaluate k-of-m in time order WITHIN each stream. Concatenating ranks of
    one job can otherwise turn simultaneous positives on several GPUs into
    apparently consecutive observations, or create an alert across a boundary
    between the end of one stream and the start of another.
    """
    if not 1 <= k <= m:
        raise ValueError("require 1 <= k <= m")
    if by not in ("run_id", "job_id"):
        raise ValueError("by must be run_id or job_id")
    scores = np.asarray(scores)
    if scores.ndim != 1 or len(scores) != len(windows):
        raise ValueError("scores must have one value per window")
    if windows.empty:
        return float("nan"), 0
    keys = ["run_id"] if by == "run_id" else ["job_id", "run_id"]
    data = windows[list(dict.fromkeys(keys + ["window_start"]))].copy()
    data["_score"] = scores
    alerts = {}
    for key, stream in data.groupby(keys, sort=False):
        values = stream.sort_values("window_start", kind="stable")._score.to_numpy() > tau
        counts = np.concatenate(([0], np.cumsum(values)))
        width = min(m, len(values))
        alert = bool(np.any(counts[width:] - counts[:-width] >= k))
        identity = key[0] if isinstance(key, tuple) else key
        alerts[identity] = alerts.get(identity, False) or alert
    return (float(np.mean(list(alerts.values()))) if alerts else float("nan"),
            len(alerts))
