#!/usr/bin/env python3
"""Per-window train/infer ground truth from the ``STEP`` launcher logs.

Workloads emit one stdout line per step::

    STEP ts=<epoch> kind=train|infer batch_idx=N [tokens=M] [passes=P] \
        [phase=rollout|update] [loss=...]

``ts`` is ``time.time()`` — the SAME epoch clock the 1 Hz telemetry collector
stamps into ``timestamp_epoch`` (renamed ``ts`` in ``threeway_improved``), and
``sliding_windows`` builds ``window_start``/``window_end`` from it. So a STEP
line bins directly into the window whose ``[window_start, window_end)`` contains
its ``ts``.

This is the reusable core behind ``scripts/label_windows_by_training.py``. Its
purpose is to give the classifier per-window ground truth so a diluted evasion
run (mostly inference, occasional training — e.g. PPO at ~12% duty cycle) is no
longer wholesale labeled "training": windows with zero real training steps are
correctly non-training, and the detector is credited for calling them so.

For PPO/decode (``adversarial_symgemm.py`` P1/DC) the ``kind`` field already
carries the right truth: a policy-update step is ``kind=train`` (``phase=update``)
and a rollout/decode step is ``kind=infer`` (``phase=rollout``), so no ``phase``
parsing is needed here.

Coverage note: only the runs whose launcher output was teed to ``logs/*.log``
have recoverable window labels (the recent evasion runs). Benign single-class
runs need none — their run-level label is correct for every window. Older
evasion runs collected without a STEP log keep their run-level label.
"""

from __future__ import annotations

import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

STEP_RE = re.compile(
    r"STEP\s+ts=(?P<ts>[\d.]+)\s+kind=(?P<kind>train|infer)\s+batch_idx=(?P<idx>\d+)")
RUNID_RE = re.compile(r"Run ID:\s*(?P<run_id>\S+)")
# Trailing key=value fields, scanned after the positional prefix. Promoted from
# rebuttal_analysis/symgemm_throughput.py, which was the ONLY tokens-aware STEP
# parser in the repo -- this module's parser matched the prefix and then threw
# `tokens=`/`passes=` away, so progress could never be joined to a window even
# though every workload emits it. Tolerating any order and any extra fields also
# means the superset dialect (node=/rank=/phase=/loss=/p_train=) parses here.
FIELD_RE = re.compile(r"(\w+)=([-\w.]+)")


def parse_step_log(path: str) -> tuple[str | None, pd.DataFrame]:
    """Return ``(run_id, DataFrame[ts, kind, tokens, passes])`` from one teed log.

    ``run_id`` is taken from the first ``Run ID:`` line (``None`` if absent, in
    which case the caller must supply an override).

    ``tokens`` is the workload's own count of USEFUL work in that step and
    ``passes`` what an observer who believed the disguise would attribute to it
    (see workloads/_step.py). Both default to 0/1 when a workload omits them, so
    older subset-dialect logs still parse. Keeping them is what lets progress be
    joined to a telemetry window, which is the missing half of any
    "detection rate vs throughput sacrificed" comparison.
    """
    run_id = None
    steps = []
    with open(path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:
            if run_id is None:
                mr = RUNID_RE.search(line)
                if mr:
                    run_id = mr.group("run_id")
            ms = STEP_RE.search(line)
            if ms:
                fields = dict(FIELD_RE.findall(line[ms.end():]))
                steps.append((
                    float(ms.group("ts")),
                    ms.group("kind"),
                    _as_int(fields.get("tokens"), 0),
                    _as_int(fields.get("passes"), 1),
                ))
    df = pd.DataFrame(steps, columns=["ts", "kind", "tokens", "passes"])
    return run_id, df


def _as_int(v, default):
    try:
        return int(float(v))
    except (TypeError, ValueError):
        return default


def match_run_id(log_run_id: str | None, override: str | None, known_run_ids) -> str | None:
    """Resolve a log's run to a run_id present in ``known_run_ids``.

    Tolerates the launcher's full run_id vs the truncated id embedded in a
    telemetry filename by matching on a shared prefix in either direction.
    """
    candidate = override or log_run_id
    if candidate is None:
        return None
    known = [str(r) for r in known_run_ids]
    if candidate in known:
        return candidate
    for rid in known:
        if rid.startswith(candidate) or candidate.startswith(rid):
            return rid
    return None


def build_step_log_map(log_globs, known_run_ids, overrides=None) -> dict:
    """Scan launcher logs and return ``{run_id: {train_ts, infer_ts, log}}``.

    ``run_id`` keys are resolved to values present in ``known_run_ids`` (the
    run_ids in the telemetry corpus), so the result can be looked up directly
    while windowing. Logs that match no known run, or carry no STEP lines, are
    skipped. ``overrides`` is an optional ``{log_path: run_id}`` map for logs
    whose ``Run ID:`` line is missing.
    """
    overrides = overrides or {}
    if isinstance(log_globs, str):
        log_globs = [log_globs]
    logs = []
    for pattern in log_globs:
        logs.extend(sorted(glob.glob(pattern)))

    out = {}
    for log_path in logs:
        log_run_id, steps = parse_step_log(log_path)
        rid = match_run_id(log_run_id, overrides.get(log_path), known_run_ids)
        if rid is None or steps.empty:
            continue
        out[rid] = {
            "train_ts": steps.loc[steps["kind"] == "train", "ts"].to_numpy(),
            "infer_ts": steps.loc[steps["kind"] == "infer", "ts"].to_numpy(),
            "log": Path(log_path).name,
        }
    return out


def window_activity(window_starts, window_ends, train_ts, infer_ts=None) -> dict:
    """Per-window step counts and a training-active flag.

    A window is ``train_active`` iff at least one ``kind=train`` step timestamp
    falls in ``[window_start, window_end)``. Returns arrays aligned to the input
    windows: ``n_train_steps``, ``n_infer_steps`` (if ``infer_ts`` given), and
    ``train_active`` (bool).
    """
    ws = np.asarray(window_starts, dtype=float)
    we = np.asarray(window_ends, dtype=float)
    train_ts = np.asarray(train_ts, dtype=float)
    n_train = np.array(
        [int(((train_ts >= s) & (train_ts < e)).sum()) for s, e in zip(ws, we)])
    result = {"n_train_steps": n_train, "train_active": n_train > 0}
    if infer_ts is not None:
        infer_ts = np.asarray(infer_ts, dtype=float)
        result["n_infer_steps"] = np.array(
            [int(((infer_ts >= s) & (infer_ts < e)).sum()) for s, e in zip(ws, we)])
    return result
