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
have recoverable window labels. Without a STEP log, single-kind programs keep
an explicitly marked program label, including startup/load windows; mixed
programs remain unresolved. They are never relabeled by their objective alone.
"""

from __future__ import annotations

import glob
import gzip
import re
from functools import lru_cache
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
                    # node/rank are the superset dialect's attribution fields.
                    # They were parsed and thrown away, which was harmless only
                    # while every multi-node workload emitted from rank 0 alone.
                    # -1 marks "this log did not say", which is not rank 0.
                    _as_int(fields.get("node"), -1),
                    _as_int(fields.get("rank"), -1),
                ))
    df = pd.DataFrame(steps, columns=["ts", "kind", "tokens", "passes",
                                      "node", "rank"])
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

    # ACCUMULATE, never replace. `rid_for()` in multinode_queue.sh is
    # deterministic and IDENTICAL on both nodes, and both nodes' logs are
    # gathered into logs/ and globbed together here. The previous version did
    # `out[rid] = {...}`, so the second log seen for a run silently discarded
    # the first. That was harmless only because every multi-node workload so far
    # emits STEP lines from rank 0 alone, leaving node1's log empty and dropped
    # by the `steps.empty` guard above.
    #
    # It stops being harmless the moment a workload emits from every rank --
    # which distributed RL must, because its ranks genuinely diverge (an actor
    # is running inference while the learner trains). There, node1's log would
    # have overwritten node0's and every label for the job would have been
    # computed from ranks 8-15 alone. Nothing would have failed; the numbers
    # would just have been wrong for exactly the cells whose per-rank
    # attribution is the entire point.
    acc = {}
    for log_path in logs:
        log_run_id, steps = parse_step_log(log_path)
        rid = match_run_id(log_run_id, overrides.get(log_path), known_run_ids)
        if rid is None or steps.empty:
            continue
        acc.setdefault(rid, []).append((Path(log_path).name, steps))

    out = {}
    for rid, parts in acc.items():
        allsteps = pd.concat([s for _, s in parts], ignore_index=True)
        # by_rank keeps the attribution the flat arrays cannot express: with an
        # actor and a learner in the same run at the same second, "was this
        # window training?" has no answer that is right for both streams.
        by_rank = {}
        if (allsteps["rank"] >= 0).any():
            for (nd, rk), g in allsteps[allsteps["rank"] >= 0].groupby(
                    ["node", "rank"], sort=True):
                by_rank[(int(nd), int(rk))] = {
                    "train_ts": np.sort(
                        g.loc[g["kind"] == "train", "ts"].to_numpy()),
                    "infer_ts": np.sort(
                        g.loc[g["kind"] == "infer", "ts"].to_numpy()),
                }
        out[rid] = {
            "train_ts": np.sort(
                allsteps.loc[allsteps["kind"] == "train", "ts"].to_numpy()),
            "infer_ts": np.sort(
                allsteps.loc[allsteps["kind"] == "infer", "ts"].to_numpy()),
            "log": ",".join(sorted(n for n, _ in parts)),
            "by_rank": by_rank,
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


# ── Per-window work labels ───────────────────────────────────────────────────
#
# A window is labeled by the work its job actually did during it, read from the
# launcher logs, never by the job's name: a gradient step makes it training,
# otherwise a forward pass makes it inference, otherwise (filler, non-ML
# compute, a stall, or nothing logged) it is other. Every emitter writes one
# STEP line per slot of work, so a line reports the work done since the previous
# line from the same emitter in the same log. Using that interval rather than
# the instant keeps a long or accumulated step training for its whole duration
# (a 10-minute parameter sync over a shaped link included), while a filler slot
# between two training steps stays filler.

WORK_OF_KIND = {"train": "ml_training", "infer": "ml_inference", "unknown": "unresolved"}
STEP_ANY_RE = re.compile(r"STEP\s+ts=(?P<ts>[\d.]+)\s+kind=(?P<kind>\w+)(?P<rest>[^\n]*)")
NODE_RE = re.compile(r"\bnode=(-?\d+)")
RANK_RE = re.compile(r"\brank=(-?\d+)")
# Programs that logged their non-ML filler slots as kind=infer, recognised from
# their own output: GPU MN-I (cuFFT/hash filler; the TPU port writes
# kind=filler) and the single-GPU interleaving strategies F and I (mining,
# cuFFT and sleep slots).
FILLER_LOGGED_AS_INFER = re.compile(r"--strategy MN-I\b|^Strategy [FI]: \w+ Interleaving", re.M)
# logs/<SESSION>__<host>__<stage>.log[.gz]; the host is where the log was written.
LOG_NAME_RE = re.compile(r"^[^_]+__(?P<host>.+?)__")
# The launcher's node rank: its header, else run_stage's <stage>_node<N>_<rid>.log.
LOG_NODE_RE = re.compile(r"(?:Multi-GPU run|Telemetry-only): .*?\bnode=(\d+)")
LOG_NAME_NODE_RE = re.compile(r"_node(\d+)_[0-9a-f]{8,}\.log")
LOG_NAME_RUN_RE = re.compile(r"_node\d+_(?P<run_id>[0-9a-f]{8,})\.log(?:\.gz)?$")
STREAM_NODE_RE = r"_node(\d+)_gpu\d+"
STREAM_RANK_RE = re.compile(r"_node(\d+)_gpu(\d+)")
HEX_ID_RE = re.compile(r"[0-9a-f]{8,}")
WORK_LABELS = ("ml_training", "ml_inference", "other")


def read_log_text(path) -> str:
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        return f.read()


def parse_work_log(path) -> dict:
    """One launcher log: run_id, host, node rank, and events[ts, work, node, rank]."""
    text = read_log_text(path)
    name = Path(path).name
    run_id = RUNID_RE.search(text)
    # Some launcher captures start at the workload subprocess, after the run
    # header was printed. Their canonical stage/node/run filename still names
    # the execution. Keep source bytes intact and use only this anchored form.
    filename_run = LOG_NAME_RUN_RE.search(name)
    if run_id and filename_run and run_id["run_id"] != filename_run["run_id"]:
        raise ValueError(f"Conflicting header/filename run IDs: {path}")
    run_source = "header" if run_id else ("filename" if filename_run else None)
    run_id = run_id or filename_run
    host = LOG_NAME_RE.match(name)
    gcs_host = re.search(r"/runs/[^/]+/([^/]+)/logs/", str(path).replace("\\", "/"))
    node = LOG_NODE_RE.search(text) or LOG_NAME_NODE_RE.search(name)
    filler_as_infer = bool(FILLER_LOGGED_AS_INFER.search(text))
    rows, pending = [], {}
    for m in STEP_ANY_RE.finditer(text):
        try:
            ts = float(m["ts"])
        except ValueError:
            continue              # two ranks' output interleaved mid-number
        kind = "filler" if filler_as_infer and m["kind"] == "infer" else m["kind"]
        step_node, rank = NODE_RE.search(m["rest"]), RANK_RE.search(m["rest"])
        fields = dict(FIELD_RE.findall(m["rest"]))
        nd, rk = int(step_node[1]) if step_node else -1, int(rank[1]) if rank else -1
        key = (nd, rk, fields.get("event_id"))
        if kind == "pending":
            if key[2] is None:
                raise ValueError(f"Pending STEP lacks event_id: {path}")
            pending[key] = ts
            continue
        explicit_start = float(fields["start_ts"]) if "start_ts" in fields else np.nan
        if "start_ts" in fields:
            if not np.isfinite(explicit_start) or explicit_start > ts:
                raise ValueError(f"Invalid STEP start/end: {path}")
            pending.pop(key, None)
        rows.append((ts, WORK_OF_KIND.get(kind, "other"),
                     nd, rk, explicit_start))
    # A killed process cannot print a completion. Its outstanding requests stay
    # unknown through the telemetry tail, rather than becoming idle negatives.
    rows.extend((np.inf, "unresolved", nd, rk, start)
                for (nd, rk, _), start in pending.items())
    return dict(run_id=run_id["run_id"] if run_id else None, run_id_source=run_source, log=str(path),
                host=host["host"] if host else (gcs_host[1] if gcs_host else None),
                node=int(node[1]) if node else None,
                events=pd.DataFrame(rows, columns=["ts", "work", "node", "rank", "start_ts"]))


def default_log_paths(root) -> list:
    logs = Path(root) / "logs"
    if not logs.is_dir():
        return []
    return sorted(p for pattern in ("*.log", "*.log.gz") for p in logs.rglob(pattern)
                  if ".nic." not in p.name and ".ray." not in p.name)


def build_work_index(paths) -> dict:
    """``{run_id: [parsed log, ...]}``, accumulated over every node's log."""
    index = {}
    for path in paths:
        parsed = parse_work_log(path)
        run_id = parsed["run_id"]
        if run_id and HEX_ID_RE.fullmatch(run_id) and not parsed["events"].empty:
            index.setdefault(run_id, []).append(parsed)
    return index


def work_intervals(events) -> pd.DataFrame:
    """Slots of work ``(start, end, work)``.

    One timeline per emitter per log: separate logs are separate executions (a
    smoke run can share its cell's run_id), so no slot ever spans two of them.
    The first line of a timeline reports one typical slot; the time before it
    (launch, model load) is not work.
    """
    # Explicit request/generation bounds support overlapping asynchronous
    # requests and exclude idle gaps. Keep legacy adjacent STEP-slot semantics.
    explicit = events["start_ts"].notna() if "start_ts" in events else pd.Series(False, index=events.index)
    bounded = events.loc[explicit].rename(columns={"start_ts": "start", "ts": "end"})
    events = events.loc[~explicit]
    keys = [c for c in ("log", "node", "rank") if c in events]
    parts = []
    if len(bounded):
        parts.append(bounded[["start", "end", "work", "node", "rank"]].copy())
    for _, s in (events.groupby(keys, dropna=False, sort=False) if keys else [(None, events)]):
        if s.empty:
            continue
        order = np.argsort(s["ts"].to_numpy(float), kind="stable")
        ts, work = s["ts"].to_numpy(float)[order], s["work"].to_numpy()[order]
        gaps = np.diff(ts)
        typical = float(np.median(gaps[gaps > 0])) if (gaps > 0).any() else 1.0
        start = np.concatenate([[ts[0] - typical], ts[:-1]])
        parts.append(pd.DataFrame({"start": start, "end": ts, "work": work,
                                   "node": int(s["node"].iloc[0]) if "node" in s else -1,
                                   "rank": int(s["rank"].iloc[0]) if "rank" in s else -1}))
    if not parts:
        return pd.DataFrame(columns=["start", "end", "work", "node", "rank"])
    return pd.concat(parts, ignore_index=True)


def own_intervals(intervals, source_path):
    """A GPU stream's own rank's slots, when its job logs work rank by rank.

    Distributed RL logs every rank (the learner trains while actors serve
    rollouts), so a GPU's windows should show that GPU's work. Only a node whose
    k logging ranks are exactly node*k .. node*k+k-1 maps GPU g to rank node*k+g;
    with any rank missing the mapping is unknown. None then, and for rank-0-only
    jobs: the job's timeline applies.
    """
    m = STREAM_RANK_RE.search(str(source_path))
    if m is None or intervals is None or "rank" not in intervals:
        return None
    node, gpu = int(m[1]), int(m[2])
    ranks = sorted(int(r) for r in intervals.loc[intervals["node"] == node, "rank"].unique() if r >= 0)
    k = len(ranks)
    if k < 2 or gpu >= k or ranks != list(range(node * k, node * k + k)):
        return None
    own = intervals[(intervals["node"] == node) & (intervals["rank"] == node * k + gpu)]
    return own.reset_index(drop=True)


def job_work_intervals(frame, index) -> dict:
    """``{job_id: intervals}`` for every job whose run_ids have STEP lines.

    An s1/s8 cell runs an independent execution on each node under one run_id,
    so only the job's own node's log counts (node rank from the telemetry file
    name, else the host); a multi-node job takes every node's lines, because
    rank 0 alone often speaks for all of them.
    """
    from classifier.multinode_features import _scope_of
    cols = [c for c in ("job_id", "run_id", "hostname", "workload_label", "source_path") if c in frame]
    ids = frame[cols].drop_duplicates()
    bounds = frame.groupby("job_id")["ts"].agg(["min", "max"]) if "ts" in frame else None
    out = {}
    for job, g in ids.groupby("job_id", sort=False, observed=True):
        rids = [r for r in g["run_id"].astype(str).unique() if HEX_ID_RE.fullmatch(r)]
        if not rids:
            continue
        per_node = _scope_of(g["workload_label"].iloc[0]) in ("s1", "s8")
        nodes = (set(g["source_path"].astype(str).str.extract(STREAM_NODE_RE)[0].dropna().astype(int))
                 if "source_path" in g else set())
        hosts = set(g["hostname"].dropna().astype(str)) if "hostname" in g else set()

        def own(log):
            if not per_node:
                return True
            if nodes and log["node"] is not None:
                return log["node"] in nodes
            return log["host"] in hosts

        parts = [log["events"].assign(log=log["log"])
                 for key, logs in index.items()
                 if any(key.startswith(r) or r.startswith(key) for r in rids)
                 for log in logs if own(log)]
        if parts:
            intervals = work_intervals(pd.concat(parts, ignore_index=True))
            if bounds is not None:
                lo, hi = bounds.loc[job, ["min", "max"]]
                intervals = intervals[(intervals.end >= lo) & (intervals.start <= hi)]
            # A reused run ID from another collection is not evidence for this
            # recording. Preserve the unresolved/program fallback in that case.
            if len(intervals):
                out[str(job)] = intervals.reset_index(drop=True)
    return out


AUTO = "auto"


@lru_cache(maxsize=4)
def _work_index(root_str: str) -> dict:
    """Parse every committed launcher log once per process.

    ~1,800 logs, a few seconds. Every windowing call would otherwise redo it.
    """
    return build_work_index(default_log_paths(Path(root_str)))


def work_events_for(frame, root=None) -> dict:
    """``{key: intervals}`` for `frame`, keyed by job_id and, where unambiguous, run_id.

    ``sliding_windows`` looks the events up by job_id when the frame carries
    one and by run_id when it does not, so both keys are provided. Identity is
    assigned here when the frame has not been through
    ``assign_stream_and_job_ids`` -- which is the usual case at the ``eval_*``
    entry points, since they window a raw load.

    A run_id that covers more than one job is left unaliased: guessing there
    would attach one execution's step timeline to another's windows.
    """
    from classifier.data_sources import ROOT
    from classifier.multinode_features import assign_stream_and_job_ids
    index = _work_index(str(root if root is not None else ROOT))
    if not index:
        return {}
    ids = frame if "job_id" in frame.columns else assign_stream_and_job_ids(frame)
    events = job_work_intervals(ids, index)
    if not events or "run_id" not in ids.columns:
        return events
    jobs_per_run = ids.groupby(ids.run_id.astype(str)).job_id.nunique()
    aliased = dict(events)
    pairs = ids[["run_id", "job_id"]].astype(str).drop_duplicates()
    for run_id, job_id in pairs.itertuples(index=False):
        if jobs_per_run.get(run_id, 2) == 1 and job_id in events:
            aliased.setdefault(run_id, events[job_id])
    return aliased


def resolve_work_events(work_events, frame):
    """Normalise the ``work_events`` argument shared by both windowers."""
    if isinstance(work_events, str):
        if work_events != AUTO:
            raise ValueError(f"work_events must be a mapping, None, or {AUTO!r}")
        return work_events_for(frame)
    return work_events


def _covered(starts, ends, x):
    """Total length of disjoint sorted intervals lying left of each x."""
    k = np.searchsorted(starts, x, side="right")
    cum = np.concatenate([[0.0], np.cumsum(ends - starts)])
    last_end = np.where(k > 0, ends[np.maximum(k - 1, 0)], -np.inf)
    return cum[k] - np.where(k > 0, np.maximum(0.0, last_end - x), 0.0)


def _union(s, e):
    """Merge overlapping intervals (ranks working at the same time)."""
    if not len(s):
        return s, e
    order = np.argsort(s, kind="stable")
    s, e = s[order], e[order]
    reach = np.maximum.accumulate(e)
    first = np.concatenate([[True], s[1:] > reach[:-1]])
    return s[first], np.maximum.reduceat(e, np.flatnonzero(first))


def label_windows(window_starts, window_ends, intervals) -> dict:
    """Seconds of each kind of work inside each window, and the window's label.

    Any training makes a window training: the question a window answers is
    whether weights were updated while it was recorded. Otherwise any inference
    makes it inference, and a window with neither is other.
    """
    ws = np.asarray(window_starts, float)
    we = np.asarray(window_ends, float)
    if not len(ws):
        return dict(label=np.array([], dtype=object), train_s=ws.copy(),
                    infer_s=ws.copy(), other_s=ws.copy(), unknown_s=ws.copy())
    seconds = {}
    for work in (*WORK_LABELS, "unresolved"):
        iv = intervals[intervals["work"] == work]
        # Pending intervals end at infinity; clip before prefix sums to avoid
        # inf-inf while preserving uncertainty at every subsequent window.
        s = np.maximum(iv["start"].to_numpy(float), ws.min())
        e = np.minimum(iv["end"].to_numpy(float), we.max())
        s, e = _union(s[e > s], e[e > s])
        seconds[work] = (_covered(s, e, we) - _covered(s, e, ws)) if len(s) else np.zeros(len(ws))
    label = np.where(seconds["ml_training"] > 0, "ml_training",
                     np.where(seconds["unresolved"] > 0, "unresolved",
                              np.where(seconds["ml_inference"] > 0, "ml_inference", "other")))
    return dict(label=label, train_s=seconds["ml_training"],
                infer_s=seconds["ml_inference"], other_s=seconds["other"],
                unknown_s=seconds["unresolved"])


def assign_work_labels(rows, intervals, workload_label, objective):
    """Relabel window rows in place by the work done inside each window.

    With STEP evidence the label is observed work. Without it, a program that
    only ever does one kind of work keeps that kind; a mixed program's windows
    are "unresolved" and must not be fitted or scored. ``job_label`` keeps the
    job's objective for reference; it is never a target or a feature.

    ``label_train_frac`` is the share of the window's wall time spent in
    training slots; one minus it is the evasion tax the window paid.
    """
    if not rows:
        return
    from classifier.workload_labels import program_work
    starts = np.asarray([r["window_start"] for r in rows], float)
    ends = np.asarray([r["window_end"] for r in rows], float)
    if intervals is not None:
        lab = label_windows(starts, ends, intervals)
        length = ends - starts
        frac = np.clip(lab["train_s"] / np.where(length > 0, length, np.nan), 0.0, 1.0)
        frac = np.where(lab["unknown_s"] > 0, np.nan, frac)
        cols = dict(threeway_label=lab["label"], label_source=np.where(lab["label"] == "unresolved", "unresolved", "steps"),
                    label_train_s=lab["train_s"], label_infer_s=lab["infer_s"],
                    label_other_s=lab["other_s"], label_unknown_s=lab["unknown_s"], label_train_frac=frac)
    else:
        # No STEP evidence. Before giving up on the window, try the schedule: a
        # deterministically diluted cell trains on batch indices 0, N, 2N, ...
        # and a measured batch rate places those in time. Camera-ready runs all
        # predate STEP logging, so this is the only per-window truth they have.
        from classifier.schedule_reconstruction import reconstruct_labels
        reconstructed, _ = reconstruct_labels(workload_label, starts, ends, float(starts.min()))
        work = program_work(workload_label)
        nan = [np.nan] * len(rows)
        if reconstructed is None:
            cols = dict(threeway_label=[work or "unresolved"] * len(rows),
                        label_source=["program" if work else "unresolved"] * len(rows),
                        label_train_s=nan, label_infer_s=nan, label_other_s=nan,
                        label_unknown_s=nan, label_train_frac=nan)
        else:
            # Windows the rate interval cannot adjudicate stay unresolved rather
            # than inheriting a label the recorded data does not support.
            resolved = np.array([r is not None for r in reconstructed])
            labels = np.where(resolved, reconstructed, work or "unresolved").astype(object)
            source = np.where(resolved, "reconstructed",
                              "program" if work else "unresolved").astype(object)
            cols = dict(threeway_label=labels, label_source=source,
                        label_train_s=nan, label_infer_s=nan, label_other_s=nan,
                        label_unknown_s=nan,
                        label_train_frac=np.where(resolved,
                                                  (labels == "ml_training").astype(float), np.nan))
    for i, row in enumerate(rows):
        row["job_label"] = objective
        for key, values in cols.items():
            row[key] = values[i]


WORK_EVENTS_SCHEMA = {"start": "float64", "end": "float64", "work": "string",
                      "node": "int64", "rank": "int64", "job_id": "string"}


def save_work_events(events, path):
    """Persist ``{job_id: intervals}`` for cache workers.

    Typed columns even when no job has STEP lines: an empty untyped job_id is
    written as null, and every filtered read of it then fails.
    """
    frames = [iv.assign(job_id=str(job)) for job, iv in events.items() if len(iv)]
    table = (pd.concat(frames, ignore_index=True) if frames
             else pd.DataFrame(columns=list(WORK_EVENTS_SCHEMA)))
    table[list(WORK_EVENTS_SCHEMA)].astype(WORK_EVENTS_SCHEMA).to_parquet(path, index=False)


def load_work_events(path, jobs) -> dict:
    """The saved intervals of ``jobs``; a job absent from the file has no STEP lines."""
    jobs = [str(j) for j in jobs]
    if not jobs:
        return {}
    table = pd.read_parquet(path, filters=[("job_id", "in", jobs)])
    table["work"] = table["work"].astype(object)
    return {job: g[["start", "end", "work", "node", "rank"]].reset_index(drop=True)
            for job, g in table.groupby("job_id", sort=False)}
