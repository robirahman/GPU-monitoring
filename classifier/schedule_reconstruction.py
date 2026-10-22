"""Per-window truth for deterministic duty-cycle jobs that predate STEP logging.

STEP logging arrived on 2026-07-24 (`8ee55e53`), after every camera-ready run
had been collected, so the diluted families in the adversarial tables have no
recorded per-window ground truth: 375 jobs and 9,139 windows of it, per
docs/camera_ready_manifest_20261022.

What *is* recorded is the schedule. Strategy L trains on batch indices
0, N, 2N, ... and infers on the rest (workloads/adversarial_round3.py), and
`batch_idx=0` begins at the start of the run. Per-batch wall-clock timing was
never logged -- the generator's own comment says so -- but batches of one run
take near-constant time, so with a batch rate r the k-th batch occupies roughly
[t0 + k/r, t0 + (k+1)/r) and the training steps land at k = 0, N, 2N, ...

The rate is not known exactly, so it is carried as an interval taken from the
slowest and fastest reference run of the same cell. A window is labelled only
when both endpoints agree; where they disagree the window is INDETERMINATE and
is excluded from scoring rather than guessed. Reconstructed labels are marked
`label_source="reconstructed"` and never counted as observed evidence.

A warning this module exists to surface: the reference runs for the sweep cells
are dataloader-starved. `L_diluted_10_rep1` completed 4 batches in 300 s (~70 s
each) while its 8-worker sibling `L_diluted_10_mw8_rep1` completed 4,647. At 4
batches a 1-in-10 schedule takes exactly one gradient step, so nearly every
window of that run really does contain no training -- but that is a property of
a starved run, not of dilution, and it is worth knowing before either number is
quoted.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from classifier.data_sources import ROOT
from classifier.window_labels import default_log_paths, parse_work_log

# Cells whose schedule is deterministic in the batch index. The trailing group
# is N, the number of batches per gradient step.
DETERMINISTIC_DILUTION = re.compile(
    r"^(?:dilution_sweep_N|adversarial_L_diluted_|whitebox_diluted_N|whitebox_lora_N)"
    r"(\d+)(?:_mw\d+)?$")
PURE_INFERENCE = re.compile(r"^(?:dilution_sweep_inference|wb_inference.*)$")


def dilution_factor(label):
    """N for a deterministically diluted cell, 1 for pure inference, else None."""
    m = DETERMINISTIC_DILUTION.match(str(label))
    return int(m[1]) if m else None


def reference_rates(label, root=ROOT):
    """(low, high, n_runs) batches/second from STEP logs of the SAME cell.

    Same cell, not same family: `L_diluted_10` and `L_diluted_10_mw8` are the
    same dilution at rates three orders of magnitude apart, and borrowing one
    for the other would relocate every reconstructed step.
    """
    # Telemetry calls the cell `adversarial_L_diluted_10`; its launcher log is
    # `L_diluted_10_rep1.log.gz`, optionally under a `<session>__<host>__` prefix.
    stem_base = re.sub(r"^adversarial_", "", str(label))
    want = re.compile(rf"^{re.escape(stem_base)}_rep\d+$")
    rates = []
    for path in default_log_paths(Path(root)):
        stem = Path(path).name.split(".log")[0]
        stem = stem.split("__")[-1]
        if not want.match(stem):
            continue
        ts = parse_work_log(path)["events"].ts.to_numpy(dtype=float)
        ts = np.sort(ts[np.isfinite(ts)])
        # n-1 intervals between n logged batches.
        if len(ts) >= 2 and ts[-1] > ts[0]:
            rates.append((len(ts) - 1) / (ts[-1] - ts[0]))
    if not rates:
        return None, None, 0
    return float(min(rates)), float(max(rates)), len(rates)


def _labels_at_rate(starts, ends, t0, n, rate):
    """Which windows contain a batch index divisible by n, at this rate."""
    # Training batch j occupies [t0 + j*n/rate, t0 + (j+1)*n/rate - (n-1)/rate).
    # Only its first instant matters for containment, so place the step there.
    period = n / rate
    first = np.ceil((starts - t0) / period)
    # The step is inside [start, end) iff some j*period lies in it.
    return (t0 + first * period) < ends


def reconstruct_labels(label, window_starts, window_ends, t0, root=ROOT):
    """Per-window labels for one diluted run, or None where not recoverable.

    Returns (labels, meta). `labels` is an object array of "ml_training",
    "ml_inference" or None; None marks a window the recorded data cannot
    adjudicate.
    """
    starts = np.asarray(window_starts, dtype=float)
    ends = np.asarray(window_ends, dtype=float)
    # Single-kind programs need no reconstruction: workload_labels.program_work
    # already states what they do, and claiming a reconstruction here would
    # overstate the provenance of a label the program name settles.
    if PURE_INFERENCE.match(str(label)):
        return None, dict(reason="pure-inference control; the program label settles it")
    n = dilution_factor(label)
    if n is None:
        return None, dict(reason="not a deterministically diluted cell")
    if n == 1:
        return None, dict(reason="N=1 trains on every batch; the program label settles it")
    lo, hi, n_runs = reference_rates(label, root)
    if lo is None:
        return None, dict(reason=f"no STEP log measures the batch rate of {label}")
    at_lo = _labels_at_rate(starts, ends, t0, n, lo)
    at_hi = _labels_at_rate(starts, ends, t0, n, hi)
    agree = at_lo == at_hi
    labels = np.where(at_lo, "ml_training", "ml_inference").astype(object)
    labels[~agree] = None
    return labels, dict(reason=f"schedule 1-in-{n} at {lo:.4g}-{hi:.4g} batches/s "
                               f"from {n_runs} reference run(s)",
                        rate_low=lo, rate_high=hi, reference_runs=n_runs,
                        determinate=float(agree.mean()))


def reconstruct_frame(windows, root=ROOT):
    """Add reconstructed per-window truth to a window frame, in place-ish.

    Adds `reconstructed_label` (None where not recoverable) and sets
    `label_source="reconstructed"` on the rows it could adjudicate. Rows of
    cells this module does not cover are left exactly as they were.
    """
    out = windows.copy()
    out["reconstructed_label"] = None
    notes = {}
    for (label, run_id), g in out.groupby(["workload_label", "run_id"], sort=False):
        labels, meta = reconstruct_labels(label, g.window_start.to_numpy(float),
                                          g.window_end.to_numpy(float),
                                          float(g.window_start.min()), root)
        notes[str(label)] = meta
        if labels is None:
            continue
        out.loc[g.index, "reconstructed_label"] = labels
    resolved = out.reconstructed_label.notna()
    if resolved.any():
        out.loc[resolved, "threeway_label"] = out.loc[resolved, "reconstructed_label"]
        out.loc[resolved, "label_source"] = "reconstructed"
    return out, notes
