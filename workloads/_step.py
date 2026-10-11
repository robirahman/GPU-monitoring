"""Shared STEP-line emitter (superset dialect) for workloads.

A STEP line is per-window ground truth. It is teed into the launcher log and
joined back to telemetry by epoch clock + run_id (classifier/window_labels.py),
so `ts` MUST be time.time() -- the same clock the collector stamps into
`timestamp_epoch`. The superset dialect adds `node`/`rank` so a multi-node
distributed job can attribute each step to a physical server; the parsers
(window_labels.STEP_RE and the relaxed symgemm regex) read `ts kind batch_idx`
positionally and treat every later `key=value` field as optional.

    STEP ts=<epoch> kind=train|infer batch_idx=N tokens=M passes=P node=r rank=g [k=v ...]

Emit from rank 0 only for a synchronous data-/model-parallel job (every rank
does the same thing each step). Emit from ALL ranks when ranks diverge -- e.g.
staggered dilution, where rank 0's train/infer pattern is not representative of
the others.
"""

import time


def emit_step(kind, batch_idx, *, tokens=0, passes=1, node=0, rank=0, **kv):
    """Print one STEP line in the superset dialect.

    `kind` names the work the slot did, and labels its telemetry windows:
    'train' is a gradient step, 'infer' a forward pass, and any other kind
    ('filler', 'nonml', 'stall', ...) is non-ML work.
    """
    line = (f"STEP ts={time.time():.3f} kind={kind} batch_idx={batch_idx} "
            f"tokens={tokens} passes={passes} node={node} rank={rank}")
    for k, v in kv.items():
        line += f" {k}={v}"
    print(line, flush=True)
