#!/usr/bin/env python3
"""Quarantine scope-ladder telemetry that no .done marker vouches for.

Why this is needed
------------------
run_workload_multi_gpu.py saves per-GPU telemetry *whether or not the workload
succeeded* -- deliberately, so a partial cell is not lost. But the queue records
success separately, as a .done marker, and ingest_session.sh globs parquets with
no reference to those markers. So a cell that died in 15 s still contributes a
file named after it, and load_all_data() will happily read 15 s of IDLE GPUs
labelled `s16_train_ddp_resnet50` -- i.e. it teaches a train-vs-inference
classifier that idle looks like training. On the 2026-08-20 session roughly 78%
of the scope-ladder parquets came from failed cells.

The rule
--------
A parquet is kept only if the queue marked its cell complete on that node:

  parquet   <label>_node<N>_gpu<I>_<gpu_name>_<rid12>.parquet
  marker    <label>_node<N>_<rid12>.done

Matching on rid is exact and rep-aware. Two wrinkles:

* --parallel cells (the inferrep rungs) give every replica its own uuid, so the
  file's rid is NOT the cell rid and can never match. For those we fall back to
  "this (label, node) has at least one .done marker", and say so in the report.
* Files whose label is not a scope-ladder cell (the repo ships ~1700 historical
  A100/B200/H200 parquets under data/) are left strictly alone.

Rejects are MOVED, never deleted, so a judgement call here is reversible.

  python scripts/gcp/prune_unmarked_cells.py --data DIR --markers DIR [--apply]
"""

import argparse
import os
import re
import shutil
import sys
from collections import defaultdict

# Scope-ladder cells only: s1_/s8_/s16_ (GPU) and tpuT<N>_ (TPU).
LADDER_RE = re.compile(r"^(?:smoke_)?(?:s(?:1|8|16)|tpuT\d+)_")
# <label>_node<N>_gpu<I>_<gpu_name>_<rid>.parquet
FILE_RE = re.compile(r"^(?P<label>.+?)_node(?P<node>\d+)_gpu(?P<gpu>\d+)_.*?_(?P<rid>[0-9a-f]{12})\.parquet(?:\.partial)?$")
# <label>_node<N>_<rid>.done
MARK_RE = re.compile(r"^(?P<label>.+?)_node(?P<node>\d+)_(?P<rid>[0-9a-f]{12})\.done$")


def load_markers(markers_dir):
    """-> (exact {(label,node,rid)}, any_rep {(label,node)})"""
    exact, any_rep = set(), set()
    for root, _, files in os.walk(markers_dir):
        for fn in files:
            m = MARK_RE.match(fn)
            if not m:
                continue
            label, node, rid = m["label"], int(m["node"]), m["rid"]
            exact.add((label, node, rid))
            any_rep.add((label, node))
    return exact, any_rep


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="directory of parquets (recursive)")
    ap.add_argument("--markers", required=True, help="directory of .done markers (recursive)")
    ap.add_argument("--quarantine", default=None,
                    help="where rejects go (default: <data>/../quarantine_unmarked)")
    ap.add_argument("--apply", action="store_true", help="actually move (default: dry run)")
    ap.add_argument("--replica-labels", default="inferrep",
                    help="comma-separated substrings identifying --parallel cells, "
                         "whose per-file rid is a per-replica uuid rather than the "
                         "cell rid and so can never match a marker (default: inferrep)")
    args = ap.parse_args()

    exact, any_rep = load_markers(args.markers)
    if not exact:
        print(f"refusing to prune: no .done markers found under {args.markers}", file=sys.stderr)
        return 2
    print(f"markers: {len(exact)} cell-reps across {len(any_rep)} (cell,node) pairs")

    quarantine = args.quarantine or os.path.join(os.path.dirname(os.path.abspath(args.data)),
                                                 "quarantine_unmarked")
    replica_pats = [t for t in (x.strip() for x in args.replica_labels.split(",")) if t]
    keep = defaultdict(int)
    drop = defaultdict(int)
    skipped = 0
    moves = []

    for root, _, files in os.walk(args.data):
        if os.path.abspath(root).startswith(os.path.abspath(quarantine)):
            continue
        for fn in sorted(files):
            if not fn.endswith((".parquet", ".parquet.partial")):
                continue
            m = FILE_RE.match(fn)
            if not m or not LADDER_RE.match(fn):
                skipped += 1          # historical / non-ladder data: untouched
                continue
            label, node, rid = m["label"], int(m["node"]), m["rid"]
            if (label, node, rid) in exact:
                keep[label] += 1
            elif (label, node) in any_rep and any(t in label for t in replica_pats):
                # A --parallel cell: the rid is a replica uuid, so exact matching is
                # impossible and "this cell succeeded on this node" is the best we
                # can do. NOT applied to rid-bearing cells -- doing so kept the
                # failed rep 2 of s16_train_fsdp30b on the strength of rep 1.
                keep[label + "  [replica uuid; cell has a .done]"] += 1
            else:
                drop[label] += 1
                moves.append(os.path.join(root, fn))

    print(f"\nnon-ladder files left alone: {skipped}")
    print(f"\nKEEP ({sum(keep.values())} files):")
    for k in sorted(keep):
        print(f"  {keep[k]:4d}  {k}")
    print(f"\nQUARANTINE ({sum(drop.values())} files -- no .done for this cell/node):")
    for k in sorted(drop):
        print(f"  {drop[k]:4d}  {k}")

    if not args.apply:
        print(f"\ndry run. re-run with --apply to move {len(moves)} file(s) to {quarantine}")
        return 0

    os.makedirs(quarantine, exist_ok=True)
    for src in moves:
        shutil.move(src, os.path.join(quarantine, os.path.basename(src)))
    print(f"\nmoved {len(moves)} file(s) -> {quarantine}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
