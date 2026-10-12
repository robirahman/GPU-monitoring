#!/usr/bin/env python3
"""Inventory the exact cache inputs, including missing LFS payloads.

Read only; never downloads or materializes data in HOME. Writes an explicit
missing/corrupt object manifest and exits 2 when the analysis cannot run.
Scope waveforms, physical sidecars and quarantines are not required cache inputs.
"""
import argparse
import csv
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.cache_provenance import input_paths
from classifier.data_sources import ROOT, pointer, sha256


def inspect_inputs(root=ROOT, objects=None, verify=True):
    root = Path(root)
    roots = [Path(objects)] if objects else []
    if not objects and os.environ.get('SPAR_LFS_OBJECTS'):
        roots.append(Path(os.environ['SPAR_LFS_OBJECTS']))
    roots.append(root / '.git/lfs/objects')
    rows, checked = [], {}
    for path in input_paths(root):
        ptr = pointer(path)
        oid, size = ptr if ptr else (None, path.stat().st_size)
        payload = path
        status = 'local'
        if ptr:
            candidates = [r / oid[:2] / oid[2:4] / oid for r in roots]
            payload = next((p for p in candidates if p.is_file() and p.stat().st_size == size), None)
            status = 'lfs_available' if payload else 'missing_or_incomplete'
        if payload is not None:
            key = str(payload)
            if key not in checked:
                checked[key] = sha256(payload) if verify or not ptr else oid
            actual = checked[key]
            if ptr and verify and actual != oid:
                status = 'corrupt'
            if not ptr:
                oid = actual
        rows.append(dict(path=str(path.relative_to(root)), sha256=oid, bytes=size,
                         lfs=bool(ptr), status=status,
                         payload=str(payload) if payload else ''))
    missing = [r for r in rows if r['status'] in ('missing_or_incomplete', 'corrupt')]
    objects_by_id = {r['sha256']: r['bytes'] for r in rows}
    missing_by_id = {r['sha256']: r['bytes'] for r in missing}
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        purpose='Exact input policy used by classifier.cache_provenance',
        ready=bool(rows) and not missing, input_paths=len(rows),
        unique_objects=len(objects_by_id), unique_bytes=sum(objects_by_id.values()),
        missing_or_corrupt_paths=len(missing), missing_or_corrupt_objects=len(missing_by_id),
        missing_or_corrupt_bytes=sum(missing_by_id.values()), verified_payload_hashes=verify,
        objects_roots=list(map(str, roots)),
        note='Readiness covers selected existing inputs, not completeness of remote collections or scientific validity.')
    return summary, rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, default=ROOT)
    ap.add_argument('--objects', type=Path)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--no-verify', action='store_true', help='Sizes only for LFS; diagnostic mode')
    a = ap.parse_args()
    summary, rows = inspect_inputs(a.repo, a.objects, not a.no_verify)
    a.out.mkdir(parents=True, exist_ok=True)
    for name, selected in [('inputs.csv', rows), ('missing.csv', [r for r in rows
                           if r['status'] in ('missing_or_incomplete', 'corrupt')])]:
        with (a.out / name).open('w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=['path','sha256','bytes','lfs','status','payload'])
            writer.writeheader(); writer.writerows(selected)
    (a.out / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    return 0 if summary['ready'] else 2


if __name__ == '__main__':
    sys.exit(main())
