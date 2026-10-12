#!/usr/bin/env python3
"""Verify a fetch_work_logs.py export and copy STEP evidence into logs/gcs.

Preserves the GCS bucket/session/run/worker path to avoid basename collisions.
The manifest also records skipped logs with no STEP evidence. No Git commit or
push is performed; inspect coverage and commit the resulting files separately.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.gcp.fetch_work_logs import inspect_log, relative_path, sha256


def import_export(export, root):
    export, root = Path(export), Path(root)
    source = json.loads((export / "manifest.json").read_text())
    if source["schema"] != "work-log-export-v1" or source["failures"]:
        raise ValueError("Export is incomplete; rerun fetch_work_logs.py before importing")
    plan, records = [], []
    for record in source["objects"]:
        relative = Path(record["local_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Export path leaves staging directory")
        src = export / relative
        if sha256(src) != record["compressed_sha256"]:
            raise ValueError(f"Corrupt exported log: {src}")
        original_hash, original_bytes = hashlib.sha256(), 0
        opener = open if record["uri"].endswith(".gz") else gzip.open
        with opener(src, "rb") as stream:
            for chunk in iter(lambda: stream.read(1024 * 1024), b""):
                original_hash.update(chunk)
                original_bytes += len(chunk)
        if original_hash.hexdigest() != record["source_sha256"] or original_bytes != record["source_bytes"]:
            raise ValueError(f"Source-byte mismatch: {src}")
        evidence = inspect_log(src)
        for key in evidence:
            if evidence[key] != record[key]:
                raise ValueError(f"Export evidence mismatch: {src}: {key}")
        target = root / "logs/gcs" / relative_path(record["uri"])
        if not target.name.endswith(".gz"):
            target = target.with_suffix(target.suffix + ".gz")
        dest = str(target.relative_to(root))
        row = dict(record, disposition="committed_step_evidence" if record["steps"] else "no_step_evidence",
                   repository_path=dest if record["steps"] else None)
        records.append(row)
        if not record["steps"]:
            continue
        if target.exists() and sha256(target) != record["compressed_sha256"]:
            raise ValueError(f"Existing committed log differs; retain both generations explicitly: {target}")
        plan.append((src, target))
    # Validate the complete export before changing the checkout.
    for src, target in plan:
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src, target)
    manifest = dict(schema="committed-work-logs-v1", selection_commit=source["selection_commit"],
                    objects=records)
    destination = root / "logs/gcs/manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        previous = json.loads(destination.read_text())
        merged = {r["uri"]: r for r in previous["objects"]}
        merged.update({r["uri"]: r for r in records})
        manifest["objects"] = sorted(merged.values(), key=lambda r: r["uri"])
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[2])
    args = ap.parse_args()
    manifest = import_export(args.export, args.repo)
    records = manifest["objects"]
    kept = [r for r in records if r["repository_path"]]
    print(f"Manifest: {len(kept)} STEP logs; {len(records)-len(kept)} logs without STEP evidence")
    print(f"STEP lines: {sum(r['steps'] for r in kept)}")
    print(f"Compressed bytes: {sum(r['compressed_bytes'] for r in kept)}")


if __name__ == "__main__":
    main()
