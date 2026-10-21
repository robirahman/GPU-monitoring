"""Versioned experiment identities, manifests and artifact verification.

This module intentionally has no GPU or dataframe dependencies. A job identifies
one experimental repetition; retries share that job but have distinct attempts.
All rank, GPU and sensor streams from a job must share its evaluation split.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import tempfile
from pathlib import Path

SCHEMA_VERSION = 1
CLASSES = {"ml_training", "ml_inference", "other"}
SENSORS = {"nvml", "dcgm", "network", "bmc", "scope", "ground_truth"}
IDENTIFIER = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,127}$")


def identifier(value: str) -> str:
    if not isinstance(value, str) or not IDENTIFIER.fullmatch(value):
        raise ValueError(f"Invalid identifier: {value!r}")
    return value


def canonical_hash(value) -> str:
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def atomic_json(path: str | Path, value) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(dir=path.parent, prefix=path.name + ".")
    try:
        with os.fdopen(fd, "w") as f:
            json.dump(value, f, sort_keys=True, indent=2, allow_nan=False)
            f.write("\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(name, path)
    finally:
        if os.path.exists(name):
            os.unlink(name)


def stream_identity(job_id, attempt_id, node_id, device_id, sensor):
    if sensor not in SENSORS:
        raise ValueError(f"Unknown sensor {sensor}")
    parts = [identifier(str(x)) for x in (job_id, attempt_id, node_id, device_id, sensor)]
    return ":".join(parts)


def artifact(path: str | Path, root: str | Path, sensor: str, **metadata):
    path, root = Path(path).resolve(), Path(root).resolve()
    if sensor not in SENSORS:
        raise ValueError(sensor)
    return {"path": str(path.relative_to(root)), "sensor_type": sensor,
            "size_bytes": path.stat().st_size, "sha256": file_hash(path), **metadata}


def validate_campaign(doc):
    if doc.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("Unsupported campaign schema")
    identifier(doc["campaign_id"])
    jobs = doc.get("jobs", [])
    if not jobs or len({j["job_id"] for j in jobs}) != len(jobs):
        raise ValueError("Campaign must contain uniquely identified jobs")
    for job in jobs:
        identifier(job["job_id"])
        identifier(job["condition"])
        if job["class_label"] not in CLASSES:
            raise ValueError("Explicit class_label required")
        for field in ("nodes", "gpus_per_node", "duration_sec", "rep"):
            if type(job[field]) is not int or job[field] <= 0:
                raise ValueError(f"{field} must be a positive integer")
        if job["split"] not in {"train", "validation", "test", "pilot"}:
            raise ValueError("Unknown split")
        if job.get("fabric") not in {"rdma", "tcp"}:
            raise ValueError("fabric must be rdma or tcp")
    expected = doc.get("manifest_sha256")
    if expected != canonical_hash({k: v for k, v in doc.items() if k != "manifest_sha256"}):
        raise ValueError("Campaign hash mismatch; regenerate instead of editing a frozen manifest")
    return doc


def verify_artifacts(doc, root):
    root = Path(root).resolve()
    for item in doc["artifacts"]:
        path = (root / item["path"]).resolve()
        if not path.is_relative_to(root) or not path.is_file():
            raise ValueError(f"Missing or unsafe artifact: {item['path']}")
        if path.stat().st_size != item["size_bytes"] or file_hash(path) != item["sha256"]:
            raise ValueError(f"Artifact checksum mismatch: {item['path']}")


def finite_number(value):
    return isinstance(value, (int, float)) and math.isfinite(value)
