"""Shared input policy; read LFS payloads in place on a separate volume.

Set SPAR_LFS_OBJECTS to an objects directory (e.g. /tmp/.../lfs/objects).
Missing payloads are errors: a partial corpus must never look like a full run.
"""
import hashlib
import json
import os
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def filename_workload_label(path):
    p = Path(path)
    if "week4/results/edge_cases/" in str(p) and p.name.endswith("_telemetry.json"):
        return p.stem.removesuffix("_telemetry")
    if re.search(r"week[47]/results/(?:ddp|dataset_scale)/", str(p)):
        return "training_dual_gpu_dp"
    label = re.split(r"_(?:NVIDIA|AMD|Google_TPU)_", p.stem)[0]
    if "week7" in p.parts:
        label = re.sub(r"_[0-9a-f]{6,}$", "", label)
    return label


def pointer(path):
    with Path(path).open("rb") as f:
        head = f.read(1024)
    if not head.startswith(b"version https://git-lfs.github.com/spec/v1\n"):
        return None
    fields = dict(line.split(" ", 1) for line in head.decode().splitlines())
    oid = fields["oid"].removeprefix("sha256:")
    if len(oid) != 64 or any(c not in "0123456789abcdef" for c in oid):
        raise ValueError(f"Invalid LFS oid in {path}")
    return oid, int(fields["size"])


def resolve_payload(path, objects=None):
    path = Path(path)
    ptr = pointer(path)
    if ptr is None:
        return path
    oid, size = ptr
    roots = [Path(objects)] if objects else []
    if not objects and os.environ.get("SPAR_LFS_OBJECTS"):
        roots.append(Path(os.environ["SPAR_LFS_OBJECTS"]))
    roots.append(ROOT / ".git/lfs/objects")
    for root in roots:
        payload = root / oid[:2] / oid[2:4] / oid
        if payload.is_file() and payload.stat().st_size == size:
            return payload
    raise FileNotFoundError(f"LFS payload missing/incomplete: {path} ({oid}, {size} bytes). "
                            "Set SPAR_LFS_OBJECTS to the completed objects directory.")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for buf in iter(lambda: f.read(1024 * 1024), b""):
            h.update(buf)
    return h.hexdigest()


def quarantined(path):
    return any("quarantine" in part.lower() for part in Path(path).parts)


def data_role(path):
    p = Path(path)
    name = p.name.lower()
    if name.startswith("nic_"):
        return "nic"
    if "sensor_logs" in p.parts or name.endswith("_sensors.parquet"):
        return "physical"
    if "wall_power" in name:
        return "wall_power"
    for role in ("scope", "dcgm", "bmc"):
        if f"_{role}_" in name or name.endswith(f"_{role}.parquet"):
            return role
    return "accelerator"


def accelerator_files(root=ROOT):
    roots = ("data", "week4/data", "week4/results/ddp", "week4/results/dataset_scale",
             "week7/single_gpu_tests/data", "week7/two_gpu_tests/data",
             "week7/edge_cases/data", "week7/results/dataset_scale")
    return sorted(p for sub in roots for p in (Path(root) / sub).rglob("*.parquet")
                  if not quarantined(p) and data_role(p) == "accelerator")


def read_parquet(path, **kwargs):
    return pd.read_parquet(resolve_payload(path), **kwargs)


def apply_quality_exclusions(frame, root=ROOT):
    """Apply reviewed exclusions by physical job identity, preserving raw files."""
    manifest = Path(root) / "data/quality_exclusions.json"
    if not manifest.is_file():
        return frame
    from classifier.multinode_features import assign_stream_and_job_ids
    excluded = {r["job_id"] for r in json.loads(manifest.read_text())["excluded_jobs"]}
    identified = assign_stream_and_job_ids(frame)
    hit = identified.job_id.isin(excluded)
    if hit.any():
        import logging
        logging.getLogger(__name__).warning("Excluded %d rows / %d jobs with failed completion evidence (%s)",
                                            hit.sum(), identified.loc[hit, "job_id"].nunique(), manifest)
    kept = identified.loc[~hit]
    # Resetting the index must not allocate another full telemetry-frame copy.
    kept.reset_index(drop=True, inplace=True)
    return kept
