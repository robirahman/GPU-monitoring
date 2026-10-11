"""Content-bound cache manifests, including uncommitted implementation changes."""
import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from classifier.data_sources import ROOT, accelerator_files, pointer, resolve_payload, sha256, quarantined
from classifier.workload_labels import LABEL_SCHEMA_VERSION, FAMILY_SCHEMA_VERSION


def provenance(root=ROOT, **config):
    root = Path(root)
    paths = accelerator_files(root)
    paths += sorted(p for p in (root / "data").rglob("nic_*.parquet") if not quarantined(p))
    paths += sorted((root / "week4/results/edge_cases").glob("*_telemetry.json"))
    if (root / "data/quality_exclusions.json").is_file():
        paths.append(root / "data/quality_exclusions.json")
    inputs = []
    for p in paths:
        payload = resolve_payload(p)  # completeness required even for cache reuse
        ptr = pointer(p)
        oid = ptr[0] if ptr else sha256(payload)
        inputs.append((str(p.relative_to(root)), oid, payload.stat().st_size))
    code = sorted((root / "classifier").rglob("*.py"))
    code += sorted((root / "scripts").glob("build*cache.py"))
    doc = dict(inputs=inputs, code=[(str(p.relative_to(root)), sha256(p)) for p in code],
               labels=LABEL_SCHEMA_VERSION, families=FAMILY_SCHEMA_VERSION, config=config,
               libraries={name: version(name) for name in ("numpy", "pandas", "scipy", "pyarrow", "scikit-learn")})
    doc["fingerprint"] = hashlib.sha256(json.dumps(doc, sort_keys=True).encode()).hexdigest()
    return doc


def cache_matches(path, manifest):
    path = Path(path)
    sidecar = Path(str(path) + ".provenance.json")
    if not path.is_file() or not sidecar.is_file():
        return False
    try:
        saved = json.loads(sidecar.read_text())
        return (saved["fingerprint"] == manifest["fingerprint"]
                and saved["artifact_sha256"] == sha256(path))
    except (ValueError, KeyError):
        return False


def stamp_cache(path, manifest):
    out = Path(str(path) + ".provenance.json")
    tmp = out.with_suffix(out.suffix + ".tmp")
    tmp.write_text(json.dumps(dict(manifest, artifact_sha256=sha256(path)), indent=2))
    tmp.replace(out)
