#!/usr/bin/env python3
"""Stage launcher logs from GCS with hashes; never download telemetry payloads.

Run in an authenticated Cloud Shell:
  python3 scripts/gcp/fetch_work_logs.py --out /tmp/gpu-step-logs

The export is resumable and stays outside the checkout. TPU objects must belong
to run IDs in tracked data/tpu filenames. GPU inference logs retain their GCS
session/host path, including older logs whose filename omits the run ID.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import datetime
import gzip
import hashlib
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import tempfile

ROOT = Path(__file__).resolve().parents[2]
BUCKET = "gs://focus-surfer-494820-g0-telemetry"
RUN_ID = re.compile(r"Run ID:\s*(\S+)")
STEP = re.compile(r"\bSTEP\s+ts=([\d.]+)\s+kind=(\w+)\b")


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked_tpu_ids(root):
    paths = subprocess.check_output(["git", "-C", str(root), "ls-files", "data/tpu"], text=True)
    return set(re.findall(r"_([0-9a-f]{8,})\.parquet$", paths, re.M))


def select_log(uri, tpu_ids):
    if not uri.startswith("gs://") or not uri.endswith((".log", ".log.gz")):
        return False
    name = uri.rsplit("/", 1)[-1]
    if any(x in name for x in (".nic.", ".ray.")) or "smoke" in uri:
        return False
    if "/tpu/" in uri:
        run = re.search(r"/runs/([0-9a-f]{8,})/w\d+/", uri)
        return bool(run and run[1] in tpu_ids and re.search(r"_w\d+\.log(?:\.gz)?$", name))
    return bool("/logs/" in uri and re.search(r"(?:inferserve|infershard|inferrep|infermem|inferhf|vllm)", name))


def relative_path(uri):
    path = PurePosixPath(uri.removeprefix("gs://"))
    if path.is_absolute() or any(p in (".", "..") for p in path.parts):
        raise ValueError(f"Unsafe object path: {uri}")
    return Path(*path.parts)


def inspect_log(path):
    ids, counts, times = set(), {}, []
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", encoding="utf-8", errors="replace") as f:
        for line in f:
            match = RUN_ID.search(line)
            if match:
                ids.add(match[1])
            match = STEP.search(line)
            if match:
                counts[match[2]] = counts.get(match[2], 0) + 1
                times.append(float(match[1]))
    return dict(run_ids=sorted(ids), step_kinds=counts, steps=sum(counts.values()),
                first_step=min(times) if times else None, last_step=max(times) if times else None)


def fetch_one(uri, out):
    rel = relative_path(uri)
    target = out / "objects" / rel
    target = target if target.name.endswith(".gz") else target.with_suffix(target.suffix + ".gz")
    metadata = target.with_suffix(target.suffix + ".json")
    if target.is_file() and metadata.is_file():
        saved = json.loads(metadata.read_text())
        if saved["compressed_sha256"] == sha256(target):
            return saved
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=out) as td:
        raw = Path(td) / ("source.log.gz" if uri.endswith(".gz") else "source.log")
        subprocess.run(["gcloud", "storage", "cp", uri, str(raw)], check=True,
                       stdout=subprocess.DEVNULL, timeout=600)
        evidence = inspect_log(raw)
        expected = re.search(r"/runs/([0-9a-f]{8,})/w\d+/", uri)
        if expected and evidence["run_ids"] != [expected[1]]:
            raise ValueError(f"Run ID mismatch or missing header: {uri}: {evidence['run_ids']}")
        record = dict(uri=uri, fetched_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                      source_sha256=sha256(raw), source_bytes=raw.stat().st_size, **evidence)
        pending = target.with_suffix(target.suffix + ".tmp")
        if uri.endswith(".gz"):
            shutil.copyfile(raw, pending)
        else:
            with raw.open("rb") as src, pending.open("wb") as dest:
                with gzip.GzipFile(filename="", fileobj=dest, mode="wb", mtime=0) as gz:
                    shutil.copyfileobj(src, gz)
        pending.replace(target)
        record.update(local_path=str(target.relative_to(out)), compressed_sha256=sha256(target),
                      compressed_bytes=target.stat().st_size)
        metadata.write_text(json.dumps(record, indent=2) + "\n")
        return record


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--bucket", default=BUCKET)
    ap.add_argument("--repo", type=Path, default=ROOT)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    ids = tracked_tpu_ids(args.repo)
    if not ids:
        raise SystemExit("No tracked TPU run IDs; run against the project checkout")
    uris = set()
    for prefix in ("tpu", "runs"):
        print(f"Listing {args.bucket}/{prefix}/", flush=True)
        result = subprocess.run(["gcloud", "storage", "ls", "--recursive",
                                 args.bucket.rstrip("/") + "/" + prefix + "/"],
                                capture_output=True, text=True, timeout=600)
        if result.returncode:
            raise SystemExit(result.stderr)
        uris.update(line.strip() for line in result.stdout.splitlines()
                    if select_log(line.strip(), ids))
    if not uris:
        raise SystemExit("No matching launcher logs found; check the bucket/prefix")
    print(f"Downloading {len(uris)} logs to {args.out} (no telemetry files)", flush=True)
    manifest = dict(schema="work-log-export-v1", bucket=args.bucket,
                    selection_commit=subprocess.check_output(["git", "-C", str(args.repo), "rev-parse", "HEAD"], text=True).strip(),
                    objects=[], failures=[])
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(fetch_one, uri, args.out): uri for uri in sorted(uris)}
        for future in as_completed(futures):
            try:
                record = future.result()
                manifest["objects"].append(record)
                print(f"{len(manifest['objects'])}/{len(uris)}: {record['steps']} STEP lines: {record['uri']}", flush=True)
            except Exception as exc:
                manifest["failures"].append(dict(uri=futures[future], error=str(exc)))
    manifest["objects"].sort(key=lambda x: x["uri"])
    manifest["failures"].sort(key=lambda x: x["uri"])
    (args.out / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(f"Saved {args.out / 'manifest.json'}; failures={len(manifest['failures'])}")
    if manifest["failures"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
