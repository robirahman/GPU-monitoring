#!/usr/bin/env python3
"""Recover the NIC and TPU candidates from the 2026-10-12 GCS audit.

Works offline from the checksum-verified audit download cache. Default is a
dry run. Original bytes are retained under a quarantine directory; only NIC
intervals paired with existing, nonexcluded accelerator runs and successfully
completed TPU jobs enter the normal loaders. No Git or cloud writes.
"""
import argparse
from collections import Counter, defaultdict
import gzip
import hashlib
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.data_sources import sha256
from classifier.workload_labels import canonical_workload
from scripts.gcp.fetch_work_logs import inspect_log, relative_path
from scripts.gcp.import_work_logs import import_export

BUCKET = "focus-surfer-494820-g0-telemetry"
DEST = Path("data/gcs_recovery_20261012")
TPU_RUNS = {"ad5e56af3e3b", "cdc1669c2763"}
MIN_NIC_SPAN = 30.0


def load_json(path):
    return json.loads(Path(path).read_text())


def same_or_absent(path, digest):
    if path.exists() and sha256(path) != digest:
        raise ValueError(f"Refusing to overwrite different existing bytes: {path}")


def interval_mask(times, intervals):
    mask = np.zeros(len(times), dtype=bool)
    for start, end in intervals:
        mask |= (times >= start) & (times <= end)
    return mask


def time_stats(frame):
    ts = np.sort(frame.timestamp_epoch.unique())
    dt = np.diff(ts)
    median = float(np.median(dt)) if len(dt) else 0.0
    return dict(samples=len(frame), t0=float(ts[0]), t1=float(ts[-1]),
                span_s=float(ts[-1] - ts[0]), median_dt=median,
                gap_capped_s=float(np.minimum(dt, 2 * median).sum()))


def merge_stream(frames, run_id, hostname, cell):
    """Merge copies/checkpoint versions, refusing ambiguous sample values."""
    frame = pd.concat(frames, ignore_index=True).drop_duplicates()
    if set(frame.run_id.astype(str)) != {run_id} or set(frame.hostname.astype(str)) != {hostname}:
        raise ValueError(f"Mixed physical identity for {run_id}/{hostname}")
    if set(frame.workload_label.map(canonical_workload)) != {cell}:
        raise ValueError(f"Mixed workload identity for {run_id}/{hostname}")
    ts = pd.to_numeric(frame.timestamp_epoch, errors="coerce")
    if not np.isfinite(ts).all() or frame.timestamp_epoch.duplicated().any():
        raise ValueError(f"Invalid or conflicting timestamps for {run_id}/{hostname}")
    return frame.sort_values("timestamp_epoch").reset_index(drop=True)


def clean_nic(frame, intervals, existing_intervals=()):
    """Restrict to accepted GPU spans and the actual NIC merge's integer key."""
    keep = interval_mask(frame.timestamp_epoch, intervals)
    # Conservative second-wide guard against the existing loader's round(ts) key.
    keep &= ~interval_mask(frame.timestamp_epoch, [(a - 1, b + 1) for a, b in existing_intervals])
    out = frame.loc[keep].copy()
    counts = dict(outside_pair_or_existing_samples=int((~keep).sum()), invalid_metric_values=0,
                  rounded_key_duplicates=0)
    metrics = [c for c in out if c.startswith("nic_") or c in ("cpu_util_pct", "mem_used_mb", "mem_total_mb")]
    for col in metrics:
        val = pd.to_numeric(out[col], errors="coerce")
        invalid = ~np.isfinite(val) | (val < 0)
        counts["invalid_metric_values"] += int(invalid.sum())
        out[col] = val.mask(invalid)
    if out.empty:
        return out, counts
    out["_second"] = out.timestamp_epoch.round().astype("int64")
    out["_distance"] = (out.timestamp_epoch - out._second).abs()
    before = len(out)
    out = out.sort_values(["_distance", "timestamp_epoch"]).drop_duplicates("_second")
    counts["rounded_key_duplicates"] = before - len(out)
    return out.drop(columns=["_second", "_distance"]).sort_values("timestamp_epoch").reset_index(drop=True), counts


def nic_usable(frame):
    if len(frame) < 31:
        return False
    valid = np.zeros(len(frame), dtype=bool)
    for rx, tx in (("nic_total_rx_mbps", "nic_total_tx_mbps"), ("nic_rx_mbps", "nic_tx_mbps")):
        if rx in frame and tx in frame:
            valid |= frame[rx].notna() & frame[tx].notna()
    times = frame.loc[valid, "timestamp_epoch"].to_numpy()
    segments = np.split(times, np.flatnonzero(np.diff(times) > 2.5) + 1)
    return any(len(s) >= 31 and s[-1] - s[0] >= MIN_NIC_SPAN for s in segments)


def tpu_completion(marker_texts, run_id):
    codes = {}
    for worker, text in marker_texts.items():
        fields = dict(re.findall(r"(rc|run_id|chips|worker)=([^\s]+)", text))
        if fields.get("run_id") != run_id or fields.get("worker") != str(worker) or fields.get("chips") != "1":
            raise ValueError(f"Invalid completion marker: {run_id}/w{worker}")
        codes[str(worker)] = int(fields["rc"])
    if set(codes) != {"0", "1000", "2000", "3000"}:
        raise ValueError(f"Missing TPU worker completion evidence: {run_id}")
    return all(v == 0 for v in codes.values()), codes


class Recovery:
    def __init__(self, repo, audit, cache, baseline, staging):
        self.repo, self.audit, self.cache, self.baseline, self.staging = map(Path, (repo, audit, cache, baseline, staging))
        self.by_hash, self.by_name = defaultdict(list), {}
        for filename in ("partial-candidates-downloaded.json", "raw-download-candidates-downloaded.json", "log-candidates-downloaded.json"):
            for obj in load_json(self.cache / filename):
                if obj["bucket"] != BUCKET:
                    raise ValueError(f"Unexpected bucket: {obj['bucket']}")
                self.by_hash[obj["sha256"]].append(obj)
                self.by_name[obj["name"]] = obj
        self.files, self.sources, self.nic, self.tpu, self.logs = {}, {}, [], [], []
        self.old = pd.read_csv(self.baseline / "streams.csv", keep_default_na=False)
        self.excluded = {x["job_id"] for x in load_json(self.repo / "data/quality_exclusions.json")["excluded_jobs"]}
        self.old_accel = self.old[(self.old.role == "accelerator") & ~self.old.quarantine & ~self.old.job_id.isin(self.excluded)]
        self.old_nic = self.old[(self.old.role == "nic") & ~self.old.quarantine]

    def stage_bytes(self, relative, payload):
        relative = Path(relative)
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Unsafe destination: {relative}")
        path = self.staging / relative
        digest = hashlib.sha256(payload).hexdigest()
        same_or_absent(self.repo / relative, digest)
        if str(relative) in self.files and self.files[str(relative)]["sha256"] != digest:
            raise ValueError(f"Two different outputs for {relative}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
        self.files[str(relative)] = dict(path=str(relative), sha256=digest, bytes=len(payload))
        return str(relative)

    def source(self, obj):
        digest = obj["sha256"]
        if digest not in self.sources:
            path = Path(obj["path"])
            if sha256(path) != digest or path.stat().st_size != int(obj["size"]):
                raise ValueError(f"Corrupt cached source: {obj['name']}")
            suffix = ".parquet" if ".parquet" in obj["name"] else ".done" if obj["name"].endswith(".done") else ".log"
            # Raw logs live outside logs/ so only the verified compressed import is indexed.
            raw_path = self.stage_bytes(DEST / "_quarantine_raw" / (digest + suffix), path.read_bytes())
            origins = [{k: v for k, v in x.items() if k != "path"} for x in self.by_hash[digest]]
            self.sources[digest] = dict(sha256=digest, raw_path=raw_path, objects=origins)
        return Path(obj["path"])

    def save_frame(self, relative, frame):
        return self.stage_bytes(relative, frame.to_parquet(index=False))

    def recover_nic(self):
        tables = [("checkpoint", pd.read_csv(self.audit / "additional_nic_checkpoints.csv", keep_default_na=False)),
                  ("finalized", pd.read_csv(self.audit / "additional_nic_streams.csv", keep_default_na=False))]
        join_keys = set()
        for kind, table in tables:
            for candidate in table.to_dict("records"):
                rid, host, cell = (str(candidate[k]) for k in ("run_id", "hostname", "cell"))
                if kind == "checkpoint":
                    objects = [self.by_hash[h][0] for h in candidate["hashes"].split("|")]
                else:
                    objects = list({self.by_name[n]["sha256"]: self.by_name[n] for n in candidate["paths"].split("|")}.values())
                frame = merge_stream([pd.read_parquet(self.source(obj)) for obj in objects], rid, host, cell)
                if len(frame) != int(candidate["samples"]):
                    raise ValueError(f"Audit sample-count mismatch: {rid}/{host}/{cell}")
                paired = self.old_accel[(self.old_accel.run_id == rid) & (self.old_accel.hostname == host) & (self.old_accel.cell == cell)
                                       & (self.old_accel.t0 <= frame.timestamp_epoch.max()) & (self.old_accel.t1 >= frame.timestamp_epoch.min())]
                prior = self.old_nic[(self.old_nic.run_id == rid) & (self.old_nic.hostname == host)]
                cleaned, counts = clean_nic(frame, list(zip(paired.t0, paired.t1)), list(zip(prior.t0, prior.t1)))
                accepted = nic_usable(cleaned)
                record = dict(kind=kind, run_id=rid, hostname=host, cell=cell,
                              source_hashes=sorted(obj["sha256"] for obj in objects), raw=time_stats(frame), **counts,
                              paired_jobs=sorted(set(paired.job_id)), paired_accelerator_paths=sorted(set(paired.paths)),
                              disposition="accepted_paired_sidecar" if accepted else "quarantined_no_accepted_pair" if paired.empty else "quarantined_insufficient_usable_overlap")
                if accepted:
                    keys = {(rid, host, int(s)) for s in cleaned.timestamp_epoch.round()}
                    if keys & join_keys:
                        raise ValueError(f"Ambiguous NIC join keys: {rid}/{host}")
                    join_keys.update(keys)
                    name = Path(objects[0]["name"]).name.removesuffix(".partial")
                    if kind == "checkpoint":
                        name = name.removesuffix(".parquet") + "_truncated.parquet"
                    if not name.startswith("nic_") or not name.endswith(".parquet"):
                        raise ValueError(f"Invalid NIC filename: {name}")
                    record["path"] = self.save_frame(DEST / "nic" / rid / host / name, cleaned)
                    record["recovered"] = time_stats(cleaned)
                self.nic.append(record)

    def recover_tpu(self):
        table = pd.read_csv(self.audit / "new_capture_dispositions.csv", keep_default_na=False)
        candidates = table[table.run_id.isin(TPU_RUNS)]
        if len(candidates) != 8 or not candidates.disposition.eq("recover_candidate_with_temporally_matched_steps").all():
            raise ValueError("Unexpected TPU candidate selection")
        for rid, group in candidates.groupby("run_id"):
            markers, log_objects, evidence = {}, {}, {}
            for worker in (0, 1000, 2000, 3000):
                prefix = f"tpu/gap_20260903_221548_ms4/runs/{rid}/w{worker}/"
                done = [obj for name, obj in self.by_name.items() if name.startswith(prefix) and name.endswith(".done")]
                logs = [obj for name, obj in self.by_name.items() if name.startswith(prefix) and name.endswith(".log")]
                if len(done) != 1 or len(logs) != 1:
                    raise ValueError(f"Missing/ambiguous worker evidence: {rid}/w{worker}")
                markers[worker] = self.source(done[0]).read_text()
                log_objects[worker] = logs[0]
                log = self.source(logs[0])
                evidence[worker] = inspect_log(log)
                ev = evidence[worker]
                if ev["run_ids"] != [rid] or set(ev["step_kinds"]) != {"train"} or ev["steps"] < 2:
                    raise ValueError(f"Missing training STEP evidence: {rid}/w{worker}")
                if worker == 0 and "on 4 chips (4 hosts)" not in log.read_text():
                    raise ValueError(f"Missing declared four-chip/four-host topology: {rid}/w{worker}")
                logged_workers = set(re.findall(r"^STEP .*\bworker=(\d+)\b", log.read_text(), re.M))
                # Multislice collector IDs are slice*1000 + local worker;
                # these four one-host slices log JAX process_index 0..3.
                if logged_workers != {str(worker // 1000)}:
                    raise ValueError(f"STEP worker mismatch: {rid}/w{worker}: {logged_workers}")
            completed, codes = tpu_completion(markers, rid)
            # The workload emits its summary only from JAX process 0.
            if completed and "TPU MN-T complete:" not in self.source(log_objects[0]).read_text():
                raise ValueError(f"No training completion summary: {rid}")
            hosts, chips = set(), set()
            for candidate in group.to_dict("records"):
                objects = [self.by_hash[h][0] for h in candidate["hashes"].split("|")]
                if len(objects) != 1:
                    raise ValueError(f"Unexpected TPU payload versions: {rid}")
                obj = objects[0]
                frame = merge_stream([pd.read_parquet(self.source(obj))], rid, candidate["hostname"], candidate["cell"])
                worker = int(re.search(r"/w(\d+)/", obj["name"])[1])
                if set(frame.worker_id.astype(str)) != {str(worker)} or set(frame.chip_index.astype(str)) != {"0"}:
                    raise ValueError(f"TPU chip identity mismatch: {obj['name']}")
                if set(frame.gpu_uuid) != {candidate["gpu_uuid"]}:
                    raise ValueError(f"TPU UUID mismatch: {obj['name']}")
                hosts.add(candidate["hostname"])
                chips.add(candidate["gpu_uuid"])
                ev = evidence[worker]
                cleaned = frame.loc[frame.timestamp_epoch.between(ev["first_step"], ev["last_step"])].copy()
                if len(cleaned) < 60 or not cleaned.tpu_metrics_live.eq(1).all() or cleaned.slice_error_detected.gt(0).any():
                    raise ValueError(f"Unusable TPU telemetry during STEP interval: {rid}/w{worker}")
                # Persist the declared width from the launcher, so completeness
                # checks do not infer topology from whichever files are present.
                cleaned["expected_ranks"] = 4
                subdir = "tpu" if completed else "_quarantine_failed/tpu"
                path = self.save_frame(DEST / subdir / rid / f"w{worker}" / Path(obj["name"]).name, cleaned)
                self.tpu.append(dict(run_id=rid, worker=worker, hostname=candidate["hostname"], gpu_uuid=candidate["gpu_uuid"],
                                     cell=candidate["cell"], hardware=candidate["hardware"], source_hashes=[obj["sha256"]], path=path,
                                     raw=time_stats(frame), recovered=time_stats(cleaned), step_evidence=ev, exit_codes=codes,
                                     disposition="accepted_completed_training" if completed else "quarantined_failed_job_training_prefix"))
            if len(hosts) != 4 or len(chips) != 4:
                raise ValueError(f"Expected four TPU hosts/chips: {rid}")
            for worker, obj in log_objects.items():
                self.export_log(obj, evidence[worker])

    def export_log(self, obj, evidence):
        uri = f"gs://{obj['bucket']}/{obj['name']}"
        rel = Path("objects") / relative_path(uri)
        rel = rel.with_suffix(rel.suffix + ".gz")
        export = self.staging / "_log_export"
        path = export / rel
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as dest:
            with gzip.GzipFile(filename="", fileobj=dest, mode="wb", mtime=0) as packed:
                packed.write(self.source(obj).read_bytes())
        digest = sha256(path)
        target = self.repo / "logs/gcs" / relative_path(uri)
        target = target.with_suffix(target.suffix + ".gz")
        same_or_absent(target, digest)
        self.logs.append(dict(uri=uri, fetched_utc="2026-10-12", generation=obj["generation"],
                              source_sha256=obj["sha256"], source_bytes=int(obj["size"]), **evidence,
                              local_path=str(rel), compressed_sha256=digest, compressed_bytes=path.stat().st_size))

    def finish(self, apply):
        selection_commit = subprocess.check_output(["git", "-C", str(self.repo), "rev-parse", "HEAD"], text=True).strip()
        export_manifest = dict(schema="work-log-export-v1", selection_commit=selection_commit, objects=self.logs, failures=[])
        export = self.staging / "_log_export"
        (export / "manifest.json").write_text(json.dumps(export_manifest, indent=2) + "\n")
        accepted_nic = [r for r in self.nic if r["disposition"] == "accepted_paired_sidecar"]
        accepted_tpu = [r for r in self.tpu if r["disposition"] == "accepted_completed_training"]
        failed_tpu = [r for r in self.tpu if r not in accepted_tpu]
        summary = dict(nic_dispositions=dict(Counter(r["disposition"] for r in self.nic)),
                       nic_source_kinds=dict(Counter(r["kind"] for r in self.nic)),
                       nic_accepted_kinds=dict(Counter(r["kind"] for r in accepted_nic)),
                       nic_raw_samples=sum(r["raw"]["samples"] for r in self.nic),
                       nic_accepted_samples=sum(r["recovered"]["samples"] for r in accepted_nic),
                       nic_accepted_host_hours=sum(r["recovered"]["gap_capped_s"] for r in accepted_nic) / 3600,
                       nic_paired_jobs=len({j for r in accepted_nic for j in r["paired_jobs"]}),
                       tpu_accepted_streams=len(accepted_tpu), tpu_quarantined_streams=len(failed_tpu),
                       tpu_accepted_samples=sum(r["recovered"]["samples"] for r in accepted_tpu),
                       tpu_accepted_accelerator_hours=sum(r["recovered"]["gap_capped_s"] for r in accepted_tpu) / 3600,
                       tpu_quarantined_samples=sum(r["recovered"]["samples"] for r in failed_tpu),
                       tpu_quarantined_accelerator_hours=sum(r["recovered"]["gap_capped_s"] for r in failed_tpu) / 3600,
                       recovered_work_logs=len(self.logs), source_payloads=len(self.sources),
                       output_bytes=sum(r["bytes"] for r in self.files.values()) + sum(r["compressed_bytes"] for r in self.logs))
        inputs = [self.baseline / "streams.csv", self.repo / "data/quality_exclusions.json"]
        inputs += [self.audit / n for n in ("additional_nic_checkpoints.csv", "additional_nic_streams.csv", "new_capture_dispositions.csv")]
        manifest = dict(schema="audited-telemetry-recovery-v1", recovered_date="2026-10-12", selection_commit=selection_commit,
                        policies=dict(nic_minimum_contiguous_span_s=MIN_NIC_SPAN, nic_units="MiB/s despite *_mbps names",
                                      nic_pairing="same run, hostname, canonical cell; within prior accepted accelerator time spans; quality exclusions applied",
                                      nic_rounding="one sample per rounded second, closest to integer; original bytes preserved",
                                      tpu_interval="first through last training STEP per worker, inclusive",
                                      tpu_acceptance="all four worker rc=0, rank-0 completion summary, live telemetry; failed jobs quarantined"),
                        inputs=[dict(path=str(p), sha256=sha256(p)) for p in inputs],
                        script_sha256=sha256(Path(__file__)), summary=summary,
                        nic=self.nic, tpu=self.tpu, sources=sorted(self.sources.values(), key=lambda r: r["sha256"]),
                        files=sorted(self.files.values(), key=lambda r: r["path"]), log_export=export_manifest)
        self.stage_bytes(DEST / "manifest.json", (json.dumps(manifest, indent=2) + "\n").encode())
        # All source hashes, identities and destination conflicts are checked before writes.
        if apply:
            # The importer independently validates the entire log export before writing.
            import_export(export, self.repo)
            for relative in self.files:
                target = self.repo / relative
                if target.exists():
                    continue
                target.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(self.staging / relative, target)
        print(json.dumps(dict(applied=apply, **summary), indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--audit", type=Path)
    parser.add_argument("--cache", type=Path, default=Path("/tmp/gcs-workload-check-2026-10-12"))
    parser.add_argument("--baseline", type=Path, default=Path("/home/robi/workload-audit-2026-10-11"))
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    audit = args.audit or args.repo / "docs/workload_review_2026-10-12/gcs_reconciliation"
    with tempfile.TemporaryDirectory(prefix="telemetry-recovery-") as staging:
        recovery = Recovery(args.repo, audit, args.cache, args.baseline, staging)
        recovery.recover_nic()
        recovery.recover_tpu()
        recovery.finish(args.apply)


if __name__ == "__main__":
    main()
