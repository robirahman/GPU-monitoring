"""Focused integration validation; does not require materializing the full LFS corpus."""
from collections import Counter
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from classifier import threeway_improved as features
from classifier.data_sources import accelerator_files, pointer, quarantined, sha256
from classifier.multinode_features import (
    assign_stream_and_job_ids, attach_nic_sidecar, normalize_comm_columns, sliding_windows_multinode,
)
from classifier.window_labels import build_work_index, default_log_paths, job_work_intervals


def main():
    dest = ROOT / "data/gcs_recovery_20261012"
    manifest = json.loads((dest / "manifest.json").read_text())
    for record in manifest["files"]:
        path = ROOT / record["path"]
        assert path.stat().st_size == record["bytes"], path
        assert sha256(path) == record["sha256"], path

    baseline = Path("/home/robi/workload-audit-2026-10-11")
    old_files = pd.read_csv(baseline / "files.csv", keep_default_na=False).set_index("path")
    cache = Path("/tmp/gcs-workload-check-2026-10-12")
    cached = {}
    for name in ("raw-download-candidates-downloaded.json", "unverified-raw-candidates-downloaded.json", "downloaded-so-far.json"):
        for record in json.loads((cache / name).read_text()):
            if "sha256" in record:
                cached[record["sha256"]] = record["path"]

    nic_records = [r for r in manifest["nic"] if r["disposition"] == "accepted_paired_sidecar"]
    nic_frames, checked_gpu_paths, join_candidates = [], set(), {}
    for record in nic_records:
        frame = pd.read_parquet(ROOT / record["path"])
        assert len(frame) == record["recovered"]["samples"]
        assert set(frame.run_id) == {record["run_id"]}
        assert set(frame.hostname) == {record["hostname"]}
        nic_frames.append(frame)
        for path_group in record["paired_accelerator_paths"]:
            for path in path_group.split("|"):
                if path in checked_gpu_paths:
                    continue
                checked_gpu_paths.add(path)
                expected = old_files.loc[path, "oid"]
                current = pointer(ROOT / path)
                assert (current[0] if current else sha256(ROOT / path)) == expected, path
                if expected in cached:
                    join_candidates[path] = (expected, cached[expected])
    all_nic = pd.concat(nic_frames, ignore_index=True)
    keys = all_nic[["run_id", "hostname"]].assign(second=all_nic.timestamp_epoch.round())
    assert not keys.duplicated().any()
    assert len(all_nic) == manifest["summary"]["nic_accepted_samples"]

    # Exercise the real merge against cached, checksum-matched GPU payloads,
    # selecting one available accelerator stream per recovered NIC identity.
    selected, frames = set(), []
    for path, (digest, payload) in sorted(join_candidates.items()):
        df = pd.read_parquet(payload)
        identity = (str(df.run_id.iloc[0]), str(df.hostname.iloc[0]))
        if identity in selected:
            continue
        assert sha256(payload) == digest
        selected.add(identity)
        frames.append(features._normalize_columns(df).assign(source_path=path))
    assert frames, "No real paired GPU payloads available for integration validation"
    gpu = pd.concat(frames, ignore_index=True)
    joined = attach_nic_sidecar(gpu, str(dest / "nic/**/nic_*.parquet"))
    assert len(joined) == len(gpu), "NIC merge multiplied GPU samples"
    assert joined.nic_total_rx_mbps.notna().any()

    # Isolate the recovered directory from the unmaterialized historical LFS
    # corpus, while using the production loader and quarantine policy unchanged.
    with tempfile.TemporaryDirectory(prefix="validate-recovered-tpu-") as temp:
        isolated = Path(temp)
        (isolated / "data").symlink_to(dest, target_is_directory=True)
        with patch.object(features, "PROJECT_ROOT", isolated):
            tpu = features.load_all_data()
        discovered = accelerator_files(isolated)
        assert len(discovered) == 4
        assert not any(quarantined(path) for path in discovered)
        assert set(tpu.run_id) == {"ad5e56af3e3b"}
        assert len(tpu) == 2232 and tpu.gpu_uuid.nunique() == 4 and tpu.hostname.nunique() == 4
        assert tpu.expected_ranks.eq(4).all()
        tpu = assign_stream_and_job_ids(normalize_comm_columns(tpu))
        assert tpu.job_id.nunique() == 1 and tpu.stream_id.nunique() == 4
        logs = [p for p in default_log_paths(ROOT) if any(rid in str(p) for rid in ("ad5e56af3e3b", "cdc1669c2763"))]
        assert len(logs) == 8
        work = job_work_intervals(tpu, build_work_index(logs))
        windows = sliding_windows_multinode(tpu, work_events=work)
        assert len(windows) > 0
        assert windows.threeway_label.eq("ml_training").all()
        assert windows.label_source.eq("steps").all()
        assert windows.meta_expected_ranks.eq(4).all() and windows.meta_n_hosts.eq(4).all()
        assert windows.meta_rank_coverage.eq(1).all() and not windows.meta_degraded.any()

    summary = dict(verified_output_files=len(manifest["files"]), verified_bytes=sum(r["bytes"] for r in manifest["files"]),
                   nic_accepted_streams=len(nic_records), nic_samples=len(all_nic), duplicate_nic_join_keys=0,
                   referenced_gpu_files_unchanged=len(checked_gpu_paths),
                   real_gpu_streams_tested=len(selected), real_gpu_samples_tested=len(gpu),
                   real_gpu_samples_with_recovered_nic=int(joined.nic_total_rx_mbps.notna().sum()),
                   tpu_loader_streams=tpu.stream_id.nunique(), tpu_loader_jobs=tpu.job_id.nunique(),
                   tpu_loader_samples=len(tpu), tpu_imported_logs=len(logs),
                   tpu_30s_windows=len(windows), tpu_window_labels=dict(Counter(windows.threeway_label)),
                   tpu_all_windows_four_hosts_four_ranks=True,
                   scope="Focused recovery validation; full historical LFS corpus not loaded; classifier not retrained")
    output = Path(__file__).with_name("validation.json")
    output.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
