"""Regressions for ground truth, physical identity, and paired generalization."""
import hashlib
import json
from pathlib import Path
import tempfile
import subprocess
import unittest
from unittest.mock import patch

import numpy as np
import pandas as pd

from classifier import threeway_improved as features
from classifier.data_sources import resolve_payload, accelerator_files, apply_quality_exclusions
from classifier.workload_labels import PHYSICAL_TRAINING, threeway_label, workload_family, workload_kind, validate_labels
from classifier.multinode_features import assign_stream_and_job_ids, sliding_windows_multinode, normalize_comm_columns
from classifier.rebuttal_analysis.paired_family_lofo import evaluate, fold_plan
from classifier.cache_provenance import provenance, stamp_cache, cache_matches
from classifier.collection_quality import assess_collection
from scripts.audit_data_inventory import time_summary, timestamp_series, inventory


class LabelTests(unittest.TestCase):
    def test_physical_truth_and_negative_controls(self):
        for label in PHYSICAL_TRAINING:
            for suffix in ("", "_rep2", "_rep2_node1_gpu0"):
                self.assertEqual(threeway_label(label + suffix), "ml_training")
        for label in ("s2_unknown", "s3_unknown", "dil_v999"):
            self.assertNotEqual(threeway_label(label), "ml_training")
        for label in ("s16_advtrain_filler_only", "tpuTMS4_advtrain_satfiller_only_rep2_w0c0"):
            self.assertEqual(threeway_label(label), "other")
            self.assertEqual(workload_kind(label), "control")
        self.assertEqual(threeway_label("s16_stalled_train_fsdp1b"), "ml_training")
        self.assertEqual(workload_kind("ev2b_baseline"), "control")
        with self.assertRaises(ValueError):
            validate_labels(["completely_unregistered_workload"])

    def test_semantics_win_over_model_name(self):
        self.assertEqual(threeway_label("s16_infer_gpt2"), "ml_inference")
        self.assertEqual(threeway_label("s16_nonml_trainbench"), "other")
        self.assertEqual(threeway_label("symgemm_L3_graddiscard_rep2"), "ml_inference")
        self.assertEqual(threeway_label("whitebox_lora_7b_long"), "ml_training")
        for label, expected in (("EC1_phantom_gpu0", "ml_inference"),
                                ("EC4_mining_gpu1", "ml_inference"),
                                ("EC5_frozen_gpu0", "ml_training"),
                                ("EC8_mem_idle_gpu1", "other"),
                                ("resnet18_small_batch_batch8_H100", "ml_training")):
            self.assertEqual(threeway_label(label), expected)

    def test_implementation_siblings_and_scopes(self):
        for a, b in (("s16_train_pipe7b", "s8_advtrain_pipe1b_mb1"),
                     ("s16_inferserve_bursty_7b", "s16_inferserve_decode_72b"),
                     ("tpuT1_train_fsdp1b_rep2_w0c0", "s16_train_fsdp70b"),
                     ("s16_train_accum64_fsdp7b", "s16_trainmem_fsdp7b_nockpt"),
                     ("s16_advtrain_interleave_f05", "tpuTMS_advtrain_satfiller_only")):
            self.assertEqual(workload_family(a), workload_family(b))
        self.assertEqual(workload_family("s16_train_fsdp1b", "backbone"),
                         workload_family("s16_train_streamdiloco_f8", "backbone"))


class InputTests(unittest.TestCase):
    def test_external_payload_missing_incomplete_and_complete(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            raw = b"payload"
            oid = hashlib.sha256(raw).hexdigest()
            p = root / "data.parquet"
            p.write_text(f"version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(raw)}\n")
            with self.assertRaises(FileNotFoundError):
                resolve_payload(p, root / "objects")
            obj = root / "objects" / oid[:2] / oid[2:4] / oid
            obj.parent.mkdir(parents=True)
            obj.write_bytes(raw[:-1])
            with self.assertRaises(FileNotFoundError):
                resolve_payload(p, root / "objects")
            obj.write_bytes(raw)
            self.assertEqual(resolve_payload(p, root / "objects"), obj)

    def test_policy_excludes_all_quarantine_and_sensor_roles(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ("good.parquet", "quarantine_q4_slowstack/bad.parquet", "_quarantine_idle/bad.parquet",
                         "cell_bmc_abc.parquet", "cell_wall_power.parquet", "nic_cell.parquet", "cell_scope_abc.parquet"):
                p = root / "data" / name
                p.parent.mkdir(parents=True, exist_ok=True)
                p.touch()
            self.assertEqual([p.name for p in accelerator_files(root)], ["good.parquet"])

    def test_mixed_label_and_device_streams_are_not_stitched(self):
        frames = []
        for i, label in enumerate(("idle", "pytorch_resnet_cifar10")):
            f = pd.DataFrame(dict(run_id="same", gpu_uuid="same", gpu_name="test",
                                  workload_label=label, ts=np.arange(10) + 1000 + i * 10))
            f["threeway_label"] = threeway_label(label)
            frames.append(features._normalize_columns(f))
        d = pd.concat(frames)
        ids = assign_stream_and_job_ids(d)
        self.assertEqual(ids.stream_id.nunique(), 2)
        self.assertEqual(ids.job_id.nunique(), 2)
        for job, sub in ids.groupby("job_id"):
            again = assign_stream_and_job_ids(sub)
            self.assertEqual(set(again.job_id), {job})
            self.assertEqual(set(again.stream_id), set(sub.stream_id))
        w = features.sliding_windows(d, 5, 5)
        self.assertEqual(w.run_id.nunique(), 2)
        self.assertEqual(set(w.threeway_label), {"other", "ml_training"})

    def test_low_utilization_is_not_failure(self):
        log = "STEP ts=123.4 kind=train batch_idx=1\nMulti-GPU run complete: abc (rc=0)"
        self.assertEqual(assess_collection("train_sparse", 0.0, log)[0], "accepted")
        self.assertEqual(assess_collection("train_actor", 0.0, "Multi-GPU run complete: abc (rc=0)")[0], "accepted")
        self.assertEqual(assess_collection("train_sparse", 0.0)[0], "uncertain")
        self.assertEqual(assess_collection("train_sparse", 90.0, "Multi-GPU run complete: abc (rc=1)")[0], "failed")

    def test_quality_exclusion_does_not_drop_independent_peer_job(self):
        frame = pd.DataFrame(dict(run_id=["shared", "shared"], gpu_uuid=["gpu0", "gpu1"],
                                   hostname=["host0", "host1"], gpu_name="test", ts=[1000., 1000.],
                                   workload_label="s1_train_fsdp1b"))
        ids = assign_stream_and_job_ids(frame)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); (root / "data").mkdir()
            (root / "data/quality_exclusions.json").write_text(json.dumps({"excluded_jobs": [
                {"job_id": ids.loc[ids.hostname.eq("host0"), "job_id"].iloc[0], "reason": "failed"}]}))
            kept = apply_quality_exclusions(frame, root)
            self.assertEqual(list(kept.hostname), ["host1"])

    def test_cache_invalidation_by_data_code_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "data").mkdir(); (root / "classifier").mkdir()
            data = root / "data/test.parquet"; data.write_bytes(b"one")
            code = root / "classifier/features.py"; code.write_text("v1")
            m = provenance(root, window=30)
            artifact = root / "cache.parquet"; artifact.write_bytes(b"cache")
            stamp_cache(artifact, m)
            self.assertTrue(cache_matches(artifact, m))
            for path in (data, code):
                old = path.read_bytes(); path.write_bytes(old + b"changed")
                self.assertFalse(cache_matches(artifact, provenance(root, window=30)))
                path.write_bytes(old)
            self.assertFalse(cache_matches(artifact, provenance(root, window=60)))
            artifact.write_bytes(b"corrupt")
            self.assertFalse(cache_matches(artifact, m))


class MultinodeFeatureTests(unittest.TestCase):
    def frame(self):
        frames = []
        for rank in range(3):
            f = pd.DataFrame(dict(run_id="job", gpu_uuid=f"gpu{rank}", hostname=f"host{rank}",
                                  gpu_name="test", workload_label="train_fixture", threeway_label="ml_training",
                                  expected_ranks=3, ts=np.arange(40, dtype=float),
                                  gpu_utilization_pct=20 + 10 * np.sin(np.arange(40) / 4 + rank),
                                  power_draw_w=100 + rank * 10))
            frames.append(features._normalize_columns(f))
        return pd.concat(frames, ignore_index=True)

    def test_rank_permutation_invariance(self):
        d = self.frame()
        a = sliding_windows_multinode(d, 20, 20)
        d.gpu_uuid = d.gpu_uuid.map({"gpu0": "gpuZ", "gpu1": "gpuA", "gpu2": "gpuM"})
        b = sliding_windows_multinode(d.sample(frac=1, random_state=1), 20, 20)
        cols = features.get_feature_cols(a)
        np.testing.assert_allclose(a[cols], b[cols], equal_nan=True, atol=1e-9)

    def test_cache_workers_write_real_chunks(self):
        from scripts import build_mn_window_cache as mn, build_window_cache as per_stream
        d = assign_stream_and_job_ids(self.frame())
        with tempfile.TemporaryDirectory() as tmp:
            raw = str(Path(tmp) / "raw.parquet")
            d.to_parquet(raw, index=False)
            with patch.object(mn, "WORK", tmp), patch.object(mn, "RAW", raw):
                _, _, count, path = mn.job_chunk((1, d.job_id.unique(), 20., 20.))
                self.assertEqual(count, 2)
                self.assertEqual(len(pd.read_parquet(path)), 2)
            d["run_id"] = d.stream_id
            d.to_parquet(raw, index=False)
            with patch.object(per_stream, "WORK", tmp), patch.object(per_stream, "RAW", raw):
                _, _, count, path = per_stream.window_chunk((1, d.run_id.unique(), 20., 20.))
                self.assertEqual(count, 6)
                self.assertEqual(pd.read_parquet(path).run_id.nunique(), 3)

    def test_expected_width_and_duration_enforced(self):
        d = self.frame()
        self.assertGreater(len(sliding_windows_multinode(d, 20, 20)), 0)
        self.assertTrue(sliding_windows_multinode(d[d.gpu_uuid != "gpu2"], 20, 20).empty)
        fragment = d[(d.gpu_uuid != "gpu2") | (d.ts < 5)]
        self.assertTrue(sliding_windows_multinode(fragment, 20, 20).empty)
        diagnostic = sliding_windows_multinode(fragment, 20, 20, keep_degraded=True)
        self.assertTrue(diagnostic.meta_degraded.all())

    def test_missing_sensor_is_not_measured_zero(self):
        d = self.frame()
        absent = sliding_windows_multinode(d, 20, 20)
        measured = sliding_windows_multinode(d.assign(nic_total_mbps=0.), 20, 20)
        self.assertTrue(absent.xragg_nic_total_mbps_mean.isna().all())
        self.assertTrue(measured.xragg_nic_total_mbps_mean.eq(0).all())
        self.assertNotIn("meta_n_ranks", features.get_feature_cols(measured))
        x = normalize_comm_columns(pd.DataFrame({"pcie_tx_mbps": [-1., 0.], "pcie_rx_mbps": [-1., 0.]}))
        self.assertTrue(np.isnan(x.pcie_total_mbps.iloc[0]))
        self.assertEqual(x.pcie_total_mbps.iloc[1], 0)
        nic = normalize_comm_columns(pd.DataFrame(dict(nic_tx_mbps=[1.], nic_rx_mbps=[2.],
                                                       nic_total_tx_mbps=[10.], nic_total_rx_mbps=[20.])))
        self.assertEqual(nic.nic_total_mbps.iloc[0], 30)
        no_power = features._normalize_columns(pd.DataFrame({
            "gpu_utilization_pct": 20 + np.sin(np.arange(30)), "ts": np.arange(30.)}))
        spectral = features.extract_enhanced_features(no_power)
        self.assertTrue(np.isnan(spectral["power_draw_w_mean"]))
        self.assertTrue(np.isfinite(spectral["gpu_util_fft_peak_power"]))


class EvaluationTests(unittest.TestCase):
    def frame(self):
        rows = []
        for cls, families in (("ml_training", ["train_a", "train_b", "train_c"]),
                               ("ml_inference", ["infer_a", "infer_b", "infer_c"]),
                               ("other", ["nonml_a", "nonml_b", "nonml_c"])):
            for family in families:
                for rep in ("old", "new"):
                    for t in (0, 15):
                        rows.append(dict(grp=family, workload_label=family, job_id=family + rep,
                                         run_id=family + rep, threeway_label=cls, window_start=t,
                                         window_end=t + 30, signal_mean=float(t), signal_std=float(t + 1),
                                         meta_n_hosts=2, y=999, expected_ranks=16))
        return pd.DataFrame(rows)

    def test_exhaustive_and_paired_training_only_ablation(self):
        d = self.frame()
        new_jobs = set(d.loc[d.job_id.str.endswith("new"), "job_id"])
        result, pred, splits = evaluate(d, new_jobs, trees=2)
        self.assertEqual(len(pred[pred.arm == "full"]), len(d))
        self.assertEqual(pred[pred.arm == "full"].row_id.nunique(), len(d))
        self.assertNotIn("y", result["features"])
        self.assertNotIn("expected_ranks", result["features"])
        self.assertEqual(result["new_jobs"]["jobs"], len(new_jobs))
        self.assertEqual(result["paired_old_jobs"]["windows"], result["baseline_old_jobs"]["windows"])
        self.assertEqual(result["all"]["windows"],
                         result["paired_old_jobs"]["windows"] + result["new_jobs"]["windows"])
        for fam in result["families"]:
            full, base = [s for s in splits if s["family"] == fam]
            self.assertEqual(set(base["train_jobs"]), set(full["train_jobs"]) - new_jobs)
            p = pred[(pred.family == fam) & pred.paired_test]
            self.assertEqual(set(p[p.arm == "full"].row_id), set(p[p.arm == "baseline"].row_id))
            self.assertFalse(set(full["test_ids"]) & set(full["train_ids"]))

    def test_absent_ablation_jobs_and_single_class_folds_fail(self):
        with self.assertRaises(ValueError):
            fold_plan(self.frame(), {"missing-job"})
        d = self.frame(); d = d[~d.grp.isin(["train_b", "train_c"])]
        with self.assertRaises(ValueError):
            evaluate(d, trees=2)


class InventoryTests(unittest.TestCase):
    def test_inventory_includes_legacy_json_without_double_counting_copies(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "repo"; (root / "data").mkdir(parents=True)
            records = [dict(ts=1700000000. + t, run_id="fixture", gpu_uuid="gpu",
                            gpu_name="test", workload_label="dil_v1") for t in range(6)]
            p = root / "data/first.parquet"
            pd.DataFrame(records).to_parquet(p, index=False)
            (root / "data/copy.parquet").write_bytes(p.read_bytes())
            edge = root / "week4/results/edge_cases"; edge.mkdir(parents=True)
            (edge / "BASELINE_TRAIN_telemetry.json").write_text(json.dumps([
                dict(t=1700000000. + t, gpu=gpu) for t in range(3) for gpu in range(2)]))
            np.save(root / "data/wave.npy", np.arange(4))
            for args in (["init", "-q"], ["add", "."], ["-c", "user.name=Fixture", "-c",
                         "user.email=fixture@example.invalid", "commit", "-qm", "fixture"]):
                subprocess.run(["git", "-C", str(root), *args], check=True, capture_output=True)
            result = inventory(root, Path(tmp) / "objects", Path(tmp) / "out")
            self.assertEqual(result["accelerator_samples"], 12)
            self.assertEqual(result["accepted_accelerator_streams"], 3)
            self.assertEqual(result["telemetry_json_paths"], 1)

    def test_duplicates_gaps_and_iso_timestamps(self):
        s = time_summary([1000, 1000, 1001, 1002, 1100, 1101])
        self.assertEqual(s["samples"], 5)
        self.assertEqual(s["span_s"], 101)
        self.assertEqual(s["gap_capped_s"], 3)
        t = timestamp_series(pd.DataFrame({"timestamp_utc": ["2026-01-01T00:00:00Z"]}))
        self.assertGreater(t.iloc[0], 1e9)
        dates = pd.to_datetime(["2026-01-01T00:00:00Z", None])
        normalized = features._normalize_columns(pd.DataFrame({"timestamp": dates}))
        self.assertEqual(normalized.ts.iloc[0], 1767225600.)
        self.assertTrue(np.isnan(normalized.ts.iloc[1]))


if __name__ == "__main__":
    unittest.main()
