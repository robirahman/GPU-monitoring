"""Regressions for ground truth, physical identity, and paired generalization."""
import gzip
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
        # No gradient step ever happens in these jobs, so none of their windows
        # is training, whatever is resident or however the job is named.
        for label in ("s16_stalled_train_fsdp1b", "s8_stalled_train_fsdp1b_node1_gpu3",
                      "s1_stalled_train_fsdp1b", "smoke_s16_stalled_train_fsdp1b_rep2",
                      "s16_advtrain_filler_only", "s8_advtrain_satfiller_only_node0_gpu2"):
            self.assertEqual(threeway_label(label), "other")
            self.assertEqual(workload_kind(label), "control")
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
        self.assertEqual(w.stream_id.nunique(), 2)
        self.assertEqual(set(w.run_id), {"same"})
        self.assertEqual(w.groupby("stream_id").threeway_label.nunique().max(), 1)
        self.assertEqual(set(w.threeway_label), {"other", "ml_training"})

    def test_sibling_ranks_share_a_cv_group(self):
        frames = []
        for host, gpu in (("host0", "gpu0"), ("host0", "gpu1"), ("host1", "gpu0"), ("host1", "gpu1")):
            frames.append(features._normalize_columns(pd.DataFrame(dict(
                run_id="job42", gpu_uuid=gpu, hostname=host, gpu_name="test",
                workload_label="s16_train_fsdp1b", threeway_label="ml_training",
                ts=np.arange(40, dtype=float) + 1000, gpu_utilization_pct=50.))))
        d = pd.concat(frames, ignore_index=True)
        w = features.sliding_windows(d, 20, 20)
        # Per-GPU features, never stitched across hosts with the same device name...
        self.assertEqual(w.stream_id.nunique(), 4)
        # ...but one run-level fold key, so ranks of one job share a CV fold.
        self.assertEqual(set(w.run_id), {"job42"})
        ided = features.sliding_windows(assign_stream_and_job_ids(d), 20, 20)
        self.assertEqual(ided.job_id.nunique(), 1)
        self.assertEqual(ided.stream_id.nunique(), 4)
        for col in ("run_id", "stream_id", "job_id"):
            self.assertNotIn(col, features.get_feature_cols(ided))

    def test_low_utilization_is_not_failure(self):
        log = "STEP ts=123.4 kind=train batch_idx=1\nMulti-GPU run complete: abc (rc=0)"
        self.assertEqual(assess_collection("train_sparse", 0.0, log)[0], "accepted")
        self.assertEqual(assess_collection("train_actor", 0.0, "Multi-GPU run complete: abc (rc=0)")[0], "accepted")
        self.assertEqual(assess_collection("train_sparse", 0.0)[0], "uncertain")
        self.assertEqual(assess_collection("train_sparse", 90.0, "Multi-GPU run complete: abc (rc=1)")[0], "failed")

    def test_parallel_replicas_are_judged_by_their_own_status(self):
        log = ("GPU 0: llm_infer_x FAIL(1) -> s16_inferrep_x_node0_gpu0_NVIDIA_T_aaaaaaaaaaaa.parquet\n"
               "GPU 1: llm_infer_x OK -> s16_inferrep_x_node0_gpu1_NVIDIA_T_bbbbbbbbbbbb.parquet\n"
               "Parallel run complete: [('llm_infer_x', 'FAIL(1)'), ('llm_infer_x', 'OK')]\n"
               "1/2 replica(s) failed")
        failed = assess_collection("s16_inferrep_x", 95.0, log,
                                   filename="s16_inferrep_x_node0_gpu0_NVIDIA_T_aaaaaaaaaaaa.parquet")
        ok = assess_collection("s16_inferrep_x", 0.0, log,
                               filename="s16_inferrep_x_node0_gpu1_NVIDIA_T_bbbbbbbbbbbb.parquet")
        self.assertEqual(failed[0], "failed")
        self.assertEqual(ok[0], "accepted")

    def test_telemetry_only_streams_take_the_heads_exit(self):
        from classifier.collection_quality import evidence_logs, launcher_logs
        rid = "abcdef123456"
        tel = (f"Telemetry-only: s16_infershard_x, 8 GPUs, node=1, run_id={rid}, dur=600s\n"
               f"GPU 0: saved s16_infershard_x_node1_gpu0_NVIDIA_T_{rid}.parquet\n"
               f"Telemetry-only run complete: {rid} (rc=0)")
        # The collector's own exit 0 is not evidence the remote workload ran.
        self.assertEqual(assess_collection("s16_infershard_x", 0.0, tel)[0], "uncertain")
        name = f"s16_infershard_x_node1_gpu0_NVIDIA_T_{rid}.parquet"
        for rc, want in ((0, "accepted"), (1, "failed")):
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                for host, stage, text in (
                        ("head", f"s16_infershard_x_node0_{rid}", f"Multi-GPU run: s16_infershard_x, 8 GPUs, "
                                                                  f"node=0, run_id={rid}\nMulti-GPU run complete: {rid} (rc={rc})"),
                        ("peer", f"s16_infershard_x_node1_{rid}", tel),
                        ("peer", f"s16_infershard_x_node1_{rid}.nic", "unrelated"),
                        # The master log names files too, next to other cells' failures.
                        ("peer", "queue_20261011", f"GPU 0: saved {name}\n"
                                                   "Multi-GPU run complete: 999999999999 (rc=1)")):
                    (root / host / "logs").mkdir(parents=True, exist_ok=True)
                    (root / host / "logs" / f"{stage}.log").write_text(text)
                used = evidence_logs(launcher_logs(root), root / "peer", name, [rid])
                self.assertEqual([log["host"].name for log in used], ["head"])
                text = "\n".join(log["text"] for log in used)
                self.assertEqual(assess_collection("s16_infershard_x", 0.0, text, filename=name)[0], want)

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
        from classifier.window_labels import save_work_events
        d = assign_stream_and_job_ids(self.frame())
        # The job trains for its first 20 s, then runs filler: two 20 s windows.
        slots = pd.DataFrame(dict(start=np.arange(0., 40., 2.), end=np.arange(2., 42., 2.),
                                  work=["ml_training"] * 10 + ["other"] * 10, node=0, rank=0))
        with tempfile.TemporaryDirectory() as tmp:
            raw, events = str(Path(tmp) / "raw.parquet"), str(Path(tmp) / "events.parquet")
            d.to_parquet(raw, index=False)
            save_work_events({str(d.job_id.iloc[0]): slots}, events)
            with patch.object(mn, "WORK", tmp), patch.object(mn, "RAW", raw):
                _, _, count, path = mn.job_chunk((1, d.job_id.unique(), 20., 20., events))
                self.assertEqual(count, 2)
                pooled = pd.read_parquet(path).sort_values("window_start")
                self.assertEqual(list(pooled.threeway_label), ["ml_training", "other"])
                self.assertEqual(set(pooled.label_source), {"steps"})
                self.assertEqual(set(pooled.job_label), {"ml_training"})
            d["run_id"] = d.stream_id
            d.to_parquet(raw, index=False)
            with patch.object(per_stream, "WORK", tmp), patch.object(per_stream, "RAW", raw):
                _, _, count, path = per_stream.window_chunk((1, d.run_id.unique(), 20., 20., events))
                self.assertEqual(count, 6)
                streams = pd.read_parquet(path)
                self.assertEqual(streams.run_id.nunique(), 3)
                self.assertEqual(streams.threeway_label.value_counts().to_dict(),
                                 {"ml_training": 3, "other": 3})

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


class WorkLabelTests(unittest.TestCase):
    """Windows are labeled by the work done inside them, read from launcher logs."""

    @staticmethod
    def write_log(root, name, lines):
        path = Path(root) / "logs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("\n".join(lines) + "\n")
        return path

    def test_a_step_line_reports_the_work_since_the_previous_line(self):
        from classifier.window_labels import work_intervals, label_windows
        # 2 s steps, then a 10-minute parameter sync that ends in a training
        # step (local SGD over a shaped link), then filler slots.
        ts = [100., 102., 104., 704., 706., 708.]
        work = ["ml_training"] * 4 + ["other"] * 2
        iv = work_intervals(pd.DataFrame(dict(ts=ts, work=work, node=0, rank=0, log="a.log")))
        lab = label_windows([60., 98., 400., 704.5], [90., 128., 430., 734.5], iv)
        self.assertEqual(list(lab["label"]), ["other", "ml_training", "ml_training", "other"])
        self.assertEqual(lab["train_s"][0], 0)                 # launch and load are not work

    def test_filler_logged_as_infer_is_not_inference(self):
        from classifier.window_labels import parse_work_log
        with tempfile.TemporaryDirectory() as tmp:
            body = ["Run ID: abcdef123456", "STEP ts=10.0 kind=train batch_idx=1",
                    "STEP ts=11.0 kind=infer batch_idx=2"]
            mni = self.write_log(tmp, "s__h0__mni_node0_abcdef123456.log",
                                 ["INFO:   Launching: python workloads/adversarial_multinode.py "
                                  "--strategy MN-I --train-frac 0.05"] + body)
            mno = self.write_log(tmp, "s__h0__mno_node0_abcdef123456.log",
                                 ["INFO:   Launching: python workloads/adversarial_multinode.py "
                                  "--strategy MN-O"] + body)
            self.assertEqual(list(parse_work_log(mni)["events"].work), ["ml_training", "other"])
            self.assertEqual(list(parse_work_log(mno)["events"].work), ["ml_training", "ml_inference"])

    def test_jobs_take_their_own_logs_and_never_bridge_executions(self):
        from classifier.window_labels import build_work_index, default_log_paths, job_work_intervals
        rid = "abcdef123456"
        with tempfile.TemporaryDirectory() as tmp:
            # An s8 cell: each node runs its own execution under one run_id.
            self.write_log(tmp, f"s__host-a__s8_train_x_node0_{rid}.log",
                           [f"Run ID: {rid}"] + [f"STEP ts={t}.0 kind=train" for t in (100, 101, 102)])
            with gzip.open(Path(tmp) / f"logs/s__host-b__s8_train_x_node1_{rid}.log.gz", "wt") as f:
                f.write(f"Run ID: {rid}\n" + "\n".join(f"STEP ts={t}.0 kind=infer" for t in (100, 101, 102)))
            index = build_work_index(default_log_paths(tmp))
            frame = pd.DataFrame(dict(job_id=["j0", "j1"], run_id=rid, hostname=["host-a", "host-b"],
                                      workload_label="s8_train_x",
                                      source_path=[f"data/s8_train_x_node0_gpu0_NVIDIA_T_{rid}.parquet",
                                                   f"data/s8_train_x_node1_gpu0_NVIDIA_T_{rid}.parquet"]))
            events = job_work_intervals(frame, index)
            self.assertEqual(set(events["j0"].work), {"ml_training"})
            self.assertEqual(set(events["j1"].work), {"ml_inference"})
            # A smoke run reusing the run_id hours later starts its own timeline.
            self.write_log(tmp, f"s__host-a__smoke_s8_train_x_node0_{rid}.log",
                           [f"Run ID: {rid}"] + [f"STEP ts={t}.0 kind=train" for t in (9000, 9001)])
            events = job_work_intervals(frame, build_work_index(default_log_paths(tmp)))
            self.assertLess((events["j0"].end - events["j0"].start).max(), 10)

    def test_each_gpu_shows_its_own_rank_when_ranks_log(self):
        from classifier.window_labels import own_intervals, work_intervals
        lines = [dict(ts=t + 0.1 * r, work="ml_training" if r == 0 else "ml_inference",
                      node=0, rank=r, log="a.log") for r in range(4) for t in range(100, 130)]
        iv = work_intervals(pd.DataFrame(lines))
        learner = own_intervals(iv, "data/s16_train_impala_node0_gpu0_NVIDIA_T_abcdef123456.parquet")
        actor = own_intervals(iv, "data/s16_train_impala_node0_gpu2_NVIDIA_T_abcdef123456.parquet")
        self.assertEqual(set(learner.work), {"ml_training"})
        self.assertEqual(set(actor.work), {"ml_inference"})
        rank0_only = iv[iv["rank"] == 0]
        self.assertIsNone(own_intervals(rank0_only, "data/x_node0_gpu3_NVIDIA_T_abcdef123456.parquet"))

    def test_without_logs_only_single_kind_programs_get_labels(self):
        from classifier.window_labels import assign_work_labels
        for label, want, source in (("s16_inferserve_poisson_7b", "ml_inference", "program"),
                                    ("s16_nonml_hpc", "other", "program"),
                                    ("s16_stalled_train_fsdp1b", "other", "program"),
                                    ("s16_train_fsdp7b", "ml_training", "program"),
                                    ("dilution_sweep_N1", "ml_training", "program"),
                                    ("dilution_sweep_N10", "unresolved", "unresolved"),
                                    ("s16_advtrain_interleave_f05", "unresolved", "unresolved"),
                                    ("s16_train_impala_a15", "unresolved", "unresolved")):
            rows = [dict(window_start=0., window_end=30.)]
            assign_work_labels(rows, None, label, threeway_label(label))
            self.assertEqual((rows[0]["threeway_label"], rows[0]["label_source"]), (want, source), label)

    def test_label_columns_never_become_features(self):
        d = MultinodeFeatureTests().frame()
        slots = pd.DataFrame(dict(start=[-2.], end=[40.], work=["ml_training"], node=0, rank=0))
        w = sliding_windows_multinode(d, 20, 20, work_events={str(assign_stream_and_job_ids(d).job_id.iloc[0]): slots})
        cols = features.get_feature_cols(w)
        self.assertTrue({"label_train_s", "label_infer_s", "label_other_s"} <= set(w))
        self.assertFalse([c for c in cols if "label" in c])


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

    def test_unresolved_windows_are_neither_fitted_nor_scored(self):
        from classifier.rebuttal_analysis.multinode_family_holdout import load_multinode
        d = self.frame().assign(grp=lambda f: "s16_" + f.grp, workload_label=lambda f: "s16_" + f.workload_label,
                                label_source="program", job_label=lambda f: f.threeway_label)
        d.loc[d.grp.eq("s16_train_c"), ["label_source", "threeway_label"]] = "unresolved"
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "windows.parquet"
            d.to_parquet(path, index=False)
            w = load_multinode(path)
        self.assertNotIn("unresolved", set(w.label_source) | set(w.threeway_label))
        self.assertEqual(w.attrs["unresolved"]["windows"], int(d.grp.eq("s16_train_c").sum()))
        result, pred, _ = evaluate(w, trees=2)
        self.assertEqual(set(result["by_label_source"]), {"program"})
        self.assertEqual(set(pred.job_label), {"ml_training", "ml_inference", "other"})

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
