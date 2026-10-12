"""GCS log identity and observed vLLM request/batch boundary regressions."""
from contextlib import redirect_stdout
import gzip
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from classifier.window_labels import parse_work_log, work_intervals, label_windows, assign_work_labels
from scripts.gcp.fetch_work_logs import select_log, inspect_log, relative_path, fetch_one
from scripts.gcp.import_work_logs import import_export
from workloads import llm_inference_vllm as offline
from workloads import llm_serving_async as serving


class WorkLogCollectionTests(unittest.TestCase):
    def intervals(self, text):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "worker.log"
            path.write_text("Run ID: abcdef123456\n" + text)
            parsed = parse_work_log(path)
        return work_intervals(parsed["events"])

    def test_overlapping_requests_exclude_idle_and_startup(self):
        iv = self.intervals(
            "STEP ts=100.0 kind=pending batch_idx=0 event_id=0\n"
            "STEP ts=105.0 kind=pending batch_idx=1 event_id=1\n"
            "STEP ts=112.0 kind=infer batch_idx=1 event_id=1 start_ts=105.0\n"
            "STEP ts=115.0 kind=infer batch_idx=0 event_id=0 start_ts=100.0\n"
            "STEP ts=140.0 kind=pending batch_idx=2 event_id=2\n"
            "STEP ts=145.0 kind=infer batch_idx=2 event_id=2 start_ts=140.0\n")
        lab = label_windows([90, 100, 120, 140], [100, 120, 140, 150], iv)
        self.assertEqual(list(lab["label"]), ["other", "ml_inference", "other", "ml_inference"])
        np.testing.assert_allclose(lab["infer_s"], [0, 15, 0, 5])

    def test_failed_and_unfinished_requests_are_unknown(self):
        iv = self.intervals(
            "STEP ts=100.0 kind=pending batch_idx=0 event_id=0\n"
            "STEP ts=110.0 kind=unknown batch_idx=0 event_id=0 start_ts=100.0\n"
            "STEP ts=130.0 kind=pending batch_idx=1 event_id=1\n")
        rows = [dict(window_start=t, window_end=t + 10) for t in (90, 100, 115, 130, 10000)]
        assign_work_labels(rows, iv, "s16_inferserve_poisson_7b", "ml_inference")
        self.assertEqual([r["threeway_label"] for r in rows], ["other", "unresolved", "other", "unresolved", "unresolved"])
        self.assertTrue(np.isnan(rows[1]["label_train_frac"]))
        self.assertEqual(rows[-1]["label_unknown_s"], 10)

    def test_invalid_explicit_bounds_fail(self):
        with self.assertRaisesRegex(ValueError, "Invalid STEP"):
            self.intervals("STEP ts=100.0 kind=infer batch_idx=0 start_ts=110.0\n")

    def test_same_basename_in_different_campaigns_never_bridges(self):
        with tempfile.TemporaryDirectory() as tmp:
            parsed = []
            for campaign, t in (("one", 100), ("two", 10000)):
                path = Path(tmp) / campaign / "cell_w0.log"
                path.parent.mkdir()
                path.write_text(f"Run ID: abcdef123456\nSTEP ts={t} kind=train\nSTEP ts={t+1} kind=train\n")
                parsed.append(parse_work_log(path))
            import pandas as pd
            iv = work_intervals(pd.concat([p["events"].assign(log=p["log"]) for p in parsed]))
            self.assertEqual((iv.end - iv.start).max(), 1)

    def test_gcs_selection_keeps_known_workers_and_inference_not_sidecars(self):
        prefix = "gs://bucket/tpu/campaign/runs/abcdef123456/w1000/"
        self.assertTrue(select_log(prefix + "cell_w1000.log", {"abcdef123456"}))
        self.assertFalse(select_log(prefix + "cell_w1000.log", {"different"}))
        self.assertFalse(select_log(prefix + "telemetry.parquet", {"abcdef123456"}))
        self.assertTrue(select_log("gs://bucket/runs/20260901/host/logs/s16_inferserve_poisson_node0.log", set()))
        self.assertFalse(select_log("gs://bucket/runs/20260901/host/logs/s16_inferserve_poisson_node0.nic.log", set()))
        with self.assertRaises(ValueError):
            relative_path("gs://bucket/../escape.log")

    def test_compressed_tpu_header_and_kinds_survive(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "tpu_w1000.log.gz"
            with gzip.open(path, "wt") as f:
                f.write("Run ID: abcdef123456\nSTEP ts=100.0 kind=filler batch_idx=0\nSTEP ts=101.0 kind=train batch_idx=1\n")
            evidence = inspect_log(path)
            self.assertEqual(evidence["run_ids"], ["abcdef123456"])
            self.assertEqual(evidence["step_kinds"], {"filler": 1, "train": 1})

    def test_download_import_resume_and_tamper_detection(self):
        uri = "gs://bucket/tpu/campaign/runs/abcdef123456/w0/cell_w0.log"
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp) / "export"; out.mkdir()
            repo = Path(tmp) / "repo"
            body = b"Run ID: abcdef123456\nSTEP ts=100.0 kind=train batch_idx=0\n"
            def download(cmd, **kwargs):
                Path(cmd[-1]).write_bytes(body)
            with patch("scripts.gcp.fetch_work_logs.subprocess.run", side_effect=download) as cp:
                record = fetch_one(uri, out)
                self.assertEqual(fetch_one(uri, out), record)
                self.assertEqual(cp.call_count, 1)
            manifest = dict(schema="work-log-export-v1", selection_commit="fixture",
                            objects=[record], failures=[])
            (out / "manifest.json").write_text(json.dumps(manifest))
            imported = import_export(out, repo)
            dest = repo / imported["objects"][0]["repository_path"]
            with gzip.open(dest, "rb") as f:
                self.assertEqual(f.read(), body)
            (out / record["local_path"]).write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "Corrupt exported log"):
                import_export(out, repo)

    def test_gcs_gpu_path_preserves_node_host_identity(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bucket/runs/20261001/host-b/logs/s8_inferrep_node1.log"
            path.parent.mkdir(parents=True)
            path.write_text("Run ID: abcdef123456\nSTEP ts=100.0 kind=infer\n")
            self.assertEqual(parse_work_log(path)["host"], "host-b")

    def test_headerless_launcher_joins_by_exact_filename_run_id(self):
        import pandas as pd
        from classifier.window_labels import build_work_index, job_work_intervals
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "bucket/runs/20261001/host-b/logs/s16_inferhf_qwen7b_node1_abcdef123456.log.gz"
            path.parent.mkdir(parents=True)
            with gzip.open(path, "wt") as f:
                f.write("STEP ts=110 kind=infer node=1 rank=8\nSTEP ts=120 kind=infer node=1 rank=8\n")
            parsed = parse_work_log(path)
            self.assertEqual(parsed["run_id_source"], "filename")
            frame = pd.DataFrame(dict(job_id=["j", "j"], run_id="abcdef123456",
                                      workload_label="s16_inferhf_qwen7b", hostname="host-b",
                                      source_path="cell_node1_gpu0.parquet", ts=[100., 130.]))
            intervals = job_work_intervals(frame, build_work_index([path]))
            self.assertEqual(list(label_windows([100., 120.], [120., 130.], intervals["j"])["label"]),
                             ["ml_inference", "other"])
            with gzip.open(path, "wt") as f:
                f.write("Run ID: 999999999999\nSTEP ts=110 kind=infer\n")
            with self.assertRaisesRegex(ValueError, "Conflicting header/filename"):
                parse_work_log(path)

    def test_unstructured_filename_does_not_supply_a_run_id(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "debug_abcdef123456.log"
            path.write_text("STEP ts=110 kind=infer\n")
            self.assertIsNone(parse_work_log(path)["run_id"])

    def test_reused_run_id_outside_telemetry_is_not_label_evidence(self):
        import pandas as pd
        from classifier.window_labels import job_work_intervals
        frame = pd.DataFrame(dict(job_id=["j", "j"], run_id="abcdef123456", workload_label="tpuTMS_advtrain_satint_f05",
                                  hostname="w0", source_path="data/tpu/cell.parquet", ts=[100., 200.]))
        entry = dict(log="other-session.log", node=0, host="w0",
                     events=pd.DataFrame(dict(ts=[9000., 9001.], work="ml_training", node=0, rank=0)))
        self.assertEqual(job_work_intervals(frame, {"abcdef123456": [entry]}), {})

    def test_offline_generate_emits_observed_bounds_and_token_counts(self):
        clock = SimpleNamespace(now=100.)
        class LLM:
            def __init__(self, **kwargs):
                pass
            def generate(self, prompts, sampling):
                clock.now += 10
                return [SimpleNamespace(outputs=[SimpleNamespace(token_ids=[1, 2, 3])])]
        args = SimpleNamespace(model=next(iter(offline.VLLM_MODELS)), tensor_parallel_size=1,
                               pipeline_parallel_size=1, distributed_executor_backend="mp",
                               gpu_memory_utilization=.9, max_model_len=0, enforce_eager=True,
                               max_tokens=3, duration=15, batch_size=1, prompt_tokens=0)
        captured = io.StringIO()
        with patch.dict("sys.modules", vllm=SimpleNamespace(LLM=LLM, SamplingParams=lambda **k: k)), \
             patch.object(offline.time, "time", side_effect=lambda: clock.now), redirect_stdout(captured):
            offline.run_vllm(args)
        iv = self.intervals(captured.getvalue())
        self.assertEqual(list(iv.start), [100, 110])
        self.assertEqual(list(iv.end), [110, 120])
        self.assertEqual(set(iv.work), {"ml_inference"})
        self.assertEqual(captured.getvalue().count("tokens=3"), 2)

    def test_async_success_and_failure_emit_separate_evidence(self):
        stats, captured = serving.Stats(), io.StringIO()
        response = io.BytesIO(json.dumps({"usage": {"prompt_tokens": 4, "completion_tokens": 3}}).encode())
        with patch.object(serving.urllib.request, "urlopen", return_value=response), \
             patch.object(serving.time, "time", side_effect=[100., 110.]), redirect_stdout(captured):
            serving.fire("http://localhost", "model", "prompt", 3, stats, 1)
        with patch.object(serving.urllib.request, "urlopen", side_effect=TimeoutError), \
             patch.object(serving.time, "time", side_effect=[120., 125.]), redirect_stdout(captured):
            serving.fire("http://localhost", "model", "prompt", 3, stats, 1)
        self.assertEqual((stats.done, stats.err, stats.ptok, stats.otok), (1, 1, 4, 3))
        iv = self.intervals(captured.getvalue())
        self.assertEqual(list(iv.work), ["ml_inference", "unresolved"])
        self.assertEqual(list(iv.start), [100, 120])
        self.assertEqual(list(iv.end), [110, 125])


if __name__ == "__main__":
    unittest.main()
