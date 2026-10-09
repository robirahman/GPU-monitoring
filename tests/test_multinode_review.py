"""CPU regressions for PR #1's merge and job-level evaluation."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd

from classifier import threeway_improved as features
from classifier.rebuttal_analysis import multinode_baseline as baseline
from classifier.run_alerts import run_alert_rate


class MultinodeReviewTests(unittest.TestCase):
    def test_single_node_cells_with_shared_run_id_remain_separate_jobs(self):
        paths = [Path(f"s8_train_fsdp1b_node{n}_gpu0_NVIDIA_H100.parquet")
                 for n in range(2)]
        frames = [pd.DataFrame({"ts": np.arange(6, dtype=float),
                                "run_id": ["shared"] * 6,
                                "hostname": [f"node{n}"] * 6}) for n in range(2)]
        with patch.object(baseline.pd, "read_parquet", side_effect=frames):
            windows = baseline.load(paths, 10, 5)
        self.assertEqual(windows.job_id.nunique(), 2)
        self.assertEqual(set(windows.job_id), {"shared|node0", "shared|node1"})

    def test_stream_time_order_and_any_stream_job_aggregation(self):
        w = pd.DataFrame({
            "job_id": ["a"] * 5 + ["b"] * 3,
            "run_id": ["a0"] * 3 + ["a1"] * 2 + ["b0"] * 3,
            "window_start": [0, 2, 1, 0, 1, 0, 1, 2],
        })
        scores = [.9, .1, .9, .1, .1, .1, .1, .1]
        # a0 alerts at t=0,1 after sorting; a1 and b0 never alert.
        self.assertEqual(run_alert_rate(w, scores, .8, 2, 2, "job_id"), (.5, 2))
        self.assertEqual(run_alert_rate(w, scores, .8, 2, 2), (1 / 3, 3))

    def test_empty_and_invalid_alert_inputs(self):
        rate, count = run_alert_rate(pd.DataFrame(), [], .8, 3, 5)
        self.assertTrue(np.isnan(rate))
        self.assertEqual(count, 0)
        with self.assertRaises(ValueError):
            run_alert_rate(pd.DataFrame(), [], .8, 6, 5)

    def test_baseline_ignores_quarantine_and_sidecars(self):
        paths = [Path("data/_quarantine_idle/s16_train_NVIDIA_H100.parquet"),
                 Path("data/nic_s16_train_NVIDIA_H100.parquet"),
                 Path("data/s16_train_scope_NVIDIA_H100.parquet"),
                 Path("data/s16_train_dcgm_NVIDIA_H100.parquet")]
        with patch.object(baseline.pd, "read_parquet") as read:
            self.assertIsNone(baseline.load(paths, 30, 15))
        read.assert_not_called()

    def test_loader_keeps_all_ranks_and_excludes_both_kinds_of_sidecar(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            data = root / "data"
            data.mkdir()
            rows = pd.DataFrame({
                "ts": [1000.0, 1000.0, 1001.0, 1001.0],
                "gpu_uuid": ["gpu0", "gpu1", "gpu0", "gpu1"],
                "run_id": ["job"] * 4,
                "workload_label": ["s16_train_fsdp1b"] * 4,
            })
            rows.to_parquet(data / "training.parquet")
            rows.to_parquet(data / "duplicate.parquet")
            for name in ("nic_sidecar.parquet", "cell_scope_data.parquet",
                         "cell_dcgm_data.parquet", "_quarantine_idle/bad.parquet"):
                p = data / name
                p.parent.mkdir(exist_ok=True)
                rows.assign(run_id=name).to_parquet(p)
            with patch.object(features, "PROJECT_ROOT", root):
                result = features.load_all_data()
            self.assertEqual(len(result), 4)
            self.assertEqual(set(result.run_id), {"job"})
            self.assertEqual(result.gpu_uuid.nunique(), 2)

    def test_job_alerts_do_not_stitch_together_independent_streams(self):
        rows = []
        for cls in ("ml_training", "ml_inference", "other"):
            for family in range(2):
                job = f"{cls}-{family}"
                for stream in range(2):
                    for t in range(2):
                        rows.append(dict(cls=cls, grp=job, job_id=job,
                                         run_id=f"{job}-gpu{stream}", is_adv=False,
                                         window_start=t, score=0.9))
        fitted_groups = []

        class FixedClassifier:
            def __init__(self, **kwargs):
                pass

            def fit(self, X, y, groups, **kwargs):
                fitted_groups.extend(groups)

            def predict_proba_training(self, X):
                return X[:, 0]

        with patch.object(baseline, "TwoStageClassifier", FixedClassifier):
            out, _, _ = baseline.evaluate(pd.DataFrame(rows), ["score"], .8, 3, 5)
        # Every stream has only TWO positive windows. Combining two streams
        # must not fabricate the three consecutive-time observations required.
        self.assertEqual(out["run_tpr"], 0)
        self.assertEqual(out["run_fpr"], 0)
        self.assertEqual(out["n_jobs"], 6)
        self.assertEqual(out["n_streams"], 12)
        self.assertFalse(any("-gpu" in group for group in fitted_groups))


if __name__ == "__main__":
    unittest.main()
