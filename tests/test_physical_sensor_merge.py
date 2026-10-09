"""CPU checks for sensor features and cross-GPU label separation."""
import unittest

import numpy as np
import pandas as pd

from classifier.physical_sensor_classifier import (
    TEMPORAL, augment_aggregates, physical_channels, windows,
)
from classifier.threeway_improved import get_feature_cols
from scripts.sensor_logger import flatten_snapshot


class PhysicalSensorMergeTests(unittest.TestCase):
    def test_aggregates_follow_activity_on_any_gpu(self):
        df = pd.DataFrame({
            "GPU0_rms_mv": [2., 12., 3.],
            "GPU1_rms_mv": [12., 3., 2.],
            "GPU2_rms_mv": [3., 2., 12.],
            "GPU0_peak_mv": [4., 24., 6.],
            "GPU1_peak_mv": [24., 6., 4.],
            "GPU2_peak_mv": [6., 4., 24.],
        })
        result = augment_aggregates(df)
        for name, expected in (("anyGPU_rms_mv", 12.), ("sumGPU_rms_mv", 17.),
                               ("nGPU_active", 1), ("anyGPU_peak_mv", 24.)):
            np.testing.assert_array_equal(result[name], [expected] * 3)

    def test_windows_keep_runs_separate_and_ignore_nvml(self):
        frames = []
        for run_id, value, label in (("train", 20., "ml_training"),
                                     ("idle", 2., "other")):
            frames.append(pd.DataFrame({
                "timestamp_epoch": np.arange(7, dtype=float),
                "run_id": run_id, "workload_label": run_id,
                "threeway_label": label, "GPU2_rms_mv": value,
                "GPU2_peak_mv": value * 2, "mic_rms": value / 10,
                "gpu2_power_w": 350., "power_total_w": 1000.,
                "gpu2_util_pct": 99.,
            }))
        df = augment_aggregates(pd.concat(frames, ignore_index=True))
        result = windows(df.sample(frac=1, random_state=42),
                         physical_channels(), TEMPORAL, 3, 3)
        self.assertEqual(len(result), 4)
        for run_id, value in (("train", 20.), ("idle", 2.)):
            rows = result[result.run_id == run_id]
            self.assertEqual(len(rows), 2)
            np.testing.assert_array_equal(rows.anyGPU_rms_mv_mean, [value] * 2)
        self.assertFalse(any("power" in c or "util" in c for c in result.columns))

    def test_snapshot_keeps_scalar_channels_without_wave_arrays(self):
        result = flatten_snapshot({
            "ts": 123., "mic": {"available": True, "rms": 0.1, "wave": [1, 2]},
            "scopes": [{"label": "GPU2", "rms_mv": 12., "peak_mv": 24.,
                        "wave": [1, 2, 3]}],
            "power": {"available": True, "total_w": 300.,
                      "gpus": [{"index": 2, "power_w": 200.}]},
        }, 124.)
        self.assertEqual(result["timestamp_epoch"], 124.)
        self.assertEqual(result["sensor_ts"], 123.)
        self.assertEqual(result["GPU2_rms_mv"], 12.)
        self.assertEqual(result["gpu2_power_w"], 200.)
        self.assertEqual(result["mic_rms"], 0.1)
        self.assertFalse(any("wave" in key for key in result))
        self.assertFalse(flatten_snapshot({}, 124.)["mic_available"])

    def test_numeric_ground_truth_never_becomes_a_feature(self):
        df = pd.DataFrame({
            "power_mean": [100., 200.], "gpu_util_mean": [30., 90.],
            "binary_label": [0, 1], "window_train_active": [0, 1],
            "run_id": [101, 102], "window_start": [0., 15.],
            "threeway_label": ["other", "ml_training"],
        })
        self.assertEqual(set(get_feature_cols(df)), {"power_mean", "gpu_util_mean"})


if __name__ == "__main__":
    unittest.main()
