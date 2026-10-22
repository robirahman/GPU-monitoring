"""Causality: a window's features must not move when its future is deleted.

The rebuttal fix (8ee55e53, 850c34b8) closed the within-run leak in
``sliding_windows`` by recomputing run-level features from ``ts < window_end``
and making that the default, which covered ~25 call sites at once. Nothing ever
tested the property, and four sibling feature builders were never swept. These
tests state it directly: build windows from a run, delete every sample after
window k's end, rebuild, and require window k's features to be unchanged.

Every stream here is quiet for its first half and loud for its second, so a
feature that reads past a window's end moves by a wide margin when the tail
goes. Against a stationary fixture these tests would pass over a real leak.
"""
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd

from classifier.adversary_defender_iteration import add_evasion_resistant_features
from classifier.eval_fourtier_evasion import BMC_FEATURE_NAMES, attach_bmc_features
from classifier.multinode_features import sliding_windows_multinode
from classifier.scope_features import SCOPE_FEATURE_NAMES, attach_scope_features
from classifier.scope_raw_spectrum import RAW_FEATURE_NAMES, attach_raw_features
from classifier.threeway_improved import RAW_SIGNALS, get_feature_cols, sliding_windows

T0 = 1_750_000_000.0
WINDOW, STRIDE = 30.0, 15.0
SPLIT = 120.0            # seconds into the run: quiet before, loud after
CUT = T0 + SPLIT         # truncation point for every test


def telemetry(n=240, seed=0, run_id="run0", label="train_resnet50", host="node0",
              gpu_uuid="GPU-0", gpu="NVIDIA A100-SXM4-80GB", rank_token=""):
    """A 1 Hz stream whose second half is four times louder than its first."""
    rng = np.random.default_rng(seed)
    idx = np.arange(n)
    loud = np.where(idx < SPLIT, 1.0, 4.0)
    frame = pd.DataFrame({
        "ts": T0 + idx.astype(float),
        "run_id": run_id,
        "workload_label": label + rank_token,
        "threeway_label": "ml_training",
        "gpu_uuid": gpu_uuid,
        "gpu_name": gpu,
        "hostname": host,
        "mem_total_mb": 81920.0,
    })
    for sig in RAW_SIGNALS:
        frame[sig] = loud * (25.0 + rng.normal(0.0, 0.5, n))
    # Memory plateaus late. This is exactly what the run-level plateau features
    # read, and reading it from the end of the run was the original leak.
    frame["mem_used_mb"] = np.where(idx < SPLIT, 2_000.0, 48_000.0) + rng.normal(0, 5, n)
    return frame


def moved_features(full, truncated, columns, case):
    """Columns that differ, over the windows that ended before the cut."""
    key = ["run_id", "window_start"] if "run_id" in full.columns else ["window_start"]
    early = truncated[truncated.window_end <= CUT]
    if early.empty:
        raise AssertionError(f"{case}: truncation left no comparable window")
    merged = early.merge(full, on=key, suffixes=("_cut", "_full"))
    if len(merged) != len(early):
        raise AssertionError(
            f"{case}: {len(early) - len(merged)} truncated window(s) have no counterpart "
            "in the full build, so their features cannot be compared")
    moved = []
    for col in columns:
        a = pd.to_numeric(merged[f"{col}_cut"], errors="coerce").to_numpy(float)
        b = pd.to_numeric(merged[f"{col}_full"], errors="coerce").to_numpy(float)
        if not np.allclose(a, b, rtol=1e-9, atol=1e-9, equal_nan=True):
            moved.append(col)
    return moved


class SlidingWindowCausality(unittest.TestCase):
    """The chokepoint every single-stream evaluation path goes through."""

    def build(self, df, mode):
        return sliding_windows(df, WINDOW, STRIDE, run_level_mode=mode)

    def moved(self, mode):
        frame = telemetry()
        full = self.build(frame, mode)
        cut = self.build(frame[frame.ts < CUT], mode)
        return moved_features(full, cut, get_feature_cols(full), mode)

    def test_causal_mode_is_invariant_to_the_future(self):
        moved = self.moved("causal")
        self.assertEqual(moved, [], f"look-ahead in sliding_windows(run_level_mode='causal'): {moved}")

    def test_none_mode_is_invariant_to_the_future(self):
        moved = self.moved("none")
        self.assertEqual(moved, [], f"look-ahead with run-level features disabled: {moved}")

    def test_orig_mode_still_leaks(self):
        """Positive control. If this passes, the fixture has gone blind."""
        self.assertTrue(self.moved("orig"),
                        "run_level_mode='orig' is the archived leaky path; it must still "
                        "differ under truncation, or these tests cannot detect a leak")


class EvasionResistantFeatureCausality(unittest.TestCase):
    """``add_evasion_resistant_features`` -- Round 3's training-as-inference answer.

    These are the "power-memory correlation + cumulative energy" features the
    paper credits for Round 3, and the set ``eval_catchup`` re-admits after
    stripping the plateau features that were already fixed.
    """

    PHYSICS = ["cumulative_energy_wh", "power_mem_correlation", "util_bimodal_score",
               "power_efficiency_ratio", "sustained_compute_frac"]

    def test_physics_features_are_invariant_to_the_future(self):
        frame = telemetry()
        windows = sliding_windows(frame, WINDOW, STRIDE, run_level_mode="causal")
        full = add_evasion_resistant_features(windows.copy(), frame)
        trunc = add_evasion_resistant_features(windows.copy(), frame[frame.ts < CUT])
        moved = moved_features(full, trunc, self.PHYSICS, "evasion-resistant")
        self.assertEqual(moved, [], f"whole-run physics stamped onto early windows: {moved}")


class MultinodeWindowCausality(unittest.TestCase):
    """``sliding_windows_multinode`` -- the pooled multi-node chokepoint."""

    @staticmethod
    def job():
        """A full s16 cell: two hosts of eight GPUs.

        The width has to match the label. `expected_rank_count` reads 16 out of
        the `s16_` prefix, and a narrower fixture is dropped as degraded before
        any feature is computed.
        """
        return pd.concat(
            [telemetry(seed=10 * node + gpu, run_id="mnrun", host=f"node{node}",
                       gpu_uuid=f"GPU-{node}-{gpu}", label="s16_train_fsdp1b",
                       rank_token=f"_node{node}_gpu{gpu}")
             for node in (0, 1) for gpu in range(8)],
            ignore_index=True)

    def test_window_features_are_invariant_to_the_future(self):
        frame = self.job()
        full = sliding_windows_multinode(frame, WINDOW, STRIDE)
        trunc = sliding_windows_multinode(frame[frame.ts < CUT], WINDOW, STRIDE)
        moved = moved_features(full, trunc, get_feature_cols(full), "multinode")
        self.assertEqual(moved, [], f"look-ahead in sliding_windows_multinode: {moved}")

    def test_window_membership_is_invariant_to_the_future(self):
        """Values can be clean while membership is not.

        The rank-qualification gate divides by a cadence median taken over the
        whole job, so whether a window is admitted at all can depend on samples
        that arrive after it ends.
        """
        frame = self.job()
        full = sliding_windows_multinode(frame, WINDOW, STRIDE)
        trunc = sliding_windows_multinode(frame[frame.ts < CUT], WINDOW, STRIDE)
        self.assertEqual(set(trunc[trunc.window_end <= CUT].window_start),
                         set(full[full.window_end <= CUT].window_start),
                         "window admission changed when the job's future was removed")


class SidecarCausality(unittest.TestCase):
    """Sidecars joined onto an existing window frame: scope, BMC, raw spectrum."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.dir = Path(self.tmp.name)
        self.windows = sliding_windows(telemetry(), WINDOW, STRIDE, run_level_mode="causal")
        self.addCleanup(self.tmp.cleanup)

    def _write(self, name, frame):
        path = self.dir / name
        frame.to_parquet(path, index=False)
        return str(path)

    def test_scope_features_are_invariant_to_the_future(self):
        """The correct template: compute_scope_window_features masks [wstart, wend)."""
        rng = np.random.default_rng(3)
        ep = T0 + np.arange(240_000) / 1000.0
        amps = np.where(ep < CUT, 1.0, 6.0) * rng.gamma(2.0, 1.0, ep.size)
        scope = pd.DataFrame({"epoch_s": ep, "amps_rms_ac": amps})
        full = attach_scope_features(self.windows.copy(),
                                     {"run0": self._write("scope_full.parquet", scope)})
        trunc = attach_scope_features(
            self.windows.copy(),
            {"run0": self._write("scope_cut.parquet", scope[scope.epoch_s < CUT])})
        moved = moved_features(full, trunc, SCOPE_FEATURE_NAMES, "scope")
        self.assertEqual(moved, [], f"look-ahead in attach_scope_features: {moved}")

    def test_bmc_features_are_invariant_to_the_future(self):
        """The nearest-reading NaN fallback can select a sample after window_end."""
        # 45 s apart, so some 30 s windows hold no reading at all and the
        # nearest-reading fallback is the code path actually under test.
        ep = T0 + np.arange(16) * 45.0
        bmc = pd.DataFrame({"epoch": ep, "bmc_power_w": np.where(ep < CUT, 300.0, 1800.0)})
        full = attach_bmc_features(self.windows.copy(),
                                   {"run0": self._write("bmc_full.parquet", bmc)})
        trunc = attach_bmc_features(self.windows.copy(),
                                    {"run0": self._write("bmc_cut.parquet", bmc[bmc.epoch < CUT])})
        moved = moved_features(full, trunc, BMC_FEATURE_NAMES, "bmc")
        self.assertEqual(moved, [], f"look-ahead in attach_bmc_features: {moved}")

    def test_raw_spectrum_refuses_window_level_use(self):
        """One capture per run cannot be a per-window feature.

        ``attach_raw_features`` broadcasts a single short snippet across every
        window of its run, so windows that end before the capture instant would
        carry features derived from telemetry that had not happened yet. The
        snippet carries no usable epoch, so the windows that precede it cannot
        be masked -- the function refuses instead, and run-level analysis opts
        in explicitly.
        """
        fs = 50_000
        wave = (3.0 * np.sin(2 * np.pi * 50 * np.arange(fs) / fs)).astype(np.float32)
        path = self.dir / "raw.npy"
        np.save(path, wave)
        with self.assertRaises(ValueError):
            attach_raw_features(self.windows.copy(), {"run0": str(path)})
        opted_in = attach_raw_features(self.windows.copy(), {"run0": str(path)},
                                       allow_run_constant=True)
        for name in RAW_FEATURE_NAMES:
            self.assertEqual(opted_in[name].nunique(dropna=False), 1,
                             f"{name} is a per-run constant by construction")


if __name__ == "__main__":
    unittest.main()
