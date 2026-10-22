"""Reconstructed labels must be recovered, not invented.

The camera-ready corpus predates STEP logging, so the diluted cells in the
adversarial tables have no recorded per-window truth. classifier/
schedule_reconstruction.py recovers it from the generator's deterministic
schedule plus a measured batch rate. These tests hold it to the line between
recovery and guesswork: it answers where the schedule and the rate interval
agree, refuses where they do not, and never claims provenance it does not have.
"""
import unittest
from unittest import mock

import numpy as np

from classifier.schedule_reconstruction import (dilution_factor, reconstruct_labels,
                                                reference_rates)

T0 = 0.0


def windows(window_sec, n, t0=T0):
    starts = t0 + np.arange(n) * (window_sec / 2)
    return starts, starts + window_sec


class FactorTests(unittest.TestCase):
    def test_reads_n_from_the_cell_name(self):
        for label, want in (("dilution_sweep_N20", 20), ("adversarial_L_diluted_10", 10),
                            ("adversarial_L_diluted_10_mw8", 10), ("whitebox_lora_N5", 5),
                            ("whitebox_diluted_N20", 20)):
            self.assertEqual(dilution_factor(label), want, label)

    def test_ignores_cells_with_no_declared_schedule(self):
        for label in ("train_resnet50", "dilution_sweep_inference", "adversarial_K_online_learning"):
            self.assertIsNone(dilution_factor(label), label)


class DeferralTests(unittest.TestCase):
    """Single-kind programs are settled by their name; do not overstate them."""

    def test_single_kind_cells_defer_to_the_program_label(self):
        starts, ends = windows(30.0, 6)
        for label in ("dilution_sweep_inference", "dilution_sweep_N1", "train_resnet50"):
            labels, meta = reconstruct_labels(label, starts, ends, T0)
            self.assertIsNone(labels, f"{label}: {meta['reason']}")

    def test_refuses_a_cell_with_no_measured_rate(self):
        starts, ends = windows(30.0, 6)
        labels, meta = reconstruct_labels("whitebox_lora_N5", starts, ends, T0)
        self.assertIsNone(labels)
        self.assertIn("batch rate", meta["reason"])


class ScheduleTests(unittest.TestCase):
    """The committed logs measure these rates; the arithmetic follows from them."""

    def test_a_starved_run_trains_in_one_window_only(self):
        """adversarial_L_diluted_10 ran ~0.0142 batches/s: 4 batches in 300 s.

        A 1-in-10 schedule then takes exactly one gradient step, at batch 0,
        so one window of the run is training and the rest are pure inference --
        yet every one of them is currently scored as a training miss.
        """
        lo, hi, n_runs = reference_rates("adversarial_L_diluted_10")
        self.assertGreater(n_runs, 0, "the reference logs for this cell are missing")
        self.assertLess(hi, 1.0, "expected a dataloader-starved reference run")
        starts, ends = windows(30.0, 18)
        labels, meta = reconstruct_labels("adversarial_L_diluted_10", starts, ends, T0)
        self.assertIsNotNone(labels)
        self.assertEqual(int((labels == "ml_training").sum()), 1)
        self.assertEqual(int((labels == "ml_inference").sum()), 17)
        self.assertEqual(meta["determinate"], 1.0)

    def test_an_unstarved_run_trains_in_every_window(self):
        """The 8-worker sibling ran ~15.6 batches/s, so the job label was right.

        The correction has to bite only where the evidence says it should.
        """
        starts, ends = windows(30.0, 18)
        labels, _ = reconstruct_labels("adversarial_L_diluted_10_mw8", starts, ends, T0)
        self.assertIsNotNone(labels)
        self.assertTrue((labels == "ml_training").all())

    def test_first_window_holds_batch_zero(self):
        """batch_idx=0 always trains and always starts the run."""
        for label in ("adversarial_L_diluted_10", "adversarial_L_diluted_20"):
            starts, ends = windows(30.0, 18)
            labels, _ = reconstruct_labels(label, starts, ends, T0)
            self.assertEqual(labels[0], "ml_training", label)


class UncertaintyTests(unittest.TestCase):
    def test_disagreement_across_the_rate_interval_is_left_unlabelled(self):
        """Where the slowest and fastest reference run disagree, say nothing."""
        import classifier.schedule_reconstruction as sr

        starts, ends = windows(30.0, 40)
        with mock.patch.object(sr, "reference_rates", return_value=(0.30, 0.45, 2)):
            labels, meta = sr.reconstruct_labels("adversarial_L_diluted_10", starts, ends, T0)
        self.assertIsNotNone(labels)
        self.assertTrue(any(v is None for v in labels),
                        "a wide rate interval must leave some window unadjudicated")
        self.assertLess(meta["determinate"], 1.0)


if __name__ == "__main__":
    unittest.main()
