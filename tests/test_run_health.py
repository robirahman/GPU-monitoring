"""The run-health check, against the four shapes that actually occurred.

Every case here is taken from a real tier-A job; the numbers in the comments
are that job's measured span and busy fraction.
"""
import unittest

import numpy as np
import pandas as pd

from scripts.run_health import assess_run_health, requested_duration_of


def frame(span_s, busy_frac, util_when_busy=90.0, hz=1.0):
    """A telemetry frame of `span_s` seconds, busy in `busy_frac` of samples."""
    n = max(int(span_s * hz) + 1, 1)
    util = np.zeros(n)
    util[: int(round(n * busy_frac))] = util_when_busy
    rng = np.random.default_rng(0)
    rng.shuffle(util)
    return pd.DataFrame({"timestamp_epoch": np.arange(n) / hz,
                         "gpu_utilization_pct": util})


class RunHealth(unittest.TestCase):
    def test_a_full_length_busy_run_is_ok(self):
        # adversarial_I_stochastic_30 on A100: 312 s, busy 0.71
        self.assertEqual(assess_run_health(frame(312, 0.71), 300)[0], "ok")

    def test_a_quiet_workload_is_not_starved(self):
        # adversarial_E_memory_minimal: mean utilisation 2.6% but busy 0.91.
        # Judging this on the mean would condemn a workload that is quiet by
        # design, which is why the test is the busy fraction.
        self.assertEqual(assess_run_health(frame(330, 0.91, util_when_busy=3.0), 300)[0], "ok")

    def test_an_idle_full_length_run_is_starved(self):
        # adversarial_L_diluted_2 on A100: 324 s, busy 0.018
        self.assertEqual(assess_run_health(frame(324, 0.018), 300)[0], "starved")

    def test_a_short_run_with_a_dead_gpu_is_aborted(self):
        # adversarial_H_mimicry_cufft on A100: 32 s of 300, busy 0.0
        self.assertEqual(assess_run_health(frame(32, 0.0), 300)[0], "aborted")

    def test_a_short_run_that_used_the_gpu_is_truncated(self):
        # adversarial_training on B200: 20 s of 300, busy 0.985 -- it ran hard
        # and stopped early, which is a different fault from never starting.
        self.assertEqual(assess_run_health(frame(20, 0.985), 300)[0], "truncated")

    def test_an_empty_frame_is_not_ok(self):
        self.assertEqual(assess_run_health(pd.DataFrame(), 300)[0], "no_samples")

    def test_duration_is_read_from_the_workload_argv(self):
        self.assertEqual(requested_duration_of(
            ["python", "w.py", "--strategy", "L", "--duration", "300"]), 300.0)
        self.assertEqual(requested_duration_of(["python", "w.py", "--duration=120"]), 120.0)
        self.assertIsNone(requested_duration_of(["python", "w.py"]))

    def test_without_a_declared_duration_only_starvation_is_judged(self):
        # No --duration means no way to know a run was cut short, so a short
        # busy run must not be called truncated on a guess.
        self.assertEqual(assess_run_health(frame(20, 0.985), None)[0], "ok")
        self.assertEqual(assess_run_health(frame(20, 0.0), None)[0], "starved")


if __name__ == "__main__":
    unittest.main()
