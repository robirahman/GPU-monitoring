"""Test topology checks without starting a distributed JAX cluster."""
import os
import unittest
from types import SimpleNamespace
from unittest.mock import Mock, patch

from workloads import tpu_transformer_common as transformer


class MultihostGateTests(unittest.TestCase):
    def runtime(self, processes):
        return SimpleNamespace(distributed=SimpleNamespace(initialize=Mock()),
                               process_count=lambda: processes,
                               process_index=lambda: 0, device_count=lambda: 8,
                               local_device_count=lambda: 4)

    def test_required_rejects_silent_single_host_degradation(self):
        runtime = self.runtime(1)
        with patch.dict(os.environ, {"SPAR_MULTIHOST": "required"}, clear=True), \
                patch.dict("sys.modules", {"jax": runtime}):
            with self.assertRaises(SystemExit):
                transformer.init_distributed("auto")
        runtime.distributed.initialize.assert_called_once()

    def test_required_accepts_the_expected_topology(self):
        runtime = self.runtime(2)
        with patch.dict(os.environ, {"SPAR_MULTIHOST": "required"}, clear=True), \
                patch.dict("sys.modules", {"jax": runtime}):
            self.assertEqual(transformer.init_distributed("auto", expect_processes=2), (2, 0))

    def test_required_rejects_incomplete_topology(self):
        runtime = self.runtime(2)
        with patch.dict(os.environ, {"SPAR_MULTIHOST": "required"}, clear=True), \
                patch.dict("sys.modules", {"jax": runtime}):
            with self.assertRaises(SystemExit):
                transformer.init_distributed("auto", expect_processes=4)


if __name__ == "__main__":
    unittest.main()
