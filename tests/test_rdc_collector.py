"""CPU checks for RDC samples and resource cleanup, without ROCm hardware."""
import importlib.util
import math
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch


spec = importlib.util.spec_from_file_location(
    "amd_collector_test", Path(__file__).resolve().parents[1] /
    "scripts/collect_amdsmi_telemetry.py")
collector_module = importlib.util.module_from_spec(spec)
with patch.dict("sys.modules", {"amdsmi": SimpleNamespace()}):
    spec.loader.exec_module(collector_module)
RdcCollector = collector_module.RdcCollector


class RdcCollectorTests(unittest.TestCase):
    def collector(self):
        obj = RdcCollector.__new__(RdcCollector)
        obj._field_map = {1: "rdc_power_usage", 2: "rdc_gpu_temp",
                          3: "rdc_gpu_clock", 4: "rdc_tensor_active"}
        obj._captured = {}
        obj._reader = Mock()
        return obj

    def test_units_and_fractional_profiling_values(self):
        obj = self.collector()
        obj._reader.process.side_effect = lambda: obj._captured.update(
            {1: 250_000_000, 2: 42_500, 3: 1_500_000_000, 4: 12.75})
        self.assertEqual(obj.sample(), {"rdc_power_usage": 250.0,
                                       "rdc_gpu_temp": 42.5,
                                       "rdc_gpu_clock": 1500.0,
                                       "rdc_tensor_active": 12.75})

    def test_failed_fields_do_not_repeat_previous_samples(self):
        obj = self.collector()
        obj._reader.process.side_effect = lambda: obj._captured.update({4: 12.75})
        self.assertEqual(obj.sample()["rdc_tensor_active"], 12.75)
        obj._reader.process.side_effect = lambda: None
        self.assertTrue(math.isnan(obj.sample()["rdc_tensor_active"]))

    def test_cleanup_releases_watch_groups_and_connection(self):
        obj = self.collector()
        obj._reader = SimpleNamespace(rdc_handle=10, gpu_group_id=20, field_group_id=30)
        obj._rdc_status_t = Mock(side_effect=lambda value: value)
        obj._rdc_status_t.RDC_ST_OK = 0
        calls = []
        def operation(name):
            def call(*args):
                calls.append((name, args))
                return 0
            return call
        names = ["rdc_field_unwatch", "rdc_group_field_destroy",
                 "rdc_group_gpu_destroy", "rdc_disconnect"]
        obj._rdc = SimpleNamespace(**{name: operation(name) for name in names})
        obj.cleanup()
        self.assertEqual(calls, list(zip(names, [(10, 20, 30), (10, 30),
                                                (10, 20), (10,)])))


if __name__ == "__main__":
    unittest.main()
