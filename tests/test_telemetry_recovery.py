"""Recovery must not turn failures, wrong identities or missing metrics into evidence."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.recover_audited_telemetry import (
    clean_nic, merge_stream, nic_usable, same_or_absent, tpu_completion,
)


class TelemetryRecoveryTests(unittest.TestCase):
    def nic(self):
        return pd.DataFrame(dict(timestamp_epoch=np.arange(100., 170.),
                                 run_id="run", hostname="host0", workload_label="s16_train_fsdp1b",
                                 nic_rx_mbps=0., nic_tx_mbps=0.))

    def test_only_paired_interval_and_no_existing_join_seconds(self):
        frame, _ = clean_nic(self.nic(), [(110., 160.)], [(145., 155.)])
        self.assertEqual(frame.timestamp_epoch.min(), 110.)
        self.assertEqual(frame.timestamp_epoch.max(), 160.)
        self.assertFalse(frame.timestamp_epoch.between(144., 156.).any())
        self.assertTrue(nic_usable(frame))  # 34 seconds of valid zero traffic
        empty, _ = clean_nic(self.nic(), [])
        self.assertFalse(nic_usable(empty))

    def test_missing_counters_are_not_zero_and_short_fragments_not_admitted(self):
        data = self.nic().assign(nic_rx_mbps=-1., nic_tx_mbps=np.inf)
        frame, stats = clean_nic(data, [(100., 170.)])
        self.assertTrue(frame.nic_rx_mbps.isna().all())
        self.assertEqual(stats["invalid_metric_values"], 140)
        self.assertFalse(nic_usable(frame))
        fragments, _ = clean_nic(self.nic(), [(100., 115.), (150., 169.)])
        self.assertFalse(nic_usable(fragments))

    def test_copies_deduplicate_but_conflicting_samples_and_hosts_fail(self):
        original = self.nic()
        self.assertEqual(len(merge_stream([original, original], "run", "host0", "s16_train_fsdp1b")), 70)
        for bad in (original.assign(hostname="host1"), original.assign(nic_rx_mbps=2.)):
            with self.assertRaises(ValueError):
                merge_stream([original, bad], "run", "host0", "s16_train_fsdp1b")

    def test_integer_join_uses_nearest_sample_without_multiplying_rows(self):
        original = self.nic()
        duplicate = original.iloc[[0]].assign(timestamp_epoch=100.2, nic_rx_mbps=999.)
        frame, stats = clean_nic(pd.concat([duplicate, original]), [(100., 170.)])
        self.assertEqual(stats["rounded_key_duplicates"], 1)
        self.assertEqual(len(frame), 70)
        self.assertEqual(frame.nic_rx_mbps.iloc[0], 0.)

    def test_all_workers_must_complete_and_markers_must_match(self):
        markers = {w: f"rc=0 run_id=abc chips=1 worker={w}" for w in (0, 1000, 2000, 3000)}
        self.assertTrue(tpu_completion(markers, "abc")[0])
        markers[1000] = "rc=124 run_id=abc chips=1 worker=1000"
        self.assertFalse(tpu_completion(markers, "abc")[0])
        with self.assertRaises(ValueError):
            tpu_completion(markers, "different")
        with self.assertRaises(ValueError):
            tpu_completion({0: markers[0]}, "abc")

    def test_recovery_cannot_overwrite_different_existing_bytes(self):
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp) / "recovered.parquet"
            digest = hashlib.sha256(b"original").hexdigest()
            same_or_absent(path, digest)
            path.write_bytes(b"original")
            same_or_absent(path, digest)
            path.write_bytes(b"different")
            with self.assertRaises(ValueError):
                same_or_absent(path, digest)


if __name__ == "__main__":
    unittest.main()
