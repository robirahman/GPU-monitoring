"""An interrupted job can contain valid data; missing coverage is never invented."""
import json
import unittest

import numpy as np
import pandas as pd

from scripts.recover_preempted_followup import NODES, admitted_interval, training_bounds
from scripts.recover_audited_telemetry import merge_stream


class PreemptionRecoveryTests(unittest.TestCase):
    def streams(self):
        return {(n, g): pd.DataFrame({'timestamp_epoch': np.arange(90., 201.), 'power_draw_w': 100.})
                for n in NODES for g in range(8)}

    def test_noncompleted_run_keeps_observed_prefix_and_completion_false(self):
        streams = self.streams()
        streams[(NODES[0], 0)] = streams[(NODES[0], 0)].iloc[:81]
        kept, bounds = admitted_interval(streams, {'first_step': 100., 'last_step': 180.}, 175.)
        self.assertEqual(bounds, (100., 170.))
        self.assertEqual(len(kept), 16)
        for frame in kept.values():
            self.assertEqual(len(frame), 71)
            self.assertFalse(frame.meta_run_complete.any())
            self.assertTrue(frame.meta_collection_status.eq('preempted').all())
            self.assertTrue(frame.expected_ranks.eq(16).all())

    def test_missing_rank_gap_short_interval_and_invalid_times_rejected(self):
        variants = []
        x = self.streams(); del x[(NODES[0], 0)]; variants.append(x)
        x = self.streams(); x[(NODES[1], 7)] = x[(NODES[1], 7)].drop(index=range(30, 40)); variants.append(x)
        x = self.streams(); x[(NODES[1], 7)] = x[(NODES[1], 7)].iloc[:31]; variants.append(x)
        x = self.streams(); x[(NODES[1], 7)].loc[50, 'timestamp_epoch'] = np.nan; variants.append(x)
        for streams in variants:
            with self.assertRaises(ValueError):
                admitted_interval(streams, {'first_step': 100., 'last_step': 180.}, 175.)

    def test_training_evidence_is_identity_and_topology_bound(self):
        seed = json.dumps(dict(rank=0, base_seed=8, run_stamp='stamp_collect'))
        log = (f'Run ID: abc\nMegatron TP: tp_size=8 dp=2 world_size=16\nRUN_SEED {seed}\n'
               'STEP ts=100.0 kind=train batch_idx=1 tokens=2 node=0 rank=0\n'
               'STEP ts=101.0 kind=train batch_idx=2 tokens=4 node=0 rank=0\n')
        self.assertEqual(training_bounds(log, 'abc', 'stamp_collect', 8)['logged_steps'], 2)
        for bad in (log.replace('batch_idx=2', 'batch_idx=4'), log.replace('dp=2', 'dp=1'),
                    log.replace('Run ID: abc', 'Run ID: wrong'), log.replace('101.0', '99.0')):
            with self.assertRaises(ValueError):
                training_bounds(bad, 'abc', 'stamp_collect', 8)

    def test_checkpoints_deduplicate_but_conflicting_samples_rejected(self):
        f = pd.DataFrame(dict(timestamp_epoch=np.arange(100., 141.), run_id='abc',
                              hostname=NODES[0], workload_label='s16_train_tp8', power_draw_w=100.))
        self.assertEqual(len(merge_stream([f, f.iloc[:20]], 'abc', NODES[0], 's16_train_tp8')), 41)
        with self.assertRaises(ValueError):
            merge_stream([f, f.assign(power_draw_w=200.)], 'abc', NODES[0], 's16_train_tp8')


if __name__ == '__main__':
    unittest.main()
