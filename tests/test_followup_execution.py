"""Scientific accounting and execution preparation regressions."""
import hashlib
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from classifier.cache_provenance import input_paths
from classifier.operational_metrics import operational_metrics, interval_union_seconds
from scripts.check_analysis_inputs import inspect_inputs
from scripts.plan_followup_repetitions import make_plan
from scripts.run_followup_evaluation import commands


class InputAvailabilityTests(unittest.TestCase):
    def test_exact_policy_missing_and_corrupt_external_objects(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); (root/'data').mkdir()
            payload=b'actual telemetry'; oid=hashlib.sha256(payload).hexdigest()
            ptr=f'version https://git-lfs.github.com/spec/v1\noid sha256:{oid}\nsize {len(payload)}\n'
            (root/'data/train.parquet').write_text(ptr)
            (root/'data/duplicate.parquet').write_text(ptr)
            (root/'data/nic_train.parquet').write_bytes(b'network')
            (root/'data/train_scope.parquet').write_text(ptr)
            (root/'data/_quarantine').mkdir()
            (root/'data/_quarantine/bad.parquet').write_text(ptr)
            objects=root/'external'; obj=objects/oid[:2]/oid[2:4]/oid
            summary, rows=inspect_inputs(root, objects)
            self.assertFalse(summary['ready'])
            self.assertEqual(summary['missing_or_corrupt_paths'], 2)
            self.assertEqual(summary['missing_or_corrupt_objects'], 1)
            self.assertEqual(summary['missing_or_corrupt_bytes'], len(payload))
            self.assertEqual({r['path'] for r in rows},
                             {str(p.relative_to(root)) for p in input_paths(root)})
            obj.parent.mkdir(parents=True); obj.write_bytes(b'x'*len(payload))
            summary, rows=inspect_inputs(root, objects)
            self.assertEqual(sum(r['status']=='corrupt' for r in rows), 2)
            obj.write_bytes(payload)
            self.assertTrue(inspect_inputs(root, objects)[0]['ready'])
            self.assertEqual((root/'data/train.parquet').read_text(), ptr)

    def test_empty_inputs_are_not_ready(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertFalse(inspect_inputs(tmp)[0]['ready'])


class OperationalTests(unittest.TestCase):
    def frame(self, starts, truth, probabilities, job='job', source='steps'):
        return pd.DataFrame(dict(job_id=job, window_start=starts,
            window_end=[s+30 for s in starts], truth=truth,
            p_training=probabilities, label_source=source))

    def test_overlaps_and_gaps_do_not_inflate_benign_exposure(self):
        f=self.frame([0,15,30,180,195], ['other']*5, [.9,.9,.1,.9,.1])
        result=operational_metrics(f)
        self.assertAlmostEqual(result['benign_job_hours'], 105/3600)
        self.assertEqual(result['false_alert_episodes'], 2)
        self.assertEqual(result['false_alert_jobs'], 1)
        self.assertAlmostEqual(result['false_alert_episodes_per_job_hour'], 2*3600/105)
        self.assertEqual(interval_union_seconds([(0,30),(0,15),(15,45)]),45)

    def test_decision_at_window_end_and_early_alert_is_not_detection(self):
        f=self.frame([0,15,30,45], ['other','other','ml_training','ml_training'], [.9,.1,.1,.9])
        result=operational_metrics(f)
        self.assertEqual(result['pretraining_alert_jobs'],1)
        self.assertEqual(result['timing_detected_jobs'],1)
        self.assertEqual(result['detected_delay_lower_median_s'],15)
        self.assertEqual(result['detected_delay_upper_median_s'],45)
        f.p_training=[.9,.1,.1,.1]
        result=operational_metrics(f)
        self.assertEqual(result['timing_missed_jobs'],1)
        self.assertIsNone(result['detected_delay_lower_median_s'])

    def test_program_labels_and_censored_onset_do_not_claim_delay(self):
        program=self.frame([0,15], ['other','ml_training'], [.1,.9], source='program')
        self.assertEqual(operational_metrics(program)['timing_eligible_jobs'],0)
        censored=self.frame([0,15], ['ml_training']*2, [.1,.9])
        self.assertEqual(operational_metrics(censored)['timing_censored_jobs'],1)
        gap=self.frame([0,180], ['other','ml_training'], [.1,.9])
        self.assertEqual(operational_metrics(gap)['timing_censored_jobs'],1)

    def test_invalid_predictions_and_duplicate_arms_fail(self):
        f=self.frame([0], ['other'], [.9])
        with self.assertRaises(ValueError):
            operational_metrics(pd.concat([f,f]))
        f.loc[0,'p_training']=float('nan')
        with self.assertRaises(ValueError):
            operational_metrics(f)
        with self.assertRaises(ValueError):
            operational_metrics(f, stride_sec=0)

    def test_independent_job_exposure_and_empty_denominators(self):
        f=self.frame([0,15], ['other']*2, [.1,.1])
        both=pd.concat([f,f.assign(job_id='other-job')])
        self.assertEqual(operational_metrics(both)['benign_job_hours'],90/3600)
        empty=operational_metrics(f.iloc[:0])
        self.assertIsNone(empty['job_false_positive_rate'])
        self.assertIsNone(empty['false_alert_episodes_per_job_hour'])


class PlanTests(unittest.TestCase):
    def test_repeats_are_unique_and_shared_between_nodes(self):
        jobs=[dict(family='train_tp',hardware='NVIDIA H100 80GB HBM3',cell='s16_train_tp8',
                   excluded_failed_job='False')]
        rows=make_plan(jobs)
        self.assertEqual(len(rows),5)
        self.assertEqual(len({r['run_stamp'] for r in rows}),5)
        self.assertEqual({r['day_slot'] for r in rows},{1,2,3})
        self.assertEqual(rows,make_plan(jobs))
        self.assertIn("ONLY_CELLS='^s16_train_tp8$'",rows[0]['queue_command'])
        self.assertTrue(all(r['status']=='planned_not_launched' for r in rows))
        with self.assertRaises(ValueError):
            make_plan(jobs,launches=2,days=3)

    def test_evaluation_plan_pins_protocol_and_does_not_fetch_or_provision(self):
        plan=commands(Path('/tmp/analysis'),2,300)
        self.assertEqual([n for n,_ in plan],['cache','mechanism_paired','backbone','steps_only'])
        self.assertTrue(all(c[0].endswith('python3') or 'python' in c[0] for _,c in plan))
        self.assertIn('--exclude-campaigns', plan[1][1])
        self.assertIn('backbone', plan[2][1])


if __name__ == '__main__':
    unittest.main()
