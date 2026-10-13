"""Regression tests for detecting corruption in saved scientific result bundles."""
import copy
import unittest

import pandas as pd

from classifier.rebuttal_analysis.paired_family_lofo import evaluate
from scripts.review_followup_evaluation import review, training_phase_jobs


class ResultReviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        rows = []
        for label, prefix in [('ml_training', 'train'), ('ml_inference', 'infer'), ('other', 'nonml')]:
            for family in ['a', 'b']:
                for campaign in ['old', '20261001']:
                    job = f'{prefix}_{family}_{campaign}'
                    for t in [0., 15.]:
                        rows.append(dict(grp=f'{prefix}_{family}', job_id=job, run_id=job,
                            window_start=t, window_end=t + 30, signal_mean=t,
                            threeway_label=label, label_source='steps', job_label=label,
                            kind='honest', gpu_name='fixture', meta_campaign=campaign,
                            label_train_frac=.03 if label == 'ml_training' else 0.))
        cls.windows = pd.DataFrame(rows)
        new = set(cls.windows.loc[cls.windows.meta_campaign.eq('20261001'), 'job_id'])
        cls.result, cls.predictions, cls.splits = evaluate(cls.windows, new, trees=2)

    def check(self, result=None, predictions=None, splits=None):
        return review(result if result is not None else self.result,
                      predictions if predictions is not None else self.predictions,
                      splits if splits is not None else self.splits, self.windows,
                      excluded_campaigns=['20261001', '20261003'])

    def test_valid_exhaustive_bundle(self):
        r = self.check()
        self.assertEqual(r['status'], 'passed')
        self.assertEqual(r['families'], 6)
        self.assertEqual(r['windows'], 24)

    def test_rejects_changed_target(self):
        p = self.predictions.copy()
        p.loc[0, 'truth'] = 'ml_training'
        if p.loc[0, 'truth'] == self.predictions.loc[0, 'truth']:
            p.loc[0, 'truth'] = 'other'
        with self.assertRaisesRegex(ValueError, 'truth differs'):
            self.check(predictions=p)

    def test_rejects_missing_test_window(self):
        with self.assertRaisesRegex(ValueError, 'Incomplete full coverage'):
            self.check(predictions=self.predictions.drop(self.predictions[self.predictions.arm.eq('full')].index[0]))

    def test_rejects_leaked_training_id(self):
        s = copy.deepcopy(self.splits)
        s[0]['train_ids'].append(s[0]['test_ids'][0])
        with self.assertRaisesRegex(ValueError, 'incorrect train_ids'):
            self.check(splits=s)

    def test_rejects_changed_baseline_population(self):
        s = copy.deepcopy(self.splits)
        base = next(x for x in s if x['arm'] == 'baseline')
        base['test_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'incorrect test_ids'):
            self.check(splits=s)

    def test_rejects_wrong_campaign_ablation(self):
        r = copy.deepcopy(self.result)
        r['ablated_jobs'].pop()
        with self.assertRaisesRegex(ValueError, 'Wrong campaign ablation'):
            self.check(result=r)

    def test_rejects_changed_accuracy(self):
        r = copy.deepcopy(self.result)
        r['all']['accuracy'] += .01
        with self.assertRaisesRegex(ValueError, 'all.accuracy'):
            self.check(result=r)

    def test_rejects_missing_fold_score(self):
        r = copy.deepcopy(self.result)
        r['per_fold'].pop()
        with self.assertRaisesRegex(ValueError, 'Missing or duplicate fold scores'):
            self.check(result=r)

    def test_rejects_invalid_probability(self):
        p = self.predictions.copy()
        p.loc[0, 'p_training'] = float('nan')
        with self.assertRaisesRegex(ValueError, 'Invalid probabilities'):
            self.check(predictions=p)

    def test_early_false_alert_is_not_training_phase_detection(self):
        p = pd.DataFrame(dict(job_id=['mixed', 'mixed', 'hit', 'benign'],
                             truth=['other', 'ml_training', 'ml_training', 'other'],
                             p_training=[.9, .1, .8, .9]))
        self.assertEqual(training_phase_jobs(p, .5),
                         dict(training_jobs=2, detected_during_training_jobs=1, recall=.5))


if __name__ == '__main__':
    unittest.main()
