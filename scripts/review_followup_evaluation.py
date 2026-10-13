#!/usr/bin/env python3
"""Audit saved holdout predictions and splits against their frozen window cache.

This never fits a model or selects a threshold. It fails closed on incomplete
coverage, changed targets, leakage, an incorrect ablation, or inconsistent scores.
Run from the repository root with PYTHONPATH=.; inputs may be on external storage.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

from classifier.operational_metrics import operational_metrics
from classifier.rebuttal_analysis.multinode_family_holdout import load_multinode
from classifier.threeway_improved import get_feature_cols
from classifier.workload_labels import workload_family


def require(condition, message):
    if not condition:
        raise ValueError(message)


def digest(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    path = Path(path)
    opener = gzip.open if path.suffix == '.gz' else open
    with opener(path, 'rt') as f:
        return json.load(f)


def close(actual, expected, name):
    if isinstance(expected, dict):
        for key, value in expected.items():
            require(key in actual, f'{name}: missing {key}')
            close(actual[key], value, f'{name}.{key}')
    elif isinstance(expected, (list, tuple)):
        require(len(actual) == len(expected), f'{name}: length differs')
        for i, value in enumerate(expected):
            close(actual[i], value, f'{name}[{i}]')
    elif isinstance(expected, (float, np.floating)):
        require(actual is not None and np.isclose(actual, expected, atol=1e-10, rtol=1e-10),
                f'{name}: {actual} != {expected}')
    else:
        require(actual == expected, f'{name}: {actual} != {expected}')


def score(part, threshold):
    """Recompute published classification statistics from saved predictions."""
    positive = part.truth.eq('ml_training')
    flag = part.p_training.ge(threshold)
    jobs = pd.DataFrame({'job': part.job_id, 'positive': positive, 'flag': flag}).groupby('job').max()
    return dict(windows=len(part), jobs=len(jobs),
                accuracy=float(accuracy_score(part.truth, part.prediction)),
                macro_f1=float(f1_score(part.truth, part.prediction, average='macro', zero_division=0)),
                training_recall=float(flag[positive].mean()) if positive.any() else None,
                false_positive_rate=float(flag[~positive].mean()) if (~positive).any() else None,
                job_any_recall=float(jobs.loc[jobs.positive, 'flag'].mean()) if jobs.positive.any() else None,
                job_any_fpr=float(jobs.loc[~jobs.positive, 'flag'].mean()) if (~jobs.positive).any() else None,
                confusion=confusion_matrix(part.truth, part.prediction,
                    labels=['ml_training', 'ml_inference', 'other']).tolist())


def training_phase_jobs(part, threshold):
    """Distinguish an alert during observed training from an earlier false alert."""
    training = part[part.truth.eq('ml_training')]
    detected = training.assign(flag=training.p_training.ge(threshold)).groupby('job_id').flag.any()
    return dict(training_jobs=len(detected), detected_during_training_jobs=int(detected.sum()),
                recall=float(detected.mean()) if len(detected) else None)


def review(result, predictions, splits, windows, *, excluded_campaigns=()):
    require(result['protocol'] == 'exhaustive_leave_one_family_out', 'Unexpected protocol')
    require(result['target'] == 'threeway_window_work', 'Unexpected target')
    require(not predictions.duplicated(['arm', 'row_id']).any(), 'Duplicate predictions')
    require(predictions.p_training.between(0, 1).all(), 'Invalid probabilities')
    require(set(predictions.prediction) <= {'ml_training', 'ml_inference', 'other'}, 'Invalid predictions')
    w = windows.copy()
    if result.get('steps_only'):
        w = w[w.label_source.eq('steps')].copy()
    w['family'] = w.grp.map(lambda x: workload_family(x, result['family_level']))
    identity_cols = ['job_id', 'run_id', 'window_start', 'window_end']
    w['row_id'] = w[identity_cols].astype(str).agg('|'.join, axis=1).map(
        lambda x: hashlib.sha256(x.encode()).hexdigest())
    require(not w.row_id.duplicated().any(), 'Duplicate cache identities')
    require(w.groupby('job_id').family.nunique().max() == 1, 'Job crosses families')
    close(result['n_windows'], len(w), 'n_windows')
    close(result['n_jobs'], w.job_id.nunique(), 'n_jobs')
    close(result['families'], sorted(w.family.unique()), 'families')
    close(result['features'], get_feature_cols(w), 'features')
    new_jobs = set(result['ablated_jobs'])
    if excluded_campaigns:
        campaigns = set(excluded_campaigns)
        hit = w.meta_campaign.fillna('').map(lambda x: bool(set(x.split('|')) & campaigns))
        require(new_jobs == set(w.loc[hit, 'job_id']), 'Wrong campaign ablation')
        require(bool(new_jobs), 'Empty requested ablation')
    else:
        require(not new_jobs, 'Unexpected ablation')
    full = predictions[predictions.arm.eq('full')].copy()
    require(set(full.row_id) == set(w.row_id) and len(full) == len(w), 'Incomplete full coverage')
    expected = w.set_index('row_id').loc[predictions.row_id]
    for field in identity_cols + ['family']:
        require(np.array_equal(predictions[field].to_numpy(), expected[field].to_numpy()),
                f'Prediction {field} differs from cache')
    require(np.array_equal(predictions.truth.to_numpy(), expected.threeway_label.to_numpy()),
            'Prediction truth differs from cache')
    for field, source in [('hardware', 'gpu_name'), ('kind', 'kind'),
                          ('label_source', 'label_source'), ('job_label', 'job_label')]:
        if source in expected:
            require(np.array_equal(predictions[field].fillna('').to_numpy(), expected[source].fillna('').to_numpy()),
                    f'Prediction {field} differs from cache')
    if 'label_train_frac' in expected:
        require(np.allclose(predictions.train_frac, expected.label_train_frac, equal_nan=True),
                'Training share differs from cache')
    require(np.array_equal(predictions.paired_test.to_numpy(), ~predictions.job_id.isin(new_jobs)),
            'Wrong paired-test flags')
    split_map = {(s['family'], s['arm']): s for s in splits}
    require(len(split_map) == len(splits), 'Duplicate split records')
    expected_keys = set()
    for family in sorted(w.family.unique()):
        for arm in ['full', 'baseline']:
            test = w[w.family.eq(family)]
            train = w[~w.family.eq(family)]
            if arm == 'baseline':
                test = test[~test.job_id.isin(new_jobs)]
                train = train[~train.job_id.isin(new_jobs)]
                if not new_jobs or test.empty:
                    continue
            key = (family, arm)
            expected_keys.add(key)
            require(key in split_map, f'Missing split {key}')
            s = split_map[key]
            for field, values in [('train_jobs', train.job_id), ('train_ids', train.row_id), ('test_ids', test.row_id)]:
                require(len(s[field]) == len(set(values)) and set(s[field]) == set(values),
                        f'{key}: incorrect {field}')
            require(not set(train.job_id) & set(test.job_id), f'{key}: job leakage')
            part = predictions[predictions.family.eq(family) & predictions.arm.eq(arm)]
            require(set(part.row_id) == set(test.row_id), f'{key}: incorrect prediction coverage')
    require(set(split_map) == expected_keys, 'Unexpected splits')
    require(set(zip(predictions.family, predictions.arm)) == expected_keys, 'Unexpected prediction arms')
    threshold = result['threshold']
    close(result['all'], score(full, threshold), 'all')
    for population, part in [('paired_old_jobs', full[full.paired_test]), ('new_jobs', full[~full.paired_test])]:
        if len(part):
            close(result[population], score(part, threshold), population)
    for axis in ['hardware', 'kind', 'label_source']:
        require(set(result['by_' + axis]) == set(map(str, full[axis].unique())),
                f'{axis}: incorrect strata')
        for key, part in full.groupby(axis, dropna=False):
            close(result['by_' + axis][str(key)], score(part, threshold), f'{axis}.{key}')
    expected_score_keys = {(family, 'full', 'all') for family in w.family.unique()}
    for family, arm in expected_keys:
        if arm == 'baseline':
            expected_score_keys.update([(family, 'full', 'old_jobs'), (family, 'baseline', 'old_jobs')])
    actual_score_keys = [(r['family'], r['arm'], r['test_population']) for r in result['per_fold']]
    require(len(actual_score_keys) == len(expected_score_keys) and set(actual_score_keys) == expected_score_keys,
            'Missing or duplicate fold scores')
    for record in result['per_fold']:
        part = predictions[predictions.family.eq(record['family']) & predictions.arm.eq(record['arm'])]
        if record['test_population'] == 'old_jobs':
            part = part[part.paired_test]
        close(record, score(part, threshold), f"fold.{record['family']}.{record['arm']}")
    known = full[full.train_frac.notna()].copy()
    if len(known):
        known['share_bin'] = pd.cut(known.train_frac, [-np.inf, 0, .05, .2, .5, np.inf],
                                    labels=['0%', '0-5%', '5-20%', '20-50%', '50-100%'])
        for label, part in known.groupby('share_bin', observed=True):
            close(result['by_train_fraction'][str(label)], score(part, threshold), f'train_fraction.{label}')
    stride = result['operational']['stride_sec']
    close(result['operational'], operational_metrics(full, threshold, stride), 'operational')
    for family, part in full.groupby('family'):
        close(result['operational_by_family'][family], operational_metrics(part, threshold, stride), family)
    if new_jobs:
        baseline = predictions[predictions.arm.eq('baseline')]
        old = full[full.paired_test]
        require(set(old.row_id) == set(baseline.row_id), 'Arms have different paired test windows')
        close(result['baseline_old_jobs'], score(baseline, threshold), 'baseline_old_jobs')
        for name, part in [('operational_baseline_old_jobs', baseline), ('operational_full_old_jobs', old)]:
            close(result[name], operational_metrics(part, threshold, stride), name)
        deltas = []
        for family in sorted(old.family.unique()):
            a, b = old[old.family.eq(family)], baseline[baseline.family.eq(family)]
            deltas.append(float(a.truth.eq(a.prediction).mean() - b.truth.eq(b.prediction).mean()))
        close(result['paired_family_count'], len(deltas), 'paired family count')
        close(result['paired_family_accuracy_delta'], float(np.mean(deltas)), 'paired delta')
        rng = np.random.default_rng(result['model']['seed'])
        boot = rng.choice(deltas, size=(2000, len(deltas)), replace=True).mean(axis=1)
        close(result['paired_family_accuracy_delta_bootstrap95'], np.quantile(boot, [.025, .975]).tolist(), 'paired CI')
    family_rows = []
    for family, part in full.groupby('family'):
        family_rows.append(dict(family=family, training_windows=int(part.truth.eq('ml_training').sum()),
                                nontraining_windows=int(part.truth.ne('ml_training').sum()), **score(part, threshold)))
    return dict(status='passed', families=len(result['families']), windows=len(w), jobs=w.job_id.nunique(),
                split_records=len(splits), prediction_rows=len(predictions),
                adversarial_families=sorted(full.loc[full.kind.eq('adversarial'), 'family'].unique()),
                single_job_families=[r['family'] for r in family_rows if r['jobs'] == 1],
                training_phase_job_detection={
                    name: training_phase_jobs(part, threshold) for name, part in
                    [('all', full), ('adversarial', full[full.kind.eq('adversarial')])]},
                checks=['cache targets and identities', 'exhaustive window coverage', 'family and job separation',
                        'exact train/test manifests', 'campaign ablation', 'paired test identity',
                        'classification metric recomputation', 'operational metric consistency'],
                family_results=family_rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--windows', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    cache = read_json(str(args.windows) + '.provenance.json')
    cache_sha = digest(args.windows)
    require(cache_sha == cache['artifact_sha256'], 'Window cache hash mismatch')
    content = {k: v for k, v in cache.items() if k not in {'fingerprint', 'artifact_sha256'}}
    fingerprint = hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()
    require(fingerprint == cache['fingerprint'], 'Cache manifest fingerprint mismatch')
    windows = load_multinode(args.windows)
    reviews = {}
    artifacts = {str(args.windows.name): cache_sha,
                 args.windows.name + '.provenance.json': digest(str(args.windows) + '.provenance.json')}
    for name in ['mechanism_paired', 'backbone', 'steps_only']:
        result_path = args.results / (name + '.json')
        pred_path = args.results / (name + '.predictions.parquet')
        split_path = args.results / (name + '.splits.json')
        if not split_path.exists():
            split_path = Path(str(split_path) + '.gz')
        result = read_json(result_path)
        require(result['windows_sha256'] == cache_sha and result['cache_fingerprint'] == fingerprint,
                f'{name}: cache provenance mismatch')
        close(result['code'], cache['code'], f'{name}.code')
        close(result['libraries'], cache['libraries'], f'{name}.libraries')
        close(result['threshold'], .5, f'{name}.threshold')
        close(result['model'], dict(type='RandomForestClassifier', trees=300, seed=0,
              max_features=.3, min_samples_leaf=2, imputation='training-fold median'), f'{name}.model')
        close(result['family_level'], 'backbone' if name == 'backbone' else 'mechanism', name)
        close(result['steps_only'], name == 'steps_only', name)
        close(result['unresolved_excluded'], windows.attrs.get('unresolved'), f'{name}.unresolved')
        reviews[name] = review(result, pd.read_parquet(pred_path), read_json(split_path), windows,
                              excluded_campaigns=['20261001', '20261003'] if name == 'mechanism_paired' else [])
        for path in [result_path, pred_path, split_path]:
            artifacts[path.name] = digest(path)
    out = dict(status='passed', cache_fingerprint=fingerprint, artifacts_sha256=artifacts, evaluations=reviews,
               limitations=['Checks saved outputs against the frozen cache; does not refit forests.',
                            'Operational consistency uses the separately regression-tested accounting function.',
                            'Metadata cannot establish unrecorded ground truth or independent repetitions.'])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2, allow_nan=False) + '\n')
    print(json.dumps({name: {k: v for k, v in r.items() if k != 'family_results'} for name, r in reviews.items()}, indent=2))


if __name__ == '__main__':
    main()
