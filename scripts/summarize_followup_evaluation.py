#!/usr/bin/env python3
"""Render reviewed holdout results as a paper-oriented report and family table."""
import argparse
import csv
import json
from pathlib import Path


def percent(value):
    return '—' if value is None else f'{100 * value:.2f}%'


def number(value):
    return '—' if value is None else f'{value:.3f}'


def table(headers, rows):
    return '\n'.join(['| ' + ' | '.join(headers) + ' |',
                       '| ' + ' | '.join(['---'] * len(headers)) + ' |'] +
                      ['| ' + ' | '.join(map(str, row)) + ' |' for row in rows])


def render(results, reviews, include_plots=False):
    main = results['mechanism_paired']
    audit = reviews['evaluations']['mechanism_paired']
    attacks = ', '.join(f'`{f}`' for f in audit['adversarial_families']) or 'none'
    single_jobs = ', '.join(f'`{f}`' for f in audit['single_job_families']) or 'none'
    rows = []
    for name, title in [('mechanism_paired', 'Mechanism holdout'), ('backbone', 'Backbone holdout'),
                         ('steps_only', 'STEP-only mechanism holdout')]:
        r = results[name]
        m, op = r['all'], r['operational']
        rows.append([title, len(r['families']), r['n_jobs'], r['n_windows'], percent(m['accuracy']),
                     number(m['macro_f1']), percent(m['training_recall']), percent(m['false_positive_rate']),
                     percent(m['job_any_fpr']), number(op['false_alert_episodes_per_job_hour'])])
    out = ['# Corrected multi-node classifier evaluations', '',
        'These results use the frozen October 12 input snapshot, with corrected physical-job identities, '
        'window labels, recovered NIC/log inputs, and exhaustive family holdouts. '
        'The two subsequently recovered Spot-preempted A100 captures remain in the corpus but are **not** '
        'part of this evaluation snapshot.', '',
        'Read the [review and priorities](REVIEW.md), [reproduction instructions](REPRODUCING.md), '
        '[storage and recovery notes](DURABILITY.md), and [sensitivity interpretation](SENSITIVITY_REVIEW.md).', '',
        '## Accuracy and detection', '',
        table(['Evaluation', 'Families', 'Jobs', 'Windows', 'Accuracy', 'Macro F1', 'Training recall',
               'Non-training window FPR', 'Benign job FPR', 'Alert episodes/benign job-hour'], rows), '',
        'Accuracy and macro F1 use three-class argmax predictions. Detection and false-positive rates '
        'use training probability ≥ 0.5. This threshold was fixed before testing; it was not calibrated '
        'to an operational false-alert budget. A job is alerted if any observed window crosses the threshold.', '',
        'The window FPR includes non-training windows within mixed jobs. The benign-job FPR includes only '
        'jobs with no training-labelled windows. Any-window training-job recall can count an alert in a '
        'non-training phase; the stricter supplementary count below requires an alert in an observed training window.', '',
        'Each model is a 300-tree random forest (seed 0, max_features=0.3, min_samples_leaf=2), '
        'with medians fitted only on its training fold. Windows span 30 seconds with a 15-second stride. '
        f'The full feature vocabulary has {len(main["features"])} telemetry features. '
        'The backbone grouping specifically merges programs sharing the custom RandomTransformerLM implementation; '
        'it is not a comprehensive independent-implementation or architecture holdout. '
        'The backbone and STEP-only rows are separate evaluations; their score differences are not '
        'paired estimates of a label-quality or feature effect.', '',
        '## Paired October campaign ablation', '',
        'The baseline removes jobs from campaigns `20261001` and `20261003` only from training. '
        'Both arms score exactly the same old-job test windows while holding out the same complete family. '
        'New-job test results are reported separately.', '',
        table(['Test population / arm', 'Jobs', 'Windows', 'Accuracy', 'Macro F1', 'Training recall', 'Window FPR'],
              [[title, main[key]['jobs'], main[key]['windows'], percent(main[key]['accuracy']),
                number(main[key]['macro_f1']), percent(main[key]['training_recall']),
                percent(main[key]['false_positive_rate'])] for key, title in
               [('baseline_old_jobs', 'Old jobs / baseline'), ('paired_old_jobs', 'Old jobs / full'),
                ('new_jobs', 'New jobs / full')]]), '',
        f'Family-macro paired accuracy change: **{100 * main["paired_family_accuracy_delta"]:+.2f} percentage points**; '
        f'95% family-bootstrap interval [{100 * main["paired_family_accuracy_delta_bootstrap95"][0]:+.2f}, '
        f'{100 * main["paired_family_accuracy_delta_bootstrap95"][1]:+.2f}] points '
        f'over {main["paired_family_count"]} paired families (2,000 resamples, seed 0). '
        'This interval describes family resampling of these saved predictions, not independent campaigns, '
        'forest seeds, or future deployments.', '',
        '## Held-out attack performance', '',
        table(['Workload kind', 'Jobs', 'Windows', 'Accuracy', 'Training recall', 'Training-job recall', 'Window FPR'],
              [[key, m['jobs'], m['windows'], percent(m['accuracy']), percent(m['training_recall']),
                percent(m['job_any_recall']), percent(m['false_positive_rate'])]
               for key, m in main['by_kind'].items()]), '',
        'The adversarial row measures the existing attack-labelled workloads under mechanism-family holdout. '
        f'The eligible attack data covers {len(audit["adversarial_families"])} mechanism groups: {attacks}. '
        'It does not evaluate a new adaptive attacker against a frozen deployed detector, certify robustness, '
        'or establish that the attacks preserved useful learning.', '',
        table(['Training-job population', 'Training jobs', 'Detected during a training window', 'Recall during training'],
              [[key, m['training_jobs'], m['detected_during_training_jobs'], percent(m['recall'])]
               for key, m in audit['training_phase_job_detection'].items()]), '',
        table(['Logged training share', 'Jobs represented', 'Windows', 'Training recall', 'Window FPR'],
              [[key, m['jobs'], m['windows'], percent(m['training_recall']), percent(m['false_positive_rate'])]
               for key, m in sorted(main.get('by_train_fraction', {}).items(),
                   key=lambda item: ['0%', '0-5%', '5-20%', '20-50%', '50-100%'].index(item[0]))]), '',
        'Training share is the logged fraction of window time attributed to training, not preserved throughput, '
        'energy efficiency, or learning utility. A job can appear in several share bins. These bins include '
        'all STEP-labelled workloads, rather than only attacks.', '',
        '## Generalization and label provenance', '',
        table(['Hardware', 'Jobs', 'Windows', 'Accuracy', 'Training recall', 'Window FPR'],
              [[key, m['jobs'], m['windows'], percent(m['accuracy']), percent(m['training_recall']),
                percent(m['false_positive_rate'])] for key, m in main['by_hardware'].items()]), '',
        'These hardware strata come from mechanism holdouts; they are not leave-one-hardware-out tests.', '',
        table(['Label source', 'Jobs represented', 'Windows', 'Accuracy', 'Training recall', 'Window FPR'],
              [[key, m['jobs'], m['windows'], percent(m['accuracy']), percent(m['training_recall']),
                percent(m['false_positive_rate'])] for key, m in main['by_label_source'].items()]), '',
        f'{main["unresolved_excluded"]["windows"]} eligible multi-node windows from '
        f'{main["unresolved_excluded"]["jobs"]} jobs lack resolved work labels and were excluded before fitting '
        'or scoring. Program-derived labels are weaker than recorded STEP evidence and do not establish activity onset.', '',
        '## Operational exposure and delay', '']
    operational_rows = []
    for name, r in results.items():
        op = r['operational']
        operational_rows.append([name, op['benign_jobs'], number(op['benign_job_hours']),
            op['false_alert_jobs'], op['false_alert_episodes'], op['timing_eligible_jobs'],
            op['timing_missed_jobs'], op['timing_censored_jobs'],
            f'{number(op["detected_delay_lower_median_s"])}–{number(op["detected_delay_upper_median_s"])}'])
    out += [table(['Evaluation', 'Benign jobs', 'Observed benign job-hours', 'False-alert jobs', 'Alert episodes',
                   'Onset-eligible jobs', 'Missed jobs', 'Censored STEP onsets', 'Detected median delay bounds (s)'],
                  operational_rows), '',
        'Exposure is the union of observed windows in entirely non-training-labelled jobs. Overlap and '
        'unobserved gaps add no exposure. Mixed jobs are excluded from the benign-hour denominator. '
        'An alert episode starts after a negative window or a gap greater than 1.5 strides. '
        'Delay bounds require adjacent STEP-backed non-training and training windows; program-only onsets '
        'and left/gap-censored onsets cannot establish delay. Delay quantiles include detected eligible jobs only; '
        'misses are reported separately. Decisions occur at window end and omit computation/transport latency.', '',
        '## Per-family results', '',
        *(['![Held-out accuracy and training recall for all 43 mechanism families](mechanism_families.png)', '',
           '[Vector figure](mechanism_families.svg)', ''] if include_plots else []),
        '[All evaluation families as CSV](family_results.csv)', '',
        'Mechanism holdouts, ordered by three-class accuracy. Small families can have extreme scores with little '
        f'independent evidence. Families with only one eligible job: {single_jobs}. '
        'See `family_results.csv` for all three evaluations.', '',
        table(['Family', 'Jobs', 'Windows', 'Accuracy', 'Training recall', 'Window FPR'],
              [[m['family'], m['jobs'], m['windows'], percent(m['accuracy']), percent(m['training_recall']),
                percent(m['false_positive_rate'])] for m in sorted(
                    reviews['evaluations']['mechanism_paired']['family_results'], key=lambda r: (r['accuracy'], r['family']))]), '',
        '## Review and reproducibility', '',
        'The result audit checks cache hashes and fingerprint, exact target and window identity, exhaustive '
        'coverage, job/family separation, every training and test manifest, campaign ablation, paired test '
        'identity, and recomputed classification and operational scores. It does not refit the forests or '
        'establish ground truth that was never recorded. See `artifacts/review.json`, `artifacts/validation.json`, '
        'and `artifacts/manifest.json` for checks, execution details, and durable object locations.', '',
        'Predictions and compressed split manifests are stored with Git LFS. Keep bulk objects on an external '
        'volume when reproducing this evaluation. The manifest identifies the frozen cache and source/input '
        'archive in cloud storage; use those exact inputs rather than a newly rebuilt cache from the expanded corpus.', '',
        '```bash',
        'PYTHONPATH=. python scripts/review_followup_evaluation.py \\',
        '  --results docs/followup_evaluation_2026-10-13/artifacts \\',
        '  --windows /external-volume/windows_30_15.parquet \\',
        '  --out /external-volume/review.json',
        '```', '',
        '## Limits and next experiments', '',
        '- Repeat sparsely sampled failure families using independent seeds and collection days; group correlated retries.',
        '- Calibrate thresholds on separate jobs and test the frozen alert rule on long benign serving/HPC runs.',
        '- Add hardware/site and chronological holdouts, matched GPU-only/NIC ablations, and missing-sensor tests.',
        '- Evaluate fresh adaptive attacks against a frozen defender while measuring preserved useful learning and cost.',
        '- Add job/campaign-aware uncertainty and model-seed repetitions before making broad performance claims.',
        '- Run a separately versioned evaluation including the admitted preempted A100 intervals; preserve this snapshot.', '']
    return '\n'.join(out)


def plot_families(review, out):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.ticker import PercentFormatter
    rows = sorted(review['family_results'], key=lambda r: (r['accuracy'], r['family']))
    y = list(range(len(rows)))
    fig, axes = plt.subplots(1, 2, figsize=(11, 14), sharey=True)
    axes[0].barh(y, [r['accuracy'] for r in rows], height=.7, color='#487a9d')
    valid = [(i, r) for i, r in enumerate(rows) if r['training_recall'] is not None]
    axes[1].barh([i for i, _ in valid], [r['training_recall'] for _, r in valid], height=.7, color='#9f552f')
    for i, r in enumerate(rows):
        if r['training_recall'] is None:
            axes[1].text(.02, i, 'no training windows', va='center', fontsize=7, color='#666666')
    axes[0].set_yticks(y, [f'{r["family"]}  (n={r["jobs"]})' for r in rows], fontsize=8)
    axes[0].invert_yaxis()
    for ax, title in zip(axes, ['Three-class accuracy', 'Training recall at p ≥ 0.5']):
        ax.set_xlim(0, 1)
        ax.xaxis.set_major_formatter(PercentFormatter(1))
        ax.set_title(title, fontsize=11)
        ax.grid(axis='x', alpha=.2)
        ax.set_axisbelow(True)
        ax.spines[['top', 'right']].set_visible(False)
        ax.tick_params(axis='y', length=0)
    fig.suptitle('Exhaustive mechanism-family holdouts', fontsize=14)
    fig.text(.5, .012, 'n = physical jobs; overlapping windows are not independent repetitions. '
             'Fixed threshold, not deployment-calibrated.', ha='center', fontsize=8)
    fig.tight_layout(rect=(0, .03, 1, .97))
    fig.savefig(out / 'mechanism_families.svg')
    fig.savefig(out / 'mechanism_families.png', dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--results', type=Path, required=True)
    ap.add_argument('--review', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--plots', action='store_true', help='Also export the mechanism-family figure (requires matplotlib)')
    args = ap.parse_args()
    reviews = json.loads(args.review.read_text())
    if reviews['status'] != 'passed':
        raise ValueError('Results must pass the audit before rendering')
    results = {name: json.loads((args.results / (name + '.json')).read_text())
               for name in ['mechanism_paired', 'backbone', 'steps_only']}
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / 'README.md').write_text(render(results, reviews,
        include_plots=args.plots or (args.out / 'mechanism_families.png').exists()))
    fields = ['evaluation', 'family', 'jobs', 'windows', 'training_windows', 'nontraining_windows', 'accuracy', 'macro_f1', 'training_recall',
              'false_positive_rate', 'job_any_recall', 'job_any_fpr']
    with (args.out / 'family_results.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fields, extrasaction='ignore')
        writer.writeheader()
        for name, r in reviews['evaluations'].items():
            writer.writerows(dict(evaluation=name, **row) for row in r['family_results'])
    if args.plots:
        plot_families(reviews['evaluations']['mechanism_paired'], args.out)


if __name__ == '__main__':
    main()
