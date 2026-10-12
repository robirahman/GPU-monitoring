"""Operational accounting for held-out, pooled-job window predictions.

Threshold is supplied, never selected from these predictions. Decision time is
window_end. Benign exposure is the union of observed window intervals for jobs
with no training-labeled window. It excludes unobserved gaps and mixed jobs.
An alert episode starts on a positive after a negative or a cadence gap.
"""
import numpy as np


def interval_union_seconds(intervals):
    end, total = None, 0.0
    for start, stop in sorted(intervals):
        total += max(0.0, stop - max(start, end if end is not None else start))
        end = max(stop, end if end is not None else stop)
    return total


def operational_metrics(predictions, threshold=0.5, stride_sec=15.0):
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError('threshold must be finite and between 0 and 1')
    if not np.isfinite(stride_sec) or stride_sec <= 0:
        raise ValueError('stride_sec must be finite and positive')
    required = {'job_id', 'window_start', 'window_end', 'truth', 'p_training'}
    if not required <= set(predictions):
        raise ValueError(f'Missing columns: {sorted(required - set(predictions))}')
    p = predictions.copy()
    if p.job_id.isna().any() or p.truth.isna().any():
        raise ValueError('Job identity and truth must be present')
    if not set(p.truth) <= {'ml_training', 'ml_inference', 'other'}:
        raise ValueError('Unresolved labels must be counted and excluded before scoring')
    numeric = p[['window_start', 'window_end', 'p_training']].to_numpy(float)
    if (not np.isfinite(numeric).all() or (p.window_end <= p.window_start).any()
            or not p.p_training.between(0, 1).all()):
        raise ValueError('Invalid times or probabilities')
    if p.duplicated(['job_id', 'window_start', 'window_end']).any():
        raise ValueError('Expected one prediction per job/window; select a single evaluation arm')
    benign_hours = 0.0
    benign_jobs = false_jobs = episodes = training_jobs = 0
    timing_eligible = timing_missed = timing_censored = pretraining_alert_jobs = 0
    delays_low, delays_high = [], []
    for _, g in p.groupby('job_id', sort=False):
        g = g.sort_values(['window_end', 'window_start'])
        flags = g.p_training.to_numpy() >= threshold
        training = g.truth.eq('ml_training').to_numpy()
        ends = g.window_end.to_numpy(float)
        if not training.any():
            benign_jobs += 1
            false_jobs += int(flags.any())
            benign_hours += interval_union_seconds(zip(g.window_start, g.window_end)) / 3600
            previous = False
            for i, flag in enumerate(flags):
                gap = i > 0 and ends[i] - ends[i-1] > 1.5 * stride_sec
                episodes += int(flag and (not previous or gap))
                previous = bool(flag)
            continue
        training_jobs += 1
        first = int(np.flatnonzero(training)[0])
        pretraining_alert_jobs += int(flags[:first].any())
        # Program labels include startup/load: they cannot establish onset.
        # Missing label provenance is likewise not temporal ground truth.
        if 'label_source' not in g or g.iloc[first].label_source != 'steps':
            continue
        # An already-training first observation or a gap before training cannot
        # bound onset. Require an adjacent STEP-backed negative observation.
        if (first == 0 or g.iloc[first-1].label_source != 'steps'
                or ends[first-1] < g.iloc[first].window_start
                or ends[first] - ends[first-1] > 1.5 * stride_sec):
            timing_censored += 1
            continue
        timing_eligible += 1
        hits = np.flatnonzero(flags & (ends >= ends[first]))
        if not len(hits):
            timing_missed += 1
            continue
        alert_end = ends[hits[0]]
        delays_low.append(float(alert_end - ends[first]))
        delays_high.append(float(alert_end - g.iloc[first].window_start))
    return dict(threshold=float(threshold), stride_sec=float(stride_sec),
        rule='any positive window; episodes rearm after a negative or >1.5-stride gap',
        exposure='union of observed windows in jobs with no training-labeled windows',
        decision_time='window_end; computation/transport delay not included',
        benign_jobs=benign_jobs, benign_job_hours=benign_hours,
        false_alert_jobs=false_jobs, false_alert_episodes=episodes,
        job_false_positive_rate=false_jobs/benign_jobs if benign_jobs else None,
        false_alert_episodes_per_job_hour=episodes/benign_hours if benign_hours else None,
        training_jobs=training_jobs, pretraining_alert_jobs=pretraining_alert_jobs,
        timing_eligible_jobs=timing_eligible, timing_missed_jobs=timing_missed,
        timing_censored_jobs=timing_censored,
        timing_detected_jobs=len(delays_low),
        detection_delay_bounds_note='Onset bounded by first STEP-labeled training window, requiring an adjacent '
                                   'STEP-backed negative. Already-training first observations and gaps are censored. '
                                   'Quantiles are detected jobs only; program-only onsets excluded. Misses reported separately.',
        detected_delay_lower_median_s=float(np.median(delays_low)) if delays_low else None,
        detected_delay_upper_median_s=float(np.median(delays_high)) if delays_high else None,
        detected_delay_lower_p90_s=float(np.quantile(delays_low, .9)) if delays_low else None,
        detected_delay_upper_p90_s=float(np.quantile(delays_high, .9)) if delays_high else None)
