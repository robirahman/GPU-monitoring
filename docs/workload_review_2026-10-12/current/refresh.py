"""Reconcile the verified inventory with committed October 12 recovery.

Run from the repository root. This updates accounting, not classifier results.
Historical stream/job identities and completion metadata are inherited. Historical
raw LFS payloads are not reread; input hashes bind this to the earlier full audit.
"""
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[3]
OUT = Path(__file__).resolve().parent
BASE = OUT.parent


def sha(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b''):
            h.update(chunk)
    return h.hexdigest()


def main():
    provenance = json.loads((BASE / 'provenance.json').read_text())
    source = Path(provenance['source_directory'])
    for directory, hashes in [(source, provenance['source_hashes']),
                              (BASE, provenance['output_sha256'])]:
        for name, expected in hashes.items():
            if sha(directory / name) != expected:
                raise ValueError(f'Changed audit input: {directory / name}')
    recovery_path = ROOT / 'data/gcs_recovery_20261012/manifest.json'
    recovery = json.loads(recovery_path.read_text())
    for row in recovery['files']:
        path = ROOT / row['path']
        if path.stat().st_size != row['bytes'] or sha(path) != row['sha256']:
            raise ValueError(f'Changed recovery output: {path}')

    jobs = pd.read_csv(BASE / 'jobs.csv', keep_default_na=False)
    streams = pd.read_csv(source / 'streams.csv', keep_default_na=False)
    streams = streams[(streams.role == 'accelerator') &
                      ~streams.quarantine.astype(str).str.lower().eq('true')].copy()
    workloads = pd.read_csv(BASE / 'workloads.csv', keep_default_na=False)
    admitted = [r for r in recovery['tpu'] if r['disposition'] == 'accepted_completed_training']
    run_ids = {r['run_id'] for r in admitted}
    if len(run_ids) != 1 or run_ids.intersection(set(streams.run_id)):
        raise ValueError('Recovery is no longer exactly one disjoint job')
    first = admitted[0]
    mask = (workloads.cell == first['cell']) & (workloads.gpu_name == first['hardware'])
    if mask.sum() != 1:
        raise ValueError('Expected one existing workload/hardware cell')
    old = workloads.loc[mask].iloc[0]
    new_streams = []
    for row in admitted:
        # Independently check the newly available Parquet timestamps and counts.
        frame = pd.read_parquet(ROOT / row['path'])
        col = next(c for c in ('ts', 'timestamp_epoch', 'timestamp') if c in frame)
        times = pd.to_numeric(frame[col]).dropna().unique()
        measured = dict(samples=len(times), t0=float(min(times)), t1=float(max(times)))
        for key, value in measured.items():
            if abs(value - row['recovered'][key]) > 1e-6:
                raise ValueError(f'Recovery metadata mismatch: {row["path"]}: {key}')
        new_streams.append(dict(cell=row['cell'], gpu_name=row['hardware'],
            span_s=measured['t1'] - measured['t0'], samples=measured['samples'],
            t0=measured['t0'], t1=measured['t1']))
    added = pd.DataFrame(new_streams)
    hours = float(added.span_s.sum() / 3600)
    elapsed = float(added.t1.max() - added.t0.min())
    new_job = dict(job_id=first['run_id'], cell=first['cell'], family=old.family,
        hardware=first['hardware'], snapshot_job_objective='ml_training',
        kind='adversarial', hosts=4, observed_devices=4, expected_devices=4,
        oct11_collection_status='not_in_oct11_snapshot', log_hosts=4,
        train_step_lines=sum(r['step_evidence']['steps'] for r in admitted),
        samples=int(added.samples.sum()), stream_hours=hours, covered_stream_hours=hours,
        elapsed_span_s=elapsed, job_objective='ml_training', category=old.category,
        excluded_failed_job=False, multi_host_observed=True,
        elapsed_job_hours=elapsed/3600, nonexcluded_accelerator_hours=hours,
        excluded_accelerator_hours=0)
    jobs = pd.concat([jobs, pd.DataFrame([new_job])], ignore_index=True)
    workloads['recovered_completed_jobs'] = 0
    for col, increment in [('streams', 4), ('jobs', 1), ('samples', int(added.samples.sum())),
                           ('stream_hours', hours), ('covered_stream_hours', hours),
                           ('elapsed_job_hours', elapsed/3600),
                           ('nonexcluded_accelerator_hours', hours),
                           ('recovered_completed_jobs', 1)]:
        workloads.loc[mask, col] += increment
    durations = pd.concat([streams.loc[(streams.cell == first['cell']) &
        (streams.gpu_name == first['hardware']), 'span_s'], added.span_s])
    for col, value in [('median_stream_seconds', durations.median()),
                       ('median_stream_minutes', durations.median()/60),
                       ('min_stream_seconds', durations.min()),
                       ('max_stream_seconds', durations.max())]:
        workloads.loc[mask, col] = value

    def aggregate(key):
        return jobs.groupby(key).agg(configurations=('cell', 'nunique'),
            jobs=('job_id', 'size'), samples=('samples', 'sum'),
            accelerator_hours=('stream_hours', 'sum'),
            elapsed_job_hours=('elapsed_job_hours', 'sum'),
            median_job_minutes=('elapsed_span_s', lambda s: s.median()/60),
            failed_jobs=('excluded_failed_job', 'sum'),
            nonexcluded_accelerator_hours=('nonexcluded_accelerator_hours', 'sum'),
            multi_host_jobs=('multi_host_observed', 'sum')).reset_index()

    workloads.to_csv(OUT / 'workloads.csv', index=False)
    jobs.to_csv(OUT / 'jobs.csv', index=False)
    aggregate('category').to_csv(OUT / 'categories.csv', index=False)
    aggregate('hardware').to_csv(OUT / 'hardware.csv', index=False)
    aggregate('family').to_csv(OUT / 'families.csv', index=False)
    multi = jobs[jobs.multi_host_observed]
    summary = dict(generated_utc=datetime.now(timezone.utc).isoformat(),
        checkout=subprocess.check_output(['git', '-C', str(ROOT), 'rev-parse', 'HEAD'], text=True).strip(),
        method='Hash-bound historical inventory plus independently read, disjoint recovery TPU job',
        configurations=int(workloads.cell.nunique()), jobs=len(jobs),
        accelerator_streams=int(workloads.streams.sum()), samples=int(workloads.samples.sum()),
        accelerator_hours=float(jobs.stream_hours.sum()),
        nonexcluded_accelerator_hours=float(jobs.nonexcluded_accelerator_hours.sum()),
        elapsed_job_hours=float(jobs.elapsed_job_hours.sum()),
        median_job_minutes=float(jobs.elapsed_span_s.median()/60),
        multi_host_jobs=len(multi), multi_host_accelerator_hours=float(multi.stream_hours.sum()),
        nonexcluded_multi_host_jobs=int((~multi.excluded_failed_job).sum()),
        nonexcluded_multi_host_accelerator_hours=float(multi.nonexcluded_accelerator_hours.sum()),
        recovery_manifest_sha256=sha(recovery_path), verified_recovery_files=len(recovery['files']),
        historical_provenance_sha256=sha(BASE / 'provenance.json'),
        recovery=recovery['summary'],
        caveats=[
            'Not a new full raw-corpus audit or fresh GCS enumeration.',
            'Historical identities and completion counts remain provisional and dated.',
            'Nonexcluded does not certify activity, success, full rank coverage, or evaluation eligibility.',
            'Accelerator-hours are per-device timestamp spans, not useful training time.',
            'Multi-host observation includes replicated workloads without cross-host computational dependence.',
            'Eight supplemental short Week 7 JSON captures remain separate.',
            'Cloud-only unadjudicated captures and quarantines remain outside main totals.'])
    columns = ['cell', 'description', 'gpu_name', 'jobs', 'streams', 'samples',
               'stream_hours', 'median_stream_minutes', 'min_devices', 'max_devices',
               'min_hosts', 'max_hosts', 'excluded_failed_jobs', 'category']
    table = workloads[columns].to_html(index=False, float_format=lambda x: f'{x:.3f}', escape=True)
    page = '''<!doctype html><html lang="en"><meta charset="utf-8"><title>Workload inventory after recovery</title>
<style>body{font:15px system-ui;margin:2em;color:#152b3c}input{padding:.7em;width:30em;max-width:90%}
table{border-collapse:collapse;margin-top:1em}th,td{padding:.5em;border:1px solid #ccd5dd;text-align:left}
th{background:#eaf1f6;position:sticky;top:0}tr:nth-child(even){background:#f7f9fb}</style>
<h1>Workload inventory after October 12 recovery</h1>
<p>ACCOUNTING_TOTALS accelerator-hours including known failures.
Hours are recorded device spans, not useful training time. Historical identities remain provisional.
Device/host counts describe observation, not proof of participation. Host count zero means unknown.</p>
<p>Based on the earlier hash-verified inventory plus one completed four-host TPU run.
225 recovered NIC streams add communication measurements, not accelerator-hours.</p>
<p><a href="workloads.csv">Workload CSV</a> · <a href="summary.json">Accounting and provenance</a> ·
<a href="../gcs_reconciliation/gcs-audit-addendum-2026-10-12.md">Cloud reconciliation</a> ·
<a href="../recovery/README.md">Recovery decisions</a></p>
<label>Filter workloads, descriptions or hardware: <input id="q" type="search"></label>'''
    page = page.replace('ACCOUNTING_TOTALS',
        f'{summary["configurations"]:,} configurations; {summary["jobs"]:,} reconstructed jobs; '
        f'{summary["accelerator_hours"]:,.3f}')
    page += table + '''<script>document.getElementById('q').addEventListener('input', e=>{
const q=e.target.value.toLowerCase();document.querySelectorAll('tbody tr').forEach(r=>{
r.hidden=!r.textContent.toLowerCase().includes(q)})});</script></html>'''
    (OUT / 'inventory.html').write_text(page)
    summary['output_sha256'] = {p.name: sha(p) for p in OUT.iterdir()
                                if p.suffix in ('.csv', '.html', '.py')}
    (OUT / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps({k: v for k, v in summary.items() if k not in ('recovery', 'output_sha256')}, indent=2))


if __name__ == '__main__':
    main()
