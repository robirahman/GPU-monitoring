#!/usr/bin/env python3
"""Preflight or run the corrected evaluation on a prepared analysis machine.

Does not provision hardware, fetch data, alter the checkout or select thresholds.
Default is preflight only. --run builds a fresh provenance-bound pooled cache and
runs mechanism/backbone holdouts plus the paired October campaign ablation.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]


def commands(work, workers, trees):
    windows = str(work/'windows_30_15.parquet')
    base = [sys.executable, '-m', 'classifier.rebuttal_analysis.paired_family_lofo',
            '--windows', windows, '--trees', str(trees), '--seed', '0', '--threshold', '0.5']
    return [
        ('cache', [sys.executable, 'scripts/build_mn_window_cache.py', windows, '30', '15', str(workers)]),
        ('mechanism_paired', base + ['--exclude-campaigns', '20261001,20261003',
                                    '--out', str(work/'mechanism_paired.json')]),
        ('backbone', base + ['--family-level', 'backbone', '--out', str(work/'backbone.json')]),
        ('steps_only', base + ['--steps-only', '--out', str(work/'steps_only.json')]),
    ]


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--work', type=Path, required=True, help='External analysis storage directory')
    ap.add_argument('--objects', type=Path, help='External LFS objects directory')
    ap.add_argument('--workers', type=int, default=2)
    ap.add_argument('--trees', type=int, default=300)
    ap.add_argument('--run', action='store_true')
    a = ap.parse_args()
    if a.workers < 1 or a.trees < 1:
        ap.error('workers and trees must be positive')
    work = a.work.resolve(); work.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ, PYTHONPATH=str(ROOT), SPAR_WORK=str(work))
    if a.objects:
        env['SPAR_LFS_OBJECTS'] = str(a.objects.resolve())
    plan = commands(work, a.workers, a.trees)
    status = dict(started_utc=datetime.now(timezone.utc).isoformat(), requested_run=a.run,
                  status='preflight', threshold_policy='fixed 0.5, not a calibrated deployment operating point',
                  plan=[dict(name=n, command=c) for n,c in plan], completed=[])
    def save():
        (work/'execution.json').write_text(json.dumps(status,indent=2)+'\n')
    save()
    with (work/'preflight.log').open('w') as log:
        check = subprocess.run([sys.executable, 'scripts/check_analysis_inputs.py',
            '--out', str(work/'inputs')], cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
    if check.returncode:
        status.update(status='blocked_inputs', returncode=check.returncode); save()
        print(f'Analysis blocked by input preflight; see {work / "inputs/summary.json"}')
        return check.returncode
    if not a.run:
        status['status']='ready_not_run'; save()
        print(f'Inputs ready; use --run to execute the plan in {work / "execution.json"}')
        return 0
    for name, cmd in plan:
        status.update(status='running', phase=name); save()
        with (work/f'{name}.log').open('w') as log:
            result = subprocess.run(cmd, cwd=ROOT, env=env, stdout=log, stderr=subprocess.STDOUT)
        if result.returncode:
            status.update(status='failed', returncode=result.returncode); save()
            return result.returncode
        status['completed'].append(name); save()
    status.update(status='complete', finished_utc=datetime.now(timezone.utc).isoformat()); save()
    return 0


if __name__ == '__main__':
    sys.exit(main())
