#!/usr/bin/env python3
"""Produce a bounded, deterministic repetition plan; does not launch VMs/jobs.

Day slots require distinct actual collection dates. Independent seeds are a
separate pending requirement: the legacy queue does not uniformly expose them.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path
import random
import re
import shlex

FAMILIES = {'train_localsgd', 'train_streamdiloco', 'train_decoupdiloco',
            'train_swarm', 'train_moe', 'train_tp', 'train_impala', 'train_apexrl',
            'train_ppodist'}
ROOT = Path(__file__).resolve().parents[1]


def make_plan(jobs, launches=5, days=3, duration=1800, seed=20261012):
    if launches < days or days < 1 or duration < 30:
        raise ValueError('Require launches >= days >= 1 and duration >= 30 seconds')
    selected = [j for j in jobs if j['family'] in FAMILIES
                and j['hardware'] == 'NVIDIA H100 80GB HBM3']
    cells = sorted({j['cell'] for j in selected})
    if not cells:
        raise ValueError('No target H100 configurations in inventory')
    rows = []
    rng = random.Random(seed)
    for launch in range(launches):
        order = cells.copy(); rng.shuffle(order)
        day = launch % days + 1
        for idx, cell in enumerate(order):
            if not re.fullmatch(r's16_[A-Za-z0-9_]+', cell):
                raise ValueError(f'Invalid distributed cell: {cell}')
            history = [j for j in selected if j['cell'] == cell]
            stamp = f'followup_d{day}_launch{launch+1}_{cell}'
            env = dict(RUN_STAMP=stamp, REPS='1', DUR=str(duration),
                       TIMEOUT=str(duration+1200), TIMEOUT_BIG=str(duration+1500),
                       TIMEOUT_SERVE=str(duration+2400),
                       ONLY_CELLS='^' + cell + '$', SMOKE='0')
            command = 'env ' + ' '.join(f'{k}={shlex.quote(v)}' for k,v in env.items())
            command += ' bash scripts/rebuttal_collection/multinode_queue.sh'
            rows.append(dict(cell=cell, family=history[0]['family'], day_slot=day,
                launch=launch+1, order_in_launch=idx+1, duration_s=duration,
                nodes=2, accelerators_per_node=8, hardware='NVIDIA H100 80GB HBM3',
                recorded_prior_jobs=len(history),
                prior_nonexcluded_jobs=sum(str(j['excluded_failed_job']).lower() != 'true' for j in history),
                run_stamp=stamp, queue_command=command,
                status='planned_not_launched', seed_control='pending_uniform_workload_seed_support'))
    return rows


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--inventory', type=Path,
        default=ROOT/'docs/workload_review_2026-10-12/current/jobs.csv')
    ap.add_argument('--out', type=Path, required=True)
    ap.add_argument('--launches', type=int, default=5)
    ap.add_argument('--days', type=int, default=3)
    ap.add_argument('--duration', type=int, default=1800)
    ap.add_argument('--seed', type=int, default=20261012)
    a = ap.parse_args()
    with a.inventory.open() as f:
        plan = make_plan(list(csv.DictReader(f)), a.launches, a.days, a.duration, a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    with (a.out/'repetitions.csv').open('w', newline='') as f:
        writer=csv.DictWriter(f, fieldnames=list(plan[0])); writer.writeheader(); writer.writerows(plan)
    summary = dict(configurations=len({r['cell'] for r in plan}), launches=len(plan),
        day_slots=a.days, seconds_per_launch=a.duration,
        nominal_pair_hours=len(plan)*a.duration/3600,
        nominal_accelerator_hours=len(plan)*a.duration*16/3600,
        schedule_seed=a.seed, inventory_sha256=hashlib.sha256(a.inventory.read_bytes()).hexdigest(),
        status='planned_not_launched',
        constraints=['Both nodes must use the identical plan and run stamps.',
            'Smoke each implementation; verify idle dedicated nodes before the legacy queue, which reaps compute processes.',
            'Run each day slot on a distinct collection date; validate all-worker completion/progress before counting a repetition.',
            'Uniform workload seed support is pending; schedule randomization is not a model/data seed.',
            'Hours exclude load, compile, sync, teardown and failed attempts. No dollar estimate without current pricing.',
            'Do not launch the whole plan automatically; select bounded batches and use durable synchronization plus peer-gated shutdown.'])
    (a.out/'summary.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__ == '__main__':
    main()
