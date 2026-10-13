#!/usr/bin/env python3
"""Admit verified TP8 intervals from interrupted follow-up captures.

Offline transformation of generation-pinned GCS downloads. Raw objects stay in
GCS, including startup, shutdown and unobserved tails. Completion is independent
of interval eligibility: these runs never earn a completed-repetition credit.
Only synchronous s16_train_tp8 is supported: rank-0 STEP events follow the
TP/DP gradient collectives and optimizer step in its recorded implementation.
"""
import argparse
import base64
import hashlib
import json
from pathlib import Path
import re
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from classifier.data_sources import sha256
from scripts.recover_audited_telemetry import clean_nic, merge_stream, nic_usable, same_or_absent, time_stats

CELL = 's16_train_tp8'
NODES = ('spar-a100-0', 'spar-a100-1')
DEST = Path('data/preempted_recovery_20261013')


def training_bounds(text, rid, stamp, seed):
    if f'Run ID: {rid}' not in text or 'tp_size=8 dp=2 world_size=16' not in text:
        raise ValueError('Missing exact run identity or synchronous TP8 topology')
    seeds = [json.loads(s) for s in re.findall(r'RUN_SEED (\{[^\n]*?\})', text)]
    if not any(s.get('rank') == 0 and s.get('base_seed') == seed and s.get('run_stamp') == stamp for s in seeds):
        raise ValueError('Missing matching rank-0 seed evidence')
    steps = [(float(t), int(b)) for t, b in re.findall(
        r'STEP\s+ts=([\d.]+)\s+kind=train\s+batch_idx=(\d+)[^\n]*\bnode=0\s+rank=0\b', text)]
    if len(steps) < 2 or any(t2 <= t1 or b2 != b1 + 1 for (t1, b1), (t2, b2) in zip(steps, steps[1:])):
        raise ValueError('Missing, nonmonotonic or discontinuous training STEP evidence')
    return dict(first_step=steps[0][0], last_step=steps[-1][0], logged_steps=len(steps),
                scope='synchronous TP8/DP2 job; rank 0 reports after gradient collectives and optimizer step')


def admitted_interval(gpus, evidence, cutoff):
    """Keep the common observed, STEP-backed interval across all 16 streams."""
    if set(gpus) != {(n, g) for n in NODES for g in range(8)}:
        raise ValueError('Expected all 16 physical GPU streams')
    lo = max(evidence['first_step'], *(float(f.timestamp_epoch.min()) for f in gpus.values()))
    hi = min(evidence['last_step'], cutoff, *(float(f.timestamp_epoch.max()) for f in gpus.values()))
    if not np.isfinite([lo, hi]).all() or hi - lo < 30:
        raise ValueError('Less than 30 seconds of common observed training')
    out = {}
    for key, frame in gpus.items():
        t = pd.to_numeric(frame.timestamp_epoch, errors='coerce')
        if not np.isfinite(t).all() or not t.is_monotonic_increasing or t.duplicated().any():
            raise ValueError('Invalid stream timestamps')
        kept = frame.loc[t.between(lo, hi)].copy().reset_index(drop=True)
        if len(kept) < 30 or kept.timestamp_epoch.diff().max() > 2.5:
            raise ValueError('Insufficient or gapped stream coverage')
        kept['expected_ranks'] = 16
        kept['meta_run_complete'] = False
        kept['meta_collection_status'] = 'preempted'
        kept['meta_observation_censored'] = True
        out[key] = kept
    return out, (lo, hi)


def recover(sources, evidence, out):
    rid, campaign = evidence['run_id'], evidence['campaign']
    if evidence['status'] != 'failed_spot_preemption' or evidence['workload'] != CELL:
        raise ValueError('This recovery requires reviewed TP8 Spot-preemption evidence')
    stamp = campaign + '_collect'
    events = [x for x in evidence['cloud_events'] if x['protoPayload'].get('methodName') == 'compute.instances.preempted']
    if not events or any(x['protoPayload']['resourceName'].rsplit('/', 1)[-1] not in NODES for x in events):
        raise ValueError('Missing matching cloud preemption event')
    cutoff = min(pd.Timestamp(x['timestamp']).timestamp() for x in events)
    selected = [x for x in sources if rid in Path(x['name']).name and f'/runs/{campaign}/' in '/' + x['name']]
    verified = []
    for obj in selected:
        path = Path(obj['path']); data = path.read_bytes()
        if (sha256(path) != obj['sha256'] or len(data) != obj['bytes']
                or base64.b64encode(hashlib.md5(data).digest()).decode() != obj['md5']
                or not obj['uri'].endswith('#' + obj['generation'])):
            raise ValueError('Unverified source object: ' + obj['name'])
        verified.append({k: v for k, v in obj.items() if k != 'path'})
    def exact(suffix):
        found = [x for x in selected if x['name'].endswith(suffix)]
        if len(found) != 1:
            raise ValueError('Missing/ambiguous source: ' + suffix)
        return found[0]
    log0 = exact(f'/spar-a100-0/logs/{CELL}_node0_{rid}.log')
    states = [json.loads(Path(exact(f'/{n}/logs/followup_{rid}.json')['path']).read_text()) for n in NODES]
    for rank, state in enumerate(states):
        if (state['run_id'] != rid or state['run_stamp'] != stamp or state['node'] != NODES[rank]
                or state['rank'] != rank or state['cell'] != CELL or state['smoke']):
            raise ValueError('Wrong state identity or smoke capture')
    seed = states[0]['seed']
    if states[1]['seed'] != seed:
        raise ValueError('Conflicting base seeds')
    ev = training_bounds(Path(log0['path']).read_text(), rid, stamp, seed)
    groups, nic_groups = {}, {}
    for obj in selected:
        name = obj['name']; p = Path(name)
        if '/data/' not in name or not re.search(r'\.parquet(?:\.partial)?$', name):
            continue
        node = p.parts[-3]
        if node not in NODES:
            raise ValueError('Unknown host')
        match = re.search(r'_gpu(\d+)_', p.name)
        if match:
            groups.setdefault((node, int(match[1])), []).append(obj)
        elif p.name.startswith('nic_'):
            nic_groups.setdefault(node, []).append(obj)
    gpus = {}
    source_refs = {}
    for key, objects in groups.items():
        node, gpu = key
        frame = merge_stream([pd.read_parquet(x['path']) for x in objects], rid, node, CELL)
        hardware = {int(parts[0]): parts[1] for line in states[NODES.index(node)]['hardware']
                    if (parts := [s.strip() for s in line.split(',')])}
        if gpu not in hardware or set(frame.gpu_uuid) != {hardware[gpu]}:
            raise ValueError('GPU identity differs from node hardware receipt')
        gpus[key] = frame
        source_refs[key] = sorted(x['sha256'] for x in objects)
    derived, interval = admitted_interval(gpus, ev, cutoff)
    outputs, streams = [], []
    def write(relative, payload):
        path = out / relative; digest = hashlib.sha256(payload).hexdigest()
        same_or_absent(path, digest); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(payload)
        outputs.append(dict(path=str(relative), sha256=digest, bytes=len(payload)))
    for (node, gpu), frame in sorted(derived.items()):
        # These two launches reuse one planned seed, so they are separate jobs
        # but not independent seeded repetitions for splitting/uncertainty.
        frame['meta_repetition_group'] = f'{CELL}|A100|seed={seed}'
        original = Path(groups[(node, gpu)][0]['name']).name.removesuffix('.partial')
        relative = Path('data/gcp') / campaign / node / original
        write(relative, frame.to_parquet(index=False))
        streams.append(dict(path=str(relative), role='accelerator', hostname=node, gpu_index=gpu,
                            source_hashes=source_refs[(node, gpu)], raw=time_stats(gpus[(node, gpu)]),
                            admitted=time_stats(frame)))
    for node in NODES:
        objects = nic_groups.get(node, [])
        if not objects:
            raise ValueError('Missing NIC sidecar')
        raw = merge_stream([pd.read_parquet(x['path']) for x in objects], rid, node, CELL)
        frame, counts = clean_nic(raw, [interval])
        if not nic_usable(frame):
            raise ValueError('Insufficient NIC overlap')
        relative = Path('data/gcp') / campaign / node / f'nic_{CELL}_node{NODES.index(node)}_{rid}.parquet'
        write(relative, frame.to_parquet(index=False))
        streams.append(dict(path=str(relative), role='nic', hostname=node,
                            source_hashes=sorted(x['sha256'] for x in objects), raw=time_stats(raw),
                            admitted=time_stats(frame), cleanup=counts))
    for rank, node in enumerate(NODES):
        log = exact(f'/{node}/logs/{CELL}_node{rank}_{rid}.log')
        # Standard log naming preserves physical host, node and the original run id.
        write(Path('logs') / f'{campaign}__{node}__{CELL}_node{rank}_{rid}.log', Path(log['path']).read_bytes())
    return dict(run_id=rid, campaign=campaign, workload=CELL, base_seed=seed,
                repetition_group=f'{CELL}|A100|seed={seed}',
                collection_status='preempted', run_completed=False, completed_repetition_credit=0,
                admitted_to_corpus=True, corpus_disposition='accepted_observed_training_interval',
                synthetic_workload=True, useful_learning_validated=False,
                first_step=ev['first_step'], last_step=ev['last_step'], step_evidence=ev,
                common_interval_s=list(interval), preemption_epoch=cutoff,
                streams=streams, sources=verified, outputs=outputs)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--sources', type=Path, required=True, help='Downloaded objects with generation, MD5 and SHA256')
    ap.add_argument('--evidence', type=Path, nargs='+', required=True)
    ap.add_argument('--out', type=Path, required=True, help='Isolated staging root; never the active analysis checkout')
    a = ap.parse_args()
    sources = json.loads(a.sources.read_text())
    runs = [recover(sources, json.loads(p.read_text()), a.out) for p in a.evidence]
    summary = dict(runs=len(runs), accelerator_streams=sum(s['role'] == 'accelerator' for r in runs for s in r['streams']),
                   nic_streams=sum(s['role'] == 'nic' for r in runs for s in r['streams']),
                   accelerator_samples=sum(s['admitted']['samples'] for r in runs for s in r['streams'] if s['role'] == 'accelerator'),
                   accelerator_hours=sum(s['admitted']['span_s'] for r in runs for s in r['streams'] if s['role'] == 'accelerator') / 3600,
                   completed_repetitions=0)
    manifest = dict(schema='preempted-followup-recovery-v1',
                    policy='Preserve raw capture; admit validated observed intervals independently of job exit status. Keep original job IDs and group all windows from each job together. No synthetic padding or completed-duration credit.',
                    source_policy='Original bytes remain at generation-pinned GCS URIs. Derived files and STEP logs are separate corpus inputs.',
                    implementation_sha256=sha256(Path(__file__)),
                    review_evidence=[dict(path=str(p), sha256=sha256(p)) for p in a.evidence],
                    runs=runs, summary=summary)
    path = a.out / DEST / 'manifest.json'; path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
