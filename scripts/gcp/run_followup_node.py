#!/usr/bin/env python3
"""Run one bounded two-node pilot cell ON an idle, prepared GPU node.

The caller must launch both ranks under systemd with a shared deadline and
KillMode=control-group. Periodic uploads preserve partial results; final uploads
are downloaded and SHA256 checked. Completion is provisional until both node
receipts and progress/telemetry have been adjudicated.

These checks establish completed-run status, not blanket corpus eligibility.
Interrupted runs retain their raw artifacts; separately validated observed
intervals may enter the corpus without a success receipt or repetition credit.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time

# Small pilot before the full randomized repetition campaign. These synthetic
# probes use random tokens: success is not a claim of useful task learning.
PILOT_CELLS = {'s16_train_localsgd_h1', 's16_train_tp8',
               's16_train_moe_bal', 's16_train_impala_a15'}


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as f:
        for chunk in iter(lambda: f.read(1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def run_id(stamp, cell):
    return hashlib.sha1(f'{stamp}_{cell}_1'.encode()).hexdigest()[:12]


def checked(cmd, **kw):
    return subprocess.run(cmd, check=True, timeout=120, **kw)


def upload(path, uri, verify=False):
    # Use a snapshot: active logs and .partial checkpoints can change mid-copy.
    with tempfile.TemporaryDirectory(prefix='spar-upload-') as tmp:
        snap = Path(tmp)/'object'
        snap.write_bytes(path.read_bytes())
        digest = sha256(snap)
        checked(['gcloud', 'storage', 'cp', str(snap), uri], stdout=subprocess.DEVNULL)
        if verify:
            remote = Path(tmp)/'download'
            checked(['gcloud', 'storage', 'cp', uri, str(remote)], stdout=subprocess.DEVNULL)
            if sha256(remote) != digest:
                raise RuntimeError(f'Cloud hash mismatch for {uri}')
        return dict(uri=uri, bytes=snap.stat().st_size, sha256=digest, verified=verify)


def artifacts(repo, rid, stamp):
    found = set()
    for directory in (repo/'data', repo/'logs'):
        if directory.exists():
            found.update(p for p in directory.rglob(f'*{rid}*') if p.is_file())
    found.update(p for p in (repo/'logs'/'.done'/stamp).glob('*') if p.is_file())
    return sorted(found)


def check_local_result(repo, rid, stamp, cell, rank, smoke, seed):
    prefix = 'smoke_' if smoke else ''
    marker = repo/'logs'/'.done'/stamp/f'{prefix}{cell}_node{rank}_{rid}.done'
    data = repo/('logs/smoke_data' if smoke else 'data')
    streams = sorted(data.glob(f'{prefix}{cell}_node{rank}_gpu*_{rid}.parquet'))
    nic = data/f'nic_{cell}_node{rank}_{rid}.parquet'
    log = repo/'logs'/f'{prefix}{cell}_node{rank}_{rid}.log'
    content = log.read_text(errors='replace') if log.exists() else ''
    seeds = []
    # torchrun may interleave stdout; malformed records leave the run incomplete.
    for record in re.findall(r'RUN_SEED (\{[^\n]*?\})', content):
        try:
            seeds.append(json.loads(record))
        except ValueError:
            pass
    seeded_ranks = {s['rank'] for s in seeds if s.get('base_seed') == seed
                    and s.get('run_stamp') == stamp}
    seed_ok = seeded_ranks == set(range(rank*8, (rank+1)*8))
    progress = {}
    for record in re.findall(r'RUN_PROGRESS (\{[^\n]*?\})', content):
        try:
            entry = json.loads(record)
            if entry.get('base_seed') == seed and entry.get('run_stamp') == stamp:
                progress[entry['rank']] = entry
        except (ValueError, KeyError):
            pass
    progress_ok = set(progress) == set(range(rank*8, (rank+1)*8))
    for worker, entry in progress.items():
        actor = cell == 's16_train_impala_a15' and worker != 0
        progress_ok &= entry.get('infer_steps' if actor else 'optimizer_updates', 0) > 0
    marker_ok = marker.exists() and bool(re.search(r'\brc=0\b', marker.read_text()))
    gpu_indices = {int(re.search(r'_gpu(\d+)_', p.name)[1]) for p in streams}
    return dict(marker_ok=marker_ok, gpu_streams=len(streams), nic_present=nic.exists(),
                seed_records_ok=seed_ok, seeded_ranks=sorted(seeded_ranks),
                progress_records_ok=progress_ok, progress=progress,
                complete=marker_ok and len(streams) == 8 and gpu_indices == set(range(8)) and nic.exists() and seed_ok and progress_ok,
                scientific_admission='pending_peer_and_progress_review')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--repo', type=Path, required=True)
    ap.add_argument('--cell', choices=sorted(PILOT_CELLS), required=True)
    ap.add_argument('--run-stamp', required=True)
    ap.add_argument('--seed', type=int, required=True)
    ap.add_argument('--duration', type=int, default=1800)
    ap.add_argument('--rank', type=int, choices=(0, 1), required=True)
    ap.add_argument('--node0', required=True); ap.add_argument('--node1', required=True)
    ap.add_argument('--master', required=True)
    ap.add_argument('--bucket', required=True); ap.add_argument('--session', required=True)
    ap.add_argument('--smoke', action='store_true')
    a = ap.parse_args()
    for value in (a.run_stamp, a.node0, a.node1, a.session):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            ap.error('Identifiers may contain only letters, digits, underscore and hyphen')
    if not 0 <= a.seed < 2**31 or not 30 <= a.duration <= 3600 or a.node0 == a.node1:
        ap.error('Require a 31-bit seed, 30..3600 second duration and two different nodes')
    if not re.fullmatch(r'gs://[a-z0-9][a-z0-9._-]+', a.bucket):
        ap.error('--bucket must be a bucket URI without an object prefix')
    repo = a.repo.resolve()
    node = (a.node0, a.node1)[a.rank]; peer = (a.node1, a.node0)[a.rank]
    if socket.gethostname().split('.')[0] != node:
        raise RuntimeError('Node identity mismatch')
    # A second check closes the gap since the driver's pair preflight. Queue
    # also refuses to reap any compute process that appears after this check.
    busy = checked(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                   capture_output=True, text=True).stdout.strip()
    if busy:
        raise RuntimeError('GPUs are occupied; no workload launched')
    gpus = checked(['nvidia-smi', '--query-gpu=index,uuid,name,memory.total,driver_version',
                    '--format=csv,noheader'], capture_output=True, text=True).stdout
    if len(gpus.strip().splitlines()) != 8:
        raise RuntimeError('Pilot requires exactly 8 GPUs per host')
    rid = run_id(a.run_stamp, a.cell)
    (repo/'logs').mkdir(exist_ok=True)
    state_path = repo/'logs'/f'followup_{rid}.json'
    if state_path.exists():
        raise RuntimeError('Run already exists: choose a fresh run stamp, never overwrite a repetition')
    queue_log = repo/'logs'/f'followup_{rid}.queue.log'
    dest = f'{a.bucket}/runs/{a.session}/{node}'
    state = dict(status='starting', started_utc=datetime.now(timezone.utc).isoformat(),
                 run_id=rid, run_stamp=a.run_stamp, cell=a.cell, rank=a.rank, node=node,
                 peer=peer, seed=a.seed, duration_s=a.duration, smoke=a.smoke,
                 hardware=gpus.strip().splitlines(), uploads=[],
                 deployment=json.loads((repo/'deployment.json').read_text()),
                 scientific_admission='pending_peer_and_progress_review')
    def save():
        state_path.write_text(json.dumps(state, indent=2)+'\n')
    save()
    # Storage must work before spending compute time; read-after-write also
    # establishes read permission required by all barriers and receipts.
    upload(state_path, f'{dest}/logs/{state_path.name}', verify=True)
    env = dict(os.environ, REPO=str(repo), REPO_DIR=str(repo), PYTHONPATH=str(repo),
               GCPDIR=str(repo/'scripts/gcp'), NODE_RANK=str(a.rank), NODE_NAME=node,
               NODE0=a.node0, NODE1=a.node1, PEER=peer, MASTER_ADDR=a.master, MASTER_PORT='29500',
               NNODES='2', RUN_STAMP=a.run_stamp, SPAR_WORKLOAD_SEED=str(a.seed),
               BUCKET=a.bucket, SESSION=a.session, ONLY_CELLS='^'+a.cell+'$',
               PY=str(repo/'venv-main/bin/python'), PY_VLLM=str(repo/'venv-vllm/bin/python'),
               REPS='1', DUR=str(a.duration), GAP='0', SMOKE=str(int(a.smoke)),
               TIMEOUT=str(a.duration+600), TIMEOUT_BIG=str(a.duration+600),
               SIDECAR_DURATION=str(a.duration+1200),
               BARRIER_WAIT='300', RENDEZVOUS_TIMEOUT='300',
               SPAR_REFUSE_BUSY_GPU='1', SPAR_STRICT_BARRIER='1')
    def interrupted(signum, frame):
        raise RuntimeError(f'Service interrupted by signal {signum}')
    signal.signal(signal.SIGTERM, interrupted)
    process = None
    try:
        with queue_log.open('w') as log:
            process = subprocess.Popen(['bash', 'scripts/rebuttal_collection/multinode_queue.sh'],
                                       cwd=repo, env=env, stdout=log, stderr=subprocess.STDOUT)
            state['status'] = 'collecting'; save()
            # systemd's RuntimeMaxSec bounds this loop and every descendant.
            while process.poll() is None:
                try:
                    process.wait(timeout=120)
                except subprocess.TimeoutExpired:
                    for path in artifacts(repo, rid, a.run_stamp):
                        try:
                            upload(path, f'{dest}/{path.relative_to(repo)}')
                        except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                            state['last_periodic_upload_error'] = str(exc)
                    save()
        state['queue_returncode'] = process.returncode
        result = check_local_result(repo, rid, a.run_stamp, a.cell, a.rank, a.smoke, a.seed)
        state['local_result'] = result
        state['status'] = 'local_complete' if process.returncode == 0 and result['complete'] else 'failed_collection'
    except BaseException as exc:
        state.update(status='interrupted', error=str(exc))
        if process is not None and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=20)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)
    finally:
        save()
        receipts = []
        for path in artifacts(repo, rid, a.run_stamp):
            if path == state_path:
                continue
            try:
                receipts.append(upload(path, f'{dest}/{path.relative_to(repo)}', verify=True))
            except (subprocess.SubprocessError, OSError, RuntimeError) as exc:
                state.setdefault('final_upload_errors', []).append(str(exc))
        state['uploads'] = receipts
        state['finished_utc'] = datetime.now(timezone.utc).isoformat()
        if state.get('final_upload_errors'):
            state['status'] = 'failed_durability'
        save()
        # A receipt is not published until final data/log uploads have succeeded.
        upload(state_path, f'{dest}/logs/{state_path.name}', verify=True)
        if state['status'] == 'local_complete':
            upload(state_path, f'{a.bucket}/runs/{a.session}/control/{a.run_stamp}/{node}.complete.json', verify=True)
            # Keep this host up while the peer finishes saving its own outputs.
            # A failed peer cannot leave this node billing indefinitely.
            deadline = time.monotonic() + 300
            peer_uri = f'{a.bucket}/runs/{a.session}/control/{a.run_stamp}/{peer}.complete.json'
            while time.monotonic() < deadline:
                try:
                    text = checked(['gcloud', 'storage', 'cat', peer_uri], capture_output=True, text=True).stdout
                    receipt = json.loads(text)
                    if (receipt['run_id'] == rid and receipt['seed'] == a.seed
                            and receipt['status'] == 'local_complete'):
                        state['peer_receipt_verified'] = True
                        break
                except (subprocess.SubprocessError, ValueError, KeyError):
                    pass
                time.sleep(10)
            state.setdefault('peer_receipt_verified', False)
            save()
            upload(state_path, f'{dest}/logs/{state_path.name}', verify=True)
    return 0 if state['status'] == 'local_complete' else 1


if __name__ == '__main__':
    raise SystemExit(main())
