#!/usr/bin/env python3
"""Stage and launch one bounded pilot on an existing idle H100 pair.

Run in an authenticated Cloud Shell with network access. Default prints the
plan. --launch requires both instances already RUNNING, checks both for idle
GPUs/queues, stages a hashed source bundle without Git credentials or data, and
starts detached systemd services. Both dedicated nodes power off afterwards.
A 60 second smoke is required before a 30 minute collection of the same cell.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess
import tarfile
import tempfile

try:
    from scripts.gcp.run_followup_node import PILOT_CELLS, run_id, sha256
except ImportError:
    from run_followup_node import PILOT_CELLS, run_id, sha256

ROOT = Path(__file__).resolve().parents[2]


def cloud(args):
    return subprocess.run(['gcloud', *args], check=True, capture_output=True,
                          text=True, timeout=180).stdout


def source_files(root):
    paths = set()
    for folder in ('workloads', 'scripts', 'classifier'):
        for ext in ('*.py', '*.sh'):
            paths.update(p for p in (root/folder).rglob(ext)
                         if not any(x.startswith('.') or x in ('__pycache__', 'venv', 'venv-main', 'venv-vllm')
                                    for x in p.relative_to(root).parts))
    paths.update(root.glob('requirements*.txt'))
    return sorted(paths)


def bundle(root, output, stamp):
    paths = source_files(root)
    manifest = dict(created_utc=datetime.now(timezone.utc).isoformat(), run_stamp=stamp,
                    source='working tree; content hashes are authoritative, local HEAD may lag',
                    files={str(p.relative_to(root)): sha256(p) for p in paths})
    with tempfile.TemporaryDirectory() as tmp:
        meta = Path(tmp)/'deployment.json'
        meta.write_text(json.dumps(manifest, indent=2)+'\n')
        with tarfile.open(output, 'w:gz') as tar:
            for path in paths:
                if path.is_symlink():
                    raise ValueError(f'Source symlink is not allowed: {path}')
                tar.add(path, arcname=str(path.relative_to(root)), recursive=False)
            tar.add(meta, arcname='deployment.json')
    return manifest


def remote_command(a, node, args):
    return cloud(['compute', 'ssh', node, '--project', a.project, '--zone', a.zone,
                  '--quiet', '--command', shlex.join(args)])


PROBE = '''set -eu
[ ! -e /run/systemd/shutdown/scheduled ]
[ -x /opt/spar/repo/venv-main/bin/python ]
[ -x /opt/spar/repo/venv-vllm/bin/ray ]
[ -x /opt/spar/repo/venv-main/bin/torchrun ]
[ -z "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]
! pgrep -f '[m]ultinode_queue.sh|[r]un_followup_node.py' >/dev/null
! pgrep -f '[t]orchrun|[v]llm|[r]aylet' >/dev/null
nvidia-smi --query-gpu=name --format=csv,noheader
'''


def check_smoke(a):
    """Require two successful durable receipts; identity must agree across hosts."""
    receipts = []
    for node in (a.node0, a.node1):
        uri = f'{a.bucket}/runs/{a.session}/control/{a.smoke_stamp}/{node}.complete.json'
        receipts.append(json.loads(cloud(['storage', 'cat', uri])))
    for rank, receipt in enumerate(receipts):
        if not (receipt['status'] == 'local_complete' and receipt['smoke']
                and receipt['cell'] == a.cell and receipt['rank'] == rank
                and receipt['node'] == (a.node0, a.node1)[rank]
                and receipt['peer'] == (a.node1, a.node0)[rank]
                and receipt['run_stamp'] == a.smoke_stamp
                and receipt['local_result']['complete']
                and receipt['uploads'] and all(u['verified'] for u in receipt['uploads'])):
            raise RuntimeError('Smoke receipts do not establish successful collection on this pair')
    if receipts[0]['seed'] != receipts[1]['seed']:
        raise RuntimeError('Smoke nodes used different seeds')
    current = {str(p.relative_to(ROOT)): sha256(p) for p in source_files(ROOT)}
    if any(r['deployment']['files'] != current for r in receipts):
        raise RuntimeError('Source changed since smoke; smoke this exact source before collection')


def service_command(a, rank, deploy):
    node = (a.node0, a.node1)[rank]
    # Bound all descendants, including torchrun's separate process sessions.
    # ExecStopPost also runs when a peer never starts or the Python runner crashes.
    seconds = a.duration + 1800
    args = ['sudo', 'systemd-run', '--unit', f'spar-followup-{a.run_stamp}',
            '--property', f'RuntimeMaxSec={seconds}', '--property', 'TimeoutStopSec=300',
            '--property', 'KillMode=control-group', '--property', 'Restart=no',
            '--property', 'ExecStopPost=/usr/sbin/shutdown -h +2',
            '/usr/bin/python3', f'{deploy}/scripts/gcp/run_followup_node.py',
            '--repo', deploy, '--cell', a.cell, '--run-stamp', a.run_stamp,
            '--seed', str(a.seed), '--duration', str(a.duration), '--rank', str(rank),
            '--node0', a.node0, '--node1', a.node1, '--master', a.node0,
            '--bucket', a.bucket, '--session', a.session]
    if a.smoke:
        args.append('--smoke')
    return args


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--project', default='focus-surfer-494820-g0')
    ap.add_argument('--zone', required=True)
    ap.add_argument('--node0', required=True); ap.add_argument('--node1', required=True)
    ap.add_argument('--cell', choices=sorted(PILOT_CELLS), default='s16_train_tp8')
    ap.add_argument('--run-stamp', required=True)
    ap.add_argument('--session', default='followup_20261012')
    ap.add_argument('--bucket', default='gs://focus-surfer-494820-g0-telemetry')
    ap.add_argument('--seed', type=int, default=20261012)
    ap.add_argument('--duration', type=int)
    ap.add_argument('--smoke', action='store_true')
    ap.add_argument('--smoke-stamp', help='Prior successful smoke of this cell, source and pair')
    ap.add_argument('--launch', action='store_true')
    a = ap.parse_args()
    a.duration = a.duration if a.duration is not None else (60 if a.smoke else 1800)
    for value in (a.run_stamp, a.session, a.project, a.zone, a.node0, a.node1):
        if not re.fullmatch(r'[A-Za-z0-9_-]+', value):
            ap.error('Identifiers may contain only letters, digits, underscore and hyphen')
    if len(a.run_stamp) > 80 or a.node0 == a.node1 or not 0 <= a.seed < 2**31:
        ap.error('Require a run stamp <=80 characters, distinct nodes and a 31-bit seed')
    if not 30 <= a.duration <= 3600 or (a.smoke and a.duration > 120):
        ap.error('Duration must be 30..3600 seconds; smoke is at most 120 seconds')
    if not a.smoke and not a.smoke_stamp:
        ap.error('Collection requires --smoke-stamp from a successful smoke')
    deploy = f'/opt/spar/followup/{a.run_stamp}'
    plan = dict(status='planned_not_launched', run_stamp=a.run_stamp,
                run_id=run_id(a.run_stamp, a.cell), cell=a.cell, seed=a.seed,
                smoke=a.smoke, nodes=[a.node0, a.node1], duration_s=a.duration,
                hard_runtime_s=a.duration+1800, stop_grace_s=300, shutdown_delay_s=120,
                deployment=deploy, output=f'{a.bucket}/runs/{a.session}',
                commands=[shlex.join(service_command(a, r, deploy)) for r in (0, 1)])
    print(json.dumps(plan, indent=2), flush=True)
    if not a.launch:
        return
    # Do not start, recreate or repurpose any instances implicitly.
    for node in (a.node0, a.node1):
        info = json.loads(cloud(['compute', 'instances', 'describe', node,
                                '--project', a.project, '--zone', a.zone, '--format=json']))
        if info['status'] != 'RUNNING':
            raise RuntimeError(f'{node} is {info["status"]}; explicitly start the intended pair first')
    with ThreadPoolExecutor(max_workers=2) as pool:
        probes = list(pool.map(lambda n: remote_command(a, n, ['bash', '-c', PROBE]), (a.node0, a.node1)))
    if any(len(p.strip().splitlines()) != 8 or any('H100' not in g for g in p.strip().splitlines()) for p in probes):
        raise RuntimeError('This pilot requires exactly eight H100 GPUs on each idle host')
    if not a.smoke:
        check_smoke(a)
    with tempfile.TemporaryDirectory(prefix='spar-deploy-') as tmp:
        archive = Path(tmp)/'source.tar.gz'
        bundle(ROOT, archive, a.run_stamp)
        digest = sha256(archive)
        # Complete deployment on BOTH nodes before starting either service.
        for node in (a.node0, a.node1):
            remote_tar = f'/tmp/spar-{a.run_stamp}.tar.gz'
            cloud(['compute', 'scp', str(archive), f'{node}:{remote_tar}',
                   '--project', a.project, '--zone', a.zone, '--quiet'])
            script = f'''set -eu
printf '%s  %s\\n' {shlex.quote(digest)} {shlex.quote(remote_tar)} | sha256sum --check
sudo mkdir {shlex.quote(deploy)}
sudo tar -xzf {shlex.quote(remote_tar)} -C {shlex.quote(deploy)}
sudo /usr/bin/python3 -c {shlex.quote('import hashlib,json,pathlib,sys; r=pathlib.Path(sys.argv[1]); m=json.loads((r/"deployment.json").read_text()); assert all(hashlib.sha256((r/p).read_bytes()).hexdigest()==h for p,h in m["files"].items()), "Deployment content mismatch"')} {shlex.quote(deploy)}
sudo ln -s /opt/spar/repo/venv-main {shlex.quote(deploy)}/venv-main
sudo ln -s /opt/spar/repo/venv-vllm {shlex.quote(deploy)}/venv-vllm
rm {shlex.quote(remote_tar)}
'''
            remote_command(a, node, ['bash', '-c', 'sudo mkdir -p /opt/spar/followup\n'+script])
        launched = []
        try:
            for rank, node in enumerate((a.node0, a.node1)):
                remote_command(a, node, service_command(a, rank, deploy))
                launched.append(node)
        finally:
            plan.update(status='services_requested', services_requested=launched)
            # This reports requests, not successful collection. Each service is
            # independently time-limited even if the second SSH call fails.
            print(json.dumps(plan, indent=2), flush=True)


if __name__ == '__main__':
    main()
