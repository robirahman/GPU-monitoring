#!/usr/bin/env python3
"""Read-only GPU pilot preflight, also sent over SSH as Python source."""
import json
import os
from pathlib import Path
import subprocess

BUSY_COMMANDS = {'multinode_queue.sh', 'run_followup_node.py', 'torchrun', 'vllm', 'raylet'}


def busy_command(argv):
    # Match actual executable/script arguments, never source embedded in a
    # parent SSH or `python -c` command. Substring pgrep would match itself.
    return any(Path(arg).name in BUSY_COMMANDS for arg in argv if '\n' not in arg)


def probe():
    for path in ('/opt/spar/repo/venv-main/bin/python',
                 '/opt/spar/repo/venv-main/bin/torchrun',
                 '/opt/spar/repo/venv-vllm/bin/ray'):
        if not os.access(path, os.X_OK):
            raise RuntimeError(f'Missing required executable: {path}')
    if Path('/run/systemd/shutdown/scheduled').exists():
        raise RuntimeError('A shutdown is already scheduled')
    active = []
    for process in Path('/proc').glob('[0-9]*'):
        try:
            argv = (process/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except FileNotFoundError:
            continue
        if busy_command(argv):
            active.append(int(process.name))
    if active:
        raise RuntimeError(f'Existing collection/training/serving processes: {active}')
    apps = subprocess.run(['nvidia-smi', '--query-compute-apps=pid', '--format=csv,noheader'],
                          check=True, capture_output=True, text=True, timeout=20).stdout.strip()
    if apps:
        raise RuntimeError('GPUs have active compute processes')
    output = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total',
                             '--format=csv,noheader,nounits'], check=True,
                            capture_output=True, text=True, timeout=20).stdout
    gpus = []
    for line in output.strip().splitlines():
        name, memory = line.rsplit(',', 1)
        gpus.append(dict(name=name.strip(), memory_mib=int(memory.strip())))
    return dict(gpus=gpus)


if __name__ == '__main__':
    print(json.dumps(probe()))
