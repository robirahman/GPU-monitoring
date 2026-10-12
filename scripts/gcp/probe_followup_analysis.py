#!/usr/bin/env python3
"""Read-only analysis-host readiness; never changes the repo or starts/stops jobs.

Sent as Python source over SSH. Reports only selected project paths and package
versions; no credential files, environment variables or remote Git URLs.
"""
import json
import os
from pathlib import Path
import shutil
import subprocess


def inspect():
    repo=Path('/opt/spar/repo')
    result=dict(host=os.uname().nodename,repo_exists=repo.is_dir(),
                work_free_bytes=shutil.disk_usage('/var/tmp').free,
                meminfo={line.split(':')[0]:line.split(':')[1].strip()
                         for line in Path('/proc/meminfo').read_text().splitlines()
                         if line.startswith(('MemTotal:','MemAvailable:'))})
    names={'run_lofo_analysis.sh','run_followup_evaluation.py','build_mn_window_cache.py',
           'paired_family_lofo.py','classifier.rebuttal_analysis.paired_family_lofo',
           'mn_binary_lofo.py','mz_watchdog.sh','campaign.sh'}
    processes=[]
    for entry in Path('/proc').glob('[0-9]*'):
        try:
            args=(entry/'cmdline').read_bytes().decode(errors='replace').split('\0')
        except (FileNotFoundError,PermissionError):
            continue
        matched=[Path(a).name for a in args if '\n' not in a and Path(a).name in names]
        if matched:
            processes.append(dict(pid=int(entry.name),programs=matched))
    result['project_processes']=processes
    if repo.is_dir():
        head=subprocess.run(['git','-C',str(repo),'rev-parse','HEAD'],capture_output=True,text=True,timeout=20)
        result['head']=head.stdout.strip() if head.returncode==0 else None
        files=list((repo/'data').rglob('*.parquet'))
        result['parquet_files']=len(files)
        result['data_file_bytes']=sum(p.stat().st_size for p in files)
        result['small_parquet_files']=sum(p.stat().st_size<256 for p in files)
        objects=repo/'.git/lfs/objects'
        result['local_lfs_directory_exists']=objects.is_dir()
    candidates=[repo/'venv-analysis/bin/python',Path('/usr/bin/python3')]
    result['interpreters']=[]
    code='import json,pandas,numpy,sklearn,pyarrow; print(json.dumps({m.__name__:m.__version__ for m in (pandas,numpy,sklearn,pyarrow)}))'
    for interpreter in candidates:
        if interpreter.exists():
            check=subprocess.run([str(interpreter),'-c',code],capture_output=True,text=True,timeout=45)
            result['interpreters'].append(dict(path=str(interpreter),ready=check.returncode==0,
                versions=json.loads(check.stdout) if check.returncode==0 else None))
    work=Path('/var/tmp/spar_analysis')
    result['existing_analysis_artifacts']=[dict(name=p.name,bytes=p.stat().st_size,mtime=p.stat().st_mtime)
        for p in sorted(work.glob('*')) if p.is_file() and (p.suffix in ('.json','.parquet') or p.name=='STATUS')]
    return result


if __name__=='__main__':
    print(json.dumps(inspect(),indent=2))
