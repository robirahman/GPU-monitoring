#!/usr/bin/env python3
"""Cloud Shell driver: smoke then one 30-minute TP8 collection on the existing A100 pair.

Print the plan by default; --launch starts the stopped pair. GPU jobs are
independently bounded by systemd and shut down their hosts. This driver records
receipts and downloads hash-verified results to /tmp, then checks basic telemetry
and per-rank progress before proceeding from smoke to collection.
"""
import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import signal
import subprocess
import sys
import time

try:
    from scripts.gcp import launch_followup_pilot as driver
except ImportError:
    import launch_followup_pilot as driver

PROJECT='focus-surfer-494820-g0'
ZONE='us-central1-c'
NODES=('spar-a100-0','spar-a100-1')
BUCKET='gs://focus-surfer-494820-g0-telemetry'
CELL='s16_train_tp8'


def read_receipt(uri):
    result=subprocess.run(['gcloud','storage','cat',uri],capture_output=True,text=True,timeout=60)
    if result.returncode:
        return None
    return json.loads(result.stdout)


def wait_stopped(deadline):
    while time.monotonic() < deadline:
        states=instance_states()
        if states == ['TERMINATED','TERMINATED']:
            return
        time.sleep(20)
    raise RuntimeError('Pair did not reach TERMINATED before the deadline')


def instance_states():
    return [json.loads(driver.cloud(['compute','instances','describe',n,
            '--project',PROJECT,'--zone',ZONE,'--format=json']))['status'] for n in NODES]


def validate_downloads(receipts, work, duration):
    import pandas as pd
    import numpy as np
    rows=[]
    for rank, receipt in enumerate(receipts):
        if receipt['rank'] != rank or receipt['node'] != NODES[rank]:
            raise RuntimeError('Receipt node identity mismatch')
        progress=receipt['local_result']['progress']
        if (not receipt['local_result']['complete']
                or {int(r) for r in progress} != set(range(rank*8,(rank+1)*8))
                or not all(u['verified'] for u in receipt['uploads'])):
            raise RuntimeError('Receipt is incomplete or missing verified worker evidence')
        for entry in progress.values():
            if entry['optimizer_updates'] < 1 or entry['elapsed_s'] < duration * .9:
                raise RuntimeError('Insufficient per-rank completed work')
        count=0
        for uploaded in receipt['uploads']:
            uri=uploaded['uri']
            prefix=f'{BUCKET}/runs/{receipt["run_stamp"].rsplit("_",1)[0]}/{NODES[rank]}/'
            if not uri.startswith(prefix):
                raise RuntimeError('Receipt contains an unexpected storage location')
            relative=Path(uri[len(prefix):])
            if '..' in relative.parts or relative.is_absolute():
                raise RuntimeError('Unsafe receipt path')
            dest=work/NODES[rank]/relative
            dest.parent.mkdir(parents=True,exist_ok=True)
            driver.cloud(['storage','cp',uri,str(dest)])
            if driver.sha256(dest) != uploaded['sha256']:
                raise RuntimeError(f'Download checksum mismatch: {uri}')
            if not dest.name.endswith('.parquet'):
                continue
            frame=pd.read_parquet(dest)
            ts=pd.to_numeric(frame['timestamp_epoch'],errors='coerce')
            if (len(frame) < max(2,duration*.5) or not np.isfinite(ts).all()
                    or not ts.is_monotonic_increasing or ts.max()-ts.min() < duration*.9):
                raise RuntimeError(f'Telemetry is short, incomplete or has invalid timestamps: {uri}')
            rows.append(dict(node=NODES[rank],path=str(dest),samples=len(frame),
                             duration_s=float(ts.max()-ts.min()),sha256=uploaded['sha256']))
            count+=1
        if count != 9:
            raise RuntimeError(f'Expected 8 GPU and 1 NIC final parquet on {NODES[rank]}, found {count}')
    return rows


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--launch',action='store_true')
    a=ap.parse_args()
    campaign=datetime.now(timezone.utc).strftime('followup_a100_%Y%m%dT%H%M%SZ')
    work=Path('/tmp')/campaign
    stages=[dict(name='smoke',duration=60,seed=20261012),
            dict(name='collect',duration=1800,seed=8123401)]
    plan=dict(campaign=campaign,zone=ZONE,nodes=NODES,gpu_model='A100',cell=CELL,
              output=f'{BUCKET}/runs/{campaign}',local_results=str(work),stages=stages,
              status='planned_not_launched',admission='pending_scientific_review')
    print(json.dumps(plan,indent=2),flush=True)
    if not a.launch:
        return
    # Fail before starting instances if the local result checker cannot run.
    import pandas, pyarrow, numpy  # noqa: F401
    work.mkdir()
    def save():
        (work/'campaign.json').write_text(json.dumps(plan,indent=2)+'\n')
    save()
    def interrupted(signum, frame):
        raise KeyboardInterrupt(f'Interrupted by signal {signum}')
    signal.signal(signal.SIGTERM,interrupted)
    owned=False
    try:
        # Ownership precondition: this controller must never stop preexisting work.
        for node in NODES:
            info=json.loads(driver.cloud(['compute','instances','describe',node,
                                          '--project',PROJECT,'--zone',ZONE,'--format=json']))
            if info['status'] != 'TERMINATED':
                raise RuntimeError(f'{node} is not stopped; this campaign will not take it over')
        owned=True
        for stage in stages:
            stamp=f'{campaign}_{stage["name"]}'
            plan.update(status='launching',stage=stage['name']);save()
            cmd=[sys.executable,str(Path(driver.__file__)), '--project',PROJECT,'--zone',ZONE,
                 '--node0',NODES[0],'--node1',NODES[1],'--gpu-model','A100',
                 '--cell',CELL,'--run-stamp',stamp,'--session',campaign,'--bucket',BUCKET,
                 '--seed',str(stage['seed']),'--duration',str(stage['duration']),
                 '--start-stopped','--launch']
            if stage['name']=='smoke':
                cmd+=['--smoke']
            else:
                cmd+=['--smoke-stamp',f'{campaign}_smoke']
            subprocess.run(cmd,check=True,timeout=1500)
            deadline=time.monotonic()+stage['duration']+2400
            plan['status']='collecting';save()
            while time.monotonic() < deadline:
                receipts=[read_receipt(f'{BUCKET}/runs/{campaign}/control/{stamp}/{n}.complete.json') for n in NODES]
                if all(r and r['status']=='local_complete' and r['run_stamp']==stamp for r in receipts):
                    break
                if instance_states() == ['TERMINATED','TERMINATED']:
                    raise RuntimeError(f'{stage["name"]}: both VMs stopped without complete receipts')
                time.sleep(30)
            else:
                raise RuntimeError(f'{stage["name"]}: both success receipts did not arrive')
            stage['receipts']=receipts
            stage['telemetry']=validate_downloads(receipts,work/stage['name'],stage['duration'])
            stage['status']='recorded_and_checked';save()
            wait_stopped(deadline)
        plan['status']='complete_pending_scientific_review';save()
        driver.cloud(['storage','cp',str(work/'campaign.json'),f'{BUCKET}/runs/{campaign}/campaign.json'])
        print(f'Completed smoke and collection; results: {work}/campaign.json',flush=True)
    except BaseException as exc:
        plan.update(status='failed',error=str(exc));save()
        if owned:
            print('Stopping the pair owned by this campaign after failure.',flush=True)
            try:
                driver.cloud(['compute','instances','stop',*NODES,'--project',PROJECT,'--zone',ZONE,'--quiet'])
            except (subprocess.SubprocessError,OSError) as stop_exc:
                print(f'AUTOMATIC CLEANUP FAILED: {stop_exc}',flush=True)
        raise


if __name__ == '__main__':
    main()
