"""Failure-path tests for seeded, bounded collection; no GPU/cloud required."""
import argparse
from contextlib import redirect_stdout
import importlib.util
import io
import json
import os
from pathlib import Path
import random
import shlex
import subprocess
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from workloads._seed import add_seed_argument, record_seed
from scripts.gcp import run_followup_node as node
from scripts.gcp import launch_followup_pilot as driver
from scripts.plan_followup_repetitions import make_plan

ROOT = Path(__file__).resolve().parents[1]


class SeedTests(unittest.TestCase):
    def test_default_environment_and_cli_priority_and_invalid_range(self):
        with patch.dict(os.environ, {'SPAR_WORKLOAD_SEED': '98'}):
            parser = argparse.ArgumentParser()
            add_seed_argument(parser)
            self.assertEqual(parser.parse_args([]).seed, 98)
            self.assertEqual(parser.parse_args(['--seed', '99']).seed, 99)
            with redirect_stdout(io.StringIO()), patch('sys.stderr', io.StringIO()):
                with self.assertRaises(SystemExit):
                    parser.parse_args(['--seed', '-1'])

    def test_recorded_auxiliary_generators_and_rank(self):
        def draw():
            output = io.StringIO()
            with patch.dict(os.environ, {'RANK':'9', 'LOCAL_RANK':'1', 'RUN_STAMP':'test'}), redirect_stdout(output):
                record_seed(82)
            return random.random(), np.random.random(), json.loads(output.getvalue().split(' ',1)[1])
        first = draw()
        self.assertEqual(first, draw())
        self.assertEqual(first[2]['rank'], 9)
        self.assertFalse(first[2]['deterministic_algorithms'])

    def test_repetitions_have_distinct_seeds(self):
        jobs = [dict(family='train_tp', hardware='NVIDIA H100 80GB HBM3',
                     cell='s16_train_tp8', excluded_failed_job='False')]
        rows = make_plan(jobs)
        self.assertEqual(len({r['workload_seed'] for r in rows}), 5)
        self.assertEqual(rows, make_plan(jobs))
        self.assertNotEqual(rows, make_plan(jobs, seed=37))
        self.assertTrue(all(f'SPAR_WORKLOAD_SEED={r["workload_seed"]}' in r['queue_command'] for r in rows))


class ReceiptTests(unittest.TestCase):
    def fixture(self, root):
        cell='s16_train_tp8'; stamp='test'; rid=node.run_id(stamp, cell)
        data=root/'data'; data.mkdir()
        logs=root/'logs'; logs.mkdir()
        markers=logs/'.done'/stamp; markers.mkdir(parents=True)
        (markers/f'{cell}_node1_{rid}.done').write_text('rc=0')
        for gpu in range(8):
            (data/f'{cell}_node1_gpu{gpu}_H100_{rid}.parquet').write_bytes(b'fixture')
        (data/f'nic_{cell}_node1_{rid}.parquet').write_bytes(b'fixture')
        log=logs/f'{cell}_node1_{rid}.log'
        log.write_text(''.join('RUN_SEED '+json.dumps(dict(rank=r, base_seed=9, run_stamp=stamp)) for r in range(8,16)))
        with log.open('a') as f:
            for r in range(8,16):
                f.write('RUN_PROGRESS '+json.dumps(dict(rank=r,base_seed=9,run_stamp=stamp,optimizer_updates=2)))
        return cell, stamp, rid, log

    def test_success_requires_all_device_files_and_correct_rank_seeds(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell,stamp,rid,log=self.fixture(root)
            def check(seed=9):
                return node.check_local_result(root,rid,stamp,cell,1,False,seed)
            self.assertTrue(check()['complete'])
            self.assertEqual(check()['scientific_admission'], 'pending_peer_and_progress_review')
            self.assertFalse(check(10)['complete'])
            path=next((root/'data').glob('*gpu7*'))
            path.rename(path.with_name(path.name.replace('gpu7','gpu8')))
            self.assertFalse(check()['complete'])
            log.write_text('RUN_SEED {broken}\n')
            self.assertFalse(check()['seed_records_ok'])

    def test_upload_verifies_download_and_fails_on_corruption(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'data'; path.write_bytes(b'payload')
            remote={}
            def transfer(cmd, **kw):
                if cmd[3].startswith('gs://'):
                    Path(cmd[4]).write_bytes(remote[cmd[3]])
                else:
                    remote[cmd[4]]=Path(cmd[3]).read_bytes()
            with patch.object(node,'checked',side_effect=transfer):
                self.assertTrue(node.upload(path,'gs://test/object',True)['verified'])
            def corrupt(cmd, **kw):
                transfer(cmd, **kw)
                if cmd[3].startswith('gs://'):
                    Path(cmd[4]).write_bytes(b'corruption')
            with patch.object(node,'checked',side_effect=corrupt):
                with self.assertRaisesRegex(RuntimeError, 'hash mismatch'):
                    node.upload(path,'gs://test/object',True)

    def test_actor_progress_does_not_require_optimizer_updates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); cell,stamp,rid,log=self.fixture(root)
            content=log.read_text().replace('"optimizer_updates": 2', '"optimizer_updates": 0, "infer_steps": 3')
            log.write_text(content)
            self.assertFalse(node.check_local_result(root,rid,stamp,cell,1,False,9)['progress_records_ok'])
            # Same records are valid actor progress on host 1 of the IMPALA cell.
            impala='s16_train_impala_a15'
            for path in list(root.rglob('*')):
                if path.is_file() and cell in path.name:
                    path.rename(path.with_name(path.name.replace(cell,impala)))
            self.assertTrue(node.check_local_result(root,rid,stamp,impala,1,False,9)['complete'])

    def test_bundle_includes_working_edits_and_no_corpus(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)/'repo'; (root/'workloads').mkdir(parents=True)
            (root/'workloads/x.py').write_text('x=2\n')
            (root/'workloads/model.pt').write_bytes(b'large')
            (root/'data').mkdir(); (root/'data/x.parquet').write_bytes(b'data')
            dest=Path(tmp)/'src.tgz'
            manifest=driver.bundle(root,dest,'test')
            self.assertEqual(manifest['files'], {'workloads/x.py':node.sha256(root/'workloads/x.py')})
            with tarfile.open(dest) as tar:
                self.assertEqual(set(tar.getnames()), {'workloads/x.py','deployment.json'})
                self.assertEqual(tar.extractfile('workloads/x.py').read(),b'x=2\n')

    def test_source_changes_invalidate_smoke(self):
        args=SimpleNamespace(bucket='gs://test',session='session',smoke_stamp='smoke',
                             node0='a',node1='b',cell='s16_train_tp8')
        receipts=[dict(status='local_complete',smoke=True,cell=args.cell,rank=r,
                       node=('a','b')[r],peer=('b','a')[r],run_stamp='smoke',seed=4,
                       local_result={'complete':True},uploads=[{'verified':True}],
                       deployment={'files':{'old':'hash'}}) for r in (0,1)]
        with patch.object(driver, 'cloud', side_effect=[json.dumps(r) for r in receipts]):
            with self.assertRaisesRegex(RuntimeError,'Source changed'):
                driver.check_smoke(args)


class QueueGuardTests(unittest.TestCase):
    def run_shell(self, script, busy=False):
        # Execute the real queue's function definitions, with inert launchers and
        # isolated logs. Never executes the large queue body or real GPU tools.
        source=(ROOT/'scripts/rebuttal_collection/multinode_queue.sh').read_text().split('# ── The queue:')[0]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); tools=root/'bin'; tools.mkdir()
            nv=tools/'nvidia-smi'; nv.write_text('#!/bin/sh\nprintf "%s" '+shlex.quote('1234' if busy else '')+'\n'); nv.chmod(0o755)
            gc=tools/'gcloud'; gc.write_text('#!/bin/sh\nexit 1\n'); gc.chmod(0o755)
            ns=tools/'netshape.sh'; ns.write_text('#!/bin/sh\ntouch "$REPO/netshape_called"\n'); ns.chmod(0o755)
            env=dict(os.environ,REPO=str(root),PATH=str(tools)+':'+os.environ['PATH'],
                     PY='/bin/true',PY_VLLM='/bin/true',TORCHRUN='/bin/true',RAY='/bin/true',
                     GCPDIR=str(tools),MASTER_ADDR='a',NODE_RANK='0',RUN_STAMP='test',
                     NNODES='2',GPU_MEM_GB='80',NODE_NAME='a',PEER='b',
                     BUCKET='gs://test',SESSION='test',SPAR_REFUSE_BUSY_GPU='1',
                     SPAR_STRICT_BARRIER='1',ONLY_CELLS='^s16_train_tp8$',GAP='0')
            result=subprocess.run(['bash','-c',source+'\n'+script],env=env,cwd=root,
                                  capture_output=True,text=True,timeout=10)
            return result,{p.name for p in root.iterdir()}

    def test_filtered_shaped_cell_never_changes_network(self):
        result,files=self.run_shell('mn_torchrun_bw s16_train_localsgd_h50_bw100 rid 100 -- /bin/true')
        self.assertEqual(result.returncode,0,result.stderr)
        self.assertNotIn('netshape_called',files)

    def test_storage_barrier_failure_refuses_workload(self):
        result,files=self.run_shell('run_stage s16_train_tp8 rid -- touch launched')
        self.assertNotEqual(result.returncode,0)
        self.assertNotIn('launched',files)

    def test_busy_gpu_refuses_workload(self):
        result,files=self.run_shell('cell_barrier() { return 0; }; run_stage s16_train_tp8 rid -- touch launched',busy=True)
        self.assertNotEqual(result.returncode,0)
        self.assertIn('refusing to terminate',result.stdout)
        self.assertNotIn('launched',files)
        self.assertNotIn('netshape_called',files)


class CheckpointTests(unittest.TestCase):
    def test_distributed_collector_checkpoints_use_final_stream_identity(self):
        # Fake only the hardware boundary; exercise the actual launcher.
        collectors=[]
        class Collector:
            def __init__(self, **kw):
                self.metadata=SimpleNamespace(gpu_name='H100'); self.kw=kw
                collectors.append(self)
            def start(self):
                self.path_at_start=self.checkpoint_path
            def stop(self): pass
            def save(self,path): self.saved=path
        spec=importlib.util.spec_from_file_location('test_multi_launcher',ROOT/'scripts/run_workload_multi_gpu.py')
        module=importlib.util.module_from_spec(spec)
        with patch.dict('sys.modules', {'scripts.collect_telemetry':SimpleNamespace(TelemetryCollector=Collector),
                                      'workloads.registry':SimpleNamespace(get_workload=None,WORKLOAD_REGISTRY={})}):
            spec.loader.exec_module(module)
        with tempfile.TemporaryDirectory() as tmp, patch.object(module.time,'sleep'), patch.object(module,'_run_reaped',return_value=(0,'OK')):
            args=SimpleNamespace(num_gpus=2,run_id='123456789abc',output_dir=tmp,workload='s16_train_tp8',
                                 node_rank=1,interval=1,no_dcgm=True,checkpoint_sec=60,timeout=60)
            self.assertEqual(module.run_multi_gpu(args,['true']),('123456789abc',0))
        self.assertEqual(len({c.path_at_start for c in collectors}),2)
        for c in collectors:
            self.assertEqual(c.path_at_start,c.saved+'.partial')
            self.assertEqual(c.kw['checkpoint_sec'],60)


if __name__ == '__main__':
    unittest.main()
