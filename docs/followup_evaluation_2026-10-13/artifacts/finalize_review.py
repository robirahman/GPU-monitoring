"""Durable review/report handoff independent of the Cloud Shell session."""
import datetime,hashlib,json,os,shutil,subprocess,tarfile,time
from pathlib import Path
BASE=Path('/var/tmp/spar_followup/20261012T232425Z')
WORK=BASE/'schema_retry/results';FROZEN=BASE/'schema_retry/repo'
PUB=BASE/'review_publication';ART=PUB/'docs/followup_evaluation_2026-10-13/artifacts'
ROOT_URI='gs://focus-surfer-494820-g0-telemetry/followup_analysis/20261012T232425Z/'
DEST=ROOT_URI+'reviewed_20261013/'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cp(uri,path):
 for attempt in range(3):
  try:return subprocess.run(['gcloud','storage','cp',uri,str(path)],check=True)
  except subprocess.CalledProcessError:
   if attempt==2:raise
   time.sleep(10)
def cloud_json(uri):
 for attempt in range(3):
  try:return json.loads(subprocess.check_output(['gcloud','storage','cat',uri],text=True))
  except subprocess.CalledProcessError:
   if attempt==2:raise
   time.sleep(10)
def save_status(status,**extra):
 d=dict(status=status,updated_utc=now(),**extra);p=PUB/'review_pipeline_status.json';p.write_text(json.dumps(d,indent=2)+'\n');cp(str(p),DEST+p.name)
PUB.mkdir(parents=True,exist_ok=True)
try:
 archive=PUB/'publication_checkpoint.tar.gz';cp(DEST+'publication_work_checkpoint.tar.gz',archive)
 assert sha(archive)=='6c9dac8bec6e90058be9f1f6d9a31955931aed821c78ffe7eca2641e7b7c547c'
 with tarfile.open(archive) as t:t.extractall(PUB)
 save_status('waiting_for_complete_evaluations',github_publication='pending; no GitHub credential copied to VM')
 deadline=time.monotonic()+10800
 while True:
  worker=cloud_json(ROOT_URI+'sensitivity_worker_20261013/worker_execution.json')
  if worker.get('status')=='failed':raise RuntimeError(worker.get('error','Sensitivity worker failed'))
  if worker.get('status')=='complete' and (WORK/'mechanism_handoff.json').is_file():break
  if time.monotonic()>deadline:raise TimeoutError('Review pipeline waited three hours')
  time.sleep(30)
 save_status('assembling_artifacts')
 (ART/'worker_execution.json').write_text(json.dumps(worker,indent=2)+'\n')
 allowed={n+s for n in ['backbone','steps_only'] for s in ['.json','.predictions.parquet','.splits.json.gz','.log']}
 for item in worker['uploads']:
  name=Path(item['name']).name
  if name not in allowed:raise ValueError('Unexpected worker artifact')
  p=ART/name;cp(item['uri']+'#'+str(item['generation']),p)
  assert sha(p)==item['sha256']
 handoff=json.loads((WORK/'mechanism_handoff.json').read_text())
 for item in handoff['artifacts']:
  p=ART/item['name'];shutil.copyfile(WORK/item['name'],p);assert sha(p)==item['sha256']
 shutil.copyfile(WORK/'mechanism_handoff.json',ART/'mechanism_handoff.json')
 # Keep the existing shared analysis environment unchanged.
 venv=PUB/'venv'
 if not venv.exists():shutil.copytree('/opt/spar/repo/venv-analysis',venv,symlinks=True)
 python=str(venv/'bin/python')
 assert Path(subprocess.check_output([python,'-c','import sys;print(sys.prefix)'],text=True).strip())==venv
 subprocess.run([python,'-m','pip','install','matplotlib==3.10.7'],check=True)
 env=dict(os.environ,PYTHONPATH=str(FROZEN),OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
 save_status('auditing_complete_results')
 with (ART/'result_audit.log').open('w') as log:
  subprocess.run([python,str(PUB/'scripts/review_followup_evaluation.py'),'--results',str(ART),'--windows',str(WORK/'windows_30_15.parquet'),'--out',str(ART/'review.json')],cwd=FROZEN,env=env,stdout=log,stderr=subprocess.STDOUT,check=True)
 subprocess.run([python,str(PUB/'scripts/summarize_followup_evaluation.py'),'--results',str(ART),'--review',str(ART/'review.json'),'--out',str(ART.parent),'--plots'],cwd=FROZEN,env=env,check=True)
 validation=json.loads((ART/'validation.json').read_text());validation.update(result_bundle_audit='passed',result_audit='review.json',audit_finished_utc=now(),github_lfs_readback='pending publication')
 (ART/'validation.json').write_text(json.dumps(validation,indent=2)+'\n')
 cache_meta=json.loads(subprocess.check_output(['gcloud','storage','objects','describe',DEST+'windows_30_15.parquet','--format=json'],text=True))
 frozen_meta=json.loads(subprocess.check_output(['gcloud','storage','objects','describe',DEST+'frozen_review_inputs.tar.gz','--format=json'],text=True))
 manifest=dict(created_utc=now(),base_commit='56ab59208009bdfc26d3d61abb52de86a716b70b',branch='results/corrected-multinode-evaluation-2026-10-13',status='automated review passed; human review and GitHub push pending',cache=dict(uri=DEST+'windows_30_15.parquet',generation=str(cache_meta['generation']),sha256='046664cd164e3be5d415816e889ef4d8e8a126333526b0ef90740a4bd881ab14'),frozen_source_and_inputs=dict(uri=DEST+'frozen_review_inputs.tar.gz',generation=str(frozen_meta['generation']),sha256='852466d4f7a444db9cd0eadef15570f23c70db5f7f5efdad231f7a9dcb518672'),cloud_artifacts=worker['uploads']+handoff['artifacts'],files_sha256={str(p.relative_to(PUB)):sha(p) for p in ART.parent.rglob('*') if p.is_file()})
 (ART/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n')
 ready=PUB/'publication_ready.tar.gz'
 with tarfile.open(ready,'w:gz') as t:
  for path in ['.gitattributes','PUBLICATION_RESUME.json','scripts/review_followup_evaluation.py','scripts/summarize_followup_evaluation.py','tests/test_followup_result_review.py','docs/followup_evaluation_2026-10-13']:t.add(PUB/path,arcname=path)
 cp(str(ready),DEST+ready.name)
 cp(str(ART.parent/'README.md'),DEST+'RESULTS.md')
 cp(str(ART/'review.json'),DEST+'review.json')
 save_status('review_ready',publication_bundle=DEST+ready.name,publication_bundle_sha256=sha(ready),github_publication='pending human review and authenticated branch push',persistent_vm_directory=str(PUB))
except Exception as exc:
 try:save_status('failed',error=repr(exc),persistent_vm_directory=str(PUB))
 finally:raise
