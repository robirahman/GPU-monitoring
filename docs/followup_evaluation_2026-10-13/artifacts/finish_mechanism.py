import datetime,gzip,hashlib,json,subprocess,time
from pathlib import Path
BASE=Path('/var/tmp/spar_followup/20261012T232425Z/schema_retry')
WORK=BASE/'results'
DEST='gs://focus-surfer-494820-g0-telemetry/followup_analysis/20261012T232425Z/reviewed_20261013/artifacts/'
UNIT='spar-followup-analysis-20261012T232425Z-schema1.service'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
start=time.monotonic()
while True:
 try:
  result=json.loads((WORK/'mechanism_paired.json').read_text())
  assert result['n_windows']==19298 and len(result['families'])==43
  assert result['windows_sha256']=='046664cd164e3be5d415816e889ef4d8e8a126333526b0ef90740a4bd881ab14'
  break
 except (FileNotFoundError,json.JSONDecodeError):
  if time.monotonic()-start>7200:raise TimeoutError('Mechanism completion watcher exceeded two hours')
  time.sleep(15)
source=WORK/'mechanism_paired.splits.json';packed=Path(str(source)+'.gz')
with source.open('rb') as inp,packed.open('wb') as out:
 with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as gz:gz.write(inp.read())
files=[WORK/('mechanism_paired'+suffix) for suffix in ['.json','.predictions.parquet','.splits.json.gz','.log']]
files += [WORK/'windows_30_15.parquet.provenance.json',WORK/'cache.log',WORK/'execution.json']
manifest=dict(status='mechanism_saved',time_utc=now(),artifacts=[])
for p in files:
 assert p.is_file() and p.stat().st_size
 subprocess.run(['gcloud','storage','cp',str(p),DEST+p.name],check=True)
 meta=json.loads(subprocess.check_output(['gcloud','storage','objects','describe',DEST+p.name,'--format=json'],text=True))
 manifest['artifacts'].append(dict(name=p.name,uri=DEST+p.name,generation=str(meta['generation']),bytes=p.stat().st_size,sha256=hashlib.file_digest(p.open('rb'),'sha256').hexdigest() if hasattr(hashlib,'file_digest') else hashlib.sha256(p.read_bytes()).hexdigest()))
try:
 worker=json.loads(subprocess.check_output(['gcloud','storage','cat','gs://focus-surfer-494820-g0-telemetry/followup_analysis/20261012T232425Z/sensitivity_worker_20261013/worker_execution.json'],text=True))
 if worker.get('status')=='complete' and set(worker.get('completed',[]))=={'backbone','steps_only'}:
  manifest['remaining_original_stages']='superseded by completed dedicated-worker evaluations'
  subprocess.run(['systemctl','stop',UNIT],check=True)
  manifest['original_evaluation_unit_stopped']=True
 else:
  manifest['remaining_original_stages']='original runner retained as fallback because dedicated worker is not complete'
except subprocess.CalledProcessError as exc:
 manifest['handoff_error']=str(exc)
p=WORK/'mechanism_handoff.json';p.write_text(json.dumps(manifest,indent=2)+'\n')
subprocess.run(['gcloud','storage','cp',str(p),DEST+p.name],check=True)
