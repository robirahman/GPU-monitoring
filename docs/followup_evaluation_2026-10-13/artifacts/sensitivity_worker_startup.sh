#!/bin/bash
set -euo pipefail
exec > >(tee -a /var/log/followup-sensitivity.log) 2>&1
export DEBIAN_FRONTEND=noninteractive
mkdir -p /opt/followup
cat > /opt/followup/cloud_io.py <<'PY'
import hashlib,json,urllib.request,urllib.parse
from pathlib import Path
BUCKET='focus-surfer-494820-g0-telemetry'
PREFIX='followup_analysis/20261012T232425Z/sensitivity_worker_20261013/'
def token():
 req=urllib.request.Request('http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/token',headers={'Metadata-Flavor':'Google'})
 return json.load(urllib.request.urlopen(req,timeout=30))['access_token']
def download():
 obj='followup_analysis/20261012T232425Z/reviewed_20261013/frozen_review_inputs.tar.gz'
 url=f'https://storage.googleapis.com/storage/v1/b/{BUCKET}/o/{urllib.parse.quote(obj,safe="")}?alt=media&generation=1791861780954107'
 req=urllib.request.Request(url,headers={'Authorization':'Bearer '+token()})
 h=hashlib.sha256()
 with urllib.request.urlopen(req,timeout=120) as response, open('/opt/followup/inputs.tar.gz','wb') as f:
  while True:
   block=response.read(1024*1024)
   if not block: break
   f.write(block);h.update(block)
 assert h.hexdigest()=='852466d4f7a444db9cd0eadef15570f23c70db5f7f5efdad231f7a9dcb518672'
def upload(path,name=None):
 path=Path(path); data=path.read_bytes(); obj=PREFIX+(name or path.name)
 url=f'https://storage.googleapis.com/upload/storage/v1/b/{BUCKET}/o?uploadType=media&name={urllib.parse.quote(obj,safe="")}'
 req=urllib.request.Request(url,data=data,method='POST',headers={'Authorization':'Bearer '+token(),'Content-Type':'application/octet-stream'})
 with urllib.request.urlopen(req,timeout=180) as response: result=json.load(response)
 return dict(name=obj,uri=f'gs://{BUCKET}/{obj}',generation=result['generation'],bytes=len(data),sha256=hashlib.sha256(data).hexdigest(),crc32c=result['crc32c'])
if __name__=='__main__':download()
PY
python3 /opt/followup/cloud_io.py
tar -xzf /opt/followup/inputs.tar.gz -C /opt/followup
apt-get update -q
apt-get install -y -q python3-venv
python3 -m venv /opt/followup/venv
/opt/followup/venv/bin/pip install numpy==2.2.6 pandas==2.3.3 scipy==1.15.3 pyarrow==25.0.1 scikit-learn==1.7.2 joblib==1.6.0 threadpoolctl==3.7.0 python-dateutil==2.9.0.post0 pytz==2026.5 tzdata==2026.5
cat > /opt/followup/run.py <<'PY'
import datetime,gzip,json,os,platform,subprocess,sys,time
from pathlib import Path
from cloud_io import upload
base=Path('/opt/followup');work=base/'results';repo=base/'repo'
now=lambda:datetime.datetime.now(datetime.timezone.utc).isoformat()
env=dict(os.environ,PYTHONPATH=str(repo),SPAR_WORK=str(work),PYTHONUNBUFFERED='1',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1')
status=dict(started_utc=now(),status='running',executor='spar-followup-eval-20261013',python=sys.version,platform=platform.platform(),cpu_count=os.cpu_count(),completed=[],stages=[],uploads=[])
def save():
 p=work/'worker_execution.json';p.write_text(json.dumps(status,indent=2)+'\n');upload(p)
def durable(path):
 for attempt in range(3):
  try:return upload(path)
  except Exception:
   if attempt==2:raise
   time.sleep(10)
try:
 save()
 for name,flags in [('backbone',['--family-level','backbone']),('steps_only',['--steps-only'])]:
  cmd=[sys.executable,'-m','classifier.rebuttal_analysis.paired_family_lofo','--windows',str(work/'windows_30_15.parquet'),'--trees','300','--seed','0','--threshold','0.5',*flags,'--out',str(work/(name+'.json'))]
  stage=dict(name=name,command=cmd,started_utc=now());status['stages'].append(stage);status['phase']=name;save()
  with (work/(name+'.log')).open('w') as log:
   result=subprocess.run(cmd,cwd=repo,env=env,stdout=log,stderr=subprocess.STDOUT)
  stage.update(returncode=result.returncode,finished_utc=now())
  if result.returncode:raise RuntimeError(f'{name} exited {result.returncode}')
  src=work/(name+'.splits.json');packed=Path(str(src)+'.gz')
  with src.open('rb') as inp,packed.open('wb') as out:
   with gzip.GzipFile(filename='',mode='wb',fileobj=out,mtime=0) as gz:gz.write(inp.read())
  for suffix in ['.json','.predictions.parquet','.splits.json.gz','.log']:
   status['uploads'].append(durable(work/(name+suffix)))
  status['completed'].append(name);save()
 status.update(status='complete',finished_utc=now());save()
 durable('/var/log/followup-sensitivity.log')
except Exception as exc:
 status.update(status='failed',error=repr(exc),finished_utc=now())
 try:
  save()
  for p in work.glob('*.log'):durable(p)
  durable('/var/log/followup-sensitivity.log')
 finally:raise
PY
/opt/followup/venv/bin/python -u /opt/followup/run.py
poweroff
