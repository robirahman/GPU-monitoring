import pathlib,json,collections,sys,hashlib
import pandas as pd,numpy as np,pyarrow.parquet as pq
sys.path.insert(0,'/home/robi/GPU-monitoring')
from scripts.audit_data_inventory import timestamp_series,filename_identity,time_summary
from classifier.workload_labels import canonical_workload
B=pathlib.Path('/tmp/gcs-workload-check-2026-10-12');O=pathlib.Path('/home/robi/GPU-monitoring/docs/workload_review_2026-10-12/gcs_reconciliation')
old=pd.read_csv('/home/robi/workload-audit-2026-10-11/streams.csv',keep_default_na=False);idx=collections.defaultdict(list)
for r in old[old.role.eq('nic')].to_dict('records'):idx[(r['run_id'],r['hostname'],canonical_workload(r['workload_label']))].append((r['t0'],r['t1']))
for r in pd.read_csv(O/'candidate_streams.csv').query("role=='nic'").to_dict('records'):idx[(r['run_id'],r['hostname'],r['cell'])].append((r['t0'],r['t1']))
known=set(pd.read_csv('/home/robi/workload-audit-2026-10-11/files.csv').oid);rs=json.load(open(B/(sys.argv[1] if len(sys.argv)>1 else 'partial-candidates-downloaded.json')))+json.load(open(B/'archive-checkpoint-members.json'));rs=list({r['sha256']:r for r in rs}.values());ss={};fr=[];errs=[]
for r in rs:
 if r['sha256'] in known:
  fr.append({'name':r['name'],'sha256':r['sha256'],'status':'exact_sha_in_prior_inventory'});continue
 try:
  pf=pq.ParquetFile(r['path']);cols=[c for c in ['timestamp_epoch','timestamp_utc','ts','hostname','run_id','workload_label'] if c in pf.schema.names];d=pf.read(columns=cols).to_pandas();d['t']=timestamp_series(d)
  for key,g in d.groupby(['run_id','hostname','workload_label']):
   key=(str(key[0]),str(key[1]),canonical_workload(str(key[2])));z=ss.setdefault(key,{'times':[],'paths':set(),'hashes':set()});z['times'].append(g.t.to_numpy());z['paths'].add(r['name']);z['hashes'].add(r['sha256'])
  fr.append({'name':r['name'],'sha256':r['sha256'],'status':'parsed_checkpoint'})
 except Exception as e:errs.append({'name':r['name'],'error':type(e).__name__+': '+str(e)})
out=[]
for (rid,host,cell),z in ss.items():
 ts=np.unique(np.concatenate(z['times']));ts=ts[np.isfinite(ts)];matches=idx.get((rid,host,cell),[]);covered=np.zeros(len(ts),dtype=bool)
 for t0,t1 in matches:covered|=(ts>=t0-0.001)&(ts<=t1+0.001)
 status='represented_by_known_span' if covered.all() else 'extends_known_span' if matches else 'unrepresented_nic_checkpoint'
 out.append({'run_id':rid,'hostname':host,'cell':cell,**time_summary(ts),'outside_known_span_samples':int((~covered).sum()),'status':status,'paths':'|'.join(sorted(z['paths'])),'hashes':'|'.join(sorted(z['hashes']))})
pd.DataFrame(out).to_csv(O/'nic_checkpoint_streams.csv',index=False);pd.DataFrame(fr).to_csv(O/'nic_checkpoint_files.csv',index=False);pd.DataFrame(errs).to_csv(O/'nic_checkpoint_read_errors.csv',index=False)
print('file checks',collections.Counter(x['status'] for x in fr),'errors',errs)
a=pd.DataFrame(out);print(a.groupby('status').agg(streams=('cell','size'),samples=('samples','sum'),extra_samples=('outside_known_span_samples','sum')).to_string());print('unrepresented cells',a[a.status.eq('unrepresented_nic_checkpoint')].groupby('cell').size().to_string())
