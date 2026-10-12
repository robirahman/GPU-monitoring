import sys,json,pathlib,collections,hashlib
import numpy as np,pandas as pd,pyarrow.parquet as pq
sys.path.insert(0,'/home/robi/GPU-monitoring')
from scripts.audit_data_inventory import timestamp_series,filename_identity,time_summary
from classifier.workload_labels import canonical_workload,threeway_label,workload_family
from classifier.data_sources import data_role
B=pathlib.Path('/tmp/gcs-workload-check-2026-10-12')
OUT=pathlib.Path('/home/robi/GPU-monitoring/docs/workload_review_2026-10-12/gcs_reconciliation');OUT.mkdir(exist_ok=True)
old=pd.read_csv('/home/robi/workload-audit-2026-10-11/streams.csv',keep_default_na=False)
old['canon']=old.workload_label.map(canonical_workload)
index=collections.defaultdict(list);loose=collections.defaultdict(list)
for r in old.to_dict('records'):
    index[(r['run_id'],r['gpu_uuid'],r['hostname'],r['canon'],r['role'])].append(r)
    loose[(r['run_id'],r['gpu_uuid'],r['canon'],r['role'])].append(r)
known_oids=set(pd.read_csv('/home/robi/workload-audit-2026-10-11/files.csv').oid)
records=json.load(open(B/'raw-download-candidates-downloaded.json'))+json.load(open(B/'downloaded-so-far.json'))
records=list({r['sha256']:r for r in records}.values())
streams={};file_rows=[];errors=[]
for i,r in enumerate(records):
    if r['sha256'] in known_oids:
        file_rows.append(dict(name=r['name'],sha256=r['sha256'],status='exact_hash_in_inventory'));continue
    try:
        pf=pq.ParquetFile(r['path']);names=set(pf.schema.names)
        if 'window_start' in names and not names & {'ts','timestamp_epoch'}:
            file_rows.append(dict(name=r['name'],sha256=r['sha256'],status='derived_features'));continue
        cols=[c for c in ('ts','timestamp_epoch','timestamp','timestamp_utc','time','epoch','epoch_s','t',
          'gpu_name','gpu_uuid','hostname','run_id','workload_label','gpu_utilization_pct','tensorcore_util_pct','gpu_index','gpu') if c in names]
        d=pf.read(columns=cols).to_pandas();d['t']=timestamp_series(d)
        d=d[np.isfinite(d.t)&d.t.between(1e8,4e9)].copy()
        if d.empty:raise ValueError('No finite epoch data')
        lab,hw,rid,host,device=filename_identity(r['name']);role=data_role(r['name'])
        for col,value in [('workload_label',lab),('gpu_name',hw),('gpu_uuid',device or r['sha256'][:12]),('hostname',host),('run_id','source:'+r['sha256'])]:
            if col not in d:d[col]=value
            d[col]=d[col].fillna(value).astype(str).replace({'':value,'nan':value,'None':value})
        d['canon']=d.workload_label.map(canonical_workload)
        for ident,g in d.groupby(['run_id','gpu_uuid','hostname','canon'],dropna=False):
            key=(*ident,role)
            z=streams.setdefault(key,dict(run_id=ident[0],gpu_uuid=ident[1],hostname=ident[2],cell=ident[3],role=role,
                hardware=str(g.gpu_name.iloc[0]),times=[],names=set(),hashes=set(),max_util=-1.,max_tensor=-1.))
            z['times'].append(g.t.to_numpy());z['names'].add(r['name']);z['hashes'].add(r['sha256'])
            for column,field in [('gpu_utilization_pct','max_util'),('tensorcore_util_pct','max_tensor')]:
                if column in g:z[field]=max(z[field],float(g[column].max()))
        file_rows.append(dict(name=r['name'],sha256=r['sha256'],status='parsed_raw_or_sidecar'))
    except Exception as exc:errors.append(dict(name=r['name'],error=type(exc).__name__+': '+str(exc)))
    if (i+1)%500==0:print('Parsed',i+1,'/',len(records),flush=True)
output=[]
for key,z in streams.items():
    ts=np.unique(np.concatenate(z.pop('times')));z.update(time_summary(ts));z['accelerator_hours']=z['span_s']/3600 if z['role']=='accelerator' else 0
    z['paths']='|'.join(sorted(z.pop('names')));z['hashes']='|'.join(sorted(z['hashes']))
    matches=index.get(key,[]);match_type='exact_identity'
    if not matches:matches=loose.get((key[0],key[1],key[3],key[4]),[]);match_type='host_alias'
    if matches:
        covered=np.zeros(len(ts),dtype=bool)
        for match in matches:covered|=(ts>=float(match['t0'])-0.001)&(ts<=float(match['t1'])+0.001)
        z['outside_known_span_samples']=int((~covered).sum())
        disposition='accepted' if any(str(m['quarantine']).lower()=='false' for m in matches) else 'quarantine'
        z['status']='represented_'+disposition if covered.all() else 'extends_'+disposition
        z['match_type']=match_type
    else:
        z['status']='unrepresented';z['outside_known_span_samples']=len(ts);z['match_type']='none'
    z['objective']=threeway_label(z['cell']);z['family']=workload_family(z['cell'])
    output.append(z)
d=pd.DataFrame(output);d.to_csv(OUT/'candidate_streams.csv',index=False)
pd.DataFrame(file_rows).to_csv(OUT/'checked_files.csv',index=False);pd.DataFrame(errors).to_csv(OUT/'read_errors.csv',index=False)
new=d[d.status.eq('unrepresented')]
g=new.groupby(['role','run_id','cell','hardware']).agg(streams=('gpu_uuid','nunique'),hosts=('hostname','nunique'),
    samples=('samples','sum'),accelerator_hours=('accelerator_hours','sum'),first_epoch=('t0','min'),last_epoch=('t1','max'),max_util=('max_util','max'),max_tensor=('max_tensor','max'),paths=('paths',lambda x:'|'.join(x))).reset_index()
g.to_csv(OUT/'unrepresented_runs.csv',index=False)
print('STATUS',d.groupby(['role','status']).agg(streams=('run_id','size'),samples=('samples','sum'),hours=('accelerator_hours','sum')).round(3).to_string())
print('NEW',g.drop(columns=['paths']).round(3).to_string(index=False))
print('errors',errors[:8])
