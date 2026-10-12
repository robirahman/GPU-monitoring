import json,pathlib,pandas as pd,collections,hashlib,base64,gzip,csv,io,datetime,shutil
B=pathlib.Path('/tmp/gcs-workload-check-2026-10-12');O=pathlib.Path('/home/robi/GPU-monitoring/docs/workload_review_2026-10-12/gcs_reconciliation')
old=pd.read_csv('/home/robi/workload-audit-2026-10-11/files.csv');oids=set(old.oid)
verified=json.load(open(B/'verified-md5-to-sha.json'))
for f in ['unverified-raw-candidates-downloaded.json','partial-candidates-downloaded.json']:
 if (B/f).exists():
  for r in json.load(open(B/f)):verified[r['md5']]={'sha256':r['sha256'],'source':'download'}
pointers={base64.b64encode(hashlib.md5(f'version https://git-lfs.github.com/spec/v1\noid sha256:{r.oid}\nsize {r.bytes}\n'.encode()).digest()).decode():r.oid for r in old.itertuples()}
for r in json.load(open(B/'log-candidates-downloaded.json')):
 if r['name'].endswith('.parquet'):
  t=pathlib.Path(r['path']).read_text();pointers[r['md5']]=t.split('sha256:')[1].split()[0]
tele=[x['metadata'] for x in json.load(open(B/'telemetry.json')) if x.get('type')=='cloud_object'];home=[x['metadata'] for x in json.load(open(B/'sigil-home-archive.json')) if x.get('type')=='cloud_object']
sigil=json.load(open(B/'sigil-complete-metadata.json')) if (B/'sigil-complete-metadata.json').exists() else []
cloud_rows=[]
for m in tele:
 if not m['name'].endswith('.parquet'):continue
 md5=m.get('md5Hash');v=verified.get(md5,{})
 status='unverified'
 if md5 in pointers:status='lfs_pointer_known_target' if pointers[md5] in oids else 'lfs_pointer_unresolved_target'
 elif '/analysis_cache/' in '/'+m['name']:status='derived_feature_cache'
 elif v:status='exact_sha_in_prior_inventory' if v['sha256'] in oids else 'new_bytes_examined'
 cloud_rows.append({'bucket':m['bucket'],'name':m['name'],'generation':m['generation'],'bytes':int(m['size']),'md5':md5,'sha256':v.get('sha256',''),'lfs_target':pointers.get(md5,''),'status':status})
pd.DataFrame(cloud_rows).to_csv(O/'cloud_parquet_reconciliation.csv',index=False)
summary={'audit_utc':datetime.datetime.now(datetime.timezone.utc).isoformat(),'project':'focus-surfer-494820-g0','scope':'All current live objects in focus-surfer-494820-g0-telemetry. Sigil and sigil-home-archive excluded at the user request. No historical generations or soft-deleted objects.', 'buckets':{},'parquet_object_status':dict(collections.Counter(r['status'] for r in cloud_rows)),'temporary_evidence_root':str(B)}
for name,ms,fn in [('telemetry',tele,'telemetry.json')]:
 summary['buckets'][name]={'objects':len(ms),'bytes':sum(int(m['size']) for m in ms),'complete':bool(ms),'manifest_sha256':hashlib.sha256((B/fn).read_bytes()).hexdigest() if (B/fn).exists() else None}
new=pd.read_csv(O/'new_capture_dispositions.csv');summary['new_capture_dispositions']=new.groupby('disposition').agg(streams=('cell','size'),samples=('samples','sum'),raw_accelerator_hours=('accelerator_hours','sum'),step_aligned_samples=('step_aligned_samples','sum'),step_aligned_hours=('step_aligned_hours','sum')).reset_index().to_dict('records')
summary['raw_new_accelerator_hours']=float(new.accelerator_hours.sum());summary['raw_new_samples']=int(new.samples.sum());summary['baseline_accelerator_hours']=1538.298071907494
summary['accepted_dataset_modified']=False
if (O/'nic_checkpoint_streams.csv').exists():
 checkpoints=pd.read_csv(O/'nic_checkpoint_streams.csv');summary['nic_checkpoint_streams']=checkpoints.groupby('status').agg(streams=('cell','size'),samples=('samples','sum'),extra_samples=('outside_known_span_samples','sum')).reset_index().to_dict('records')

checkpoint_rows=[]
for m in tele:
 if not m['name'].endswith('.parquet.partial'):continue
 v=verified.get(m.get('md5Hash'),{});sha=v.get('sha256','')
 checkpoint_rows.append({'bucket':m['bucket'],'name':m['name'],'generation':m['generation'],'bytes':int(m['size']),'md5':m.get('md5Hash'),'sha256':sha,'status':'exact_sha_in_prior_inventory' if sha in oids else 'checkpoint_or_known_final_payload_examined' if sha else 'unverified'})
pd.DataFrame(checkpoint_rows).to_csv(O/'cloud_checkpoint_manifest.csv',index=False)
summary['checkpoint_object_status']=dict(collections.Counter(r['status'] for r in checkpoint_rows))
summary['additional_nic_finalized_streams']=27
summary['additional_nic_finalized_samples']=10281
summary['additional_nic_checkpoint_streams']=467
summary['additional_nic_checkpoint_samples']=203929
summary['checkpoint_streams_overlapping_prior_main_accelerators']=223

(O/'summary.json').write_text(json.dumps(summary,indent=2))
for f in ['archive-candidates-downloaded.json']:
 shutil.copyfile(B/f,O/f)
# Small manifests remain in the review; large bucket listings stay in /tmp with their hashes above.
for fn in ['compare_captures.py','finish_metadata_check.py','check_nic_checkpoints.py','build_addendum.py']:
 shutil.copyfile(B/fn,O/fn)
print(json.dumps(summary,indent=2))
