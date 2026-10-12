import csv
import hashlib
import html
import json
import subprocess
from pathlib import Path

import pandas as pd

from classifier.workload_labels import LABEL_SCHEMA_VERSION, threeway_label, workload_kind

SOURCE = Path('/home/robi/workload-audit-2026-10-11')
OUT = Path('/home/robi/GPU-monitoring/docs/workload_review_2026-10-12')
OUT.mkdir(parents=True, exist_ok=True)

def digest(p):
    return hashlib.sha256(Path(p).read_bytes()).hexdigest()

def category(row):
    f = row['family']
    if f == 'physical_resnet_evasion': return 'Physical-sensor evasion probes'
    if f in ('adversarial_mm', 'adversarial_llmc'): return 'llm.c memory and timing disguises'
    if f.startswith('symgemm') or f == 'ppo_rlhf': return 'Symmetric-GEMM ladder and single-GPU PPO'
    if f in ('train_localsgd', 'train_streamdiloco', 'train_decoupdiloco', 'train_swarm'):
        return 'Local SGD, DiLoCo and SWARM'
    if f in ('train_impala', 'train_ppodist', 'train_apexrl'): return 'Distributed RL'
    if f in ('train_moe', 'train_tp', 'train_contrastive', 'train_diffusion'):
        return 'MoE, tensor-parallel, contrastive and diffusion training'
    if f in ('train_fsdp', 'train_ddp', 'train_lora', 'train_pipe'):
        return 'Distributed FSDP, DDP, LoRA and pipeline training'
    if f in ('multinode_interleave', 'advtrain_bubble', 'advtrain_flatarena', 'advtrain_typeconf'):
        return 'Distributed evasion and matched filler controls'
    if f == 'inferserve_async': return 'Online LLM serving'
    if f.startswith(('inferrep', 'infershard', 'inferhf')):
        return 'Replicated and sharded LLM inference'
    if f.startswith(('inferasr', 'infercnn', 'inferdiff', 'inferembed', 'inferrec', 'infervis')):
        return 'Replicated modality and distributed recommendation inference'
    if f.startswith('nonml_') or f == 'stalled_train': return 'Distributed non-ML and stall controls'
    if f.startswith(('adversarial', 'whitebox', 'dilution_sweep')):
        return 'Legacy evasion, white-box extensions and controls'
    return {'ml_training':'Legacy conventional training', 'ml_inference':'Legacy conventional inference',
            'other':'Legacy non-ML, idle and communication controls'}[row['job_objective']]

jobs = pd.read_csv(SOURCE / 'jobs.csv', keep_default_na=False)
jobs = jobs.rename(columns={'threeway_label':'snapshot_job_objective', 'collection_status':'oct11_collection_status'})
jobs['job_objective'] = jobs.cell.map(threeway_label)
jobs['kind'] = jobs.cell.map(workload_kind)
jobs['category'] = jobs.apply(category, axis=1)
excluded = {r['job_id'] for r in json.loads(Path('data/quality_exclusions.json').read_text())['excluded_jobs']}
jobs['excluded_failed_job'] = jobs.job_id.isin(excluded)
jobs['multi_host_observed'] = jobs.hosts > 1
jobs['elapsed_job_hours'] = jobs.elapsed_span_s / 3600
jobs['nonexcluded_accelerator_hours'] = jobs.stream_hours.where(~jobs.excluded_failed_job, 0)
jobs['excluded_accelerator_hours'] = jobs.stream_hours.where(jobs.excluded_failed_job, 0)
jobs.to_csv(OUT / 'jobs.csv', index=False)

w = pd.read_csv(SOURCE / 'workloads.csv')
w = w.rename(columns={'threeway_label':'snapshot_job_objective',
                      'log_completed_jobs':'oct11_log_completed_jobs',
                      'failed_or_incomplete_jobs':'oct11_failed_or_incomplete_jobs'})
w['job_objective'] = w.cell.map(threeway_label)
w['kind'] = w.cell.map(workload_kind)
w['category'] = w.apply(category, axis=1)
ex = jobs.groupby(['cell','hardware']).agg(excluded_failed_jobs=('excluded_failed_job','sum'),
    excluded_accelerator_hours=('excluded_accelerator_hours','sum'),
    nonexcluded_accelerator_hours=('nonexcluded_accelerator_hours','sum')).reset_index()
w = w.merge(ex, left_on=['cell','gpu_name'], right_on=['cell','hardware'], validate='1:1').drop(columns='hardware')
w['median_stream_minutes'] = w.median_stream_seconds / 60
w.to_csv(OUT / 'workloads.csv', index=False)

def aggregate(keys):
    return jobs.groupby(keys).agg(cells=('cell','nunique'), jobs=('job_id','size'),
        samples=('samples','sum'), accelerator_hours=('stream_hours','sum'),
        elapsed_job_hours=('elapsed_job_hours','sum'),
        median_job_minutes=('elapsed_span_s',lambda v:v.median()/60),
        excluded_failed_jobs=('excluded_failed_job','sum'),
        nonexcluded_accelerator_hours=('nonexcluded_accelerator_hours','sum'),
        multi_host_jobs=('multi_host_observed','sum'),
        hardware=('hardware',lambda v:'; '.join(sorted(set(v)))),
        observed_devices=('observed_devices',lambda v:','.join(map(str,sorted(set(v))))),
        observed_hosts=('hosts',lambda v:','.join(map(str,sorted(set(v)))))).reset_index()

c = aggregate('category')
c.to_csv(OUT / 'categories.csv', index=False)
h = jobs.groupby('hardware').agg(
    cells=('cell','nunique'),jobs=('job_id','size'),samples=('samples','sum'),
    accelerator_hours=('stream_hours','sum'),multi_host_jobs=('multi_host_observed','sum'),
    nonexcluded_accelerator_hours=('nonexcluded_accelerator_hours','sum'),
    observed_devices=('observed_devices',lambda v:','.join(map(str,sorted(set(v))))),
    observed_hosts=('hosts',lambda v:','.join(map(str,sorted(set(v)))))).reset_index()
h.to_csv(OUT / 'hardware.csv',index=False)
aggregate('family').to_csv(OUT / 'families.csv',index=False)

# These eight small, non-overlapping captures were absent from the saved scan.
supp = []
for p in sorted(Path('week7/results/edge_cases').glob('*_telemetry.json')):
    d = pd.DataFrame(json.loads(p.read_text()))
    spans = d.groupby('gpu').t.agg(['min','max'])
    supp.append(dict(path=str(p),sha256=digest(p),cell=p.name.removesuffix('_telemetry.json'),
        samples=len(d),observed_devices=d.gpu.nunique(),hardware='unknown in telemetry JSON',
        first_epoch=d.t.min(),last_epoch=d.t.max(),elapsed_seconds=d.t.max()-d.t.min(),
        accelerator_hours=((spans['max']-spans['min']).sum()/3600),
        note='Supplemental legacy capture, excluded from snapshot totals; too short for a 30-second window.'))
supp = pd.DataFrame(supp)
supp.to_csv(OUT / 'supplemental_traces.csv',index=False)

checks = {'pointer_paths':0,'nonpointer_paths':0,'missing_paths':[], 'changed_paths':[]}
for row in csv.DictReader((SOURCE / 'files.csv').open()):
    p = Path(row['path'])
    if not p.is_file(): checks['missing_paths'].append(str(p)); continue
    with p.open('rb') as f: head = f.read(200)
    if head.startswith(b'version https://git-lfs.github.com/spec/v1'):
        checks['pointer_paths'] += 1
        fields = dict(x.split(' ',1) for x in head.decode().splitlines())
        same = fields['oid'].split(':')[1] == row['oid'] and int(fields['size']) == int(row['bytes'])
    else:
        checks['nonpointer_paths'] += 1
        same = digest(p) == row['oid'] and p.stat().st_size == int(row['bytes'])
    if not same: checks['changed_paths'].append(str(p))
assert not checks['missing_paths'] and not checks['changed_paths'], checks
assert len(jobs) == 2888 and jobs.cell.nunique() == 529
assert abs(jobs.stream_hours.sum()-1538.298071907494) < 1e-6
assert jobs.excluded_failed_job.sum() == 21
assert abs(c.accelerator_hours.sum()-jobs.stream_hours.sum()) < 1e-6
assert w.jobs.sum() == len(jobs)

provenance = dict(
    review_date='2026-10-12',checkout=subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
    source_directory=str(SOURCE),source_hashes={p.name:digest(p) for p in SOURCE.glob('*.csv')},
    original_summary=json.loads((SOURCE/'summary.json').read_text()),label_schema=LABEL_SCHEMA_VERSION,
    changed_objective_jobs=int((jobs.job_objective != jobs.snapshot_job_objective).sum()),
    checkout_verification=checks,
    external_objects_path_present=Path('/tmp/gpu-workload-audit/lfs/objects').exists(),
    local_objects_path_present=Path('.git/lfs/objects').exists(),
    supplemental_samples=int(supp.samples.sum()),supplemental_accelerator_hours=float(supp.accelerator_hours.sum()),
    caveats=[
        'Main totals derive from the October 11 hash-verified scan, not a new raw-payload scan.',
        'All saved data paths and their pointer identities or local bytes match this checkout.',
        'Job objective labels are refreshed from current code; per-window labels are not calculated here.',
        'October 11 job identities, completion metadata and expected widths remain provisional and dated.',
        'Nonexcluded does not mean validated successful or eligible for classifier evaluation.',
        'Accelerator-hours are summed per-device timestamp spans, not optimizer time.',
        'Elapsed job hours sum reconstructed job durations; concurrent independent jobs may overlap.',
        'Host count zero means unknown. Observed device counts include collector artifacts or incomplete collections.',
        'Supplemental Week 7 JSONs are reported separately, not added to the saved-corpus totals.',
        'This does not establish completeness of remote GCS buckets.'
    ])
(OUT/'provenance.json').write_text(json.dumps(provenance,indent=2)+'\n')

display = w[['cell','description','category','gpu_name','job_objective','kind','jobs','samples',
             'stream_hours','nonexcluded_accelerator_hours','median_stream_minutes','min_hosts','max_hosts',
             'min_devices','max_devices','excluded_failed_jobs']].copy()
display = display.round({'stream_hours':3,'nonexcluded_accelerator_hours':3,'median_stream_minutes':2})
page = '''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Workload inventory review — October 12, 2026</title>
<style>body{font:15px system-ui;margin:32px;color:#17252d;background:#fafcfd}h1{font-size:27px}p{max-width:1050px;line-height:1.5}table{border-collapse:collapse;font-size:13px;background:white}td,th{padding:9px;border:1px solid #ddd;text-align:left;vertical-align:top}th{position:sticky;top:0;background:#e4eef4}input{padding:12px;width:460px;max-width:90%;margin:15px 0}.scroll{overflow:auto;max-height:75vh}a{color:#145885}.stat{font-size:19px}</style>
<h1>Workload inventory review — October 12, 2026</h1>
<p class="stat">529 workload cells · 2,888 reconstructed jobs · 5,590,675 samples · 1,538.298 accelerator-hours</p>
<p>Main totals use the October 11 hash-verified inventory. Its file identities match the current checkout, but the external raw LFS cache is not available in this session. Current objective labels correct seven stalled jobs (11.360 accelerator-hours) to “other.” The 21 known failed jobs contribute 47.973 hours; excluding them leaves 1,490.325 hours, still including unverified collections. Eight additional brief Week 7 traces are listed separately.</p>
<p>“Cell” includes parameter and scope variants; it is not an independent algorithm family. Durations are per-device timestamp spans, not useful training time. Device/host counts describe telemetry, not proven participation; zero hosts means unknown. Completion metadata predates the October 12 log recovery. No new classifier accuracy is claimed.</p>
<p>Download <a href="workloads.csv">workloads by hardware</a> · <a href="jobs.csv">jobs</a> · <a href="categories.csv">category totals</a> · <a href="hardware.csv">hardware totals</a> · <a href="families.csv">families</a> · <a href="supplemental_traces.csv">supplemental traces</a> · <a href="provenance.json">provenance</a></p>
'''
page += '<p>Category totals (mutually exclusive; include failed collections):</p>'+c[['category','cells','jobs','samples','accelerator_hours','median_job_minutes','excluded_failed_jobs']].round(2).to_html(index=False)
page += '<p>Search the complete hardware-specific workload catalog:</p><input id="q" placeholder="Search workload, hardware, description…" aria-label="Filter workloads"><span id="count"></span><div class="scroll">'
page += display.to_html(index=False,table_id='catalog')+'</div>'
page += '''<script>const q=document.querySelector('#q'),rows=[...document.querySelectorAll('#catalog tbody tr')],c=document.querySelector('#count');function filter(){const term=q.value.toLowerCase();let n=0;for(const row of rows){const match=row.textContent.toLowerCase().includes(term);row.hidden=!match;n+=match}c.textContent=n+' hardware-specific rows'}q.addEventListener('input',filter);filter()</script></html>'''
(OUT/'inventory.html').write_text(page)
print(c[['category','cells','jobs','accelerator_hours','median_job_minutes','excluded_failed_jobs','hardware']].round(2).to_string(index=False))
print('VERIFICATION', json.dumps(checks))
print('SUPPLEMENT',len(supp),supp.samples.sum(),supp.accelerator_hours.sum())
print('CURRENT OBJECTIVES',jobs.groupby('job_objective').stream_hours.sum().to_dict())
