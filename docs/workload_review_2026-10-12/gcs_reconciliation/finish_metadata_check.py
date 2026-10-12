import pathlib,json,tarfile,hashlib,base64,pandas as pd,collections
B=pathlib.Path('/tmp/gcs-workload-check-2026-10-12');index={}
for f in ['raw-download-candidates-downloaded.json','downloaded-so-far.json']:
 for r in json.load(open(B/f)):index[r.get('md5',r.get('md5Hash'))]={'sha256':r['sha256'],'source':'download'}
p=next(x['path'] for x in json.load(open(B/'archive-candidates-downloaded.json')) if 'analysis_bundle' in x['name'])
with tarfile.open(p) as t:
 for m in t:
  if m.isfile() and m.name.endswith('.parquet'):
   blob=t.extractfile(m).read();index[base64.b64encode(hashlib.md5(blob).digest()).decode()]={'sha256':hashlib.sha256(blob).hexdigest(),'source':'verified_archive'}
(B/'verified-md5-to-sha.json').write_text(json.dumps(index));candidates=json.load(open(B/'all-raw-candidates.json'));unknown=[r for r in candidates if r['md5Hash'] not in index];(B/'unverified-raw-candidates.json').write_text(json.dumps(unknown));print('allraw unique',len(candidates),'verified',len(candidates)-len(unknown),'unknown',len(unknown),'bytes',sum(int(r['size']) for r in unknown));print('examples',[r['name'] for r in unknown[:12]])
