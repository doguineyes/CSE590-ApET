"""Build portable images + verified historical input E/zero records; no inference."""

import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from analyze_rq1_scores import PLAN, load_sources
from rq1_protocol import REPO, sha256


def prepare(results, destination):
    results, destination = Path(results), Path(destination)
    if destination.exists(): raise ValueError('Choose a new bundle filename')
    plan = json.loads(PLAN.read_text())
    # Fully audit canonical source ZIPs, arithmetic, images and feature hashes first.
    records, audits = load_sources(results, plan)
    audits = [{**a, 'archive':Path(a['archive']).name} for a in audits]
    entries, files = [], {}
    for dataset, info in plan['sources'].items():
        with zipfile.ZipFile(results/(info['run']+'.zip')) as z:
            report = json.loads(z.read('report.json'))
            lookup = {json.loads(z.read(n))['case_id']: n for n in z.namelist()
                      if n.startswith('questions/') and n.endswith('.json')}
            files[f'references/{dataset}/report.json'] = z.read('report.json')
            for row in report['questions']:
                cid = row['case_id']; q = json.loads(z.read(lookup[cid]))
                qpath = f'references/{dataset}/questions/{cid.split(":")[1]}.json'
                fpath = f'references/{dataset}/features/{q["image_sha256"]}.json'
                files[qpath] = z.read(lookup[cid])
                files[fpath] = z.read('features/'+q['image_sha256']+'.json')
                entries.append({'case_id':cid, 'dataset':dataset, 'question_file':qpath, 'feature_file':fpath})
    index = {'version':'rq1-input-ih-references-v1', 'sources':plan['sources'], 'source_audits':audits,
        'cases':entries, 'files_sha256':{name:hashlib.sha256(raw).hexdigest() for name,raw in sorted(files.items())},
        'scope':'Exact original question/feature JSON copied from fully verified canonical ZIPs. No tensors unpickled, no inference, no cohort/score changes.'}
    raw_index = (json.dumps(index,indent=2,allow_nan=False)+'\n').encode()
    files['references/index.json'] = raw_index
    source = results/'rq1-inputs-v1.zip'
    with zipfile.ZipFile(source) as old:
        if old.testzip() is not None: raise ValueError('Existing image bundle CRC failed')
        for name in old.namelist():
            if name in files or '..' in Path(name).parts or name.startswith('/'): raise ValueError('Unsafe/duplicate path')
            files[name] = old.read(name)
    destination.parent.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(destination,'x',zipfile.ZIP_DEFLATED,compresslevel=6) as z:
        for name,raw in sorted(files.items()):
            item=zipfile.ZipInfo(name,date_time=(1980,1,1,0,0,0));item.compress_type=zipfile.ZIP_DEFLATED
            z.writestr(item,raw)
    return {'archive':str(destination), 'archive_sha256':sha256(destination),
            'reference_index_sha256':hashlib.sha256(raw_index).hexdigest(),
            'questions':len(entries), 'calibration_case_ids':[e['case_id'] for dataset in plan['sources']
                for e in [r for r in entries if r['dataset']==dataset][:2]],
            'case_ids':[e['case_id'] for e in entries], 'bytes':destination.stat().st_size}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--results',type=Path,default=REPO/'research/results')
    p.add_argument('--output',type=Path,default=REPO/'research/results/rq1-ih-inputs-v1.zip')
    a=p.parse_args();print(json.dumps(prepare(a.results,a.output),indent=2))
