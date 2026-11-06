"""Build the portable, answer-independent CPU top96 checks from the saved seed run."""
import argparse
import hashlib
import json
from pathlib import Path
import zipfile
from read_cpu_tensor_archive import read_tensors
from textvqa_patch_properties import detail_without_padding,padding_fraction
from selector_grid_protocol import choose


def prepare(archive,output):
    with archive.open('rb') as f:digest=hashlib.file_digest(f,'sha256').hexdigest()
    result=dict(version='selector-grid-textvqa-reference-v1',source_archive=archive.name,source_archive_sha256=digest,cases={})
    with zipfile.ZipFile(archive) as z:
        report=json.loads(z.read('report.json'))
        if report['status']!='passed' or report['completed_predictions']!=2200:raise ValueError('Require complete seed run')
        for name in sorted(z.namelist()):
            if not(name.startswith('questions/') and name.endswith('.json')):continue
            q=json.loads(z.read(name));f=q['feature_dump'];blob=z.read(f['file'])
            if hashlib.sha256(blob).hexdigest()!=f['sha256']:raise ValueError('Tensor hash mismatch')
            t=read_tensors(blob);key=q['case']['image_sha256']
            if hashlib.sha256(t['clip_patches'].tobytes()).hexdigest()!=f['input_diagnostics']['source_patches']['sha256']:raise ValueError('Decoded feature mismatch')
            detail=detail_without_padding(t['pixels'],*f['image_size']);pad=padding_fraction(*f['image_size']);e=f['input_diagnostics']['input']['errors']
            result['cases'][q['case']['case_id']]=dict(image_sha256=key,image_size=f['image_size'],source_patches_sha256=f['input_diagnostics']['source_patches']['sha256'],
                detail=detail.tolist(),padding=pad.tolist(),errors=e,
                expected_detail_top96=choose('detail',96,pad,detail,e,key,0,'input'),expected_E_top96=choose('error',96,pad,detail,e,key,0,'input'),
                historical={k:dict(response=v['response'],generated_ids=v['generation_tokens']['generated_token_ids'],scoring=v['scoring']) for k,v in q['arms'].items()})
    if len(result['cases'])!=100:raise ValueError('Incomplete cohort')
    output.write_text(json.dumps(result,separators=(',',':'))+'\n')

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('archive',type=Path);p.add_argument('output',type=Path);args=p.parse_args()
    if args.output.exists():p.error('Choose a fresh output path')
    prepare(args.archive,args.output)
